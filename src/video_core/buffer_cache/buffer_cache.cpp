// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <intrin.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <immintrin.h>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "common/alignment.h"
#include "common/unmapped_allocations.h"
#include "common/io_file.h"
#include "common/perf_monitor.h"
#include "core/emulator_settings.h"
#include "video_core/buffer_cache/fault_download_ranges.h"
#include "video_core/buffer_cache/hot_page_prehasher.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/proto_import.h"
#include "video_core/buffer_cache/parallel_hasher.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image_aliasing.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>
#ifdef _WIN32
#include <windows.h>
#include "core/guest_hash.h"
#include "ryty_gpu/ryty_diag.h"
#endif

namespace VideoCore {

namespace {
// PERF-027: copies guest memory into upload buffers on several cores at once. The command
// thread hands out 64 KB pieces, copies pieces itself, and returns only when every piece is
// copied, so the copy finishes at the same point in the command stream as before.
class ParallelUploadCopier {
public:
    struct Piece {
        VAddr source;
        u8* destination;
        u64 size;
    };

    static ParallelUploadCopier& Instance() {
        static ParallelUploadCopier copier;
        return copier;
    }

    /// The pieces of the next batch, to fill before Run.
    std::vector<Piece>& BeginBatch() {
        // PERF-046 review: a helper that woke late for the previous batch must be out before it
        // is replaced, and none may join until Run opens the new one. Both are decided under the
        // lock: checking `active` outside it let a helper that slept through a batch join while
        // the pieces were being refilled.
        std::unique_lock lk{mutex};
        while (active.load(std::memory_order_acquire) != 0) {
            lk.unlock();
            _mm_pause();
            lk.lock();
        }
        batch_open = false;
        pieces.clear();
        return pieces;
    }

    void Run(Core::MemoryManager* memory, const Common::ReadCapture* capture) {
        {
            std::scoped_lock lk{mutex};
            memory_manager = memory;
            read_capture = capture;
            next.store(0, std::memory_order_relaxed);
            done.store(0, std::memory_order_relaxed);
            batch_open = true;
            ++generation;
        }
        cv.notify_all();
        CopyPieces();
        while (done.load(std::memory_order_acquire) != pieces.size()) {
            _mm_pause();
        }
    }

private:
    ParallelUploadCopier() {
        const u32 cores = std::max(std::thread::hardware_concurrency(), 4u);
        const u32 helpers = std::clamp(cores / 4, 1u, 3u);
        for (u32 i = 0; i < helpers; ++i) {
            threads.emplace_back([this](std::stop_token stop) { Work(stop); });
        }
    }

    void Work(std::stop_token stop) {
        Common::SetCurrentThreadName("shadGT:UploadCopy");
        u64 seen = 0;
        while (true) {
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, stop, [&] { return batch_open && generation != seen; });
                if (stop.stop_requested()) {
                    return;
                }
                seen = generation;
                active.fetch_add(1, std::memory_order_acq_rel);
            }
            CopyPieces();
            active.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    void CopyPieces() {
        const size_t count = pieces.size();
        for (size_t i = next.fetch_add(1, std::memory_order_relaxed); i < count;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
            const auto& piece = pieces[i];
            memory_manager->CopySparseMemory(piece.source, piece.destination, piece.size,
                                             read_capture);
            done.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    std::vector<Piece> pieces;
    Core::MemoryManager* memory_manager{};
    const Common::ReadCapture* read_capture{};
    std::atomic<size_t> next{};
    std::atomic<size_t> done{};
    std::atomic<u32> active{};
    u64 generation{};
    bool batch_open{}; ///< Between Run and the next BeginBatch; changed under the mutex.
    std::mutex mutex;
    std::condition_variable_any cv;
    std::vector<std::jthread> threads;
};
} // namespace

static std::atomic<HotPagePrehasher*> s_prehasher{nullptr}; // PERF-049
static std::atomic<BufferCache*> s_own_write_cache{nullptr}; // FIX-073

// DIAG-072: SHADGT_TRACE_READBACKS=1 keeps the last readbacks written into guest memory.
static bool TraceReadbacks() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_TRACE_READBACKS");
        return env && env[0] == '1';
    }();
    return enabled;
}

static s64 TraceNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// FIX-073 (described at NoteGpuWriteOverReadbackPages). -DisablePerf 99 turns it off.
static bool ReadbackHashEnabled() {
    static const bool enabled = Common::PerfFeatureEnabled(99);
    return enabled;
}

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
// FIX-188: Intel Arc (#65) gets twice the stream rings, against ring wraps in heavy frames.
static size_t StreamBufferSize(const Vulkan::Instance& instance) {
    return instance.IsIntel() ? 2 * STREAM_BUFFER_SIZE : STREAM_BUFFER_SIZE;
}

// PERF-094: the stream ring (small read-only buffers copied for each draw) is in system memory,
// which the GPU reads over PCIe, instead of video memory the recorder writes over PCIe. Each copy
// took 1,620 cycles in video memory and 950 in system memory (GT Sport race, RTX 4070 Ti, about
// 200,000 copies per second), with the same GPU load. Idea from the shadPS4-eltutz fork's
// findings ("ring host-direct"). SHADGT_STREAM_HOST=0 or -DisablePerf 111 keeps video memory.
static MemoryType StreamRingType() {
    const char* env = std::getenv("SHADGT_STREAM_HOST");
    const bool host = !(env && env[0] == '0') && Common::PerfFeatureEnabled(111);
    return host ? MemoryType::HostUncached : MemoryType::Stream;
}

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)}, page_manager{tracker},
      stream_buffer{instance, scheduler, StreamRingType(), StreamBufferSize(instance_)},
      ahead_stream{instance, scheduler, MemoryType::Stream, StreamBufferSize(instance_)},
      ahead_staging{instance, scheduler, MemoryType::HostUncached, StreamBufferSize(instance_)},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance},
      // PERF-011 is disabled: skipping GPU-written bytes in uploads loses CPU stores to those
      // bytes that the snapshot check cannot see (a range superseded by a newer GPU write, or a
      // store of the old value). With exact store sizes it produced corrupted (blue, exploded)
      // vertices around the track in a race, so CPU write faults drain the GPU again.
      split_write_faults{false},
      // PERF-015 is disabled: without the drain, GPU results near those records (GT Sport's
      // sparks) no longer reached guest memory before the game's CPU used them, and sparks
      // came with exploded vertices. The drain's wait for all earlier GPU work is what the game
      // relies on there.
      split_gds_write_faults{false} {
    if (DirectMemory::Enabled() && instance.IsExternalMemoryHostSupported()) { // D1
        direct_memory = std::make_unique<DirectMemory>(instance, scheduler);
    }
    for (u32 i = 1; i < StageWorkerCount(); ++i) { // R5
        worker_streams[i] = std::make_unique<StreamBuffer>(instance, scheduler, MemoryType::Stream,
                                                           StreamBufferSize(instance_) / 2);
        worker_stagings[i] = std::make_unique<StreamBuffer>(
            instance, scheduler, MemoryType::HostUncached, StreamBufferSize(instance_) / 2);
    }
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    LOG_WARNING(Render_Vulkan, "P3: arena sparse memory types {:#x}, alignment {:#x}, block {:#x}",
                reqs.memoryTypeBits, reqs.alignment, block_size);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);

    // PERF-049 (opt-in, SHADGT_PREHASH=1): not in precise readback mode, where GPU-written pages
    // are read-protected and the prehasher could fault on one. Off by default: in the Nurburgring
    // benchmark only about 11% of its hashes were used (epochs change faster than the recorder
    // reaches them), and the recorder waited on its lock while it built each batch (18% of the
    // recorder's samples in RefreshReadPages).
    const char* prehash_env = std::getenv("SHADGT_PREHASH");
    if (prehash_env && prehash_env[0] == '1' && Common::PerfFeatureEnabled(61) &&
        EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Precise) {
        prehasher = std::make_unique<HotPagePrehasher>(
            1, BYTES_PER_PAGE, g_upload_epoch,
            [this](std::vector<VAddr>& pages) {
                std::scoped_lock lk{uploaded_pages_mutex};
                pages.reserve(uploaded_page_contents.Size());
                uploaded_page_contents.ForEachPage([&](VAddr page) { pages.push_back(page); });
            },
            [this](const std::function<void()>& func) { memory->WithMappingsStable(func); },
            [this](VAddr page) { return memory->IsMappedAddress(page); });
        s_prehasher = prehasher.get();
        g_upload_epoch_listener.store(
            [](u32 epoch) {
                if (auto* hasher = s_prehasher.load(std::memory_order_acquire)) {
                    hasher->OnEpoch(epoch);
                }
            },
            std::memory_order_release);
    }
#ifdef RYTY_GPU
    // FIX-073: the renderer's own writes into guest memory keep readback page hashes current.
    s_own_write_cache = this;
    Core::RytyMemory::SetOwnWriteObserver([](VAddr address, u64 size) {
        if (auto* cache = s_own_write_cache.load(std::memory_order_acquire)) {
            cache->RefreshReadbackPageHashes(address, size);
        }
    });
#endif
}

BufferCache::~BufferCache() {
#ifdef RYTY_GPU
    Core::RytyMemory::SetOwnWriteObserver(nullptr);
    s_own_write_cache.store(nullptr, std::memory_order_release);
#endif
    g_upload_epoch_listener.store(nullptr, std::memory_order_release);
    s_prehasher.store(nullptr, std::memory_order_release);
    prehasher.reset();
}

u64 BufferCache::SparseResidentBytes() {
    std::scoped_lock buffer_lock{mutex};
    u64 bytes = 0;
    for (const auto& range : resident_ranges) bytes += (range.end - range.start) << block_shift;
    return bytes;
}

u64 BufferCache::SparseFreedBytes() {
    std::scoped_lock buffer_lock{mutex};
    return sparse_freed_bytes;
}

void BufferCache::NoteArenaMapping(VAddr address, u64 size) {
    if (!size || address > ~u64{0} - size) return;
    std::scoped_lock lock{arena_unmap_mutex};
    unmapped_arena_bytes.Subtract(address, size);
}

void BufferCache::TickFrame() {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::RecordGdsReadback(VAddr address, u32 gds_offset, u32 size) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    auto readback = std::make_shared<AsyncReadback>();
    readback->address = address;
    readback->size = size;
    readback->is_gds = true;
    ++num_valid_gds_readbacks;
    readback->download = staging_pool.Request(size, MemoryType::HostCached, 4, true);
    const vk::BufferCopy copy = {
        .srcOffset = gds_offset,
        .dstOffset = readback->download.offset,
        .size = size,
    };
    runtime.CopyBuffer(&gds_buffer, readback->download.buffer, std::span{&copy, 1});
    std::scoped_lock lk{async_readbacks_mutex};
    pending_async_readbacks.push_back(std::move(readback));
    num_pending_async_readbacks = static_cast<u32>(pending_async_readbacks.size());
    num_tracked_async_readbacks =
        static_cast<u32>(pending_async_readbacks.size() + inflight_async_readbacks.size());
}

std::vector<std::shared_ptr<BufferCache::AsyncReadback>> BufferCache::TakePendingAsyncReadbacks() {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    std::scoped_lock lk{async_readbacks_mutex};
    auto readbacks = std::move(pending_async_readbacks);
    pending_async_readbacks.clear();
    inflight_async_readbacks.insert(inflight_async_readbacks.end(), readbacks.begin(),
                                    readbacks.end());
    num_pending_async_readbacks = 0;
    return readbacks;
}

void BufferCache::NoteCpuReadFault(VAddr address, u64 size, u64 exact_write_size) {
    // PERF-010: a CPU access that needed GPU-written data drains the GPU. Pages that saw such an
    // access are read back after each later GPU write, so the next access finds current data.
    ++hot_page_stats.faults;
    // PERF-DIAG-008: does the access itself touch GPU-written bytes, or only share their page?
    constexpr u64 LineSize = 64;
    const VAddr line = Common::AlignDown(address, LineSize);
    const bool touches_gpu_bytes =
        exact_write_size != 0
            ? gpu_modified_ranges.Intersects(address, exact_write_size)
            : gpu_modified_ranges.Intersects(
                  line, Common::AlignUp(address + std::max<u64>(size, 1), LineSize) - line);
    if (!touches_gpu_bytes) {
        ++hot_page_stats.page_only_faults;
    }
    const VAddr fault_page = Common::AlignDown(address, 4_KB);
    auto& page_stat = hot_page_stats.pages[fault_page];
    if (page_stat.faults++ == 0) {
        gpu_modified_ranges.ForEachInRange(fault_page, 4_KB, [&](VAddr start, VAddr end) {
            page_stat.gpu_bytes +=
                fmt::format("{}{:#x}+{:#x}", page_stat.gpu_bytes.empty() ? "" : ",",
                            start - fault_page, end - start);
        });
    }
    page_stat.last_offset = address - fault_page;
    page_stat.last_size = exact_write_size != 0 ? exact_write_size : size;
    page_stat.touches_gpu_bytes |= touches_gpu_bytes;
    constexpr u64 PageSize = 4_KB;
    const VAddr end = address + std::max<u64>(size, 1);
    for (VAddr page = Common::AlignDown(address, PageSize); page < end; page += PageSize) {
        if (!hot_pages.insert(page).second) {
            continue;
        }
        if (ReadbackHashEnabled()) {
            // FIX-073: its snapshot is taken when the fault's GPU data reaches guest memory.
            std::scoped_lock lk{readback_hash_mutex};
            if (pending_snapshot_pages.size() > 4096) {
                pending_snapshot_pages.clear();
            }
            if (!readback_page_snapshots.contains(page)) {
                pending_snapshot_pages.insert(page);
            }
            if (TraceReadbacks()) {
                hot_page_since_ns[page] = TraceNowNs();
            }
        }
        ++hot_page_stats.new_pages;
        hot_page_order.push_back(page);
        if (hot_page_order.size() > MaxHotPages) {
            hot_pages.erase(hot_page_order.front());
            hot_page_order.pop_front();
        }
    }
}

bool BufferCache::RecordHotPageReadbacks() {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    LogHotPageStats();
    if (hot_page_order.empty()) {
        return false;
    }
    constexpr u64 PageSize = 4_KB;
    // PERF-087: the pages with readbacks in flight are listed only when a page needs one (this
    // runs at every fence, and most fences find no hot page the GPU wrote since).
    boost::container::small_vector<VAddr, 64> inflight_pages;
    bool inflight_listed = false;
    const auto has_inflight = [&](VAddr page) {
        if (!inflight_listed) {
            inflight_listed = true;
            std::scoped_lock lk{async_readbacks_mutex};
            for (const auto* list : {&pending_async_readbacks, &inflight_async_readbacks}) {
                for (const auto& readback : *list) {
                    if (readback->valid) {
                        inflight_pages.push_back(Common::AlignDown(readback->address, PageSize));
                    }
                }
            }
            std::ranges::sort(inflight_pages);
        }
        return std::ranges::binary_search(inflight_pages, page);
    };
    std::vector<std::shared_ptr<AsyncReadback>> recorded;
    std::vector<std::pair<VAddr, VAddr>> cpu_newer_parts;   // FIX-069
    std::vector<std::pair<VAddr, VAddr>> cpu_rewrote_parts; // FIX-073
    for (const VAddr page : hot_page_order) {
        // A CPU-modified page may hold CPU writes over GPU ranges (PERF-011); those ranges are
        // only read back with a snapshot check, at write faults.
        if (!memory_tracker->IsRegionGpuModified(page, PageSize) ||
            memory_tracker->IsRegionCpuModified(page, PageSize) || has_inflight(page)) {
            continue;
        }
        const u64 block = page >> block_shift;
        const auto* arena = GetArena(block, block);
        // Same ranges as DownloadMemory: only bytes the GPU wrote, so CPU writes not yet uploaded
        // to the arena are never overwritten.
        memory_tracker->ForEachDownloadRange<false>(page, PageSize, [&](u64 address, u64 size) {
            gpu_modified_ranges.ForEachInRange(address, size, [&](VAddr start_all, VAddr end_all) {
                SplitByWriter(start_all, end_all, [&](VAddr start, VAddr end) {
                if (CpuRewroteReadbackRange(start, end)) {
                    // FIX-073: the CPU changed these bytes after the GPU wrote them, unseen.
                    cpu_rewrote_parts.emplace_back(start, end);
                    return;
                }
                auto readback = std::make_shared<AsyncReadback>();
                readback->address = start;
                readback->size = static_cast<u32>(end - start);
                readback->download =
                    staging_pool.Request(readback->size, MemoryType::HostCached, 4, true);
                // FIX-071: the guest bytes now, so the completion keeps them when the CPU has
                // written them meanwhile (the game may reuse the memory once a fence before
                // the completion was signaled).
                readback->snapshot.resize(readback->size);
                Common::CopyGuest(readback->snapshot.data(), std::bit_cast<const void*>(start),
                                  readback->size);
                if (TraceReadbacks()) { // DIAG-072
                    readback->writer = DescribeGpuWriters(start, end - start);
                    readback->recorded_ns = TraceNowNs();
                }
                const vk::BufferCopy copy = {
                    .srcOffset = start - arena->cpu_addr,
                    .dstOffset = readback->download.offset,
                    .size = readback->size,
                };
                runtime.CopyBuffer(arena, readback->download.buffer, std::span{&copy, 1});
                hot_page_stats.recorded_bytes += readback->size;
                recorded.push_back(std::move(readback));
                }, [&](VAddr s, VAddr e) { cpu_newer_parts.emplace_back(s, e); }); // FIX-069
            });
        });
    }
    for (const auto& [s, e] : cpu_newer_parts) {
        gpu_modified_ranges.Subtract(s, e - s);
        ++writebacks_cpu_newer;
    }
    for (const auto& [s, e] : cpu_rewrote_parts) {
        // FIX-073: the CPU's bytes are newer; they go to the GPU on its next use of them.
        gpu_modified_ranges.Subtract(s, e - s);
        NoteCpuChange(s, e - s);
        memory_tracker->MarkRegionAsCpuModified(s, e - s);
        ++readbacks_cpu_rewrote;
    }
    if (recorded.empty()) {
        return false;
    }
    hot_page_stats.recorded += static_cast<u32>(recorded.size());
    std::scoped_lock lk{async_readbacks_mutex};
    pending_async_readbacks.insert(pending_async_readbacks.end(), recorded.begin(), recorded.end());
    num_pending_async_readbacks = static_cast<u32>(pending_async_readbacks.size());
    num_tracked_async_readbacks =
        static_cast<u32>(pending_async_readbacks.size() + inflight_async_readbacks.size());
    return true;
}

void BufferCache::LogHotPageStats() {
    // PERF-DIAG-006: CPU faults on GPU-written data and the readbacks recorded for them.
    auto& stats = hot_page_stats;
    const auto now = std::chrono::steady_clock::now();
    if (stats.window_start == std::chrono::steady_clock::time_point{}) {
        stats.window_start = now;
        return;
    }
    if (now - stats.window_start < std::chrono::seconds{2}) {
        return;
    }
    std::vector<std::pair<VAddr, const HotPageStats::PageFaults*>> pages;
    for (const auto& [page, stat] : stats.pages) {
        pages.emplace_back(page, &stat);
    }
    std::ranges::sort(pages, std::greater{}, [](const auto& p) { return p.second->faults; });
    std::string top;
    for (size_t i = 0; i < std::min<size_t>(4, pages.size()); ++i) {
        const auto& [page, stat] = pages[i];
        top += fmt::format(" {:#x}x{} (last +{:#x} size {:#x}, {}, gpu bytes [{}])", page,
                           stat->faults, stat->last_offset, stat->last_size,
                           stat->touches_gpu_bytes ? "touches them" : "page only", stat->gpu_bytes);
    }
    LOG_WARNING(Render_Vulkan,
                "Readback pages in 2.0 s: {} CPU faults ({} only share a page with GPU writes, "
                "{} handled without a drain, {} beside GDS copies, {} GPU ranges then written "
                "by the CPU), {} new pages ({} tracked), {} readbacks ({} KB), {} completed, {} "
                "superseded; FIX-066: {} uploads over GPU-written bytes; FIX-069: {} parts kept by the "
                "CPU; FIX-073: {} parts the CPU rewrote unseen, {} unseen CPU bytes sent to the "
                "GPU;{}",
                stats.faults, stats.page_only_faults, stats.split_faults, stats.gds_split_faults,
                stats.cpu_overwrote.exchange(0), stats.new_pages, hot_page_order.size(),
                stats.recorded, stats.recorded_bytes / 1024, stats.completed.exchange(0),
                stats.invalidated.exchange(0), stats.uploads_over_gpu_bytes.load(),
                std::exchange(writebacks_cpu_newer, 0), std::exchange(readbacks_cpu_rewrote, 0),
                std::exchange(unseen_cpu_bytes_synced, 0), top);
    if (auto& sync = image_sync_stats; sync.calls != 0) {
        const u64 generation = texture_cache.RegistryGeneration();
        LOG_WARNING(Render_Vulkan,
                    "DIAG-048: image checks of buffers in 2.0 s: {} calls, {} exact memo hits, {} "
                    "block memo hits, {} walked and free ({} MB walked), {} overlapping images; "
                    "{} image registry changes; PERF-078: {} image lists reused, {} with nothing "
                    "to export (PERF-086: {} known quiet); FIX-065: {} old images on unwatched "
                    "pages not exported; {} walks of 64 MB or more ({} over images):{}",
                    sync.calls, sync.exact_hits, sync.block_hits, sync.full_free,
                    sync.full_bytes >> 20, sync.with_images,
                    sync.generation ? generation - sync.generation : 0, sync.overlap_hits,
                    sync.nothing_to_export, sync.quiet_hits, sync.stale_skipped,
                    sync.large,
                    sync.large_with_images, sync.samples);
        sync = {};
        sync.generation = generation;
    }
    // PERF-033: hot-page uploads avoided because the bytes had not changed.
    const u64 unchanged = hot_pages_unchanged.exchange(0);
    const u64 uploaded = hot_pages_uploaded.exchange(0);
    if (unchanged + uploaded != 0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-033 hot pages in 2.0 s: {} unchanged and not uploaded ({} MB), {} "
                    "uploaded; PERF-049: {} hashes taken ahead, {} on the recorder; PERF-064: "
                    "{} cooled ({} written while cooling)",
                    unchanged, unchanged * BYTES_PER_PAGE >> 20, uploaded, prehashed_hits.load(),
                    prehashed_misses.load(), hot_pages_cooled.load(), hot_pages_cooled_changed.load());
    }
    hot_pages_cooled = 0;
    hot_pages_cooled_changed = 0;
    prehashed_hits = 0;
    prehashed_misses = 0;
    if (stage_prehash_hits + stage_prehash_misses != 0) {
        LOG_WARNING(Render_Vulkan, "PERF-087: hot pages hashed by the buffer stage before the "
                    "lock: {} used, {} hashed under it", stage_prehash_hits.load(), stage_prehash_misses.load());
        stage_prehash_hits = 0;
        stage_prehash_misses = 0;
    }
    if (ahead_hits + ahead_misses != 0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-062: hot-page hashes taken by the command thread: {} used, {} hashed "
                    "here; {} hashed ahead",
                    ahead_hits.load(), ahead_misses.load(), ahead_hashed.exchange(0));
    }
    ahead_hits = 0;
    ahead_misses = 0;
    if (const u64 stores = write_through_stores.exchange(0); stores != 0) {
        LOG_WARNING(Render_Vulkan,
                    "FIX-079: {} CPU stores written through GPU-written pages in 2.0 s ({} bytes, "
                    "{} of them GPU bytes another thread wrote meanwhile), {} left to the drain "
                    "(pages stored to often)",
                    stores, write_through_bytes.exchange(0), write_through_gpu_bytes.exchange(0),
                    write_through_refused.exchange(0));
    }
    if (const u64 flushes = async_fault_flushes.exchange(0); flushes != 0) {
        const u64 wait_us = async_fault_wait_us.exchange(0);
        LOG_WARNING(Render_Vulkan,
                    "PERF-052: {} fault downloads waited for off the recorder, {:.1f} ms in total "
                    "(avg {:.2f} ms)",
                    flushes, wait_us / 1000.0, wait_us / 1000.0 / flushes);
    }
    stats.page_only_faults = 0;
    stats.drain_reports = 0;
    stats.split_faults = 0;
    stats.gds_split_faults = 0;
    stats.uploads_over_gpu_bytes = 0;
    stats.pages.clear();
    stats.window_start = now;
    stats.faults = 0;
    stats.new_pages = 0;
    stats.recorded = 0;
    stats.recorded_bytes = 0;
}

std::string BufferCache::DescribeRecentReadbacks(VAddr address, u64 size) {
    if (!TraceReadbacks()) {
        return " (SHADGT_TRACE_READBACKS=1 to trace)";
    }
    const s64 now = TraceNowNs();
    std::string out;
    std::scoped_lock lk{readback_trace_mutex};
    for (auto it = readback_trace.rbegin(); it != readback_trace.rend(); ++it) {
        if (it->address < address + size && address < it->address + it->size) {
            out += fmt::format(" [{:#x}+{:#x} recorded {:.1f} ms ago, written {:.1f} ms ago, "
                               "{} bytes changed; writers:{}]",
                               it->address, it->size, (now - it->recorded_ns) / 1e6,
                               (now - it->completed_ns) / 1e6, it->changed_bytes,
                               it->writer.substr(0, 400));
        }
    }
    return out.empty() ? std::string{" none"} : out;
}

// DIAG-053: GPU data written back into guest memory over a recent command buffer.
static void NoteWritebackOverCommands(AmdGpu::Liverpool* liverpool, const char* writer,
                                      VAddr address, u64 size) {
    if (!liverpool) {
        return;
    }
    const auto cmd = liverpool->OverlappingRecentCmdBuffer(address, size);
    if (!cmd) {
        return;
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 40) {
        LOG_WARNING(Render_Vulkan,
                    "DIAG-053: {} wrote GPU data {:#x}+{:#x} over command buffer {:#x}+{:#x} "
                    "({} buffers ago)",
                    writer, address, size, cmd->address, cmd->size,
                    liverpool->CmdBufferSequence() - cmd->sequence);
    }
}

void BufferCache::CompleteAsyncReadbacks(
    std::span<const std::shared_ptr<AsyncReadback>> readbacks) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    std::scoped_lock lk{async_readbacks_mutex};
    // FIX-106: guest bytes are read through CopySparseMemory, which does not fault on pages
    // protected for tracking. A plain read faulted on this helper thread while it held the
    // buffer cache lock, and the fault's flush waited for the recorder, which waited for that
    // lock (early fences froze at car select).
    thread_local std::vector<u8> current;
    const auto guest_differs = [&](const AsyncReadback& readback) {
        current.resize(readback.size);
        memory->CopySparseMemory(readback.address, current.data(), readback.size);
        return std::memcmp(current.data(), readback.snapshot.data(), readback.size) != 0;
    };
    for (const auto& readback : readbacks) {
        if (readback->valid && memory->IsValidMapping(readback->address, readback->size) &&
            !readback->snapshot.empty() && guest_differs(*readback)) {
            // PERF-011: the CPU wrote the range after the fault; its value is newer than the
            // GPU's. Stop excluding the range from uploads and upload the CPU value.
            completed_readback_ranges.emplace_back(readback->address, readback->size);
            num_completed_readback_ranges = static_cast<u32>(completed_readback_ranges.size());
            memory_tracker->MarkRegionAsCpuModified(readback->address, readback->size);
            ++hot_page_stats.cpu_overwrote;
        } else if (readback->valid && memory->IsValidMapping(readback->address, readback->size)) {
            readback->download.Invalidate();
            NoteWritebackOverCommands(liverpool, "async readback", readback->address,
                                      readback->size);
            if (TraceReadbacks()) { // DIAG-072
                const u8* guest = std::bit_cast<const u8*>(readback->address);
                u32 changed = 0;
                for (u32 i = 0; i < readback->size; ++i) {
                    changed += guest[i] != readback->download.mapped[i];
                }
                std::scoped_lock trace_lock{readback_trace_mutex};
                readback_trace.push_back({readback->address, readback->size, changed,
                                          readback->recorded_ns, TraceNowNs(), readback->writer});
                if (readback_trace.size() > 512) {
                    readback_trace.pop_front();
                }
            }
            memory->TryWriteBacking(std::bit_cast<void*>(readback->address),
                                    readback->download.mapped, readback->size);
            completed_readback_ranges.emplace_back(readback->address, readback->size);
            num_completed_readback_ranges = static_cast<u32>(completed_readback_ranges.size());
            // The guest copy is now current. The page stops being GPU-modified once no other
            // GPU-written bytes remain on it (ApplyCompletedReadbacks, GPU thread).
            ++hot_page_stats.completed;
        } else {
            ++hot_page_stats.invalidated;
        }
        if (readback->is_gds && readback->valid) {
            readback->valid = false;
            --num_valid_gds_readbacks;
        }
        std::erase(inflight_async_readbacks, readback);
        finished_async_downloads.push_back(readback->download);
    }
    num_tracked_async_readbacks =
        static_cast<u32>(pending_async_readbacks.size() + inflight_async_readbacks.size());
}

void BufferCache::ReleaseFinishedAsyncReadbacks() {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    std::vector<Vulkan::StagingBufferRef> finished;
    {
        std::scoped_lock lk{async_readbacks_mutex};
        finished.swap(finished_async_downloads);
    }
    for (const auto& download : finished) {
        staging_pool.FreeDeferred(download);
    }
}

void BufferCache::InvalidateAsyncReadbacks(VAddr address, u64 size) {
    // A later GPU write to the destination makes the read-back value stale; keep the
    // GPU-modified state so the exact path downloads the newer data.
    if (num_tracked_async_readbacks.load() == 0) {
        return;
    }
    // Completion clears GPU-modified state per 4 KB tracker page, so any readback sharing a page
    // with the write is stale too.
    constexpr u64 PageSize = 4_KB;
    const VAddr start = Common::AlignDown(address, PageSize);
    const VAddr end = Common::AlignUp(address + size, PageSize);
    std::scoped_lock lk{async_readbacks_mutex};
    const auto invalidate = [&](const std::shared_ptr<AsyncReadback>& readback) {
        if (!readback->valid) {
            return;
        }
        if (readback->is_gds && split_gds_write_faults) {
            // PERF-015: its page stays GPU-modified until ApplyCompletedReadbacks finds no
            // GPU-written bytes left, so only a write to its own bytes makes it stale.
            if (readback->address < address + size &&
                address < readback->address + readback->size) {
                readback->valid = false;
                --num_valid_gds_readbacks;
            }
            return;
        }
        if (readback->address < end && start < readback->address + readback->size) {
            readback->valid = false;
            if (readback->is_gds) {
                --num_valid_gds_readbacks;
            }
        }
    };
    std::ranges::for_each(pending_async_readbacks, invalidate);
    std::ranges::for_each(inflight_async_readbacks, invalidate);
}

void BufferCache::ApplyCompletedReadbacks() {
    if (num_completed_readback_ranges.load() == 0) {
        return;
    }
    std::vector<std::pair<VAddr, u64>> ranges;
    {
        std::scoped_lock lk{async_readbacks_mutex};
        ranges.swap(completed_readback_ranges);
        num_completed_readback_ranges = 0;
    }
    boost::container::small_vector<VAddr, 16> unmark;
    {
        std::scoped_lock gpu_lk{gpu_ranges_mutex}; // R3
        for (const auto& [address, size] : ranges) {
            gpu_modified_ranges.Subtract(address, size);
        }
        for (const auto& [address, size] : ranges) {
            const VAddr first_page = Common::AlignDown(address, 4_KB);
            const VAddr last_page = Common::AlignDown(address + size - 1, 4_KB);
            for (VAddr page = first_page; page <= last_page; page += 4_KB) {
                if (!gpu_modified_ranges.Intersects(page, 4_KB)) {
                    unmark.push_back(page);
                }
            }
        }
    }
    for (const VAddr page : unmark) {
        memory_tracker->UnmarkRegionAsGpuModified(page, 4_KB, false);
    }
}

bool BufferCache::TrySplitGdsWriteFault(VAddr address, u64 exact_write_size) {
    // PERF-015: GT Sport keeps 16-byte records whose second half a GDS copy writes (PERF-009)
    // and whose first half the CPU writes, about once per frame, forcing a full drain. When every
    // GPU-written byte of the page is a pending GDS copy and the store (exact size, PERF-011b)
    // misses them, those bytes are kept out of uploads until their value reaches guest memory
    // (SynchronizeMemory), and a CPU store to them meanwhile is detected by the snapshot and wins.
    if (!split_gds_write_faults || exact_write_size == 0 || num_valid_gds_readbacks.load() == 0 ||
        gpu_modified_ranges.Intersects(address, exact_write_size)) {
        return false;
    }
    const VAddr page = Common::AlignDown(address, 4_KB);
    if (HasGpuImageAlias(page, 4_KB)) {
        return false;
    }
    std::scoped_lock lk{async_readbacks_mutex};
    std::vector<std::shared_ptr<AsyncReadback>> page_readbacks;
    for (const auto* list : {&pending_async_readbacks, &inflight_async_readbacks}) {
        for (const auto& readback : *list) {
            if (readback->valid && readback->address < page + 4_KB &&
                page < readback->address + readback->size) {
                if (!readback->is_gds) {
                    return false;
                }
                page_readbacks.push_back(readback);
            }
        }
    }
    if (page_readbacks.empty()) {
        return false;
    }
    bool covered = true;
    gpu_modified_ranges.ForEachInRange(page, 4_KB, [&](VAddr start, VAddr end) {
        for (VAddr byte = start; byte < end;) {
            const auto it = std::ranges::find_if(page_readbacks, [&](const auto& readback) {
                return readback->address <= byte && byte < readback->address + readback->size;
            });
            if (it == page_readbacks.end()) {
                covered = false;
                return;
            }
            byte = (*it)->address + (*it)->size;
        }
    });
    if (!covered) {
        return false;
    }
    for (const auto& readback : page_readbacks) {
        if (readback->snapshot.empty()) {
            readback->snapshot.resize(readback->size);
            std::memcpy(readback->snapshot.data(), std::bit_cast<const void*>(readback->address),
                        readback->size);
        }
    }
    ++hot_page_stats.gds_split_faults;
    return true;
}

bool BufferCache::TrySplitWriteFault(const Buffer* arena, VAddr address, u64 size,
                                     VAddr window_start, VAddr window_end, u64 exact_write_size) {
    // PERF-011: the drain on a CPU write fault only protects GPU-written bytes of the page from
    // being overwritten by the later whole-page upload. When the write does not touch them, the
    // upload skips them instead (SynchronizeMemory) and their values reach guest memory
    // asynchronously, like the drain's window download did, without waiting for the GPU.
    // Relaxed mode only: GPU-written pages stay readable there, so the guest bytes can be
    // snapshotted here.
    if (!split_write_faults) {
        return false;
    }
    // With the store's exact size its bytes are checked; otherwise its 64-byte line.
    constexpr u64 LineSize = 64;
    const VAddr line = Common::AlignDown(address, LineSize);
    const VAddr line_end = Common::AlignUp(address + std::max<u64>(size, 1), LineSize);
    if (exact_write_size != 0 ? gpu_modified_ranges.Intersects(address, exact_write_size)
                              : gpu_modified_ranges.Intersects(line, line_end - line)) {
        return false;
    }
    if (HasGpuImageAlias(window_start, window_end - window_start)) {
        return false;
    }
    std::vector<std::shared_ptr<AsyncReadback>> recorded;
    memory_tracker->ForEachDownloadRange<false>(
        window_start, window_end - window_start, [&](u64 range_address, u64 range_size) {
            gpu_modified_ranges.ForEachInRange(
                range_address, range_size, [&](VAddr start, VAddr end) {
                    if (!memory->IsValidMapping(start, end - start)) {
                        return;
                    }
                    auto readback = std::make_shared<AsyncReadback>();
                    readback->address = start;
                    readback->size = static_cast<u32>(end - start);
                    readback->download =
                        staging_pool.Request(readback->size, MemoryType::HostCached, 4, true);
                    readback->snapshot.resize(readback->size);
                    std::memcpy(readback->snapshot.data(), std::bit_cast<const void*>(start),
                                readback->size);
                    const vk::BufferCopy copy = {
                        .srcOffset = start - arena->cpu_addr,
                        .dstOffset = readback->download.offset,
                        .size = readback->size,
                    };
                    runtime.CopyBuffer(arena, readback->download.buffer, std::span{&copy, 1});
                    recorded.push_back(std::move(readback));
                });
        });
    ++hot_page_stats.split_faults;
    if (!recorded.empty()) {
        std::scoped_lock lk{async_readbacks_mutex};
        pending_async_readbacks.insert(pending_async_readbacks.end(), recorded.begin(),
                                       recorded.end());
        num_pending_async_readbacks = static_cast<u32>(pending_async_readbacks.size());
        num_tracked_async_readbacks =
            static_cast<u32>(pending_async_readbacks.size() + inflight_async_readbacks.size());
    }
    return true;
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks,
                                   u64 exact_write_size) {
    NoteCpuChange(device_addr, size); // FIX-069
    static const bool readbacks_disabled =
        EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Disabled &&
        Common::PerfFeatureEnabled(91);
    if (readbacks_disabled) {
        // FIX-068: without readbacks a CPU write makes the CPU's bytes the current ones and the
        // next upload sends them to the GPU. The pages stopped being GPU-written only when a
        // readback flushed them, which never happens in this mode, so once written by the GPU
        // they stayed GPU-written for good: never streamed again, deferred fences behind every
        // GPU completion (GT Sport raced at 27 FPS against 42 with readbacks).
        memory_tracker->InvalidateRegion(device_addr, size, [] {});
        const VAddr first = Common::AlignDown(device_addr, 4_KB);
        const VAddr end = Common::AlignUp(device_addr + std::max<u64>(size, 1), 4_KB);
        std::scoped_lock buffer_lock{mutex};
        if (gpu_modified_ranges.Intersects(first, end - first)) {
            gpu_modified_ranges.Subtract(first, end - first);
            memory_tracker->UnmarkRegionAsGpuModified(first, end - first, false);
        }
        return;
    }
    memory_tracker->InvalidateRegion(
        device_addr, size, [this, device_addr, size, assume_locks, exact_write_size] {
            ReadMemory(device_addr, size, true, assume_locks, exact_write_size);
        });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks,
                             u64 exact_write_size) {
    RYDIAG_SCOPE(FaultFlush, device_addr);
    const auto flush_request = [this, device_addr, size, is_write, exact_write_size] {
        std::scoped_lock buffer_lock{mutex}; // PERF-067
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        ApplyCompletedReadbacks();
        NoteCpuReadFault(device_addr, size, exact_write_size);
        if (is_write && (TrySplitGdsWriteFault(device_addr, exact_write_size) ||
                         TrySplitWriteFault(arena, device_addr, size, window_start, window_end,
                                            exact_write_size))) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
            return;
        }
        if (hot_page_stats.drain_reports < 3) {
            ++hot_page_stats.drain_reports;
            const VAddr page = Common::AlignDown(device_addr, 4_KB);
            const auto it = small_gpu_writers.find(page);
            LOG_WARNING(Render_Vulkan,
                        "Drain for CPU {} at {:#x}+{:#x}: last small GPU write to the page: {}",
                        is_write ? "write" : "read", device_addr, exact_write_size,
                        it == small_gpu_writers.end()
                            ? std::string("none recorded")
                            : fmt::format("{} {:#x} at {:#x}+{:#x}", it->second.kind,
                                          it->second.tag, it->second.address, it->second.size));
        }
        DownloadMemory(arena, window_start, window_end - window_start, device_addr, size);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
        return;
    }
    // PERF-052 (opt-in, SHADGT_ASYNC_FAULTS=1): with the draw pipe, the recorder records the
    // download and goes on; this thread waits for the GPU, and the write to guest memory runs in
    // GPU order on the scheduler's completion thread. Off by default: the later write-back can
    // land over bytes the CPU wrote meanwhile on unprotected (hot) pages. In the Nurburgring
    // benchmark it corrupted a command buffer ("Invalid PM4 type 0") on the way to the race.
    // PERF-052b: on by default again with the snapshot guard below; the "Invalid PM4" crash
    // first blamed on it was FIX-049 (fence order). SHADGT_ASYNC_FAULTS=0 or -DisablePerf 64
    // turns it off.
    static const bool async_flush = [] {
        const char* env = std::getenv("SHADGT_ASYNC_FAULTS");
        return !(env && env[0] == '0') && Common::PerfFeatureEnabled(64);
    }();
    if (async_flush && liverpool->Pipelined()) {
        std::shared_ptr<FaultDownload> fault;
        liverpool->SendFaultCommand(
            [&] { fault = BeginFaultFlush(device_addr, size, is_write, exact_write_size); });
        if (!fault) {
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        fault->done.acquire();
        async_fault_wait_us.fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - start)
                                 .count()),
            std::memory_order_relaxed);
        async_fault_flushes.fetch_add(1, std::memory_order_relaxed);
        liverpool->SendFaultCommand([&] { FinishFaultFlush(*fault); });
        return;
    }
    liverpool->SendFaultCommand(std::move(flush_request));
}

void BufferCache::NoteGpuWrite(VAddr address, u64 size) {
    // PERF-052: only needed while fault downloads are pending.
    if (num_fault_downloads.load(std::memory_order_relaxed) == 0) {
        return;
    }
    gpu_write_log.emplace_back(++gpu_write_seq, address, address + size);
}

void BufferCache::WaitForFaultWritebacks(u64 tick) {
    if (num_fault_downloads.load(std::memory_order_acquire) == 0) {
        return;
    }
    RYDIAG_SCOPE(ReadbackWait, tick);
    std::vector<std::shared_ptr<FaultDownload>> older;
    {
        std::scoped_lock lk{fault_downloads_mutex};
        for (const auto& fault : fault_downloads) {
            if (fault->tick < tick) {
                older.push_back(fault);
            }
        }
    }
    for (const auto& fault : older) {
        while (!fault->applied.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
}

// FIX-069: GT Sport rewrites its per-frame memory every frame, often on hot pages it writes
// without faulting. GPU results recorded over that memory frames earlier were later copied
// back over the newer CPU bytes (a fault flush or a hot-page readback only checks that guest
// memory did not change after the download was recorded): an earlier frame's vertex data
// over the race summary's draw constants and descriptor tables (the driver figure flickered
// and glowed green). CPU changes (write faults, hot pages found changed) and GPU writes now
// share one sequence; a part whose page the CPU changed after the GPU wrote it keeps the CPU's
// bytes. -DisablePerf 92 copies back as before.
static bool WriterOrderEnabled() {
    static const bool enabled = Common::PerfFeatureEnabled(92);
    return enabled;
}

// FIX-076: the sequence is the steady clock in nanoseconds (kept increasing), so a CPU change
// and a GPU write can also be compared in time (see SplitByWriter).
u64 BufferCache::NextWriteSeq() {
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    u64 prev = write_seq.load(std::memory_order_relaxed);
    while (true) {
        const u64 next = std::max(prev + 1, now);
        if (write_seq.compare_exchange_weak(prev, next, std::memory_order_relaxed)) {
            return next;
        }
    }
}

// FIX-076: whether the GPU wrote this range less than 50 ms ago (results a frame's effects read
// back), with GT Sport's particle effects on.
bool BufferCache::FreshGpuWrite(VAddr start, VAddr end) {
    static const bool enabled =
        WriterOrderEnabled() && Common::PerfFeatureEnabled(100) &&
        EmulatorSettings.IsParticleEffects();
    if (!enabled) {
        return false;
    }
    u64 oldest = ~u64{0};
    bool any = false;
    gpu_write_seqs.ForEachInRange(start, end - start, [&](VAddr, VAddr, u64 seq) {
        oldest = std::min(oldest, seq);
        any = true;
    });
    return any && NextWriteSeq() - oldest < 50'000'000;
}

void BufferCache::NoteCpuChange(VAddr address, u64 size) {
    if (!WriterOrderEnabled()) {
        return;
    }
    const u64 seq = NextWriteSeq();
    const VAddr end = address + std::max<u64>(size, 1);
    std::scoped_lock lk{cpu_change_mutex};
    if (cpu_change_seqs.size() > (1u << 21)) {
        cpu_change_seqs.clear();
    }
    for (VAddr page = Common::AlignDown(address, 4_KB); page < end; page += 4_KB) {
        cpu_change_seqs[page] = seq;
    }
}

void BufferCache::NoteGpuWriteSeq(VAddr address, u64 size) {
    if (!WriterOrderEnabled() || size == 0) {
        return;
    }
    const u64 seq = NextWriteSeq();
    gpu_write_seqs.Subtract(address, size);
    gpu_write_seqs.Add(address, size, seq);
    NoteGpuWriteOverReadbackPages(address, size); // FIX-073
}

// FIX-073: hot-page readbacks (PERF-010) copied GPU data back over CPU writes. The CPU rewrites
// GT Sport's per-frame memory on these pages without faulting (they are not write-protected), so
// its writes never reached the GPU copy. A shader writing part of the page was bound over the
// whole range, the readback copied the whole range back, and the bytes the shader did not write
// came back stale: an earlier frame's vertex data over the race summary's vertex buffer tables
// (garbage V#s, FIX-053; DIAG-071/072 traced it). A faster recorder made it common. Each readback
// page keeps the guest bytes the GPU copy last took in; before a shader writes over the page, the
// bytes the CPU changed since are copied to the GPU, and a readback is not recorded over bytes
// the CPU changed after that. -DisablePerf 99 copies back as before.
void BufferCache::NoteSnapshotUpdate(VAddr page, const char* reason) {
    // DIAG-073 (under readback_hash_mutex)
    if (TraceReadbacks()) {
        if (snapshot_updates.size() > 8192) {
            snapshot_updates.clear();
        }
        snapshot_updates[page] = {TraceNowNs(), reason};
    }
}

std::string BufferCache::DescribeReadbackPage(VAddr address) {
    const VAddr page = Common::AlignDown(address, 4_KB);
    const s64 now = TraceNowNs();
    std::scoped_lock lk{readback_hash_mutex};
    std::string out = fmt::format("page {:#x}:", page);
    if (const auto it = hot_page_since_ns.find(page); it != hot_page_since_ns.end()) {
        out += fmt::format(" hot since {:.1f} ms ago;", (now - it->second) / 1e6);
    } else {
        out += " not seen hot;";
    }
    out += pending_snapshot_pages.contains(page) ? " snapshot pending;" : "";
    if (const auto it = readback_page_snapshots.find(page);
        it != readback_page_snapshots.end() && it->second) {
        std::array<u8, 4096> guest;
        Common::CopyGuest(guest.data(), std::bit_cast<const void*>(page), 4_KB);
        u32 differ = 0;
        for (u32 i = 0; i < 4096; ++i) {
            differ += guest[i] != (*it->second)[i];
        }
        out += fmt::format(" snapshot differs from guest in {} bytes;", differ);
    } else {
        out += " no snapshot;";
    }
    if (const auto it = snapshot_updates.find(page); it != snapshot_updates.end()) {
        out += fmt::format(" snapshot last set {:.1f} ms ago by {}", (now - it->second.first) / 1e6,
                           it->second.second);
    }
    return out;
}

void BufferCache::NoteGpuWriteOverReadbackPages(VAddr address, u64 size) {
    if (!ReadbackHashEnabled() || hot_page_order.empty()) {
        return;
    }
    const VAddr end = address + size;
    for (const VAddr page : hot_page_order) {
        if (page >= end || address >= page + 4_KB) {
            continue;
        }
        std::scoped_lock lk{readback_hash_mutex};
        auto& snapshot = readback_page_snapshots[page];
        if (!snapshot) {
            if (readback_page_snapshots.size() > 4096) {
                readback_page_snapshots.clear();
                continue;
            }
            // First seen: the GPU copy is taken to match guest memory.
            snapshot = std::make_unique<std::array<u8, 4096>>();
            Common::CopyGuest(snapshot->data(), std::bit_cast<const void*>(page), 4_KB);
            pending_snapshot_pages.erase(page);
            NoteSnapshotUpdate(page, "GPU write (first)");
        }
    }
}

void BufferCache::SyncUnseenCpuWrites(const Buffer* arena, VAddr address, u64 size) {
    if (!ReadbackHashEnabled() || hot_page_order.empty()) {
        return;
    }
    const VAddr end = address + size;
    std::array<u8, 4096> now;
    for (const VAddr page : hot_page_order) {
        if (page >= end || address >= page + 4_KB) {
            continue;
        }
        std::scoped_lock lk{readback_hash_mutex};
        const auto it = readback_page_snapshots.find(page);
        if (it == readback_page_snapshots.end() || !it->second) {
            continue;
        }
        auto& snapshot = *it->second;
        Common::CopyGuest(now.data(), std::bit_cast<const void*>(page), 4_KB);
        const u32 last = static_cast<u32>(std::min(end, page + 4_KB) - page);
        u32 i = static_cast<u32>(std::max(address, page) - page);
        while (i < last) {
            while (i < last && now[i] == snapshot[i]) {
                ++i;
            }
            if (i >= last) {
                break;
            }
            u32 j = i;
            while (j < last && now[j] != snapshot[j]) {
                ++j;
            }
            // vkCmdUpdateBuffer works in 4-byte units.
            const u32 lo = i & ~3u;
            const u32 hi = std::min<u32>((j + 3) & ~3u, 4096);
            runtime.InlineData(arena, page + lo - arena->cpu_addr,
                               std::span<const u8>{now.data() + lo, hi - lo});
            std::memcpy(snapshot.data() + lo, now.data() + lo, hi - lo);
            unseen_cpu_bytes_synced += hi - lo;
            NoteSnapshotUpdate(page, "unseen CPU write sync");
            i = j;
        }
    }
}

bool BufferCache::CpuRewroteReadbackRange(VAddr start, VAddr end) {
    if (!ReadbackHashEnabled()) {
        return false;
    }
    std::scoped_lock lk{readback_hash_mutex};
    for (VAddr page = Common::AlignDown(start, 4_KB); page < end; page += 4_KB) {
        const auto it = readback_page_snapshots.find(page);
        if (it == readback_page_snapshots.end() || !it->second) {
            continue;
        }
        const VAddr s = std::max(start, page);
        const VAddr e = std::min<VAddr>(end, page + 4_KB);
        // Page tracking can protect the guest view while GPU data is pending.
        // Match RefreshReadbackPageHashes: read through the sparse/alias copy path
        // instead of faulting in memcmp while the recorder holds cache locks.
        std::array<u8, 4096> current;
        memory->CopySparseMemory(s, current.data(), e - s);
        if (std::memcmp(current.data(), it->second->data() + (s - page), e - s) != 0) {
            return true;
        }
    }
    return false;
}

void BufferCache::RefreshReadbackPageHashes(VAddr address, u64 size) {
    if (!ReadbackHashEnabled()) {
        return;
    }
    // The renderer wrote GPU data into guest memory (or the GPU copy took in guest bytes): those
    // bytes now match the GPU copy.
    std::scoped_lock lk{readback_hash_mutex};
    if (readback_page_snapshots.empty() && pending_snapshot_pages.empty()) {
        return;
    }
    const VAddr end = address + std::max<u64>(size, 1);
    const auto refresh = [&](VAddr page, std::array<u8, 4096>& snapshot) {
        const VAddr s = std::max(address, page);
        const VAddr e = std::min<VAddr>(end, page + 4_KB);
        // FIX-107: through CopySparseMemory, which reads tracked and guarded pages through an
        // alias view: a fault here (often on the readback thread holding the buffer-cache lock)
        // waited for the recorder, which waited for that lock (early fences froze).
        memory->CopySparseMemory(s, snapshot.data() + (s - page), e - s);
        NoteSnapshotUpdate(page, "renderer write or upload");
    };
    const VAddr first = Common::AlignDown(address, 4_KB);
    // A page that became hot at a CPU fault gets its snapshot once the fault's GPU data is in
    // guest memory, before the faulting thread goes on to write it.
    for (auto it = pending_snapshot_pages.begin(); it != pending_snapshot_pages.end();) {
        const VAddr page = *it;
        if (page < end && first < page + 4_KB) {
            auto& snapshot = readback_page_snapshots[page];
            if (!snapshot) {
                snapshot = std::make_unique<std::array<u8, 4096>>();
                memory->CopySparseMemory(page, snapshot->data(), 4_KB); // FIX-107
                NoteSnapshotUpdate(page, "fault write-back");
            }
            it = pending_snapshot_pages.erase(it);
        } else {
            ++it;
        }
    }
    if ((end - first) / 4_KB > readback_page_snapshots.size()) {
        for (auto& [page, snapshot] : readback_page_snapshots) {
            if (snapshot && page < end && first < page + 4_KB) {
                refresh(page, *snapshot);
            }
        }
        return;
    }
    for (VAddr page = first; page < end; page += 4_KB) {
        const auto it = readback_page_snapshots.find(page);
        if (it != readback_page_snapshots.end() && it->second) {
            refresh(page, *it->second);
        }
    }
}

template <typename Keep, typename Drop>
void BufferCache::SplitByWriter(VAddr start, VAddr end, Keep&& keep, Drop&& drop,
                                bool keep_fresh) {
    if (!WriterOrderEnabled()) {
        keep(start, end);
        return;
    }
    std::scoped_lock lk{cpu_change_mutex};
    VAddr run_start = start;
    bool run_cpu = false;
    for (VAddr page = Common::AlignDown(start, 4_KB); page < end; page += 4_KB) {
        const VAddr s = std::max(start, page);
        const VAddr e = std::min<VAddr>(end, page + 4_KB);
        bool cpu_newer = false;
        if (const auto it = cpu_change_seqs.find(page); it != cpu_change_seqs.end()) {
            u64 gpu = ~u64{0};
            bool any = false;
            gpu_write_seqs.ForEachInRange(s, e - s, [&](VAddr, VAddr, u64 seq) {
                gpu = std::min(gpu, seq);
                any = true;
            });
            // FIX-076: GT Sport's effects (tire smoke, sparks) read GPU results that share a page
            // with bytes the CPU writes within the same frame; FIX-069 kept the CPU's page and
            // the effects never saw their results, so they were never drawn. The CPU's page wins
            // only when its change came at least FreshGpuNs after the GPU write (the race
            // summary's stale data came frames later). -DisablePerf 100 drops them as before.
            // Only CPU accesses (fault write-backs) keep fresh GPU results; hot-page readbacks,
            // which run ahead of any access, still drop them (they cost frames as before).
            static const bool fresh_enabled =
                Common::PerfFeatureEnabled(100) && EmulatorSettings.IsParticleEffects();
            constexpr u64 FreshGpuNs = 50'000'000;
            cpu_newer = !any || (it->second > gpu && (!keep_fresh || !fresh_enabled ||
                                                      it->second - gpu >= FreshGpuNs));
        }
        if (s == start) {
            run_cpu = cpu_newer;
        } else if (cpu_newer != run_cpu) {
            if (run_cpu) {
                drop(run_start, s);
            } else {
                keep(run_start, s);
            }
            run_start = s;
            run_cpu = cpu_newer;
        }
    }
    if (run_cpu) {
        drop(run_start, end);
    } else {
        keep(run_start, end);
    }
}

// FIX-063: a CPU fault writes GPU results back for the whole 512 KB window around it. On a page
// nobody watches (a hot page tracked by hashing), the CPU writes without faulting, so its bytes
// can be newer than GPU-written ones still recorded there; writing those back put stale GPU data
// over them. GT Sport reuses its per-frame memory (draw constants and descriptor tables) where an
// earlier frame's GPU work wrote vertex data, and a flush for a fault nearby put that vertex data
// back over the new frame's tessellation constants and tables (the race summary's driver figure
// flickered and glowed green). Only the faulting pages and pages still watched get GPU bytes;
// the others keep the CPU's. -DisablePerf 86 writes back the whole window as before.
static void SplitWriteback(VAddr start, VAddr end, VAddr fault_addr, u64 fault_size,
                           const PageManager& pages,
                           const std::function<void(VAddr, VAddr)>& keep,
                           const std::function<void(VAddr, VAddr)>& drop,
                           const std::function<bool(VAddr, VAddr)>& fresh = {}) {
    static const bool enabled = Common::PerfFeatureEnabled(86);
    // -DisablePerf 87 also writes back pages still watched, as in the first version.
    static const bool faulting_pages_only = Common::PerfFeatureEnabled(87);
    if (!enabled) {
        keep(start, end);
        return;
    }
    const VAddr fault_first = Common::AlignDown(fault_addr, 4_KB);
    const VAddr fault_end = Common::AlignUp(fault_addr + std::max<u64>(fault_size, 1), 4_KB);
    VAddr run_start = start;
    bool run_kept = true;
    for (VAddr page = Common::AlignDown(start, 4_KB); page < end; page += 4_KB) {
        const VAddr s = std::max(start, page);
        // Only the pages being faulted on: GPU-written ranges elsewhere in the window can be
        // whole bound buffers whose GPU copy is older game data (GT Sport binds buffers of
        // hundreds of MB), and they get their own fault when the game touches them.
        // FIX-076: also pages whose GPU results are fresh (the effects' smoke reads them on
        // other pages of the window without faulting there).
        const bool kept = (page >= fault_first && page < fault_end) ||
                          (!faulting_pages_only && !pages.IsUnwatched(page)) ||
                          (fresh && fresh(s, std::min<VAddr>(end, page + 4_KB)));
        if (s == start) {
            run_kept = kept;
        } else if (kept != run_kept) {
            (run_kept ? keep : drop)(run_start, s);
            run_start = s;
            run_kept = kept;
        }
    }
    (run_kept ? keep : drop)(run_start, end);
}

std::shared_ptr<BufferCache::FaultDownload> BufferCache::BeginFaultFlush(VAddr device_addr,
                                                                         u64 size, bool is_write,
                                                                         u64 exact_write_size) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    // As ReadMemory's flush_request up to DownloadMemory's GPU wait.
    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    constexpr u64 WindowSize = 512_KB;
    const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
    const VAddr window_end =
        std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
    ApplyCompletedReadbacks();
    NoteCpuReadFault(device_addr, size, exact_write_size);
    if (is_write && (TrySplitGdsWriteFault(device_addr, exact_write_size) ||
                     TrySplitWriteFault(arena, device_addr, size, window_start, window_end,
                                        exact_write_size))) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        return nullptr;
    }
    auto fault = std::make_shared<FaultDownload>();
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    std::vector<std::pair<VAddr, VAddr>> dropped; // FIX-063
    std::vector<std::pair<VAddr, VAddr>> cpu_newer; // FIX-069
    memory_tracker->ForEachDownloadRange<false>(
        window_start, window_end - window_start, [&](u64 address, u64 range_size) {
            gpu_modified_ranges.ForEachInRange(address, range_size, [&](VAddr start_all,
                                                                        VAddr end_all) {
                SplitByWriter(start_all, end_all, [&](VAddr start, VAddr end) {
                SplitWriteback(
                    start, end, device_addr, size, page_manager,
                    [&](VAddr s, VAddr e) {
                        fault->copies.push_back(vk::BufferCopy{
                            .srcOffset = s - arena_base,
                            .dstOffset = total_size_bytes,
                            .size = e - s,
                        });
                        fault->ranges.emplace_back(s, e);
                        total_size_bytes += Common::AlignUp(e - s, 64);
                    },
                    [&](VAddr s, VAddr e) { dropped.emplace_back(s, e); },
                    [this](VAddr s, VAddr e) { return FreshGpuWrite(s, e); });
                }, [&](VAddr s, VAddr e) { cpu_newer.emplace_back(s, e); }, true);
            });
        });
    for (const auto& [s, e] : cpu_newer) {
        // FIX-069: the CPU changed these bytes after the GPU wrote them; they are the CPU's.
        gpu_modified_ranges.Subtract(s, e - s);
        ++writebacks_cpu_newer;
    }
    for (const auto& [s, e] : dropped) {
        // On an unwatched page the CPU's bytes are the current ones now; a watched page keeps
        // its GPU-written bytes until a fault on it flushes them.
        if (page_manager.IsUnwatched(Common::AlignDown(s, 4_KB))) {
            gpu_modified_ranges.Subtract(s, e - s);
        }
    }
    if (total_size_bytes == 0) {
        // Nothing GPU-written: DownloadMemory would return at once.
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
        return nullptr;
    }
    fault->device_addr = device_addr;
    fault->size = size;
    fault->is_write = is_write;
    fault->window_start = window_start;
    fault->window_end = window_end;
    if (!dropped.empty()) {
        // FIX-063: only the faulting pages stop being GPU-written when this flush finishes.
        fault->window_start = std::max(window_start, Common::AlignDown(device_addr, 4_KB));
        fault->window_end = std::min(window_end, Common::AlignUp(device_addr + size, 4_KB));
    }
    fault->download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 0, true);
    for (auto& copy : fault->copies) {
        copy.dstOffset += fault->download.offset;
    }
    runtime.CopyBuffer(arena, fault->download.buffer, fault->copies);
    fault->snapshots.reserve(fault->copies.size());
    for (const auto& copy : fault->copies) {
        const u8* guest = std::bit_cast<const u8*>(arena_base + copy.srcOffset);
#ifdef RYTY_GPU
        // Memory the game released or protected since the GPU wrote it is neither read nor
        // written back (an empty snapshot skips the copy).
        if (!Core::RytyMemory::IsGuestReadable(arena_base + copy.srcOffset, copy.size)) {
            fault->snapshots.emplace_back();
            continue;
        }
#endif
        fault->snapshots.emplace_back(guest, guest + copy.size);
    }
    // A readback recorded before this download must not complete over it.
    InvalidateAsyncReadbacks(window_start, window_end - window_start);
    fault->seq = gpu_write_seq;
    fault->tick = scheduler.CurrentTick();
    {
        std::scoped_lock lk{fault_downloads_mutex};
        fault_downloads.push_back(fault);
        num_fault_downloads.store(static_cast<u32>(fault_downloads.size()),
                                  std::memory_order_release);
    }
    scheduler.DeferPriorityOperation([this, fault, arena_base] {
        fault->download.buffer->Invalidate(fault->download.offset, fault->download.size);
        std::scoped_lock lk{fault_downloads_mutex};
        for (size_t c = 0; c < fault->copies.size(); ++c) {
            const auto& copy = fault->copies[c];
            const VAddr start = arena_base + copy.srcOffset;
            const u8* src = fault->download.mapped + (copy.dstOffset - fault->download.offset);
            if (fault->snapshots[c].empty()) {
                continue;
            }
#ifdef RYTY_GPU
            if (!Core::RytyMemory::IsGuestReadable(start, copy.size)) {
                continue;
            }
#endif
            const u8* snapshot = fault->snapshots[c].data();
            // FIX-108: the guest bytes are compared from a copy CopySparseMemory takes through
            // an alias view where pages are protected: a fault here, with fault_downloads_mutex
            // held, waited for the recorder, which waited for that lock (early fences froze).
            thread_local std::vector<u8> guest_copy;
            guest_copy.resize(copy.size);
            memory->CopySparseMemory(start, guest_copy.data(), copy.size);
            const u8* guest = guest_copy.data();
            for (const auto& [s, e] : PartsWithoutNewerWrites(start, start + copy.size,
                                                              fault->released)) {
                // PERF-052b: only bytes still as they were when the download was recorded; a
                // byte the CPU wrote since (an unprotected page) keeps the CPU's value.
                VAddr run = s;
                while (run < e) {
                    while (run < e && guest[run - start] != snapshot[run - start]) {
                        ++run;
                    }
                    VAddr run_end = run;
                    while (run_end < e && guest[run_end - start] == snapshot[run_end - start]) {
                        ++run_end;
                    }
                    if (run_end > run) {
                        memory->TryWriteBacking(std::bit_cast<u8*>(run), src + (run - start),
                                                run_end - run);
                    }
                    run = run_end;
                }
            }
        }
        fault->applied.store(true, std::memory_order_release);
        fault->done.release();
    });
    scheduler.Flush();
    last_drain = std::chrono::steady_clock::now();
    return fault;
}

void BufferCache::FinishFaultFlush(FaultDownload& fault) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    // The rest of DownloadMemory and of the flush, for the bytes no GPU write has changed since
    // the download: those stay GPU-written and their guest copy is stale, as for any GPU write.
    boost::container::small_vector<std::pair<VAddr, VAddr>, 8> newer;
    for (const auto& [seq, start, end] : gpu_write_log) {
        if (seq > fault.seq && start < fault.window_end && fault.window_start < end) {
            newer.emplace_back(start, end);
        }
    }
    const std::span<const std::pair<VAddr, VAddr>> newer_writes{newer.data(), newer.size()};
    std::vector<std::pair<VAddr, VAddr>> released;
    for (const auto& [start, end] : fault.ranges) {
        for (const auto& [s, e] : PartsWithoutNewerWrites(start, end, newer_writes)) {
            gpu_modified_ranges.Subtract(s, e - s);
            released.emplace_back(s, e);
        }
    }
    {
        // Later fault downloads of these bytes that have not reached guest memory yet hold the
        // same GPU bytes; once this flush returns the CPU may write them, so they skip them.
        std::scoped_lock lk{fault_downloads_mutex};
        for (const auto& other : fault_downloads) {
            if (other.get() != &fault && other->tick >= fault.tick &&
                !other->applied.load(std::memory_order_acquire)) {
                other->released.insert(other->released.end(), released.begin(), released.end());
            }
        }
    }
    if (newer.empty()) {
        memory_tracker->UnmarkRegionAsGpuModified(fault.window_start,
                                                  fault.window_end - fault.window_start, false);
    } else {
        for (VAddr page = Common::AlignDown(fault.window_start, 4_KB); page < fault.window_end;
             page += 4_KB) {
            const VAddr start = std::max(page, fault.window_start);
            const VAddr end = std::min<VAddr>(page + 4_KB, fault.window_end);
            if (!TouchedByNewerWrite(page, page + 4_KB, newer_writes)) {
                memory_tracker->UnmarkRegionAsGpuModified(start, end - start, false);
            }
        }
    }
    if (fault.is_write) {
        memory_tracker->MarkRegionAsCpuModified(fault.device_addr, fault.size);
    }
    staging_pool.FreeDeferred(fault.download);
    std::scoped_lock lk{fault_downloads_mutex};
    std::erase_if(fault_downloads, [&](const auto& f) { return f.get() == &fault; });
    num_fault_downloads.store(static_cast<u32>(fault_downloads.size()), std::memory_order_release);
    if (fault_downloads.empty()) {
        gpu_write_log.clear();
    }
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size,
                                 VAddr fault_addr, u64 fault_size) {
    // FIX-063: GPU-written parts not written back here, on pages still watched.
    std::vector<std::pair<VAddr, VAddr>> kept_gpu;
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, [&](VAddr start_all, VAddr end_all) {
            SplitByWriter(start_all, end_all, [&](VAddr start, VAddr end) {
            SplitWriteback(start, end, fault_addr, fault_size, page_manager, add_download,
                           [&](VAddr s, VAddr e) { kept_gpu.emplace_back(s, e); }, // FIX-063
                           [this](VAddr s, VAddr e) { return FreshGpuWrite(s, e); });
            }, [&](VAddr, VAddr) { ++writebacks_cpu_newer; }); // FIX-069
        });
        gpu_modified_ranges.Subtract(address, size);
    });
    if (total_size_bytes == 0) {
        return;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    const u64 download_tick = scheduler.CurrentTick();
    scheduler.Finish();
    // PERF-052: older fault downloads reach guest memory first, so they cannot land over this.
    WaitForFaultWritebacks(download_tick);
    last_drain = std::chrono::steady_clock::now();
    // This download is current; a readback recorded earlier must not complete over it after the
    // CPU has written the range.
    InvalidateAsyncReadbacks(device_addr, size);

    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& copy : copies) {
        auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
        NoteWritebackOverCommands(liverpool, "fault download", arena_base + copy.srcOffset,
                                  copy.size);
        memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                copy.size);
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
    for (const auto& [s, e] : kept_gpu) {
        if (!page_manager.IsUnwatched(Common::AlignDown(s, 4_KB))) {
            gpu_modified_ranges.Add(s, e - s);
            memory_tracker->MarkRegionAsGpuModified(s, e - s);
        }
    }
}

// DIAG-111 (with SHADGT_DIAG_OBTAIN=1): cycles spent in obtains by kind, logged with DIAG-109.
static std::atomic<u64> g_obtain_cycles[4]{}; // stream, arena read, arena written, in place (P2)
static std::atomic<u64> g_obtain_counts[4]{};

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    // EXP-127 (SHADGT_CEIL_NOBUFFERS=1, measurement only, rendering goes wrong): 190 s in, every
    // buffer obtain returns a 1 GB dummy buffer without any other work: the recorder's cost per
    // draw if buffer memory needed no CPU work at all (the direct renderer's model).
    static const bool ceil_nobuffers = [] {
        const char* env = std::getenv("SHADGT_CEIL_NOBUFFERS");
        return env && env[0] == '1';
    }();
    if (ceil_nobuffers && t_ceil_draw && !is_written && size <= 1_GB && !is_texel_buffer) {
        static const auto start = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds{190}) {
            static Buffer dummy{instance, 0, 1_GB, MemoryType::DeviceLocal, "EXP-127 dummy"};
            return {&dummy, 0};
        }
    }
    static const bool diag_time = [] {
        const char* env = std::getenv("SHADGT_DIAG_OBTAIN");
        return env && env[0] == '1';
    }();
    if (diag_time) {
        const u64 start = __rdtsc();
        const auto result = ObtainBufferImpl(device_addr, size, is_written, is_texel_buffer);
        const int kind = result.first == &stream_buffer ? 0
                         : result.first->mem_type == MemoryType::HostUncached ? 3 // P2 in place
                         : is_written ? 2 : 1;
        g_obtain_cycles[kind] += __rdtsc() - start;
        // DIAG-115: arena read cycles by 16 MB block (top blocks every 2 s, with DIAG-109).
        if (kind == 1) {
            static std::unordered_map<VAddr, std::pair<u64, u64>> by_block;
            static auto window = std::chrono::steady_clock::now();
            auto& entry = by_block[device_addr >> 24];
            entry.first += __rdtsc() - start;
            ++entry.second;
            if (const auto now = std::chrono::steady_clock::now();
                now - window >= std::chrono::seconds{2}) {
                window = now;
                std::vector<std::pair<VAddr, std::pair<u64, u64>>> top(by_block.begin(), by_block.end());
                std::ranges::sort(top, std::greater{}, [](const auto& e) { return e.second.first; });
                std::string text;
                for (size_t i = 0; i < std::min<size_t>(top.size(), 10); ++i) {
                    text += fmt::format(" {:#x}: {} Mcycles/{} (view {:#x})", top[i].first << 24,
                                        top[i].second.first / 1000000, top[i].second.second,
                                        GuestViewBase(top[i].first << 24));
                }
                LOG_WARNING(Render_Vulkan, "DIAG-115: arena reads by 16 MB block:{}", text);
                by_block.clear();
            }
        }
        ++g_obtain_counts[kind];
        return result;
    }
    return ObtainBufferImpl(device_addr, size, is_written, is_texel_buffer);
}

std::optional<bool> BufferCache::ImagesOverQuick(VAddr device_addr, u32 size) const {
    // R6 (recorder): SynchronizeMemoryFromImage's answer when its memos have it and nothing is
    // to be exported; nothing when the images must be walked or exported.
    if (size == 0) {
        return false;
    }
    if (!Common::PerfFeatureEnabled(67)) {
        return std::nullopt;
    }
    const u64 generation = texture_cache.RegistryGeneration();
    const auto& slot =
        no_image_memo[((device_addr >> 6) ^ (device_addr >> 21) ^ size) % no_image_memo.size()];
    if (slot.address == device_addr && slot.size == size && slot.generation == generation) {
        return false;
    }
    if (Common::PerfFeatureEnabled(77)) {
        constexpr u32 BlockBits = 16;
        const VAddr first = device_addr >> BlockBits;
        const VAddr last = (device_addr + size - 1) >> BlockBits;
        bool all_free = last - first < 16;
        for (VAddr block = first; all_free && block <= last; ++block) {
            const auto& entry = no_image_blocks[block % no_image_blocks.size()];
            all_free = entry.block == block && entry.generation == generation;
        }
        if (all_free) {
            return false;
        }
    }
    if (Common::PerfFeatureEnabled(98)) {
        const u64 export_state = g_image_export_state.load(std::memory_order_relaxed);
        const auto& quiet =
            quiet_memo[((device_addr >> 8) ^ (device_addr >> 24) ^ size) % quiet_memo.size()];
        if (quiet.address == device_addr && quiet.size == size &&
            quiet.generation == generation && quiet.export_state == export_state) {
            return true;
        }
    }
    return std::nullopt;
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::ObtainBufferShared(
    VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer) {
    // R6 (recorder, cache lock held shared): the common obtain, while the stage workers keep
    // obtaining. Nothing (and nothing changed but recorded ahead uploads) when it needs an
    // image walk or export, a new arena, residency or metadata: the exclusive path then.
    if (is_texel_buffer) {
        return std::nullopt;
    }
    const auto images = ImagesOverQuick(device_addr, size);
    if (!images) {
        return std::nullopt;
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    const bool stream = !is_written && size <= STREAM_THRESHOLD && !*images;
    const Buffer* arena = address_space[first_page];
    if (!stream) {
        if (!arena || arena != address_space[last_page]) {
            return std::nullopt;
        }
        bool resident = true;
        resident_ranges.ForEachGap(first_block, last_block + 1,
                                   [&](u64, u64) { resident = false; });
        if (!resident) {
            return std::nullopt;
        }
    }
    RecordPendingAheadUploads(device_addr, size);
    if (is_written) {
        InvalidateAsyncReadbacks(device_addr, size);
    }
    ApplyCompletedReadbacks();
    ++Common::GetWorkCounters().obtain_buffer;
    if (stream && !IsRegionGpuModified(device_addr, size)) {
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size, read_capture);
        stream_buffer.Commit();
        ++Common::GetWorkCounters().obtain_stream;
        return std::pair<const Buffer*, u64>{&stream_buffer, offset};
    }
    if (!arena || arena != address_space[last_page]) {
        return std::nullopt; // GPU-written small range without an arena yet
    }
    if (stream) {
        bool resident = true;
        resident_ranges.ForEachGap(first_block, last_block + 1,
                                   [&](u64, u64) { resident = false; });
        if (!resident) {
            return std::nullopt;
        }
    }
    SynchronizeMemory(arena, device_addr, size, is_written, false);
    if (is_written) {
        SyncUnseenCpuWrites(arena, device_addr, size); // FIX-073
        {
            std::scoped_lock gpu_lk{gpu_ranges_mutex};
            gpu_modified_ranges.Add(device_addr, size);
        }
        NoteGpuWriteSeq(device_addr, size); // FIX-069
        NoteGpuWrite(device_addr, size);
        if (size <= 4_KB) {
            if (small_gpu_writers.size() > 65536) {
                small_gpu_writers.clear();
            }
            small_gpu_writers[Common::AlignDown(device_addr, 4_KB)] =
                GpuWriter{g_gpu_write_kind, g_gpu_write_tag, device_addr, size};
        } else {
            if (large_gpu_writers.size() > 16384) {
                large_gpu_writers.clear();
            }
            large_gpu_writers[device_addr] =
                GpuWriter{g_gpu_write_kind, g_gpu_write_tag, device_addr, size};
        }
    }
    return std::pair<const Buffer*, u64>{arena, arena->Offset(device_addr)};
}

const Buffer* BufferCache::TryImportView(VAddr device_addr, u32 size) {
    // P2: up to 512 views of at most 64 MB without images or GPU writes. A view refused for
    // what it holds is not tried again; a range reaching past its view is left as it is.
    std::scoped_lock lk{import_mutex};
    if (imported_buffers.size() >= 512) {
        return nullptr;
    }
    VAddr begin{}, end{};
    // The view's extent, remembered per 2 MB block (VirtualQuery walks cost ~1 us each).
    auto& extent = view_extents[device_addr >> 21];
    if (extent.second == 0) {
        extent.first = GuestViewBase(device_addr);
        extent.second = extent.first ? GuestViewEnd(extent.first, 64_MB) : 1;
    }
    const VAddr base = extent.first;
    if (base == 0 || device_addr + size > extent.second || import_tried.contains(base)) {
        return nullptr;
    }
    import_tried.insert(base);
    // SHADGT_PROTO_INPLACE_MAX=<hex address>: only views below it (the game decommits pool
    // memory higher up, and an imported page made inaccessible loses the device).
    static const VAddr limit = [] {
        const char* env = std::getenv("SHADGT_PROTO_INPLACE_MAX");
        return env ? static_cast<VAddr>(std::strtoull(env, nullptr, 16)) : ~VAddr{0};
    }();
    if (base >= limit) {
        ++import_rejected;
        return nullptr;
    }
    const VAddr view_end = GuestViewEnd(base, 64_MB);
    if (view_end <= base || view_end - base > 64_MB ||
        texture_cache.AnyImageInRegion(base, view_end - base) ||
        IsRegionGpuModified(base, view_end - base)) {
        if (++import_rejected <= 30) {
            LOG_WARNING(Render_Vulkan, "P2: view {:#x}+{:#x} not imported (images {}, GPU-written {})",
                        base, view_end - base,
                        view_end > base && texture_cache.AnyImageInRegion(base, view_end - base),
                        view_end > base && IsRegionGpuModified(base, view_end - base));
        }
        return nullptr;
    }
    const vk::Buffer handle = ImportGuestView(instance, device_addr, size, 64_MB, &begin, &end);
    if (!handle) {
        ++import_rejected;
        return nullptr;
    }
    auto& buffer = imported_buffers.emplace_back(
        std::make_unique<Buffer>(instance, begin, end - begin, handle));
    {
        std::unique_lock views_lk{g_imported_mutex};
        g_imported_views.push_back({begin, end, buffer.get()});
        std::ranges::sort(g_imported_views, {}, &ImportedView::begin);
        g_imported_count.store(static_cast<u32>(g_imported_views.size()), std::memory_order_release);
    }
    LOG_WARNING(Render_Vulkan, "P2: imported view {:#x}+{:#x} ({} views)", begin, end - begin,
                imported_buffers.size());
    return buffer.get();
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferImpl(VAddr device_addr, u32 size,
                                                            bool is_written, bool is_texel_buffer) {
    // D1: guest memory obtained here is imported for the GPU (coverage logged every 2 s).
    // Diagnostic only (SHADGT_DIAG_D1=1): it imports memory on every obtain.
    static const bool diag_d1 = [] {
        const char* env = std::getenv("SHADGT_DIAG_D1");
        return env && env[0] == '1';
    }();
    if (diag_d1 && direct_memory && !is_texel_buffer) {
        static u64 covered = 0, missed = 0;
        static auto window = std::chrono::steady_clock::now();
        if (direct_memory->EnsureImported(device_addr, size)) {
            ++covered;
        } else {
            ++missed;
        }
        if (const auto now = std::chrono::steady_clock::now(); now - window >= std::chrono::seconds{2}) {
            window = now;
            const auto stats = direct_memory->GetStats();
            LOG_WARNING(Render_Vulkan, "D1: obtains in 2 s: {} in imported memory, {} not; {} views ({} MiB), {} refused",
                        std::exchange(covered, 0), std::exchange(missed, 0), stats.views, stats.bytes >> 20, stats.failed);
        }
    }
    // EXP-125 (SHADGT_CEIL_REGION=<lo hex>-<hi hex>, measurement only, rendering goes wrong):
    // obtains inside the region skip every synchronization and use the arena as it is: the frame
    // rate if that memory were read in place.
    static const std::pair<VAddr, VAddr> ceil_region = [] {
        const char* env = std::getenv("SHADGT_CEIL_REGION");
        if (!env) {
            return std::pair<VAddr, VAddr>{0, 0};
        }
        char* rest{};
        const VAddr lo = std::strtoull(env, &rest, 16);
        const VAddr hi = rest && *rest == '-' ? std::strtoull(rest + 1, nullptr, 16) : 0;
        return std::pair<VAddr, VAddr>{lo, hi};
    }();
    static const auto ceil_start = std::chrono::steady_clock::now();
    if (ceil_region.second != 0 &&
        std::chrono::steady_clock::now() - ceil_start > std::chrono::seconds{190} &&
        device_addr >= ceil_region.first &&
        device_addr + size <= ceil_region.second && !is_texel_buffer) {
        std::scoped_lock ceil_lock{mutex};
        const u64 first_block = device_addr >> block_shift;
        const u64 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        EnsureResident(arena, first_block, last_block);
        return {arena, arena->Offset(device_addr)};
    }
    // P2 (SHADGT_PROTO_INPLACE=1, prototype): ranges in imported guest views are used in place.
    static const bool proto_inplace = [] {
        const char* env = std::getenv("SHADGT_PROTO_INPLACE");
        return env && env[0] == '1';
    }();
    if (proto_inplace) {
        static auto window = std::chrono::steady_clock::now();
        if (const auto now = std::chrono::steady_clock::now();
            now - window >= std::chrono::seconds{2}) {
            window = now;
            LOG_WARNING(Render_Vulkan, "P2: in 2 s: {} obtains in place, {} views imported, {} refused",
                        inplace_obtains.exchange(0), g_imported_count.load(),
                        import_rejected.load());
        }
    }
    if (proto_inplace && !is_texel_buffer) {
        if (const auto* view = FindImportedView(device_addr, size)) {
            ++inplace_obtains;
            return {static_cast<const Buffer*>(view->buffer), device_addr - view->begin};
        }
        if (const Buffer* imported = TryImportView(device_addr, size)) {
            ++inplace_obtains;
            return {imported, imported->Offset(device_addr)};
        }
    }
    // R6: the common case under the shared lock (stage workers obtain meanwhile). -DisablePerf
    // 132 always takes it exclusively.
    static const bool shared_obtain =
        Common::PerfFeatureEnabled(132) && StageWorkerCount() > 1;
    if (shared_obtain && t_recorder_thread && !CacheMutex::HeldExclusive()) {
        std::shared_lock shared{mutex};
        if (const auto result = ObtainBufferShared(device_addr, size, is_written, is_texel_buffer)) {
            ++shared_obtains;
            return *result;
        }
    }
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    // EXP-124 (SHADGT_CEIL_STREAMFAST=1, measurement only): small read-only ranges are copied
    // into the stream ring with no other bookkeeping (no image or GPU-write checks).
    static const bool ceil_stream = [] {
        const char* env = std::getenv("SHADGT_CEIL_STREAMFAST");
        return env && env[0] == '1';
    }();
    if (ceil_stream && !is_written && size <= STREAM_THRESHOLD && !read_capture) {
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    RecordPendingAheadUploads(device_addr, size);
    if (is_written) {
        InvalidateAsyncReadbacks(device_addr, size);
    }
    // After the invalidation: a readback that completed before it must not later remove the
    // ranges this write adds.
    ApplyCompletedReadbacks();
    ++Common::GetWorkCounters().obtain_buffer;
    // DIAG-109 (SHADGT_DIAG_OBTAIN=1): how many read-only obtains repeat a range already obtained
    // in the same upload epoch (what a per-epoch memo of obtains could skip).
    static const bool diag_obtain = [] {
        const char* env = std::getenv("SHADGT_DIAG_OBTAIN");
        return env && env[0] == '1';
    }();
    if (diag_obtain) {
        static std::unordered_set<u64> seen;
        static u32 seen_epoch = 0;
        static u64 calls = 0, repeats = 0, small_calls = 0, small_repeats = 0, epochs = 0;
        static auto window = std::chrono::steady_clock::now();
        const u32 epoch = g_upload_epoch.load(std::memory_order_acquire);
        if (epoch != seen_epoch) {
            seen.clear();
            seen_epoch = epoch;
            ++epochs;
        }
        const bool small = size <= STREAM_THRESHOLD;
        ++calls;
        small_calls += small;
        if (!is_written && !seen.insert(device_addr ^ (u64(size) << 40)).second) {
            ++repeats;
            small_repeats += small;
        }
        if (const auto now = std::chrono::steady_clock::now(); now - window >= std::chrono::seconds{2}) {
            LOG_WARNING(Render_Vulkan,
                        "DIAG-109: obtains in 2 s: {} ({} small), {} repeat a read-only range of "
                        "the same epoch ({} small); {} epochs; PERF-117 arena memo {} hits {} "
                        "stores; DIAG-111 Mcycles/count: stream "
                        "{}/{}, arena read {}/{}, arena written {}/{}, in place {}/{}",
                        calls, small_calls, repeats, small_repeats, epochs,
                        std::exchange(arena_memo_hits, 0), std::exchange(arena_memo_stores, 0),
                        g_obtain_cycles[0].exchange(0) / 1000000, g_obtain_counts[0].exchange(0),
                        g_obtain_cycles[1].exchange(0) / 1000000, g_obtain_counts[1].exchange(0),
                        g_obtain_cycles[2].exchange(0) / 1000000, g_obtain_counts[2].exchange(0),
                        g_obtain_cycles[3].exchange(0) / 1000000, g_obtain_counts[3].exchange(0));
            std::string sources;
            for (u32 i = 0; i < g_epoch_bump_sources.size(); ++i) {
                if (const u32 n = g_epoch_bump_sources[i].exchange(0); n != 0) {
                    sources += fmt::format(" {}:{}={}", i >= 8192 ? "rasterizer" : "liverpool",
                                           i & 8191, n);
                }
            }
            LOG_WARNING(Render_Vulkan, "DIAG-112: upload epoch bumps in 2 s by source:{}", sources);
            calls = repeats = small_calls = small_repeats = epochs = 0;
            window = now;
        }
    }
    // PERF-117: a read-only arena range synchronized earlier in this upload epoch is current
    // while no page of its regions was marked CPU-modified since and no image over it changed:
    // later draws of the epoch bind it without walking, hashing or protecting its pages again.
    // Draws bind the same pool ranges over and over (~1 arena read per draw, ~2.7 us each).
    // Only ranges above the stream threshold, which always come from the arena.
    // Opt-in (SHADGT_ARENA_MEMO=1): no measurable gain in the Tokyo A/B, kept for the ceiling
    // tests and the draw-state cache.
    static const bool arena_memo_enabled = [] {
        const char* env = std::getenv("SHADGT_ARENA_MEMO");
        return Common::PerfFeatureEnabled(117) && env && env[0] == '1';
    }();
    ArenaMemo* memo_lookup_slot = nullptr;
    u64 memo_versions = 0;
    bool memo_skip_sync = false;
    // Written ranges too (compute dispatches write the same buffers every frame): their
    // synchronization is skipped, the GPU-write bookkeeping below still runs.
    if (arena_memo_enabled && (is_written || size > STREAM_THRESHOLD) && !read_capture) {
        memo_lookup_slot = &arena_memos[((device_addr >> 6) ^ (device_addr >> 22) ^ size ^
                                         (u64(is_texel_buffer) << 5) ^ (u64(is_written) << 4)) %
                                        arena_memos.size()];
        memo_versions = memory_tracker->CpuVersionSum(device_addr, size);
        const auto& m = *memo_lookup_slot;
        // EXP-118 (SHADGT_CEIL_NOSYNC=1, measurement only, rendering goes wrong): a range is
        // synchronized once and never again, for the frame rate without arena synchronization.
        static const bool ceil_nosync = [] {
            const char* env = std::getenv("SHADGT_CEIL_NOSYNC");
            return env && env[0] == '1';
        }();
        if (ceil_nosync && !is_written && m.address == device_addr && m.size == size &&
            m.texel == is_texel_buffer && !m.written) {
            ++arena_memo_hits;
            return {m.arena, m.arena->Offset(device_addr)};
        }
        if (m.address == device_addr && m.size == size && m.texel == is_texel_buffer &&
            m.written == is_written &&
            m.arena == GetArena(device_addr >> block_shift, (device_addr + size - 1) >> block_shift) &&
            m.epoch == g_upload_epoch.load(std::memory_order_acquire) &&
            m.versions == memo_versions && m.registry == texture_cache.RegistryGeneration() &&
            m.export_state == g_image_export_state.load(std::memory_order_relaxed)) {
            ++arena_memo_hits;
            if (!is_written) {
                return {m.arena, m.arena->Offset(device_addr)};
            }
            memo_skip_sync = true;
        }
    }
    const u32 memo_epoch = g_upload_epoch.load(std::memory_order_acquire);
    const u64 memo_registry = texture_cache.RegistryGeneration();
    const u64 memo_export_state = g_image_export_state.load(std::memory_order_relaxed);
    const bool images_overlap = !memo_skip_sync && SynchronizeMemoryFromImage(device_addr, size);
    // PERF-043: with no image over the range, it has no GPU image alias either; the texture
    // cache is not walked again. -DisablePerf 55 walks it again.
    static const bool skip_alias_walk = Common::PerfFeatureEnabled(55);
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size) &&
        ((skip_alias_walk && !images_overlap) || !HasGpuImageAlias(device_addr, size))) {
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size, read_capture);
        stream_buffer.Commit();
        ++Common::GetWorkCounters().obtain_stream;
        return {&stream_buffer, offset};
    }
    // P1 (SHADGT_PROTO_IMPORT=1): once, 200 s in, on a read range of 64 KB or more.
    static const bool proto_import = [] {
        const char* env = std::getenv("SHADGT_PROTO_IMPORT");
        return env && (env[0] == '1' || env[0] == '2' || env[0] == '3');
    }();
    if (proto_import && !is_written && size >= 64_KB) {
        static const auto start = std::chrono::steady_clock::now();
        static bool done = false;
        if (!done && std::chrono::steady_clock::now() - start > std::chrono::seconds{200}) {
            done = true;
            const char* mode = std::getenv("SHADGT_PROTO_IMPORT");
            if (mode && mode[0] == '3') {
                scheduler.Finish();
                ProtoAliasTest(instance, scheduler, device_addr);
            } else if (mode && mode[0] == '2') {
                scheduler.Finish();
                ProtoImportSurvey(instance, 0x200000000ULL, 0x10000000000ULL);
            } else {
                ProtoImportTest(instance, scheduler, device_addr, size);
            }
        }
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    ArenaMemo* arena_memo = memo_lookup_slot;
    const u64 versions = memo_versions;
    // EXP-126 (SHADGT_CEIL_ONCE_PER_SUBMIT=1, measurement only): a read-only arena range is
    // synchronized once per graphics submission, not again in its later upload epochs: the frame
    // rate if those reads cost nothing after the first.
    static const bool once_per_submit = [] {
        const char* env = std::getenv("SHADGT_CEIL_ONCE_PER_SUBMIT");
        return env && env[0] == '1';
    }();
    if (once_per_submit && !is_written && !is_texel_buffer && !memo_skip_sync) {
        static std::unordered_map<u64, u64> synced; // range -> submission
        const u64 key = device_addr ^ (u64(size) << 40) ^ (u64(size) >> 24);
        const u64 submit = g_gfx_submit_count.load(std::memory_order_acquire);
        auto [it, inserted] = synced.try_emplace(key, submit);
        if (!inserted && it->second == submit) {
            ++ceil_skipped_syncs;
            EnsureResident(arena, first_block, last_block);
            memo_skip_sync = true;
        } else {
            it->second = submit;
        }
        if (synced.size() > (1u << 20)) {
            synced.clear();
        }
    }
    if (!memo_skip_sync) {
        EnsureResident(arena, first_block, last_block);
        SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    }
    if (arena_memo && !memo_skip_sync) {
        // The state from before this synchronization: anything it changed shows up as a
        // different version or generation, and the next lookup synchronizes again.
        *arena_memo = {device_addr, size,     is_texel_buffer, is_written, arena,
                       memo_epoch,  versions, memo_registry,   memo_export_state};
        ++arena_memo_stores;
    }
    if (is_written) {
        SyncUnseenCpuWrites(arena, device_addr, size); // FIX-073
        gpu_modified_ranges.Add(device_addr, size);
        NoteGpuWriteSeq(device_addr, size); // FIX-069
        NoteGpuWrite(device_addr, size);
        if (size <= 4_KB) {
            if (small_gpu_writers.size() > 65536) {
                small_gpu_writers.clear();
            }
            small_gpu_writers[Common::AlignDown(device_addr, 4_KB)] =
                GpuWriter{g_gpu_write_kind, g_gpu_write_tag, device_addr, size};
        } else {
            if (large_gpu_writers.size() > 16384) {
                large_gpu_writers.clear();
            }
            large_gpu_writers[device_addr] =
                GpuWriter{g_gpu_write_kind, g_gpu_write_tag, device_addr, size};
        }
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    RecordPendingAheadUploads(device_addr, size);
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size, read_capture);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

bool BufferCache::HasGpuImageAlias(VAddr addr, size_t size) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    if (size == 0) {
        return false;
    }
    bool found = false;
    texture_cache.ForEachImageInRegion(addr, size, [&](ImageId, Image& image) {
        found |= image.SafeToDownload();
        return found;
    });
    return found;
}

bool BufferCache::DirectReadHitsImage(VAddr addr, size_t size) {
    // FIX-122: a range the GPU reads in place must not hold a picture the GPU drew into: its
    // pixels are in the image, not in guest memory, until they are exported (the usual path
    // exports them first). GT Sport reads such ranges on its car screens; read in place, about
    // one frame in ten had the car view blown out to white and the dashboard panel white or
    // magenta, and the frame after picking a car flashed white. Most ranges are constant and
    // vertex rings in blocks that hold no images, so those are remembered per thread until an
    // image is registered or unregistered. -DisablePerf 144 reads in place as before.
    static const bool enabled = Common::PerfFeatureEnabled(144);
    if (!enabled || size == 0) {
        return false;
    }
    constexpr u32 BlockBits = 16;
    struct Block {
        VAddr block;
        u64 generation;
    };
    thread_local std::vector<Block> free_blocks(4096, Block{~VAddr{0}, 0});
    const u64 generation = texture_cache.RegistryGeneration();
    const VAddr first = addr >> BlockBits;
    const VAddr last = (addr + size - 1) >> BlockBits;
    if (last - first < 64) {
        bool all_free = true;
        for (VAddr block = first; all_free && block <= last; ++block) {
            auto& entry = free_blocks[block % free_blocks.size()];
            if (entry.block == block && entry.generation == generation) {
                continue;
            }
            bool any = false;
            {
                std::scoped_lock buffer_lock{mutex};
                texture_cache.ForEachImageInRegion(block << BlockBits, VAddr{1} << BlockBits,
                                                   [&](ImageId, Image&) {
                                                       any = true;
                                                       return true;
                                                   });
            }
            if (any) {
                all_free = false;
            } else {
                entry = {block, generation};
            }
        }
        if (all_free) {
            return false;
        }
    }
    return HasGpuImageAlias(addr, size);
}

void BufferCache::SynchronizeDmaBuffers() {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    RecordPendingAheadUploads(0, ~u64{0} >> 1);
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    // Conservatively cancel retirement if a range is used again through any obtain path.
    NoteArenaMapping(first_block << block_shift, (last_block - first_block + 1) << block_shift);
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    // FIX-075: 64-bit counts. GT Sport's showroom makes a range of 4 GB + 64 KB resident at once;
    // the 32-bit block loop below never reached its end and wrote past the staging buffer.
    u64 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    if ((resident_blocks << block_shift) >= (1ull << 30)) {
        LOG_WARNING(Render_Vulkan,
                    "FIX-075: making {} MB resident at once ({:#x}..{:#x} in arena {:#x}+{:#x})",
                    (resident_blocks << block_shift) >> 20, first_block << block_shift,
                    (last_block + 1) << block_shift, arena->CpuAddr(), arena->SizeBytes());
    }

    const vk::MemoryAllocateInfo alloc_info = {
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u64 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

// PERF-046: hashes count pages (page_at(i) gives each address) into hashes, on several
// threads from 128 pages (512 KB) on; -DisablePerf 58 hashes them on this thread.
// PERF-087: hashes of hot pages the buffer stage took for the obtain it runs (ascending).
static thread_local std::vector<std::pair<VAddr, u64>> t_stage_prehash;
thread_local u32 BufferCache::draw_epoch{};
thread_local const Common::ReadCapture* BufferCache::read_capture{};
thread_local BufferCache::AheadUploads* BufferCache::ahead_uploads{};

template <typename PageAt>
static void HashPages(size_t count, PageAt&& page_at, std::span<u64> hashes) {
    static constexpr size_t ParallelFrom = 128;
    static const bool parallel = Common::PerfFeatureEnabled(58);
    static std::mutex hasher_mutex;
    std::unique_lock hasher_lock{hasher_mutex, std::defer_lock};
    if (parallel && count >= ParallelFrom && hasher_lock.try_lock()) {
        static VideoCore::ParallelPageHasher hasher{
            std::clamp(std::max(std::thread::hardware_concurrency(), 4u) / 4, 1u, 3u)};
        static std::vector<const u8*> pointers;
        pointers.resize(count);
        for (size_t i = 0; i < count; ++i) {
            pointers[i] = std::bit_cast<const u8*>(page_at(i));
        }
        hasher.Hash(pointers, BYTES_PER_PAGE, hashes.first(count));
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        hashes[i] = Core::HashGuestBytes(page_at(i), BYTES_PER_PAGE);
    }
}

void BufferCache::HashPagesAhead(std::span<const std::pair<VAddr, u64>> ranges, u32 epoch) {
    // Pages that have an upload record are the ones SynchronizeMemory hashes (when hot); others
    // are uploaded whole anyway. Each page is hashed once per epoch.
    constexpr size_t MaxPages = 2048;
    boost::container::small_vector<VAddr, 64> pages;
    {
        std::scoped_lock lk{uploaded_pages_mutex};
        for (const auto& [address, size] : ranges) {
            for (VAddr page = Common::AlignUp(address, BYTES_PER_PAGE);
                 page + BYTES_PER_PAGE <= address + size && pages.size() < MaxPages;
                 page += BYTES_PER_PAGE) {
                if (!ahead_hashes.Has(page, epoch) && uploaded_page_contents.Contains(page)) {
                    pages.push_back(page);
                }
            }
        }
    }
    if (pages.empty()) {
        return;
    }
    // No mapping can go away while the pages are read.
    memory->WithMappingsStable([&] {
        for (const VAddr page : pages) {
            if (memory->IsMappedAddress(page)) {
                ahead_hashes.Store(page, epoch,
                                   Core::HashGuestBytes(page, BYTES_PER_PAGE));
            }
        }
    });
    ahead_hashed.fetch_add(pages.size(), std::memory_order_relaxed);
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::ObtainBufferAhead(
    VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer,
    AheadUploads& uploads, bool* overlapped_out) {
    // PERF-087: the hot pages this obtain will check are hashed before the lock is taken; the
    // recorder waited on the lock while the stage hashed them under it. A page the CPU writes
    // after this hash is no different from one it writes after a hash taken under the lock.
    // -DisablePerf 106 hashes them under the lock.
    static const bool prehash = Common::PerfFeatureEnabled(106);
    auto& prehashed = t_stage_prehash;
    prehashed.clear();
    if (prehash && size >= BYTES_PER_PAGE && !is_written) {
        boost::container::small_vector<VAddr, 32> pages;
        memory_tracker->ForEachHotPageToCheck(device_addr, size, [&](VAddr page) {
            if (page >= device_addr && page + BYTES_PER_PAGE <= device_addr + size &&
                pages.size() < 4096) {
                pages.push_back(page);
            }
        });
        if (!pages.empty()) {
            memory->WithMappingsStable([&] {
                for (const VAddr page : pages) {
                    if (memory->IsMappedAddress(page)) {
                        prehashed.emplace_back(page, Core::HashGuestBytes(page, BYTES_PER_PAGE));
                    }
                }
            });
        }
    }
    struct ClearPrehash {
        ~ClearPrehash() {
            t_stage_prehash.clear();
        }
    } clear_prehash;
    // R3: a read is obtained under the shared lock (with other stage workers; the recorder's
    // obtains are exclusive). -DisablePerf 131 takes it exclusively.
    static const bool shared_reads = Common::PerfFeatureEnabled(131);
    std::unique_lock<CacheMutex> lk_ex{mutex, std::defer_lock};
    std::shared_lock<CacheMutex> lk_sh{mutex, std::defer_lock};
    if (shared_reads && !is_written) {
        lk_sh.lock();
    } else {
        lk_ex.lock();
    }
    // PERF-079: a range under images is obtained ahead through the arena as well; the recorder
    // exports the images before using it (ExportImagesOver). In GT Sport's races a third of the
    // buffers the stage was given lie under images, and each stopped it. -DisablePerf 84 leaves
    // them to the recorder.
    static const bool overlaps_ahead = Common::PerfFeatureEnabled(84);
    bool overlapped = false;
    if (size != 0 && overlaps_ahead && overlapped_out &&
        texture_cache.AnyImageInRegion(device_addr, size)) {
        overlapped = true;
        *overlapped_out = true;
    } else if (size == 0 || texture_cache.AnyImageInRegion(device_addr, size)) {
        ++ahead_refusals[1];
        static std::atomic<u32> logged{};
        if (size != 0 && logged.fetch_add(1) < 40) {
            LOG_WARNING(Render_Vulkan, "PERF-067: buffer {:#x}+{:#x} ({}) left to the recorder: "
                        "an image overlaps it",
                        device_addr, size, is_written ? "written" : "read");
        }
        return std::nullopt; // Image exports and aliases are the recorder's.
    }
    if (is_written) {
        InvalidateAsyncReadbacks(device_addr, size);
    }
    ApplyCompletedReadbacks();
    ++Common::GetWorkCounters().obtain_buffer;
    if (!is_written && size <= STREAM_THRESHOLD && !overlapped &&
        !IsRegionGpuModified(device_addr, size)) {
        // MT-008: the ring is the stage's own, so it is filled without the lock (the recorder
        // waited on it for these copies). -DisablePerf 130 fills it under the lock.
        static const bool copy_unlocked = Common::PerfFeatureEnabled(130);
        if (copy_unlocked) {
            if (lk_sh.owns_lock()) {
                lk_sh.unlock();
            } else {
                lk_ex.unlock();
            }
        }
        auto& ahead_stream = AheadStream(); // R5
        const auto [data, offset] = ahead_stream.Map(size, instance.UniformMinAlignment(), false);
        if (!data) {
            ++ahead_refusals[2];
            return std::nullopt; // The ring would have to wait for the GPU.
        }
        memory->CopySparseMemory(device_addr, data, size);
        ahead_stream.Commit();
        ++Common::GetWorkCounters().obtain_stream;
        return std::pair<const Buffer*, u64>{&ahead_stream, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    if (is_texel_buffer && !is_written) {
        ++ahead_refusals[0];
        return std::nullopt; // Metadata (HTILE) is the texture cache's (arena path only).
    }
    if (!address_space[first_page] || address_space[first_page] != address_space[last_page]) {
        ++ahead_refusals[3];
        return std::nullopt; // A new or migrating arena records binds.
    }
    bool resident = true;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64, u64) { resident = false; });
    if (!resident) {
        ++ahead_refusals[4];
        return std::nullopt;
    }
    // The uploads' staging must not wait for the GPU either: size of everything this range
    // could upload.
    {
        const auto [probe, probe_offset] = AheadStaging().Map(size, 16, false);
        if (!probe) {
            ++ahead_refusals[5];
            return std::nullopt;
        }
    }
    const Buffer* arena = address_space[first_page];
    ahead_uploads = &uploads;
    // R5: the recorder waits for this before it uses a range over it (SettleAheadRange).
    auto& sync = worker_syncs[t_stage_worker];
    sync.end.store(device_addr + size, std::memory_order_relaxed);
    sync.begin.store(device_addr, std::memory_order_seq_cst);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    sync.begin.store(0, std::memory_order_release);
    ahead_uploads = nullptr;
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
        NoteGpuWriteSeq(device_addr, size); // FIX-069
        NoteGpuWrite(device_addr, size);
        if (size <= 4_KB) {
            if (small_gpu_writers.size() > 65536) {
                small_gpu_writers.clear();
            }
            small_gpu_writers[Common::AlignDown(device_addr, 4_KB)] =
                GpuWriter{g_gpu_write_kind, g_gpu_write_tag, device_addr, size};
        } else {
            if (large_gpu_writers.size() > 16384) {
                large_gpu_writers.clear();
            }
            large_gpu_writers[device_addr] =
                GpuWriter{g_gpu_write_kind, g_gpu_write_tag, device_addr, size};
        }
    }
    return std::pair<const Buffer*, u64>{arena, arena->Offset(device_addr)};
}

u32 BufferCache::StageWorkerCount() {
    static const u32 count = [] {
        const char* env = std::getenv("SHADGT_STAGE_WORKERS");
        const int n = env ? std::atoi(env) : 1;
        return static_cast<u32>(std::clamp(n, 1, static_cast<int>(MaxStageWorkers)));
    }();
    return count;
}

void BufferCache::SettleAheadRange(VAddr device_addr, u64 size) {
    const VAddr end = device_addr + size;
    for (u32 i = 0; i < StageWorkerCount(); ++i) {
        auto& sync = worker_syncs[i];
        while (true) {
            const VAddr begin = sync.begin.load(std::memory_order_seq_cst);
            if (begin == 0 || begin >= end || sync.end.load(std::memory_order_relaxed) <= device_addr) {
                break;
            }
            _mm_pause();
        }
    }
    std::shared_lock buffer_lock{mutex};
    RecordPendingAheadUploads(device_addr, size);
}

void BufferCache::RecordAheadUpload(AheadUpload& upload) {
    if (upload.recorded) {
        return;
    }
    upload.recorded = true;
    if (runtime.HoistsUploads()) {
        runtime.CopyBufferHoisted(upload.staging, upload.arena, upload.copies);
    } else {
        runtime.CopyBuffer(upload.staging, upload.arena, upload.copies);
    }
}

void BufferCache::RecordAheadUploads(const AheadUploads& uploads) {
    // R3: the uploads and the pending list need no cache state; shared with the stage's syncs.
    std::shared_lock buffer_lock{mutex};
    std::scoped_lock pending_lk{pending_uploads_mutex};
    for (const auto& upload : uploads) {
        RecordAheadUpload(*upload);
    }
    while (!pending_ahead_uploads.empty() && pending_ahead_uploads.front()->recorded) {
        pending_ahead_uploads.pop_front();
    }
}

void BufferCache::RecordPendingAheadUploads(VAddr device_addr, u64 size) {
    // Lock held. An upload prepared for a later draw over these bytes is recorded now, with
    // the ones prepared before it, so this use sees them.
    std::scoped_lock pending_lk{pending_uploads_mutex}; // R3
    if (pending_ahead_uploads.empty()) {
        return;
    }
    size_t last = pending_ahead_uploads.size();
    for (size_t i = pending_ahead_uploads.size(); i-- > 0;) {
        const auto& upload = *pending_ahead_uploads[i];
        if (!upload.recorded && upload.begin < device_addr + size && device_addr < upload.end) {
            last = i;
            break;
        }
    }
    if (last == pending_ahead_uploads.size()) {
        return;
    }
    for (size_t i = 0; i <= last; ++i) {
        RecordAheadUpload(*pending_ahead_uploads[i]);
    }
    while (!pending_ahead_uploads.empty() && pending_ahead_uploads.front()->recorded) {
        pending_ahead_uploads.pop_front();
    }
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    ApplyCompletedReadbacks();
    const auto add_upload = [&](VAddr start, VAddr end) {
        copies.emplace_back(total_size_bytes, start, end - start);
        total_size_bytes += end - start;
    };
    // PERF-015: bytes of pending GDS copies stay as they are in the arena until their value
    // reaches guest memory; uploading the stale guest copy would overwrite them.
    boost::container::small_vector<std::pair<VAddr, VAddr>, 8> gds_pending;
    if (split_gds_write_faults && num_valid_gds_readbacks.load() != 0) {
        std::scoped_lock lk{async_readbacks_mutex};
        for (const auto* list : {&pending_async_readbacks, &inflight_async_readbacks}) {
            for (const auto& readback : *list) {
                if (readback->valid && readback->is_gds && readback->address < device_addr + size &&
                    device_addr < readback->address + readback->size) {
                    gds_pending.emplace_back(readback->address, readback->address + readback->size);
                }
            }
        }
        std::ranges::sort(gds_pending);
    }
    const auto upload_range = [&](u64 addr, u64 size) {
        // PERF-011: bytes the GPU wrote and guest memory does not have yet stay as they are in
        // the arena; uploading the stale guest copy would overwrite them.
        std::unique_lock gpu_lk{gpu_ranges_mutex}; // R3
        if (split_write_faults && gpu_modified_ranges.Intersects(addr, size)) {
            VAddr cursor = addr;
            gpu_modified_ranges.ForEachInRange(addr, size, [&](VAddr start, VAddr end) {
                if (start > cursor) {
                    add_upload(cursor, start);
                }
                cursor = std::max(cursor, end);
            });
            if (cursor < addr + size) {
                add_upload(cursor, addr + size);
            }
            return;
        }
        gpu_lk.unlock();
        VAddr cursor = addr;
        for (const auto& [start, end] : gds_pending) {
            if (end <= cursor || start >= addr + size) {
                continue;
            }
            if (start > cursor) {
                add_upload(cursor, start);
            }
            cursor = std::max(cursor, end);
        }
        if (cursor < addr + size) {
            add_upload(cursor, addr + size);
        }
    };
    // PERF-033: hot pages whose bytes match what their last upload copied are left out. The
    // ranges are collected first; hot pages are looked up without the region lock held.
    static const bool skip_unchanged_hot_pages = Common::PerfFeatureEnabled(48);
    struct HotUpload {
        VAddr page;
        u16 generation;
        u64 hash;
    };
    boost::container::small_vector<HotUpload, 16> hot_uploads;
    boost::container::small_vector<std::pair<VAddr, u64>, 8> ranges;
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
        if (skip_unchanged_hot_pages) {
            ranges.emplace_back(addr, size);
        } else {
            upload_range(addr, size);
        }
    });
    const auto whole_page_from_guest = [&](VAddr page) {
        if (split_write_faults) {
            std::scoped_lock gpu_lk{gpu_ranges_mutex}; // R3
            if (gpu_modified_ranges.Intersects(page, BYTES_PER_PAGE)) {
                return false;
            }
        }
        return std::ranges::none_of(gds_pending, [&](const auto& gds) {
            return gds.first < page + BYTES_PER_PAGE && page < gds.second;
        });
    };
    // PERF-046: the hot pages of all ranges are hashed first, on several threads when many.
    // PERF-053: into scratch vectors kept by the thread (thousands of hot pages per call), and
    // looked up under one hold of the record lock.
    struct HotScratch {
        std::vector<std::pair<VAddr, u16>> hot;
        std::vector<size_t> range_end; // one past each range's hot pages
        std::vector<u64> hashes;
        std::vector<size_t> misses;
        std::vector<u64> miss_hashes;
        std::vector<u8> unchanged;
        bool in_use = false;
    };
    static thread_local HotScratch thread_scratch;
    HotScratch local_scratch;
    HotScratch& scratch = thread_scratch.in_use ? local_scratch : thread_scratch;
    scratch.in_use = true;
    struct Release {
        HotScratch& scratch;
        ~Release() {
            scratch.in_use = false;
        }
    } release{scratch};
    auto& hot = scratch.hot;
    auto& range_end = scratch.range_end;
    auto& hot_hashes = scratch.hashes;
    hot.clear();
    range_end.clear();
    for (const auto& [addr, range_size] : ranges) {
        memory_tracker->ForEachHotPage(addr, range_size, [&](VAddr page, u16 generation) {
            if (page >= addr && page + BYTES_PER_PAGE <= addr + range_size) {
                hot.emplace_back(page, generation);
            }
        });
        range_end.push_back(hot.size());
    }
    hot_hashes.resize(hot.size());
    if (!t_stage_prehash.empty() && !hot.empty()) {
        // PERF-087: hashed by the buffer stage before it took the lock (ascending pages).
        auto& misses = scratch.misses;
        misses.clear();
        for (size_t i = 0; i < hot.size(); ++i) {
            const auto it = std::ranges::lower_bound(t_stage_prehash, hot[i].first, {},
                                                     &std::pair<VAddr, u64>::first);
            if (it != t_stage_prehash.end() && it->first == hot[i].first) {
                hot_hashes[i] = it->second;
            } else {
                misses.push_back(i);
            }
        }
        stage_prehash_hits += hot.size() - misses.size();
        stage_prehash_misses += misses.size();
        for (const size_t i : misses) {
            hot_hashes[i] = Core::HashGuestBytes(hot[i].first, BYTES_PER_PAGE);
        }
    } else if (read_capture && !hot.empty()) {
        // PERF-063: pages captured at decode come with their hash; the rest are hashed here.
        auto& misses = scratch.misses;
        misses.clear();
        for (size_t i = 0; i < hot.size(); ++i) {
            if (!read_capture->Find(hot[i].first, &hot_hashes[i])) {
                misses.push_back(i);
            }
        }
        capture_hash_hits += hot.size() - misses.size();
        capture_hash_misses += misses.size();
        auto& miss_hashes = scratch.miss_hashes;
        miss_hashes.resize(misses.size());
        HashPages(misses.size(), [&](size_t j) { return hot[misses[j]].first; },
                  std::span<u64>{miss_hashes.data(), miss_hashes.size()});
        for (size_t j = 0; j < misses.size(); ++j) {
            hot_hashes[misses[j]] = miss_hashes[j];
        }
    } else if (draw_epoch != 0 && !hot.empty()) {
        // PERF-062: hashes the command thread took when it decoded this draw, in its epoch.
        auto& misses = scratch.misses;
        misses.clear();
        for (size_t i = 0; i < hot.size(); ++i) {
            if (!ahead_hashes.Lookup(hot[i].first, draw_epoch, hot_hashes[i])) {
                misses.push_back(i);
            }
        }
        ahead_hits += hot.size() - misses.size();
        ahead_misses += misses.size();
        auto& miss_hashes = scratch.miss_hashes;
        miss_hashes.resize(misses.size());
        HashPages(misses.size(), [&](size_t j) { return hot[misses[j]].first; },
                  std::span<u64>{miss_hashes.data(), miss_hashes.size()});
        for (size_t j = 0; j < misses.size(); ++j) {
            hot_hashes[misses[j]] = miss_hashes[j];
        }
    } else if (prehasher && !hot.empty()) {
        // PERF-049: hashes taken ahead in the current epoch; the rest are hashed here.
        const u32 epoch = g_upload_epoch.load(std::memory_order_acquire);
        const auto batch = prehasher->Current();
        auto& misses = scratch.misses;
        misses.clear();
        for (size_t i = 0; i < hot.size(); ++i) {
            if (!batch || !batch->Lookup(hot[i].first, epoch, hot_hashes[i])) {
                misses.push_back(i);
            }
        }
        prehashed_hits += hot.size() - misses.size();
        prehashed_misses += misses.size();
        auto& miss_hashes = scratch.miss_hashes;
        miss_hashes.resize(misses.size());
        HashPages(misses.size(), [&](size_t j) { return hot[misses[j]].first; },
                  std::span<u64>{miss_hashes.data(), miss_hashes.size()});
        for (size_t j = 0; j < misses.size(); ++j) {
            hot_hashes[misses[j]] = miss_hashes[j];
        }
    } else {
        HashPages(hot.size(), [&](size_t i) { return hot[i].first; },
                  std::span<u64>{hot_hashes.data(), hot_hashes.size()});
    }
    auto& unchanged_pages = scratch.unchanged;
    unchanged_pages.resize(hot.size());
    if (!hot.empty()) {
        std::scoped_lock lk{uploaded_pages_mutex};
        for (size_t i = 0; i < hot.size(); ++i) {
            unchanged_pages[i] =
                uploaded_page_contents.Unchanged(hot[i].first, hot[i].second, hot_hashes[i]);
        }
    }
    u64 unchanged_count = 0;
    size_t hot_index = 0;
    // PERF-064: hot pages unchanged for this many checks in a row go back to normal tracking;
    // the game writes them rarely now. A page written every frame changes within ~10 checks.
    // -DisablePerf 72 keeps every hot page hot.
    static const bool cool_hot_pages = Common::PerfFeatureEnabled(72);
    constexpr u8 CoolAfterChecks = 32;
    boost::container::small_vector<std::pair<VAddr, u64>, 8> cool;
    for (size_t range_index = 0; range_index < ranges.size(); ++range_index) {
        const auto [addr, range_size] = ranges[range_index];
        VAddr cursor = addr;
        for (; hot_index < range_end[range_index]; ++hot_index) {
            const auto [page, generation] = hot[hot_index];
            const u64 hash = hot_hashes[hot_index];
            if (unchanged_pages[hot_index]) {
                if (page > cursor) {
                    upload_range(cursor, page - cursor);
                }
                cursor = page + BYTES_PER_PAGE;
                ++unchanged_count;
                if (cool_hot_pages && !read_capture) {
                    std::scoped_lock streak_lk{streak_mutex}; // R3
                    u8& streak = hot_unchanged_streak[page];
                    if (++streak >= CoolAfterChecks) {
                        cool.emplace_back(page, hash);
                    }
                }
            } else if (whole_page_from_guest(page)) {
                if (cool_hot_pages) {
                    std::scoped_lock streak_lk{streak_mutex}; // R3
                    hot_unchanged_streak.erase(page);
                }
                hot_uploads.push_back({page, generation, hash});
                NoteCpuChange(page, BYTES_PER_PAGE); // FIX-069
            } else {
                NoteCpuChange(page, BYTES_PER_PAGE); // FIX-069
                std::scoped_lock lk{uploaded_pages_mutex};
                uploaded_page_contents.Forget(page);
            }
        }
        if (cursor < addr + range_size) {
            upload_range(cursor, addr + range_size - cursor);
        }
    }
    hot_pages_unchanged += unchanged_count;
    for (const auto& [page, hash] : cool) {
        {
            std::scoped_lock streak_lk{streak_mutex}; // R3
            hot_unchanged_streak.erase(page);
        }
        // Clean and write-protected again: the GPU copy holds these bytes (the last upload).
        memory_tracker->CoolHotPages(page, BYTES_PER_PAGE);
        ++hot_pages_cooled;
        // A CPU write between the check and the protection did not fault; it is caught here.
        if (Core::HashGuestBytes(page, BYTES_PER_PAGE) != hash) {
            NoteCpuChange(page, BYTES_PER_PAGE); // FIX-069
            memory_tracker->MarkRegionAsCpuModified(page, BYTES_PER_PAGE);
            ++hot_pages_cooled_changed;
        }
    }
    {
        std::scoped_lock streak_lk{streak_mutex}; // R3
        if (hot_unchanged_streak.size() > 1 << 20) {
            hot_unchanged_streak.clear();
        }
    }
    if (!copies.empty()) {
        for (const auto& copy : copies) {
            RecordWatchedUploads(copy.dstOffset, copy.dstOffset + copy.size);
            // FIX-073: the GPU copy takes in these guest bytes.
            RefreshReadbackPageHashes(copy.dstOffset, copy.size);
        }
        // FIX-066: bytes uploaded from guest memory are no longer GPU-only: the GPU copy now
        // holds the CPU's bytes. They stayed listed as GPU-written, and the next CPU fault on
        // the page wrote that copy back, which by then was older CPU data than guest memory
        // held (a page the game rewrote while it was hot and unwatched). GT Sport reuses its
        // per-frame memory this way; an earlier frame's vertex data came back over a later
        // frame's draw constants and descriptor tables (the race summary's driver figure
        // flickered and glowed green). -DisablePerf 90 keeps them listed.
        static const bool forget_uploaded = Common::PerfFeatureEnabled(90);
        if (forget_uploaded) {
            boost::container::small_vector<VAddr, 16> unmark;
            {
                std::scoped_lock gpu_lk{gpu_ranges_mutex}; // R3
                for (const auto& copy : copies) {
                    const VAddr start = copy.dstOffset;
                    const VAddr end = start + copy.size;
                    if (!gpu_modified_ranges.Intersects(start, copy.size)) {
                        continue;
                    }
                    gpu_modified_ranges.Subtract(start, copy.size);
                    for (VAddr page = Common::AlignDown(start, 4_KB); page < end; page += 4_KB) {
                        if (!gpu_modified_ranges.Intersects(page, 4_KB)) {
                            unmark.push_back(page);
                        }
                    }
                    ++hot_page_stats.uploads_over_gpu_bytes;
                }
            }
            for (const VAddr page : unmark) {
                memory_tracker->UnmarkRegionAsGpuModified(page, 4_KB, false);
            }
        }
        Common::GetWorkCounters().uploads += copies.size();
        Common::GetWorkCounters().upload_bytes += total_size_bytes;
        if (ahead_uploads) {
            // PERF-067: on the buffer stage; the ring and the free check were done before.
            auto& ahead_staging = AheadStaging(); // R5
            const auto [mapped, staging_offset] = ahead_staging.Map(total_size_bytes, 16, false);
            ASSERT(mapped);
            for (auto& copy : copies) {
                memory->CopySparseMemory(copy.dstOffset, mapped + copy.srcOffset, copy.size);
                copy.srcOffset += staging_offset;
                copy.dstOffset -= arena->cpu_addr;
            }
            ahead_staging.Commit();
            auto upload = std::make_shared<AheadUpload>();
            upload->staging = &ahead_staging;
            upload->arena = arena;
            upload->copies.assign(copies.begin(), copies.end());
            upload->begin = ~VAddr{0};
            upload->end = 0;
            for (const auto& copy : copies) {
                upload->begin = std::min(upload->begin, arena->cpu_addr + copy.dstOffset);
                upload->end = std::max(upload->end, arena->cpu_addr + copy.dstOffset + copy.size);
            }
            ahead_uploads->push_back(upload);
            std::scoped_lock pending_lk{pending_uploads_mutex}; // R3
            pending_ahead_uploads.push_back(std::move(upload));
            copies.clear();
        }
    }
    if (!copies.empty()) {
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        // PERF-027: large uploads are copied on several cores; -DisablePerf 27 copies them on
        // this thread only.
        static const bool parallel_copies = Common::PerfFeatureEnabled(27);
        constexpr u64 ParallelThreshold = 192_KB;
        constexpr u64 PieceSize = 64_KB;
        // The copier serves one caller at a time; any other copies on its own thread.
        static std::mutex copier_mutex;
        std::unique_lock copier_lock{copier_mutex, std::defer_lock};
        if (parallel_copies && total_size_bytes >= ParallelThreshold && copier_lock.try_lock()) {
            auto& copier = ParallelUploadCopier::Instance();
            auto& pieces = copier.BeginBatch();
            for (const auto& copy : copies) {
                for (u64 done_bytes = 0; done_bytes < copy.size; done_bytes += PieceSize) {
                    pieces.push_back({copy.dstOffset + done_bytes,
                                      staging.mapped + copy.srcOffset + done_bytes,
                                      std::min(PieceSize, copy.size - done_bytes)});
                }
            }
            copier.Run(memory, read_capture);
            for (auto& copy : copies) {
                copy.srcOffset += staging.offset;
                copy.dstOffset -= arena->cpu_addr;
            }
        } else {
            for (auto& copy : copies) {
                memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset,
                                         copy.size, read_capture);
                copy.srcOffset += staging.offset;
                copy.dstOffset -= arena->cpu_addr;
            }
        }
        staging.Flush();
        if (runtime.HoistsUploads()) {
            // PERF-066: ahead of the session's draws, without ending the render pass.
            runtime.CopyBufferHoisted(staging.buffer, arena, copies);
        } else {
            runtime.CopyBuffer(staging.buffer, arena, copies);
        }
    }
    if (!hot_uploads.empty()) {
        // A page the CPU wrote while it was copied gets no record and is uploaded next time.
        boost::container::small_vector<u64, 16> after(hot_uploads.size());
        if (read_capture) {
            // PERF-063: the upload copied the captured bytes, which do not change.
            for (size_t i = 0; i < hot_uploads.size(); ++i) {
                after[i] = HashGuestPage(hot_uploads[i].page);
            }
        } else {
            HashPages(hot_uploads.size(), [&](size_t i) { return hot_uploads[i].page; },
                      std::span<u64>{after.data(), after.size()});
        }
        std::scoped_lock lk{uploaded_pages_mutex};
        for (size_t i = 0; i < hot_uploads.size(); ++i) {
            const auto& upload = hot_uploads[i];
            uploaded_page_contents.Record(upload.page, upload.generation, upload.hash, after[i]);
        }
        hot_pages_uploaded += hot_uploads.size();
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMetadata(arena, device_addr, size);
    }
    return false;
}

static constexpr u64 WatchedPageBits = 12;
static constexpr u64 WatchedPageSize = u64{1} << WatchedPageBits;

u64 BufferCache::HashGuestPage(VAddr page) const {
    u64 hash;
    if (read_capture && read_capture->Find(page, &hash)) {
        return hash;
    }
    return Core::HashGuestBytes(page, BYTES_PER_PAGE);
}

void BufferCache::RecordWatchedUploads(VAddr start, VAddr end) {
    std::scoped_lock lk{vertex_pages_mutex};
    if (vertex_page_hashes.empty()) {
        return;
    }
    // Only whole pages: a partial upload leaves the rest of the page as it was.
    for (VAddr page = Common::AlignUp(start, WatchedPageSize); page + WatchedPageSize <= end;
         page += WatchedPageSize) {
        const auto it = vertex_page_hashes.find(page >> WatchedPageBits);
        if (it != vertex_page_hashes.end()) {
            it->second = HashGuestPage(page);
        }
    }
}

void BufferCache::DumpRange(VAddr address, u64 size, const std::filesystem::path& base) {
    const u64 first_block = address >> block_shift;
    const u64 last_block = (address + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    const auto download = staging_pool.Request(size, MemoryType::HostCached);
    const vk::BufferCopy copy{
        .srcOffset = arena->Offset(address),
        .dstOffset = download.offset,
        .size = size,
    };
    runtime.CopyBuffer(arena, download.buffer, std::span{&copy, 1});
    scheduler.Finish();
    download.buffer->Invalidate(download.offset, size);
    auto gpu_path = base;
    gpu_path += ".gpu.bin";
    Common::FS::IOFile{gpu_path, Common::FS::FileAccessMode::Create}.WriteRaw<u8>(download.mapped,
                                                                                  size);
    auto guest_path = base;
    guest_path += ".guest.bin";
    Common::FS::IOFile{guest_path, Common::FS::FileAccessMode::Create}.WriteRaw<u8>(
        std::bit_cast<const u8*>(address), size);
}

void BufferCache::InlineGuestWrite(VAddr address, std::span<const u8> data) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    RecordPendingAheadUploads(0, ~u64{0} >> 1);
    const u64 size = data.size();
    const u64 first_block = address >> block_shift;
    const u64 last_block = (address + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    // Pending CPU writes around the bytes first, so they do not later land over them.
    SynchronizeMemory(arena, address, u32(size), false, false);
    runtime.InlineData(arena, arena->Offset(address), data);
}

bool BufferCache::CanWriteThrough(VAddr address, u64 size) {
    // FIX-079: a CPU store to a page that also holds GPU-written bytes, but none of them (nearly
    // all of GT Sport's write faults: per-object records sitting beside GPU results), waited
    // for the GPU to finish all its queued work so the page's GPU bytes could be downloaded
    // first (about 1 ms on average, up to 100 ms; 54 s in an 8-minute session). The store goes
    // through instead: its bytes are copied to the GPU copy, and the page stays watched and
    // GPU-written, so reads of its GPU bytes and its readbacks work as before.
    // -DisablePerf 103 turns it off.
    // Opt-in (SHADGT_WRITE_THROUGH=1) since Lance's release sessions froze with it: the game
    // stopped drawing during a scene change, while memory was being unmapped and remapped.
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_WRITE_THROUGH");
        return env && env[0] == '1' && Common::PerfFeatureEnabled(103);
    }();
    if (!enabled || !liverpool || !liverpool->CanSendFaultCommandAsync()) {
        return false;
    }
    constexpr u64 PageSize = 4_KB;
    const VAddr page = Common::AlignDown(address, PageSize);
    if (size == 0 || address + size > page + PageSize) {
        return false;
    }
    {
        // A page stored to many times in a short while is drained once, as before, and then
        // written freely (each write-through costs a fault and a step).
        std::scoped_lock lk{write_through_mutex};
        const auto now = std::chrono::steady_clock::now();
        if (now - write_through_window > std::chrono::milliseconds{16}) {
            write_through_window = now;
            write_through_counts.clear();
        }
        if (++write_through_counts[page] > 24) {
            ++write_through_refused;
            return false;
        }
    }
    std::scoped_lock buffer_lock{mutex};
    // Stores over GPU-written bytes are left to the drain: letting them through (the page then
    // stayed watched, so each later store faulted too) flooded the recorder with tiny uploads
    // and the race fell to 12 FPS.
    if (!gpu_modified_ranges.Intersects(page, PageSize) ||
        gpu_modified_ranges.Intersects(address, size) || HasGpuImageAlias(page, PageSize) ||
        !memory->IsValidMapping(page, PageSize)) {
        return false;
    }
    return true;
}

void BufferCache::FinishWriteThrough(VAddr page, std::span<const u8> before, VAddr store,
                                     u64 store_size) {
    constexpr u64 PageSize = 4_KB;
    const auto* now = std::bit_cast<const u8*>(page);
    const u64 store_begin = store - page;
    const u64 store_end = store_begin + store_size;
    const auto changed = [&](u64 i) {
        return now[i] != before[i] || (i >= store_begin && i < store_end);
    };
    struct Run {
        VAddr address;
        std::vector<u8> bytes;
    };
    std::vector<Run> runs;
    for (u64 i = 0; i < PageSize;) {
        if (!changed(i)) {
            ++i;
            continue;
        }
        u64 end = i + 1;
        while (end < PageSize && changed(end)) {
            ++end;
        }
        runs.push_back({page + i, std::vector<u8>(now + i, now + end)});
        i = end;
    }
    {
        std::scoped_lock buffer_lock{mutex};
        for (const auto& run : runs) {
            // The store's bytes, and any another thread wrote while the page was open: the
            // CPU's are the newer ones there.
            if (gpu_modified_ranges.Intersects(run.address, run.bytes.size())) {
                gpu_modified_ranges.Subtract(run.address, run.bytes.size());
                write_through_gpu_bytes += run.bytes.size();
            }
        }
    }
    for (const auto& run : runs) {
        NoteCpuChange(run.address, run.bytes.size()); // FIX-069
        write_through_bytes += run.bytes.size();
    }
    ++write_through_stores;
    liverpool->SendFaultCommandAsync([this, runs = std::move(runs)] {
        for (const auto& run : runs) {
            WriteThroughUpload(run.address, run.bytes);
        }
    });
}

void BufferCache::WriteThroughUpload(VAddr address, std::span<const u8> data) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    // As InlineGuestWrite: earlier uploads of these bytes are recorded first, so they do not
    // land over them later.
    RecordPendingAheadUploads(0, ~u64{0} >> 1);
    const u64 size = data.size();
    const u64 first_block = address >> block_shift;
    const u64 last_block = (address + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    SynchronizeMemory(arena, address, u32(size), false, false);
    const auto staging = staging_pool.Request(size, MemoryType::HostUncached);
    std::memcpy(staging.mapped, data.data(), size);
    staging.Flush();
    const vk::BufferCopy copy = {
        .srcOffset = staging.offset,
        .dstOffset = address - arena->cpu_addr,
        .size = size,
    };
    runtime.CopyBuffer(staging.buffer, arena, std::span{&copy, 1});
}

std::string BufferCache::DescribeGpuWriters(VAddr address, u64 size) {
    std::string writers;
    auto it = large_gpu_writers.upper_bound(address + size);
    for (u32 n = 0; it != large_gpu_writers.begin() && n < 64; ++n) {
        --it;
        const auto& w = it->second;
        if (w.address + w.size > address && writers.size() < 600) {
            writers += fmt::format(" [{} {:#x} at {:#x}+{:#x}]", w.kind, w.tag, w.address, w.size);
        }
    }
    for (VAddr page = Common::AlignDown(address, 4_KB);
         page < address + std::min<u64>(size, 64_MB) && writers.size() < 600; page += 4_KB) {
        const auto small = small_gpu_writers.find(page);
        if (small != small_gpu_writers.end()) {
            const auto& w = small->second;
            writers += fmt::format(" [{} {:#x} at {:#x}+{:#x}]", w.kind, w.tag, w.address, w.size);
        }
    }
    return writers.empty() ? std::string(" none recorded") : writers;
}

std::string BufferCache::DescribeRange(VAddr address, u64 size) {
    constexpr u64 Limit = 256_MB;
    std::scoped_lock lk{vertex_pages_mutex};
    u64 pages = 0, gpu = 0, cpu = 0, hot = 0, changed = 0, unwatched = 0, unmapped = 0;
    for (VAddr page = Common::AlignDown(address, WatchedPageSize);
         page < address + std::min(size, Limit); page += WatchedPageSize) {
        ++pages;
        if (!memory->IsValidMapping(page, WatchedPageSize) || !memory->IsMappedAddress(page)) {
            ++unmapped;
            continue;
        }
        if (memory_tracker->IsRegionGpuModified(page, WatchedPageSize)) {
            ++gpu;
            continue;
        }
        cpu += memory_tracker->IsRegionCpuModified(page, WatchedPageSize);
        hot += memory_tracker->IsRegionHot(page, WatchedPageSize);
        const auto it = vertex_page_hashes.find(page >> WatchedPageBits);
        if (it == vertex_page_hashes.end()) {
            ++unwatched;
        } else if (Core::HashGuestBytes(page, WatchedPageSize) != it->second) {
            ++changed;
        }
    }
    return fmt::format("{} pages{}: {} GPU-written, {} CPU-modified, {} hot, {} changed since "
                       "upload, {} unwatched, {} unmapped; writers:{}",
                       pages, size > Limit ? " (first 256 MB)" : "", gpu, cpu, hot, changed,
                       unwatched, unmapped, DescribeGpuWriters(address, size));
}

void BufferCache::RefreshReadPages(VAddr address, u64 size, u64 shader_hash, bool quad_vertices) {
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    // FIX-017: the Nurburgring grass is drawn as quad lists and stretched across the screen.
    // In a capture its vertex pages were never uploaded during the frame, so the GPU drew an
    // older copy. Before the draw's uploads, each vertex page is compared with the guest bytes
    // it was last uploaded from. A changed page is either a PERF-012 hot page already uploaded
    // in this epoch ("hot"), or one the CPU changed without a tracked write ("untracked"); both
    // are uploaded again. -DisablePerf 29 only counts them.
    static const bool refresh = Common::PerfFeatureEnabled(29);
    constexpr u64 BigRange = 16_KB;
    std::scoped_lock lk{vertex_pages_mutex};
    auto& stats = vertex_page_stats;
    if (quad_vertices && size >= BigRange && grass_vertex_dumps_left.load() != 0) {
        // DIAG-032: the vertices the grass draws read, after the dumped grass dispatch.
        const u32 n = grass_vertex_dumps_left--;
        const auto base =
            grass_dump_dir / fmt::format("vertices_{}_{:#x}_vs{:#x}", 6 - n, address, shader_hash);
        DumpRange(address, size, base);
        LOG_WARNING(Render_Vulkan, "DIAG-032: dumped quad-list vertices {:#x}+{:#x} to {}", address,
                    size, base.string());
    }
    if (quad_vertices && size >= BigRange) {
        // DIAG-029: where the vertices of large quad-list draws (the grass) come from.
        const u64 count = ++big_quad_draws_logged;
        if (count <= 60 || count % 1000 == 0) {
            u32 pages = 0;
            u32 gpu_pages = 0;
            for (VAddr page = Common::AlignDown(address, WatchedPageSize); page < address + size;
                 page += WatchedPageSize) {
                ++pages;
                gpu_pages += memory_tracker->IsRegionGpuModified(page, WatchedPageSize);
            }
            LOG_WARNING(Render_Vulkan,
                        "DIAG-029: quad-list draw {} (vs {:#x}) vertices {:#x}+{:#x}: {} of {} "
                        "pages GPU-written; writers:{}",
                        count, shader_hash, address, size, gpu_pages, pages,
                        DescribeGpuWriters(address, size));
        }
    }
    // PERF-049: hashes taken ahead in the current epoch are used here too.
    const u32 epoch = g_upload_epoch.load(std::memory_order_acquire);
    const auto prehashed = prehasher ? prehasher->Current() : nullptr;
    VAddr mapped_from = 0;
    VAddr mapped_until = 0;
    for (VAddr page = Common::AlignDown(address, WatchedPageSize); page < address + size;
         page += WatchedPageSize) {
        // FIX-038: ClampRangeSize does not look up mappings below 1 GB, and IsValidMapping
        // accepts free areas, so an unmapped page (a buffer of the dealership shader's garbage
        // V#s) was hashed and faulted.
        // PERF-058: one map lookup per mapped area rather than two per page. A page counts as
        // mapped when the area containing its start, a mapped one, extends over all of it.
        if (page < mapped_from || page + WatchedPageSize > mapped_until) {
            mapped_from = page;
            mapped_until = memory->MappedAreaEnd(page, address + size);
        }
        if (page + WatchedPageSize > mapped_until ||
            memory_tracker->IsRegionGpuModified(page, WatchedPageSize)) {
            continue;
        }
        ++stats.checked;
        static_assert(WatchedPageSize == BYTES_PER_PAGE);
        u64 hash;
        if (!prehashed || !prehashed->Lookup(page, epoch, hash)) {
            hash = HashGuestPage(page);
        }
        const auto [it, inserted] = vertex_page_hashes.try_emplace(page >> WatchedPageBits, hash);
        if (inserted) {
            // What the GPU copy holds is unknown; upload it so the next check has a reference.
            ++stats.first_seen;
            if (refresh) {
                memory_tracker->ForceUpload(page, WatchedPageSize);
            }
            continue;
        }
        if (it->second == hash) {
            continue;
        }
        const bool hot = memory_tracker->IsRegionHot(page, WatchedPageSize);
        if (!hot && memory_tracker->IsRegionCpuModified(page, WatchedPageSize)) {
            // A tracked write: this draw's uploads copy the page.
            continue;
        }
        ++(hot ? stats.stale_hot : stats.stale_untracked);
        if ((!quad_vertices || size >= BigRange) && stats.big_stale_logged < 60) {
            ++stats.big_stale_logged;
            LOG_WARNING(Render_Vulkan,
                        "FIX-017: {} page {:#x} of shader {:#x} changed since its upload ({}) in "
                        "range {:#x}+{:#x}",
                        quad_vertices ? "quad-list vertex" : "watched input", page, shader_hash,
                        hot ? "hot page" : "untracked", address, size);
        }
        if (refresh) {
            memory_tracker->ForceUpload(page, WatchedPageSize);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - vertex_stats_time >= std::chrono::seconds{2}) {
        if (stats.stale_hot + stats.stale_untracked != 0) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-017: quad-list vertex and watched input pages in 2.0 s: {} checked, "
                        "{} first seen, "
                        "{} changed on hot pages, {} changed untracked{}",
                        stats.checked, stats.first_seen, stats.stale_hot, stats.stale_untracked,
                        refresh ? "; uploaded again" : "");
        }
        stats = {.big_stale_logged = stats.big_stale_logged};
        vertex_stats_time = now;
    }
}

bool BufferCache::SynchronizeMetadata(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    return false;
}

bool BufferCache::ExportImagesOver(VAddr device_addr, u32 size) {
    std::scoped_lock buffer_lock{mutex};
    const u64 before = image_exports;
    SynchronizeMemoryFromImage(device_addr, size);
    return image_exports != before;
}

bool BufferCache::SynchronizeMemoryFromImage(VAddr device_addr, u32 size) {
    if (size == 0) {
        return false;
    }
    // PERF-057: a range found free of images stays so until an image is registered or
    // unregistered. -DisablePerf 67 walks the texture cache every time.
    static const bool memo = Common::PerfFeatureEnabled(67);
    const u64 generation = texture_cache.RegistryGeneration();
    auto& slot =
        no_image_memo[((device_addr >> 6) ^ (device_addr >> 21) ^ size) % no_image_memo.size()];
    ++image_sync_stats.calls;
    if (memo && slot.address == device_addr && slot.size == size && slot.generation == generation) {
        ++image_sync_stats.exact_hits;
        return false;
    }
    // PERF-077: the same by 64 KB block. Constant and vertex rings put each draw's data at a new
    // address, so the exact-range memo above rarely hits; their blocks hold no images.
    // -DisablePerf 77 turns it off.
    static const bool block_memo = memo && Common::PerfFeatureEnabled(77);
    if (block_memo) {
        constexpr u32 BlockBits = 16;
        const VAddr first = device_addr >> BlockBits;
        const VAddr last = (device_addr + size - 1) >> BlockBits;
        bool all_free = last - first < 16;
        for (VAddr block = first; all_free && block <= last; ++block) {
            auto& entry = no_image_blocks[block % no_image_blocks.size()];
            if (entry.block == block && entry.generation == generation) {
                continue;
            }
            bool any = false;
            texture_cache.ForEachImageInRegion(block << BlockBits, VAddr{1} << BlockBits,
                                               [&](ImageId, Image&) { any = true; });
            if (any) {
                all_free = false;
            } else {
                entry = {block, generation};
            }
        }
        if (all_free) {
            ++image_sync_stats.block_hits;
            slot = {device_addr, size, generation};
            return false;
        }
    }
    image_sync_stats.full_bytes += size;
    // PERF-086: a range with nothing to export keeps nothing to export until an image is
    // registered or unregistered or an image's GpuModified, Dirty or BufferCoherent bit changes.
    // GT Sport binds a 588 MB buffer over its ~1,100 texture-pool images with many draws, and the
    // check below read every one of those images each time. -DisablePerf 98 checks every time.
    static const bool quiet_memo_enabled = memo && Common::PerfFeatureEnabled(98);
    const u64 export_state = g_image_export_state.load(std::memory_order_relaxed);
    auto& quiet =
        quiet_memo[((device_addr >> 8) ^ (device_addr >> 24) ^ size) % quiet_memo.size()];
    if (quiet_memo_enabled && quiet.address == device_addr && quiet.size == size &&
        quiet.generation == generation && quiet.export_state == export_state) {
        // DIAG-070: SHADGT_VERIFY_QUIET=1 checks the images anyway and logs disagreements.
        static const bool verify_quiet = [] {
            const char* env = std::getenv("SHADGT_VERIFY_QUIET");
            return env && env[0] == '1';
        }();
        if (verify_quiet) {
            const auto& ids = overlap_memo[((device_addr >> 8) ^ (device_addr >> 24) ^ size) %
                                           overlap_memo.size()];
            std::vector<ImageId> fresh;
            texture_cache.ForEachImageInRegion(device_addr, size,
                                               [&](ImageId id, Image&) { fresh.push_back(id); });
            for (const ImageId id : fresh) {
                const auto& image = texture_cache.GetImage(id);
                const ImageFlagBits flags = image.flags;
                if (True(flags & ImageFlagBits::GpuModified) &&
                    False(flags & ImageFlagBits::Dirty) &&
                    False(flags & ImageFlagBits::BufferCoherent)) {
                    static std::atomic<u32> logged{};
                    if (logged++ < 40) {
                        LOG_WARNING(Render_Vulkan,
                                    "DIAG-070: quiet range {:#x}+{:#x} has image {} at {:#x}+{:#x} "
                                    "to export (flags {:#x}, memo list {} of {}, state {})",
                                    device_addr, size, id.index, image.info.guest_address,
                                    image.info.guest_size, u32(flags),
                                    std::ranges::find(ids.ids, id) != ids.ids.end() ? "in" : "not in",
                                    ids.ids.size(), export_state);
                    }
                    break;
                }
            }
        }
        ++image_sync_stats.quiet_hits;
        ++image_sync_stats.with_images;
        ++image_sync_stats.nothing_to_export;
        return true;
    }
    // PERF-078: the images over a range stay the same until one is registered or unregistered,
    // so a repeated range reuses the list of its last walk. GT Sport binds buffers over its
    // texture pools with every draw (one of 588 MB over 2347 images), and each was walked and
    // its images searched for duplicates (quadratic) every time. One walk reports an image once,
    // so its list needs no search. -DisablePerf 78 walks and searches every time.
    static const bool overlap_memo_enabled = memo && Common::PerfFeatureEnabled(78);
    auto& overlap =
        overlap_memo[((device_addr >> 8) ^ (device_addr >> 24) ^ size) % overlap_memo.size()];
    // PERF-113: one list per recorder thread, reused (a new vector per call was ~1% of it).
    thread_local std::vector<ImageId> walked;
    walked.clear();
    const std::vector<ImageId>* first_level = &walked;
    if (overlap_memo_enabled && overlap.address == device_addr && overlap.size == size &&
        overlap.generation == generation) {
        first_level = &overlap.ids;
        ++image_sync_stats.overlap_hits;
    } else if (overlap_memo_enabled) {
        texture_cache.ForEachImageInRegion(device_addr, size,
                                           [&](ImageId id, Image&) { walked.push_back(id); });
        if (!walked.empty()) {
            overlap.address = device_addr;
            overlap.size = size;
            overlap.generation = generation;
            overlap.ids = walked;
        }
    } else {
        texture_cache.ForEachImageInRegion(device_addr, size, [&](ImageId id, Image&) {
            if (std::ranges::find(walked, id) == walked.end()) {
                walked.push_back(id);
            }
        });
    }
    const size_t requested_images = first_level->size();
    if (size >= (64u << 20)) {
        ++image_sync_stats.large;
        image_sync_stats.large_with_images += requested_images != 0;
        if (image_sync_stats.large <= 3) {
            image_sync_stats.samples += fmt::format(" {:#x}+{:#x} ({} images)", device_addr, size,
                                                    requested_images);
        }
    }
    if (requested_images == 0) {
        ++image_sync_stats.full_free;
        slot = {device_addr, size, generation};
        return false;
    }
    ++image_sync_stats.with_images;
    // PERF-078: only an image that is current on the GPU and not exported yet can be exported
    // (PlanImageAliasExports); with none over the range there is nothing to plan.
    if (overlap_memo_enabled && std::ranges::none_of(*first_level, [&](ImageId id) {
            const auto flags = texture_cache.GetImage(id).flags;
            return True(flags & ImageFlagBits::GpuModified) &&
                   False(flags & ImageFlagBits::Dirty) &&
                   False(flags & ImageFlagBits::BufferCoherent);
        })) {
        ++image_sync_stats.nothing_to_export;
        quiet = {device_addr, size, generation, export_state};
        return true;
    }
    std::vector<ImageId> image_ids = *first_level;
    // PERF-085: images already listed are marked by slot index (a hash set of every image over
    // the range cost about 5% of the recorder in races).
    if (++listed_stamp == 0) {
        std::ranges::fill(listed_stamps, 0u);
        listed_stamp = 1;
    }
    const auto mark_listed = [&](ImageId id) {
        if (id.index >= listed_stamps.size()) {
            listed_stamps.resize(std::max<size_t>(id.index + 1, listed_stamps.size() * 2), 0u);
        }
        u32& stamp = listed_stamps[id.index];
        if (stamp == listed_stamp) {
            return false;
        }
        stamp = listed_stamp;
        return true;
    };
    for (const auto id : image_ids) {
        mark_listed(id);
    }
    const auto collect = [&](ImageId id, Image&) {
        if (mark_listed(id)) {
            image_ids.push_back(id);
        }
    };
    // Full exports must also consider aliases outside the requested buffer slice.
    for (size_t i = 0; i < requested_images; ++i) {
        const auto& image = texture_cache.GetImage(image_ids[i]);
        if (image.SafeToDownload() && False(image.flags & ImageFlagBits::BufferCoherent)) {
            texture_cache.ForEachImageInRegion(image.info.guest_address, image.info.guest_size,
                                               collect);
        }
    }
    std::vector<ImageAliasFootprint> footprints;
    footprints.reserve(image_ids.size());
    // FIX-065: an image the GPU drew into a while ago, on memory whose CPU writes nobody watches
    // (a hot page tracked by hashing), may have had its memory reused by the game since: the
    // CPU's newer bytes would not mark it Dirty. GT Sport draws vertex data into render targets
    // in its per-frame memory; exporting such an old target into a buffer bound over it, and
    // writing that back on the next CPU fault, put old vertex data over a later frame's draw
    // constants and descriptor tables (the race summary's driver figure flickered and glowed
    // green). Those are not exported. -DisablePerf 89 exports them as before.
    static const bool skip_stale_exports = Common::PerfFeatureEnabled(89);
    constexpr s64 StaleNs = 50'000'000;
    const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    const auto may_be_stale = [&](const Image& image) {
        // Only an image that would be exported is checked, on at most 64 of its pages spread
        // over it (the whole walk cost about 2% of the recorder in races).
        if (!skip_stale_exports || False(image.flags & ImageFlagBits::GpuModified) ||
            True(image.flags & ImageFlagBits::Dirty) ||
            True(image.flags & ImageFlagBits::BufferCoherent) ||
            now_ns - image.gpu_write_ns < StaleNs) {
            return false;
        }
        const VAddr first = Common::AlignDown(image.info.guest_address, 4_KB);
        const VAddr end = image.info.guest_address + image.info.guest_size;
        const u64 pages = (end - first + 4_KB - 1) / 4_KB;
        const u64 step = std::max<u64>(1, pages / 64);
        for (u64 i = 0; i < pages; i += step) {
            if (page_manager.IsUnwatched(first + i * 4_KB)) {
                return true;
            }
        }
        return page_manager.IsUnwatched(Common::AlignDown(end - 1, 4_KB));
    };
    for (const auto id : image_ids) {
        const auto& image = texture_cache.GetImage(id);
        const auto& info = image.info;
        const bool stale = may_be_stale(image);
        image_sync_stats.stale_skipped += stale;
        const bool exportable = !stale && info.num_samples == 1 &&
                                image.backing->num_samples == 1 && !info.props.has_stencil &&
                                info.guest_size <= UINT32_MAX &&
                                info.pixel_format == image.backing->image.image_ci.format;
        // A stale image is treated as overwritten by the CPU: it neither exports nor holds back
        // newer images over the same memory.
        footprints.push_back(DescribeImageAlias(
            info.guest_address, info.guest_size,
            stale ? image.flags | ImageFlagBits::Dirty : image.flags,
            gpu_modified_ranges.Intersects(info.guest_address, info.guest_size), exportable));
    }
    for (const auto& export_image : PlanImageAliasExports(footprints, device_addr, size)) {
        Image& image = texture_cache.GetImage(image_ids[export_image.index]);
        const u64 first_block = export_image.address >> block_shift;
        const u64 last_block = (export_image.address + export_image.size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        EnsureResident(arena, first_block, last_block);
        const auto arena_offset =
            ImageAliasExportOffset(export_image, arena->cpu_addr, arena->size_bytes);
        ASSERT(arena_offset.has_value());
        SynchronizeMemory(arena, export_image.address, export_image.size, false, false);

        boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
        for (u32 mip = 0; mip < image.info.resources.levels; ++mip) {
            const auto& mip_info = image.info.mips_layout[mip];
            buffer_copies.push_back(vk::BufferImageCopy{
                .bufferOffset = mip_info.offset,
                .bufferRowLength = mip_info.pitch,
                .bufferImageHeight = mip_info.height,
                .imageSubresource{
                    .aspectMask = image.aspect_mask,
                    .mipLevel = mip,
                    .baseArrayLayer = 0,
                    .layerCount = image.info.resources.layers,
                },
                .imageOffset = {0, 0, 0},
                .imageExtent = {std::max(image.info.size.width >> mip, 1u),
                                std::max(image.info.size.height >> mip, 1u),
                                std::max(image.info.size.depth >> mip, 1u)},
            });
        }
        texture_cache.GetTileManager().TileImage(image, buffer_copies, arena, *arena_offset, true);
        InvalidateAsyncReadbacks(export_image.address, export_image.size);
        ApplyCompletedReadbacks();
        memory_tracker->MarkRegionAsGpuModified(export_image.address, export_image.size);
        gpu_modified_ranges.Add(export_image.address, export_image.size);
        NoteGpuWriteSeq(export_image.address, export_image.size); // FIX-069
        NoteGpuWrite(export_image.address, export_image.size);
        if (large_gpu_writers.size() > 16384) {
            large_gpu_writers.clear();
        }
        large_gpu_writers[export_image.address] =
            GpuWriter{"image export", 0, export_image.address, export_image.size};
        image.flags |= ImageFlagBits::BufferCoherent;
        ++image_exports;
        if (image_alias_exports_logged < 16) {
            ++image_alias_exports_logged;
            LOG_INFO(
                Render,
                "Preserved image alias base={:#x}, size={:#x}, extent={}x{}, request={:#x}+{:#x}",
                export_image.address, export_image.size, image.info.size.width,
                image.info.size.height, device_addr, size);
        }
    }
    return true;
}

void BufferCache::ReleaseUnmappedArenaMemory(VAddr address, u64 size) {
    if (size == 0 || address > ~u64{0} - size) return;
    std::scoped_lock buffer_lock{mutex};
    {
        std::scoped_lock lock{arena_unmap_mutex};
        // Preserve byte coverage: several 16 KiB guest unmaps can retire a larger
        // Vulkan sparse page. Remember only ranges actually backed by this cache.
        resident_ranges.ForEachInRange(address >> block_shift,
            ((address + size - 1) >> block_shift) + 1, [&](const Backing& range) {
                const u64 begin = std::max(address, range.start << block_shift);
                const u64 end = std::min(address + size, range.end << block_shift);
                if (begin < end) unmapped_arena_bytes.Add(begin, end - begin);
            });
    }
    std::map<vk::DeviceMemory, bool> whole;
    std::vector<Backing> released;
    IntervalList<Backing> retained;
    const auto collect = [&] {
        std::scoped_lock lock{arena_unmap_mutex};
        whole = Common::FullyUnmappedAllocations(resident_ranges, [&](const Backing& range) {
            return unmapped_arena_bytes.Contains(range.start << block_shift,
                                                (range.end - range.start) << block_shift);
        });
        released.clear();
        retained.Clear();
        for (const auto& range : resident_ranges) {
            if (whole[range.memory]) released.push_back(range);
            else retained.Add(range);
        }
        return !released.empty();
    };
    if (!collect()) return;
    // No marker lock across GPU submission/wait: map callbacks hold the guest registry.
    scheduler.Finish();
    // Submission callbacks may have changed cache residency; revalidate after finishing.
    if (!collect()) return;
    std::vector<ArenaBinds> unbinds;
    unbinds.reserve(arenas.size());
    for (const auto& arena : arenas) {
        ArenaBinds item{&arena};
        for (const auto& range : released) {
            const u64 begin_addr = std::max(range.start << block_shift, arena.cpu_addr);
            const u64 end_addr = std::min(range.end << block_shift,
                                         arena.cpu_addr + arena.size_bytes);
            if (begin_addr < end_addr) {
                item.binds.push_back(vk::SparseMemoryBind{
                    .resourceOffset = begin_addr - arena.cpu_addr,
                    .size = end_addr - begin_addr,
                    .memory = {},
                });
            }
        }
        if (!item.binds.empty()) unbinds.push_back(std::move(item));
    }
    std::vector<vk::SparseBufferMemoryBindInfo> infos;
    for (const auto& item : unbinds) {
        infos.push_back({item.arena->Handle(), static_cast<u32>(item.binds.size()), item.binds.data()});
    }
    const auto device = instance.GetDevice();
    const auto fence = Vulkan::Check(device.createFence({}));
    const vk::BindSparseInfo sparse_info{
        .bufferBindCount = static_cast<u32>(infos.size()),
        .pBufferBinds = infos.data(),
    };
    {
        std::scoped_lock queue_lock{Vulkan::Scheduler::submit_mutex};
        const auto result = instance.GetGraphicsQueue().bindSparse(sparse_info, fence);
        ASSERT_MSG(result == vk::Result::eSuccess, "Sparse unbind failed: {}", vk::to_string(result));
    }
    const auto waited = device.waitForFences(fence, true, ~u64{0});
    ASSERT_MSG(waited == vk::Result::eSuccess, "Sparse unbind wait failed: {}", vk::to_string(waited));
    device.destroyFence(fence);
    resident_ranges = std::move(retained);
    u64 freed_bytes = 0;
    for (const auto& range : released) {
        const VAddr start = range.start << block_shift;
        const u64 bytes = (range.end - range.start) << block_shift;
        freed_bytes += bytes;
        NoteArenaMapping(start, bytes);
        runtime.FillBuffer(bda_pagetable_buffer.get(), range.start * sizeof(vk::DeviceAddress),
                           (range.end - range.start) * sizeof(vk::DeviceAddress), 0u);
        gpu_modified_ranges.Subtract(start, bytes);
        memory_tracker->UnmarkRegionAsGpuModified(start, bytes, false);
        memory_tracker->MarkRegionAsCpuModified(start, bytes);
        std::scoped_lock uploaded_lock{uploaded_pages_mutex};
        for (VAddr page = start; page < start + bytes; page += 4_KB) {
            uploaded_page_contents.Forget(page);
        }
    }
    for (auto& memo : arena_memos) memo = {};
    for (const auto& [allocation, complete] : whole) {
        if (complete) device.freeMemory(allocation);
    }
    sparse_freed_bytes += freed_bytes;
    LOG_DEBUG(Render_Vulkan, "Released {} MiB of unmapped sparse buffer backing", freed_bytes >> 20);
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (instance.IsIntel()) {
        scheduler.SyncCommandThread(); // FIX-188: Intel Arc (#65)
    }
    std::scoped_lock buffer_lock{mutex}; // PERF-067
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    // PERF-084: the scheduler no longer holds the queue lock around its submit callback.
    std::scoped_lock queue_lock{Vulkan::Scheduler::submit_mutex};
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
