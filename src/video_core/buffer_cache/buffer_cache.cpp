// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <immintrin.h>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "common/alignment.h"
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

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

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
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
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
}

BufferCache::~BufferCache() {
    g_upload_epoch_listener.store(nullptr, std::memory_order_release);
    s_prehasher.store(nullptr, std::memory_order_release);
    prehasher.reset();
}

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::RecordGdsReadback(VAddr address, u32 gds_offset, u32 size) {
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
        ++hot_page_stats.new_pages;
        hot_page_order.push_back(page);
        if (hot_page_order.size() > MaxHotPages) {
            hot_pages.erase(hot_page_order.front());
            hot_page_order.pop_front();
        }
    }
}

bool BufferCache::RecordHotPageReadbacks() {
    LogHotPageStats();
    if (hot_page_order.empty()) {
        return false;
    }
    constexpr u64 PageSize = 4_KB;
    std::unordered_map<VAddr, bool> inflight_pages;
    {
        std::scoped_lock lk{async_readbacks_mutex};
        for (const auto* list : {&pending_async_readbacks, &inflight_async_readbacks}) {
            for (const auto& readback : *list) {
                if (readback->valid) {
                    inflight_pages[Common::AlignDown(readback->address, PageSize)] = true;
                }
            }
        }
    }
    std::vector<std::shared_ptr<AsyncReadback>> recorded;
    for (const VAddr page : hot_page_order) {
        // A CPU-modified page may hold CPU writes over GPU ranges (PERF-011); those ranges are
        // only read back with a snapshot check, at write faults.
        if (inflight_pages.contains(page) || !memory_tracker->IsRegionGpuModified(page, PageSize) ||
            memory_tracker->IsRegionCpuModified(page, PageSize)) {
            continue;
        }
        const u64 block = page >> block_shift;
        const auto* arena = GetArena(block, block);
        // Same ranges as DownloadMemory: only bytes the GPU wrote, so CPU writes not yet uploaded
        // to the arena are never overwritten.
        memory_tracker->ForEachDownloadRange<false>(page, PageSize, [&](u64 address, u64 size) {
            gpu_modified_ranges.ForEachInRange(address, size, [&](VAddr start, VAddr end) {
                auto readback = std::make_shared<AsyncReadback>();
                readback->address = start;
                readback->size = static_cast<u32>(end - start);
                readback->download =
                    staging_pool.Request(readback->size, MemoryType::HostCached, 4, true);
                const vk::BufferCopy copy = {
                    .srcOffset = start - arena->cpu_addr,
                    .dstOffset = readback->download.offset,
                    .size = readback->size,
                };
                runtime.CopyBuffer(arena, readback->download.buffer, std::span{&copy, 1});
                hot_page_stats.recorded_bytes += readback->size;
                recorded.push_back(std::move(readback));
            });
        });
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
                "superseded;{}",
                stats.faults, stats.page_only_faults, stats.split_faults, stats.gds_split_faults,
                stats.cpu_overwrote.exchange(0), stats.new_pages, hot_page_order.size(),
                stats.recorded, stats.recorded_bytes / 1024, stats.completed.exchange(0),
                stats.invalidated.exchange(0), top);
    // PERF-033: hot-page uploads avoided because the bytes had not changed.
    const u64 unchanged = hot_pages_unchanged.exchange(0);
    const u64 uploaded = hot_pages_uploaded.exchange(0);
    if (unchanged + uploaded != 0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-033 hot pages in 2.0 s: {} unchanged and not uploaded ({} MB), {} "
                    "uploaded; PERF-049: {} hashes taken ahead, {} on the recorder",
                    unchanged, unchanged * BYTES_PER_PAGE >> 20, uploaded, prehashed_hits,
                    prehashed_misses);
    }
    prehashed_hits = 0;
    prehashed_misses = 0;
    if (ahead_hits + ahead_misses != 0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-062: hot-page hashes taken by the command thread: {} used, {} hashed "
                    "here; {} hashed ahead",
                    ahead_hits, ahead_misses, ahead_hashed.exchange(0));
    }
    ahead_hits = 0;
    ahead_misses = 0;
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
    stats.pages.clear();
    stats.window_start = now;
    stats.faults = 0;
    stats.new_pages = 0;
    stats.recorded = 0;
    stats.recorded_bytes = 0;
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
    std::scoped_lock lk{async_readbacks_mutex};
    for (const auto& readback : readbacks) {
        if (readback->valid && memory->IsValidMapping(readback->address, readback->size) &&
            !readback->snapshot.empty() &&
            std::memcmp(std::bit_cast<const void*>(readback->address), readback->snapshot.data(),
                        readback->size) != 0) {
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
    for (const auto& [address, size] : ranges) {
        gpu_modified_ranges.Subtract(address, size);
    }
    for (const auto& [address, size] : ranges) {
        const VAddr first_page = Common::AlignDown(address, 4_KB);
        const VAddr last_page = Common::AlignDown(address + size - 1, 4_KB);
        for (VAddr page = first_page; page <= last_page; page += 4_KB) {
            if (!gpu_modified_ranges.Intersects(page, 4_KB)) {
                memory_tracker->UnmarkRegionAsGpuModified(page, 4_KB, false);
            }
        }
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
    memory_tracker->InvalidateRegion(
        device_addr, size, [this, device_addr, size, assume_locks, exact_write_size] {
            ReadMemory(device_addr, size, true, assume_locks, exact_write_size);
        });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks,
                             u64 exact_write_size) {
    const auto flush_request = [this, device_addr, size, is_write, exact_write_size] {
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
        DownloadMemory(arena, window_start, window_end - window_start);
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

std::shared_ptr<BufferCache::FaultDownload> BufferCache::BeginFaultFlush(VAddr device_addr,
                                                                         u64 size, bool is_write,
                                                                         u64 exact_write_size) {
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
    memory_tracker->ForEachDownloadRange<false>(
        window_start, window_end - window_start, [&](u64 address, u64 range_size) {
            gpu_modified_ranges.ForEachInRange(address, range_size, [&](VAddr start, VAddr end) {
                fault->copies.push_back(vk::BufferCopy{
                    .srcOffset = start - arena_base,
                    .dstOffset = total_size_bytes,
                    .size = end - start,
                });
                fault->ranges.emplace_back(start, end);
                total_size_bytes += Common::AlignUp(end - start, 64);
            });
        });
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
    fault->download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 0, true);
    for (auto& copy : fault->copies) {
        copy.dstOffset += fault->download.offset;
    }
    runtime.CopyBuffer(arena, fault->download.buffer, fault->copies);
    fault->snapshots.reserve(fault->copies.size());
    for (const auto& copy : fault->copies) {
        const u8* guest = std::bit_cast<const u8*>(arena_base + copy.srcOffset);
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
            const u8* snapshot = fault->snapshots[c].data();
            const u8* guest = std::bit_cast<const u8*>(start);
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

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
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
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
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
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    if (is_written) {
        InvalidateAsyncReadbacks(device_addr, size);
    }
    // After the invalidation: a readback that completed before it must not later remove the
    // ranges this write adds.
    ApplyCompletedReadbacks();
    ++Common::GetWorkCounters().obtain_buffer;
    const bool images_overlap = SynchronizeMemoryFromImage(device_addr, size);
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
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
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

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
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
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
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

        for (u32 block = 0; block < bind.size; block += block_size) {
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
        hashes[i] = XXH3_64bits(std::bit_cast<const void*>(page_at(i)), BYTES_PER_PAGE);
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
                                   XXH3_64bits(std::bit_cast<const void*>(page), BYTES_PER_PAGE));
            }
        }
    });
    ahead_hashed.fetch_add(pages.size(), std::memory_order_relaxed);
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
        if (split_write_faults && gpu_modified_ranges.Intersects(page, BYTES_PER_PAGE)) {
            return false;
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
    if (read_capture && !hot.empty()) {
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
            } else if (whole_page_from_guest(page)) {
                hot_uploads.push_back({page, generation, hash});
            } else {
                std::scoped_lock lk{uploaded_pages_mutex};
                uploaded_page_contents.Forget(page);
            }
        }
        if (cursor < addr + range_size) {
            upload_range(cursor, addr + range_size - cursor);
        }
    }
    hot_pages_unchanged += unchanged_count;
    if (!copies.empty()) {
        for (const auto& copy : copies) {
            RecordWatchedUploads(copy.dstOffset, copy.dstOffset + copy.size);
        }
        Common::GetWorkCounters().uploads += copies.size();
        Common::GetWorkCounters().upload_bytes += total_size_bytes;
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
        runtime.CopyBuffer(staging.buffer, arena, copies);
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
    return XXH3_64bits(std::bit_cast<const void*>(page), BYTES_PER_PAGE);
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
    const u64 size = data.size();
    const u64 first_block = address >> block_shift;
    const u64 last_block = (address + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    // Pending CPU writes around the bytes first, so they do not later land over them.
    SynchronizeMemory(arena, address, u32(size), false, false);
    runtime.InlineData(arena, arena->Offset(address), data);
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
        } else if (XXH3_64bits(std::bit_cast<const void*>(page), WatchedPageSize) != it->second) {
            ++changed;
        }
    }
    return fmt::format("{} pages{}: {} GPU-written, {} CPU-modified, {} hot, {} changed since "
                       "upload, {} unwatched, {} unmapped; writers:{}",
                       pages, size > Limit ? " (first 256 MB)" : "", gpu, cpu, hot, changed,
                       unwatched, unmapped, DescribeGpuWriters(address, size));
}

void BufferCache::RefreshReadPages(VAddr address, u64 size, u64 shader_hash, bool quad_vertices) {
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
            mapped_until = memory->MappedAreaEnd(page);
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
    if (memo && slot.address == device_addr && slot.size == size && slot.generation == generation) {
        return false;
    }
    std::vector<ImageId> image_ids;
    const auto collect = [&](ImageId id, Image&) {
        if (std::ranges::find(image_ids, id) == image_ids.end()) {
            image_ids.push_back(id);
        }
    };
    texture_cache.ForEachImageInRegion(device_addr, size, collect);
    const size_t requested_images = image_ids.size();
    if (requested_images == 0) {
        slot = {device_addr, size, generation};
        return false;
    }
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
    for (const auto id : image_ids) {
        const auto& image = texture_cache.GetImage(id);
        const auto& info = image.info;
        const bool exportable = info.num_samples == 1 && image.backing->num_samples == 1 &&
                                !info.props.has_stencil && info.guest_size <= UINT32_MAX &&
                                info.pixel_format == image.backing->image.image_ci.format;
        footprints.push_back(DescribeImageAlias(
            info.guest_address, info.guest_size, image.flags,
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
        NoteGpuWrite(export_image.address, export_image.size);
        if (large_gpu_writers.size() > 16384) {
            large_gpu_writers.clear();
        }
        large_gpu_writers[export_image.address] =
            GpuWriter{"image export", 0, export_image.address, export_image.size};
        image.flags |= ImageFlagBits::BufferCoherent;
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

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
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
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
