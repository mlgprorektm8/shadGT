// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

/// PERF-DIAG-012: what the command thread is doing when it writes GPU memory, for reports.
inline thread_local const char* g_gpu_write_kind = "unknown";
inline thread_local u64 g_gpu_write_tag = 0;

class TextureCache;
struct Image;
class MemoryTracker;
class PageManager;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false,
                          u64 exact_write_size = 0);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false,
                    u64 exact_write_size = 0);

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Returns true when host-memory shortcuts would bypass rendered image contents.
    [[nodiscard]] bool HasGpuImageAlias(VAddr addr, size_t size);

    /// GPU writes read back to guest memory once the GPU has executed them, so later CPU
    /// accesses do not have to drain the GPU: GDS-to-memory copies (PERF-009) and GPU-written
    /// pages the CPU has faulted on before (PERF-010). Recorded on the GPU thread.
    struct AsyncReadback {
        VAddr address;
        u32 size;
        Vulkan::StagingBufferRef download;
        bool valid = true;
        /// PERF-011: guest bytes when recorded; a mismatch at completion means the CPU wrote the
        /// range meanwhile, and its value is kept.
        std::vector<u8> snapshot;
        /// PERF-015: a GDS-to-memory copy; invalidated only by writes to its own bytes, and its
        /// bytes are kept out of uploads until its value reaches guest memory.
        bool is_gds = false;
    };
    void RecordGdsReadback(VAddr address, u32 gds_offset, u32 size);
    [[nodiscard]] bool HasPendingAsyncReadbacks() const {
        return num_pending_async_readbacks.load() != 0;
    }
    /// PERF-016: whether a CPU access drained the GPU in the last 100 ms (GPU thread).
    [[nodiscard]] bool DrainedRecently() const {
        return last_drain != std::chrono::steady_clock::time_point{} &&
               std::chrono::steady_clock::now() - last_drain < std::chrono::milliseconds{100};
    }

    /// Remembers pages the CPU read while they held GPU-written data (GPU thread).
    void NoteCpuReadFault(VAddr address, u64 size, u64 exact_write_size = 0);
    /// Records readbacks of remembered pages that hold GPU-written data again (GPU thread).
    /// Returns true when any were recorded.
    bool RecordHotPageReadbacks();
    /// Hands the recorded readbacks to a deferred fence (GPU thread).
    std::vector<std::shared_ptr<AsyncReadback>> TakePendingAsyncReadbacks();
    /// Writes completed readbacks to guest memory and clears their GPU-modified state, unless a
    /// later GPU write covered them (any thread, after the GPU work completed).
    void CompleteAsyncReadbacks(std::span<const std::shared_ptr<AsyncReadback>> readbacks);
    /// Frees staging memory of completed readbacks (GPU thread).
    void ReleaseFinishedAsyncReadbacks();

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

    /// FIX-017: before a quad-list draw's uploads, compares its vertex pages with the guest
    /// bytes they were last uploaded from and uploads again the ones that changed unnoticed.
    void RefreshReadPages(VAddr address, u64 size, u64 shader_hash, bool quad_vertices);

    /// DIAG-030: page states of a range (GPU-written, CPU-modified, hot, changed since its
    /// upload) and the recorded GPU writers overlapping it.
    std::string DescribeRange(VAddr address, u64 size);

private:
    void InvalidateAsyncReadbacks(VAddr address, u64 size);
    /// PERF-011: handles a CPU write fault on a page with GPU-written bytes that the write does
    /// not touch, without waiting for the GPU. Returns false when the exact path is needed.
    bool TrySplitWriteFault(const Buffer* arena, VAddr address, u64 size, VAddr window_start,
                            VAddr window_end, u64 exact_write_size);
    /// PERF-015: TrySplitWriteFault for pages whose GPU-written bytes are all pending GDS copies.
    bool TrySplitGdsWriteFault(VAddr address, u64 exact_write_size);
    /// Removes ranges whose read-back value reached guest memory from gpu_modified_ranges
    /// (GPU thread).
    void ApplyCompletedReadbacks();
    void LogHotPageStats();
    struct GpuWriter {
        const char* kind;
        u64 tag;
        VAddr address;
        u64 size;
    };
    std::unordered_map<VAddr, GpuWriter> small_gpu_writers;
    /// DIAG-029: GPU writes larger than 4 KB, by start address.
    std::map<VAddr, GpuWriter> large_gpu_writers;
    u64 big_quad_draws_logged{};

    std::mutex async_readbacks_mutex;
    std::vector<std::shared_ptr<AsyncReadback>> pending_async_readbacks;
    std::vector<std::shared_ptr<AsyncReadback>> inflight_async_readbacks;
    std::vector<Vulkan::StagingBufferRef> finished_async_downloads;
    std::atomic<u32> num_pending_async_readbacks{};
    std::vector<std::pair<VAddr, u64>> completed_readback_ranges;
    std::atomic<u32> num_completed_readback_ranges{};
    std::atomic<u32> num_valid_gds_readbacks{};
    const bool split_gds_write_faults;
    std::chrono::steady_clock::time_point last_drain{};
    const bool split_write_faults;

    static constexpr size_t MaxHotPages = 1024;
    std::unordered_set<VAddr> hot_pages;
    std::deque<VAddr> hot_page_order;
    struct HotPageStats {
        std::chrono::steady_clock::time_point window_start{};
        u32 faults{};
        u32 page_only_faults{};
        u32 split_faults{};
        u32 gds_split_faults{};
        std::atomic<u32> cpu_overwrote{};
        struct PageFaults {
            u32 faults{};
            u64 last_offset{};
            u64 last_size{};
            bool touches_gpu_bytes{};
            std::string gpu_bytes;
        };
        std::unordered_map<VAddr, PageFaults> pages;
        u32 drain_reports{};
        u32 new_pages{};
        u32 recorded{};
        u64 recorded_bytes{};
        std::atomic<u32> completed{};
        std::atomic<u32> invalidated{};
    } hot_page_stats;
    std::atomic<u32> num_tracked_async_readbacks{};

    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMetadata(const Buffer* arena, VAddr device_addr, u32 size);

    void SynchronizeMemoryFromImage(VAddr device_addr, u32 size);
    /// DIAG-028: records the guest bytes of watched pages as they are uploaded.
    void RecordWatchedUploads(VAddr start, VAddr end);
    /// DIAG-029: the recorded GPU writes overlapping a range.
    std::string DescribeGpuWriters(VAddr address, u64 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    u32 image_alias_exports_logged{};
    /// DIAG-028: hash of the guest bytes each watched page (4 KB) was last uploaded from.
    std::mutex vertex_pages_mutex;
    std::unordered_map<u64, u64> vertex_page_hashes;
    struct VertexPageStats {
        u64 checked{};
        u64 first_seen{};
        u64 stale_hot{};
        u64 stale_untracked{};
        u64 big_stale_logged{};
    } vertex_page_stats;
    std::chrono::steady_clock::time_point vertex_stats_time{};

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};
    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
