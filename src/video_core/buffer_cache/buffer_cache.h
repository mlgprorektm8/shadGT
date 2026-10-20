// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/texture_cache/types.h"
#include "common/read_capture.h"
#include "common/recursive_shared_mutex.h"
#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/buffer_cache/page_hash_table.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/buffer_cache/uploaded_pages.h"
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

class HotPagePrehasher; // PERF-049

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
    u64 SparseResidentBytes();

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

    /// PERF-062 (command thread): hashes the recorded pages lying wholly inside these read
    /// ranges for the upload epoch the draw was decoded in, so the recorder need not.
    void HashPagesAhead(std::span<const std::pair<VAddr, u64>> ranges, u32 epoch);

    /// PERF-062 (recorder): the upload epoch the draw being recorded was decoded in (0: none).
    void SetDrawEpoch(u32 epoch) {
        draw_epoch = epoch;
    }

    /// PERF-063 (recorder): the capture of the job being recorded (null: guest memory as is).
    void SetReadCapture(const Common::ReadCapture* capture) {
        read_capture = capture;
    }

    /// PERF-063: XXH3 of a guest page, from the current capture when it holds the page.
    u64 HashGuestPage(VAddr page) const;

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false);
    std::pair<const Buffer*, u64> ObtainBufferImpl(VAddr device_addr, u32 size, bool is_written,
                                                   bool is_texel_buffer);
    /// R6
    std::optional<std::pair<const Buffer*, u64>> ObtainBufferShared(VAddr device_addr, u32 size,
                                                                    bool is_written,
                                                                    bool is_texel_buffer);
    std::optional<bool> ImagesOverQuick(VAddr device_addr, u32 size) const;
    /// P2
    const Buffer* TryImportView(VAddr device_addr, u32 size);
    std::mutex import_mutex;
    std::unordered_set<VAddr> import_tried; // view bases
    std::unordered_map<VAddr, std::pair<VAddr, VAddr>> view_extents; // per 2 MB block
    std::vector<std::unique_ptr<Buffer>> imported_buffers;
    std::atomic<u64> import_rejected{};
    std::atomic<u64> inplace_obtains{};
    std::atomic<u64> ceil_skipped_syncs{}; // EXP-126
    std::atomic<u64> shared_obtains{};

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// PERF-067: an upload the buffer stage prepared; the recorder records its copy.
    struct AheadUpload {
        const Buffer* staging;
        const Buffer* arena;
        boost::container::small_vector<vk::BufferCopy, 4> copies;
        VAddr begin; // guest range the copies write
        VAddr end;
        bool recorded{};
    };
    using AheadUploads = std::vector<std::shared_ptr<AheadUpload>>;
    /// PERF-067 (buffer stage thread): ObtainBuffer for the simple, common case, ahead of the
    /// recorder: no image over the range, no metadata, the arena resident. Uploads it needs go
    /// to `uploads` for the recorder to record. Nothing (and nothing changed) otherwise; the
    /// recorder then obtains the buffer itself.
    /// PERF-079: a range under images is obtained too (through the arena); `overlapped` says so,
    /// and the recorder exports the images before it uses the buffer (ExportImagesOver).
    [[nodiscard]] std::optional<std::pair<const Buffer*, u64>> ObtainBufferAhead(
        VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer, AheadUploads& uploads,
        bool* overlapped = nullptr);
    /// PERF-079 (recorder): exports the images over a range obtained ahead, as ObtainBuffer
    /// would have first. True when it exported any (the buffer is then obtained again).
    bool ExportImagesOver(VAddr device_addr, u32 size);
    /// PERF-067 (recorder): records the copies of a draw's prepared uploads not recorded yet
    /// (ahead of the session's draws when PERF-066 is on).
    void RecordAheadUploads(const AheadUploads& uploads);
    /// PERF-067: the cache is used by the buffer stage thread and the recorder; every entry
    /// point takes this.
    struct MutexTag {};
    using CacheMutex = Common::RecursiveSharedMutex<MutexTag>;
    CacheMutex& Mutex() noexcept {
        return mutex;
    }

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Returns true when host-memory shortcuts would bypass rendered image contents.
    [[nodiscard]] bool HasGpuImageAlias(VAddr addr, size_t size);

    /// FIX-122 (direct reads): HasGpuImageAlias with a per-thread memo of 64 KB blocks that
    /// hold no images, for ranges the GPU would read in place from guest memory.
    [[nodiscard]] bool DirectReadHitsImage(VAddr addr, size_t size);

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
        /// DIAG-072: the GPU writers of the range and when it was recorded.
        std::string writer;
        s64 recorded_ns{};
    };
    /// DIAG-072 (SHADGT_TRACE_READBACKS=1): recent readbacks written into guest memory over a
    /// range: when recorded and completed, how many bytes they changed, and their GPU writers.
    std::string DescribeRecentReadbacks(VAddr address, u64 size);
    /// DIAG-073: the readback state of a page (hot since, snapshot, CPU bytes changed).
    std::string DescribeReadbackPage(VAddr address);
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
    /// Release whole sparse allocations covered by an explicit guest unmap, on the recorder.
    void ReleaseUnmappedArenaMemory(VAddr address, u64 size);

    /// FIX-017: before a quad-list draw's uploads, compares its vertex pages with the guest
    /// bytes they were last uploaded from and uploads again the ones that changed unnoticed.
    void RefreshReadPages(VAddr address, u64 size, u64 shader_hash, bool quad_vertices);

    /// DIAG-032: writes the GPU copy of a guest range, and the guest bytes, to files (waits for
    /// the GPU). For one-off dumps of the Nurburgring grass buffers.
    void DumpRange(VAddr address, u64 size, const std::filesystem::path& base);
    /// DIAG-032: the next large quad-list vertex ranges to dump after a grass dispatch dump.
    std::atomic<u32> grass_vertex_dumps_left{};
    std::filesystem::path grass_dump_dir;

    /// FIX-019: writes guest bytes into the GPU copy of their memory in command order, without
    /// changing how the memory is tracked (guest memory receives the same bytes later).
    void InlineGuestWrite(VAddr address, std::span<const u8> data);

    /// FIX-079 (faulting thread): whether a CPU store of `size` bytes at `address` may write
    /// through a page that holds GPU-written bytes instead of draining the GPU.
    bool CanWriteThrough(VAddr address, u64 size);
    /// FIX-079 (faulting thread, after the store): the page's bytes that changed since `before`
    /// are sent to the GPU copy in order; the page stays watched and GPU-written.
    void FinishWriteThrough(VAddr page, std::span<const u8> before, VAddr store, u64 store_size);

    /// DIAG-030: page states of a range (GPU-written, CPU-modified, hot, changed since its
    /// upload) and the recorded GPU writers overlapping it.
    std::string DescribeRange(VAddr address, u64 size);

private:
    /// PERF-052: a game-thread fault's download, recorded on the recorder; the GPU wait and the
    /// write to guest memory happen off the recorder, in GPU order with async readbacks.
    struct FaultDownload {
        VAddr device_addr{};
        u64 size{};
        bool is_write{};
        VAddr window_start{};
        VAddr window_end{};
        u64 tick{};
        u64 seq{}; // GPU writes logged after this one are newer than the download
        Vulkan::StagingBufferRef download{};
        boost::container::small_vector<vk::BufferCopy, 4> copies;
        boost::container::small_vector<std::pair<VAddr, VAddr>, 4> ranges;
        /// Bytes an earlier fault released before this one reached guest memory: the CPU may
        /// have written them since, so they are not written again (fault_downloads_mutex).
        std::vector<std::pair<VAddr, VAddr>> released;
        /// Guest bytes of each copy when the download was recorded; a byte the CPU changed
        /// since is newer than the GPU's and is not written (PERF-052b).
        std::vector<std::vector<u8>> snapshots;
        std::atomic<bool> applied{};
        std::binary_semaphore done{0};
    };
    /// Recorder. Null when nothing had to come from the GPU (the flush is complete).
    std::shared_ptr<FaultDownload> BeginFaultFlush(VAddr device_addr, u64 size, bool is_write,
                                                   u64 exact_write_size);
    /// Recorder, once the download reached guest memory.
    void FinishFaultFlush(FaultDownload& fault);
    /// Recorder: guest writes of fault downloads older than tick have happened.
    void WaitForFaultWritebacks(u64 tick);
    void NoteGpuWrite(VAddr address, u64 size);
    std::mutex fault_downloads_mutex;
    std::deque<std::shared_ptr<FaultDownload>> fault_downloads;
    std::atomic<u32> num_fault_downloads{};
    u64 gpu_write_seq{};
    std::vector<std::tuple<u64, VAddr, VAddr>> gpu_write_log; // while downloads are pending
    std::atomic<u64> async_fault_flushes{};
    std::atomic<u64> async_fault_wait_us{};

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
    std::atomic<u64> stage_prehash_hits{};   // PERF-087
    std::atomic<u64> stage_prehash_misses{};
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
        std::atomic<u32> uploads_over_gpu_bytes{}; // FIX-066
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
    // PERF-033: hot pages whose bytes match their last upload are not uploaded again.
    UploadedPageContents uploaded_page_contents;
    std::mutex uploaded_pages_mutex;
    // PERF-057: ranges SynchronizeMemoryFromImage found free of images.
    struct NoImageRange {
        VAddr address{};
        u32 size{};
        u64 generation{};
    };
    std::vector<NoImageRange> no_image_memo = std::vector<NoImageRange>(65536);
    /// PERF-077: 64 KB blocks found free of images, with the registry generation of the check.
    struct NoImageBlock {
        VAddr block;
        u64 generation;
    };
    std::vector<NoImageBlock> no_image_blocks = std::vector<NoImageBlock>(16384);
    /// PERF-078: the images over a range, with the registry generation of the walk.
    struct ImageOverlapMemo {
        VAddr address;
        u32 size;
        u64 generation;
        std::vector<ImageId> ids;
    };
    std::vector<ImageOverlapMemo> overlap_memo = std::vector<ImageOverlapMemo>(4096);
    /// PERF-086: ranges over images with nothing to export, with the registry generation and
    /// the image export state (g_image_export_state) of the check.
    struct QuietRange {
        VAddr address;
        u32 size;
        u64 generation;
        u64 export_state;
    };
    std::vector<QuietRange> quiet_memo = std::vector<QuietRange>(4096);
    /// PERF-117: the last synchronization of a read-only arena range (see ObtainBuffer).
    struct ArenaMemo {
        VAddr address{};
        u32 size{};
        bool texel{};
        bool written{};
        const Buffer* arena{};
        u32 epoch{};
        u64 versions{};
        u64 registry{};
        u64 export_state{};
    };
    std::vector<ArenaMemo> arena_memos = std::vector<ArenaMemo>(8192);
    u64 arena_memo_hits{};
    u64 arena_memo_stores{};
    /// PERF-085: images already listed in this export plan, by slot index.
    std::vector<u32> listed_stamps;
    u32 listed_stamp{};
    /// DIAG-048: how SynchronizeMemoryFromImage answers, per 2 s.
    struct ImageSyncStats {
        u64 calls, exact_hits, block_hits, full_free, with_images, exported, generation;
        u64 overlap_hits, nothing_to_export, stale_skipped, quiet_hits;
        u64 full_bytes, full_pages_with_entries;
        u64 large, large_with_images;
        std::string samples;
    } image_sync_stats{};
    u64 image_exports{}; // PERF-079: images exported to buffers so far
    std::atomic<u64> hot_pages_unchanged{};
    std::atomic<u64> hot_pages_uploaded{};
    // PERF-064: per hot page, how many checks in a row found it unchanged.
    tsl::robin_map<VAddr, u8> hot_unchanged_streak;
    std::mutex streak_mutex;     // R3: hot_unchanged_streak (shared-mode syncs)
    std::mutex gpu_ranges_mutex; // R3: gpu_modified_ranges between shared-mode syncs
    std::mutex pending_uploads_mutex; // R3: pending_ahead_uploads
    std::atomic<u64> hot_pages_cooled{};
    std::atomic<u64> hot_pages_cooled_changed{};
    // PERF-062: hot-page hashes taken by the command thread as it decodes draws.
    PageHashTable ahead_hashes;
    // R1: per thread (the buffer stage and the recorder each have their own job).
    static thread_local u32 draw_epoch;
    static thread_local const Common::ReadCapture* read_capture;
    std::atomic<u64> ahead_hashed{};
    std::atomic<u64> ahead_hits{};
    std::atomic<u64> capture_hash_hits{};   // PERF-063
    std::atomic<u64> capture_hash_misses{};
    std::atomic<u64> ahead_misses{};
    // PERF-049: hot-page hashes taken ahead on a background thread.
    std::unique_ptr<HotPagePrehasher> prehasher;
    std::atomic<u64> prehashed_hits{};
    std::atomic<u64> prehashed_misses{};
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

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size, VAddr fault_addr,
                        u64 fault_size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMetadata(const Buffer* arena, VAddr device_addr, u32 size);

    /// Returns whether any image overlaps [device_addr, device_addr + size).
    bool SynchronizeMemoryFromImage(VAddr device_addr, u32 size);
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
    PageManager& page_manager; // FIX-063

    StreamBuffer stream_buffer;
    // PERF-067: rings the buffer stage fills ahead of the recorder.
    StreamBuffer ahead_stream;
    StreamBuffer ahead_staging;
public:
    /// R5: buffer stage workers (SHADGT_STAGE_WORKERS, 1-4). Each has its own rings; worker 0
    /// uses the ones above.
    static constexpr u32 MaxStageWorkers = 4;
    static u32 StageWorkerCount();
    /// D1: the direct memory model (null unless SHADGT_DIRECT=1).
    DirectMemory* GetDirectMemory() noexcept {
        return direct_memory.get();
    }

    /// R5 (stage worker thread): which worker this thread is.
    static void SetStageWorker(u32 index) noexcept {
        t_stage_worker = index;
    }
    /// R6: called on the recorder thread when it starts (its obtains may run shared).
    /// EXP-127: set while a direct graphics draw binds its buffers.
    static inline thread_local bool t_ceil_draw{};

    static void MarkRecorderThread() noexcept {
        t_recorder_thread = true;
    }
    /// R5 (recorder, before using a buffer obtained ahead): waits until no stage worker is
    /// synchronizing memory over the range, then records the uploads prepared over it, so a
    /// range another worker found dirty and uploads for a later draw is current for this one.
    void SettleAheadRange(VAddr device_addr, u64 size);
private:
    static inline thread_local u32 t_stage_worker{};
    static inline thread_local bool t_recorder_thread{};
    StreamBuffer& AheadStream() noexcept {
        return t_stage_worker == 0 ? ahead_stream : *worker_streams[t_stage_worker];
    }
    StreamBuffer& AheadStaging() noexcept {
        return t_stage_worker == 0 ? ahead_staging : *worker_stagings[t_stage_worker];
    }
    std::array<std::unique_ptr<StreamBuffer>, MaxStageWorkers> worker_streams;
    std::array<std::unique_ptr<StreamBuffer>, MaxStageWorkers> worker_stagings;
    struct WorkerSync {
        std::atomic<VAddr> begin{};
        std::atomic<VAddr> end{};
    };
    std::array<WorkerSync, MaxStageWorkers> worker_syncs;
    std::unique_ptr<DirectMemory> direct_memory; // D1
    static thread_local AheadUploads* ahead_uploads; // set while ObtainBufferAhead runs
public:
    /// PERF-067: why the stage left buffers to the recorder (texel read, image, stream ring,
    /// arena, residency, staging ring).
    std::array<std::atomic<u64>, 6> ahead_refusals{};
private:
    // Prepared uploads not recorded yet, oldest first. The stage has already marked their
    // bytes uploaded; anything the recorder obtains over them records them first.
    std::deque<std::shared_ptr<AheadUpload>> pending_ahead_uploads;
    void RecordPendingAheadUploads(VAddr device_addr, u64 size);
    void RecordAheadUpload(AheadUpload& upload);
    mutable CacheMutex mutex; // R2: exclusive unless a path is known safe shared
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    /// FIX-079: the bytes a write-through store changed, copied to the arena on the recorder.
    void WriteThroughUpload(VAddr address, std::span<const u8> data);
    std::mutex write_through_mutex;
    std::unordered_map<VAddr, u32> write_through_counts;
    std::chrono::steady_clock::time_point write_through_window{};
    std::atomic<u64> write_through_stores{};
    std::atomic<u64> write_through_bytes{};
    std::atomic<u64> write_through_refused{};
    std::atomic<u64> write_through_gpu_bytes{};
    // FIX-069: which of the CPU and the GPU wrote memory last, in one sequence. GPU bytes reach
    // guest memory only where the GPU wrote after the CPU last changed the page.
    RangeMap<u64> gpu_write_seqs;
    std::mutex cpu_change_mutex;
    std::unordered_map<VAddr, u64> cpu_change_seqs; // per 4 KB page
    std::atomic<u64> write_seq{1};
    u64 writebacks_cpu_newer{};
    void NoteCpuChange(VAddr address, u64 size);
    u64 NextWriteSeq();
    bool FreshGpuWrite(VAddr start, VAddr end);
    void NoteGpuWriteSeq(VAddr address, u64 size);
    // FIX-073: the guest bytes of each readback page as the GPU copy last took them in (when the
    // GPU wrote over them, or the renderer wrote GPU data into them). Bytes that differ now are
    // CPU writes nobody saw (these pages are not write-protected).
    std::mutex readback_hash_mutex;
    std::unordered_map<VAddr, std::unique_ptr<std::array<u8, 4096>>> readback_page_snapshots;
    std::unordered_map<VAddr, s64> hot_page_since_ns; // DIAG-073: when each page became hot
    std::unordered_set<VAddr> pending_snapshot_pages; // FIX-073: hot, snapshot at next write-back
    void NoteSnapshotUpdate(VAddr page, const char* reason);
    std::unordered_map<VAddr, std::pair<s64, const char*>> snapshot_updates; // DIAG-073
    u64 unseen_cpu_bytes_synced{};
    void SyncUnseenCpuWrites(const Buffer* arena, VAddr address, u64 size);
    bool CpuRewroteReadbackRange(VAddr start, VAddr end);
    struct ReadbackTrace {
        VAddr address;
        u32 size;
        u32 changed_bytes;
        s64 recorded_ns;
        s64 completed_ns;
        std::string writer;
    };
    std::mutex readback_trace_mutex;
    std::deque<ReadbackTrace> readback_trace; // DIAG-072
    u64 readbacks_cpu_rewrote{};
    void NoteGpuWriteOverReadbackPages(VAddr address, u64 size);

public:
    /// FIX-073: a renderer write into guest memory (keeps readback page hashes current).
    void RefreshReadbackPageHashes(VAddr address, u64 size);

private:
    template <typename Keep, typename Drop>
    void SplitByWriter(VAddr start, VAddr end, Keep&& keep, Drop&& drop, bool keep_fresh = false);
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
