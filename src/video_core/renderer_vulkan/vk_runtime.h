// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class BlitHelper;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Scheduler;

class Runtime {
public:
    explicit Runtime(const Instance& instance, Scheduler& scheduler);
    ~Runtime() = default;

    const Instance& GetInstance() const {
        return instance;
    }

    StagingBufferPool& GetStagingPool() {
        return staging_pool;
    }

    void TickFrame();

    void CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                    std::span<const vk::BufferCopy> copies);

    void FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value);

    void InlineData(VideoCore::Buffer* dst, u64 offset, u32 value);

    /// FIX-019: writes up to 64 KB of dword-aligned data into a buffer in command order.
    void InlineData(const VideoCore::Buffer* dst, u64 offset, std::span<const u8> data);

    bool Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                 vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                 std::optional<VideoCore::SubresourceRange> subres_range = {});

    void UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                     std::span<const vk::BufferImageCopy> upload_copies,
                     bool preserve_buffer_coherence = false);
    void DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                       std::span<const vk::BufferImageCopy> download_copies);

    void CopyImage(VideoCore::Image* src, VideoCore::Image* dst);
    void CopySubrect(VideoCore::Image* src, VideoCore::Image* dst);
    void CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                             const VideoCore::Buffer* buffer, u64 offset,
                             std::optional<VideoCore::SubresourceRange> sub_range = {});
    void CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice);

    void CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst);

    void CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                          const VideoCore::SubresourceRange& sub_range);

    void ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                      const VideoCore::SubresourceRange& src_range,
                      const VideoCore::SubresourceRange& dst_range);
    void ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                    const vk::ClearValue& clear_value);

    void SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing = true);

    void AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size,
                      vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access);

    bool IsBufferAccessed(const VideoCore::Buffer* handle, u64 offset, u64 size,
                          bool check_read_access = false);

    void FlushBarriers();

    /// PERF-066 (SHADGT_HOIST_UPLOADS=1): whether buffer uploads go to the session's upload
    /// command buffer.
    bool HoistsUploads() const noexcept {
        return hoist_uploads;
    }
    /// PERF-066: a guest-memory upload recorded in the session's upload command buffer, ahead
    /// of the session's draws; recorded in place (as CopyBuffer) when a command already
    /// recorded in this session uses the bytes.
    void CopyBufferHoisted(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                           std::span<const vk::BufferCopy> copies);
    /// PERF-066: buffer ranges the draw just bound use in the session.
    void NoteSessionDraw(std::span<const std::tuple<vk::Buffer, u64, u64, bool>> buffers);

    /// DIAG-057 (SHADGT_DIAG_BATCH=1): how many consecutive draws could be recorded as one
    /// batch, with every GPU copy and layout change for them hoisted before them, before one
    /// touches a resource an earlier draw of the batch uses. Measures the parallel recorder's
    /// batches without changing anything.
    void DiagBatchDraw(std::span<const std::tuple<vk::Buffer, u64, u64, bool>> buffers,
                       std::span<const vk::Image> images);
    void DiagBatchCut(int reason);

private:
    void MakeCurrent(const VideoCore::Buffer* handle);

private:
    const Instance& instance;
    Scheduler& scheduler;
    std::unique_ptr<VideoCore::BlitHelper> blit_helper;
    StagingBufferPool staging_pool;
    struct BufferBarriers {
        const VideoCore::Buffer* handle;
        using AccessList = IntervalList<Interval>;
        AccessList read_ranges;
        AccessList write_ranges;
    };
    BufferBarriers* resource{};
    std::vector<BufferBarriers> resources;
    VideoCore::Image::Barriers image_barriers;
    vk::MemoryBarrier2 memory_barrier{};
    // DIAG-057
    void DiagBatchBufferWrite(vk::Buffer buffer, u64 offset, u64 size, int reason);
    void DiagBatchBufferRead(vk::Buffer buffer, u64 offset, u64 size, int reason);
    void DiagBatchImage(vk::Image image, int reason);
    struct DiagBatch {
        bool enabled{};
        std::vector<std::tuple<vk::Buffer, u64, u64, bool>> buffers; // used by the batch
        std::vector<vk::Image> images;
        u32 draws{};
        std::array<u64, 6> cuts{};      // by reason
        std::array<u64, 6> histogram{}; // batch sizes: 1, 2-4, 5-16, 17-64, 65-256, 257+
        u64 total_draws{};
        u64 batches{};
        std::chrono::steady_clock::time_point since{};
    } diag_batch;
    // PERF-066
    void NoteSessionUse(vk::Buffer buffer, u64 offset, u64 size);
    void ReportHoisting();
    bool hoist_uploads{};
    u64 hoist_session{};
    // Per buffer, the byte ranges commands of the current session use.
    std::unordered_map<VkBuffer, IntervalList<>> session_used;
    u64 hoisted_copies{};
    u64 hoist_cuts{};
    u64 hoist_draws{};
    std::chrono::steady_clock::time_point hoist_report{};
};

} // namespace Vulkan
