// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include <array>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    /// exact_write_size: bytes a faulting CPU store writes from addr, when known (PERF-011b).
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false,
                          u64 exact_write_size = 0);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    [[nodiscard]] bool HasPendingAsyncReadbacks() const {
        return buffer_cache.HasPendingAsyncReadbacks();
    }
    /// DIAG-016: where a synchronous GPU drain came from.
    enum class DrainSource : u32 {
        Submit,
        GfxEos,
        GfxEop,
        GfxWriteData,
        AscWriteData,
        AscReleaseMem,
        GdsStore,
        Count,
    };
    void OnFence(DrainSource source);
    void FinishForGds();
    /// Defers a graphics fence signal until the GPU completes when readbacks are pending or
    /// another fence is already deferred. Returns false if the caller should signal now.
    /// value: the bytes the fence will write, when known when it is processed (PERF-014).
    bool DeferFenceSignal(VAddr address, Common::UniqueFunction<void>&& signal,
                          bool compute_queue = false, std::span<const u8> value = {});
    /// FIX-019: a WRITE_DATA packet whose guest write was deferred also writes the GPU copy of
    /// the memory now, in command order, as hardware does.
    void InlineDeferredWrite(VAddr address, std::span<const u8> data);
    /// PERF-014: the dword at address once all deferred fence writes that cover it land.
    std::optional<u32> PendingFenceDword(VAddr address);
    [[nodiscard]] bool HasReadbackFences() const {
        return readback_fences.load() != 0;
    }
    /// Submits recorded GPU work if deferred fences are outstanding (before blocking waits).
    void FlushForDeferredFences();
    void SubmitChunkIfNeeded();
    void RecordDeferredFenceLatency(std::chrono::steady_clock::time_point deferred_at);

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed,
                            bool quad_triangles = false) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed, bool quad_triangles) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    void LogInvalidTextureContext(const Shader::Info& stage, u32 sharp_offset);
    /// Copies flattened constants whose guest source is GPU-modified from the buffer cache into
    /// the uploaded flat buffer, so the shader sees them in GPU execution order.
    void RefreshGpuWrittenConstants(const Shader::Info& stage, const VideoCore::Buffer& flat_buffer,
                                    u64 flat_offset);
    void RebindTextures(bool is_compute, u32 num_color_targets);
    void FinalizeTextureLayouts(RenderState* render_state = nullptr);
    bool BindResources(const Pipeline* pipeline);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0);
    bool CanDrawQuadListAsTriangles(bool is_indexed, u32 index_offset);
    u32 BindQuadListIndices(bool is_indexed, u32 index_offset);

    void ResetBindings(bool is_compute);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundTexture {
        u32 descriptor_index;
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
        vk::Image backing_image;
    };
    boost::container::static_vector<BoundTexture, Shader::NUM_IMAGES> bound_textures;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        bool is_written;
        VAddr guest_address{};
    };
    boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS> bound_buffers;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    vk::ImageAspectFlags attachment_feedback_loop{};
    bool needs_barrier{};
    u32 invalid_texture_context_count{};
    std::unordered_set<u64> logged_gpu_constant_shaders;
    std::atomic<u32> deferred_fences{};
    /// PERF-014b: deferred fences that will also bring GPU data back to guest memory.
    std::atomic<u32> readback_fences{};
    u32 draws_since_submit{};
    struct DrainStats {
        std::chrono::steady_clock::time_point window_start{};
        std::array<u32, u32(DrainSource::Count)> count{};
        std::array<u64, u32(DrainSource::Count)> total_us{};
    } drain_stats;
    void ProcessDownloadsTimed(DrainSource source);
    void RecordDrain(DrainSource source, std::chrono::steady_clock::time_point start);
    std::mutex deferred_fences_mutex;
    struct DeferredFenceStats {
        std::chrono::steady_clock::time_point window_start{};
        u64 window_frame{};
        u64 count{};
        u64 total_us{};
        u64 max_us{};
    } deferred_fence_stats;
    std::unordered_map<VAddr, u32> deferred_fence_addresses;
    std::unordered_map<VAddr, std::vector<u8>> pending_fence_bytes;
    u64 gpu_constant_refresh_count{};
    u64 next_gpu_constant_refresh_report{1000};
};

} // namespace Vulkan
