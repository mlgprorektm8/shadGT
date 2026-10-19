// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-100: native mode (RYGT_NATIVE=1, ryGT builds); see vk_rasterizer_native.cpp.

#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/regs_texture.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"

namespace AmdGpu {
union Regs;
}

namespace Shader {
struct Info;
}

namespace VideoCore {
class Sampler;
}

namespace Native {
class ImageEntry;
class ImageRegistry;
struct MadeImage;
namespace Rhi {
class Buffer;
class CommandList;
class Device;
class Fence;
} // namespace Rhi
} // namespace Native

namespace Vulkan {

class Rasterizer;
class GraphicsPipeline;
class ComputePipeline;

/// NAT-100: whether native mode was asked for (RYGT_NATIVE=1).
bool NativeModeRequested();

class NativeRenderer {
public:
    explicit NativeRenderer(Rasterizer& rasterizer);
    ~NativeRenderer();

    NativeRenderer(const NativeRenderer&) = delete;
    NativeRenderer& operator=(const NativeRenderer&) = delete;

    void Draw(bool is_indexed, u32 index_offset);
    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    /// WRITE_DATA in command order: later work on the GPU reads the new value. False when the
    /// address can't be written by the GPU.
    bool InlineWrite(VAddr address, std::span<const u8> data);
    /// Guest memory is unmapped: textures over it are uploaded again when next used.
    void OnUnmap(VAddr address, u64 size);
    /// NAT-105: the CPU wrote watched memory (a page fault on any thread): the textures there
    /// are uploaded again before their next use, and their pages are no longer watched.
    void OnCpuWrite(VAddr address, u64 size, u64 exact);
    u32 ReadGds(u32 gds_offset);

    /// Ends the command list recorded so far and runs it in the scheduler's current command
    /// buffer (called before anything else is recorded there or it is submitted).
    void Splice();

    /// The picture at a display buffer, for the presenter (none when nothing drew there).
    struct Display {
        vk::Image image;
        vk::ImageView view;
        u32 width;
        u32 height;
    };
    std::optional<Display> FindDisplay(const VideoCore::ImageInfo& info, vk::Format view_format);

private:
    enum class ViewUse : u8 { Sampled, Storage, Attachment };

    struct Chunk {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 retire_tick = 0;
    };

    /// What a draw renders into (the scheduler's RenderState, in the general layout).
    struct Targets {
        RenderState state{};
        std::array<const Native::ImageEntry*, 8> colors{};
        const Native::ImageEntry* depth = nullptr;
    };

    /// The bindings of one draw or dispatch.
    struct DrawBindings {
        Pipeline::DescriptorWrites writes;
        boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffers;
        boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> images;
        Shader::PushData push{};
        std::vector<const Native::ImageEntry*> sampled;
        std::vector<const Native::ImageEntry*> stored;
        bool writes_buffers = false;
        bool failed = false;
        void Clear();
    };

    /// The command list being recorded and its state.
    struct Stream {
        Native::Rhi::CommandList* list = nullptr;
        vk::CommandBuffer cmd{};
        bool rendering = false;
        RenderState render{};
        DynamicState dynamic{};
        vk::Pipeline pipeline{};
        std::vector<const Native::ImageEntry*> written; ///< since the last barrier
        bool buffers_written = false;                   ///< by shaders since the last barrier
        std::vector<Chunk> chunks;
        u64 chunk_used = 0;
        std::vector<vk::DescriptorPool> pools; ///< for pipelines without push descriptors
        std::vector<std::unique_ptr<Native::Rhi::Buffer>> keep; ///< alive until the list ran
        u32 commands = 0;
    };

    struct ViewRecord {
        VideoCore::ImageViewInfo info;
        ViewUse use;
        vk::UniqueImageView view;
    };

    Native::MadeImage MakeImage(const VideoCore::ImageInfo& info, bool render_target);
    vk::ImageView View(const Native::ImageEntry& entry, const VideoCore::ImageViewInfo& info,
                       ViewUse use);
    vk::Sampler SamplerFor(const AmdGpu::Sampler& sharp, AmdGpu::BorderColorBuffer border,
                           bool is_depth);

    Stream& Begin();
    void EndRendering();
    void BeginRendering(const RenderState& state);
    void GlobalBarrier();
    void PrepareImages();
    void ChunkFlushCheck();

    /// Upload memory of the stream's list: (buffer, offset, CPU pointer).
    std::tuple<vk::Buffer, u64, u8*> Allocate(u64 size, u64 align);
    Chunk TakeChunk(u64 size);
    vk::DescriptorPool TakeDescriptorPool();

    /// NAT-114: a range shaders write that no single imported view holds lives in a buffer of
    /// its own on the GPU (a region), filled from guest memory when made; every later binding
    /// over it uses the region, and CPU writes to its pages are copied in before the next use.
    struct Region {
        VAddr begin = 0;
        VAddr end = 0;
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        std::vector<VAddr> dirty_pages; ///< written by the CPU since the last use
    };
    std::map<VAddr, Region> regions; ///< by begin; guarded by mutex
    std::mutex region_dirty_mutex;   ///< pending_region_writes (CPU write faults: any thread)
    std::vector<std::pair<VAddr, VAddr>> pending_region_writes;
    std::atomic<bool> has_pending_region_writes{false};
    void SortRegionWrites();
    Region* RegionOver(VAddr address, u64 size);
    Region* MakeRegion(VAddr address, u64 size);
    void RefreshRegion(Region& region);
    /// NAT-107: a guest range gathered on the GPU from the imported views holding it, into a
    /// buffer of the list (none when part of it is not imported).
    std::optional<std::pair<vk::Buffer, u64>> GatherOnGpu(VAddr address, u64 size);
    /// A guest range as a buffer the GPU can bind: imported guest memory read and written in
    /// place, else (reads only) a copy in upload memory.
    std::optional<std::pair<vk::Buffer, u64>> GuestBuffer(VAddr address, u64 size, bool write);

    bool ResolveTargets(const AmdGpu::Regs& regs, const GraphicsPipeline& pipeline, Targets& out);
    void BindStages(std::span<const Shader::Info* const> stages, const AmdGpu::Regs& regs,
                    bool compute, DrawBindings& b);
    void BindBuffers(const Shader::Info& stage, const AmdGpu::Regs& regs, bool compute,
                     Shader::Backend::Bindings& binding, DrawBindings& b);
    void BindTextures(const Shader::Info& stage, const AmdGpu::Regs& regs,
                      Shader::Backend::Bindings& binding, DrawBindings& b);
    void PushBindings(const Pipeline& pipeline, vk::PipelineBindPoint bind_point,
                      DrawBindings& b);
    struct VertexIndexState;
    void ResolveVertexBuffers(const GraphicsPipeline& pipeline, const AmdGpu::Regs& regs,
                              VertexIndexState& v);
    bool ResolveIndices(const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset,
                        bool quad_triangles, VertexIndexState& v);
    void RecordVertexIndex(const VertexIndexState& v);
    bool NeedsBarrierFor(const DrawBindings& b) const;
    void NoteWrites(const Targets* targets, const DrawBindings& b);
    /// NAT-105: CPU writes to the pages of a range fault (once) from now on.
    void WatchPages(VAddr address, u64 size);
    /// NAT-112: small render targets drawn into are written back to guest memory (GT Sport's
    /// CPU reads its brightness chain there for auto exposure).
    void FlushWritebacks();
    std::vector<Native::ImageEntry*> writebacks;
    /// NAT-104: the picture was written now (in GPU order).
    void Written(const Native::ImageEntry* entry);
    /// NAT-104: copies the newer pictures of the render targets a texture is made of into it.
    void RefreshSources(Native::ImageEntry& texture);

    void ResolveDraw(const AmdGpu::Regs& regs);
    bool MetaClearDispatch(const Shader::Info& cs);
    void Report();

    Rasterizer& r;
    std::unique_ptr<Native::Rhi::Device> device;
    std::unique_ptr<Native::Rhi::Fence> timeline;
    std::unique_ptr<Native::ImageRegistry> registry;
    std::unique_ptr<Native::Rhi::Buffer> gds;
    std::unique_ptr<Native::Rhi::Buffer> lds;

    std::recursive_mutex mutex; ///< v0: the recorder thread records; flips may come from others
    Stream stream;
    DrawBindings bindings;
    u64 frame = 0;

    std::mutex chunk_mutex;
    std::vector<Chunk> free_chunks;
    std::deque<Chunk> retiring_chunks;
    std::vector<vk::DescriptorPool> free_pools;
    std::deque<std::pair<u64, vk::DescriptorPool>> retiring_pools;

    std::mutex sampler_mutex;
    std::unordered_map<u64, std::unique_ptr<VideoCore::Sampler>> samplers;

    std::mutex views_mutex;
    std::unordered_map<const Native::ImageEntry*, std::vector<ViewRecord>> views;
    /// NAT-108: views of evicted images, destroyed once the GPU passed the tick.
    std::deque<std::pair<u64, vk::UniqueImageView>> retired_views;
    void EvictImages();
    struct ReportBase {
        u64 uploads = 0, upload_bytes = 0, evicted = 0;
        std::array<u64, 3> stale{};
    } report_base;

    /// NAT-104: the write clock, and which version of its sources each subresource of a
    /// texture holds (by mip << 16 | layer).
    u64 write_clock = 0;
    std::unordered_map<const Native::ImageEntry*, std::unordered_map<u32, u64>> held_versions;

    /// NAT-105: the pages native textures watch (each watched once in the page manager, which
    /// counts at most 255 watchers a page; many textures share pages).
    std::mutex watch_mutex;
    std::unordered_set<VAddr> watched_pages;

    /// CMASK / HTILE addresses the game cleared: the next pass into their target clears it.
    std::unordered_set<VAddr> cleared_meta;
    /// Color targets by their CMASK address, depth targets by their HTILE address (so a clear
    /// of the metadata can be told from a clear of a picture).
    std::unordered_set<VAddr> known_meta;

    struct Stats {
        u64 draws = 0, dispatches = 0, skipped = 0, no_pipeline = 0, passes = 0, barriers = 0,
            splices = 0, copied_reads = 0, copied_bytes = 0, unbound = 0, null_images = 0,
            fills = 0, copies = 0, meta_clears = 0, inline_writes = 0, source_copies = 0,
            draw_ns = 0, dispatch_ns = 0, gathers = 0, gathered_bytes = 0, writebacks = 0,
            region_binds = 0, regions_made = 0, region_refreshes = 0;
        /// Images bound empty: no address or format, not mapped for the GPU, format the
        /// device lacks, bad geometry, outside guest memory, FIX-081, no image made.
        std::array<u64, 7> null_reasons{};
        /// NAT-110: time in the draw's phases: pipeline, targets, buffers, textures, sources,
        /// vertices, prepare (inits and uploads), record.
        std::array<u64, 8> phase_ns{};
        std::atomic<u64> cpu_writes{0}; // NAT-105 (faults come from any thread)
    } stats;
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();
};

} // namespace Vulkan
