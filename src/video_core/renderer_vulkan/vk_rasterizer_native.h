// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-100: native mode (RYGT_NATIVE=1, ryGT builds); see vk_rasterizer_native.cpp.

#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <filesystem>
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

    /// NAT-116: saves the render targets noted while the rasterizer collected a frame.
    void DumpFrame(const std::filesystem::path& dir);

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
        Pipeline::WriteSets write_sets; ///< SHD-002: the set of each write
        boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffers;
        boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> images;
        Shader::PushData push{};
        std::vector<const Native::ImageEntry*> sampled;
        std::vector<const Native::ImageEntry*> stored;
        /// NAT-119: guest memory the draw or dispatch writes through storage buffers.
        boost::container::small_vector<std::pair<VAddr, u64>, 8> gpu_writes;
        /// NAT-120: the dispatch is GT Sport's fill shader: what it fills, with what.
        std::optional<std::array<u32, 2>> fill_pattern;
        VAddr fill_address = 0;
        u64 fill_size = 0;
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
        /// NAT-140: attachments drawn since the last pass or global barrier, and pictures
        /// sampled since the last global barrier: what a new pass has to wait for.
        std::vector<const Native::ImageEntry*> attach_dirty;
        std::vector<const Native::ImageEntry*> read;
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
    /// NAT-140: attachment writes before it finish before attachment access after it (a pass
    /// drawing into what an earlier pass drew), without waiting for anything else.
    void PassBarrier();
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
    /// NAT-150: copies in video memory of guest memory the GPU reads as vertices and indices
    /// and the CPU leaves alone (meshes), instead of reading it over PCIe at every draw. A range
    /// bound in a second frame is copied at the start of the next list; CPU writes to its pages
    /// are copied in before its next use; 64 KiB granules the CPU writes in two frames (rings of
    /// per-frame data) are read in place again for a while. Mirrors never overlap regions or
    /// each other. RYGT_NATIVE_MIRRORS=0 turns them off; RYGT_NATIVE_MIRROR_MB is the budget.
    struct Mirror {
        VAddr begin = 0;
        VAddr end = 0;
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        std::vector<VAddr> dirty_pages; ///< written by the CPU (or written back) since copied
        u64 used_frame = 0;
    };
    struct MirrorCandidate {
        VAddr end = 0;
        u64 frame = 0; ///< when last considered
    };
    std::map<VAddr, Mirror> mirrors; ///< by begin; guarded by mutex
    u64 mirror_bytes = 0;
    std::unordered_map<VAddr, MirrorCandidate> mirror_candidates; ///< by page of the range start
    std::vector<std::pair<VAddr, VAddr>> pending_mirrors;          ///< made at a list's start
    std::unordered_map<u64, u64> mirror_writes;    ///< granule -> frame the CPU wrote a mirror there
    std::unordered_map<u64, u64> dynamic_granules; ///< granule -> frame it was found CPU-written
    /// The first mirror that may overlap memory from `begin` on.
    std::map<VAddr, Mirror>::iterator FirstMirror(VAddr begin);
    /// The mirror holding all of the range, if any.
    Mirror* MirrorHolding(VAddr address, u64 size);
    bool IsDynamic(VAddr begin, VAddr end);
    std::optional<std::pair<vk::Buffer, u64>> MirrorBuffer(VAddr address, u64 size);
    void RefreshMirror(Mirror& mirror);
    void MakePendingMirrors();
    void DropMirrors(VAddr address, u64 size);
    /// NAT-150: what the last GuestBuffer bound (for the NAT-142 counters).
    enum class Bound : u8 { InPlace, Region, Mirror };
    Bound last_bound = Bound::InPlace;
    /// OPT-003: what a guest range resolved to the last time (a region, a mirror, or imported
    /// guest memory read in place), valid while no region or mirror was made or dropped and no
    /// memory unmapped since (buffer_generation); a region or mirror with pages to copy in takes
    /// the full lookup.
    struct BufferMemo {
        VAddr address = 0;
        u64 size = 0;
        u32 flags = 0;
        u64 generation = ~u64{0};
        vk::Buffer buffer{};
        u64 offset = 0;
        Region* region = nullptr;
        Mirror* mirror = nullptr;
        Bound bound = Bound::InPlace;
    };
    std::array<BufferMemo, 2048> buffer_memos{};
    u64 buffer_generation = 0;
    /// OPT-003: what the last lookup can be remembered as (set by GuestBufferLookup).
    enum class MemoKind : u8 { None, Region, Mirror, InPlace };
    MemoKind memo_kind = MemoKind::None;
    Region* memo_region = nullptr;
    Mirror* memo_mirror = nullptr;
    std::optional<std::pair<vk::Buffer, u64>> GuestBufferLookup(VAddr address, u64 size, bool write,
                                                                bool mirror);
    /// OPT-002: what a T# resolved to the last time (picture and views), by the T# and how the
    /// shader reads it, valid while the registry made or dropped no picture and no memory was
    /// unmapped since.
    struct TextureMemo {
        std::array<u64, 4> sharp{};
        u32 flags = 0;
        u64 generation = ~u64{0};
        Native::ImageEntry* entry = nullptr;
        std::array<vk::ImageView, 4> views{};
    };
    std::array<TextureMemo, 4096> texture_memos{};
    u64 unmap_generation = 0;
    /// NAT-107: a guest range gathered on the GPU from the imported views holding it, into a
    /// buffer of the list (none when part of it is not imported).
    std::optional<std::pair<vk::Buffer, u64>> GatherOnGpu(VAddr address, u64 size);
    /// A guest range as a buffer the GPU can bind: imported guest memory read and written in
    /// place, else (reads only) a copy in upload memory.
    /// `mirror`: read-only data worth a copy in video memory (NAT-150: vertices, indices).
    std::optional<std::pair<vk::Buffer, u64>> GuestBuffer(VAddr address, u64 size, bool write,
                                                          bool mirror = false);

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
    std::vector<const Native::ImageEntry*> dump_entries; // NAT-116
    /// NAT-123: GPU-written images the dumped frame sampled, by texture key.
    std::vector<std::pair<std::string, const Native::ImageEntry*>> dump_textures;
    /// NAT-120: depth images by the address of their stencil plane (fills clear it there).
    std::unordered_map<VAddr, boost::container::small_vector<Native::ImageEntry*, 2>>
        stencil_owners;
    /// NAT-120: GPU work wrote guest memory (targets there get a pending rewrite).
    void GpuWrote(VAddr address, u64 size, const std::array<u32, 2>* pattern, bool exact);
    /// NAT-120: a target's pending rewrite, before it is used: a clear, or an upload.
    void ApplyRewrite(Native::ImageEntry& entry);
    /// NAT-132: fills the picture with a 32-bit pattern on the GPU (rows != 0: only those rows
    /// of the first mip and layer); false when it can't (compressed, multisampled).
    bool FillImage(Native::ImageEntry& entry, u32 lo, u32 hi, u32 rows);
    /// NAT-134: a device buffer holding the pattern (lo, hi, lo, hi...), at least `bytes` long.
    const Native::Rhi::Buffer* PatternBuffer(u32 lo, u32 hi, u64 bytes);
    std::unordered_map<u64, std::unique_ptr<Native::Rhi::Buffer>> fill_patterns;
    std::vector<std::unique_ptr<Native::Rhi::Buffer>> retired_patterns;
    /// NAT-120: GT Sport's fill shader (cs 0x8aa6e00c): its pattern and range, from the
    /// arguments it reads from memory.
    void ReadFillShader(const Shader::Info& cs, DrawBindings& b);
    /// NAT-104: the picture was written now (in GPU order).
    /// `at` != 0: the version is that clock value (NAT-148: a rewrite applied late).
    void Written(const Native::ImageEntry* entry, u64 at = 0);
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
    u64 timing_tick = 0; ///< OPT-001
    /// NAT-142: guest memory the GPU reads or writes in place (64 KB pages, every 8th frame):
    /// vertex buffers, indices, read-only storage through the page table, other storage
    /// buffers read, storage buffers written; NAT-150: then what is bound from video memory
    /// instead (mirrors, regions).
    struct InPlace {
        std::array<std::unordered_set<u64>, 7> pages;
        std::array<u64, 7> total{};
        u64 frames = 0;
        void Note(u32 kind, VAddr address, u64 size) {
            if (size == 0) {
                return;
            }
            for (u64 page = address >> 16; page <= (address + size - 1) >> 16; ++page) {
                pages[kind].insert(page);
            }
        }
    } in_place;
    bool InPlaceSampled() const {
        return (frame & 7) == 0;
    }
    /// NAT-150: the counter for a binding of that kind, by what GuestBuffer bound.
    u32 BoundKind(u32 kind) const {
        return last_bound == Bound::Mirror ? 5 : (last_bound == Bound::Region ? 6 : kind);
    }

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
        u64 rewrite_clears = 0, rewrite_uploads = 0, fill_shaders = 0; ///< NAT-120
        u64 rewrite_partial = 0; ///< NAT-124
        u64 pass_barriers = 0, passes_unblocked = 0; ///< NAT-140
        u64 region_refresh_pages = 0, region_refresh_ns = 0; ///< NAT-145
        u64 guest_buffer_calls = 0, guest_buffer_ns = 0;     ///< NAT-145
        u64 rewrite_fills = 0;   ///< NAT-132
        u64 timed_draws = 0;     ///< OPT-001
        u64 texture_memo_hits = 0, texture_lookups = 0; ///< OPT-002
        u64 buffer_memo_hits = 0;                        ///< OPT-003
        u64 mirror_binds = 0, mirrors_made = 0, mirror_made_bytes = 0, mirrors_dropped = 0,
            mirrors_evicted = 0, mirror_over_budget = 0, mirror_refresh_pages = 0; ///< NAT-150
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
