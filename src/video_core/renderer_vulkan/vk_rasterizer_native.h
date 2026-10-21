// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-100: native mode (RYGT_NATIVE=1, ryGT builds); see vk_rasterizer_native.cpp.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
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

    /// OPT-010 (RYGT_NATIVE_PREPARE=1): whether game job threads prepare draws.
    static bool PrepareEnabled();
    /// OPT-012: on the command thread, whether a job thread prepared the draw at the packet
    /// (with its stages): then the command thread doesn't select its pipeline.
    bool HasPrepared(const void* packet);
    /// OPT-010: on a game job thread whose walk reached a draw (`regs` its state there) and
    /// found its built pipeline, the stage infos holding the draw's user data: the draw's
    /// bindings worked out ahead for the recorder (see PreparedDraw).
    void PrepareJobDraw(const void* packet, u32 epoch, const AmdGpu::Regs& regs,
                        const GraphicsPipeline& pipeline,
                        std::span<const Shader::Info* const> stages, bool is_indexed,
                        u32 index_offset);

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
        /// OPT-005: the vertex input layout, vertex buffers and index buffer last recorded in
        /// this list (the same ones are not recorded again).
        bool vertex_state_known = false;
        boost::container::static_vector<vk::VertexInputBindingDescription2EXT, 32> input_bindings;
        boost::container::static_vector<vk::VertexInputAttributeDescription2EXT, 32> attributes;
        boost::container::static_vector<vk::Buffer, 32> vertex_buffers;
        boost::container::static_vector<vk::DeviceSize, 32> vertex_offsets;
        boost::container::static_vector<vk::DeviceSize, 32> vertex_sizes;
        boost::container::static_vector<vk::DeviceSize, 32> vertex_strides;
        vk::Buffer index_buffer{};
        u64 index_offset = 0;
        vk::IndexType index_type = vk::IndexType::eUint32;
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
    /// FIX-173: region pages as guest memory held them before the CPU's first write since they
    /// were last copied in (taken at the write fault; guarded by region_dirty_mutex).
    std::unordered_map<VAddr, std::unique_ptr<std::array<u8, 4096>>> page_snapshots;
    std::atomic<u32> logged_partial{};
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
        std::vector<u64> hashes; ///< FIX-176: guest memory per 64 KB granule when last verified
    };
    static constexpr u64 MirrorGranule = u64{64} * 1024; ///< FIX-176
    VAddr mirror_verify_cursor = 0;            ///< FIX-176
    static u64 MirrorVerifyBudget();
    static bool MirrorKind(char kind); ///< FIX-177
    /// FIX-179 (diagnosis): a vertex range bound from a mirror, copied out with guest memory.
    struct MirrorCheck {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u64 size;
        VAddr base;
        u64 frame;
        VAddr mirror_begin;
        u64 mirror_size;
        u64 mirror_used;
        u64 dirty;
    };
    std::vector<MirrorCheck> mirror_checks;
    u64 mirror_check_counter = 0;
    void CheckMirrorBind(Mirror& mirror, VAddr base, u64 size);
    /// FIX-181 (diagnosis): how far a draw reads past its vertex V#s and their mirrors.
    void CheckVertexReach(const GraphicsPipeline& pipeline, const AmdGpu::Regs& regs,
                          bool is_indexed, u32 index_offset);
    u64 reach_check_counter = 0;
    void VerifyMirrors();
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
    std::array<BufferMemo, 16384> buffer_memos{}; ///< OPT-006: 2048 thrashed
    u64 buffer_generation = 0;
    /// OPT-003: what the last lookup can be remembered as (set by GuestBufferLookup).
    enum class MemoKind : u8 { None, Region, Mirror, InPlace };
    MemoKind memo_kind = MemoKind::None;
    Region* memo_region = nullptr;
    Mirror* memo_mirror = nullptr;
    /// OPT-004: the mirror the last GuestBuffer bound from (none when it bound anything else).
    Mirror* last_mirror = nullptr;
    /// FIX-170 (diagnosis): RYGT_NATIVE_SNAPSHOT_BELOW, 0 when off.
    static u64 SnapshotBelow();
    /// A copy of the range as it is now (NAT-106 zeros where the CPU can't read).
    std::optional<std::pair<vk::Buffer, u64>> SnapshotRange(VAddr address, u64 size);
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
    std::array<TextureMemo, 16384> texture_memos{}; ///< OPT-006
    std::atomic<u64> unmap_generation{0}; ///< (OPT-010: read by job threads)
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
    /// A draw's vertex and index buffers: looked up before its pass starts (a lookup may gather
    /// on the GPU, outside rendering) and bound inside it.
    struct VertexIndexState {
        boost::container::static_vector<vk::VertexInputAttributeDescription2EXT, 32> attributes;
        boost::container::static_vector<vk::VertexInputBindingDescription2EXT, 32> input_bindings;
        boost::container::static_vector<vk::Buffer, 32> buffers;
        boost::container::static_vector<vk::DeviceSize, 32> offsets;
        boost::container::static_vector<vk::DeviceSize, 32> sizes;
        boost::container::static_vector<vk::DeviceSize, 32> strides;
        vk::Buffer index_buffer{};
        u64 index_offset = 0;
        vk::IndexType index_type = vk::IndexType::eUint32;
        u32 index_count = 0;
    };
    void ResolveVertexBuffers(const GraphicsPipeline& pipeline, const AmdGpu::Regs& regs,
                              VertexIndexState& v);
    bool ResolveIndices(const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset,
                        bool quad_triangles, VertexIndexState& v);
    void RecordVertexIndex(const VertexIndexState& v);
    bool NeedsBarrierFor(const DrawBindings& b) const;
    void NoteWrites(const Targets* targets, const DrawBindings& b);
    /// NAT-105: CPU writes to the pages of a range fault (once) from now on.
    void WatchPages(VAddr address, u64 size);
    /// NAT-105, FIX-153: the watched pages of [begin, end) (page aligned) are watched no more.
    void UnwatchPages(VAddr begin, VAddr end);
    /// NAT-112: small render targets drawn into are written back to guest memory (GT Sport's
    /// CPU reads its brightness chain there for auto exposure).
    void FlushWritebacks();
    /// FIX-156 (diagnosis): reads back the pictures at RYGT_NATIVE_WATCH_TEXTURE now and then.
    void ProbeWatchedPictures();
    /// FIX-157 (diagnosis): a T# at a watched address and the picture it resolved to.
    void WatchBind(const Native::ImageEntry& entry, const AmdGpu::Image& tsharp, bool storage,
                   bool memo, u64 program);
    struct Probe {
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 tick;
        u64 size;
        u64 frame;
        u64 version;
        u32 width, height;
        bool target;
        VAddr address;
        bool packed_r11g11b10 = false; // FIX-171
        u32 mips = 0, layers = 0;      // FIX-171: an untiled copy of every mip and layer
    };
    std::vector<Probe> probes;
    /// FIX-175 (diagnosis): rectangles of presented pictures on their way back from the GPU.
    std::vector<Probe> display_probes;
    std::array<double, 3> display_last_part{}; ///< FIX-182
    std::vector<u8> display_last_bytes;        ///< FIX-182
    u32 display_glitch_dumps = 0;              ///< FIX-182
    void ProbeDisplay(const Native::ImageEntry& entry);
    /// FIX-157: bytes of pictures kept past their idle time.
    u64 kept_bytes = 0;
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

    /// OPT-010: a draw's bindings worked out on the game job thread that recorded it, before
    /// the command processor reaches it: BindStages' lookups (buffers bound from regions or
    /// guest memory, textures from existing pictures and their views, samplers) with nothing
    /// done that must happen in GPU order (no picture or region made, nothing copied, uploaded
    /// or rewritten; upload memory is filled later). The recorder takes it by the draw packet's
    /// address and uses it when it was made for the same pipeline and user data and no
    /// picture, region or mirror over what it binds was made or dropped since (and no memory
    /// unmapped); then it does the ordered part (pending rewrites, copies in, upload memory).
    struct PreparedDraw {
        const void* packet = nullptr;
        u32 epoch = 0;
        const GraphicsPipeline* pipeline = nullptr;
        std::array<u64, Shader::MaxStageTypes> stage_hashes{};
        u64 image_generation = 0;
        u64 buffer_seq = 0;
        u64 unmap_gen = 0;
        DrawBindings bindings;
        boost::container::small_vector<Region*, 4> regions;
        std::vector<std::pair<VAddr, VAddr>> image_ranges;
        std::vector<std::pair<VAddr, VAddr>> buffer_ranges;
        /// Upload memory (flattened user data, clip planes, D3 tables): bytes copied at commit
        /// into the list's upload memory, and the buffer info that binds them.
        struct Upload {
            u32 info = 0;
            u32 offset = 0;
            u32 size = 0;
            s32 flat_stage = -1; ///< >= 0: that stage's flattened user data, the recorder's
        };
        std::vector<u8> upload_bytes;
        boost::container::small_vector<Upload, 4> uploads;
        u32 null_images = 0;
        /// OPT-011: vertex inputs and buffers and the index buffer (not for quad lists), as
        /// ResolveVertexBuffers and ResolveIndices work them out; the V#s they came from (a
        /// hash), the mirrors bound (copied in at commit) and the ranges read in place (mirror
        /// candidates, noted at commit).
        bool has_vertices = false;
        std::unique_ptr<VertexIndexState> vertex_index;
        u64 vertex_hash = 0;
        bool indexed = false;
        u32 index_offset = 0;
        VAddr index_address = 0;
        boost::container::small_vector<Mirror*, 4> mirrors;
        boost::container::small_vector<std::pair<VAddr, u64>, 4> in_place;
        /// OPT-012: the stages as the job thread matched them (what the command thread's
        /// selection would copy): the recorder installs them when the command thread didn't
        /// select (HasPrepared).
        struct StageCopy {
            bool present = false;
            VAddr pgm_base = 0;
            u32 num_user_data = 0;
            std::array<u32, 16> user_data{};
            std::vector<u32> flattened;
        };
        std::array<StageCopy, Shader::MaxStageTypes> stage_copies;
        void Reset();
    };
    /// OPT-010: the guest ranges whose region or mirror was made or dropped, by sequence number
    /// (written and checked by the recorder; job threads read the sequence before looking up).
    class ChangeLog {
    public:
        void Note(VAddr begin, VAddr end);
        /// Whether none of the ranges changed after `since` (false when the log no longer
        /// reaches back that far).
        bool Untouched(u64 since, std::span<const std::pair<VAddr, VAddr>> ranges) const;
        u64 Seq() const {
            return seq.load(std::memory_order_acquire);
        }

    private:
        struct Entry {
            u64 seq = 0;
            VAddr begin = 0;
            VAddr end = 0;
        };
        std::array<Entry, 1024> entries{};
        std::atomic<u64> seq{0};
    };
    ChangeLog buffer_log;
    /// OPT-010: regions and mirrors are added and removed under it (exclusive); job threads
    /// look regions up under it (shared).
    std::shared_mutex buffer_map_mutex;
    struct PreparedShard {
        std::mutex mutex;
        std::unordered_map<const void*, PreparedDraw*> draws;
    };
    std::array<PreparedShard, 64> prepared_shards;
    PreparedShard& ShardFor(const void* packet) {
        return prepared_shards[(reinterpret_cast<u64>(packet) * 0x9E3779B97F4A7C15ULL) >> 58];
    }
    std::mutex prepared_pool_mutex;
    std::vector<std::unique_ptr<PreparedDraw>> prepared_all;
    std::vector<PreparedDraw*> prepared_free;
    PreparedDraw* NewPrepared();
    void FreePrepared(PreparedDraw* draw);
    /// The prepared draw for the packet, taken out of the table (none if there is none).
    PreparedDraw* TakePrepared(const void* packet);
    bool PrepareBuffers(const Shader::Info& stage, const AmdGpu::Regs& regs,
                        Shader::Backend::Bindings& binding, PreparedDraw& p);
    bool PrepareTextures(const Shader::Info& stage, const AmdGpu::Regs& regs,
                         Shader::Backend::Bindings& binding, PreparedDraw& p);
    bool PrepareVertices(PreparedDraw& p, const AmdGpu::Regs& regs,
                         const GraphicsPipeline& pipeline, const Shader::Info& vs,
                         bool is_indexed, u32 index_offset);
    /// A vertex or index range from a region, a mirror or guest memory read in place (none
    /// when it would need ordered work).
    std::optional<std::pair<vk::Buffer, u64>> JobVertexBuffer(VAddr address, u64 size,
                                                              PreparedDraw& p,
                                                              Mirror** mirror_out);
    bool CommitPreparedVertices(PreparedDraw& p, const GraphicsPipeline& pipeline,
                                const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset,
                                bool quad_triangles, bool trusted);
    static u64 VertexHash(const GraphicsPipeline& pipeline, const Shader::Info& vs,
                          u32 step_rate_0, u32 step_rate_1);
    /// GuestBufferLookup's lookups without its side effects: none when it would make a region,
    /// gather or copy.
    std::optional<std::pair<vk::Buffer, u64>> JobGuestBuffer(VAddr address, u64 size, bool write,
                                                             PreparedDraw& p);
    /// Recorder: whether the prepared draw holds for the draw being recorded (its ordered part
    /// is then done).
    bool CommitPrepared(PreparedDraw& p, const GraphicsPipeline& pipeline,
                        const AmdGpu::Regs& regs, bool trusted);
    /// OPT-012: installs the prepared draw's stages for its pipeline (nullptr if they don't
    /// fit it).
    const GraphicsPipeline* InstallPrepared(PreparedDraw& p);
    static u64 StageHash(const Shader::Info& info);
    std::atomic<u64> shared_frame{0}; ///< OPT-010: `frame`, for job threads
    std::atomic<u32> recorder_epoch{0};
    struct PrepareStats {
        std::atomic<u64> made{0}, failed{0}, replaced{0}, stale{0}, ns{0};
    } prepare_stats;

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
        u64 region_partial_pages = 0, region_partial_bytes = 0; ///< FIX-173
        u64 mirror_untrapped = 0;                               ///< FIX-176
        u64 mirror_checks = 0, mirror_check_differ = 0;         ///< FIX-179
        u64 reach_checks = 0, reach_past_sharp = 0, reach_past_mirror = 0; ///< FIX-181
        u64 guest_buffer_calls = 0, guest_buffer_ns = 0;     ///< NAT-145
        u64 rewrite_fills = 0;   ///< NAT-132
        u64 timed_draws = 0;     ///< OPT-001
        u64 vertex_near = 0;     ///< OPT-004
        u64 vertex_skipped = 0;  ///< OPT-005
        /// OPT-010: draws bound from a prepared draw, without one, and prepared draws not used:
        /// another pipeline, other user data, memory unmapped, pictures or buffers changed.
        u64 prepared_used = 0, prepared_missing = 0;
        std::array<u64, 5> prepared_rejected{};
        u64 prepared_vertices = 0, prepared_vertices_rejected = 0; ///< OPT-011
        u64 prepared_installed = 0; ///< OPT-012: pipelines from a job thread's selection
        /// OPT-011: the record phase's parts: barrier choice, pass begin, dynamic state, vertex
        /// and index binds, pipeline bind, descriptors and push constants, the draw, write notes.
        std::array<u64, 8> record_ns{};
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
