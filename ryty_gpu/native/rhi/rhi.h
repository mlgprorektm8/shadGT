// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-010: the native renderer's backend interface (RHI). The renderer records through it and
// never touches Vulkan or Direct3D 12 itself, so a second backend is a second implementation of
// these classes. It is shaped by what GT Sport's draws need:
// - command lists are recorded by many threads at once (one per render job) and submitted in
//   the GPU order the game chose;
// - resources are bound per draw: one set of descriptors pushed with the draw (Vulkan push
//   descriptors; D3D12 a descriptor table copied into a ring) plus push constants;
// - images stay in one general layout; synchronisation is global memory barriers placed where
//   the game flushes caches, as on the PS4;
// - guest memory can be imported as a buffer, so shaders read the game's buffers in place.

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <span>
#include <string_view>

#include "common/types.h"

namespace Native::Rhi {

enum class BackendKind : u8 { Vulkan, D3D12 };

/// Formats use Vulkan's numbering (VkFormat), which the GCN translation already produces; the
/// D3D12 backend maps them to DXGI formats.
enum class Format : u32 { Undefined = 0 };
constexpr Format FormatFromVk(u32 vk_format) {
    return static_cast<Format>(vk_format);
}

enum class MemoryKind : u8 {
    DeviceLocal, ///< GPU-only
    Upload,      ///< CPU-written each frame, persistently mapped
    Readback,    ///< GPU-written, read by the CPU
};

enum BufferUsage : u32 {
    BufferUsageNone = 0,
    BufferUsageVertex = 1 << 0,
    BufferUsageIndex = 1 << 1,
    BufferUsageUniform = 1 << 2,
    BufferUsageStorage = 1 << 3,
    BufferUsageTexel = 1 << 4,      ///< uniform/storage texel buffers
    BufferUsageIndirect = 1 << 5,
    BufferUsageTransferSrc = 1 << 6,
    BufferUsageTransferDst = 1 << 7,
    BufferUsageDeviceAddress = 1 << 8,
};

struct BufferDesc {
    u64 size = 0;
    u32 usage = BufferUsageNone;
    MemoryKind memory = MemoryKind::DeviceLocal;
    std::string_view name;
};

enum class ImageDim : u8 { D1, D2, D3 };

enum ImageUsage : u32 {
    ImageUsageNone = 0,
    ImageUsageSampled = 1 << 0,
    ImageUsageStorage = 1 << 1,
    ImageUsageColorTarget = 1 << 2,
    ImageUsageDepthTarget = 1 << 3,
    ImageUsageTransferSrc = 1 << 4,
    ImageUsageTransferDst = 1 << 5,
    ImageUsageCube = 1 << 6,          ///< array layers can be viewed as cube faces
    ImageUsageMutableFormat = 1 << 7, ///< views may use other formats of the same size
};

struct ImageDesc {
    ImageDim dim = ImageDim::D2;
    Format format = Format::Undefined;
    u32 width = 1;
    u32 height = 1;
    u32 depth = 1;
    u32 layers = 1;
    u32 mips = 1;
    u32 samples = 1;
    u32 usage = ImageUsageNone;
    std::string_view name;
};

enum class ViewDim : u8 { D1, D1Array, D2, D2Array, D3, Cube, CubeArray };
enum class Swizzle : u8 { Identity, Zero, One, R, G, B, A };
enum ImageAspect : u8 { AspectColor = 1, AspectDepth = 2, AspectStencil = 4 };

struct ImageViewDesc {
    ViewDim dim = ViewDim::D2;
    Format format = Format::Undefined; ///< Undefined: the image's format
    u32 base_mip = 0;
    u32 mips = 1;
    u32 base_layer = 0;
    u32 layers = 1;
    u8 aspect = AspectColor;
    std::array<Swizzle, 4> swizzle{};
    bool storage = false; ///< for storage-image access instead of sampling
};

enum class Filter : u8 { Nearest, Linear };
enum class AddressMode : u8 { Repeat, MirroredRepeat, ClampToEdge, ClampToBorder, MirrorClampToEdge };
enum class CompareOp : u8 { Never, Less, Equal, LessOrEqual, Greater, NotEqual, GreaterOrEqual, Always };
enum class BorderColor : u8 { TransparentBlack, OpaqueBlack, OpaqueWhite };

struct SamplerDesc {
    Filter min_filter = Filter::Linear;
    Filter mag_filter = Filter::Linear;
    Filter mip_filter = Filter::Linear;
    std::array<AddressMode, 3> address{AddressMode::Repeat, AddressMode::Repeat, AddressMode::Repeat};
    float mip_lod_bias = 0.0f;
    float min_lod = 0.0f;
    float max_lod = 1000.0f;
    float max_anisotropy = 1.0f; ///< 1: off
    bool compare = false;
    CompareOp compare_op = CompareOp::Never;
    BorderColor border = BorderColor::TransparentBlack;
};

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual u64 Size() const = 0;
    /// The buffer's GPU virtual address (Vulkan buffer device address / D3D12 GPU VA).
    virtual u64 GpuAddress() const = 0;
    /// CPU pointer of Upload/Readback memory; nullptr for device-local memory.
    virtual u8* Mapped() const = 0;
    /// Invalidate CPU cache for mapped memory ranges (required for non-coherent host memory).
    virtual void Invalidate(u64 offset = 0, u64 size = ~0ULL) {}
    /// The backend's own object (Vulkan: VkBuffer), for recording outside this interface.
    virtual u64 NativeHandle() const = 0;
};

class Image {
public:
    virtual ~Image() = default;
    virtual const ImageDesc& Desc() const = 0;
    /// The backend's own object (Vulkan: VkImage).
    virtual u64 NativeHandle() const = 0;
};

class ImageView {
public:
    virtual ~ImageView() = default;
    /// The backend's own object (Vulkan: VkImageView).
    virtual u64 NativeHandle() const = 0;
};

class Sampler {
public:
    virtual ~Sampler() = default;
    /// The backend's own object (Vulkan: VkSampler).
    virtual u64 NativeHandle() const = 0;
};

enum class ShaderStage : u8 { Vertex, Hull, Domain, Geometry, Pixel, Compute };
enum StageMask : u32 {
    StageVertex = 1 << 0,
    StageHull = 1 << 1,
    StageDomain = 1 << 2,
    StageGeometry = 1 << 3,
    StagePixel = 1 << 4,
    StageCompute = 1 << 5,
    StageAllGraphics = 0x1f,
};

/// A specialization constant (SPIR-V constant_id).
struct SpecConstant {
    u32 id = 0;
    u32 value = 0;
};

struct ShaderCode {
    ShaderStage stage = ShaderStage::Vertex;
    std::span<const u32> spirv; ///< D3D12 converts it to DXIL
    std::string_view entry = "main";
    std::span<const SpecConstant> constants;
};

enum class BindingType : u8 {
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
    Sampler,
    UniformTexelBuffer,
    StorageTexelBuffer,
};

/// One binding of the per-draw descriptor set (Vulkan: set 0, binding `slot`).
struct BindingSlot {
    u32 slot = 0;
    BindingType type = BindingType::UniformBuffer;
    u32 stages = StageAllGraphics;
};

struct PipelineLayoutDesc {
    std::span<const BindingSlot> bindings;
    u32 push_constant_bytes = 0;
    u32 push_constant_stages = StageAllGraphics;
};

class PipelineLayout {
public:
    virtual ~PipelineLayout() = default;
};

enum class Topology : u8 {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
    TriangleFan,
    PatchList,
};
enum class CullMode : u8 { None, Front, Back, FrontAndBack };
enum class FillMode : u8 { Solid, Wireframe, Point };
enum class BlendFactor : u8 {
    Zero, One, SrcColor, OneMinusSrcColor, DstColor, OneMinusDstColor, SrcAlpha,
    OneMinusSrcAlpha, DstAlpha, OneMinusDstAlpha, ConstantColor, OneMinusConstantColor,
    ConstantAlpha, OneMinusConstantAlpha, SrcAlphaSaturate, Src1Color, OneMinusSrc1Color,
    Src1Alpha, OneMinusSrc1Alpha,
};
enum class BlendOp : u8 { Add, Subtract, ReverseSubtract, Min, Max };
enum class StencilOp : u8 { Keep, Zero, Replace, IncrementClamp, DecrementClamp, Invert, IncrementWrap, DecrementWrap };

struct BlendTarget {
    bool enable = false;
    BlendFactor src_color = BlendFactor::One;
    BlendFactor dst_color = BlendFactor::Zero;
    BlendOp color_op = BlendOp::Add;
    BlendFactor src_alpha = BlendFactor::One;
    BlendFactor dst_alpha = BlendFactor::Zero;
    BlendOp alpha_op = BlendOp::Add;
    u8 write_mask = 0xf;
};

struct StencilFace {
    StencilOp fail = StencilOp::Keep;
    StencilOp depth_fail = StencilOp::Keep;
    StencilOp pass = StencilOp::Keep;
    CompareOp compare = CompareOp::Always;
};

struct VertexAttribute {
    u32 location = 0;
    u32 binding = 0;
    Format format = Format::Undefined;
    u32 offset = 0;
};
struct VertexBinding {
    u32 binding = 0;
    u32 stride = 0;
    bool per_instance = false;
    u32 divisor = 1;
};

constexpr u32 kMaxColorTargets = 8;

struct GraphicsPipelineDesc {
    PipelineLayout* layout = nullptr;
    std::span<const ShaderCode> shaders;
    std::span<const VertexBinding> vertex_bindings;
    std::span<const VertexAttribute> vertex_attributes;
    Topology topology = Topology::TriangleList;
    u32 patch_control_points = 0;
    bool primitive_restart = false;
    // Rasterizer
    CullMode cull = CullMode::None;
    bool front_counter_clockwise = false;
    FillMode fill = FillMode::Solid;
    bool depth_clamp = false;
    bool depth_clip = true;
    bool depth_bias = false;
    bool rasterizer_discard = false;
    u32 samples = 1;
    bool alpha_to_coverage = false;
    // Depth / stencil
    bool depth_test = false;
    bool depth_write = false;
    CompareOp depth_compare = CompareOp::Always;
    bool depth_bounds_test = false;
    bool stencil_test = false;
    StencilFace stencil_front{};
    StencilFace stencil_back{};
    // Colour
    u32 color_count = 0;
    std::array<Format, kMaxColorTargets> color_formats{};
    std::array<BlendTarget, kMaxColorTargets> blend{};
    bool logic_op_enable = false;
    u8 logic_op = 0;
    Format depth_format = Format::Undefined;
    Format stencil_format = Format::Undefined;
    /// Cull mode, front face, depth and stencil tests, depth bias enable and primitive restart
    /// come from SetRasterState / SetDepthStencilState instead of the fields above, so fewer
    /// pipelines are needed (Caps().dynamic_raster_depth_stencil; D3D12 bakes them in).
    bool dynamic_raster_depth_stencil = false;
    std::string_view name;
};

struct ComputePipelineDesc {
    PipelineLayout* layout = nullptr;
    ShaderCode shader{};
    std::string_view name;
};

class Pipeline {
public:
    virtual ~Pipeline() = default;
};

/// One resource for the per-draw descriptor set.
struct Binding {
    u32 slot = 0;
    BindingType type = BindingType::UniformBuffer;
    const Buffer* buffer = nullptr;
    u64 offset = 0;
    u64 size = 0;
    Format texel_format = Format::Undefined; ///< texel buffers
    const ImageView* view = nullptr;
    const Sampler* sampler = nullptr;
};

enum class LoadOp : u8 { Load, Clear, DontCare };
enum class StoreOp : u8 { Store, DontCare };

struct ColorAttachment {
    const ImageView* view = nullptr;
    LoadOp load = LoadOp::Load;
    StoreOp store = StoreOp::Store;
    std::array<float, 4> clear{};
};
struct DepthAttachment {
    const ImageView* view = nullptr;
    LoadOp depth_load = LoadOp::Load;
    StoreOp depth_store = StoreOp::Store;
    LoadOp stencil_load = LoadOp::Load;
    StoreOp stencil_store = StoreOp::Store;
    float clear_depth = 1.0f;
    u8 clear_stencil = 0;
    bool read_only = false;
};
struct RenderingDesc {
    u32 width = 0;
    u32 height = 0;
    u32 layers = 1;
    u32 color_count = 0;
    std::array<ColorAttachment, kMaxColorTargets> colors{};
    DepthAttachment depth{};
};

struct Viewport {
    float x = 0, y = 0, width = 0, height = 0, min_depth = 0, max_depth = 1;
};
struct Rect {
    s32 x = 0, y = 0;
    u32 width = 0, height = 0;
};

enum class IndexType : u8 { U16, U32 };

/// Pipeline state set while recording, for pipelines made with dynamic_raster_depth_stencil.
struct DynamicRasterState {
    CullMode cull = CullMode::None;
    bool front_counter_clockwise = false;
    bool depth_bias = false;
    bool primitive_restart = false;
    bool rasterizer_discard = false;
};
struct DynamicDepthStencilState {
    bool depth_test = false;
    bool depth_write = false;
    CompareOp depth_compare = CompareOp::Always;
    bool depth_bounds_test = false;
    bool stencil_test = false;
    StencilFace stencil_front{};
    StencilFace stencil_back{};
};

enum class CommandListLevel : u8 {
    Primary,   ///< submitted with Queue::Submit
    Secondary, ///< run inside another API's command buffer (see NativeHandle, MarkExecuted)
};

/// What a barrier waits for and what it holds back (global: all resources).
enum BarrierScope : u32 {
    ScopeNone = 0,
    ScopeDraw = 1 << 0,     ///< vertex .. colour/depth output of draws
    ScopeCompute = 1 << 1,  ///< compute shaders
    ScopeTransfer = 1 << 2, ///< copies and clears
    ScopeHost = 1 << 3,     ///< CPU access of mapped memory
    ScopeAll = 0xf,
};

struct BufferImageCopy {
    u64 buffer_offset = 0;
    u32 buffer_row_texels = 0; ///< 0: tightly packed
    u32 buffer_image_rows = 0;
    u32 mip = 0;
    u32 base_layer = 0;
    u32 layers = 1;
    std::array<s32, 3> offset{};
    std::array<u32, 3> extent{1, 1, 1};
    u8 aspect = AspectColor;
};

/// A list of GPU commands, recorded by one thread at a time.
class CommandList {
public:
    virtual ~CommandList() = default;

    virtual void Begin() = 0;
    virtual void End() = 0;

    virtual void BeginRendering(const RenderingDesc& desc) = 0;
    virtual void EndRendering() = 0;

    virtual void SetViewports(std::span<const Viewport> viewports) = 0;
    virtual void SetScissors(std::span<const Rect> scissors) = 0;
    virtual void SetBlendConstants(const std::array<float, 4>& constants) = 0;
    virtual void SetStencilReference(u8 front, u8 back) = 0;
    virtual void SetStencilMasks(u8 compare_front, u8 compare_back, u8 write_front, u8 write_back) = 0;
    virtual void SetDepthBias(float constant, float clamp, float slope) = 0;
    virtual void SetDepthBounds(float min, float max) = 0;
    virtual void SetLineWidth(float width) = 0;

    virtual void SetRasterState(const DynamicRasterState& state) = 0;
    virtual void SetDepthStencilState(const DynamicDepthStencilState& state) = 0;

    virtual void BindPipeline(const Pipeline* pipeline) = 0;
    /// The per-draw descriptor set of the bound pipeline's layout.
    virtual void PushBindings(std::span<const Binding> bindings) = 0;
    virtual void PushConstants(u32 offset, std::span<const u8> data) = 0;
    virtual void BindVertexBuffers(u32 first, std::span<const Buffer* const> buffers,
                                   std::span<const u64> offsets) = 0;
    virtual void BindIndexBuffer(const Buffer* buffer, u64 offset, IndexType type) = 0;

    virtual void Draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance) = 0;
    virtual void DrawIndexed(u32 indices, u32 instances, u32 first_index, s32 vertex_offset,
                             u32 first_instance) = 0;
    virtual void Dispatch(u32 x, u32 y, u32 z) = 0;

    virtual void Barrier(u32 wait_for, u32 hold_back) = 0;
    virtual void CopyBuffer(const Buffer* src, u64 src_offset, const Buffer* dst, u64 dst_offset,
                            u64 size) = 0;
    virtual void CopyBufferToImage(const Buffer* src, const Image* dst,
                                   std::span<const BufferImageCopy> regions) = 0;
    virtual void CopyImageToBuffer(const Image* src, const Buffer* dst,
                                   std::span<const BufferImageCopy> regions) = 0;
    virtual void FillBuffer(const Buffer* dst, u64 offset, u64 size, u32 value) = 0;
    virtual void ClearColorImage(const Image* image, const std::array<float, 4>& color) = 0;

    virtual void BeginMarker(std::string_view name) = 0;
    virtual void EndMarker() = 0;

    /// Runs secondary lists (Caps().secondary_lists) here, outside any rendering.
    virtual void ExecuteSecondary(std::span<CommandList* const> lists) = 0;

    /// The backend's own object (Vulkan: VkCommandBuffer), for running a secondary list inside
    /// a command buffer the renderer records elsewhere.
    virtual u64 NativeHandle() const = 0;
};

/// A GPU timeline: Signal from a submission, wait from the CPU or another submission.
class Fence {
public:
    virtual ~Fence() = default;
    virtual u64 Completed() const = 0;
    /// Waits until the fence reaches `value` or `timeout_ns` passes; true when reached.
    virtual bool Wait(u64 value, u64 timeout_ns) const = 0;
};

struct FenceValue {
    Fence* fence = nullptr;
    u64 value = 0;
};

class Queue {
public:
    virtual ~Queue() = default;
    /// Runs the lists in order after `waits`, then signals `signal`.
    virtual void Submit(std::span<CommandList* const> lists, std::span<const FenceValue> waits,
                        FenceValue signal) = 0;
};

/// What the device can do, for choices the renderer makes.
struct DeviceCaps {
    BackendKind backend = BackendKind::Vulkan;
    bool import_host_memory = false; ///< guest memory as a buffer
    u64 host_import_alignment = 0;
    bool geometry_shaders = false;
    bool tessellation = false;
    bool depth_bounds = false;
    bool depth_clip_control = false;
    bool dynamic_raster_depth_stencil = false;
    bool secondary_lists = false; ///< CommandListLevel::Secondary and ExecuteSecondary
    u32 max_push_constant_bytes = 128;
    u32 max_push_bindings = 0; ///< more per-draw bindings than this use a descriptor set per draw
};

class Device {
public:
    virtual ~Device() = default;
    virtual const DeviceCaps& Caps() const = 0;

    virtual std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) = 0;
    /// Wraps existing CPU memory (page aligned, `Caps().host_import_alignment`) as a buffer the
    /// GPU reads and writes in place; nullptr when unsupported.
    /// NAT-107: a buffer made elsewhere (Vulkan: a VkBuffer), not owned: bound and copied
    /// like the others, never destroyed by this object.
    virtual std::unique_ptr<Buffer> WrapBuffer(u64 native_handle, u64 size) = 0;
    virtual std::unique_ptr<Buffer> ImportHostMemory(void* memory, u64 size, u32 usage,
                                                     std::string_view name) = 0;
    virtual std::unique_ptr<Image> CreateImage(const ImageDesc& desc) = 0;
    virtual std::unique_ptr<ImageView> CreateImageView(const Image* image,
                                                       const ImageViewDesc& desc) = 0;
    virtual std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) = 0;
    virtual std::unique_ptr<PipelineLayout> CreatePipelineLayout(const PipelineLayoutDesc& desc) = 0;
    /// May run on any thread; nullptr when the driver refuses the pipeline.
    virtual std::unique_ptr<Pipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc& desc) = 0;
    virtual std::unique_ptr<Pipeline> CreateComputePipeline(const ComputePipelineDesc& desc) = 0;
    virtual std::unique_ptr<Fence> CreateFence(u64 initial) = 0;
    /// A timeline created elsewhere (Vulkan: a timeline VkSemaphore), not owned.
    virtual std::unique_ptr<Fence> ImportFence(u64 native_handle) = 0;

    /// A command list from the calling thread's pool, ready for Begin. A list goes back to its
    /// pool once the GPU has run the submission it was part of (or when Release is called on a
    /// list that was never submitted).
    virtual CommandList* AcquireCommandList(CommandListLevel level = CommandListLevel::Primary) = 0;
    /// Wraps an externally managed command buffer (e.g. from the scheduler's primary session).
    virtual std::unique_ptr<CommandList> WrapCommandBuffer(u64 native_cmdbuf) { return nullptr; }
    virtual void ReleaseUnsubmitted(CommandList* list) = 0;
    /// A list run by someone else's submission (a secondary list) goes back to its pool once
    /// `retire` is reached.
    virtual void MarkExecuted(CommandList* list, FenceValue retire) = 0;
    virtual Queue& GraphicsQueue() = 0;

    /// Recycles command lists and frees objects the GPU has finished with.
    virtual void Collect() = 0;

    /// For lists another renderer submits (secondary lists run in its command buffers): GPU
    /// progress is read from `fence`, and `current` gives the value the work recorded now will
    /// signal. Deferred destruction then follows that timeline instead of Queue::Submit's.
    virtual void UseExternalTimeline(Fence* fence, std::function<u64()> current) = 0;
    /// Records the first layout change of the images created since the last call (Queue::Submit
    /// does it on its own); for lists another renderer runs, before any list using the images.
    virtual void RecordPendingImageInit(CommandList& list) = 0;
    /// Whether images were created since the last RecordPendingImageInit.
    virtual bool PendingImageInit() = 0;
};

} // namespace Native::Rhi
