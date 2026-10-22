// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/assert.h"
#include "common/logging/log.h"
#include "ryty_gpu/native/rhi/rhi_vulkan.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <vk_mem_alloc.h>

namespace Native::Rhi {
namespace {

vk::Format ToVk(Format format) {
    return static_cast<vk::Format>(static_cast<u32>(format));
}

bool IsDepthFormat(vk::Format format) {
    switch (format) {
    case vk::Format::eD16Unorm:
    case vk::Format::eD32Sfloat:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return true;
    default:
        return false;
    }
}
bool HasStencil(vk::Format format) {
    return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint ||
           format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eS8Uint;
}

vk::ImageAspectFlags AspectOf(vk::Format format) {
    if (!IsDepthFormat(format)) {
        return format == vk::Format::eS8Uint ? vk::ImageAspectFlagBits::eStencil
                                             : vk::ImageAspectFlagBits::eColor;
    }
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eDepth;
    if (HasStencil(format)) {
        aspect |= vk::ImageAspectFlagBits::eStencil;
    }
    return aspect;
}

vk::ImageAspectFlags ToVkAspect(u8 aspect) {
    vk::ImageAspectFlags out{};
    if (aspect & AspectColor) out |= vk::ImageAspectFlagBits::eColor;
    if (aspect & AspectDepth) out |= vk::ImageAspectFlagBits::eDepth;
    if (aspect & AspectStencil) out |= vk::ImageAspectFlagBits::eStencil;
    return out;
}

vk::ShaderStageFlags ToVkStages(u32 mask) {
    vk::ShaderStageFlags out{};
    if (mask & StageVertex) out |= vk::ShaderStageFlagBits::eVertex;
    if (mask & StageHull) out |= vk::ShaderStageFlagBits::eTessellationControl;
    if (mask & StageDomain) out |= vk::ShaderStageFlagBits::eTessellationEvaluation;
    if (mask & StageGeometry) out |= vk::ShaderStageFlagBits::eGeometry;
    if (mask & StagePixel) out |= vk::ShaderStageFlagBits::eFragment;
    if (mask & StageCompute) out |= vk::ShaderStageFlagBits::eCompute;
    return out;
}

vk::ShaderStageFlagBits ToVkStage(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex: return vk::ShaderStageFlagBits::eVertex;
    case ShaderStage::Hull: return vk::ShaderStageFlagBits::eTessellationControl;
    case ShaderStage::Domain: return vk::ShaderStageFlagBits::eTessellationEvaluation;
    case ShaderStage::Geometry: return vk::ShaderStageFlagBits::eGeometry;
    case ShaderStage::Pixel: return vk::ShaderStageFlagBits::eFragment;
    case ShaderStage::Compute: return vk::ShaderStageFlagBits::eCompute;
    }
    return vk::ShaderStageFlagBits::eVertex;
}

vk::DescriptorType ToVk(BindingType type) {
    switch (type) {
    case BindingType::UniformBuffer: return vk::DescriptorType::eUniformBuffer;
    case BindingType::StorageBuffer: return vk::DescriptorType::eStorageBuffer;
    case BindingType::SampledImage: return vk::DescriptorType::eSampledImage;
    case BindingType::StorageImage: return vk::DescriptorType::eStorageImage;
    case BindingType::Sampler: return vk::DescriptorType::eSampler;
    case BindingType::UniformTexelBuffer: return vk::DescriptorType::eUniformTexelBuffer;
    case BindingType::StorageTexelBuffer: return vk::DescriptorType::eStorageTexelBuffer;
    }
    return vk::DescriptorType::eUniformBuffer;
}

vk::Filter ToVk(Filter filter) {
    return filter == Filter::Linear ? vk::Filter::eLinear : vk::Filter::eNearest;
}
vk::SamplerMipmapMode ToVkMip(Filter filter) {
    return filter == Filter::Linear ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest;
}
vk::SamplerAddressMode ToVk(AddressMode mode) {
    switch (mode) {
    case AddressMode::Repeat: return vk::SamplerAddressMode::eRepeat;
    case AddressMode::MirroredRepeat: return vk::SamplerAddressMode::eMirroredRepeat;
    case AddressMode::ClampToEdge: return vk::SamplerAddressMode::eClampToEdge;
    case AddressMode::ClampToBorder: return vk::SamplerAddressMode::eClampToBorder;
    case AddressMode::MirrorClampToEdge: return vk::SamplerAddressMode::eMirrorClampToEdge;
    }
    return vk::SamplerAddressMode::eRepeat;
}
vk::CompareOp ToVk(CompareOp op) {
    return static_cast<vk::CompareOp>(static_cast<u32>(op)); // same order as VkCompareOp
}
vk::StencilOp ToVk(StencilOp op) {
    return static_cast<vk::StencilOp>(static_cast<u32>(op)); // same order as VkStencilOp
}
vk::BlendFactor ToVk(BlendFactor factor) {
    return static_cast<vk::BlendFactor>(static_cast<u32>(factor)); // same order as VkBlendFactor
}
vk::BlendOp ToVk(BlendOp op) {
    return static_cast<vk::BlendOp>(static_cast<u32>(op)); // same order as VkBlendOp
}
vk::PrimitiveTopology ToVk(Topology topology) {
    switch (topology) {
    case Topology::PointList: return vk::PrimitiveTopology::ePointList;
    case Topology::LineList: return vk::PrimitiveTopology::eLineList;
    case Topology::LineStrip: return vk::PrimitiveTopology::eLineStrip;
    case Topology::TriangleList: return vk::PrimitiveTopology::eTriangleList;
    case Topology::TriangleStrip: return vk::PrimitiveTopology::eTriangleStrip;
    case Topology::TriangleFan: return vk::PrimitiveTopology::eTriangleFan;
    case Topology::PatchList: return vk::PrimitiveTopology::ePatchList;
    }
    return vk::PrimitiveTopology::eTriangleList;
}
vk::ComponentSwizzle ToVk(Swizzle swizzle) {
    switch (swizzle) {
    case Swizzle::Identity: return vk::ComponentSwizzle::eIdentity;
    case Swizzle::Zero: return vk::ComponentSwizzle::eZero;
    case Swizzle::One: return vk::ComponentSwizzle::eOne;
    case Swizzle::R: return vk::ComponentSwizzle::eR;
    case Swizzle::G: return vk::ComponentSwizzle::eG;
    case Swizzle::B: return vk::ComponentSwizzle::eB;
    case Swizzle::A: return vk::ComponentSwizzle::eA;
    }
    return vk::ComponentSwizzle::eIdentity;
}
vk::ImageViewType ToVk(ViewDim dim) {
    switch (dim) {
    case ViewDim::D1: return vk::ImageViewType::e1D;
    case ViewDim::D1Array: return vk::ImageViewType::e1DArray;
    case ViewDim::D2: return vk::ImageViewType::e2D;
    case ViewDim::D2Array: return vk::ImageViewType::e2DArray;
    case ViewDim::D3: return vk::ImageViewType::e3D;
    case ViewDim::Cube: return vk::ImageViewType::eCube;
    case ViewDim::CubeArray: return vk::ImageViewType::eCubeArray;
    }
    return vk::ImageViewType::e2D;
}

// Pipeline stages and memory accesses of a barrier scope.
void ScopeToVk(u32 scope, bool source, vk::PipelineStageFlags2& stages, vk::AccessFlags2& access) {
    using S = vk::PipelineStageFlagBits2;
    using A = vk::AccessFlagBits2;
    if (scope & ScopeDraw) {
        stages |= S::eAllGraphics;
        access |= source ? (A::eShaderWrite | A::eColorAttachmentWrite | A::eDepthStencilAttachmentWrite)
                         : (A::eMemoryRead | A::eMemoryWrite);
    }
    if (scope & ScopeCompute) {
        stages |= S::eComputeShader;
        access |= source ? A::eShaderWrite : (A::eMemoryRead | A::eMemoryWrite);
    }
    if (scope & ScopeTransfer) {
        stages |= S::eAllTransfer;
        access |= source ? A::eTransferWrite : (A::eTransferRead | A::eTransferWrite);
    }
    if (scope & ScopeHost) {
        stages |= S::eHost;
        access |= source ? A::eHostWrite : (A::eHostRead | A::eHostWrite);
    }
    if (!stages) {
        stages = source ? S::eTopOfPipe : S::eBottomOfPipe;
    }
}

class VulkanDevice;

// Objects the GPU may still use are destroyed once the submissions before them have finished.
struct Retired {
    u64 tick;
    std::function<void()> destroy;
};

class VulkanBuffer final : public Buffer {
public:
    VulkanBuffer(VulkanDevice& device, vk::Buffer buffer, VmaAllocation allocation,
                 vk::DeviceMemory memory, u64 size, u64 address, u8* mapped)
        : device{device}, buffer{buffer}, allocation{allocation}, memory{memory}, size{size},
          address{address}, mapped{mapped} {}
    ~VulkanBuffer() override;
    u64 Size() const override {
        return size;
    }
    u64 GpuAddress() const override {
        return address;
    }
    u8* Mapped() const override {
        return mapped;
    }
    void Invalidate(u64 offset = 0, u64 size = ~0ULL) override;
    u64 NativeHandle() const override {
        return reinterpret_cast<u64>(static_cast<VkBuffer>(buffer));
    }

    VulkanDevice& device;
    vk::Buffer buffer;
    VmaAllocation allocation;
    vk::DeviceMemory memory; // imported memory, when not from VMA
    bool owned = true;       // NAT-107: false for WrapBuffer
    u64 size;
    u64 address;
    u8* mapped;
};

class VulkanImage final : public Image {
public:
    VulkanImage(VulkanDevice& device, const ImageDesc& desc, vk::Image image,
                VmaAllocation allocation)
        : device{device}, desc{desc}, image{image}, allocation{allocation} {
        this->desc.name = {};
    }
    ~VulkanImage() override;
    const ImageDesc& Desc() const override {
        return desc;
    }
    u64 NativeHandle() const override {
        return reinterpret_cast<u64>(static_cast<VkImage>(image));
    }

    VulkanDevice& device;
    ImageDesc desc;
    vk::Image image;
    VmaAllocation allocation;
};

class VulkanImageView final : public ImageView {
public:
    VulkanImageView(VulkanDevice& device, vk::ImageView view) : device{device}, view{view} {}
    ~VulkanImageView() override;
    u64 NativeHandle() const override {
        return reinterpret_cast<u64>(static_cast<VkImageView>(view));
    }
    VulkanDevice& device;
    vk::ImageView view;
};

class VulkanSampler final : public Sampler {
public:
    VulkanSampler(VulkanDevice& device, vk::Sampler sampler) : device{device}, sampler{sampler} {}
    ~VulkanSampler() override;
    u64 NativeHandle() const override {
        return reinterpret_cast<u64>(static_cast<VkSampler>(sampler));
    }
    VulkanDevice& device;
    vk::Sampler sampler;
};

class VulkanPipelineLayout final : public PipelineLayout {
public:
    VulkanPipelineLayout(VulkanDevice& device, vk::DescriptorSetLayout set_layout,
                         vk::PipelineLayout layout, vk::ShaderStageFlags push_stages,
                         bool push_bindings)
        : device{device}, set_layout{set_layout}, layout{layout}, push_stages{push_stages},
          push_bindings{push_bindings} {}
    ~VulkanPipelineLayout() override;
    VulkanDevice& device;
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout layout;
    vk::ShaderStageFlags push_stages;
    bool push_bindings; // push descriptors; otherwise a descriptor set per draw
};

class VulkanPipeline final : public Pipeline {
public:
    VulkanPipeline(VulkanDevice& device, vk::Pipeline pipeline, const VulkanPipelineLayout* layout,
                   vk::PipelineBindPoint bind_point)
        : device{device}, pipeline{pipeline}, layout{layout}, bind_point{bind_point} {}
    ~VulkanPipeline() override;
    VulkanDevice& device;
    vk::Pipeline pipeline;
    const VulkanPipelineLayout* layout;
    vk::PipelineBindPoint bind_point;
};

class VulkanFence final : public Fence {
public:
    VulkanFence(vk::Device device, vk::Semaphore semaphore, bool owned = true)
        : device{device}, semaphore{semaphore}, owned{owned} {}
    ~VulkanFence() override {
        if (owned) {
            device.destroySemaphore(semaphore);
        }
    }
    u64 Completed() const override {
        const auto [result, value] = device.getSemaphoreCounterValue(semaphore);
        return result == vk::Result::eSuccess ? value : 0;
    }
    bool Wait(u64 value, u64 timeout_ns) const override {
        const vk::SemaphoreWaitInfo info{
            .semaphoreCount = 1,
            .pSemaphores = &semaphore,
            .pValues = &value,
        };
        return device.waitSemaphores(info, timeout_ns) == vk::Result::eSuccess;
    }
    vk::Device device;
    vk::Semaphore semaphore;
    bool owned;
};

struct ThreadPool;

class VulkanCommandList final : public CommandList {
public:
    VulkanCommandList(VulkanDevice& device, ThreadPool* pool, vk::CommandBuffer cmd,
                      CommandListLevel level)
        : device{device}, pool{pool}, cmd{cmd}, level{level} {}

    void Begin() override;
    void End() override;
    void BeginRendering(const RenderingDesc& desc) override;
    void EndRendering() override;
    void SetViewports(std::span<const Viewport> viewports) override;
    void SetScissors(std::span<const Rect> scissors) override;
    void SetBlendConstants(const std::array<float, 4>& constants) override;
    void SetStencilReference(u8 front, u8 back) override;
    void SetStencilMasks(u8 compare_front, u8 compare_back, u8 write_front, u8 write_back) override;
    void SetDepthBias(float constant, float clamp, float slope) override;
    void SetDepthBounds(float min, float max) override;
    void SetLineWidth(float width) override;
    void SetRasterState(const DynamicRasterState& state) override;
    void SetDepthStencilState(const DynamicDepthStencilState& state) override;
    void BindPipeline(const Pipeline* pipeline) override;
    void PushBindings(std::span<const Binding> bindings) override;
    void PushConstants(u32 offset, std::span<const u8> data) override;
    void BindVertexBuffers(u32 first, std::span<const Buffer* const> buffers,
                           std::span<const u64> offsets) override;
    void BindIndexBuffer(const Buffer* buffer, u64 offset, IndexType type) override;
    void Draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance) override;
    void DrawIndexed(u32 indices, u32 instances, u32 first_index, s32 vertex_offset,
                     u32 first_instance) override;
    void Dispatch(u32 x, u32 y, u32 z) override;
    void Barrier(u32 wait_for, u32 hold_back) override;
    void CopyBuffer(const Buffer* src, u64 src_offset, const Buffer* dst, u64 dst_offset,
                    u64 size) override;
    void CopyBufferToImage(const Buffer* src, const Image* dst,
                           std::span<const BufferImageCopy> regions) override;
    void CopyImageToBuffer(const Image* src, const Buffer* dst,
                           std::span<const BufferImageCopy> regions) override;
    void FillBuffer(const Buffer* dst, u64 offset, u64 size, u32 value) override;
    void ClearColorImage(const Image* image, const std::array<float, 4>& color) override;
    void BeginMarker(std::string_view name) override;
    void EndMarker() override;
    void ExecuteSecondary(std::span<CommandList* const> lists) override;
    u64 NativeHandle() const override {
        return reinterpret_cast<u64>(static_cast<VkCommandBuffer>(cmd));
    }

    /// Whether the GPU is done with the list (its retire fence reached the value).
    bool Retired() const;

    VulkanDevice& device;
    ThreadPool* pool;
    vk::CommandBuffer cmd;
    CommandListLevel level;
    // The fence and value that free the list: the device timeline after Queue::Submit, another
    // timeline after MarkExecuted, or nothing (value 0) after ReleaseUnsubmitted.
    std::atomic<const Fence*> retire_fence{nullptr};
    std::atomic<u64> retire_value{~0ULL};
    vk::DescriptorPool descriptor_pool{}; // descriptor sets of layouts too big to push
    const VulkanPipeline* graphics = nullptr;
    const VulkanPipeline* compute = nullptr;
    vk::PipelineBindPoint last_bind = vk::PipelineBindPoint::eGraphics;
};

// Command lists are recorded by one thread at a time, so each thread has its own pool; lists the
// GPU has finished with are reset by that thread when it next asks for one.
struct ThreadPool {
    vk::CommandPool pool;
    std::vector<VulkanCommandList*> free;
    std::vector<VulkanCommandList*> free_secondary;
    std::deque<VulkanCommandList*> used;
    std::vector<std::unique_ptr<VulkanCommandList>> all;
};

class VulkanQueue final : public Queue {
public:
    explicit VulkanQueue(VulkanDevice& device) : device{device} {}
    void Submit(std::span<CommandList* const> lists, std::span<const FenceValue> waits,
                FenceValue signal) override;
    VulkanDevice& device;
};

// The Vulkan objects a device runs on: the renderer's own (shared with the presenter), or ones
// created for tests and tools (owned).
struct VulkanContext {
    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    VmaAllocator allocator = nullptr;
    u32 queue_family = 0;
    vk::Queue queue;
    std::mutex* submit_mutex = nullptr;
    bool external_memory_host = false;
    bool geometry = false;
    bool tessellation = false;
    bool depth_bounds = false;
    bool depth_clip_control = false;
    u32 max_push_descriptors = 32;
    std::function<void()> destroy; // owned objects
};

class VulkanDevice final : public Device {
public:
    explicit VulkanDevice(VulkanContext context);
    ~VulkanDevice() override;

    const DeviceCaps& Caps() const override {
        return caps;
    }
    std::unique_ptr<Buffer> CreateBuffer(const BufferDesc& desc) override;
    std::unique_ptr<Buffer> WrapBuffer(u64 native_handle, u64 size) override {
        auto buffer = std::make_unique<VulkanBuffer>(
            *this, vk::Buffer{reinterpret_cast<VkBuffer>(native_handle)}, VmaAllocation{},
            vk::DeviceMemory{}, size, 0, nullptr);
        buffer->owned = false;
        return buffer;
    }
    std::unique_ptr<Buffer> ImportHostMemory(void* memory, u64 size, u32 usage,
                                             std::string_view name) override;
    std::unique_ptr<Image> CreateImage(const ImageDesc& desc) override;
    std::unique_ptr<ImageView> CreateImageView(const Image* image,
                                               const ImageViewDesc& desc) override;
    std::unique_ptr<Sampler> CreateSampler(const SamplerDesc& desc) override;
    std::unique_ptr<PipelineLayout> CreatePipelineLayout(const PipelineLayoutDesc& desc) override;
    std::unique_ptr<Pipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc& desc) override;
    std::unique_ptr<Pipeline> CreateComputePipeline(const ComputePipelineDesc& desc) override;
    std::unique_ptr<Fence> CreateFence(u64 initial) override;
    std::unique_ptr<Fence> ImportFence(u64 native_handle) override;
    CommandList* AcquireCommandList(CommandListLevel level) override;
    std::unique_ptr<CommandList> WrapCommandBuffer(u64 native_cmdbuf) override;
    void ReleaseUnsubmitted(CommandList* list) override;
    void MarkExecuted(CommandList* list, FenceValue retire) override;
    Queue& GraphicsQueue() override {
        return queue;
    }
    void Collect() override;
    void UseExternalTimeline(Fence* fence, std::function<u64()> current) override;
    void RecordPendingImageInit(CommandList& list) override;
    bool PendingImageInit() override {
        std::scoped_lock lock{init_mutex};
        return !pending_init.empty();
    }

    /// Destroys GPU objects once the GPU has run the lists recorded so far. Those are submitted
    /// with the next submission at the latest (render jobs record during a frame and the frame is
    /// submitted once), so one more submission is a safe margin.
    void Retire(std::function<void()>&& destroy) {
        const u64 now = external_current ? external_current()
                                         : submitted.load(std::memory_order_relaxed);
        std::scoped_lock lock{retire_mutex};
        retired.push_back({now + 2, std::move(destroy)});
    }
    u64 CompletedTick() const {
        if (external_fence) {
            return external_fence->Completed();
        }
        const auto [result, value] = device.getSemaphoreCounterValue(timeline);
        return result == vk::Result::eSuccess ? value : 0;
    }
    /// Queues the first layout change of a new image (UNDEFINED to GENERAL).
    void InitImageLayout(vk::Image image, vk::ImageAspectFlags aspect);
    std::unique_ptr<Image> CreateImageFrom(const VkImageCreateInfo& image_ci, const ImageDesc& desc);
    void RecordImageInit(vk::CommandBuffer cmd, std::vector<std::pair<vk::Image, vk::ImageAspectFlags>>& images);

    VulkanContext ctx;
    vk::Device device;
    VmaAllocator allocator;
    DeviceCaps caps{};
    VulkanQueue queue{*this};
    vk::Semaphore timeline;           // one tick per submission
    std::unique_ptr<VulkanFence> timeline_fence;
    std::atomic<u64> submitted{0};    // ticks handed out
    std::mutex retire_mutex;
    std::deque<Retired> retired;
    std::mutex pools_mutex;
    std::unordered_map<std::thread::id, std::unique_ptr<ThreadPool>> pools;
    // New images go from UNDEFINED to GENERAL in a command buffer submitted before the next list.
    std::mutex init_mutex;
    vk::CommandPool init_pool;
    std::deque<std::pair<u64, vk::CommandBuffer>> init_used; // (tick, buffer)
    std::vector<std::pair<vk::Image, vk::ImageAspectFlags>> pending_init;
    Fence* external_fence = nullptr;
    std::function<u64()> external_current;
};

void VulkanBuffer::Invalidate(u64 offset, u64 size) {
    if (allocation) {
        vmaInvalidateAllocation(device.allocator, allocation, offset,
                                size == ~0ULL ? VK_WHOLE_SIZE : size);
    }
}

VulkanBuffer::~VulkanBuffer() {
    if (!owned) {
        return; // NAT-107: wrapped
    }
    device.Retire([dev = device.device, alloc = device.allocator, buffer = buffer,
                   allocation = allocation, memory = memory] {
        if (allocation) {
            vmaDestroyBuffer(alloc, static_cast<VkBuffer>(buffer), allocation);
        } else {
            dev.destroyBuffer(buffer);
            dev.freeMemory(memory);
        }
    });
}
VulkanImage::~VulkanImage() {
    device.Retire([alloc = device.allocator, image = image, allocation = allocation] {
        vmaDestroyImage(alloc, static_cast<VkImage>(image), allocation);
    });
}
VulkanImageView::~VulkanImageView() {
    device.Retire([dev = device.device, view = view] { dev.destroyImageView(view); });
}
VulkanSampler::~VulkanSampler() {
    device.Retire([dev = device.device, sampler = sampler] { dev.destroySampler(sampler); });
}
VulkanPipelineLayout::~VulkanPipelineLayout() {
    device.Retire([dev = device.device, set_layout = set_layout, layout = layout] {
        dev.destroyPipelineLayout(layout);
        dev.destroyDescriptorSetLayout(set_layout);
    });
}
VulkanPipeline::~VulkanPipeline() {
    device.Retire([dev = device.device, pipeline = pipeline] { dev.destroyPipeline(pipeline); });
}

VulkanDevice::VulkanDevice(VulkanContext context)
    : ctx{std::move(context)}, device{ctx.device}, allocator{ctx.allocator} {
    caps.backend = BackendKind::Vulkan;
    caps.import_host_memory = ctx.external_memory_host;
    caps.geometry_shaders = ctx.geometry;
    caps.tessellation = ctx.tessellation;
    caps.depth_bounds = ctx.depth_bounds;
    caps.depth_clip_control = ctx.depth_clip_control;
    const auto properties = ctx.physical_device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();
    caps.max_push_constant_bytes =
        properties.get<vk::PhysicalDeviceProperties2>().properties.limits.maxPushConstantsSize;
    caps.host_import_alignment =
        properties.get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>()
            .minImportedHostPointerAlignment;
    caps.max_push_bindings = ctx.max_push_descriptors;
    caps.dynamic_raster_depth_stencil = true; // Vulkan 1.3 core dynamic state
    caps.secondary_lists = true;

    const vk::SemaphoreTypeCreateInfo timeline_info{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0,
    };
    const auto [result, semaphore] =
        device.createSemaphore(vk::SemaphoreCreateInfo{.pNext = &timeline_info});
    ASSERT_MSG(result == vk::Result::eSuccess, "NAT-011: timeline semaphore failed");
    timeline = semaphore;
    timeline_fence = std::make_unique<VulkanFence>(device, semaphore, false);

    const auto [pool_result, pool] = device.createCommandPool(vk::CommandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = ctx.queue_family,
    });
    ASSERT_MSG(pool_result == vk::Result::eSuccess, "NAT-011: command pool failed");
    init_pool = pool;
}

VulkanDevice::~VulkanDevice() {
    std::ignore = device.waitIdle();
    Collect();
    for (auto& retired_object : retired) {
        retired_object.destroy();
    }
    for (auto& [id, pool] : pools) {
        for (auto& list : pool->all) {
            if (list->descriptor_pool) {
                device.destroyDescriptorPool(list->descriptor_pool);
            }
        }
        device.destroyCommandPool(pool->pool);
    }
    device.destroyCommandPool(init_pool);
    device.destroySemaphore(timeline);
    if (ctx.destroy) {
        ctx.destroy();
    }
}

std::unique_ptr<Buffer> VulkanDevice::CreateBuffer(const BufferDesc& desc) {
    vk::BufferUsageFlags usage{};
    if (desc.usage & BufferUsageVertex) usage |= vk::BufferUsageFlagBits::eVertexBuffer;
    if (desc.usage & BufferUsageIndex) usage |= vk::BufferUsageFlagBits::eIndexBuffer;
    if (desc.usage & BufferUsageUniform) usage |= vk::BufferUsageFlagBits::eUniformBuffer;
    if (desc.usage & BufferUsageStorage) usage |= vk::BufferUsageFlagBits::eStorageBuffer;
    if (desc.usage & BufferUsageTexel) {
        usage |= vk::BufferUsageFlagBits::eUniformTexelBuffer |
                 vk::BufferUsageFlagBits::eStorageTexelBuffer;
    }
    if (desc.usage & BufferUsageIndirect) usage |= vk::BufferUsageFlagBits::eIndirectBuffer;
    if (desc.usage & BufferUsageTransferSrc) usage |= vk::BufferUsageFlagBits::eTransferSrc;
    if (desc.usage & BufferUsageTransferDst) usage |= vk::BufferUsageFlagBits::eTransferDst;
    if (desc.usage & BufferUsageDeviceAddress) {
        usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    const VkBufferCreateInfo buffer_ci = static_cast<VkBufferCreateInfo>(vk::BufferCreateInfo{
        .size = desc.size,
        .usage = usage,
    });
    VmaAllocationCreateInfo alloc_ci{};
    switch (desc.memory) {
    case MemoryKind::DeviceLocal:
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        break;
    case MemoryKind::Upload:
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        break;
    case MemoryKind::Readback:
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        break;
    }
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator, &buffer_ci, &alloc_ci, &buffer, &allocation, &info) != VK_SUCCESS) {
        return nullptr;
    }
    u64 address = 0;
    if (desc.usage & BufferUsageDeviceAddress) {
        address = device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = buffer});
    }
    return std::make_unique<VulkanBuffer>(*this, buffer, allocation, vk::DeviceMemory{}, desc.size,
                                          address, static_cast<u8*>(info.pMappedData));
}

std::unique_ptr<Buffer> VulkanDevice::ImportHostMemory(void* memory, u64 size, u32 usage,
                                                       std::string_view name) {
    if (!caps.import_host_memory) {
        return nullptr;
    }
    const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
    VkMemoryHostPointerPropertiesEXT pointer_properties{
        VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (d.vkGetMemoryHostPointerPropertiesEXT(device,
                                              VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                              memory, &pointer_properties) != VK_SUCCESS ||
        pointer_properties.memoryTypeBits == 0) {
        return nullptr;
    }
    VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr,
                                         VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            &flags_info,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            memory};
    u32 memory_type_index = static_cast<u32>(std::countr_zero(pointer_properties.memoryTypeBits));
    // FIX-188: Intel Arc (#65) imports into cached host memory.
    const bool intel = ctx.physical_device.getProperties().vendorID == 0x8086;
    const auto mem_props = ctx.physical_device.getMemoryProperties();
    for (u32 i = 0; intel && i < mem_props.memoryTypeCount; ++i) {
        if ((pointer_properties.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCached)) {
            memory_type_index = i;
            break;
        }
    }
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import, size,
                                  memory_type_index};
    VkDeviceMemory device_memory{};
    if (d.vkAllocateMemory(device, &allocate, nullptr, &device_memory) != VK_SUCCESS) {
        return nullptr;
    }
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                              nullptr,
                                              VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (usage & BufferUsageTexel) {
        buffer_info.usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                             VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    }
    VkBuffer buffer{};
    if (d.vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS ||
        d.vkBindBufferMemory(device, buffer, device_memory, 0) != VK_SUCCESS) {
        if (buffer) {
            d.vkDestroyBuffer(device, buffer, nullptr);
        }
        d.vkFreeMemory(device, device_memory, nullptr);
        return nullptr;
    }
    const u64 address = device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = buffer});
    return std::make_unique<VulkanBuffer>(*this, buffer, VmaAllocation{}, device_memory, size,
                                          address, static_cast<u8*>(memory));
}

void VulkanDevice::InitImageLayout(vk::Image image, vk::ImageAspectFlags aspect) {
    std::scoped_lock lock{init_mutex};
    pending_init.emplace_back(image, aspect);
}

void VulkanDevice::RecordImageInit(vk::CommandBuffer cmd,
                                   std::vector<std::pair<vk::Image, vk::ImageAspectFlags>>& images) {
    std::vector<vk::ImageMemoryBarrier2> barriers;
    barriers.reserve(images.size());
    for (const auto& [image, aspect] : images) {
        barriers.push_back({
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
    }
    if (!barriers.empty()) {
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
            .pImageMemoryBarriers = barriers.data(),
        });
    }
    images.clear();
}

void VulkanDevice::RecordPendingImageInit(CommandList& list) {
    std::vector<std::pair<vk::Image, vk::ImageAspectFlags>> images;
    {
        std::scoped_lock lock{init_mutex};
        images.swap(pending_init);
    }
    RecordImageInit(static_cast<VulkanCommandList&>(list).cmd, images);
}

void VulkanDevice::UseExternalTimeline(Fence* fence, std::function<u64()> current) {
    external_fence = fence;
    external_current = std::move(current);
}

std::unique_ptr<Image> VulkanDevice::CreateImage(const ImageDesc& desc) {
    const vk::Format format = ToVk(desc.format);
    vk::ImageUsageFlags usage{};
    if (desc.usage & ImageUsageSampled) usage |= vk::ImageUsageFlagBits::eSampled;
    if (desc.usage & ImageUsageStorage) usage |= vk::ImageUsageFlagBits::eStorage;
    if (desc.usage & ImageUsageColorTarget) usage |= vk::ImageUsageFlagBits::eColorAttachment;
    if (desc.usage & ImageUsageDepthTarget) usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
    if (desc.usage & ImageUsageTransferSrc) usage |= vk::ImageUsageFlagBits::eTransferSrc;
    if (desc.usage & ImageUsageTransferDst) usage |= vk::ImageUsageFlagBits::eTransferDst;
    vk::ImageCreateFlags flags{};
    if (desc.usage & ImageUsageCube) flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    if (desc.usage & ImageUsageMutableFormat) {
        flags |= vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage;
    }
    if (desc.dim == ImageDim::D3) flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
    const VkImageCreateInfo image_ci = static_cast<VkImageCreateInfo>(vk::ImageCreateInfo{
        .flags = flags,
        .imageType = desc.dim == ImageDim::D1   ? vk::ImageType::e1D
                     : desc.dim == ImageDim::D3 ? vk::ImageType::e3D
                                                : vk::ImageType::e2D,
        .format = format,
        .extent{desc.width, desc.height, desc.dim == ImageDim::D3 ? desc.depth : 1},
        .mipLevels = desc.mips,
        .arrayLayers = desc.dim == ImageDim::D3 ? 1 : desc.layers,
        .samples = static_cast<vk::SampleCountFlagBits>(desc.samples),
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    // An image the device can't make with this format and usage is refused here, so the caller
    // can ask again with less usage (the driver may not reject it itself).
    VkImageFormatProperties limits{};
    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceImageFormatProperties(
            ctx.physical_device, image_ci.format, image_ci.imageType, image_ci.tiling,
            image_ci.usage, image_ci.flags, &limits) != VK_SUCCESS ||
        limits.maxMipLevels < desc.mips || limits.maxArrayLayers < image_ci.arrayLayers ||
        !(limits.sampleCounts & image_ci.samples)) {
        return nullptr;
    }
    return CreateImageFrom(image_ci, desc);
}

std::unique_ptr<Image> VulkanDevice::CreateImageFrom(const VkImageCreateInfo& image_ci,
                                                    const ImageDesc& desc) {
    const VmaAllocationCreateInfo alloc_ci{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImage image{};
    VmaAllocation allocation{};
    if (vmaCreateImage(allocator, &image_ci, &alloc_ci, &image, &allocation, nullptr) != VK_SUCCESS) {
        return nullptr;
    }
    InitImageLayout(image, AspectOf(static_cast<vk::Format>(image_ci.format)));
    return std::make_unique<VulkanImage>(*this, desc, image, allocation);
}

std::unique_ptr<ImageView> VulkanDevice::CreateImageView(const Image* image,
                                                         const ImageViewDesc& desc) {
    const auto* vulkan_image = static_cast<const VulkanImage*>(image);
    const Format format = desc.format == Format::Undefined ? vulkan_image->desc.format : desc.format;
    const vk::ImageViewUsageCreateInfo usage_info{
        .usage = desc.storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled,
    };
    const bool limit_usage = (vulkan_image->desc.usage & ImageUsageMutableFormat) != 0;
    const auto [result, view] = device.createImageView(vk::ImageViewCreateInfo{
        .pNext = limit_usage ? &usage_info : nullptr,
        .image = vulkan_image->image,
        .viewType = ToVk(desc.dim),
        .format = ToVk(format),
        .components{ToVk(desc.swizzle[0]), ToVk(desc.swizzle[1]), ToVk(desc.swizzle[2]),
                    ToVk(desc.swizzle[3])},
        .subresourceRange{
            .aspectMask = ToVkAspect(desc.aspect),
            .baseMipLevel = desc.base_mip,
            .levelCount = desc.mips,
            .baseArrayLayer = desc.base_layer,
            .layerCount = desc.layers,
        },
    });
    if (result != vk::Result::eSuccess) {
        return nullptr;
    }
    return std::make_unique<VulkanImageView>(*this, view);
}

std::unique_ptr<Sampler> VulkanDevice::CreateSampler(const SamplerDesc& desc) {
    const vk::BorderColor border = desc.border == BorderColor::OpaqueBlack
                                       ? vk::BorderColor::eFloatOpaqueBlack
                                   : desc.border == BorderColor::OpaqueWhite
                                       ? vk::BorderColor::eFloatOpaqueWhite
                                       : vk::BorderColor::eFloatTransparentBlack;
    const auto [result, sampler] = device.createSampler(vk::SamplerCreateInfo{
        .magFilter = ToVk(desc.mag_filter),
        .minFilter = ToVk(desc.min_filter),
        .mipmapMode = ToVkMip(desc.mip_filter),
        .addressModeU = ToVk(desc.address[0]),
        .addressModeV = ToVk(desc.address[1]),
        .addressModeW = ToVk(desc.address[2]),
        .mipLodBias = desc.mip_lod_bias,
        .anisotropyEnable = desc.max_anisotropy > 1.0f,
        .maxAnisotropy = desc.max_anisotropy,
        .compareEnable = desc.compare,
        .compareOp = ToVk(desc.compare_op),
        .minLod = desc.min_lod,
        .maxLod = desc.max_lod,
        .borderColor = border,
    });
    if (result != vk::Result::eSuccess) {
        return nullptr;
    }
    return std::make_unique<VulkanSampler>(*this, sampler);
}

std::unique_ptr<PipelineLayout> VulkanDevice::CreatePipelineLayout(const PipelineLayoutDesc& desc) {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    bindings.reserve(desc.bindings.size());
    for (const BindingSlot& slot : desc.bindings) {
        bindings.push_back({
            .binding = slot.slot,
            .descriptorType = ToVk(slot.type),
            .descriptorCount = 1,
            .stageFlags = ToVkStages(slot.stages),
        });
    }
    const bool push_bindings = bindings.size() <= caps.max_push_bindings;
    const auto [set_result, set_layout] =
        device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{
            .flags = push_bindings ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                                   : vk::DescriptorSetLayoutCreateFlagBits{},
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    if (set_result != vk::Result::eSuccess) {
        return nullptr;
    }
    const vk::ShaderStageFlags push_stages = ToVkStages(desc.push_constant_stages);
    const vk::PushConstantRange push_range{
        .stageFlags = push_stages,
        .offset = 0,
        .size = desc.push_constant_bytes,
    };
    const auto [layout_result, layout] = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = desc.push_constant_bytes ? 1u : 0u,
        .pPushConstantRanges = &push_range,
    });
    if (layout_result != vk::Result::eSuccess) {
        device.destroyDescriptorSetLayout(set_layout);
        return nullptr;
    }
    return std::make_unique<VulkanPipelineLayout>(*this, set_layout, layout, push_stages,
                                                  push_bindings);
}

std::unique_ptr<Pipeline> VulkanDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc& desc) {
    const auto* layout = static_cast<const VulkanPipelineLayout*>(desc.layout);
    std::vector<vk::ShaderModule> modules;
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    std::vector<std::string> entries;
    entries.reserve(desc.shaders.size());
    std::vector<std::vector<vk::SpecializationMapEntry>> spec_entries(desc.shaders.size());
    std::vector<vk::SpecializationInfo> spec_infos(desc.shaders.size());
    const auto destroy_modules = [&] {
        for (const vk::ShaderModule module : modules) {
            device.destroyShaderModule(module);
        }
    };
    for (const ShaderCode& code : desc.shaders) {
        const auto [result, module] = device.createShaderModule(vk::ShaderModuleCreateInfo{
            .codeSize = code.spirv.size_bytes(),
            .pCode = code.spirv.data(),
        });
        if (result != vk::Result::eSuccess) {
            destroy_modules();
            return nullptr;
        }
        modules.push_back(module);
        entries.emplace_back(code.entry);
        const size_t index = stages.size();
        for (u32 i = 0; i < code.constants.size(); ++i) {
            spec_entries[index].push_back({
                .constantID = code.constants[i].id,
                .offset = static_cast<u32>(i * sizeof(SpecConstant) + offsetof(SpecConstant, value)),
                .size = sizeof(u32),
            });
        }
        spec_infos[index] = {
            .mapEntryCount = static_cast<u32>(spec_entries[index].size()),
            .pMapEntries = spec_entries[index].data(),
            .dataSize = code.constants.size_bytes(),
            .pData = code.constants.data(),
        };
        stages.push_back({
            .stage = ToVkStage(code.stage),
            .module = module,
            .pName = entries.back().c_str(),
            .pSpecializationInfo = code.constants.empty() ? nullptr : &spec_infos[index],
        });
    }
    std::vector<vk::VertexInputBindingDescription> vertex_bindings;
    std::vector<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    for (const VertexBinding& binding : desc.vertex_bindings) {
        vertex_bindings.push_back({
            .binding = binding.binding,
            .stride = binding.stride,
            .inputRate = binding.per_instance ? vk::VertexInputRate::eInstance
                                              : vk::VertexInputRate::eVertex,
        });
        if (binding.per_instance && binding.divisor != 1) {
            divisors.push_back({.binding = binding.binding, .divisor = binding.divisor});
        }
    }
    std::vector<vk::VertexInputAttributeDescription> attributes;
    for (const VertexAttribute& attribute : desc.vertex_attributes) {
        attributes.push_back({
            .location = attribute.location,
            .binding = attribute.binding,
            .format = ToVk(attribute.format),
            .offset = attribute.offset,
        });
    }
    const vk::PipelineVertexInputDivisorStateCreateInfoEXT divisor_info{
        .vertexBindingDivisorCount = static_cast<u32>(divisors.size()),
        .pVertexBindingDivisors = divisors.data(),
    };
    const vk::PipelineVertexInputStateCreateInfo vertex_input{
        .pNext = divisors.empty() ? nullptr : &divisor_info,
        .vertexBindingDescriptionCount = static_cast<u32>(vertex_bindings.size()),
        .pVertexBindingDescriptions = vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = ToVk(desc.topology),
        .primitiveRestartEnable = desc.primitive_restart,
    };
    const vk::PipelineTessellationStateCreateInfo tessellation{
        .patchControlPoints = desc.patch_control_points,
    };
    const vk::PipelineViewportStateCreateInfo viewport{
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const vk::PipelineRasterizationDepthClipStateCreateInfoEXT depth_clip{
        .depthClipEnable = desc.depth_clip,
    };
    const vk::PipelineRasterizationStateCreateInfo raster{
        .pNext = &depth_clip,
        .depthClampEnable = desc.depth_clamp,
        .rasterizerDiscardEnable = desc.rasterizer_discard,
        .polygonMode = desc.fill == FillMode::Wireframe ? vk::PolygonMode::eLine
                       : desc.fill == FillMode::Point   ? vk::PolygonMode::ePoint
                                                        : vk::PolygonMode::eFill,
        .cullMode = static_cast<vk::CullModeFlagBits>(static_cast<u32>(desc.cull)),
        .frontFace = desc.front_counter_clockwise ? vk::FrontFace::eCounterClockwise
                                                  : vk::FrontFace::eClockwise,
        .depthBiasEnable = desc.depth_bias,
        .lineWidth = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = static_cast<vk::SampleCountFlagBits>(desc.samples),
        .alphaToCoverageEnable = desc.alpha_to_coverage,
    };
    const auto stencil = [](const StencilFace& face) {
        return vk::StencilOpState{
            .failOp = ToVk(face.fail),
            .passOp = ToVk(face.pass),
            .depthFailOp = ToVk(face.depth_fail),
            .compareOp = ToVk(face.compare),
        };
    };
    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = desc.depth_test,
        .depthWriteEnable = desc.depth_write,
        .depthCompareOp = ToVk(desc.depth_compare),
        .depthBoundsTestEnable = desc.depth_bounds_test,
        .stencilTestEnable = desc.stencil_test,
        .front = stencil(desc.stencil_front),
        .back = stencil(desc.stencil_back),
    };
    std::array<vk::PipelineColorBlendAttachmentState, kMaxColorTargets> blend{};
    std::array<vk::Format, kMaxColorTargets> color_formats{};
    for (u32 i = 0; i < desc.color_count; ++i) {
        const BlendTarget& target = desc.blend[i];
        blend[i] = {
            .blendEnable = target.enable,
            .srcColorBlendFactor = ToVk(target.src_color),
            .dstColorBlendFactor = ToVk(target.dst_color),
            .colorBlendOp = ToVk(target.color_op),
            .srcAlphaBlendFactor = ToVk(target.src_alpha),
            .dstAlphaBlendFactor = ToVk(target.dst_alpha),
            .alphaBlendOp = ToVk(target.alpha_op),
            .colorWriteMask = static_cast<vk::ColorComponentFlagBits>(target.write_mask),
        };
        color_formats[i] = ToVk(desc.color_formats[i]);
    }
    const vk::PipelineColorBlendStateCreateInfo color_blend{
        .logicOpEnable = desc.logic_op_enable,
        .logicOp = static_cast<vk::LogicOp>(desc.logic_op),
        .attachmentCount = desc.color_count,
        .pAttachments = blend.data(),
    };
    std::vector<vk::DynamicState> dynamic_states{
        vk::DynamicState::eViewport,          vk::DynamicState::eScissor,
        vk::DynamicState::eBlendConstants,    vk::DynamicState::eStencilReference,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eDepthBias,         vk::DynamicState::eDepthBounds,
        vk::DynamicState::eLineWidth,
    };
    if (desc.dynamic_raster_depth_stencil) {
        dynamic_states.insert(dynamic_states.end(), {
            vk::DynamicState::eCullMode,           vk::DynamicState::eFrontFace,
            vk::DynamicState::eDepthBiasEnable,    vk::DynamicState::ePrimitiveRestartEnable,
            vk::DynamicState::eRasterizerDiscardEnable, vk::DynamicState::eDepthTestEnable,
            vk::DynamicState::eDepthWriteEnable,   vk::DynamicState::eDepthCompareOp,
            vk::DynamicState::eDepthBoundsTestEnable, vk::DynamicState::eStencilTestEnable,
            vk::DynamicState::eStencilOp,
        });
    }
    const vk::PipelineDynamicStateCreateInfo dynamic{
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = desc.color_count,
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat = ToVk(desc.depth_format),
        .stencilAttachmentFormat = ToVk(desc.stencil_format),
    };
    const auto [result, pipeline] = device.createGraphicsPipeline(
        {}, vk::GraphicsPipelineCreateInfo{
                .pNext = &rendering,
                .stageCount = static_cast<u32>(stages.size()),
                .pStages = stages.data(),
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pTessellationState = desc.patch_control_points ? &tessellation : nullptr,
                .pViewportState = &viewport,
                .pRasterizationState = &raster,
                .pMultisampleState = &multisample,
                .pDepthStencilState = &depth_stencil,
                .pColorBlendState = &color_blend,
                .pDynamicState = &dynamic,
                .layout = layout->layout,
            });
    destroy_modules();
    if (result != vk::Result::eSuccess) {
        return nullptr;
    }
    return std::make_unique<VulkanPipeline>(*this, pipeline, layout,
                                            vk::PipelineBindPoint::eGraphics);
}

std::unique_ptr<Pipeline> VulkanDevice::CreateComputePipeline(const ComputePipelineDesc& desc) {
    const auto* layout = static_cast<const VulkanPipelineLayout*>(desc.layout);
    const auto [module_result, module] = device.createShaderModule(vk::ShaderModuleCreateInfo{
        .codeSize = desc.shader.spirv.size_bytes(),
        .pCode = desc.shader.spirv.data(),
    });
    if (module_result != vk::Result::eSuccess) {
        return nullptr;
    }
    const std::string entry{desc.shader.entry};
    std::vector<vk::SpecializationMapEntry> spec_entries;
    for (u32 i = 0; i < desc.shader.constants.size(); ++i) {
        spec_entries.push_back({
            .constantID = desc.shader.constants[i].id,
            .offset = static_cast<u32>(i * sizeof(SpecConstant) + offsetof(SpecConstant, value)),
            .size = sizeof(u32),
        });
    }
    const vk::SpecializationInfo spec_info{
        .mapEntryCount = static_cast<u32>(spec_entries.size()),
        .pMapEntries = spec_entries.data(),
        .dataSize = desc.shader.constants.size_bytes(),
        .pData = desc.shader.constants.data(),
    };
    const auto [result, pipeline] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage{
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = module,
                    .pName = entry.c_str(),
                    .pSpecializationInfo = spec_entries.empty() ? nullptr : &spec_info,
                },
                .layout = layout->layout,
            });
    device.destroyShaderModule(module);
    if (result != vk::Result::eSuccess) {
        return nullptr;
    }
    return std::make_unique<VulkanPipeline>(*this, pipeline, layout,
                                            vk::PipelineBindPoint::eCompute);
}

std::unique_ptr<Fence> VulkanDevice::CreateFence(u64 initial) {
    const vk::SemaphoreTypeCreateInfo timeline_info{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = initial,
    };
    const auto [result, semaphore] =
        device.createSemaphore(vk::SemaphoreCreateInfo{.pNext = &timeline_info});
    if (result != vk::Result::eSuccess) {
        return nullptr;
    }
    return std::make_unique<VulkanFence>(device, semaphore);
}

std::unique_ptr<Fence> VulkanDevice::ImportFence(u64 native_handle) {
    return std::make_unique<VulkanFence>(
        device, vk::Semaphore{reinterpret_cast<VkSemaphore>(native_handle)}, false);
}

bool VulkanCommandList::Retired() const {
    const u64 value = retire_value.load(std::memory_order_acquire);
    if (value == 0) {
        return true; // released without being submitted
    }
    const Fence* fence = retire_fence.load(std::memory_order_acquire);
    return fence && fence->Completed() >= value;
}

CommandList* VulkanDevice::AcquireCommandList(CommandListLevel level) {
    ThreadPool* pool;
    {
        std::scoped_lock lock{pools_mutex};
        auto& slot = pools[std::this_thread::get_id()];
        if (!slot) {
            slot = std::make_unique<ThreadPool>();
            const auto [result, command_pool] = device.createCommandPool(vk::CommandPoolCreateInfo{
                .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = ctx.queue_family,
            });
            ASSERT_MSG(result == vk::Result::eSuccess, "NAT-011: command pool failed");
            slot->pool = command_pool;
        }
        pool = slot.get();
    }
    // Lists retire in roughly the order they were used; one still running stops the scan.
    for (size_t scanned = 0; !pool->used.empty() && scanned < 64; ++scanned) {
        VulkanCommandList* list = pool->used.front();
        if (!list->Retired()) {
            break;
        }
        pool->used.pop_front();
        std::ignore = list->cmd.reset();
        if (list->descriptor_pool) {
            std::ignore = device.resetDescriptorPool(list->descriptor_pool);
        }
        (list->level == CommandListLevel::Secondary ? pool->free_secondary : pool->free)
            .push_back(list);
    }
    auto& free = level == CommandListLevel::Secondary ? pool->free_secondary : pool->free;
    VulkanCommandList* list;
    if (!free.empty()) {
        list = free.back();
        free.pop_back();
    } else {
        const auto [result, buffers] = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = pool->pool,
            .level = level == CommandListLevel::Secondary ? vk::CommandBufferLevel::eSecondary
                                                          : vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        });
        ASSERT_MSG(result == vk::Result::eSuccess, "NAT-011: command buffer failed");
        pool->all.push_back(std::make_unique<VulkanCommandList>(*this, pool, buffers[0], level));
        list = pool->all.back().get();
    }
    list->retire_fence.store(nullptr, std::memory_order_relaxed);
    list->retire_value.store(~0ULL, std::memory_order_relaxed);
    pool->used.push_back(list);
    return list;
}

std::unique_ptr<CommandList> VulkanDevice::WrapCommandBuffer(u64 native_cmdbuf) {
    vk::CommandBuffer cmdbuf{reinterpret_cast<VkCommandBuffer>(native_cmdbuf)};
    return std::make_unique<VulkanCommandList>(*this, nullptr, cmdbuf, CommandListLevel::Primary);
}

void VulkanDevice::ReleaseUnsubmitted(CommandList* list) {
    if (!list) return;
    static_cast<VulkanCommandList*>(list)->retire_value.store(0, std::memory_order_release);
}

void VulkanDevice::MarkExecuted(CommandList* list, FenceValue retire) {
    if (!list) return;
    auto* vulkan_list = static_cast<VulkanCommandList*>(list);
    vulkan_list->retire_fence.store(retire.fence, std::memory_order_relaxed);
    vulkan_list->retire_value.store(retire.value, std::memory_order_release);
}

void VulkanDevice::Collect() {
    const u64 completed = CompletedTick();
    std::vector<std::function<void()>> ready;
    {
        std::scoped_lock lock{retire_mutex};
        while (!retired.empty() && retired.front().tick <= completed) {
            ready.push_back(std::move(retired.front().destroy));
            retired.pop_front();
        }
    }
    for (auto& destroy : ready) {
        destroy();
    }
}

void VulkanQueue::Submit(std::span<CommandList* const> lists, std::span<const FenceValue> waits,
                         FenceValue signal) {
    std::vector<vk::CommandBuffer> buffers;
    buffers.reserve(lists.size() + 1);
    std::unique_lock init_lock{device.init_mutex};
    vk::CommandBuffer init_cmd{};
    if (!device.pending_init.empty()) {
        const u64 completed = device.CompletedTick();
        if (!device.init_used.empty() && device.init_used.front().first <= completed) {
            init_cmd = device.init_used.front().second;
            device.init_used.pop_front();
            std::ignore = init_cmd.reset();
        } else {
            const auto [result, allocated] =
                device.device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
                    .commandPool = device.init_pool,
                    .level = vk::CommandBufferLevel::ePrimary,
                    .commandBufferCount = 1,
                });
            ASSERT(result == vk::Result::eSuccess);
            init_cmd = allocated[0];
        }
        std::ignore = init_cmd.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        });
        device.RecordImageInit(init_cmd, device.pending_init);
        std::ignore = init_cmd.end();
        buffers.push_back(init_cmd);
    }
    for (CommandList* list : lists) {
        buffers.push_back(static_cast<VulkanCommandList*>(list)->cmd);
    }
    std::vector<vk::Semaphore> wait_semaphores;
    std::vector<u64> wait_values;
    std::vector<vk::PipelineStageFlags> wait_stages;
    for (const FenceValue& wait : waits) {
        wait_semaphores.push_back(static_cast<VulkanFence*>(wait.fence)->semaphore);
        wait_values.push_back(wait.value);
        wait_stages.push_back(vk::PipelineStageFlagBits::eAllCommands);
    }
    std::scoped_lock submit_lock{*device.ctx.submit_mutex};
    const u64 tick = device.submitted.fetch_add(1, std::memory_order_relaxed) + 1;
    if (init_cmd) {
        device.init_used.emplace_back(tick, init_cmd);
    }
    init_lock.unlock();
    std::array<vk::Semaphore, 2> signal_semaphores{device.timeline};
    std::array<u64, 2> signal_values{tick};
    u32 signal_count = 1;
    if (signal.fence) {
        signal_semaphores[1] = static_cast<VulkanFence*>(signal.fence)->semaphore;
        signal_values[1] = signal.value;
        signal_count = 2;
    }
    const vk::TimelineSemaphoreSubmitInfo timeline_info{
        .waitSemaphoreValueCount = static_cast<u32>(wait_values.size()),
        .pWaitSemaphoreValues = wait_values.data(),
        .signalSemaphoreValueCount = signal_count,
        .pSignalSemaphoreValues = signal_values.data(),
    };
    const vk::SubmitInfo submit{
        .pNext = &timeline_info,
        .waitSemaphoreCount = static_cast<u32>(wait_semaphores.size()),
        .pWaitSemaphores = wait_semaphores.data(),
        .pWaitDstStageMask = wait_stages.data(),
        .commandBufferCount = static_cast<u32>(buffers.size()),
        .pCommandBuffers = buffers.data(),
        .signalSemaphoreCount = signal_count,
        .pSignalSemaphores = signal_semaphores.data(),
    };
    const vk::Result result = device.ctx.queue.submit(submit);
    ASSERT_MSG(result == vk::Result::eSuccess, "NAT-011: queue submit failed: {}",
               vk::to_string(result));
    for (CommandList* list : lists) {
        auto* vulkan_list = static_cast<VulkanCommandList*>(list);
        vulkan_list->retire_fence.store(device.timeline_fence.get(), std::memory_order_relaxed);
        vulkan_list->retire_value.store(tick, std::memory_order_release);
    }
}

// ---- command list ----

void VulkanCommandList::Begin() {
    graphics = nullptr;
    compute = nullptr;
    if (!pool) {
        return; // Externally begun
    }
    // Secondary lists run outside render passes of the command buffer that executes them; they
    // begin and end their own rendering.
    const vk::CommandBufferInheritanceInfo inheritance{};
    std::ignore = cmd.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        .pInheritanceInfo = level == CommandListLevel::Secondary ? &inheritance : nullptr,
    });
}

void VulkanCommandList::End() {
    if (!pool) {
        return; // Externally ended
    }
    std::ignore = cmd.end();
}

void VulkanCommandList::BeginRendering(const RenderingDesc& desc) {
    const auto load = [](LoadOp op) {
        return op == LoadOp::Clear     ? vk::AttachmentLoadOp::eClear
               : op == LoadOp::Load    ? vk::AttachmentLoadOp::eLoad
                                       : vk::AttachmentLoadOp::eDontCare;
    };
    const auto store = [](StoreOp op) {
        return op == StoreOp::Store ? vk::AttachmentStoreOp::eStore : vk::AttachmentStoreOp::eDontCare;
    };
    std::array<vk::RenderingAttachmentInfo, kMaxColorTargets> colors{};
    for (u32 i = 0; i < desc.color_count; ++i) {
        const ColorAttachment& color = desc.colors[i];
        colors[i] = {
            .imageView = color.view ? static_cast<const VulkanImageView*>(color.view)->view
                                    : vk::ImageView{},
            .imageLayout = vk::ImageLayout::eGeneral,
            .loadOp = load(color.load),
            .storeOp = store(color.store),
            .clearValue{.color{.float32 = color.clear}},
        };
    }
    const DepthAttachment& depth = desc.depth;
    const vk::ImageView depth_view =
        depth.view ? static_cast<const VulkanImageView*>(depth.view)->view : vk::ImageView{};
    const vk::RenderingAttachmentInfo depth_info{
        .imageView = depth_view,
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = load(depth.depth_load),
        .storeOp = store(depth.depth_store),
        .clearValue{.depthStencil{.depth = depth.clear_depth, .stencil = depth.clear_stencil}},
    };
    const vk::RenderingAttachmentInfo stencil_info{
        .imageView = depth_view,
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = load(depth.stencil_load),
        .storeOp = store(depth.stencil_store),
        .clearValue{.depthStencil{.depth = depth.clear_depth, .stencil = depth.clear_stencil}},
    };
    cmd.beginRendering(vk::RenderingInfo{
        .renderArea{.offset{0, 0}, .extent{desc.width, desc.height}},
        .layerCount = desc.layers,
        .colorAttachmentCount = desc.color_count,
        .pColorAttachments = colors.data(),
        .pDepthAttachment = depth.view ? &depth_info : nullptr,
        .pStencilAttachment = depth.view ? &stencil_info : nullptr,
    });
}

void VulkanCommandList::EndRendering() {
    cmd.endRendering();
}

void VulkanCommandList::SetViewports(std::span<const Viewport> viewports) {
    std::array<vk::Viewport, 16> out{};
    const u32 count = std::min<u32>(static_cast<u32>(viewports.size()), 16);
    for (u32 i = 0; i < count; ++i) {
        out[i] = {viewports[i].x, viewports[i].y, viewports[i].width, viewports[i].height,
                  viewports[i].min_depth, viewports[i].max_depth};
    }
    cmd.setViewport(0, count, out.data());
}

void VulkanCommandList::SetScissors(std::span<const Rect> scissors) {
    std::array<vk::Rect2D, 16> out{};
    const u32 count = std::min<u32>(static_cast<u32>(scissors.size()), 16);
    for (u32 i = 0; i < count; ++i) {
        out[i] = {{scissors[i].x, scissors[i].y}, {scissors[i].width, scissors[i].height}};
    }
    cmd.setScissor(0, count, out.data());
}

void VulkanCommandList::SetBlendConstants(const std::array<float, 4>& constants) {
    cmd.setBlendConstants(constants.data());
}
void VulkanCommandList::SetStencilReference(u8 front, u8 back) {
    cmd.setStencilReference(vk::StencilFaceFlagBits::eFront, front);
    cmd.setStencilReference(vk::StencilFaceFlagBits::eBack, back);
}
void VulkanCommandList::SetStencilMasks(u8 compare_front, u8 compare_back, u8 write_front,
                                        u8 write_back) {
    cmd.setStencilCompareMask(vk::StencilFaceFlagBits::eFront, compare_front);
    cmd.setStencilCompareMask(vk::StencilFaceFlagBits::eBack, compare_back);
    cmd.setStencilWriteMask(vk::StencilFaceFlagBits::eFront, write_front);
    cmd.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, write_back);
}
void VulkanCommandList::SetDepthBias(float constant, float clamp, float slope) {
    cmd.setDepthBias(constant, clamp, slope);
}
void VulkanCommandList::SetDepthBounds(float min, float max) {
    cmd.setDepthBounds(min, max);
}
void VulkanCommandList::SetLineWidth(float width) {
    cmd.setLineWidth(width);
}

void VulkanCommandList::SetRasterState(const DynamicRasterState& state) {
    cmd.setCullMode(static_cast<vk::CullModeFlagBits>(static_cast<u32>(state.cull)));
    cmd.setFrontFace(state.front_counter_clockwise ? vk::FrontFace::eCounterClockwise
                                                   : vk::FrontFace::eClockwise);
    cmd.setDepthBiasEnable(state.depth_bias);
    cmd.setPrimitiveRestartEnable(state.primitive_restart);
    cmd.setRasterizerDiscardEnable(state.rasterizer_discard);
}

void VulkanCommandList::SetDepthStencilState(const DynamicDepthStencilState& state) {
    cmd.setDepthTestEnable(state.depth_test);
    cmd.setDepthWriteEnable(state.depth_write);
    cmd.setDepthCompareOp(ToVk(state.depth_compare));
    cmd.setDepthBoundsTestEnable(state.depth_bounds_test);
    cmd.setStencilTestEnable(state.stencil_test);
    const auto op = [&](vk::StencilFaceFlags face, const StencilFace& s) {
        cmd.setStencilOp(face, ToVk(s.fail), ToVk(s.pass), ToVk(s.depth_fail), ToVk(s.compare));
    };
    op(vk::StencilFaceFlagBits::eFront, state.stencil_front);
    op(vk::StencilFaceFlagBits::eBack, state.stencil_back);
}

void VulkanCommandList::BindPipeline(const Pipeline* pipeline) {
    const auto* vulkan_pipeline = static_cast<const VulkanPipeline*>(pipeline);
    cmd.bindPipeline(vulkan_pipeline->bind_point, vulkan_pipeline->pipeline);
    if (vulkan_pipeline->bind_point == vk::PipelineBindPoint::eCompute) {
        compute = vulkan_pipeline;
    } else {
        graphics = vulkan_pipeline;
    }
    last_bind = vulkan_pipeline->bind_point;
}

void VulkanCommandList::PushBindings(std::span<const Binding> bindings) {
    const VulkanPipeline* pipeline =
        last_bind == vk::PipelineBindPoint::eCompute ? compute : graphics;
    if (!pipeline || bindings.empty()) {
        return;
    }
    constexpr size_t kMax = 64;
    std::array<vk::WriteDescriptorSet, kMax> writes{};
    std::array<vk::DescriptorBufferInfo, kMax> buffer_infos{};
    std::array<vk::DescriptorImageInfo, kMax> image_infos{};
    std::array<vk::BufferView, kMax> texel_views{};
    const size_t count = std::min(bindings.size(), kMax);
    for (size_t i = 0; i < count; ++i) {
        const Binding& binding = bindings[i];
        vk::WriteDescriptorSet& write = writes[i];
        write.dstBinding = binding.slot;
        write.descriptorCount = 1;
        write.descriptorType = ToVk(binding.type);
        switch (binding.type) {
        case BindingType::UniformBuffer:
        case BindingType::StorageBuffer:
            buffer_infos[i] = {
                .buffer = binding.buffer ? static_cast<const VulkanBuffer*>(binding.buffer)->buffer
                                         : vk::Buffer{},
                .offset = binding.offset,
                .range = binding.size ? binding.size : VK_WHOLE_SIZE,
            };
            write.pBufferInfo = &buffer_infos[i];
            break;
        case BindingType::SampledImage:
        case BindingType::StorageImage:
            image_infos[i] = {
                .imageView = binding.view ? static_cast<const VulkanImageView*>(binding.view)->view
                                          : vk::ImageView{},
                .imageLayout = vk::ImageLayout::eGeneral,
            };
            write.pImageInfo = &image_infos[i];
            break;
        case BindingType::Sampler:
            image_infos[i] = {
                .sampler = binding.sampler ? static_cast<const VulkanSampler*>(binding.sampler)->sampler
                                           : vk::Sampler{},
            };
            write.pImageInfo = &image_infos[i];
            break;
        case BindingType::UniformTexelBuffer:
        case BindingType::StorageTexelBuffer:
            // Texel buffer views are created per draw; they are retired with the device's
            // deferred destruction once the submission has run.
            if (binding.buffer) {
                const auto [result, view] = device.device.createBufferView(vk::BufferViewCreateInfo{
                    .buffer = static_cast<const VulkanBuffer*>(binding.buffer)->buffer,
                    .format = ToVk(binding.texel_format),
                    .offset = binding.offset,
                    .range = binding.size ? binding.size : VK_WHOLE_SIZE,
                });
                if (result == vk::Result::eSuccess) {
                    texel_views[i] = view;
                    device.Retire([dev = device.device, view] { dev.destroyBufferView(view); });
                }
            }
            write.pTexelBufferView = &texel_views[i];
            break;
        }
    }
    if (pipeline->layout->push_bindings) {
        cmd.pushDescriptorSetKHR(pipeline->bind_point, pipeline->layout->layout, 0,
                                 static_cast<u32>(count), writes.data());
        return;
    }
    // More bindings than can be pushed: a descriptor set from this list's pool, freed with it.
    if (!descriptor_pool) {
        static constexpr std::array sizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1024},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 4096},
            vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
            vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 4096},
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1024},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1024},
        };
        const auto [result, created] = device.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .maxSets = 1024,
            .poolSizeCount = static_cast<u32>(sizes.size()),
            .pPoolSizes = sizes.data(),
        });
        if (result != vk::Result::eSuccess) {
            return;
        }
        descriptor_pool = created;
    }
    const auto [result, sets] = device.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &pipeline->layout->set_layout,
    });
    if (result != vk::Result::eSuccess) {
        return; // pool exhausted: the draw goes without its resources
    }
    for (size_t i = 0; i < count; ++i) {
        writes[i].dstSet = sets[0];
    }
    device.device.updateDescriptorSets(static_cast<u32>(count), writes.data(), 0, nullptr);
    cmd.bindDescriptorSets(pipeline->bind_point, pipeline->layout->layout, 0, sets[0], {});
}

void VulkanCommandList::PushConstants(u32 offset, std::span<const u8> data) {
    const VulkanPipeline* pipeline =
        last_bind == vk::PipelineBindPoint::eCompute ? compute : graphics;
    if (!pipeline) {
        return;
    }
    cmd.pushConstants(pipeline->layout->layout, pipeline->layout->push_stages, offset,
                      static_cast<u32>(data.size()), data.data());
}

void VulkanCommandList::BindVertexBuffers(u32 first, std::span<const Buffer* const> buffers,
                                          std::span<const u64> offsets) {
    std::array<vk::Buffer, 32> out{};
    const u32 count = std::min<u32>(static_cast<u32>(buffers.size()), 32);
    for (u32 i = 0; i < count; ++i) {
        out[i] = buffers[i] ? static_cast<const VulkanBuffer*>(buffers[i])->buffer : vk::Buffer{};
    }
    cmd.bindVertexBuffers(first, count, out.data(), offsets.data());
}

void VulkanCommandList::BindIndexBuffer(const Buffer* buffer, u64 offset, IndexType type) {
    cmd.bindIndexBuffer(static_cast<const VulkanBuffer*>(buffer)->buffer, offset,
                        type == IndexType::U16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);
}

void VulkanCommandList::Draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance) {
    cmd.draw(vertices, instances, first_vertex, first_instance);
}

void VulkanCommandList::DrawIndexed(u32 indices, u32 instances, u32 first_index,
                                    s32 vertex_offset, u32 first_instance) {
    cmd.drawIndexed(indices, instances, first_index, vertex_offset, first_instance);
}

void VulkanCommandList::Dispatch(u32 x, u32 y, u32 z) {
    cmd.dispatch(x, y, z);
}

void VulkanCommandList::Barrier(u32 wait_for, u32 hold_back) {
    vk::MemoryBarrier2 barrier{};
    ScopeToVk(wait_for, true, barrier.srcStageMask, barrier.srcAccessMask);
    ScopeToVk(hold_back, false, barrier.dstStageMask, barrier.dstAccessMask);
    cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
}

void VulkanCommandList::CopyBuffer(const Buffer* src, u64 src_offset, const Buffer* dst,
                                   u64 dst_offset, u64 size) {
    const vk::BufferCopy region{.srcOffset = src_offset, .dstOffset = dst_offset, .size = size};
    cmd.copyBuffer(static_cast<const VulkanBuffer*>(src)->buffer,
                   static_cast<const VulkanBuffer*>(dst)->buffer, region);
}

namespace {
std::vector<vk::BufferImageCopy> ToVkRegions(std::span<const BufferImageCopy> regions) {
    std::vector<vk::BufferImageCopy> out;
    out.reserve(regions.size());
    for (const BufferImageCopy& region : regions) {
        out.push_back({
            .bufferOffset = region.buffer_offset,
            .bufferRowLength = region.buffer_row_texels,
            .bufferImageHeight = region.buffer_image_rows,
            .imageSubresource{
                .aspectMask = ToVkAspect(region.aspect),
                .mipLevel = region.mip,
                .baseArrayLayer = region.base_layer,
                .layerCount = region.layers,
            },
            .imageOffset{region.offset[0], region.offset[1], region.offset[2]},
            .imageExtent{region.extent[0], region.extent[1], region.extent[2]},
        });
    }
    return out;
}
} // namespace

void VulkanCommandList::CopyBufferToImage(const Buffer* src, const Image* dst,
                                          std::span<const BufferImageCopy> regions) {
    const auto out = ToVkRegions(regions);
    cmd.copyBufferToImage(static_cast<const VulkanBuffer*>(src)->buffer,
                          static_cast<const VulkanImage*>(dst)->image, vk::ImageLayout::eGeneral,
                          static_cast<u32>(out.size()), out.data());
}

void VulkanCommandList::CopyImageToBuffer(const Image* src, const Buffer* dst,
                                          std::span<const BufferImageCopy> regions) {
    const auto out = ToVkRegions(regions);
    cmd.copyImageToBuffer(static_cast<const VulkanImage*>(src)->image, vk::ImageLayout::eGeneral,
                          static_cast<const VulkanBuffer*>(dst)->buffer,
                          static_cast<u32>(out.size()), out.data());
}

void VulkanCommandList::FillBuffer(const Buffer* dst, u64 offset, u64 size, u32 value) {
    cmd.fillBuffer(static_cast<const VulkanBuffer*>(dst)->buffer, offset, size, value);
}

void VulkanCommandList::ClearColorImage(const Image* image, const std::array<float, 4>& color) {
    const vk::ClearColorValue value{.float32 = color};
    const vk::ImageSubresourceRange range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
    cmd.clearColorImage(static_cast<const VulkanImage*>(image)->image, vk::ImageLayout::eGeneral,
                        value, range);
}

void VulkanCommandList::BeginMarker(std::string_view name) {
    if (!VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT) {
        return;
    }
    const std::string label{name};
    cmd.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{.pLabelName = label.c_str()});
}

void VulkanCommandList::ExecuteSecondary(std::span<CommandList* const> lists) {
    std::array<vk::CommandBuffer, 64> buffers{};
    for (size_t first = 0; first < lists.size(); first += buffers.size()) {
        const size_t count = std::min(buffers.size(), lists.size() - first);
        for (size_t i = 0; i < count; ++i) {
            buffers[i] = static_cast<VulkanCommandList*>(lists[first + i])->cmd;
        }
        cmd.executeCommands(static_cast<u32>(count), buffers.data());
    }
    // State bound by the secondary lists does not carry over.
    graphics = nullptr;
    compute = nullptr;
}

void VulkanCommandList::EndMarker() {
    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndDebugUtilsLabelEXT) {
        cmd.endDebugUtilsLabelEXT();
    }
}

} // namespace

std::unique_ptr<Image> CreateVulkanImage(Device& device, const VkImageCreateInfo& image_ci,
                                         const ImageDesc& desc) {
    return static_cast<VulkanDevice&>(device).CreateImageFrom(image_ci, desc);
}

std::unique_ptr<Device> CreateVulkanDevice(const Vulkan::Instance& instance) {
    VulkanContext context{
        .instance = instance.GetInstance(),
        .physical_device = instance.GetPhysicalDevice(),
        .device = instance.GetDevice(),
        .allocator = instance.GetAllocator(),
        .queue_family = instance.GetGraphicsQueueFamilyIndex(),
        .queue = instance.GetGraphicsQueue(),
        .submit_mutex = &Vulkan::Scheduler::submit_mutex,
        .external_memory_host = instance.IsExternalMemoryHostSupported(),
        .geometry = instance.IsGeometryStageSupported(),
        .tessellation = instance.IsTessellationSupported(),
        .depth_bounds = instance.IsDepthBoundsSupported(),
        .depth_clip_control = instance.IsDepthClipControlSupported(),
        .max_push_descriptors = instance.MaxPushDescriptors(),
    };
    return std::make_unique<VulkanDevice>(std::move(context));
}

std::unique_ptr<Device> CreateStandaloneVulkanDevice(std::string* error) {
    const auto fail = [error](std::string why) -> std::unique_ptr<Device> {
        if (error) {
            *error = std::move(why);
        }
        return nullptr;
    };
    static std::mutex standalone_submit_mutex;
    vk::UniqueInstance owned_instance =
        Vulkan::CreateInstance(Frontend::WindowSystemType::Headless, false, false);
    if (!owned_instance) {
        return fail("no Vulkan instance");
    }
    const vk::Instance instance = *owned_instance;
    const auto [enum_result, physical_devices] = instance.enumeratePhysicalDevices();
    if (enum_result != vk::Result::eSuccess || physical_devices.empty()) {
        return fail("no Vulkan devices");
    }
    // A discrete GPU with Vulkan 1.3 first.
    vk::PhysicalDevice physical{};
    for (const int pass : {0, 1}) {
        for (const vk::PhysicalDevice candidate : physical_devices) {
            const auto properties = candidate.getProperties();
            if (properties.apiVersion < VK_API_VERSION_1_3) {
                continue;
            }
            if (pass == 0 && properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu) {
                continue;
            }
            physical = candidate;
            break;
        }
        if (physical) {
            break;
        }
    }
    if (!physical) {
        return fail("no Vulkan 1.3 device");
    }
    u32 family = ~0u;
    const auto families = physical.getQueueFamilyProperties();
    for (u32 i = 0; i < families.size(); ++i) {
        if ((families[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            (families[i].queueFlags & vk::QueueFlagBits::eCompute)) {
            family = i;
            break;
        }
    }
    if (family == ~0u) {
        return fail("no graphics queue");
    }
    const auto [ext_result, extensions] = physical.enumerateDeviceExtensionProperties();
    const auto has_extension = [&](std::string_view name) {
        return std::ranges::any_of(extensions, [name](const vk::ExtensionProperties& e) {
            return std::string_view{e.extensionName.data()} == name;
        });
    };
    if (!has_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        return fail("no VK_KHR_push_descriptor");
    }
    std::vector<const char*> enabled{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    const bool host_import = has_extension(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    if (host_import) {
        enabled.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    }
    const bool clip_enable = has_extension(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
    if (clip_enable) {
        enabled.push_back(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
    }
    const auto features = physical.getFeatures();
    vk::PhysicalDeviceDepthClipEnableFeaturesEXT clip_features{.depthClipEnable = clip_enable};
    vk::PhysicalDeviceVulkan13Features vk13{
        .pNext = clip_enable ? &clip_features : nullptr,
        .synchronization2 = true,
        .dynamicRendering = true,
        .maintenance4 = true,
    };
    vk::PhysicalDeviceVulkan12Features vk12{
        .pNext = &vk13,
        .scalarBlockLayout = true,
        .timelineSemaphore = true,
        .bufferDeviceAddress = true,
    };
    const vk::PhysicalDeviceFeatures2 features2{
        .pNext = &vk12,
        .features{
            .independentBlend = features.independentBlend,
            .geometryShader = features.geometryShader,
            .tessellationShader = features.tessellationShader,
            .depthClamp = features.depthClamp,
            .fillModeNonSolid = features.fillModeNonSolid,
            .depthBounds = features.depthBounds,
            .wideLines = features.wideLines,
            .samplerAnisotropy = features.samplerAnisotropy,
            .shaderInt64 = features.shaderInt64,
            .shaderInt16 = features.shaderInt16,
        },
    };
    const float priority = 1.0f;
    const vk::DeviceQueueCreateInfo queue_info{
        .queueFamilyIndex = family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const auto [device_result, device] = physical.createDevice(vk::DeviceCreateInfo{
        .pNext = &features2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = static_cast<u32>(enabled.size()),
        .ppEnabledExtensionNames = enabled.data(),
    });
    if (device_result != vk::Result::eSuccess) {
        return fail("device creation failed: " + vk::to_string(device_result));
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
    const VmaVulkanFunctions functions{
        .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
    };
    const VmaAllocatorCreateInfo allocator_info{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physical,
        .device = device,
        .pVulkanFunctions = &functions,
        .instance = instance,
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };
    VmaAllocator allocator{};
    if (vmaCreateAllocator(&allocator_info, &allocator) != VK_SUCCESS) {
        device.destroy();
        return fail("VMA failed");
    }
    const auto chain = physical.getProperties2<vk::PhysicalDeviceProperties2,
                                               vk::PhysicalDevicePushDescriptorPropertiesKHR>();
    const u32 max_push_descriptors =
        chain.get<vk::PhysicalDevicePushDescriptorPropertiesKHR>().maxPushDescriptors;
    auto* kept_instance = new vk::UniqueInstance(std::move(owned_instance));
    VulkanContext context{
        .instance = instance,
        .physical_device = physical,
        .device = device,
        .allocator = allocator,
        .queue_family = family,
        .queue = device.getQueue(family, 0),
        .submit_mutex = &standalone_submit_mutex,
        .external_memory_host = host_import,
        .geometry = features.geometryShader != 0,
        .tessellation = features.tessellationShader != 0,
        .depth_bounds = features.depthBounds != 0,
        .depth_clip_control = false,
        .max_push_descriptors = max_push_descriptors,
        .destroy =
            [device, allocator, kept_instance] {
                vmaDestroyAllocator(allocator);
                device.destroy();
                delete kept_instance;
            },
    };
    return std::make_unique<VulkanDevice>(std::move(context));
}

} // namespace Native::Rhi
