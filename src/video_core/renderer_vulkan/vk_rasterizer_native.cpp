// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-100: native mode (RYGT_NATIVE=1, ryGT builds). Draws, dispatches and copies are recorded
// with the native image registry and guest memory read in place, instead of the texture and
// buffer caches, which keep images and buffers in step with guest memory draw by draw and are
// most of the recorder's cost. Version 0 records on the recorder thread, in GPU order, into a
// secondary command buffer; before anything else goes into the scheduler's command buffer (a
// submit, a flip) that list is ended and run there, so fences, labels and presenting work as
// before. Images stay in the general layout. Synchronisation is by global memory barriers: when
// a pass starts with other targets, before a draw reads a picture written since the last
// barrier, and around dispatches and copies. Later versions record the game's render jobs on its
// own threads with this code (ryty_gpu/native/DESIGN.md, N4).

#include "video_core/renderer_vulkan/vk_rasterizer_native.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>

#include <fmt/ranges.h>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "core/memory.h"
#include "ryty_gpu/native/image_registry.h"
#include "ryty_gpu/native/job_walker.h"
#include "ryty_gpu/native/rhi/rhi_vulkan.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/renderer_vulkan/depth_attachment.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/resource_binding.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image_descriptor.h"
#include "video_core/texture_cache/sampler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {
vk::ImageViewType ConvertImageViewType(AmdGpu::ImageType type);
}

namespace Vulkan {
namespace {

bool EnvFlag(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] == '1';
}

/// Logs the first few times and then every 1000th.
bool Sample(std::atomic<u32>& counter) {
    const u32 n = ++counter;
    return n <= 8 || n % 1000 == 0;
}

// What vk_rasterizer.cpp's ClassifyDraw decides, from the registers alone.
enum class Filter { Draw, EliminateFastClear, FmaskDecompress, Resolve, NoPrimitive, Copy };

Filter Classify(const AmdGpu::Regs& regs) {
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        return Filter::EliminateFastClear;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        return Filter::FmaskDecompress;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        return Filter::Resolve;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return Filter::NoPrimitive;
    }
    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const bool depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const bool stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        return Filter::Copy;
    }
    return Filter::Draw;
}

// As vk_rasterizer.cpp's MakeUserData.
Shader::PushData UserData(const AmdGpu::Regs& regs) {
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

// As vk_rasterizer.cpp's GetDrawOffsets.
std::pair<u32, u32> DrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
                                const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

// PERF-028's quad corners (vk_rasterizer.cpp).
constexpr std::array<u32, 6> QuadFirstVertex{0, 1, 2, 0, 2, 3};
constexpr std::array<u32, 6> QuadLastVertex{0, 1, 3, 1, 2, 3};
constexpr u64 MaxQuadIndexBytes = 16ULL << 20;

template <typename T>
u32 ExpandQuads(const T* in, u32 count, const std::optional<T> restart,
                const std::array<u32, 6>& corners, T* out) {
    u32 written = 0;
    if (!restart) {
        for (u32 quad = 0; quad + 4 <= count; quad += 4) {
            for (const u32 corner : corners) {
                out[written++] = in[quad + corner];
            }
        }
        return written;
    }
    std::array<T, 4> quad{};
    u32 filled = 0;
    for (u32 i = 0; i < count; ++i) {
        if (in[i] == *restart) {
            filled = 0;
            continue;
        }
        quad[filled++] = in[i];
        if (filled == 4) {
            for (const u32 corner : corners) {
                out[written++] = quad[corner];
            }
            filled = 0;
        }
    }
    return written;
}

// The texture cache's image usage, features and type (texture_cache/image.cpp), so native
// images are made as its images are.
vk::ImageUsageFlags UsageOf(const Instance& instance, const VideoCore::ImageInfo& info) {
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst |
                                vk::ImageUsageFlagBits::eSampled;
    if (!info.props.is_block) {
        if (info.props.is_depth) {
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
        } else {
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            if (info.num_samples == 1 || instance.IsMultisampleStorageImageSupported()) {
                usage |= vk::ImageUsageFlagBits::eStorage;
            }
        }
    } else {
        usage |= vk::ImageUsageFlagBits::eStorage;
    }
    return usage;
}

vk::FormatFeatureFlags2 FeaturesOf(vk::ImageUsageFlags usage) {
    vk::FormatFeatureFlags2 features{};
    if (usage & vk::ImageUsageFlagBits::eTransferSrc) {
        features |= vk::FormatFeatureFlagBits2::eTransferSrc;
    }
    if (usage & vk::ImageUsageFlagBits::eTransferDst) {
        features |= vk::FormatFeatureFlagBits2::eTransferDst;
    }
    if (usage & vk::ImageUsageFlagBits::eSampled) {
        features |= vk::FormatFeatureFlagBits2::eSampledImage;
    }
    if (usage & vk::ImageUsageFlagBits::eColorAttachment) {
        features |= vk::FormatFeatureFlagBits2::eColorAttachment;
    }
    if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        features |= vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    }
    return features;
}

vk::ImageType TypeOf(AmdGpu::ImageType type) {
    return type == AmdGpu::ImageType::Color3D ? vk::ImageType::e3D : vk::ImageType::e2D;
}

u32 RhiUsage(vk::ImageUsageFlags usage) {
    u32 result = 0;
    if (usage & vk::ImageUsageFlagBits::eSampled) {
        result |= Native::Rhi::ImageUsageSampled;
    }
    if (usage & vk::ImageUsageFlagBits::eStorage) {
        result |= Native::Rhi::ImageUsageStorage;
    }
    if (usage & vk::ImageUsageFlagBits::eColorAttachment) {
        result |= Native::Rhi::ImageUsageColorTarget;
    }
    if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        result |= Native::Rhi::ImageUsageDepthTarget;
    }
    if (usage & vk::ImageUsageFlagBits::eTransferSrc) {
        result |= Native::Rhi::ImageUsageTransferSrc;
    }
    if (usage & vk::ImageUsageFlagBits::eTransferDst) {
        result |= Native::Rhi::ImageUsageTransferDst;
    }
    return result | Native::Rhi::ImageUsageMutableFormat;
}

template <typename T>
T* Handle(u64 handle) {
    return reinterpret_cast<T*>(handle);
}

/// NAT-110: adds the time until the end of the scope to a counter.
struct PhaseClock {
    u64& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~PhaseClock() {
        total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count();
    }
};

constexpr u64 ChunkSize = 8_MB;
constexpr u32 SpliceAfter = 512; ///< commands per list before it is run and submitted

} // namespace

bool NativeModeRequested() {
    static const bool requested = EnvFlag("RYGT_NATIVE");
    return requested;
}

void NativeRenderer::DrawBindings::Clear() {
    writes.clear();
    buffers.clear();
    images.clear();
    push = {};
    sampled.clear();
    stored.clear();
    writes_buffers = false;
    failed = false;
}

NativeRenderer::NativeRenderer(Rasterizer& rasterizer) : r{rasterizer} {
    device = Native::Rhi::CreateVulkanDevice(r.instance);
    ASSERT_MSG(device, "NAT-100: no native device");
    const VkSemaphore semaphore = r.scheduler.GetWorkSemaphore()->Handle();
    timeline = device->ImportFence(reinterpret_cast<u64>(semaphore));
    device->UseExternalTimeline(timeline.get(), [this] { return r.scheduler.CurrentTick(); });
    registry = std::make_unique<Native::ImageRegistry>(*device);
    registry->SetImageFactory([this](const VideoCore::ImageInfo& info, bool render_target) {
        return MakeImage(info, render_target);
    });
    registry->SetMemoryClamp([this](VAddr address, u64 size) -> u64 {
        // Mapped, and readable by the CPU (GPU-only memory is not: NAT-106).
        if (!r.memory->IsValidMapping(address)) {
            return 0;
        }
        const u64 mapped = r.memory->ClampRangeSize(address, size);
        return Native::ReadablePrefix(reinterpret_cast<const void*>(address), mapped);
    });
    registry->SetUploadSource([this](const Native::ImageEntry& entry,
                                     Native::Rhi::CommandList& list) {
        // NAT-107: textures are uploaded from guest memory as the GPU sees it, in GPU order.
        Native::ImageRegistry::UploadSource source;
        auto* direct = r.buffer_cache.GetDirectMemory();
        const VAddr address = entry.info.guest_address;
        const u64 size = entry.info.guest_size;
        if (!direct || size == 0 || !direct->EnsureImported(address, size)) {
            return source;
        }
        if (const auto found = direct->FindBuffer(address, size);
            found && found->second % r.instance.StorageMinAlignment() == 0) {
            auto wrapped = device->WrapBuffer(
                reinterpret_cast<u64>(static_cast<VkBuffer>(found->first->Handle())),
                found->first->SizeBytes());
            source.buffer = wrapped.get();
            source.offset = found->second;
            source.keep.push_back(std::move(wrapped));
            return source;
        }
        auto gathered = device->CreateBuffer({.size = size,
                                              .usage = Native::Rhi::BufferUsageStorage |
                                                       Native::Rhi::BufferUsageTransferSrc |
                                                       Native::Rhi::BufferUsageTransferDst,
                                              .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                              .name = "native upload gather"});
        if (!gathered) {
            return source;
        }
        const vk::CommandBuffer cmd{Handle<VkCommandBuffer_T>(list.NativeHandle())};
        const vk::Buffer dst{Handle<VkBuffer_T>(gathered->NativeHandle())};
        const bool all = direct->ForEachPart(
            address, size, [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
                cmd.copyBuffer(view.Handle(), dst,
                               vk::BufferCopy{.srcOffset = view_offset, .dstOffset = offset,
                                              .size = bytes});
            });
        if (all) {
            list.Barrier(Native::Rhi::ScopeTransfer,
                         Native::Rhi::ScopeCompute | Native::Rhi::ScopeTransfer);
            source.buffer = gathered.get();
        }
        source.keep.push_back(std::move(gathered));
        return source;
    });
    registry->SetUploadHook([this](Native::ImageEntry& entry) {
        // NAT-105: CPU writes to a texture's memory from now on fault once, and the texture
        // is uploaded again.
        WatchPages(entry.info.guest_address, std::max<u64>(entry.info.guest_size, 1));
    });
    gds = device->CreateBuffer({.size = 64_KB,
                                .usage = Native::Rhi::BufferUsageStorage |
                                         Native::Rhi::BufferUsageTransferSrc |
                                         Native::Rhi::BufferUsageTransferDst,
                                .memory = Native::Rhi::MemoryKind::Upload,
                                .name = "native gds"});
    ASSERT_MSG(gds && gds->Mapped(), "NAT-100: no GDS buffer");
    std::memset(gds->Mapped(), 0, gds->Size());
    LOG_INFO(Render_Vulkan, "NAT-100: native mode: draws, dispatches and copies are recorded with "
                            "native images and guest memory read in place");
}

NativeRenderer::~NativeRenderer() {
    std::scoped_lock lock{mutex};
    const vk::Device vk_device = r.instance.GetDevice();
    for (const auto pool : free_pools) {
        vk_device.destroyDescriptorPool(pool);
    }
    for (const auto& [tick, pool] : retiring_pools) {
        vk_device.destroyDescriptorPool(pool);
    }
    for (const auto pool : stream.pools) {
        vk_device.destroyDescriptorPool(pool);
    }
    views.clear();
    samplers.clear();
}

Native::MadeImage NativeRenderer::MakeImage(const VideoCore::ImageInfo& info, bool) {
    Native::MadeImage made{};
    if (info.pixel_format == vk::Format::eUndefined) {
        return made;
    }
    const Instance& instance = r.instance;
    vk::ImageCreateFlags flags{vk::ImageCreateFlagBits::eMutableFormat |
                               vk::ImageCreateFlagBits::eExtendedUsage};
    if (info.props.is_volume) {
        flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
        if (instance.Is2dViewOf3dSupported()) {
            flags |= vk::ImageCreateFlagBits::e2DViewCompatibleEXT;
        }
    }
    if (info.props.is_block && instance.IsBlockTexelViewSupported()) {
        flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
    }
    const vk::ImageUsageFlags usage = UsageOf(instance, info);
    const vk::FormatFeatureFlags2 features = FeaturesOf(usage);
    const vk::Format format = instance.GetSupportedFormat(info.pixel_format, features);
    const vk::ImageType type = TypeOf(info.type);
    const auto properties =
        instance.GetPhysicalDevice().getImageFormatProperties2(vk::PhysicalDeviceImageFormatInfo2{
            .format = format,
            .type = type,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = usage,
            .flags = flags,
        });
    if (properties.result != vk::Result::eSuccess) {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_ERROR(Render_Vulkan, "NAT-100: no image of format {} ({}x{}) at {:#x}",
                      vk::to_string(format), info.size.width, info.size.height,
                      info.guest_address);
        }
        return made;
    }
    const vk::ImageCreateInfo image_ci{
        .flags = flags,
        .imageType = type,
        .format = format,
        .extent{info.size.width, info.size.height, std::max(info.size.depth, 1u)},
        .mipLevels = std::max<u32>(info.resources.levels, 1),
        .arrayLayers = type == vk::ImageType::e3D ? 1u : std::max<u32>(info.resources.layers, 1),
        .samples = LiverpoolToVK::NumSamples(
            info.num_samples, properties.value.imageFormatProperties.sampleCounts),
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    Native::Rhi::ImageDesc desc{};
    desc.dim = type == vk::ImageType::e3D ? Native::Rhi::ImageDim::D3 : Native::Rhi::ImageDim::D2;
    desc.format = Native::Rhi::FormatFromVk(static_cast<u32>(format));
    desc.width = image_ci.extent.width;
    desc.height = image_ci.extent.height;
    desc.depth = image_ci.extent.depth;
    desc.mips = image_ci.mipLevels;
    desc.layers = image_ci.arrayLayers;
    desc.samples = static_cast<u32>(image_ci.samples);
    desc.usage = RhiUsage(usage);
    made.image = Native::Rhi::CreateVulkanImage(
        *device, static_cast<const VkImageCreateInfo&>(image_ci), desc);
    made.usage = static_cast<u32>(static_cast<VkImageUsageFlags>(usage));
    made.features = static_cast<u64>(static_cast<VkFormatFeatureFlags2>(features));
    return made;
}

vk::ImageView NativeRenderer::View(const Native::ImageEntry& entry,
                                   const VideoCore::ImageViewInfo& requested, ViewUse use) {
    std::scoped_lock lock{views_mutex};
    auto& records = views[&entry];
    for (const auto& record : records) {
        if (record.use == use && record.info == requested) {
            return *record.view;
        }
    }
    // As the texture cache's ImageView (texture_cache/image_view.cpp), on the native image.
    const auto& image_desc = entry.image->Desc();
    const vk::Image image{Handle<VkImage_T>(entry.image->NativeHandle())};
    const vk::Format image_format = static_cast<vk::Format>(image_desc.format);
    const bool volume = image_desc.dim == Native::Rhi::ImageDim::D3;
    VideoCore::ImageViewInfo info = requested;
    // A view never reaches past the image (a T# may name more levels or layers than the
    // layout the image was made for).
    const u32 mips = std::max<u32>(image_desc.mips, 1);
    const u32 layers = volume ? 1u : std::max<u32>(image_desc.layers, 1);
    info.range.base.level = std::min<u32>(info.range.base.level, mips - 1);
    info.range.extent.levels =
        std::clamp<u32>(info.range.extent.levels, 1, mips - info.range.base.level);
    info.range.base.layer = std::min<u32>(info.range.base.layer, layers - 1);
    info.range.extent.layers =
        std::clamp<u32>(info.range.extent.layers, 1, layers - info.range.base.layer);

    vk::ImageUsageFlags usage{entry.backend_usage};
    if (use != ViewUse::Storage) {
        usage &= ~vk::ImageUsageFlagBits::eStorage;
    }
    vk::ImageViewUsageCreateInfo usage_ci{.usage = usage};
    vk::ImageViewMinLodCreateInfoEXT min_lod_ci{};
    if (info.min_lod != 0 && r.instance.IsImageViewMinLodSupported()) {
        const float last_level =
            static_cast<float>(info.range.base.level + info.range.extent.levels - 1);
        min_lod_ci.minLod = std::min(static_cast<float>(info.min_lod) / 256.f, last_level);
        usage_ci.pNext = &min_lod_ci;
    }
    vk::Format format = info.format;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
    if (entry.info.props.is_depth) {
        // A depth image: attachments see depth (and stencil); a T# names R32/R16 for depth and
        // R8 for stencil.
        const vk::ImageAspectFlags all =
            entry.info.props.has_stencil
                ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil
                : vk::ImageAspectFlagBits::eDepth;
        if (use == ViewUse::Attachment) {
            aspect = all;
        } else if (entry.info.props.has_stencil && !LiverpoolToVK::IsFormatDepthCompatible(format) &&
                   LiverpoolToVK::IsFormatStencilCompatible(format)) {
            aspect = vk::ImageAspectFlagBits::eStencil;
        } else {
            aspect = vk::ImageAspectFlagBits::eDepth;
        }
        format = image_format;
    } else {
        format = r.instance.GetSupportedFormat(format, vk::FormatFeatureFlags2{entry.backend_features});
    }
    vk::ImageViewType view_type = VideoCore::ConvertImageViewType(info.type);
    if (volume) {
        view_type = vk::ImageViewType::e3D;
    } else if (view_type == vk::ImageViewType::e3D) {
        view_type = info.range.extent.layers > 1 ? vk::ImageViewType::e2DArray
                                                 : vk::ImageViewType::e2D;
    } else if (view_type == vk::ImageViewType::e2D && info.range.extent.layers > 1) {
        view_type = vk::ImageViewType::e2DArray;
    }
    const vk::ImageViewCreateInfo view_ci{
        .pNext = &usage_ci,
        .image = image,
        .viewType = view_type,
        .format = format,
        .components = info.mapping == AmdGpu::IdentityMapping
                          ? vk::ComponentMapping{}
                          : LiverpoolToVK::ComponentMapping(info.mapping),
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = info.range.base.level,
            .levelCount = info.range.extent.levels,
            .baseArrayLayer = info.range.base.layer,
            .layerCount = info.range.extent.layers,
        },
    };
    auto [result, view] = r.instance.GetDevice().createImageViewUnique(view_ci);
    if (result != vk::Result::eSuccess) {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_ERROR(Render_Vulkan, "NAT-100: no view {} of image {:#x} ({})",
                      vk::to_string(format), entry.info.guest_address, vk::to_string(result));
        }
        return {};
    }
    const vk::ImageView handle = *view;
    records.push_back({requested, use, std::move(view)});
    return handle;
}

vk::Sampler NativeRenderer::SamplerFor(const AmdGpu::Sampler& sharp,
                                       AmdGpu::BorderColorBuffer border, bool is_depth) {
    const u64 hash = XXH3_64bits(&sharp, sizeof(sharp)) ^
                     (is_depth ? 0x9E3779B97F4A7C15ULL : 0) ^ (u64(border.Address()) << 1);
    std::scoped_lock lock{sampler_mutex};
    auto& slot = samplers[hash];
    if (!slot) {
        slot = std::make_unique<VideoCore::Sampler>(r.instance, sharp, border, is_depth);
    }
    return slot->Handle();
}

NativeRenderer::Stream& NativeRenderer::Begin() {
    Stream& s = stream;
    if (s.list) {
        return s;
    }
    s.list = device->AcquireCommandList(Native::Rhi::CommandListLevel::Secondary);
    s.list->Begin();
    s.cmd = vk::CommandBuffer{Handle<VkCommandBuffer_T>(s.list->NativeHandle())};
    s.rendering = false;
    s.dynamic.Invalidate();
    s.pipeline = vk::Pipeline{};
    s.commands = 0;
    // What was recorded before this list (earlier lists, the presenter) is finished first.
    GlobalBarrier();
    return s;
}

void NativeRenderer::EndRendering() {
    if (stream.rendering) {
        stream.cmd.endRendering();
        stream.rendering = false;
    }
}

void NativeRenderer::BeginRendering(const RenderState& state) {
    Stream& s = stream;
    if (s.rendering && s.render == state) {
        return;
    }
    EndRendering();
    // As Scheduler::BeginRendering.
    std::array<vk::RenderingAttachmentInfo, 8> colors;
    for (u32 i = 0; i < state.num_color_attachments; ++i) {
        const auto& cb = state.color_attachments[i];
        colors[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = vk::ImageLayout::eGeneral,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }
    const auto& db = state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth{
        .imageView = db.image_view,
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{
                                         .depth = std::bit_cast<float>(db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil{
        .imageView = db.image_view,
        .imageLayout = vk::ImageLayout::eGeneral,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };
    s.cmd.beginRendering(vk::RenderingInfo{
        .renderArea{.offset = {0, 0}, .extent = {state.width, state.height}},
        .layerCount = state.num_layers,
        .colorAttachmentCount = state.num_color_attachments,
        .pColorAttachments = colors.data(),
        .pDepthAttachment = db.has_depth ? &depth : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil : nullptr,
    });
    s.rendering = true;
    s.render = state;
    ++stats.passes;
}

void NativeRenderer::GlobalBarrier() {
    Stream& s = stream;
    EndRendering();
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
    s.written.clear();
    s.buffers_written = false;
    ++stats.barriers;
    FlushWritebacks();
}

void NativeRenderer::FlushWritebacks() {
    if (writebacks.empty() || !stream.list) {
        return;
    }
    Stream& s = stream;
    auto* direct = r.buffer_cache.GetDirectMemory();
    const u64 alignment = r.instance.StorageMinAlignment();
    for (auto* entry : writebacks) {
        const VAddr address = entry->info.guest_address;
        const u64 size = entry->info.guest_size;
        if (!direct || !direct->EnsureImported(address, size)) {
            continue;
        }
        const auto found = direct->FindBuffer(address, size);
        if (!found || found->second % alignment != 0) {
            continue;
        }
        auto out = device->WrapBuffer(
            reinterpret_cast<u64>(static_cast<VkBuffer>(found->first->Handle())),
            found->first->SizeBytes());
        if (registry->RecordDownload(*s.list, *entry, *out, found->second)) {
            ++stats.writebacks;
        }
        s.keep.push_back(std::move(out));
    }
    writebacks.clear();
    s.pipeline = vk::Pipeline{}; // the re-tiling binds its own pipeline
    // What reads guest memory next (shaders, the CPU after the GPU) sees the pictures.
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
}

void NativeRenderer::PrepareImages() {
    const bool init = device->PendingImageInit();
    const bool uploads = registry->HasPendingUploads();
    if (!init && !uploads) {
        return;
    }
    Stream& s = stream;
    EndRendering();
    if (init) {
        device->RecordPendingImageInit(*s.list);
    }
    if (uploads) {
        GlobalBarrier(); // NAT-107: uploads read guest memory as earlier GPU work left it
        registry->RecordUploads(*s.list);
        s.pipeline = vk::Pipeline{}; // the uploads bind their own pipelines
    }
}

void NativeRenderer::ChunkFlushCheck() {
    // Long lists are run and submitted as they grow, so the GPU starts on them (as the old
    // path's SubmitChunkIfNeeded does).
    if (stream.commands >= SpliceAfter) {
        Splice();
        r.scheduler.Flush();
    }
}

NativeRenderer::Chunk NativeRenderer::TakeChunk(u64 size) {
    auto* semaphore = r.scheduler.GetWorkSemaphore();
    std::scoped_lock lock{chunk_mutex};
    if (free_chunks.empty() && !retiring_chunks.empty()) {
        semaphore->Refresh();
    }
    const u64 known = semaphore->KnownGpuTick();
    while (!retiring_chunks.empty() && retiring_chunks.front().retire_tick <= known) {
        if (retiring_chunks.front().buffer->Size() == ChunkSize) {
            free_chunks.push_back(std::move(retiring_chunks.front()));
        }
        retiring_chunks.pop_front(); // larger ones are freed (after the GPU, by the device)
    }
    if (size <= ChunkSize && !free_chunks.empty()) {
        Chunk chunk = std::move(free_chunks.back());
        free_chunks.pop_back();
        return chunk;
    }
    Chunk chunk;
    chunk.buffer = device->CreateBuffer({
        .size = std::max(size, ChunkSize),
        .usage = Native::Rhi::BufferUsageStorage | Native::Rhi::BufferUsageUniform |
                 Native::Rhi::BufferUsageIndex | Native::Rhi::BufferUsageVertex |
                 Native::Rhi::BufferUsageIndirect | Native::Rhi::BufferUsageTransferSrc,
        .memory = Native::Rhi::MemoryKind::Upload,
        .name = "native upload",
    });
    ASSERT_MSG(chunk.buffer && chunk.buffer->Mapped(), "NAT-100: no upload memory ({} bytes)",
               size);
    return chunk;
}

std::tuple<vk::Buffer, u64, u8*> NativeRenderer::Allocate(u64 size, u64 align) {
    Stream& s = stream;
    align = std::max<u64>(align, 16);
    if (s.chunks.empty() ||
        Common::AlignUp(s.chunk_used, align) + size > s.chunks.back().buffer->Size()) {
        s.chunks.push_back(TakeChunk(size));
        s.chunk_used = 0;
    }
    const auto& buffer = *s.chunks.back().buffer;
    const u64 offset = Common::AlignUp(s.chunk_used, align);
    s.chunk_used = offset + size;
    return {vk::Buffer{Handle<VkBuffer_T>(buffer.NativeHandle())}, offset, buffer.Mapped() + offset};
}

vk::DescriptorPool NativeRenderer::TakeDescriptorPool() {
    const vk::Device vk_device = r.instance.GetDevice();
    {
        std::scoped_lock lock{chunk_mutex};
        const u64 known = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
        while (!retiring_pools.empty() && retiring_pools.front().first <= known) {
            std::ignore = vk_device.resetDescriptorPool(retiring_pools.front().second);
            free_pools.push_back(retiring_pools.front().second);
            retiring_pools.pop_front();
        }
        if (!free_pools.empty()) {
            const auto pool = free_pools.back();
            free_pools.pop_back();
            return pool;
        }
    }
    static constexpr std::array<vk::DescriptorPoolSize, 4> sizes{{
        {vk::DescriptorType::eStorageBuffer, 8192},
        {vk::DescriptorType::eSampledImage, 8192},
        {vk::DescriptorType::eStorageImage, 1024},
        {vk::DescriptorType::eSampler, 4096},
    }};
    const auto [result, pool] = vk_device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
        .maxSets = 512,
        .poolSizeCount = static_cast<u32>(sizes.size()),
        .pPoolSizes = sizes.data(),
    });
    ASSERT_MSG(result == vk::Result::eSuccess, "NAT-100: no descriptor pool");
    return pool;
}

NativeRenderer::Region* NativeRenderer::RegionOver(VAddr address, u64 size) {
    auto it = regions.upper_bound(address);
    if (it != regions.begin()) {
        Region& region = std::prev(it)->second;
        if (address < region.end) {
            return &region;
        }
    }
    if (it != regions.end() && it->second.begin < address + size) {
        return &it->second;
    }
    return nullptr;
}

NativeRenderer::Region* NativeRenderer::MakeRegion(VAddr address, u64 size) {
    auto* direct = r.buffer_cache.GetDirectMemory();
    // The new region takes in every region it overlaps (their GPU-written contents kept).
    VAddr begin = Common::AlignDown(address, 4096);
    VAddr end = Common::AlignUp(address + size, 4096);
    std::vector<Region> merged;
    for (auto it = regions.begin(); it != regions.end();) {
        if (it->second.begin < end && it->second.end > begin) {
            begin = std::min(begin, it->second.begin);
            end = std::max(end, it->second.end);
            merged.push_back(std::move(it->second));
            it = regions.erase(it);
        } else {
            ++it;
        }
    }
    if (!direct || end - begin > 1_GB || !direct->EnsureImported(begin, end - begin)) {
        return nullptr;
    }
    Region region;
    region.begin = begin;
    region.end = end;
    region.buffer = device->CreateBuffer({.size = end - begin,
                                          .usage = Native::Rhi::BufferUsageStorage |
                                                   Native::Rhi::BufferUsageVertex |
                                                   Native::Rhi::BufferUsageIndex |
                                                   Native::Rhi::BufferUsageIndirect |
                                                   Native::Rhi::BufferUsageTransferSrc |
                                                   Native::Rhi::BufferUsageTransferDst,
                                          .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                          .name = "native region"});
    if (!region.buffer) {
        return nullptr;
    }
    Stream& s = Begin();
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(region.buffer->NativeHandle())};
    // Guest memory first, then what the GPU wrote into the regions taken in.
    direct->ForEachPart(begin, end - begin,
                        [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
                            s.cmd.copyBuffer(view.Handle(), dst,
                                             vk::BufferCopy{.srcOffset = view_offset,
                                                            .dstOffset = offset,
                                                            .size = bytes});
                        });
    for (auto& old : merged) {
        s.cmd.copyBuffer(vk::Buffer{Handle<VkBuffer_T>(old.buffer->NativeHandle())}, dst,
                         vk::BufferCopy{.srcOffset = 0,
                                        .dstOffset = old.begin - begin,
                                        .size = old.end - old.begin});
        s.keep.push_back(std::move(old.buffer)); // freed once the list ran
    }
    s.buffers_written = true;
    // CPU writes to it from now on are copied in before its next use.
    WatchPages(begin, end - begin);
    ++stats.regions_made;
    LOG_INFO(Render_Vulkan, "NAT-114: region {:#x}+{:#x} on the GPU ({} taken in)", begin,
             end - begin, merged.size());
    auto [it, inserted] = regions.emplace(begin, std::move(region));
    return &it->second;
}

void NativeRenderer::SortRegionWrites() {
    if (!has_pending_region_writes.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    std::vector<std::pair<VAddr, VAddr>> writes;
    {
        std::scoped_lock lock{region_dirty_mutex};
        writes.swap(pending_region_writes);
    }
    for (const auto& [begin, end] : writes) {
        for (auto& [base, region] : regions) {
            if (region.begin < end && region.end > begin) {
                for (VAddr page = std::max(begin, region.begin); page < std::min(end, region.end);
                     page += 4096) {
                    region.dirty_pages.push_back(page);
                }
            }
        }
    }
}

void NativeRenderer::RefreshRegion(Region& region) {
    SortRegionWrites();
    std::vector<VAddr> pages;
    pages.swap(region.dirty_pages);
    if (pages.empty()) {
        return;
    }
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct) {
        return;
    }
    Stream& s = Begin();
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(region.buffer->NativeHandle())};
    std::ranges::sort(pages);
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    for (const VAddr page : pages) {
        if (page < region.begin || page >= region.end) {
            continue;
        }
        direct->ForEachPart(page, std::min<u64>(4096, region.end - page),
                            [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                u64 bytes) {
                                s.cmd.copyBuffer(view.Handle(), dst,
                                                 vk::BufferCopy{.srcOffset = view_offset,
                                                                .dstOffset = page - region.begin +
                                                                             offset,
                                                                .size = bytes});
                            });
    }
    s.buffers_written = true;
    WatchPages(region.begin, region.end - region.begin);
    ++stats.region_refreshes;
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::GuestBuffer(VAddr address, u64 size,
                                                                      bool write) {
    if (size == 0) {
        return std::nullopt;
    }
    // NAT-114: ranges in (or reaching into) a region use it.
    if (Region* region = RegionOver(address, size)) {
        if (address < region->begin || address + size > region->end) {
            region = MakeRegion(address, size);
        }
        if (region) {
            RefreshRegion(*region);
            ++stats.region_binds;
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
    }
    if (auto* direct = r.buffer_cache.GetDirectMemory();
        direct && direct->EnsureImported(address, size)) {
        if (const auto found = direct->FindBuffer(address, size)) {
            return std::pair{found->first->Handle(), found->second};
        }
    }
    if (!write) {
        if (const auto gathered = GatherOnGpu(address, size)) {
            return gathered;
        }
    }
    if (write) {
        if (Region* region = MakeRegion(address, size)) {
            ++stats.region_binds;
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
        ++stats.unbound;
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan,
                        "NAT-101: written range {:#x}+{:#x} is not in one imported view; bound "
                        "empty",
                        address, size);
        }
        return std::nullopt;
    }
    if (size > 64_MB) {
        ++stats.unbound;
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan,
                        "NAT-101: read range {:#x}+{:#x} is not in one imported view and too "
                        "large to copy; bound empty",
                        address, size);
        }
        return std::nullopt;
    }
    // A snapshot of what the range holds now, in the list's upload memory (zeros where the CPU
    // can't read it: NAT-106).
    const auto [buffer, offset, data] = Allocate(size, r.instance.StorageMinAlignment());
    const u64 readable = Native::ReadablePrefix(reinterpret_cast<const void*>(address), size);
    std::memcpy(data, reinterpret_cast<const void*>(address), readable);
    if (readable < size) {
        std::memset(data + readable, 0, size - readable);
    }
    ++stats.copied_reads;
    stats.copied_bytes += size;
    return std::pair{buffer, offset};
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::GatherOnGpu(VAddr address, u64 size) {
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct || size > 256_MB || !direct->EnsureImported(address, size)) {
        return std::nullopt;
    }
    auto gathered = device->CreateBuffer({.size = size,
                                          .usage = Native::Rhi::BufferUsageStorage |
                                                   Native::Rhi::BufferUsageVertex |
                                                   Native::Rhi::BufferUsageIndex |
                                                   Native::Rhi::BufferUsageIndirect |
                                                   Native::Rhi::BufferUsageTransferDst,
                                          .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                          .name = "native gather"});
    if (!gathered) {
        return std::nullopt;
    }
    Stream& s = Begin();
    // Copies run outside rendering, after what earlier work wrote; the draw waits for them
    // (the next barrier: they count as written buffers).
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(gathered->NativeHandle())};
    const bool all = direct->ForEachPart(
        address, size, [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset, u64 bytes) {
            s.cmd.copyBuffer(view.Handle(), dst,
                             vk::BufferCopy{.srcOffset = view_offset, .dstOffset = offset,
                                            .size = bytes});
        });
    s.buffers_written = true;
    s.keep.push_back(std::move(gathered));
    if (!all) {
        return std::nullopt;
    }
    ++stats.gathers;
    stats.gathered_bytes += size;
    return std::pair{dst, u64{0}};
}

bool NativeRenderer::ResolveTargets(const AmdGpu::Regs& regs, const GraphicsPipeline& pipeline,
                                    Targets& out) {
    const auto& key = pipeline.GetGraphicsKey();
    RenderState& state = out.state;
    state = {};
    out.colors = {};
    out.depth = nullptr;
    u32 width = r.instance.GetMaxFramebufferWidth();
    u32 height = r.instance.GetMaxFramebufferHeight();
    u32 num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = static_cast<u16>(std::bit_width(key.mrt_mask));
    const bool skip_cb = regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        auto& attachment = state.color_attachments[cb];
        attachment = {};
        attachment.image_layout = vk::ImageLayout::eGeneral;
        const auto& col_buf = regs.color_buffers[cb];
        if (skip_cb || !col_buf || !regs.color_target_mask.GetMask(cb) ||
            (key.mrt_mask & (1u << cb)) == 0) {
            continue;
        }
        const VideoCore::TextureCache::ImageDesc desc{col_buf, r.liverpool->DrawCbExtent(cb)};
        Native::ImageEntry* entry = registry->Find(desc.info, true, frame);
        if (!entry) {
            ++stats.null_images;
            continue;
        }
        const vk::ImageView view = View(*entry, desc.view_info, ViewUse::Attachment);
        if (!view) {
            continue;
        }
        out.colors[cb] = entry;
        const u32 mip = desc.view_info.range.base.level;
        width = std::min<u32>(width, std::max(entry->info.size.width >> mip, 1u));
        height = std::min<u32>(height, std::max(entry->info.size.height >> mip, 1u));
        num_layers = std::min<u32>(num_layers, desc.view_info.range.extent.layers);
        const VAddr cmask = col_buf.CmaskAddress();
        bool is_clear = false;
        if (cmask != 0) {
            known_meta.insert(cmask);
            is_clear = cleared_meta.erase(cmask) != 0;
        }
        attachment.image_view = view;
        attachment.clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf).color.uint32
                     : std::array<u32, 4>{};
        attachment.is_clear = is_clear;
    }
    auto& db = state.depth_stencil_attachment;
    db = {};
    db.image_layout = vk::ImageLayout::eGeneral;
    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const VAddr htile = regs.depth_htile_data_base.GetAddress();
        const VideoCore::TextureCache::ImageDesc desc{regs.depth_buffer, regs.depth_view,
                                                      regs.depth_control, htile,
                                                      r.liverpool->DrawDbExtent()};
        Native::ImageEntry* entry = registry->Find(desc.info, true, frame);
        const vk::ImageView view =
            entry ? View(*entry, desc.view_info, ViewUse::Attachment) : vk::ImageView{};
        if (view) {
            bool meta_clear = false;
            if (htile != 0) {
                known_meta.insert(htile);
                meta_clear = cleared_meta.erase(htile) != 0;
            }
            const auto aspects =
                DepthStencilClearAspects(regs.depth_buffer, regs.depth_control,
                                         regs.depth_render_control, meta_clear);
            const bool depth_clear = bool(aspects & vk::ImageAspectFlagBits::eDepth);
            const bool stencil_clear = bool(aspects & vk::ImageAspectFlagBits::eStencil);
            db.image_view = view;
            if (regs.depth_buffer.DepthValid()) {
                db.clear_value[0] = depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
                db.has_depth = true;
                db.depth_clear = depth_clear;
            }
            if (regs.depth_buffer.StencilValid() && entry->info.props.has_stencil) {
                db.clear_value[1] = stencil_clear ? regs.stencil_clear : 0u;
                db.has_stencil = true;
                db.stencil_clear = stencil_clear;
            }
            width = std::min<u32>(width, entry->info.size.width);
            height = std::min<u32>(height, entry->info.size.height);
            num_layers = std::min<u32>(num_layers, desc.view_info.range.extent.layers);
            out.depth = entry;
        } else {
            ++stats.null_images;
        }
    }
    state.width = static_cast<u16>(std::min<u32>(width, 0xffff));
    state.height = static_cast<u16>(std::min<u32>(height, 0xffff));
    state.num_layers = static_cast<u16>(
        num_layers == std::numeric_limits<u16>::max() ? 1u : std::max(num_layers, 1u));
    return true;
}

void NativeRenderer::BindStages(std::span<const Shader::Info* const> stages,
                                const AmdGpu::Regs& regs, bool compute, DrawBindings& b) {
    size_t total = 0;
    for (const auto* stage : stages) {
        if (stage) {
            total += stage->buffers.size() + stage->images.size() + stage->samplers.size();
        }
    }
    b.writes.reserve(total);
    Shader::Backend::Bindings binding{};
    std::array<StageResourceBindings, Shader::MaxStageTypes> plans{};
    for (u32 i = 0; i < stages.size(); ++i) {
        const auto* stage = stages[i];
        if (!stage) {
            continue;
        }
        plans[i] = PlanStageResourceBindings(*stage, binding);
        auto buffer_binding = plans[i].buffers;
        PhaseClock clock{stats.phase_ns[2]};
        BindBuffers(*stage, regs, compute, buffer_binding, b);
        binding = plans[i].next;
    }
    for (u32 i = 0; i < stages.size(); ++i) {
        if (const auto* stage = stages[i]) {
            auto texture_binding = plans[i].textures;
            PhaseClock clock{stats.phase_ns[3]};
            BindTextures(*stage, regs, texture_binding, b);
        }
    }
}

void NativeRenderer::BindBuffers(const Shader::Info& stage, const AmdGpu::Regs& regs, bool compute,
                                 Shader::Backend::Bindings& binding, DrawBindings& b) {
    const u64 alignment = r.instance.StorageMinAlignment();
    auto* direct = stage.uses_direct ? r.buffer_cache.GetDirectMemory() : nullptr;
    // D3's table: page table L1 and zero page, then base, size and valid of each buffer read
    // through it.
    static thread_local std::vector<std::array<u32, 4>> direct_entries;
    direct_entries.assign(stage.buffers.size(), {});
    constexpr u64 GpuAddressLimit = 1ULL << 40;
    u32 index = 0;
    for (const auto& desc : stage.buffers) {
        const u32 buffer_index = index++;
        vk::DescriptorBufferInfo info{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
        if (desc.IsSpecial()) {
            switch (desc.buffer_type) {
            case Shader::BufferType::GdsBuffer:
                info = {Handle<VkBuffer_T>(gds->NativeHandle()), 0, gds->Size()};
                break;
            case Shader::BufferType::Flatbuf: {
                const u32 size = static_cast<u32>(stage.flattened_ud_buf.size() * sizeof(u32));
                if (size != 0) {
                    const auto [buffer, offset, data] = Allocate(size, alignment);
                    std::memcpy(data, stage.flattened_ud_buf.data(), size);
                    info = {buffer, offset, size};
                }
                break;
            }
            case Shader::BufferType::ClipPlanes:
                if (regs.clipper_control.user_clip_plane_enable != 0) {
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const auto [buffer, offset, data] = Allocate(sizeof(planes), alignment);
                    std::memcpy(data, planes.data(), sizeof(planes));
                    info = {buffer, offset, sizeof(planes)};
                }
                break;
            case Shader::BufferType::DirectTable: {
                const u32 words = static_cast<u32>(4 + stage.buffers.size() * 4);
                const auto [buffer, offset, data] = Allocate(words * sizeof(u32), alignment);
                auto* table = reinterpret_cast<u32*>(data);
                std::memset(table, 0, words * sizeof(u32));
                if (auto* memory = r.buffer_cache.GetDirectMemory()) {
                    const u64 l1 = memory->L1Address();
                    const u64 zero = memory->ZeroPageAddress();
                    table[0] = u32(l1);
                    table[1] = u32(l1 >> 32);
                    table[2] = u32(zero);
                    table[3] = u32(zero >> 32);
                    for (size_t i = 0; i < stage.buffers.size(); ++i) {
                        std::memcpy(&table[4 + i * 4], direct_entries[i].data(), 16);
                    }
                }
                info = {buffer, offset, words * sizeof(u32)};
                break;
            }
            case Shader::BufferType::SharedMemory: {
                const auto& cs_program = r.liverpool->DrawCsRegs();
                const u64 lds_size =
                    std::max<u64>(u64(cs_program.SharedMemSize()) * cs_program.NumWorkgroups(), 16);
                // GCN LDS is undefined when a workgroup starts: one GPU-only buffer serves every
                // dispatch (they are separated by barriers).
                if (!lds || lds->Size() < lds_size) {
                    lds = device->CreateBuffer({
                        .size = std::max<u64>(Common::AlignUp(lds_size, 1_MB), 4_MB),
                        .usage = Native::Rhi::BufferUsageStorage,
                        .memory = Native::Rhi::MemoryKind::DeviceLocal,
                        .name = "native lds",
                    });
                }
                if (lds) {
                    info = {Handle<VkBuffer_T>(lds->NativeHandle()), 0, lds_size};
                }
                break;
            }
            default: {
                static std::atomic<u32> logged{};
                if (Sample(logged)) {
                    LOG_WARNING(Render_Vulkan,
                                "NAT-101: shader {:#x} uses buffer type {}, not supported in "
                                "native mode; bound empty",
                                stage.pgm_hash, u32(desc.buffer_type));
                }
                break;
            }
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            const u64 base = vsharp.base_address;
            const u64 sharp_size = vsharp.GetSize();
            // FIX-031 / FIX-036 / FIX-046: V#s that are not buffers are bound empty.
            const bool impossible =
                vsharp.num_records != UINT32_MAX &&
                (sharp_size >= GpuAddressLimit || base + sharp_size > GpuAddressLimit ||
                 (sharp_size >= 1_GB && base != 0 && !r.memory->IsValidMapping(base)));
            const bool unmapped =
                !impossible && base != 0 && sharp_size != 0 && !r.IsMappedStart(base);
            if (base != 0 && sharp_size != 0 && !impossible && !unmapped) {
                const u64 size = r.memory->ClampRangeSize(base, sharp_size);
                if (direct && desc.IsDirectEligible() && size < (u64{1} << 32) &&
                    !RegionOver(base, size) && direct->EnsureImported(base, size)) {
                    // D3: read in place through the page table; nothing bound.
                    direct_entries[buffer_index] = {u32(base), u32(base >> 32), u32(size), 1u};
                } else if (const auto found = GuestBuffer(base, size, desc.is_written)) {
                    const u64 aligned = Common::AlignDown(found->second, alignment);
                    const u64 adjust = found->second - aligned;
                    b.push.AddOffset(binding.buffer, static_cast<u32>(adjust));
                    info = {found->first, aligned, size + adjust};
                    b.writes_buffers |= desc.is_written;
                }
            }
        }
        b.buffers.push_back(info);
        b.writes.push_back(vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &b.buffers.back(),
        });
        ++binding.buffer;
    }
    (void)compute;
}

void NativeRenderer::BindTextures(const Shader::Info& stage, const AmdGpu::Regs& regs,
                                  Shader::Backend::Bindings& binding, DrawBindings& b) {
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        const u32 num_bindings = image_desc.NumBindings(stage);
        const u32 first = static_cast<u32>(b.images.size());
        const bool storage = image_desc.is_written;
        const auto add_write = [&] {
            b.writes.push_back(vk::WriteDescriptorSet{
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = binding.unified,
                .dstArrayElement = 0,
                .descriptorCount = num_bindings,
                .descriptorType =
                    storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage,
                .pImageInfo = &b.images[first],
            });
            binding.unified += num_bindings;
        };
        const auto bind_null = [&](u32 reason) {
            for (u32 i = 0; i < num_bindings; ++i) {
                b.images.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral});
            }
            ++stats.null_images;
            ++stats.null_reasons[reason];
            add_write();
        };
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            bind_null(0);
            continue;
        }
        if (!r.memory->IsValidGpuMapping(tsharp.Address(), 0)) {
            bind_null(1);
            continue;
        }
        if (LiverpoolToVK::TrySurfaceFormat(data_fmt, num_fmt) == vk::Format::eUndefined) {
            bind_null(2);
            continue;
        }
        if (VideoCore::CheckImageDescriptorGeometry(tsharp, r.instance.GetImageLimits()) !=
            VideoCore::ImageDescriptorGeometryError::None) {
            bind_null(3);
            continue;
        }
        const VideoCore::TextureCache::ImageDesc desc{tsharp, image_desc};
        // FIX-046: an image must start inside the guest address space. FIX-081: GT Sport's grass
        // shader reads garbage T#s from one slot when the texture is unused; from there only an
        // image the game already made is bound.
        if (!r.IsMappedStart(desc.info.guest_address)) {
            bind_null(4);
            continue;
        }
        if (stage.pgm_hash == 0xaa3822a3 && image_desc.sharp_fetch.offsets[0] == 109 &&
            !registry->HasImageAt(desc.info.guest_address)) {
            bind_null(5);
            continue;
        }
        Native::ImageEntry* entry = registry->Find(desc.info, storage, frame);
        if (!entry) {
            bind_null(6);
            continue;
        }
        if (!storage && entry->from_cpu) {
            const auto start = std::chrono::steady_clock::now();
            RefreshSources(*entry); // NAT-104
            // (part of the textures' time; reported apart)
            stats.phase_ns[4] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
        }
        const auto fallback = image_desc.mip_fallback_mode;
        for (u32 i = 0; i < num_bindings; ++i) {
            auto view_info = desc.view_info;
            if (fallback == Shader::MipStorageFallbackMode::ConstantIndex) {
                view_info.range.base.level += image_desc.constant_mip_index;
                view_info.range.extent.levels = 1;
            } else if (fallback == Shader::MipStorageFallbackMode::DynamicIndex) {
                view_info.range.base.level += i;
                view_info.range.extent.levels = 1;
            }
            const vk::ImageView view =
                View(*entry, view_info, storage ? ViewUse::Storage : ViewUse::Sampled);
            b.images.push_back({VK_NULL_HANDLE, view, vk::ImageLayout::eGeneral});
        }
        (storage ? b.stored : b.sampled).push_back(entry);
        add_write();
    }
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                regs.ta_bc_base.Address() == 0)) {
            ssharp = AmdGpu::Sampler{};
        }
        b.images.push_back({SamplerFor(ssharp, regs.ta_bc_base, sampler.is_depth), VK_NULL_HANDLE,
                            vk::ImageLayout::eGeneral});
        b.writes.push_back(vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = &b.images.back(),
        });
    }
}

void NativeRenderer::PushBindings(const Pipeline& pipeline, vk::PipelineBindPoint bind_point,
                                  DrawBindings& b) {
    Stream& s = stream;
    const CmdBuf cmdbuf{s.cmd};
    const auto stage_flags =
        pipeline.IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    cmdbuf.pushConstants(pipeline.GetLayout(), stage_flags, 0u, sizeof(b.push), &b.push);
    if (b.writes.empty()) {
        return;
    }
    if (pipeline.UsesPushDescriptors()) {
        cmdbuf.pushDescriptorSetKHR(bind_point, pipeline.GetLayout(), 0, b.writes);
        return;
    }
    const vk::Device vk_device = r.instance.GetDevice();
    const vk::DescriptorSetLayout layout = pipeline.DescriptorSetLayout();
    vk::DescriptorSet set{};
    for (u32 attempt = 0; attempt < 2 && !set; ++attempt) {
        if (s.pools.empty() || attempt == 1) {
            s.pools.push_back(TakeDescriptorPool());
        }
        vk::DescriptorSetAllocateInfo alloc{
            .descriptorPool = s.pools.back(),
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        VkDescriptorSet raw{};
        if (vk_device.allocateDescriptorSets(&alloc, reinterpret_cast<vk::DescriptorSet*>(&raw)) ==
            vk::Result::eSuccess) {
            set = raw;
        }
    }
    if (!set) {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_ERROR(Render_Vulkan, "NAT-100: no descriptor set for a draw");
        }
        return;
    }
    for (auto& write : b.writes) {
        write.dstSet = set;
    }
    vk_device.updateDescriptorSets(b.writes, {});
    cmdbuf.bindDescriptorSets(bind_point, pipeline.GetLayout(), 0, set, {});
}

/// A draw's vertex and index buffers: looked up before its pass starts (a lookup may gather on
/// the GPU, outside rendering) and bound inside it.
struct NativeRenderer::VertexIndexState {
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> input_bindings;
    VertexInputs<vk::Buffer> buffers;
    VertexInputs<vk::DeviceSize> offsets;
    VertexInputs<vk::DeviceSize> sizes;
    VertexInputs<vk::DeviceSize> strides;
    vk::Buffer index_buffer{};
    u64 index_offset = 0;
    vk::IndexType index_type = vk::IndexType::eUint32;
    u32 index_count = 0;
};

void NativeRenderer::ResolveVertexBuffers(const GraphicsPipeline& pipeline,
                                          const AmdGpu::Regs& regs, VertexIndexState& v) {
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(v.attributes, v.input_bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    for (const auto& guest : guest_buffers) {
        std::optional<std::pair<vk::Buffer, u64>> found;
        u64 size = 0;
        if (guest.base_address != 0 && guest.GetSize() > 0 &&
            r.memory->MappedAreaEnd(guest.base_address, guest.base_address + 1) != 0) {
            size = r.memory->ClampRangeSize(guest.base_address, guest.GetSize());
            found = GuestBuffer(guest.base_address, size, false);
        }
        v.buffers.push_back(found ? found->first : vk::Buffer{});
        v.offsets.push_back(found ? found->second : 0);
        v.sizes.push_back(found ? size : 0);
        v.strides.push_back(guest.GetStride());
    }
}

bool NativeRenderer::ResolveIndices(const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset,
                                    bool quad_triangles, VertexIndexState& v) {
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    v.index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    v.index_count = regs.num_indices;
    if (quad_triangles) {
        const auto& corners =
            regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last
                ? QuadLastVertex
                : QuadFirstVertex;
        const u32 count = regs.num_indices;
        const u32 max_indices = count / 4 * 6;
        v.index_count = 0;
        if (max_indices == 0) {
            return true;
        }
        if (!is_indexed) {
            const auto [buffer, offset, data] = Allocate(u64(max_indices) * sizeof(u32), 4);
            auto* out = reinterpret_cast<u32*>(data);
            for (u32 quad = 0; quad + 4 <= count; quad += 4) {
                for (const u32 corner : corners) {
                    *out++ = quad + corner;
                }
            }
            v.index_buffer = buffer;
            v.index_offset = offset;
            v.index_type = vk::IndexType::eUint32;
            v.index_count = max_indices;
            return true;
        }
        const VAddr address =
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
        if (!Native::Readable(reinterpret_cast<const void*>(address), u64(count) * index_size)) {
            return false; // NAT-106: indices the CPU can't read
        }
        const bool restart = (regs.enable_primitive_restart & 1) != 0;
        const auto [buffer, offset, data] = Allocate(u64(max_indices) * index_size, 4);
        if (is_index16) {
            v.index_count = ExpandQuads<u16>(
                reinterpret_cast<const u16*>(address), count,
                restart ? std::optional<u16>{u16(regs.primitive_restart_index)} : std::nullopt,
                corners, reinterpret_cast<u16*>(data));
        } else {
            v.index_count = ExpandQuads<u32>(
                reinterpret_cast<const u32*>(address), count,
                restart ? std::optional<u32>{regs.primitive_restart_index} : std::nullopt,
                corners, reinterpret_cast<u32*>(data));
        }
        v.index_buffer = buffer;
        v.index_offset = offset;
        return true;
    }
    if (!is_indexed) {
        return true;
    }
    const VAddr address =
        regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
    const auto found = GuestBuffer(address, u64(regs.num_indices) * index_size, false);
    if (!found) {
        return false;
    }
    v.index_buffer = found->first;
    v.index_offset = found->second;
    return true;
}

void NativeRenderer::RecordVertexIndex(const VertexIndexState& v) {
    const CmdBuf cmdbuf{stream.cmd};
    const bool dynamic_input = r.instance.IsVertexInputDynamicState();
    if (dynamic_input) {
        cmdbuf.setVertexInputEXT(v.input_bindings, v.attributes);
    }
    if (const u32 count = static_cast<u32>(v.buffers.size()); count != 0) {
        if (dynamic_input) {
            cmdbuf.bindVertexBuffers(0, count, v.buffers.data(), v.offsets.data());
        } else {
            cmdbuf.bindVertexBuffers2(0, count, v.buffers.data(), v.offsets.data(),
                                      v.sizes.data(), v.strides.data());
        }
    }
    if (v.index_buffer) {
        cmdbuf.bindIndexBuffer(v.index_buffer, v.index_offset, v.index_type);
    }
}

bool NativeRenderer::NeedsBarrierFor(const DrawBindings& b) const {
    const Stream& s = stream;
    if (s.buffers_written) {
        return true;
    }
    for (const auto* entry : b.sampled) {
        if (std::ranges::find(s.written, entry) != s.written.end()) {
            return true;
        }
    }
    for (const auto* entry : b.stored) {
        if (std::ranges::find(s.written, entry) != s.written.end()) {
            return true;
        }
    }
    return false;
}

void NativeRenderer::Written(const Native::ImageEntry* entry) {
    if (!entry) {
        return;
    }
    auto* written = const_cast<Native::ImageEntry*>(entry);
    written->write_version.store(++write_clock, std::memory_order_relaxed);
    // NAT-111: the textures made of it copy its picture in before their next read.
    std::scoped_lock lock{written->sources_mutex};
    for (auto* texture : written->dependents) {
        texture->sources_dirty.store(true, std::memory_order_relaxed);
    }
}

void NativeRenderer::RefreshSources(Native::ImageEntry& texture) {
    // NAT-111: only when one of its targets was drawn since the last time.
    if (!texture.sources_dirty.exchange(false, std::memory_order_relaxed)) {
        return;
    }
    static thread_local std::vector<Native::SourceLink> sources;
    {
        std::scoped_lock lock{texture.sources_mutex};
        if (texture.sources.empty()) {
            return;
        }
        sources.assign(texture.sources.begin(), texture.sources.end());
    }
    // Per subresource of the texture, the newest picture drawn for it (several layouts take
    // turns in the same memory), when newer than what the texture holds.
    struct Pick {
        Native::ImageEntry* source;
        Native::SourcePlace place;
        u64 version;
    };
    boost::container::small_vector<Pick, 16> picks;
    auto& held = held_versions[&texture];
    const u64 own = texture.write_version.load(std::memory_order_relaxed);
    for (const auto& [source, place] : sources) {
        const u64 version = source->write_version.load(std::memory_order_relaxed);
        if (version == 0 || !source->image) {
            continue;
        }
        const u32 key = (u32(place.mip) << 16) | u32(place.layer);
        if (version <= std::max(held[key], own)) {
            continue;
        }
        const auto same = std::ranges::find_if(picks, [&](const Pick& pick) {
            return pick.place.mip == place.mip && pick.place.layer == place.layer;
        });
        if (same == picks.end()) {
            picks.push_back({source, place, version});
        } else if (version > same->version) {
            *same = {source, place, version};
        }
    }
    if (picks.empty()) {
        return;
    }
    Stream& s = Begin();
    PrepareImages();  // the texture's own upload goes first
    GlobalBarrier();  // the sources' pictures are written
    const vk::Image dst{Handle<VkImage_T>(texture.image->NativeHandle())};
    for (const auto& pick : picks) {
        const auto& place = pick.place;
        const u32 width = std::min({place.width, std::max(texture.info.size.width >> place.mip, 1u),
                                    pick.source->info.size.width});
        const u32 height =
            std::min({place.height, std::max(texture.info.size.height >> place.mip, 1u),
                      pick.source->info.size.height});
        const vk::ImageCopy region{
            .srcSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = 0,
                            .baseArrayLayer = 0,
                            .layerCount = 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = u32(place.mip),
                            .baseArrayLayer = u32(place.layer),
                            .layerCount = 1},
            .dstOffset = {0, 0, 0},
            .extent = {width, height, 1},
        };
        CmdBuf{s.cmd}.copyImage(vk::Image{Handle<VkImage_T>(pick.source->image->NativeHandle())},
                                vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eGeneral, region);
        held[(u32(place.mip) << 16) | u32(place.layer)] = pick.version;
    }
    s.written.push_back(&texture); // read after the next barrier
    stats.source_copies += picks.size();
    ++s.commands;
}

void NativeRenderer::NoteWrites(const Targets* targets, const DrawBindings& b) {
    Stream& s = stream;
    const auto note = [&](const Native::ImageEntry* entry) {
        Written(entry);
        if (entry && std::ranges::find(s.written, entry) == s.written.end()) {
            s.written.push_back(entry);
        }
    };
    if (targets) {
        for (const auto* entry : targets->colors) {
            note(entry);
            // NAT-112: small pictures go back to guest memory.
            constexpr u64 WritebackSize = 64_KB;
            if (entry && entry->info.guest_size <= WritebackSize && entry->info.num_samples == 1 &&
                std::ranges::find(writebacks, entry) == writebacks.end()) {
                writebacks.push_back(const_cast<Native::ImageEntry*>(entry));
            }
        }
        note(targets->depth);
    }
    for (const auto* entry : b.stored) {
        note(entry);
        constexpr u64 WritebackSize = 64_KB; // NAT-112, as for render targets
        if (entry->info.guest_size <= WritebackSize && entry->info.num_samples == 1 &&
            !entry->info.props.is_depth &&
            std::ranges::find(writebacks, entry) == writebacks.end()) {
            writebacks.push_back(const_cast<Native::ImageEntry*>(entry));
        }
    }
    s.buffers_written |= b.writes_buffers;
}

void NativeRenderer::Draw(bool is_indexed, u32 index_offset) {
    std::scoped_lock lock{mutex};
    const auto start = std::chrono::steady_clock::now();
    struct Timed {
        std::chrono::steady_clock::time_point start;
        u64& total;
        ~Timed() {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
    } timed{start, stats.draw_ns};
    r.scheduler.PopPendingOperations();
    const auto& regs = r.liverpool->DrawRegs();
    switch (Classify(regs)) {
    case Filter::EliminateFastClear: {
        // Fast-cleared tiles get the clear colour now when nothing drew into the target since
        // its CMASK was cleared (otherwise the pass that drew started with the clear).
        const auto& col_buf = regs.color_buffers[0];
        if (col_buf && col_buf.info.fast_clear && cleared_meta.erase(col_buf.CmaskAddress())) {
            const VideoCore::TextureCache::ImageDesc desc{col_buf, r.liverpool->DrawCbExtent(0)};
            if (Native::ImageEntry* entry = registry->Find(desc.info, true, frame)) {
                Stream& s = Begin();
                PrepareImages();
                GlobalBarrier();
                const vk::ImageSubresourceRange range{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = desc.view_info.range.base.layer,
                    .layerCount = desc.view_info.range.extent.layers,
                };
                CmdBuf{s.cmd}.clearColorImage(
                    vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())},
                    vk::ImageLayout::eGeneral, LiverpoolToVK::ColorBufferClearValue(col_buf).color,
                    range);
                Written(entry);
                s.buffers_written = true; // the next work waits for the clear
                ++s.commands;
            }
        }
        ++stats.skipped;
        return;
    }
    case Filter::FmaskDecompress:
    case Filter::NoPrimitive:
        ++stats.skipped;
        return;
    case Filter::Resolve:
        ResolveDraw(regs);
        return;
    case Filter::Copy: {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan, "NAT-102: depth/stencil copy draw skipped");
        }
        ++stats.skipped;
        return;
    }
    case Filter::Draw:
        break;
    }

    const bool quad_list = regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
    bool quad_triangles = quad_list && Common::PerfFeatureEnabled(28) &&
                          u64(regs.num_indices) / 4 * 6 * sizeof(u32) <= MaxQuadIndexBytes;
    const GraphicsPipeline* pipeline = [&] {
        PhaseClock clock{stats.phase_ns[0]};
        return r.AcquireGraphicsPipeline(
            {
                .vertex_sgpr_offset = 0,
                .instance_sgpr_offset = 0,
                .tessellate_quads = quad_list && !quad_triangles,
            },
            !quad_list);
    }();
    if (!pipeline) {
        ++stats.no_pipeline;
        return;
    }
    Stream& s = Begin();
    Targets targets;
    {
        PhaseClock clock{stats.phase_ns[1]};
        ResolveTargets(regs, *pipeline, targets);
    }
    DrawBindings& b = bindings;
    b.Clear();
    b.push = UserData(regs);
    BindStages(pipeline->GetStages(), regs, false, b);
    if (b.failed) {
        ++stats.skipped;
        return;
    }
    static thread_local VertexIndexState vertex_index;
    {
        PhaseClock clock{stats.phase_ns[5]};
        vertex_index = {};
        ResolveVertexBuffers(*pipeline, regs, vertex_index);
        if (!ResolveIndices(regs, is_indexed, index_offset, quad_triangles, vertex_index)) {
            ++stats.skipped;
            return;
        }
    }
    {
        PhaseClock clock{stats.phase_ns[6]};
        PrepareImages();
    }
    PhaseClock record_clock{stats.phase_ns[7]};
    if (!s.rendering || !(s.render == targets.state) || NeedsBarrierFor(b)) {
        GlobalBarrier();
    }
    BeginRendering(targets.state);

    // The renderer's dynamic state code, filling this list's state.
    {
        DynamicState* const saved = Rasterizer::t_dynamic_state;
        Rasterizer::t_dynamic_state = &s.dynamic;
        r.ctx->attachment_feedback_loop = {};
        r.UpdateViewportScissorState();
        r.UpdateDepthStencilState();
        r.UpdatePrimitiveState(is_indexed, quad_triangles);
        r.UpdateRasterizationState();
        r.UpdateColorBlendingState(pipeline);
        s.dynamic.Commit(r.instance, CmdBuf{s.cmd});
        Rasterizer::t_dynamic_state = saved;
    }
    RecordVertexIndex(vertex_index);
    const u32 index_count = vertex_index.index_count;
    const CmdBuf cmdbuf{s.cmd};
    if (s.pipeline != pipeline->Handle()) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
        s.pipeline = pipeline->Handle();
    }
    PushBindings(*pipeline, vk::PipelineBindPoint::eGraphics, b);
    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto [vertex_offset, instance_offset] =
        DrawOffsets(regs, vs_info, pipeline->GetFetchShader());
    const u32 instances = regs.num_instances.NumInstances();
    if (quad_triangles) {
        if (index_count != 0) {
            cmdbuf.drawIndexed(index_count, instances, 0, s32(vertex_offset), instance_offset);
        }
    } else if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, instances, 0, s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, instances, vertex_offset, instance_offset);
    }
    NoteWrites(&targets, b);
    ++stats.draws;
    ++s.commands;
    ChunkFlushCheck();
    Report();
}

bool NativeRenderer::MetaClearDispatch(const Shader::Info& cs) {
    // As IsComputeMetaClear: a shader writing a known CMASK/HTILE without tile address math
    // (no XOR) and reading none clears it. The dispatch still runs (FIX-112: GT Sport reuses
    // such memory for pictures).
    for (const auto& desc : cs.buffers) {
        if (!desc.IsSpecial() && !desc.is_written &&
            known_meta.contains(desc.GetSharp(cs).base_address)) {
            return false;
        }
    }
    if (cs.has_bitwise_xor) {
        return false;
    }
    bool cleared = false;
    for (const auto& desc : cs.buffers) {
        const VAddr address = desc.GetSharp(cs).base_address;
        if (!desc.IsSpecial() && desc.is_written && known_meta.contains(address)) {
            cleared_meta.insert(address);
            cleared = true;
        }
    }
    stats.meta_clears += cleared;
    return cleared;
}

void NativeRenderer::DispatchDirect() {
    std::scoped_lock lock{mutex};
    const auto start = std::chrono::steady_clock::now();
    struct Timed {
        std::chrono::steady_clock::time_point start;
        u64& total;
        ~Timed() {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
    } timed{start, stats.dispatch_ns};
    r.scheduler.PopPendingOperations();
    const auto& cs_program = r.liverpool->DrawCsRegs();
    const ComputePipeline* pipeline = r.AcquireComputePipeline();
    if (!pipeline) {
        ++stats.no_pipeline;
        return;
    }
    const auto& regs = r.liverpool->DrawRegs();
    MetaClearDispatch(pipeline->GetStage(Shader::SwStage::Compute));
    Stream& s = Begin();
    DrawBindings& b = bindings;
    b.Clear();
    b.push = UserData(regs);
    BindStages(pipeline->GetStages(), regs, true, b);
    if (b.failed) {
        ++stats.skipped;
        return;
    }
    PrepareImages();
    GlobalBarrier();
    const CmdBuf cmdbuf{s.cmd};
    if (s.pipeline != pipeline->Handle()) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
        s.pipeline = pipeline->Handle();
    }
    PushBindings(*pipeline, vk::PipelineBindPoint::eCompute, b);
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    NoteWrites(nullptr, b);
    s.buffers_written = true; // what the dispatch wrote is seen after the next barrier
    ++stats.dispatches;
    ++s.commands;
    ChunkFlushCheck();
    Report();
}

void NativeRenderer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    std::scoped_lock lock{mutex};
    r.scheduler.PopPendingOperations();
    const ComputePipeline* pipeline = r.AcquireComputePipeline();
    if (!pipeline) {
        ++stats.no_pipeline;
        return;
    }
    const auto& regs = r.liverpool->DrawRegs();
    Stream& s = Begin();
    const auto args = GuestBuffer(address + offset, std::max<u32>(size, 12), false);
    if (!args) {
        ++stats.skipped;
        return;
    }
    DrawBindings& b = bindings;
    b.Clear();
    b.push = UserData(regs);
    BindStages(pipeline->GetStages(), regs, true, b);
    PrepareImages();
    GlobalBarrier();
    const CmdBuf cmdbuf{s.cmd};
    if (s.pipeline != pipeline->Handle()) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
        s.pipeline = pipeline->Handle();
    }
    PushBindings(*pipeline, vk::PipelineBindPoint::eCompute, b);
    cmdbuf.dispatchIndirect(args->first, args->second);
    NoteWrites(nullptr, b);
    s.buffers_written = true;
    ++stats.dispatches;
    ++s.commands;
    ChunkFlushCheck();
}

void NativeRenderer::ResolveDraw(const AmdGpu::Regs& regs) {
    const VideoCore::TextureCache::ImageDesc src_desc{regs.color_buffers[0],
                                                      r.liverpool->DrawCbExtent(0)};
    const VideoCore::TextureCache::ImageDesc dst_desc{regs.color_buffers[1],
                                                      r.liverpool->DrawCbExtent(1)};
    Native::ImageEntry* src = registry->Find(src_desc.info, true, frame);
    Native::ImageEntry* dst = registry->Find(dst_desc.info, true, frame);
    if (!src || !dst) {
        return;
    }
    Stream& s = Begin();
    PrepareImages();
    GlobalBarrier();
    const auto layers = std::min(src_desc.view_info.range.extent.layers,
                                 dst_desc.view_info.range.extent.layers);
    const vk::ImageResolve region{
        .srcSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = src_desc.view_info.range.base.layer,
                        .layerCount = layers},
        .dstSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = dst_desc.view_info.range.base.layer,
                        .layerCount = layers},
        .extent{std::min(src->info.size.width, dst->info.size.width),
                std::min(src->info.size.height, dst->info.size.height), 1},
    };
    const vk::Image src_image{Handle<VkImage_T>(src->image->NativeHandle())};
    const vk::Image dst_image{Handle<VkImage_T>(dst->image->NativeHandle())};
    if (src->image->Desc().samples > 1 && dst->image->Desc().samples == 1) {
        CmdBuf{s.cmd}.resolveImage(src_image, vk::ImageLayout::eGeneral, dst_image,
                                   vk::ImageLayout::eGeneral, region);
    } else {
        const vk::ImageCopy copy{
            .srcSubresource = region.srcSubresource,
            .dstSubresource = region.dstSubresource,
            .extent = region.extent,
        };
        CmdBuf{s.cmd}.copyImage(src_image, vk::ImageLayout::eGeneral, dst_image,
                                vk::ImageLayout::eGeneral, copy);
    }
    Written(dst);
    s.buffers_written = true;
    ++s.commands;
}

void NativeRenderer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    if (num_bytes == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    Stream& s = Begin();
    if (is_gds) {
        GlobalBarrier();
        CmdBuf{s.cmd}.fillBuffer(vk::Buffer{Handle<VkBuffer_T>(gds->NativeHandle())}, address,
                                 std::min<u64>(num_bytes, gds->Size() - address), value);
        s.buffers_written = true;
        ++s.commands;
        return;
    }
    if (known_meta.contains(address)) {
        cleared_meta.insert(address); // a CMASK/HTILE cleared by the CP
        ++stats.meta_clears;
    }
    ++stats.fills;
    if (const auto found = GuestBuffer(address, num_bytes, true)) {
        GlobalBarrier();
        CmdBuf{s.cmd}.fillBuffer(found->first, found->second, num_bytes, value);
        s.buffers_written = true;
        ++s.commands;
    } else {
        // Not GPU-visible as one range: written now (before the GPU runs what precedes it).
        if (r.memory->IsValidMapping(address, num_bytes)) {
            std::fill_n(reinterpret_cast<u32*>(address), num_bytes / 4, value);
        }
    }
    registry->Invalidate(address, num_bytes, Native::ImageRegistry::Stale::GpuWrite);
}

void NativeRenderer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (num_bytes == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    Stream& s = Begin();
    ++stats.copies;
    const vk::Buffer gds_buffer{Handle<VkBuffer_T>(gds->NativeHandle())};
    const auto source = src_gds ? std::optional{std::pair{gds_buffer, u64(src)}}
                                : GuestBuffer(src, num_bytes, false);
    const auto dest = dst_gds ? std::optional{std::pair{gds_buffer, u64(dst)}}
                              : GuestBuffer(dst, num_bytes, true);
    if (source && dest) {
        GlobalBarrier();
        CmdBuf{s.cmd}.copyBuffer(source->first, dest->first,
                                 vk::BufferCopy{.srcOffset = source->second,
                                                .dstOffset = dest->second,
                                                .size = num_bytes});
        s.buffers_written = true;
        ++s.commands;
    } else if (!dst_gds && !src_gds) {
        if (r.memory->IsValidMapping(dst, num_bytes) &&
            Native::Readable(reinterpret_cast<const void*>(src), num_bytes)) {
            std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
                        num_bytes);
        }
    } else {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan, "NAT-101: GDS copy {:#x} <- {:#x} ({} bytes) dropped", dst,
                        src, num_bytes);
        }
    }
    if (!dst_gds) {
        registry->Invalidate(dst, num_bytes, Native::ImageRegistry::Stale::GpuWrite);
    }
}

bool NativeRenderer::InlineWrite(VAddr address, std::span<const u8> data) {
    if (data.empty() || data.size() % 4 != 0 || data.size() > 65536 || address % 4 != 0) {
        return false;
    }
    std::scoped_lock lock{mutex};
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct || !direct->EnsureImported(address, data.size())) {
        return false;
    }
    const auto found = direct->FindBuffer(address, data.size());
    if (!found) {
        return false;
    }
    Stream& s = Begin();
    GlobalBarrier();
    CmdBuf{s.cmd}.updateBuffer(found->first->Handle(), found->second, data.size(), data.data());
    s.buffers_written = true;
    ++s.commands;
    ++stats.inline_writes;
    return true;
}

void NativeRenderer::OnUnmap(VAddr address, u64 size) {
    registry->Invalidate(address, size, Native::ImageRegistry::Stale::Unmap);
}

void NativeRenderer::EvictImages() {
    // NAT-108: images not used for two seconds go (GT Sport's scratch pool holds render
    // targets of ever-changing sizes: the UI's blur panels, effects).
    constexpr u64 KeepFrames = 120;
    const u64 tick = r.scheduler.CurrentTick();
    {
        const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
        while (!retired_views.empty() && retired_views.front().first <= done) {
            retired_views.pop_front();
        }
    }
    if (frame <= KeepFrames || frame % 30 != 0) {
        return;
    }
    registry->Evict(frame - KeepFrames, [&](Native::ImageEntry& entry) {
        {
            std::scoped_lock lock{views_mutex};
            if (const auto it = views.find(&entry); it != views.end()) {
                for (auto& record : it->second) {
                    retired_views.emplace_back(tick, std::move(record.view));
                }
                views.erase(it);
            }
        }
        held_versions.erase(&entry);
        std::erase(stream.written, &entry);
        std::erase(writebacks, &entry);
    });
}

void NativeRenderer::WatchPages(VAddr address, u64 size) {
    const VAddr begin = VideoCore::PageManager::GetPageAddr(address);
    const VAddr end = VideoCore::PageManager::GetNextPageAddr(address + size - 1);
    constexpr u64 PageSize = 4096;
    std::scoped_lock lock{watch_mutex};
    VAddr run = 0; // start of a run of pages not watched yet
    for (VAddr page = begin; page < end; page += PageSize) {
        const bool fresh = watched_pages.insert(page).second;
        if (fresh && run == 0) {
            run = page;
        } else if (!fresh && run != 0) {
            r.page_manager.UpdatePageWatchers(run, page - run, VideoCore::PageOp::Track);
            run = 0;
        }
    }
    if (run != 0) {
        r.page_manager.UpdatePageWatchers(run, end - run, VideoCore::PageOp::Track);
    }
}

void NativeRenderer::OnCpuWrite(VAddr address, u64 size, u64 exact) {
    // The written pages stop being watched (so they open up and the write goes on), and the
    // textures over them are uploaded again when next used (which watches them again).
    const VAddr begin = VideoCore::PageManager::GetPageAddr(address);
    const VAddr end = VideoCore::PageManager::GetNextPageAddr(address + std::max<u64>(size, 1) - 1);
    constexpr u64 PageSize = 4096;
    {
        std::scoped_lock lock{watch_mutex};
        VAddr run = 0;
        for (VAddr page = begin; page < end; page += PageSize) {
            const bool watched = watched_pages.erase(page) != 0;
            if (watched && run == 0) {
                run = page;
            } else if (!watched && run != 0) {
                r.page_manager.UpdatePageWatchers(run, page - run, VideoCore::PageOp::Untrack);
                run = 0;
            }
        }
        if (run != 0) {
            r.page_manager.UpdatePageWatchers(run, end - run, VideoCore::PageOp::Untrack);
        }
    }
    registry->CpuWrite(begin, end - begin, exact != 0 ? (address - begin) + exact : 0);
    ++stats.cpu_writes;
    // NAT-114: pages of regions written by the CPU (a fault on any thread) are copied in before
    // the region's next use; the recorder sorts them to their regions.
    {
        std::scoped_lock pending_lock{region_dirty_mutex};
        pending_region_writes.emplace_back(begin, end);
    }
    has_pending_region_writes.store(true, std::memory_order_release);
}

u32 NativeRenderer::ReadGds(u32 gds_offset) {
    u32 value = 0;
    if (gds_offset + sizeof(u32) <= gds->Size()) {
        std::memcpy(&value, gds->Mapped() + gds_offset, sizeof(value));
    }
    return value;
}

void NativeRenderer::Splice() {
    std::scoped_lock lock{mutex};
    Stream& s = stream;
    if (!s.list) {
        return;
    }
    EndRendering();
    if (!writebacks.empty()) {
        GlobalBarrier(); // NAT-112: written back in this submission
    }
    s.list->End();
    r.scheduler.EndRendering();
    const vk::CommandBuffer secondary = s.cmd;
    r.scheduler.CommandBuffer().executeCommands(secondary);
    const u64 tick = r.scheduler.CurrentTick();
    device->MarkExecuted(s.list, {timeline.get(), tick});
    {
        std::scoped_lock chunk_lock{chunk_mutex};
        for (auto& chunk : s.chunks) {
            chunk.retire_tick = tick;
            retiring_chunks.push_back(std::move(chunk));
        }
        for (const auto pool : s.pools) {
            retiring_pools.emplace_back(tick, pool);
        }
    }
    s.chunks.clear();
    s.chunk_used = 0;
    s.pools.clear();
    s.keep.clear(); // freed by the device once the GPU ran this list (NAT-107)
    device->Collect(); // objects the GPU finished with (staging buffers of uploads)
    s.list = nullptr;
    s.cmd = vk::CommandBuffer{};
    s.rendering = false;
    ++stats.splices;
}

std::optional<NativeRenderer::Display> NativeRenderer::FindDisplay(const VideoCore::ImageInfo& info,
                                                                   vk::Format view_format) {
    std::scoped_lock lock{mutex};
    ++frame;
    // The game's render target there (its layout from the CB registers may differ in details
    // from the video out description), else an image of the display layout.
    Native::ImageEntry* entry =
        registry->FindExisting(info.guest_address, [&](const Native::ImageEntry& candidate) {
            return !candidate.from_cpu && !candidate.info.props.is_depth &&
                   candidate.info.size.width == info.size.width &&
                   candidate.info.size.height == info.size.height &&
                   candidate.info.num_bits == info.num_bits;
        });
    if (!entry) {
        static std::atomic<u32> logged{};
        if (Sample(logged)) {
            LOG_WARNING(Render_Vulkan,
                        "NAT-103: nothing drew the display buffer {:#x} ({}x{}, {} bits) yet",
                        info.guest_address, info.size.width, info.size.height, info.num_bits);
        }
        entry = registry->Find(info, true, frame);
    }
    if (!entry || !entry->image) {
        return std::nullopt;
    }
    VideoCore::ImageViewInfo view_info{};
    view_info.format = view_format;
    view_info.mapping.a = AmdGpu::CompSwizzle::One; // as the presenter: no alpha blending
    const vk::ImageView view = View(*entry, view_info, ViewUse::Sampled);
    if (!view) {
        return std::nullopt;
    }
    // A display buffer nothing drew into yet starts in the general layout too.
    if (device->PendingImageInit()) {
        Begin();
        PrepareImages();
    }
    Splice();
    const Display display{vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())}, view,
                          entry->info.size.width, entry->info.size.height};
    EvictImages(); // (the display picture was just used: it stays)
    return display;
}

void NativeRenderer::Report() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    last_report = now;
    const auto& registry_stats = registry->GetStats();
    const u64 uploads = registry_stats.uploads.load();
    const u64 upload_bytes = registry_stats.upload_bytes.load();
    const u64 evicted = registry_stats.evicted.load();
    std::array<u64, 3> stale{};
    for (u32 i = 0; i < stale.size(); ++i) {
        stale[i] = registry_stats.stale[i].load();
    }
    // The textures uploaded most in the window.
    std::vector<std::tuple<u32, VAddr, u32, u32, u64>> busy;
    registry->ForEachOverlapping(0, ~VAddr{0}, [&](Native::ImageEntry& entry) {
        if (const u32 count = entry.upload_count.exchange(0); count >= 4) {
            busy.emplace_back(count, entry.info.guest_address, entry.info.size.width,
                              entry.info.size.height, entry.info.guest_size);
        }
    });
    std::ranges::sort(busy, std::greater{});
    std::string busiest;
    for (size_t i = 0; i < std::min<size_t>(busy.size(), 5); ++i) {
        const auto& [count, address, width, height, size] = busy[i];
        busiest += fmt::format(" {:#x} {}x{} ({} KB) x{};", address, width, height, size / 1024,
                               count);
    }
    std::string crowded;
    for (const auto& [address, count] : registry->Crowded(4)) {
        crowded += fmt::format(" {:#x} x{};", address, count);
    }
    LOG_INFO(Render_Vulkan,
             "NAT-108 images (10 s): {} uploads ({:.1f} MB), {} skipped as unchanged (NAT-109), "
             "textures made stale by CPU writes {}, unmaps {}, GPU copies {}; {} evicted, {} "
             "live, most at one address:{}; uploaded most:{}",
             uploads - report_base.uploads, (upload_bytes - report_base.upload_bytes) / 1048576.0,
             registry->GetUploadStats().unchanged.exchange(0),
             stale[0] - report_base.stale[0], stale[1] - report_base.stale[1],
             stale[2] - report_base.stale[2], evicted - report_base.evicted, registry->Count(),
             crowded,
             busiest.empty() ? std::string{" none"} : busiest);
    report_base = {uploads, upload_bytes, evicted, stale};
    LOG_INFO(Render_Vulkan,
             "NAT-100 native (10 s): {} draws, {} dispatches, {} skipped, {} without pipeline; "
             "{} passes, {} barriers, {} lists run; {} reads copied ({:.1f} MB), {} ranges bound "
             "empty, {} images bound empty; {} fills, {} copies, {} meta clears, {} inline "
             "writes, {} pictures copied from targets, {} written back, {} CPU writes to textures, "
             "{} ranges gathered, region binds/made/refreshes {}/{}/{} (NAT-114), "
             "on the GPU ({:.1f} MB), uploads from guest memory on the GPU/CPU {}/{}; images so far: "
             "{} lookups, {} textures and {} targets made, {} uploads ({:.1f} MB); {:.0f} draws/s, "
             "{:.2f} us a draw, {:.2f} us a dispatch on the recorder; images bound empty by "
             "reason (none/unmapped/format/geometry/outside/FIX-081/no image) {}; us per draw in "
             "pipeline/targets/buffers/textures(sources)/vertices/prepare/record {}",
             stats.draws, stats.dispatches, stats.skipped, stats.no_pipeline, stats.passes,
             stats.barriers, stats.splices, stats.copied_reads, stats.copied_bytes / 1048576.0,
             stats.unbound, stats.null_images, stats.fills, stats.copies, stats.meta_clears,
             stats.inline_writes, stats.source_copies, stats.writebacks,
             stats.cpu_writes.exchange(0),
             stats.region_binds, stats.regions_made, stats.region_refreshes, stats.gathers,
             stats.gathered_bytes / 1048576.0,
             registry->GetUploadStats().gpu.load(), registry->GetUploadStats().cpu.load(),
             registry_stats.lookups.load(),
             registry_stats.created_textures.load(), registry_stats.created_targets.load(),
             registry_stats.uploads.load(), registry_stats.upload_bytes.load() / 1048576.0,
             stats.draws / seconds, stats.draws ? stats.draw_ns / 1000.0 / stats.draws : 0.0,
             stats.dispatches ? stats.dispatch_ns / 1000.0 / stats.dispatches : 0.0,
             fmt::join(stats.null_reasons, "/"), [&] {
                 std::string text;
                 for (u32 i = 0; i < stats.phase_ns.size(); ++i) {
                     text += fmt::format(
                         "{}{:.2f}", i == 0 ? "" : (i == 4 ? "(" : (i == 5 ? ")/" : "/")),
                         stats.draws ? stats.phase_ns[i] / 1000.0 / stats.draws : 0.0);
                 }
                 return text;
             }());
    const u64 cpu_writes = stats.cpu_writes.load();
    stats.draws = stats.dispatches = stats.skipped = stats.no_pipeline = stats.passes = 0;
    stats.barriers = stats.splices = stats.copied_reads = stats.copied_bytes = stats.unbound = 0;
    stats.null_images = stats.fills = stats.copies = stats.meta_clears = stats.inline_writes = 0;
    stats.source_copies = stats.draw_ns = stats.dispatch_ns = 0;
    stats.gathers = stats.gathered_bytes = stats.writebacks = 0;
    stats.region_binds = stats.regions_made = stats.region_refreshes = 0;
    stats.null_reasons = {};
    stats.phase_ns = {};
    (void)cpu_writes;
}

} // namespace Vulkan
