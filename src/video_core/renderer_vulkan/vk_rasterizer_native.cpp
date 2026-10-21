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
#include "common/ryty_config.h"
// (ryty_config.h brings in windows.h, whose `near` and `far` macros would rename variables here.)
#undef near
#undef far
#include "core/guest_hash.h"
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
    push_data.quad_sample_mask = {regs.reg_array[0xa30e], regs.reg_array[0xa30f]}; // PA_SC_AA_MASK
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
// OPT-001: the phase timers read the clock on one draw in 16 (reading it in every phase of every
// draw was 7% of the recorder's time); the report divides by the draws timed.
thread_local bool t_timed = true;
struct PhaseClock {
    u64* total;
    std::chrono::steady_clock::time_point start{};
    explicit PhaseClock(u64& total_) : total{t_timed ? &total_ : nullptr} {
        if (total) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~PhaseClock() {
        if (total) {
            *total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
        }
    }
};

constexpr u64 ChunkSize = 8_MB;
constexpr u32 SpliceAfter = 512; ///< commands per list before it is run and submitted

} // namespace

bool NativeModeRequested() {
    // NAT-160: the launcher's renderer choice (GPU.native_renderer in the game's settings, so
    // GTSport.exe started on its own follows it too); RYGT_NATIVE=1 or =0 overrides it.
    static const bool requested = [] {
        if (const char* env = std::getenv("RYGT_NATIVE"); env && *env) {
            return env[0] == '1';
        }
        const auto value = Common::RytyGameConfigValue("native_renderer");
        return value && *value == "true";
    }();
    return requested;
}

void NativeRenderer::DrawBindings::Clear() {
    writes.clear();
    write_sets.clear();
    buffers.clear();
    images.clear();
    push = {};
    sampled.clear();
    stored.clear();
    gpu_writes.clear();
    fill_pattern.reset();
    fill_address = 0;
    fill_size = 0;
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
        // NAT-146: the GPU's bytes there are in a region (shader-written memory lives in video
        // memory): the upload reads the region.
        if (Region* region = RegionOver(address, size);
            region && address >= region->begin && address + size <= region->end &&
            (address - region->begin) % r.instance.StorageMinAlignment() == 0) {
            RefreshRegion(*region);
            auto wrapped = device->WrapBuffer(region->buffer->NativeHandle(), region->buffer->Size());
            source.buffer = wrapped.get();
            source.offset = address - region->begin;
            source.kind = 1;
            source.keep.push_back(std::move(wrapped));
            return source;
        }
        if (const auto found = direct->FindBuffer(address, size);
            found && found->second % r.instance.StorageMinAlignment() == 0) {
            auto wrapped = device->WrapBuffer(
                reinterpret_cast<u64>(static_cast<VkBuffer>(found->first->Handle())),
                found->first->SizeBytes());
            source.buffer = wrapped.get();
            source.offset = found->second;
            source.kind = 2;
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
            source.kind = 3;
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
    s.vertex_state_known = false; // OPT-005
    s.commands = 0;
    // What was recorded before this list (earlier lists, the presenter) is finished first.
    GlobalBarrier();
    MakePendingMirrors(); // NAT-150: behind the barrier the list starts with anyway
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
    s.attach_dirty.clear();
    s.read.clear();
    ++stats.barriers;
    FlushWritebacks();
}

void NativeRenderer::PassBarrier() {
    Stream& s = stream;
    EndRendering();
    constexpr auto stages = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                            vk::PipelineStageFlagBits2::eLateFragmentTests;
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = stages,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .dstStageMask = stages,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                         vk::AccessFlagBits2::eColorAttachmentWrite |
                         vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
    };
    s.cmd.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
    s.attach_dirty.clear();
    ++stats.pass_barriers;
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
            // NAT-146: regions over the picture's memory copy it in before their next use.
            if (Region* region = RegionOver(address, size)) {
                for (VAddr page = Common::AlignDown(std::max<VAddr>(address, region->begin), 4096);
                     page < std::min<VAddr>(address + size, region->end); page += 4096) {
                    region->dirty_pages.push_back(page);
                }
            }
            // NAT-150: so do mirrors.
            for (auto it = FirstMirror(address);
                 it != mirrors.end() && it->second.begin < address + size; ++it) {
                Mirror& mirror = it->second;
                for (VAddr page = Common::AlignDown(std::max<VAddr>(address, mirror.begin), 4096);
                     page < std::min<VAddr>(address + size, mirror.end); page += 4096) {
                    mirror.dirty_pages.push_back(page);
                }
            }
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
    ++buffer_generation; // OPT-003: regions are taken in or made
    auto* direct = r.buffer_cache.GetDirectMemory();
    // The new region takes in every region it overlaps (their GPU-written contents kept).
    VAddr begin = Common::AlignDown(address, 4096);
    VAddr end = Common::AlignUp(address + size, 4096);
    std::vector<Region> merged;
    {
        std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
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
    }
    buffer_log.Note(begin, end); // OPT-010
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
        // FIX-173: CPU writes to it not copied in yet are copied in at the new one's next use.
        region.dirty_pages.insert(region.dirty_pages.end(), old.dirty_pages.begin(),
                                  old.dirty_pages.end());
    }
    s.buffers_written = true;
    // CPU writes to it from now on are copied in before its next use.
    WatchPages(begin, end - begin);
    DropMirrors(begin, end - begin); // NAT-150: reads there use the region from now on
    ++stats.regions_made;
    LOG_INFO(Render_Vulkan, "NAT-114: region {:#x}+{:#x} on the GPU ({} taken in)", begin,
             end - begin, merged.size());
    std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
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
        // FIX-173: copies of pages no region holds are not needed.
        if (end - begin <= 64_KB) {
            std::scoped_lock lock{region_dirty_mutex};
            for (VAddr page = begin; page < end; page += 4096) {
                if (!RegionOver(page, 4096)) {
                    page_snapshots.erase(page);
                }
            }
        }
        // NAT-150: mirrors copy the pages in before their next use, unless the CPU wrote that
        // granule in another recent frame too: then the mirror goes and the granule is read in
        // place for a while.
        for (auto it = FirstMirror(begin); it != mirrors.end() && it->second.begin < end;) {
            Mirror& mirror = it->second;
            const VAddr from = std::max(begin, mirror.begin);
            const VAddr to = std::min(end, mirror.end);
            bool dynamic = false;
            for (u64 granule = from >> 16; granule <= (to - 1) >> 16; ++granule) {
                const auto [written, first] = mirror_writes.try_emplace(granule, frame);
                if (!first && written->second != frame && frame - written->second < 600) {
                    dynamic = true;
                    dynamic_granules[granule] = frame;
                }
                written->second = frame;
            }
            if (dynamic) {
                mirror_bytes -= mirror.end - mirror.begin;
                ++stats.mirrors_dropped;
                ++buffer_generation; // OPT-003
                buffer_log.Note(mirror.begin, mirror.end); // OPT-010
                std::unique_lock map_lock{buffer_map_mutex};
                it = mirrors.erase(it); // freed once the GPU is done with it
                continue;
            }
            for (VAddr page = from; page < to; page += 4096) {
                mirror.dirty_pages.push_back(page);
            }
            ++it;
        }
    }
    if (mirror_writes.size() > 1'000'000) {
        mirror_writes.clear();
    }
}

std::map<VAddr, NativeRenderer::Mirror>::iterator NativeRenderer::FirstMirror(VAddr begin) {
    auto it = mirrors.upper_bound(begin);
    if (it != mirrors.begin() && std::prev(it)->second.end > begin) {
        --it;
    }
    return it;
}

NativeRenderer::Mirror* NativeRenderer::MirrorHolding(VAddr address, u64 size) {
    auto it = mirrors.upper_bound(address);
    if (it == mirrors.begin()) {
        return nullptr;
    }
    Mirror& mirror = std::prev(it)->second;
    return address >= mirror.begin && address + size <= mirror.end ? &mirror : nullptr;
}

bool NativeRenderer::IsDynamic(VAddr begin, VAddr end) {
    constexpr u64 DynamicFrames = 600; // then the granule may be mirrored again
    if (dynamic_granules.empty()) {
        return false;
    }
    for (u64 granule = begin >> 16; granule <= (end - 1) >> 16; ++granule) {
        const auto it = dynamic_granules.find(granule);
        if (it == dynamic_granules.end()) {
            continue;
        }
        if (frame - it->second < DynamicFrames) {
            return true;
        }
        dynamic_granules.erase(it);
    }
    return false;
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::MirrorBuffer(VAddr address, u64 size) {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_NATIVE_MIRRORS");
        return !(env && env[0] == '0');
    }();
    if (!enabled || size == 0 || size > 64_MB) {
        return std::nullopt;
    }
    SortRegionWrites();
    Mirror* mirror = MirrorHolding(address, size);
    if (mirror && !mirror->dirty_pages.empty() && !stream.list) {
        Begin(); // a new list makes the pending mirrors first (they can take this one in)
        mirror = MirrorHolding(address, size);
    }
    if (mirror) {
        RefreshMirror(*mirror);
        mirror->used_frame = frame;
        ++stats.mirror_binds;
        memo_kind = MemoKind::Mirror; // OPT-003
        memo_mirror = mirror;
        return std::pair{vk::Buffer{Handle<VkBuffer_T>(mirror->buffer->NativeHandle())},
                         u64{address - mirror->begin}};
    }
    // Not mirrored: a range bound in an earlier frame too gets a mirror at the start of the next
    // list (read in place until then).
    const VAddr begin = Common::AlignDown(address, 4096);
    const VAddr end = Common::AlignUp(address + size, 4096);
    if (mirror_candidates.size() > 65536) {
        mirror_candidates.clear(); // per-frame data at ever-new addresses
    }
    const auto [it, inserted] = mirror_candidates.try_emplace(begin, MirrorCandidate{end, frame});
    MirrorCandidate& candidate = it->second;
    if (inserted) {
        return std::nullopt;
    }
    candidate.end = std::max(candidate.end, end);
    if (candidate.frame == frame) {
        return std::nullopt;
    }
    candidate.frame = frame; // looked at again next frame at the earliest
    if (IsDynamic(begin, candidate.end)) {
        return std::nullopt;
    }
    pending_mirrors.emplace_back(begin, candidate.end);
    mirror_candidates.erase(it);
    return std::nullopt;
}

void NativeRenderer::RefreshMirror(Mirror& mirror) {
    if (mirror.dirty_pages.empty()) {
        return;
    }
    std::vector<VAddr> pages;
    pages.swap(mirror.dirty_pages);
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct) {
        return;
    }
    Stream& s = Begin();
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(mirror.buffer->NativeHandle())};
    std::ranges::sort(pages);
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    for (const VAddr page : pages) {
        if (page < mirror.begin || page >= mirror.end) {
            continue;
        }
        direct->ForEachPart(page, std::min<u64>(4096, mirror.end - page),
                            [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                u64 bytes) {
                                s.cmd.copyBuffer(view.Handle(), dst,
                                                 vk::BufferCopy{.srcOffset = view_offset,
                                                                .dstOffset = page - mirror.begin +
                                                                             offset,
                                                                .size = bytes});
                            });
        WatchPages(page, std::min<u64>(4096, mirror.end - page));
    }
    s.buffers_written = true;
    stats.mirror_refresh_pages += pages.size();
}

void NativeRenderer::MakePendingMirrors() {
    if (pending_mirrors.empty() || !stream.list) {
        return;
    }
    std::vector<std::pair<VAddr, VAddr>> pending;
    pending.swap(pending_mirrors);
    ++buffer_generation; // OPT-003: mirrors are made, taken in or evicted
    std::ranges::sort(pending);
    std::vector<std::pair<VAddr, VAddr>> ranges; // overlapping or touching ranges joined
    for (const auto& [begin, end] : pending) {
        if (!ranges.empty() && begin <= ranges.back().second) {
            ranges.back().second = std::max(ranges.back().second, end);
        } else {
            ranges.emplace_back(begin, end);
        }
    }
    static const u64 budget = [] {
        const char* env = std::getenv("RYGT_NATIVE_MIRROR_MB");
        return (env ? std::strtoull(env, nullptr, 10) : 1024) * 1_MB;
    }();
    auto* direct = r.buffer_cache.GetDirectMemory();
    Stream& s = stream;
    bool copied = false;
    bool evicted = false;
    for (auto [begin, end] : ranges) {
        if (!direct || RegionOver(begin, end - begin) || IsDynamic(begin, end)) {
            continue; // shaders write there (the region holds it), or the CPU does
        }
        // The new mirror takes in the mirrors it overlaps.
        u64 taken = 0;
        for (auto it = FirstMirror(begin); it != mirrors.end() && it->second.begin < end; ++it) {
            begin = std::min(begin, it->second.begin);
            end = std::max(end, it->second.end);
            taken += it->second.end - it->second.begin;
        }
        if (end - begin > 256_MB) {
            continue;
        }
        if (mirror_bytes - taken + (end - begin) > budget && !evicted) {
            // Mirrors unused for two seconds go first.
            evicted = true;
            std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
            for (auto it = mirrors.begin(); it != mirrors.end();) {
                if (it->second.used_frame + 120 < frame &&
                    !(it->second.begin < end && it->second.end > begin)) {
                    mirror_bytes -= it->second.end - it->second.begin;
                    ++stats.mirrors_evicted;
                    buffer_log.Note(it->second.begin, it->second.end);
                    it = mirrors.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (mirror_bytes - taken + (end - begin) > budget) {
            ++stats.mirror_over_budget;
            continue;
        }
        if (!direct->EnsureImported(begin, end - begin)) {
            continue;
        }
        auto buffer = device->CreateBuffer({.size = end - begin,
                                            .usage = Native::Rhi::BufferUsageStorage |
                                                     Native::Rhi::BufferUsageVertex |
                                                     Native::Rhi::BufferUsageIndex |
                                                     Native::Rhi::BufferUsageTransferSrc |
                                                     Native::Rhi::BufferUsageTransferDst,
                                            .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                            .name = "native mirror"});
        if (!buffer) {
            continue;
        }
        {
            std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
            for (auto it = FirstMirror(begin); it != mirrors.end() && it->second.begin < end;) {
                mirror_bytes -= it->second.end - it->second.begin;
                it = mirrors.erase(it); // freed once the GPU is done with it
            }
        }
        buffer_log.Note(begin, end); // OPT-010
        const vk::Buffer dst{Handle<VkBuffer_T>(buffer->NativeHandle())};
        direct->ForEachPart(begin, end - begin,
                            [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                u64 bytes) {
                                s.cmd.copyBuffer(view.Handle(), dst,
                                                 vk::BufferCopy{.srcOffset = view_offset,
                                                                .dstOffset = offset,
                                                                .size = bytes});
                            });
        // CPU writes to it from now on are copied in before its next use.
        WatchPages(begin, end - begin);
        mirror_bytes += end - begin;
        ++stats.mirrors_made;
        stats.mirror_made_bytes += end - begin;
        std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
        mirrors.emplace(begin, Mirror{.begin = begin,
                                      .end = end,
                                      .buffer = std::move(buffer),
                                      .dirty_pages = {},
                                      .used_frame = frame});
        copied = true;
    }
    if (copied) {
        s.buffers_written = true; // the next draw waits for the copies
    }
}

void NativeRenderer::DropMirrors(VAddr address, u64 size) {
    for (auto it = FirstMirror(address); it != mirrors.end() && it->second.begin < address + size;) {
        mirror_bytes -= it->second.end - it->second.begin;
        ++buffer_generation; // OPT-003
        buffer_log.Note(it->second.begin, it->second.end); // OPT-010
        std::unique_lock map_lock{buffer_map_mutex};
        it = mirrors.erase(it); // freed once the GPU is done with it
    }
}

void NativeRenderer::RefreshRegion(Region& region) {
    SortRegionWrites();
    std::vector<VAddr> pages;
    pages.swap(region.dirty_pages);
    if (pages.empty()) {
        return;
    }
    const auto refresh_start = std::chrono::steady_clock::now();
    struct Timed {
        std::chrono::steady_clock::time_point start;
        u64& total;
        ~Timed() {
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
        }
    } timed{refresh_start, stats.region_refresh_ns};
    stats.region_refresh_pages += pages.size();
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (!direct) {
        return;
    }
    Stream& s = Begin();
    GlobalBarrier();
    const vk::Buffer dst{Handle<VkBuffer_T>(region.buffer->NativeHandle())};
    std::ranges::sort(pages);
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    const auto copy_in = [&](VAddr from, u64 size) {
        direct->ForEachPart(from, size,
                            [&](const VideoCore::Buffer& view, u64 view_offset, u64 offset,
                                u64 bytes) {
                                s.cmd.copyBuffer(view.Handle(), dst,
                                                 vk::BufferCopy{.srcOffset = view_offset,
                                                                .dstOffset = from - region.begin +
                                                                             offset,
                                                                .size = bytes});
                            });
    };
    for (const VAddr page : pages) {
        if (page < region.begin || page >= region.end) {
            continue;
        }
        const u64 page_bytes = std::min<u64>(4096, region.end - page);
        // FIX-173: the page as it was before the CPU wrote it (taken at the write fault).
        std::unique_ptr<std::array<u8, 4096>> before;
        {
            std::scoped_lock lock{region_dirty_mutex};
            if (const auto it = page_snapshots.find(page); it != page_snapshots.end()) {
                before = std::move(it->second);
                page_snapshots.erase(it);
            }
        }
        // NAT-145: only the pages copied in are watched again (the region was watched whole
        // when made; re-watching all of it at each refresh cost ~9 us a draw with big regions).
        // FIX-173: watched before guest memory is compared, so a CPU write from now on faults
        // (and takes a new copy) instead of slipping in between the compare and the watch.
        WatchPages(page, page_bytes);
        std::array<u8, 4096> now;
        if (!before || !Core::RytyMemory::GuardedCopy(now.data(),
                                                      reinterpret_cast<const void*>(page),
                                                      page_bytes)) {
            copy_in(page, page_bytes);
            continue;
        }
        // The runs of 16-byte pieces the CPU changed (gaps under 64 bytes joined).
        constexpr u64 Piece = 16;
        u64 run_begin = 0, run_end = 0, changed = 0;
        bool in_run = false;
        for (u64 at = 0; at < page_bytes; at += Piece) {
            const u64 n = std::min(Piece, page_bytes - at);
            if (std::memcmp(now.data() + at, before->data() + at, n) == 0) {
                continue;
            }
            if (in_run && at - run_end < 64) {
                run_end = at + n;
                continue;
            }
            if (in_run) {
                copy_in(page + run_begin, run_end - run_begin);
                changed += run_end - run_begin;
            }
            in_run = true;
            run_begin = at;
            run_end = at + n;
        }
        if (in_run) {
            copy_in(page + run_begin, run_end - run_begin);
            changed += run_end - run_begin;
        }
        ++stats.region_partial_pages;
        stats.region_partial_bytes += changed;
    }
    s.buffers_written = true;
    ++stats.region_refreshes;
    if (stats.region_partial_pages != 0 && Sample(logged_partial)) {
        LOG_INFO(Render_Vulkan, "FIX-173: region pages copied in part: {} ({:.1f} KB of {:.1f} KB)",
                 stats.region_partial_pages, stats.region_partial_bytes / 1024.0,
                 stats.region_partial_pages * 4.0);
    }
}

u64 NativeRenderer::SnapshotBelow() {
    static const u64 below = [] {
        const char* env = std::getenv("RYGT_NATIVE_SNAPSHOT_BELOW");
        return env && *env ? std::strtoull(env, nullptr, 0) : u64{0};
    }();
    return below;
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::GuestBuffer(VAddr address, u64 size,
                                                                      bool write, bool mirror) {
    if (size == 0) {
        return std::nullopt;
    }
    PhaseClock guest_clock{stats.guest_buffer_ns}; // NAT-145
    ++stats.guest_buffer_calls;
    last_mirror = nullptr; // OPT-004
    // OPT-003: the range's last result, when nothing it depends on changed.
    const u32 flags = u32(write) | (u32(mirror) << 1);
    const std::array<u64, 3> memo_key{address, size, flags};
    BufferMemo& memo =
        buffer_memos[XXH3_64bits(memo_key.data(), sizeof(memo_key)) & (buffer_memos.size() - 1)];
    if (memo.generation == buffer_generation && memo.address == address && memo.size == size &&
        memo.flags == flags) {
        if (memo.region || memo.mirror) {
            SortRegionWrites(); // CPU writes since: pages to copy in (may drop a mirror)
        }
        const bool fresh =
            memo.generation == buffer_generation &&
            (!memo.region || memo.region->dirty_pages.empty()) &&
            (!memo.mirror || memo.mirror->dirty_pages.empty());
        if (fresh) {
            if (memo.region) {
                ++stats.region_binds;
            } else if (memo.mirror) {
                ++stats.mirror_binds;
                memo.mirror->used_frame = frame;
            }
            last_bound = memo.bound;
            last_mirror = memo.mirror; // OPT-004
            ++stats.buffer_memo_hits;
            return std::pair{memo.buffer, memo.offset};
        }
    }
    memo_kind = MemoKind::None;
    memo_region = nullptr;
    memo_mirror = nullptr;
    const auto result = GuestBufferLookup(address, size, write, mirror);
    if (result && memo_kind == MemoKind::Mirror) {
        last_mirror = memo_mirror; // OPT-004
    }
    if (result && memo_kind != MemoKind::None) {
        memo = BufferMemo{
            .address = address,
            .size = size,
            .flags = flags,
            .generation = buffer_generation,
            .buffer = result->first,
            .offset = result->second,
            .region = memo_region,
            .mirror = memo_mirror,
            .bound = last_bound,
        };
    }
    return result;
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::GuestBufferLookup(VAddr address, u64 size,
                                                                            bool write,
                                                                            bool mirror) {
    last_bound = Bound::Region;
    // NAT-114: ranges in (or reaching into) a region use it.
    if (Region* region = RegionOver(address, size)) {
        if (address < region->begin || address + size > region->end) {
            region = MakeRegion(address, size);
        }
        if (region) {
            RefreshRegion(*region);
            ++stats.region_binds;
            memo_kind = MemoKind::Region; // OPT-003
            memo_region = region;
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
    }
    // NAT-144: every range shaders write lives in video memory (a region), not in guest memory
    // over PCIe (the GPU was 85% busy waiting on PCIe, 55% with this; the race went from 25 to
    // 33-39 FPS). What the GPU writes there doesn't reach guest memory: GT Sport's CPU doesn't
    // read it back (the old renderer runs with buffer readbacks off too); small render targets
    // still go back (NAT-112). RYGT_NATIVE_WRITE_REGIONS=0: written ranges in guest memory.
    static const bool write_regions = [] {
        const char* env = std::getenv("RYGT_NATIVE_WRITE_REGIONS");
        return !(env && env[0] == '0');
    }();
    // FIX-150: GT Sport's small buffers from 0xf000000000 up are ones the GPU writes for the CPU
    // to read back (in a region the CPU never saw them: no rain drops on the windshield); written
    // ranges there stay in guest memory. RYGT_NATIVE_INPLACE_FROM=<hex address> moves the limit
    // (ffffffffffffffff: every written range gets a region, as before).
    static const VAddr inplace_from = [] {
        const char* env = std::getenv("RYGT_NATIVE_INPLACE_FROM");
        return env ? static_cast<VAddr>(std::strtoull(env, nullptr, 16)) : VAddr{0xf000000000};
    }();
    // FIX-172 (diagnosis, RYGT_NATIVE_INPLACE_RANGES=<hex address>+<hex size>,...): written
    // ranges meeting one of these stay in guest memory too.
    static const std::vector<std::pair<VAddr, VAddr>> inplace_ranges = [] {
        std::vector<std::pair<VAddr, VAddr>> list;
        const char* env = std::getenv("RYGT_NATIVE_INPLACE_RANGES");
        for (const char* at = env; at && *at;) {
            char* end = nullptr;
            const VAddr begin = std::strtoull(at, &end, 16);
            if (end == at || *end != '+') {
                break;
            }
            at = end + 1;
            const u64 length = std::strtoull(at, &end, 16);
            list.emplace_back(begin, begin + std::max<u64>(length, 1));
            at = *end == ',' ? end + 1 : end;
        }
        return list;
    }();
    const bool listed_in_place = std::ranges::any_of(inplace_ranges, [&](const auto& range) {
        return address < range.second && address + size > range.first;
    });
    if (write && write_regions && size <= 256_MB && address < inplace_from && !listed_in_place) {
        if (Region* region = MakeRegion(address, size)) {
            ++stats.region_binds;
            memo_kind = MemoKind::Region; // OPT-003
            memo_region = region;
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
    }
    // FIX-170 (diagnosis, RYGT_NATIVE_SNAPSHOT_BELOW=<bytes>): small read-only shader buffers
    // (constants) are copied when the draw is recorded instead of read in place by the GPU later.
    if (!mirror && !write && size <= SnapshotBelow()) {
        last_bound = Bound::InPlace;
        return SnapshotRange(address, size);
    }
    if (mirror && !write) {
        if (const auto mirrored = MirrorBuffer(address, size)) {
            last_bound = Bound::Mirror; // NAT-150
            return mirrored;
        }
    }
    last_bound = Bound::InPlace;
    if (write) {
        DropMirrors(address, size); // NAT-150: the GPU writes guest memory there
    }
    if (auto* direct = r.buffer_cache.GetDirectMemory();
        direct && direct->EnsureImported(address, size)) {
        if (const auto found = direct->FindBuffer(address, size)) {
            if (!mirror) {
                memo_kind = MemoKind::InPlace; // OPT-003 (vertices and indices keep looking:
            }                                  // they become mirrors when bound again)
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
    return SnapshotRange(address, size);
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::SnapshotRange(VAddr address, u64 size) {
    // A snapshot of what the range holds now, in the list's upload memory (zeros where the CPU
    // can't read it: NAT-106).
    const auto [buffer, offset, data] = Allocate(size, r.instance.StorageMinAlignment());
    const u64 readable = Native::ReadablePrefix(reinterpret_cast<const void*>(address), size);
    // FIX-160: the game can release part of the range meanwhile (zeros there then).
    if (readable != 0 &&
        !Core::RytyMemory::GuardedCopy(data, reinterpret_cast<const void*>(address), readable)) {
        r.memory->CopySparseMemory(address, data, readable);
    }
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
        ApplyRewrite(*entry); // NAT-120
        if (Native::WatchedTexture(entry->info.guest_address, entry->info.guest_address + 1) &&
            Native::WatchSample()) {
            LOG_WARNING(Render_Vulkan, "FIX-153 watch: draw into target {:#x} {}x{} layers {} "
                        "({:#x} bytes), view layers {}+{} mip {}",
                        entry->info.guest_address, entry->info.size.width,
                        entry->info.size.height, entry->info.resources.layers,
                        entry->info.guest_size, desc.view_info.range.base.layer,
                        desc.view_info.range.extent.layers, desc.view_info.range.base.level);
        }
        out.colors[cb] = entry;
        // FIX-154: the draw covers the target it names, not the larger picture it may share
        // (a clear of the small one must leave the rest alone).
        const u32 mip = desc.view_info.range.base.level;
        width = std::min<u32>(width, std::max(std::min(entry->info.size.width,
                                                       desc.info.size.width) >> mip, 1u));
        height = std::min<u32>(height, std::max(std::min(entry->info.size.height,
                                                         desc.info.size.height) >> mip, 1u));
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
            ApplyRewrite(*entry); // NAT-120
            if (regs.depth_buffer.StencilValid() && entry->info.props.has_stencil) {
                auto& owners = stencil_owners[regs.depth_buffer.StencilAddress()];
                if (std::ranges::find(owners, entry) == owners.end()) {
                    owners.push_back(entry);
                }
            }
            bool meta_clear = false;
            bool fresh = false;
            if (htile != 0) {
                known_meta.insert(htile);
                meta_clear = cleared_meta.erase(htile) != 0;
                // FIX-156: a depth buffer with HTILE the GPU never drew into starts out cleared,
                // as in the old renderer (a new depth image's HTILE counts as cleared). The game
                // fast-clears the HTILE before the first pass that uses the buffer, while that
                // memory is not yet known as metadata. GT Sport draws the track from above once
                // while a race loads, and the cockpit's night lighting reads a map made of that
                // depth: started from the zeros of fresh memory, every fragment failed the depth
                // test, the map came out empty, and the cockpit went dark but for the odd frame.
                if (!meta_clear && desc.info.meta_info.htile_addr != 0 && !entry->from_cpu &&
                    entry->write_version.load(std::memory_order_relaxed) == 0) {
                    meta_clear = fresh = true;
                    ++stats.meta_clears;
                }
            }
            const auto aspects =
                DepthStencilClearAspects(regs.depth_buffer, regs.depth_control,
                                         regs.depth_render_control, meta_clear);
            const bool depth_clear = bool(aspects & vk::ImageAspectFlagBits::eDepth);
            const bool stencil_clear = bool(aspects & vk::ImageAspectFlagBits::eStencil);
            if (Native::WatchedTexture(entry->info.guest_address, entry->info.guest_address + 1) &&
                Native::WatchSample()) {
                const auto& dc = regs.depth_control;
                LOG_WARNING(Render_Vulkan, "FIX-156 watch: depth pass {:#x} {}x{} htile {:#x} "
                            "(tile surface {}) version {} meta clear {}{} -> depth clear {} ({}) "
                            "stencil clear {} | test {} write {} func {} stencil {} | render "
                            "control clear {}/{} (frame {})",
                            entry->info.guest_address, entry->info.size.width,
                            entry->info.size.height, htile,
                            u32(regs.depth_buffer.z_info.tile_surface_enable),
                            entry->write_version.load(), meta_clear, fresh ? " (fresh)" : "",
                            depth_clear, regs.depth_clear, stencil_clear, u32(dc.depth_enable),
                            u32(dc.depth_write_enable), u32(dc.depth_func),
                            u32(dc.stencil_enable),
                            u32(regs.depth_render_control.depth_clear_enable),
                            u32(regs.depth_render_control.stencil_clear_enable), frame);
            }
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
            width = std::min<u32>(width, std::min(entry->info.size.width, desc.info.size.width));
            height =
                std::min<u32>(height, std::min(entry->info.size.height, desc.info.size.height));
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
        // SHD-002: each stage binds in its own set, from its own start.
        plans[i] = PlanStageResourceBindings(*stage,
                                             Shader::Backend::StageStart(u32(stage->sw_stage)));
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
    // DIAG-150 (SHADGT_WATCH_SHADERS=<hex,...>): a watched shader's buffers, how they are bound
    // and the first bytes guest memory holds there, and its flattened user data.
    if (Rasterizer::ExposureLogOn()) {
        r.ExposureLogShader(stage, "native"); // FIX-153 (diagnosis)
    }
    static const bool watching = std::getenv("SHADGT_WATCH_SHADERS") != nullptr;
    const bool watched = watching && IsWatchedShader(stage.pgm_hash);
    static std::atomic<u64> watched_binds{};
    const u64 watched_bind = watched ? ++watched_binds : 0;
    const bool log_watched = watched && (watched_bind <= 8 || watched_bind % 300 == 0);
    if (log_watched) {
        std::string flat;
        for (size_t i = 0; i < std::min<size_t>(stage.flattened_ud_buf.size(), 192); ++i) {
            flat += fmt::format(" {:08x}", stage.flattened_ud_buf[i]);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-150: native shader {:#x} use {}: {} dwords flattened:{}",
                    stage.pgm_hash, watched_bind, stage.flattened_ud_buf.size(), flat);
        // FIX-153 (diagnosis): where the walker read each run of flattened dwords.
        std::string sources;
        const auto& src = stage.flattened_ud_src;
        for (size_t i = 0; i < src.size();) {
            size_t j = i + 1;
            while (j < src.size() && src[j] != 0 && src[j] == src[j - 1] + 4) {
                ++j;
            }
            if (src[i] != 0) {
                sources += fmt::format(" [{}..{}] {:#x}", i, j - 1, src[i]);
            }
            i = j;
        }
        LOG_WARNING(Render_Vulkan, "DIAG-150: native shader {:#x} use {} sources:{}",
                    stage.pgm_hash, watched_bind, sources);
    }
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
                // FIX-174 (diagnosis, RYGT_NATIVE_WATCH_BUFFER=<hex address>+<hex size>): the
                // shaders that bind a buffer over that range (the first ones, then now and then).
                static const std::pair<VAddr, VAddr> watch_buffer = [] {
                    const char* env = std::getenv("RYGT_NATIVE_WATCH_BUFFER");
                    if (!env || !*env) {
                        return std::pair<VAddr, VAddr>{};
                    }
                    char* end = nullptr;
                    const VAddr at = std::strtoull(env, &end, 16);
                    const u64 length = end && *end == '+' ? std::strtoull(end + 1, nullptr, 16) : 1;
                    return std::pair<VAddr, VAddr>{at, at + std::max<u64>(length, 1)};
                }();
                if (watch_buffer.first != 0 && base < watch_buffer.second &&
                    base + size > watch_buffer.first) {
                    static std::atomic<u32> watch_logged{};
                    if (Sample(watch_logged)) {
                        LOG_WARNING(Render_Vulkan,
                                    "FIX-174 watch: {} shader {:#x} binds buffer {} {:#x}+{:#x} "
                                    "(frame {})",
                                    stage.sw_stage == Shader::SwStage::Compute ? "compute" : "draw",
                                    stage.pgm_hash, desc.is_written ? "write" : "read", base,
                                    size, frame);
                    }
                }
                const auto log_buffer = [&](const char* how) {
                    if (!log_watched) {
                        return;
                    }
                    std::array<u32, 4> head{};
                    const bool read = Core::RytyMemory::GuardedCopy(head.data(),
                                                        reinterpret_cast<const void*>(base),
                                                        std::min<u64>(size, sizeof(head)));
                    LOG_WARNING(Render_Vulkan,
                                "DIAG-150: native shader {:#x} use {} buffer {} {} {:#x}+{:#x} "
                                "stride {} records {:#x}: {}; memory {}",
                                stage.pgm_hash, watched_bind, buffer_index,
                                desc.is_written ? "write" : "read", base, size, vsharp.GetStride(),
                                vsharp.num_records, how,
                                read ? fmt::format("{:08x} {:08x} {:08x} {:08x}", head[0], head[1],
                                                   head[2], head[3])
                                     : std::string{"unreadable"});
                };
                if (direct && desc.IsDirectEligible() && size < (u64{1} << 32) &&
                    (desc.is_written || size > SnapshotBelow()) && // FIX-170
                    !RegionOver(base, size) && direct->EnsureImported(base, size)) {
                    // D3: read in place through the page table; nothing bound.
                    direct_entries[buffer_index] = {u32(base), u32(base >> 32), u32(size), 1u};
                    if (InPlaceSampled()) {
                        in_place.Note(2, base, size); // NAT-142
                    }
                    log_buffer("in place through the page table (D3)");
                } else if (const auto found = GuestBuffer(base, size, desc.is_written)) {
                    log_buffer(last_bound == Bound::Region   ? "region"
                               : last_bound == Bound::Mirror ? "mirror"
                                                             : "in place");
                    const u64 aligned = Common::AlignDown(found->second, alignment);
                    const u64 adjust = found->second - aligned;
                    b.push.AddOffset(binding.buffer, static_cast<u32>(adjust));
                    info = {found->first, aligned, size + adjust};
                    b.writes_buffers |= desc.is_written;
                    if (InPlaceSampled()) {
                        in_place.Note(BoundKind(desc.is_written ? 4 : 3), base, size); // NAT-142
                    }
                    if (desc.is_written) {
                        b.gpu_writes.emplace_back(base, size); // NAT-119
                    }
                }
            }
        }
        b.buffers.push_back(info);
        b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
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
            b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
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
        // OPT-002: the picture and views this T# resolved to last time.
        ++stats.texture_lookups;
        TextureMemo* memo = nullptr;
        std::array<u64, 4> raw_sharp{};
        static_assert(sizeof(tsharp) == sizeof(raw_sharp));
        std::memcpy(raw_sharp.data(), &tsharp, sizeof(raw_sharp));
        const u32 memo_flags = u32(storage) | (u32(image_desc.is_array) << 1) |
                               (u32(image_desc.is_depth) << 2) |
                               (u32(image_desc.mip_fallback_mode) << 3) |
                               ((image_desc.constant_mip_index & 0xff) << 6) |
                               (num_bindings << 16);
        if (num_bindings <= 4 && stage.pgm_hash != 0xaa3822a3 && !r.FrameDumpCollecting()) {
            memo = &texture_memos[XXH3_64bits_withSeed(raw_sharp.data(), sizeof(raw_sharp),
                                                       memo_flags) &
                                  (texture_memos.size() - 1)];
            if (memo->entry && memo->flags == memo_flags && memo->sharp == raw_sharp &&
                memo->generation == registry->Generation() + unmap_generation) {
                Native::ImageEntry* entry = memo->entry;
                entry->last_use.store(frame, std::memory_order_relaxed);
                if (Native::WatchedTexture(entry->info.guest_address,
                                           entry->info.guest_address + 1)) {
                    WatchBind(*entry, tsharp, storage, true, stage.pgm_hash); // FIX-157
                }
                ApplyRewrite(*entry); // NAT-120
                if (!storage && entry->from_cpu) {
                    RefreshSources(*entry); // NAT-104
                }
                for (u32 i = 0; i < num_bindings; ++i) {
                    b.images.push_back({VK_NULL_HANDLE, memo->views[i], vk::ImageLayout::eGeneral});
                }
                (storage ? b.stored : b.sampled).push_back(entry);
                add_write();
                ++stats.texture_memo_hits;
                continue;
            }
        }
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
        if (Native::WatchedTexture(desc.info.guest_address, desc.info.guest_address + 1)) {
            WatchBind(*entry, tsharp, storage, false, stage.pgm_hash); // FIX-157 (diagnosis)
        }
        ApplyRewrite(*entry); // NAT-120 (NAT-133: textures too)
        if (!storage && r.FrameDumpCollecting()) {
            // FIX-152 (diagnosis): RYGT_FRAME_DUMP_ALL_TEXTURES=1 dumps uploaded ones too.
            static const bool dump_all = [] {
                const char* env = std::getenv("RYGT_FRAME_DUMP_ALL_TEXTURES");
                return env && env[0] == '1';
            }();
            bool gpu_written = dump_all || !entry->from_cpu;
            if (!gpu_written) {
                std::scoped_lock lock{entry->sources_mutex};
                gpu_written = !entry->sources.empty();
            }
            if (gpu_written) {
                auto key = Rasterizer::FrameDumpTextureKey(desc.info); // NAT-123
                if (std::ranges::find(dump_textures, key, &decltype(dump_textures)::value_type::first) ==
                    dump_textures.end()) {
                    dump_textures.emplace_back(std::move(key), entry);
                }
            }
        }
        if (!storage && entry->from_cpu) {
            // (part of the textures' time; reported apart; OPT-001: on timed draws only)
            PhaseClock sources_clock{stats.phase_ns[4]};
            RefreshSources(*entry); // NAT-104
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
        if (memo) {
            // OPT-002: remembered for the next draws reading this T# the same way.
            memo->sharp = raw_sharp;
            memo->flags = memo_flags;
            memo->entry = entry;
            for (u32 i = 0; i < num_bindings; ++i) {
                memo->views[i] = b.images[first + i].imageView;
            }
            memo->generation = registry->Generation() + unmap_generation;
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
        b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
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
    // SHD-002: each stage's writes go to its own set; one set may be pushed.
    std::array<boost::container::small_vector<vk::WriteDescriptorSet, 32>,
               Shader::Backend::MaxDescriptorSets>
        by_set;
    for (size_t i = 0; i < b.writes.size(); ++i) {
        by_set[i < b.write_sets.size() ? b.write_sets[i] : 0].push_back(b.writes[i]);
    }
    const vk::Device vk_device = r.instance.GetDevice();
    for (u32 index = 0; index < by_set.size(); ++index) {
        auto& writes = by_set[index];
        if (writes.empty()) {
            continue;
        }
        if (index == pipeline.PushSet()) {
            cmdbuf.pushDescriptorSetKHR(bind_point, pipeline.GetLayout(), index, writes);
            continue;
        }
        const vk::DescriptorSetLayout layout = pipeline.SetLayout(index);
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
            if (vk_device.allocateDescriptorSets(&alloc,
                                                 reinterpret_cast<vk::DescriptorSet*>(&raw)) ==
                vk::Result::eSuccess) {
                set = raw;
            }
        }
        if (!set) {
            static std::atomic<u32> logged{};
            if (Sample(logged)) {
                LOG_ERROR(Render_Vulkan, "NAT-100: no descriptor set for a draw");
            }
            continue;
        }
        for (auto& write : writes) {
            write.dstSet = set;
        }
        vk_device.updateDescriptorSets(writes, {});
        cmdbuf.bindDescriptorSets(bind_point, pipeline.GetLayout(), index, set, {});
    }
}

void NativeRenderer::ResolveVertexBuffers(const GraphicsPipeline& pipeline,
                                          const AmdGpu::Regs& regs, VertexIndexState& v) {
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(v.attributes, v.input_bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    // OPT-004: a draw's attributes mostly read one mesh (~12 V#s a draw in GT Sport races): a
    // range inside the mirror the previous attribute was bound from, while it is there with
    // nothing to copy in, is bound from it without a lookup (what the lookup would return).
    Mirror* near = nullptr;
    u64 near_generation = 0;
    const bool sampled = InPlaceSampled();
    for (const auto& guest : guest_buffers) {
        std::optional<std::pair<vk::Buffer, u64>> found;
        u64 size = 0;
        const VAddr base = guest.base_address;
        const u64 sharp_size = guest.GetSize();
        if (near && !sampled && base != 0 && sharp_size > 0 &&
            near_generation == buffer_generation && base >= near->begin &&
            base + sharp_size <= near->end && near->dirty_pages.empty()) {
            size = sharp_size;
            found = std::pair{vk::Buffer{Handle<VkBuffer_T>(near->buffer->NativeHandle())},
                              u64{base - near->begin}};
            ++stats.mirror_binds;
            ++stats.vertex_near;
        } else if (base != 0 && sharp_size > 0 && r.memory->MappedAreaEnd(base, base + 1) != 0) {
            size = r.memory->ClampRangeSize(base, sharp_size);
            found = GuestBuffer(base, size, false, true);
            if (sampled) {
                in_place.Note(BoundKind(0), base, size); // NAT-142
            }
            near = found ? last_mirror : nullptr;
            near_generation = buffer_generation;
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
    const auto found = GuestBuffer(address, u64(regs.num_indices) * index_size, false, true);
    if (!found) {
        return false;
    }
    if (InPlaceSampled()) {
        in_place.Note(BoundKind(1), address, u64(regs.num_indices) * index_size); // NAT-142
    }
    v.index_buffer = found->first;
    v.index_offset = found->second;
    return true;
}

void NativeRenderer::RecordVertexIndex(const VertexIndexState& v) {
    Stream& s = stream;
    const CmdBuf cmdbuf{s.cmd};
    const bool dynamic_input = r.instance.IsVertexInputDynamicState();
    // OPT-005: what the list already has is not recorded again (vertex input state and vertex
    // and index bindings outlive pipeline binds; the list's other pipelines are compute).
    const bool known = s.vertex_state_known;
    s.vertex_state_known = true;
    if (dynamic_input) {
        if (!known || !std::ranges::equal(s.input_bindings, v.input_bindings) ||
            !std::ranges::equal(s.attributes, v.attributes)) {
            cmdbuf.setVertexInputEXT(v.input_bindings, v.attributes);
            s.input_bindings.assign(v.input_bindings.begin(), v.input_bindings.end());
            s.attributes.assign(v.attributes.begin(), v.attributes.end());
        } else {
            ++stats.vertex_skipped;
        }
    }
    if (const u32 count = static_cast<u32>(v.buffers.size()); count != 0) {
        const bool same = known && std::ranges::equal(s.vertex_buffers, v.buffers) &&
                          std::ranges::equal(s.vertex_offsets, v.offsets) &&
                          (dynamic_input || (std::ranges::equal(s.vertex_sizes, v.sizes) &&
                                             std::ranges::equal(s.vertex_strides, v.strides)));
        if (!same) {
            if (dynamic_input) {
                cmdbuf.bindVertexBuffers(0, count, v.buffers.data(), v.offsets.data());
            } else {
                cmdbuf.bindVertexBuffers2(0, count, v.buffers.data(), v.offsets.data(),
                                          v.sizes.data(), v.strides.data());
            }
            s.vertex_buffers.assign(v.buffers.begin(), v.buffers.end());
            s.vertex_offsets.assign(v.offsets.begin(), v.offsets.end());
            s.vertex_sizes.assign(v.sizes.begin(), v.sizes.end());
            s.vertex_strides.assign(v.strides.begin(), v.strides.end());
        }
    }
    if (v.index_buffer) {
        if (!known || s.index_buffer != v.index_buffer || s.index_offset != v.index_offset ||
            s.index_type != v.index_type) {
            cmdbuf.bindIndexBuffer(v.index_buffer, v.index_offset, v.index_type);
            s.index_buffer = v.index_buffer;
            s.index_offset = v.index_offset;
            s.index_type = v.index_type;
        }
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

void NativeRenderer::Written(const Native::ImageEntry* entry, u64 at) {
    if (!entry) {
        return;
    }
    auto* written = const_cast<Native::ImageEntry*>(entry);
    written->write_version.store(at != 0 ? at : ++write_clock, std::memory_order_relaxed);
    if (Native::WatchedTexture(entry->info.guest_address, entry->info.guest_address + 1)) {
        // FIX-159 (diagnosis, RYGT_FRAME_DUMP_ON_STORE=<seconds>): from that long after start,
        // the first write of a watched picture starts a frame dump there (the frames a
        // time-triggered dump never lands on).
        static const double on_store = [] {
            const char* env = std::getenv("RYGT_FRAME_DUMP_ON_STORE");
            return env && *env ? std::atof(env) : -1.0;
        }();
        static const auto started = std::chrono::steady_clock::now();
        static bool dumped = false;
        if (on_store >= 0 && !dumped &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >=
                on_store) {
            dumped = true;
            Native::g_dump_request = true;
        }
        if (Native::WatchSample()) {
            LOG_WARNING(Render_Vulkan, "FIX-157 watch: {} {:#x} {}x{} layers {} written, version {} "
                        "(frame {})", entry->from_cpu ? "texture" : "target",
                        entry->info.guest_address, entry->info.size.width,
                        entry->info.size.height, entry->info.resources.layers,
                        written->write_version.load(), frame);
        }
    }
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
        if (version == 0 || !source->image || source->reused.load(std::memory_order_relaxed)) {
            continue; // FIX-157: a reused one is stale
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
    for (const auto& pick : picks) {
        ApplyRewrite(*pick.source); // NAT-122
    }
    PrepareImages();  // the texture's own upload goes first
    GlobalBarrier();  // the sources' pictures are written
    const vk::Image dst{Handle<VkImage_T>(texture.image->NativeHandle())};
    for (const auto& pick : picks) {
        const auto& place = pick.place;
        const u32 width = std::min(
            {place.width, std::max(texture.info.size.width >> place.mip, 1u),
             std::max(pick.source->info.size.width >> place.src_mip, 1u)});
        const u32 height = std::min(
            {place.height, std::max(texture.info.size.height >> place.mip, 1u),
             std::max(pick.source->info.size.height >> place.src_mip, 1u)});
        // NAT-125: a layered target (a cube's faces drawn in one pass, the layer from the
        // vertex shader) gives all its layers; NAT-131: from the layer the texture starts at.
        const u32 layers = std::max<u32>(
            1u, std::min<u32>(std::max<u32>(pick.source->image->Desc().layers, 1u) -
                                  u32(place.src_layer),
                              std::max<u32>(texture.image->Desc().layers, 1u) - u32(place.layer)));
        const vk::ImageCopy region{
            .srcSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = u32(place.src_mip),
                            .baseArrayLayer = u32(place.src_layer),
                            .layerCount = layers},
            .srcOffset = {0, 0, 0},
            .dstSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                            .mipLevel = u32(place.mip),
                            .baseArrayLayer = u32(place.layer),
                            .layerCount = layers},
            .dstOffset = {0, 0, 0},
            .extent = {width, height, 1},
        };
        CmdBuf{s.cmd}.copyImage(vk::Image{Handle<VkImage_T>(pick.source->image->NativeHandle())},
                                vk::ImageLayout::eGeneral, dst, vk::ImageLayout::eGeneral, region);
        if (Rasterizer::Traced(texture.info.guest_address, texture.info.guest_size)) {
            LOG_WARNING(Render_Vulkan,
                        "NAT-130: texture {:#x} {}x{} mips {} layers {} <- target {:#x} {}x{} "
                        "layers {} at mip {} layer {} ({} layers, version {})",
                        texture.info.guest_address, texture.info.size.width,
                        texture.info.size.height, texture.info.resources.levels,
                        texture.info.resources.layers, pick.source->info.guest_address,
                        pick.source->info.size.width, pick.source->info.size.height,
                        pick.source->info.resources.layers, place.mip, place.layer, layers,
                        pick.version);
        }
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
        if (entry && r.FrameDumpCollecting() &&
            std::ranges::find(dump_entries, entry) == dump_entries.end()) {
            dump_entries.push_back(entry); // NAT-116
        }
        if (entry && std::ranges::find(s.written, entry) == s.written.end()) {
            s.written.push_back(entry);
        }
    };
    for (const auto* entry : b.sampled) {
        if (entry && std::ranges::find(s.read, entry) == s.read.end()) {
            s.read.push_back(entry); // NAT-140
        }
    }
    if (targets) {
        const auto attached = [&](const Native::ImageEntry* entry) {
            if (entry && std::ranges::find(s.attach_dirty, entry) == s.attach_dirty.end()) {
                s.attach_dirty.push_back(entry); // NAT-140
            }
        };
        for (const auto* entry : targets->colors) {
            attached(entry);
        }
        attached(targets->depth);
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
    // NAT-119: images the shader overwrote in memory are uploaded again (after this work, in
    // GPU order: the next PrepareImages records the uploads behind a barrier).
    for (const auto& [address, size] : b.gpu_writes) {
        if (b.fill_pattern && address == b.fill_address) {
            GpuWrote(address, std::min(size, b.fill_size), &*b.fill_pattern, true); // NAT-120
        } else {
            GpuWrote(address, size, nullptr, false); // NAT-129: what the shader may write
        }
    }
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
    t_timed = (++timing_tick & 15) == 0; // OPT-001
    stats.timed_draws += t_timed ? 1 : 0;
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
                ApplyRewrite(*entry); // NAT-122
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
    // OPT-010: the draw a game job thread prepared, if any.
    static const bool prepare = PrepareEnabled();
    PreparedDraw* prepared = nullptr;
    if (prepare && !r.FrameDumpCollecting()) {
        prepared = TakePrepared(Rasterizer::t_draw_packet);
        stats.prepared_missing += prepared ? 0 : 1;
    }
    struct GiveBack {
        NativeRenderer& n;
        PreparedDraw* p;
        ~GiveBack() {
            if (p) {
                n.FreePrepared(p);
            }
        }
    } give_back{*this, prepared};
    // OPT-012: when the command thread didn't select (it saw the prepared draw), the job
    // thread's selection is installed and trusted.
    bool trusted = false;
    const GraphicsPipeline* pipeline = [&]() -> const GraphicsPipeline* {
        PhaseClock clock{stats.phase_ns[0]};
        if (prepared && !quad_list && r.selected_pipeline == nullptr && prepared->pipeline) {
            if (const GraphicsPipeline* installed = InstallPrepared(*prepared)) {
                trusted = true;
                return installed;
            }
        }
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
    // DIAG-152 (RYGT_NATIVE_SKIP_PS=<hex,...>): draws with these pixel shaders are skipped.
    static const std::vector<u64> skip_ps = [] {
        std::vector<u64> hashes;
        if (const char* env = std::getenv("RYGT_NATIVE_SKIP_PS"); env && *env) {
            const std::string list{env};
            for (size_t start = 0; start < list.size();) {
                const size_t end = std::min(list.find(',', start), list.size());
                hashes.push_back(std::stoull(list.substr(start, end - start), nullptr, 16));
                start = end + 1;
            }
        }
        return hashes;
    }();
    if (!skip_ps.empty()) {
        const auto& stages = pipeline->GetStages();
        const Shader::Info* fs = stages[u32(Shader::SwStage::Fragment)];
        if (fs && std::ranges::find(skip_ps, fs->pgm_hash) != skip_ps.end()) {
            ++stats.skipped;
            return;
        }
    }
    Stream& s = Begin();
    Targets targets;
    {
        PhaseClock clock{stats.phase_ns[1]};
        ResolveTargets(regs, *pipeline, targets);
    }
    if (r.FrameDumpActive() || Rasterizer::ExposureLogOn()) {
        r.FrameDumpPass(regs, targets.state, pipeline); // NAT-117
    }
    // OPT-010: the bindings the job thread worked out ahead, when they hold.
    bool from_prepared = false;
    if (prepared) {
        PhaseClock clock{stats.phase_ns[2]};
        from_prepared = CommitPrepared(*prepared, *pipeline, regs, trusted);
    }
    DrawBindings& b = from_prepared ? prepared->bindings : bindings;
    if (!from_prepared) {
        b.Clear();
        b.push = UserData(regs);
        BindStages(pipeline->GetStages(), regs, false, b);
    }
    if (b.failed) {
        ++stats.skipped;
        return;
    }
    static thread_local VertexIndexState vertex_index;
    const VertexIndexState* vi = &vertex_index;
    {
        PhaseClock clock{stats.phase_ns[5]};
        if (from_prepared && prepared->has_vertices &&
            CommitPreparedVertices(*prepared, *pipeline, regs, is_indexed, index_offset,
                                   quad_triangles, trusted)) {
            vi = prepared->vertex_index.get(); // OPT-011
        } else {
            vertex_index = {};
            ResolveVertexBuffers(*pipeline, regs, vertex_index);
            if (!ResolveIndices(regs, is_indexed, index_offset, quad_triangles, vertex_index)) {
                ++stats.skipped;
                return;
            }
        }
    }
    {
        PhaseClock clock{stats.phase_ns[6]};
        PrepareImages();
    }
    PhaseClock record_clock{stats.phase_ns[7]};
    {
        PhaseClock part{stats.record_ns[0]}; // OPT-011
        // NAT-140: a full barrier only when the draw reads what was written since the last
        // one, or draws into what was read since; a pass that draws into pictures earlier
        // passes drew waits for their attachment writes only; other passes don't wait.
        static const bool pass_barriers = [] {
            const char* env = std::getenv("RYGT_NATIVE_PASS_BARRIERS");
            return !(env && env[0] == '0');
        }();
        const bool pass_change = !s.rendering || !(s.render == targets.state);
        const auto targets_in = [&](const std::vector<const Native::ImageEntry*>& list) {
            if (list.empty()) {
                return false;
            }
            for (const auto* entry : targets.colors) {
                if (entry && std::ranges::find(list, entry) != list.end()) {
                    return true;
                }
            }
            return targets.depth && std::ranges::find(list, targets.depth) != list.end();
        };
        if (!pass_barriers) {
            if (pass_change || NeedsBarrierFor(b)) {
                GlobalBarrier();
            }
        } else if (NeedsBarrierFor(b) || (pass_change && targets_in(s.read))) {
            GlobalBarrier();
        } else if (pass_change && targets_in(s.attach_dirty)) {
            PassBarrier();
        } else if (pass_change) {
            ++stats.passes_unblocked;
        }
    }
    {
        PhaseClock part{stats.record_ns[1]};
        BeginRendering(targets.state);
    }

    // The renderer's dynamic state code, filling this list's state.
    {
        PhaseClock part{stats.record_ns[2]};
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
    {
        PhaseClock part{stats.record_ns[3]};
        RecordVertexIndex(*vi);
    }
    const u32 index_count = vi->index_count;
    const CmdBuf cmdbuf{s.cmd};
    {
        PhaseClock part{stats.record_ns[4]};
        if (s.pipeline != pipeline->Handle()) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
            s.pipeline = pipeline->Handle();
        }
    }
    {
        PhaseClock part{stats.record_ns[5]};
        PushBindings(*pipeline, vk::PipelineBindPoint::eGraphics, b);
    }
    {
        PhaseClock part{stats.record_ns[6]};
        const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
        const auto [vertex_offset, instance_offset] =
            DrawOffsets(regs, vs_info, pipeline->GetFetchShader());
        const u32 instances = regs.num_instances.NumInstances();
        if (quad_triangles) {
            if (index_count != 0) {
                cmdbuf.drawIndexed(index_count, instances, 0, s32(vertex_offset),
                                   instance_offset);
            }
        } else if (is_indexed) {
            cmdbuf.drawIndexed(regs.num_indices, instances, 0, s32(vertex_offset),
                               instance_offset);
        } else {
            cmdbuf.draw(regs.num_indices, instances, vertex_offset, instance_offset);
        }
    }
    {
        PhaseClock part{stats.record_ns[7]};
        NoteWrites(&targets, b);
    }
    ++stats.draws;
    ++s.commands;
    ChunkFlushCheck();
    Report();
}

bool NativeRenderer::PrepareEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_NATIVE_PREPARE");
        return env && env[0] == '1';
    }();
    return enabled;
}

void NativeRenderer::PreparedDraw::Reset() {
    packet = nullptr;
    epoch = 0;
    pipeline = nullptr;
    stage_hashes = {};
    bindings.Clear();
    regions.clear();
    image_ranges.clear();
    buffer_ranges.clear();
    upload_bytes.clear();
    uploads.clear();
    null_images = 0;
    has_vertices = false;
    vertex_hash = 0;
    indexed = false;
    index_offset = 0;
    index_address = 0;
    mirrors.clear();
    in_place.clear();
    for (auto& copy : stage_copies) {
        copy.present = false;
    }
}

void NativeRenderer::ChangeLog::Note(VAddr begin, VAddr end) {
    const u64 next = seq.load(std::memory_order_relaxed) + 1;
    entries[next % entries.size()] = {next, begin, end};
    seq.store(next, std::memory_order_release);
}

bool NativeRenderer::ChangeLog::Untouched(u64 since,
                                          std::span<const std::pair<VAddr, VAddr>> ranges) const {
    const u64 now = seq.load(std::memory_order_acquire);
    if (now == since) {
        return true;
    }
    if (now - since > entries.size()) {
        return false;
    }
    for (u64 at = since + 1; at <= now; ++at) {
        const Entry& entry = entries[at % entries.size()];
        for (const auto& [begin, end] : ranges) {
            if (entry.begin < end && entry.end > begin) {
                return false;
            }
        }
    }
    return true;
}

NativeRenderer::PreparedDraw* NativeRenderer::NewPrepared() {
    std::scoped_lock lock{prepared_pool_mutex};
    if (!prepared_free.empty()) {
        PreparedDraw* draw = prepared_free.back();
        prepared_free.pop_back();
        return draw;
    }
    prepared_all.push_back(std::make_unique<PreparedDraw>());
    return prepared_all.back().get();
}

void NativeRenderer::FreePrepared(PreparedDraw* draw) {
    draw->Reset();
    std::scoped_lock lock{prepared_pool_mutex};
    prepared_free.push_back(draw);
}

bool NativeRenderer::HasPrepared(const void* packet) {
    if (!packet) {
        return false;
    }
    PreparedShard& shard = ShardFor(packet);
    std::scoped_lock lock{shard.mutex};
    return shard.draws.contains(packet);
}

const GraphicsPipeline* NativeRenderer::InstallPrepared(PreparedDraw& p) {
    // As AcquireGraphicsPipeline with the command thread's selection: the stages' user data
    // and flattened user data, the flattened buffers moved in (and the old ones kept here).
    std::array<Rasterizer::SelectedStage, Shader::MaxStageTypes> selected{};
    for (u32 i = 0; i < selected.size(); ++i) {
        const auto& copy = p.stage_copies[i];
        auto& stage = selected[i];
        stage.present = copy.present;
        if (!copy.present) {
            continue;
        }
        stage.pgm_base = copy.pgm_base;
        stage.num_user_data = copy.num_user_data;
        std::copy_n(copy.user_data.begin(), copy.num_user_data, stage.user_data.begin());
        stage.flattened = std::move(p.stage_copies[i].flattened);
    }
    const bool installed = r.InstallDrawStages(p.pipeline, selected, true);
    for (u32 i = 0; i < selected.size(); ++i) {
        p.stage_copies[i].flattened = std::move(selected[i].flattened);
    }
    if (!installed) {
        return nullptr;
    }
    ++stats.prepared_installed;
    return p.pipeline;
}

NativeRenderer::PreparedDraw* NativeRenderer::TakePrepared(const void* packet) {
    if (!packet) {
        return nullptr;
    }
    PreparedShard& shard = ShardFor(packet);
    std::scoped_lock lock{shard.mutex};
    const auto it = shard.draws.find(packet);
    if (it == shard.draws.end()) {
        return nullptr;
    }
    PreparedDraw* draw = it->second;
    shard.draws.erase(it);
    return draw;
}

u64 NativeRenderer::StageHash(const Shader::Info& info) {
    // What the bindings were worked out from: the stage's sharps (the user data registers the
    // program doesn't read hold whatever earlier work left, which a job and the command
    // processor need not agree on).
    static thread_local std::vector<u32> words;
    words.clear();
    const auto add = [](const auto& sharp) {
        const auto* data = reinterpret_cast<const u32*>(&sharp);
        words.insert(words.end(), data, data + sizeof(sharp) / sizeof(u32));
    };
    for (const auto& desc : info.buffers) {
        if (!desc.IsSpecial()) {
            add(desc.GetSharp(info));
        }
    }
    for (const auto& desc : info.images) {
        add(desc.GetSharp(info));
    }
    for (const auto& desc : info.samplers) {
        add(desc.GetSharp(info));
    }
    return XXH3_64bits_withSeed(words.data(), words.size() * sizeof(u32), info.pgm_base);
}

void NativeRenderer::PrepareJobDraw(const void* packet, u32 epoch, const AmdGpu::Regs& regs,
                                    const GraphicsPipeline& pipeline,
                                    std::span<const Shader::Info* const> stages, bool is_indexed,
                                    u32 index_offset) {
    const auto start = std::chrono::steady_clock::now();
    PreparedDraw* p = NewPrepared();
    p->packet = packet;
    p->epoch = epoch;
    p->pipeline = &pipeline;
    // What the lookups depend on, read before them.
    p->image_generation = registry->Generation();
    p->buffer_seq = buffer_log.Seq();
    p->unmap_gen = unmap_generation.load(std::memory_order_acquire);
    p->bindings.push = UserData(regs);
    const u32 count = std::min<u32>(static_cast<u32>(stages.size()), Shader::MaxStageTypes - 1);
    std::array<StageResourceBindings, Shader::MaxStageTypes> plans{};
    bool ok = true;
    size_t total = 0;
    for (u32 i = 0; i < count; ++i) {
        if (const auto* stage = stages[i]) {
            total += stage->buffers.size() + stage->images.size() + stage->samplers.size();
        }
    }
    p->bindings.writes.reserve(total);
    for (u32 i = 0; i < count; ++i) {
        // OPT-012: what the recorder installs (as the command thread's selection would copy).
        auto& copy = p->stage_copies[i];
        const auto* stage = stages[i];
        copy.present = stage != nullptr;
        if (stage) {
            copy.pgm_base = stage->pgm_base;
            copy.num_user_data =
                static_cast<u32>(std::min<size_t>(stage->user_data.size(), copy.user_data.size()));
            std::copy_n(stage->user_data.begin(), copy.num_user_data, copy.user_data.begin());
            copy.flattened.assign(stage->flattened_ud_buf.begin(), stage->flattened_ud_buf.end());
        }
    }
    for (u32 i = 0; ok && i < count; ++i) {
        const auto* stage = stages[i];
        p->stage_hashes[i] = stage ? StageHash(*stage) : 0;
        if (!stage) {
            continue;
        }
        plans[i] = PlanStageResourceBindings(*stage,
                                             Shader::Backend::StageStart(u32(stage->sw_stage)));
        auto buffer_binding = plans[i].buffers;
        ok = PrepareBuffers(*stage, regs, buffer_binding, *p);
    }
    for (u32 i = 0; ok && i < count; ++i) {
        if (const auto* stage = stages[i]) {
            auto texture_binding = plans[i].textures;
            ok = PrepareTextures(*stage, regs, texture_binding, *p);
        }
    }
    if (!ok) {
        ++prepare_stats.failed;
        FreePrepared(p);
        return;
    }
    // OPT-011: the vertex and index buffers too, when they need nothing ordered.
    constexpr u32 VertexStage = static_cast<u32>(Shader::SwStage::Vertex);
    if (VertexStage < count && stages[VertexStage]) {
        p->has_vertices =
            PrepareVertices(*p, regs, pipeline, *stages[VertexStage], is_indexed, index_offset);
    }
    std::vector<PreparedDraw*> dropped;
    {
        PreparedShard& shard = ShardFor(packet);
        std::scoped_lock lock{shard.mutex};
        if (shard.draws.size() > 1024) {
            // Draws the recorder never took (not submitted, or filtered): from a few
            // submissions ago, they go.
            for (auto it = shard.draws.begin(); it != shard.draws.end();) {
                if (epoch - it->second->epoch > 4) {
                    dropped.push_back(it->second);
                    it = shard.draws.erase(it);
                } else {
                    ++it;
                }
            }
        }
        const auto [it, inserted] = shard.draws.try_emplace(packet, p);
        if (!inserted) {
            dropped.push_back(it->second);
            it->second = p;
            ++prepare_stats.replaced;
        }
    }
    prepare_stats.stale += dropped.size();
    for (PreparedDraw* draw : dropped) {
        FreePrepared(draw);
    }
    ++prepare_stats.made;
    prepare_stats.ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::JobGuestBuffer(VAddr address, u64 size,
                                                                         bool write,
                                                                         PreparedDraw& p) {
    {
        std::shared_lock map_lock{buffer_map_mutex};
        if (Region* region = RegionOver(address, size)) {
            if (address < region->begin || address + size > region->end) {
                return std::nullopt; // the region grows over it first
            }
            p.regions.push_back(region);
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
    }
    if (write) {
        return std::nullopt; // a region is made for it (NAT-144), or mirrors dropped
    }
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (direct && direct->EnsureImported(address, size)) {
        if (const auto found = direct->FindBuffer(address, size)) {
            return std::pair{found->first->Handle(), found->second};
        }
    }
    return std::nullopt; // gathered or copied, in GPU order
}

bool NativeRenderer::PrepareBuffers(const Shader::Info& stage, const AmdGpu::Regs& regs,
                                    Shader::Backend::Bindings& binding, PreparedDraw& p) {
    // As BindBuffers, without its side effects.
    DrawBindings& b = p.bindings;
    const u64 alignment = r.instance.StorageMinAlignment();
    auto* direct = stage.uses_direct ? r.buffer_cache.GetDirectMemory() : nullptr;
    static thread_local std::vector<std::array<u32, 4>> direct_entries;
    direct_entries.assign(stage.buffers.size(), {});
    constexpr u64 GpuAddressLimit = 1ULL << 40;
    const auto upload = [&](const void* data, u32 size) {
        const u32 offset = static_cast<u32>(p.upload_bytes.size());
        const auto* bytes = static_cast<const u8*>(data);
        p.upload_bytes.insert(p.upload_bytes.end(), bytes, bytes + size);
        p.uploads.push_back({static_cast<u32>(b.buffers.size()), offset, size});
        return vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, size};
    };
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
                // The recorder's own copy of the stage's user data (the registers the program
                // doesn't read may differ here), uploaded at commit.
                const u32 size = static_cast<u32>(stage.flattened_ud_buf.size() * sizeof(u32));
                if (size != 0) {
                    p.uploads.push_back({static_cast<u32>(b.buffers.size()), 0, size,
                                         static_cast<s32>(stage.sw_stage)});
                    info = vk::DescriptorBufferInfo{VK_NULL_HANDLE, 0, size};
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
                    info = upload(planes.data(), sizeof(planes));
                }
                break;
            case Shader::BufferType::DirectTable: {
                const u32 words = static_cast<u32>(4 + stage.buffers.size() * 4);
                static thread_local std::vector<u32> table;
                table.assign(words, 0);
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
                info = upload(table.data(), words * sizeof(u32));
                break;
            }
            default:
                return false; // compute-only kinds and the unsupported ones: the recorder
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            const u64 base = vsharp.base_address;
            const u64 sharp_size = vsharp.GetSize();
            const bool impossible =
                vsharp.num_records != UINT32_MAX &&
                (sharp_size >= GpuAddressLimit || base + sharp_size > GpuAddressLimit ||
                 (sharp_size >= 1_GB && base != 0 && !r.memory->IsValidMapping(base)));
            const bool unmapped =
                !impossible && base != 0 && sharp_size != 0 && !r.IsMappedStart(base);
            if (base != 0 && sharp_size != 0 && !impossible && !unmapped) {
                const u64 size = r.memory->ClampRangeSize(base, sharp_size);
                bool region_over;
                {
                    std::shared_lock map_lock{buffer_map_mutex};
                    region_over = RegionOver(base, size) != nullptr;
                }
                p.buffer_ranges.emplace_back(base, base + size);
                if (direct && desc.IsDirectEligible() && size < (u64{1} << 32) && !region_over &&
                    direct->EnsureImported(base, size)) {
                    direct_entries[buffer_index] = {u32(base), u32(base >> 32), u32(size), 1u};
                } else if (const auto found = JobGuestBuffer(base, size, desc.is_written, p)) {
                    const u64 aligned = Common::AlignDown(found->second, alignment);
                    const u64 adjust = found->second - aligned;
                    b.push.AddOffset(binding.buffer, static_cast<u32>(adjust));
                    info = {found->first, aligned, size + adjust};
                    b.writes_buffers |= desc.is_written;
                    if (desc.is_written) {
                        b.gpu_writes.emplace_back(base, size); // NAT-119
                    }
                } else {
                    return false;
                }
            }
        }
        b.buffers.push_back(info);
        b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
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
    return true;
}

bool NativeRenderer::PrepareTextures(const Shader::Info& stage, const AmdGpu::Regs& regs,
                                     Shader::Backend::Bindings& binding, PreparedDraw& p) {
    // As BindTextures, without its side effects: pictures that exist only, no rewrites or
    // copies of sources (the recorder does them).
    DrawBindings& b = p.bindings;
    const u64 now_frame = shared_frame.load(std::memory_order_relaxed);
    static thread_local std::array<TextureMemo, 4096> memos{};
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        const u32 num_bindings = image_desc.NumBindings(stage);
        const u32 first = static_cast<u32>(b.images.size());
        const bool storage = image_desc.is_written;
        const auto add_write = [&] {
            b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
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
        const auto bind_null = [&] {
            for (u32 i = 0; i < num_bindings; ++i) {
                b.images.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral});
            }
            ++p.null_images;
            add_write();
        };
        std::array<u64, 4> raw_sharp{};
        static_assert(sizeof(tsharp) == sizeof(raw_sharp));
        std::memcpy(raw_sharp.data(), &tsharp, sizeof(raw_sharp));
        const u32 memo_flags = u32(storage) | (u32(image_desc.is_array) << 1) |
                               (u32(image_desc.is_depth) << 2) |
                               (u32(image_desc.mip_fallback_mode) << 3) |
                               ((image_desc.constant_mip_index & 0xff) << 6) |
                               (num_bindings << 16);
        if (num_bindings > 4 || stage.pgm_hash == 0xaa3822a3) {
            return false; // the recorder (FIX-081's grass shader, long arrays)
        }
        const u64 memo_generation = p.image_generation + p.unmap_gen;
        TextureMemo& memo = memos[XXH3_64bits_withSeed(raw_sharp.data(), sizeof(raw_sharp),
                                                       memo_flags) &
                                  (memos.size() - 1)];
        const VAddr address = tsharp.Address();
        if (memo.entry && memo.flags == memo_flags && memo.sharp == raw_sharp &&
            memo.generation == memo_generation) {
            Native::ImageEntry* entry = memo.entry;
            entry->last_use.store(now_frame, std::memory_order_relaxed);
            for (u32 i = 0; i < num_bindings; ++i) {
                b.images.push_back({VK_NULL_HANDLE, memo.views[i], vk::ImageLayout::eGeneral});
            }
            (storage ? b.stored : b.sampled).push_back(entry);
            p.image_ranges.emplace_back(address, address + std::max<u64>(entry->info.guest_size, 1));
            add_write();
            continue;
        }
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (address == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt) ||
            !r.memory->IsValidGpuMapping(address, 0) ||
            LiverpoolToVK::TrySurfaceFormat(data_fmt, num_fmt) == vk::Format::eUndefined ||
            VideoCore::CheckImageDescriptorGeometry(tsharp, r.instance.GetImageLimits()) !=
                VideoCore::ImageDescriptorGeometryError::None) {
            bind_null();
            continue;
        }
        const VideoCore::TextureCache::ImageDesc desc{tsharp, image_desc};
        if (!r.IsMappedStart(desc.info.guest_address)) {
            bind_null();
            continue;
        }
        Native::ImageEntry* entry = registry->Find(desc.info, storage, now_frame, false);
        if (!entry) {
            return false; // made by the recorder, in GPU order
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
        memo.sharp = raw_sharp;
        memo.flags = memo_flags;
        memo.entry = entry;
        for (u32 i = 0; i < num_bindings; ++i) {
            memo.views[i] = b.images[first + i].imageView;
        }
        memo.generation = memo_generation;
        (storage ? b.stored : b.sampled).push_back(entry);
        p.image_ranges.emplace_back(desc.info.guest_address,
                                    desc.info.guest_address +
                                        std::max<u64>(desc.info.guest_size, 1));
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
        b.write_sets.push_back(u8(Shader::Backend::StageSet(u32(stage.sw_stage)))); // SHD-002
        b.writes.push_back(vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = &b.images.back(),
        });
    }
    return true;
}

u64 NativeRenderer::VertexHash(const GraphicsPipeline& pipeline, const Shader::Info& vs,
                                u32 step_rate_0, u32 step_rate_1) {
    // The V#s the fetch shader reads (what the vertex buffers were worked out from).
    const auto& fetch = pipeline.GetFetchShader();
    std::array<u32, 2 + 4 * MaxVertexBufferCount> words{step_rate_0, step_rate_1};
    u32 count = 2;
    if (!fetch.Empty()) {
        for (const auto& attrib : fetch.attributes) {
            if (count + 4 > words.size()) {
                break;
            }
            const auto sharp = attrib.GetSharp(vs);
            static_assert(sizeof(sharp) == 4 * sizeof(u32));
            std::memcpy(&words[count], &sharp, sizeof(sharp));
            count += 4;
        }
    }
    return XXH3_64bits(words.data(), count * sizeof(u32));
}

std::optional<std::pair<vk::Buffer, u64>> NativeRenderer::JobVertexBuffer(VAddr address, u64 size,
                                                                          PreparedDraw& p,
                                                                          Mirror** mirror_out) {
    // As GuestBufferLookup for vertices and indices (a region, else a mirror, else in place).
    // The caller holds buffer_map_mutex shared.
    *mirror_out = nullptr;
    {
        if (Region* region = RegionOver(address, size)) {
            if (address < region->begin || address + size > region->end) {
                return std::nullopt;
            }
            if (p.regions.empty() || p.regions.back() != region) {
                p.regions.push_back(region);
            }
            return std::pair{vk::Buffer{Handle<VkBuffer_T>(region->buffer->NativeHandle())},
                             u64{address - region->begin}};
        }
        static const bool mirrors_on = [] {
            const char* env = std::getenv("RYGT_NATIVE_MIRRORS");
            return !(env && env[0] == '0');
        }();
        if (mirrors_on && size <= 64_MB) {
            if (Mirror* mirror = MirrorHolding(address, size)) {
                if (std::ranges::find(p.mirrors, mirror) == p.mirrors.end()) {
                    p.mirrors.push_back(mirror);
                }
                *mirror_out = mirror;
                return std::pair{vk::Buffer{Handle<VkBuffer_T>(mirror->buffer->NativeHandle())},
                                 u64{address - mirror->begin}};
            }
        }
    }
    auto* direct = r.buffer_cache.GetDirectMemory();
    if (direct && direct->EnsureImported(address, size)) {
        if (const auto found = direct->FindBuffer(address, size)) {
            p.in_place.emplace_back(address, size);
            return std::pair{found->first->Handle(), found->second};
        }
    }
    return std::nullopt;
}

bool NativeRenderer::PrepareVertices(PreparedDraw& p, const AmdGpu::Regs& regs,
                                     const GraphicsPipeline& pipeline, const Shader::Info& vs,
                                     bool is_indexed, u32 index_offset) {
    if (regs.primitive_type == AmdGpu::PrimitiveType::QuadList) {
        return false; // the recorder makes their triangles
    }
    if (!p.vertex_index) {
        p.vertex_index = std::make_unique<VertexIndexState>();
    }
    VertexIndexState& v = *p.vertex_index;
    v = {};
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(v.attributes, v.input_bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1, vs);
    p.vertex_hash = VertexHash(pipeline, vs, regs.vgt_instance_step_rate_0,
                               regs.vgt_instance_step_rate_1);
    // Regions and mirrors stay put while this thread reads them.
    std::shared_lock map_lock{buffer_map_mutex};
    Mirror* near = nullptr;
    for (const auto& guest : guest_buffers) {
        std::optional<std::pair<vk::Buffer, u64>> found;
        u64 size = 0;
        const VAddr base = guest.base_address;
        const u64 sharp_size = guest.GetSize();
        if (near && base != 0 && sharp_size > 0 && base >= near->begin &&
            base + sharp_size <= near->end) {
            size = sharp_size; // OPT-004 on this thread
            found = std::pair{vk::Buffer{Handle<VkBuffer_T>(near->buffer->NativeHandle())},
                              u64{base - near->begin}};
        } else if (base != 0 && sharp_size > 0 && r.memory->MappedAreaEnd(base, base + 1) != 0) {
            size = r.memory->ClampRangeSize(base, sharp_size);
            found = JobVertexBuffer(base, size, p, &near);
            if (!found) {
                return false;
            }
        }
        if (size != 0) {
            p.buffer_ranges.emplace_back(base, base + size);
        }
        v.buffers.push_back(found ? found->first : vk::Buffer{});
        v.offsets.push_back(found ? found->second : 0);
        v.sizes.push_back(found ? size : 0);
        v.strides.push_back(guest.GetStride());
    }
    p.indexed = is_indexed;
    p.index_offset = index_offset;
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    v.index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    v.index_count = regs.num_indices;
    if (is_indexed) {
        const VAddr address =
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
        const u64 bytes = u64(regs.num_indices) * index_size;
        if (bytes == 0) {
            return false;
        }
        Mirror* unused = nullptr;
        const auto found = JobVertexBuffer(address, bytes, p, &unused);
        if (!found) {
            return false;
        }
        p.buffer_ranges.emplace_back(address, address + bytes);
        p.index_address = address;
        v.index_buffer = found->first;
        v.index_offset = found->second;
    }
    return true;
}

bool NativeRenderer::CommitPreparedVertices(PreparedDraw& p, const GraphicsPipeline& pipeline,
                                            const AmdGpu::Regs& regs, bool is_indexed,
                                            u32 index_offset, bool quad_triangles,
                                            bool trusted) {
    // (The ranges were checked with the bindings'.)
    const auto reject = [&] {
        ++stats.prepared_vertices_rejected;
        return false;
    };
    if (quad_triangles || p.indexed != is_indexed || p.index_offset != index_offset) {
        return reject();
    }
    if (!trusted &&
        VertexHash(pipeline, pipeline.GetStage(Shader::SwStage::Vertex),
                   regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1) != p.vertex_hash) {
        return reject();
    }
    const auto& v = *p.vertex_index;
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    if (v.index_count != regs.num_indices ||
        v.index_type != (is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32)) {
        return reject();
    }
    if (is_indexed) {
        const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
        if (p.index_address !=
            regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size) {
            return reject();
        }
    }
    // The ordered part: pages the CPU wrote into the mirrors, and the ranges read in place
    // (noted as mirror candidates).
    for (Mirror* mirror : p.mirrors) {
        RefreshMirror(*mirror);
        mirror->used_frame = frame;
        ++stats.mirror_binds;
    }
    for (const auto& [address, size] : p.in_place) {
        MirrorBuffer(address, size);
    }
    ++stats.prepared_vertices;
    return true;
}

bool NativeRenderer::CommitPrepared(PreparedDraw& p, const GraphicsPipeline& pipeline,
                                    const AmdGpu::Regs& regs, bool trusted) {
    const auto reject = [&](u32 why) {
        ++stats.prepared_rejected[why];
        return false;
    };
    if (p.pipeline != &pipeline) {
        return reject(0);
    }
    const auto stages = pipeline.GetStages();
    // OPT-012: stages installed from this prepared draw are what it was worked out from.
    for (u32 i = 0; !trusted && i < stages.size() && i < p.stage_hashes.size(); ++i) {
        if ((stages[i] ? StageHash(*stages[i]) : 0) != p.stage_hashes[i]) {
            return reject(1);
        }
    }
    for (const auto& upload : p.uploads) {
        if (upload.flat_stage >= 0) {
            const Shader::Info* stage = stages[upload.flat_stage];
            if (!stage || stage->flattened_ud_buf.size() * sizeof(u32) != upload.size) {
                return reject(1);
            }
        }
    }
    if (p.unmap_gen != unmap_generation.load(std::memory_order_acquire)) {
        return reject(2);
    }
    if (!registry->Untouched(p.image_generation, p.image_ranges)) {
        return reject(3);
    }
    SortRegionWrites(); // CPU writes since (may drop mirrors)
    if (!buffer_log.Untouched(p.buffer_seq, p.buffer_ranges)) {
        return reject(4);
    }
    // The ordered part, as BindBuffers and BindTextures do it.
    for (Region* region : p.regions) {
        RefreshRegion(*region);
        ++stats.region_binds;
    }
    const u64 alignment = r.instance.StorageMinAlignment();
    for (const auto& upload : p.uploads) {
        const u8* source = p.upload_bytes.data() + upload.offset;
        u32 size = upload.size;
        if (upload.flat_stage >= 0) {
            source = reinterpret_cast<const u8*>(stages[upload.flat_stage]->flattened_ud_buf.data());
        }
        const auto [buffer, offset, data] = Allocate(size, alignment);
        std::memcpy(data, source, size);
        p.bindings.buffers[upload.info] = vk::DescriptorBufferInfo{buffer, offset, size};
    }
    for (const auto* stored : p.bindings.stored) {
        auto& entry = *const_cast<Native::ImageEntry*>(stored);
        entry.last_use.store(frame, std::memory_order_relaxed);
        ApplyRewrite(entry); // NAT-120
    }
    for (const auto* sampled : p.bindings.sampled) {
        auto& entry = *const_cast<Native::ImageEntry*>(sampled);
        entry.last_use.store(frame, std::memory_order_relaxed);
        ApplyRewrite(entry); // NAT-120, NAT-133
        if (entry.from_cpu) {
            RefreshSources(entry); // NAT-104
        }
    }
    // The viewport transform the shaders get, from the registers as recorded.
    const Shader::PushData fresh = UserData(regs);
    p.bindings.push.xoffset = fresh.xoffset;
    p.bindings.push.xscale = fresh.xscale;
    p.bindings.push.yoffset = fresh.yoffset;
    p.bindings.push.yscale = fresh.yscale;
    p.bindings.push.quad_sample_mask = fresh.quad_sample_mask;
    stats.null_images += p.null_images;
    ++stats.prepared_used;
    return true;
}

void NativeRenderer::GpuWrote(VAddr address, u64 size, const std::array<u32, 2>* pattern,
                              bool exact) {
    const u64 clock = ++write_clock; // NAT-148: when the memory was rewritten
    registry->GpuWrote(address, size, pattern, exact, clock);
    // A fill of a stencil plane: depth images whose stencil is there clear it at next use.
    const auto it = stencil_owners.find(address);
    if (it == stencil_owners.end() || !pattern) {
        return;
    }
    const u32 lo = (*pattern)[0];
    const bool uniform = (*pattern)[0] == (*pattern)[1] && lo == (lo & 0xffu) * 0x01010101u;
    if (!uniform) {
        return;
    }
    for (auto* entry : it->second) {
        entry->stencil_clear_pending = true;
        entry->stencil_clear_value = static_cast<u8>(lo);
        entry->rewrite_clock = std::max(entry->rewrite_clock, clock);
    }
}

const Native::Rhi::Buffer* NativeRenderer::PatternBuffer(u32 lo, u32 hi, u64 bytes) {
    // NAT-134: a device buffer holding lo, hi, lo, hi... for fills, made once per pattern and
    // grown when a bigger picture needs it (only ever read afterwards).
    const u64 key = u64(lo) | (u64(hi) << 32);
    auto& slot = fill_patterns[key];
    if (slot && slot->Size() >= bytes) {
        return slot.get();
    }
    const u64 size = std::max<u64>(Common::AlignUp(bytes, 64_KB), 1_MB);
    auto buffer = device->CreateBuffer({.size = size,
                                        .usage = Native::Rhi::BufferUsageTransferDst |
                                                 Native::Rhi::BufferUsageTransferSrc,
                                        .memory = Native::Rhi::MemoryKind::DeviceLocal,
                                        .name = "native fill pattern"});
    if (!buffer) {
        return slot.get();
    }
    Stream& s = Begin();
    EndRendering();
    const vk::Buffer handle{Handle<VkBuffer_T>(buffer->NativeHandle())};
    const auto transfer_barrier = [&] {
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
        };
        s.cmd.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    };
    if (lo == hi) {
        s.cmd.fillBuffer(handle, 0, VK_WHOLE_SIZE, lo);
    } else {
        // 8 KB of the pattern written in, then doubled by copies until the buffer is full.
        std::array<u32, 2048> words;
        for (size_t i = 0; i < words.size(); i += 2) {
            words[i] = lo;
            words[i + 1] = hi;
        }
        u64 filled = sizeof(words);
        s.cmd.updateBuffer(handle, 0, filled, words.data());
        while (filled < size) {
            transfer_barrier();
            const u64 chunk = std::min(filled, size - filled);
            s.cmd.copyBuffer(handle, handle, vk::BufferCopy{.srcOffset = 0, .dstOffset = filled,
                                                            .size = chunk});
            filled += chunk;
        }
    }
    transfer_barrier();
    ++s.commands;
    if (slot) {
        retired_patterns.push_back(std::move(slot)); // still read by recorded copies
    }
    slot = std::move(buffer);
    return slot.get();
}

bool NativeRenderer::FillImage(Native::ImageEntry& entry, u32 lo, u32 hi, u32 rows) {
    // NAT-132/134: a picture filled with the pattern (two dwords, repeated), as memory holds it
    // after a fill: copied in from a buffer holding it (any format: every texel is the pattern's
    // bytes). `rows` != 0: only those rows of the first mip and layer (NAT-124).
    const auto& info = entry.info;
    const u32 texel = info.props.is_depth
                          ? (info.pixel_format == vk::Format::eD16Unorm ||
                                     info.pixel_format == vk::Format::eD16UnormS8Uint
                                 ? 2u
                                 : 4u)
                          : info.num_bits / 8;
    if (texel == 0 || info.props.is_block || info.num_samples != 1 || !entry.image ||
        (lo != hi && texel < 8 && texel != 4)) {
        return false;
    }
    const auto& desc = entry.image->Desc();
    const u32 layers = desc.dim == Native::Rhi::ImageDim::D3 ? 1u : std::max(desc.layers, 1u);
    const u32 depth = desc.dim == Native::Rhi::ImageDim::D3 ? std::max(desc.depth, 1u) : 1u;
    const u32 mips = rows ? 1u : std::max(desc.mips, 1u);
    const u32 height0 = rows ? std::min(rows, desc.height) : desc.height;
    const u64 bytes = u64(desc.width) * height0 * texel * (rows ? 1u : layers * depth);
    const Native::Rhi::Buffer* pattern = PatternBuffer(lo, hi, bytes);
    if (!pattern || pattern->Size() < bytes) {
        return false;
    }
    Stream& s = Begin();
    GlobalBarrier(); // the work before is done with the picture
    const vk::Buffer buffer{Handle<VkBuffer_T>(pattern->NativeHandle())};
    boost::container::small_vector<vk::BufferImageCopy, 16> copies;
    for (u32 m = 0; m < mips; ++m) {
        const u32 w = std::max(desc.width >> m, 1u);
        const u32 h = rows ? height0 : std::max(desc.height >> m, 1u);
        const u32 d = std::max(depth >> m, 1u);
        copies.push_back(vk::BufferImageCopy{
            .bufferOffset = 0,
            .bufferRowLength = w,
            .bufferImageHeight = h,
            .imageSubresource{.aspectMask = info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                                : vk::ImageAspectFlagBits::eColor,
                              .mipLevel = m,
                              .baseArrayLayer = 0,
                              .layerCount = rows ? 1u : layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {w, h, d},
        });
    }
    s.cmd.copyBufferToImage(buffer, vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())},
                            vk::ImageLayout::eGeneral, static_cast<u32>(copies.size()),
                            copies.data());
    Written(&entry, entry.rewrite_clock); // NAT-148
    s.buffers_written = true; // the next work waits for the fill
    ++s.commands;
    return true;
}

void NativeRenderer::ApplyRewrite(Native::ImageEntry& entry) {
    using Rewrite = Native::ImageEntry::Rewrite;
    if (entry.rewrite == Rewrite::None && !entry.stencil_clear_pending) {
        return;
    }
    const auto& info = entry.info;
    if (Native::WatchedTexture(info.guest_address, info.guest_address + 1) &&
        Native::WatchSample()) {
        LOG_WARNING(Render_Vulkan, "FIX-153 watch: apply {} to {} {:#x} ({:#x} bytes of {:#x}, "
                    "pattern {:#x} {:#x})",
                    entry.rewrite == Rewrite::Clear    ? "clear"
                    : entry.rewrite == Rewrite::Upload ? "upload"
                                                       : "stencil clear",
                    entry.from_cpu ? "texture" : "target", info.guest_address, entry.rewrite_bytes,
                    info.guest_size, entry.rewrite_pattern[0], entry.rewrite_pattern[1]);
    }
    const Rewrite rewrite = std::exchange(entry.rewrite, Rewrite::None);
    const bool stencil = std::exchange(entry.stencil_clear_pending, false) &&
                         info.props.has_stencil && entry.image;
    const u32 lo = entry.rewrite_pattern[0];
    const u32 hi = entry.rewrite_pattern[1];
    const bool partial = rewrite == Rewrite::Clear && entry.rewrite_bytes < info.guest_size;
    // What a clear command can do: a depth plane filled with one value (32-bit float, or 16-bit
    // unorm), a color picture filled with zeros.
    std::optional<float> depth_value;
    bool color_zero = false;
    if (rewrite == Rewrite::Clear && !partial) {
        if (info.props.is_depth) {
            const auto format = info.pixel_format;
            const bool d16 = format == vk::Format::eD16Unorm || format == vk::Format::eD16UnormS8Uint;
            if (!d16 && lo == hi) {
                depth_value = std::bit_cast<float>(lo);
            } else if (d16 && lo == hi && (lo >> 16) == (lo & 0xffffu)) {
                depth_value = float(lo & 0xffffu) / 65535.0f;
            }
        } else if (!info.props.is_block && lo == 0 && hi == 0) {
            color_zero = true;
        }
    }
    if (rewrite == Rewrite::Clear && partial) {
        // NAT-124: a fill of the first bytes only: those rows of the first mip and layer.
        const u32 texel = info.props.is_depth ? 4u : std::max(info.num_bits / 8, 1u);
        const u64 row_bytes = u64(std::max(info.pitch, info.size.width)) * texel;
        const u32 rows = u32(std::min<u64>((entry.rewrite_bytes + row_bytes - 1) / row_bytes,
                                           info.size.height));
        if (rows != 0 && FillImage(entry, lo, hi, rows)) {
            ++stats.rewrite_partial;
        }
    } else if (rewrite == Rewrite::Clear && !depth_value && !color_zero) {
        // NAT-132: other patterns (GT Sport fills pictures in its scratch pool with non-zero
        // values) are filled in on the GPU; uploading them from memory cost 1-3 GB a second in
        // the menus.
        if (FillImage(entry, lo, hi, 0)) {
            ++stats.rewrite_fills;
        } else if (info.num_samples == 1) {
            registry->RequestUpload(entry); // in GPU order, before the next draw or dispatch
            ++stats.rewrite_uploads;
        }
    } else if (rewrite == Rewrite::Upload && info.num_samples == 1) {
        registry->RequestUpload(entry);
        ++stats.rewrite_uploads;
    }
    if (!depth_value && !color_zero && !stencil) {
        return;
    }
    Stream& s = Begin();
    GlobalBarrier(); // the work before is done with the picture
    const vk::Image image{Handle<VkImage_T>(entry.image->NativeHandle())};
    const auto range = [](vk::ImageAspectFlags aspect) {
        return vk::ImageSubresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = VK_REMAINING_ARRAY_LAYERS,
        };
    };
    if (depth_value) {
        s.cmd.clearDepthStencilImage(image, vk::ImageLayout::eGeneral,
                                     vk::ClearDepthStencilValue{.depth = *depth_value},
                                     range(vk::ImageAspectFlagBits::eDepth));
    }
    if (stencil) {
        s.cmd.clearDepthStencilImage(
            image, vk::ImageLayout::eGeneral,
            vk::ClearDepthStencilValue{.stencil = entry.stencil_clear_value},
            range(vk::ImageAspectFlagBits::eStencil));
    }
    if (color_zero) {
        s.cmd.clearColorImage(image, vk::ImageLayout::eGeneral, vk::ClearColorValue{},
                              range(vk::ImageAspectFlagBits::eColor));
    }
    Written(&entry, entry.rewrite_clock); // NAT-148
    s.buffers_written = true; // the next work waits for the clear
    ++s.commands;
    ++stats.rewrite_clears;
}

void NativeRenderer::ReadFillShader(const Shader::Info& cs, DrawBindings& b) {
    // GT Sport's fill shader reads [lo, hi, count] from its first buffer and writes lo, hi,
    // lo, hi... (count dwords) to its second: a memset of depth buffers, stencil planes and
    // pictures.
    constexpr u64 FillShader = 0x8aa6e00c;
    if (cs.pgm_hash != FillShader || cs.buffers.size() < 2) {
        return;
    }
    const Shader::BufferResource* args = nullptr;
    const Shader::BufferResource* out = nullptr;
    for (const auto& desc : cs.buffers) {
        if (desc.IsSpecial()) {
            continue;
        }
        if (desc.is_written) {
            out = out ? out : &desc;
        } else {
            args = args ? args : &desc;
        }
    }
    if (!args || !out) {
        return;
    }
    const auto args_sharp = args->GetSharp(cs);
    const auto out_sharp = out->GetSharp(cs);
    const void* arguments = reinterpret_cast<const void*>(args_sharp.base_address);
    if (args_sharp.base_address == 0 || !Native::Readable(arguments, 12)) {
        return;
    }
    std::array<u32, 3> words{};
    std::memcpy(words.data(), arguments, sizeof(words));
    b.fill_pattern = std::array<u32, 2>{words[0], words[1]};
    b.fill_address = out_sharp.base_address;
    b.fill_size = std::min<u64>(u64(words[2]) * 4, out_sharp.GetSize());
    ++stats.fill_shaders;
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
            if (r.FrameDumpActive()) {
                r.FrameDumpEvent(fmt::format("dispatch: metadata clear {:#x}", address));
            }
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
    t_timed = false; // OPT-001: the phase timers are the draws'
    r.scheduler.PopPendingOperations();
    const auto& cs_program = r.liverpool->DrawCsRegs();
    const ComputePipeline* pipeline = r.AcquireComputePipeline();
    if (!pipeline) {
        ++stats.no_pipeline;
        return;
    }
    const auto& regs = r.liverpool->DrawRegs();
    MetaClearDispatch(pipeline->GetStage(Shader::SwStage::Compute));
    if (r.FrameDumpActive()) {
        r.FrameDumpDispatch(pipeline->GetStage(Shader::SwStage::Compute), cs_program,
                            false); // NAT-118
    }
    Stream& s = Begin();
    DrawBindings& b = bindings;
    b.Clear();
    b.push = UserData(regs);
    BindStages(pipeline->GetStages(), regs, true, b);
    if (b.failed) {
        ++stats.skipped;
        return;
    }
    ReadFillShader(pipeline->GetStage(Shader::SwStage::Compute), b); // NAT-120
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
    if (Native::g_dump_request.exchange(false)) {
        r.StartFrameDumpNow(); // FIX-156 (diagnosis): a watched fill
    }
    s.buffers_written = true; // what the dispatch wrote is seen after the next barrier
    ++stats.dispatches;
    ++s.commands;
    ChunkFlushCheck();
    Report();
}

void NativeRenderer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    std::scoped_lock lock{mutex};
    t_timed = false; // OPT-001
    r.scheduler.PopPendingOperations();
    const ComputePipeline* pipeline = r.AcquireComputePipeline();
    if (!pipeline) {
        ++stats.no_pipeline;
        return;
    }
    const auto& regs = r.liverpool->DrawRegs();
    if (r.FrameDumpActive()) {
        r.FrameDumpDispatch(pipeline->GetStage(Shader::SwStage::Compute),
                            r.liverpool->DrawCsRegs(), true); // NAT-118
    }
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
    // NAT-122: a fill before the resolve is applied before it (left pending, it cleared the
    // resolved picture at its first read: GT Sport's minimaps, resolved into the scratch pool).
    ApplyRewrite(*src);
    ApplyRewrite(*dst);
    if (r.FrameDumpActive()) {
        r.FrameDumpEvent(fmt::format("resolve {:#x} {}x{} x{} -> {:#x} {}x{} x{}",
                                     src->info.guest_address, src->info.size.width,
                                     src->info.size.height, src->info.num_samples,
                                     dst->info.guest_address, dst->info.size.width,
                                     dst->info.size.height, dst->info.num_samples));
        if (r.FrameDumpCollecting() &&
            std::ranges::find(dump_entries, dst) == dump_entries.end()) {
            dump_entries.push_back(dst);
        }
    }
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
        // FIX-154: the targets the resolve names (they may share larger pictures).
        .extent{std::min({src->info.size.width, dst->info.size.width, src_desc.info.size.width,
                          dst_desc.info.size.width}),
                std::min({src->info.size.height, dst->info.size.height,
                          src_desc.info.size.height, dst_desc.info.size.height}),
                1},
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
    const bool meta = known_meta.contains(address);
    if (meta) {
        cleared_meta.insert(address); // a CMASK/HTILE cleared by the CP
        ++stats.meta_clears;
    }
    if (r.FrameDumpActive()) {
        r.FrameDumpEvent(fmt::format("fill {:#x}+{:#x} = {:#x}{}", address, num_bytes, value,
                                     meta ? " (metadata clear)" : "")); // NAT-117
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
    const std::array<u32, 2> pattern{value, value};
    GpuWrote(address, num_bytes, &pattern, true); // NAT-119/120
}

void NativeRenderer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (num_bytes == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    Stream& s = Begin();
    ++stats.copies;
    if (r.FrameDumpActive()) {
        r.FrameDumpEvent(fmt::format("copy {:#x}{} <- {:#x}{} ({:#x} bytes)", dst,
                                     dst_gds ? " (GDS)" : "", src, src_gds ? " (GDS)" : "",
                                     num_bytes)); // NAT-117
    }
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
        GpuWrote(dst, num_bytes, nullptr, true); // NAT-119
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
    // NAT-146: a region over the range holds what the GPU reads there: it gets the bytes too.
    for (auto& [base, region] : regions) {
        const VAddr begin = std::max<VAddr>(address, region.begin);
        const VAddr end = std::min<VAddr>(address + data.size(), region.end);
        if (begin < end) {
            CmdBuf{s.cmd}.updateBuffer(vk::Buffer{Handle<VkBuffer_T>(region.buffer->NativeHandle())},
                                       begin - region.begin, end - begin,
                                       data.data() + (begin - address));
        }
    }
    // NAT-150: and mirrors.
    for (auto it = FirstMirror(address);
         it != mirrors.end() && it->second.begin < address + data.size(); ++it) {
        const Mirror& mirror = it->second;
        const VAddr begin = std::max<VAddr>(address, mirror.begin);
        const VAddr end = std::min<VAddr>(address + data.size(), mirror.end);
        CmdBuf{s.cmd}.updateBuffer(vk::Buffer{Handle<VkBuffer_T>(mirror.buffer->NativeHandle())},
                                   begin - mirror.begin, end - begin,
                                   data.data() + (begin - address));
    }
    s.buffers_written = true;
    ++s.commands;
    ++stats.inline_writes;
    return true;
}

void NativeRenderer::OnUnmap(VAddr address, u64 size) {
    if (Native::WatchedTexture(address, address + size)) {
        LOG_WARNING(Render_Vulkan, "FIX-153 watch: unmap {:#x}+{:#x}", address, size);
    }
    Native::ForgetReadable(address, size); // NAT-147
    // FIX-153: the pages stop being watched (Ryty tells before it unmaps them). Memory mapped
    // there again comes writable; a watch left in place counted as one in the page manager but
    // protected nothing, and was never set again (the page was "watched"). GT Sport maps a car's
    // models and textures into the same memory each time: the showroom's 32x32 lighting texture
    // kept the zeros of the fresh mapping, and the car screens came out blown out.
    if (size != 0) {
        UnwatchPages(VideoCore::PageManager::GetPageAddr(address),
                     VideoCore::PageManager::GetNextPageAddr(address + size - 1));
    }
    registry->Invalidate(address, size, Native::ImageRegistry::Stale::Unmap);
    // NAT-146: regions over unmapped memory go (the device frees their buffers once the GPU is
    // done with them).
    std::scoped_lock lock{mutex};
    {
        std::unique_lock map_lock{buffer_map_mutex}; // OPT-010
        for (auto it = regions.begin(); it != regions.end();) {
            if (it->second.begin < address + size && it->second.end > address) {
                buffer_log.Note(it->second.begin, it->second.end);
                it = regions.erase(it);
            } else {
                ++it;
            }
        }
    }
    DropMirrors(address, size); // NAT-150
    {
        std::scoped_lock pending_lock{region_dirty_mutex}; // FIX-173
        std::erase_if(page_snapshots, [&](const auto& snapshot) {
            return snapshot.first >= address && snapshot.first < address + size;
        });
    }
    ++buffer_generation;        // OPT-003
    ++unmap_generation;         // OPT-002
}

void NativeRenderer::WatchBind(const Native::ImageEntry& entry, const AmdGpu::Image& tsharp,
                               bool storage, bool memo, u64 program) {
    if (!Native::WatchSample()) {
        return;
    }
    u32 sources = 0;
    {
        std::scoped_lock lock{const_cast<Native::ImageEntry&>(entry).sources_mutex};
        sources = static_cast<u32>(entry.sources.size());
    }
    LOG_WARNING(Render_Vulkan, "FIX-157 watch: {} {:#x} {}x{} layers {}-{} type {} fmt {}/{}{} "
                "(program {:#x}) -> {} {}x{} layers {} version {}, {} sources (frame {})",
                storage ? "store" : "sample", tsharp.Address(), tsharp.width + 1,
                tsharp.height + 1, u32(tsharp.base_array), u32(tsharp.last_array),
                u32(tsharp.GetType()), u32(tsharp.GetDataFmt()), u32(tsharp.GetNumberFmt()),
                memo ? " (memo)" : "", program, entry.from_cpu ? "texture" : "target",
                entry.info.size.width, entry.info.size.height, entry.info.resources.layers,
                entry.write_version.load(), sources, frame);
}

void NativeRenderer::ProbeWatchedPictures() {
    // FIX-156 (diagnosis, RYGT_NATIVE_WATCH_TEXTURE): what the pictures at the watched address
    // hold, every half second (read back from the GPU), to see when one loses its content.
    const auto& watched_list = Native::WatchedTextures();
    if (watched_list.empty()) {
        return;
    }
    const u64 done = r.scheduler.GetWorkSemaphore()->KnownGpuTick();
    for (auto it = probes.begin(); it != probes.end();) {
        if (it->tick > done) {
            ++it;
            continue;
        }
        const u8* bytes = it->buffer->Mapped();
        const u64 count = it->size / 4;
        double sum = 0;
        float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
        u64 zeros = 0;
        // FIX-171: B10G11R11 pictures (GT Sport's light cubes) as their mean colour channel.
        const auto small_float = [](u32 bits, u32 mantissa_bits) {
            const u32 e = bits >> mantissa_bits;
            const float m = float(bits & ((1u << mantissa_bits) - 1)) / float(1u << mantissa_bits);
            return e == 0 ? std::ldexp(m, -14) : e == 31 ? 0.0f : std::ldexp(1.0f + m, int(e) - 15);
        };
        for (u64 i = 0; i < count; ++i) {
            float v;
            std::memcpy(&v, bytes + i * 4, 4);
            if (it->packed_r11g11b10) {
                u32 word;
                std::memcpy(&word, bytes + i * 4, 4);
                v = (small_float(word & 0x7ff, 6) + small_float((word >> 11) & 0x7ff, 6) +
                     small_float(word >> 22, 5)) / 3.0f;
            }
            if (!std::isfinite(v)) {
                continue;
            }
            sum += v;
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            zeros += v == 0.0f ? 1 : 0;
        }
        if (it->packed_r11g11b10 && it->layers != 0) {
            // FIX-171: per mip, the mean of each layer (untiled copy, mip after mip).
            std::string text;
            u64 at = 0;
            for (u32 mip = 0; mip < it->mips; ++mip) {
                const u64 w = std::max(it->width >> mip, 1u), h = std::max(it->height >> mip, 1u);
                text += fmt::format(" m{}:", mip);
                for (u32 layer = 0; layer < it->layers; ++layer) {
                    double layer_sum = 0;
                    for (u64 i = 0; i < w * h && (at + i + 1) * 4 <= it->size; ++i) {
                        u32 word;
                        std::memcpy(&word, bytes + (at + i) * 4, 4);
                        layer_sum += (small_float(word & 0x7ff, 6) +
                                      small_float((word >> 11) & 0x7ff, 6) +
                                      small_float(word >> 22, 5)) / 3.0f;
                    }
                    at += w * h;
                    text += fmt::format(" {:.3f}", layer_sum / double(w * h));
                }
            }
            if (it->layers == 1 && it->height >= 3) {
                // A plain picture (the frame's main view): its top, middle and bottom thirds.
                const u64 w = it->width, third = it->height / 3;
                std::array<double, 3> part{};
                for (u32 band = 0; band < 3; ++band) {
                    for (u64 y = band * third; y < (band + 1) * third; ++y) {
                        for (u64 x = 0; x < w; ++x) {
                            u32 word;
                            std::memcpy(&word, bytes + (y * w + x) * 4, 4);
                            part[band] += (small_float(word & 0x7ff, 6) +
                                           small_float((word >> 11) & 0x7ff, 6) +
                                           small_float(word >> 22, 5)) / 3.0f;
                        }
                    }
                    part[band] /= double(third * w);
                }
                text += fmt::format(" thirds: {:.4f} {:.4f} {:.4f}", part[0], part[1], part[2]);
            }
            LOG_WARNING(Render_Vulkan, "FIX-171 watch: frame {}: {} {:#x} {}x{} version {}:{}",
                        it->frame, it->target ? "target" : "texture", it->address, it->width,
                        it->height, it->version, text);
            it = probes.erase(it);
            continue;
        }
        LOG_WARNING(Render_Vulkan, "FIX-156 watch: frame {}: {} {:#x} {}x{} version {} holds "
                    "mean {:.4f} min {:.4f} max {:.4f}, {} of {} words zero (as 32-bit floats)",
                    it->frame, it->target ? "target" : "texture", it->address, it->width,
                    it->height,
                    it->version, count ? sum / count : 0.0, lo, hi, zeros, count);
        it = probes.erase(it);
    }
    // FIX-171 (diagnosis, RYGT_NATIVE_WATCH_EVERY=<frames>): how often (every 30 frames).
    static const u64 every = [] {
        const char* env = std::getenv("RYGT_NATIVE_WATCH_EVERY");
        return env && *env ? std::max<u64>(std::strtoull(env, nullptr, 0), 1) : u64{30};
    }();
    if (frame % every != 0 || probes.size() >= (every >= 30 ? 1u : 16u)) {
        return;
    }
    Begin();
    for (const VAddr watched : watched_list) {
        registry->ForEachOverlapping(watched, watched + 1, [&](Native::ImageEntry& entry) {
            if (entry.info.guest_address != watched || !entry.image ||
                entry.info.num_samples != 1) {
                return;
            }
            const bool packed = entry.info.pixel_format == vk::Format::eB10G11R11UfloatPack32;
            const auto& desc = entry.image->Desc();
            // FIX-171 (RYGT_NATIVE_WATCH_RECT=x,y,w,h): of a plain picture only that rectangle.
            static const std::array<u32, 4> rect = [] {
                std::array<u32, 4> r{};
                if (const char* env = std::getenv("RYGT_NATIVE_WATCH_RECT"); env && *env) {
                    std::sscanf(env, "%u,%u,%u,%u", &r[0], &r[1], &r[2], &r[3]);
                }
                return r;
            }();
            const bool in_rect = packed && rect[2] != 0 && rect[3] != 0 && desc.mips == 1 &&
                                 desc.layers <= 1 && rect[0] + rect[2] <= desc.width &&
                                 rect[1] + rect[3] <= desc.height;
            u64 untiled = 0;
            for (u32 mip = 0; mip < desc.mips; ++mip) {
                untiled += u64(std::max(desc.width >> mip, 1u)) *
                           std::max(desc.height >> mip, 1u) * std::max(desc.layers, 1u) * 4;
            }
            if (in_rect) {
                untiled = u64(rect[2]) * rect[3] * 4;
            }
            const u64 size = packed ? untiled : entry.info.guest_size;
            auto buffer = device->CreateBuffer({.size = size,
                                                .usage = Native::Rhi::BufferUsageStorage |
                                                         Native::Rhi::BufferUsageTransferDst,
                                                .memory = Native::Rhi::MemoryKind::Readback,
                                                .name = "watch probe"});
            if (!buffer || !buffer->Mapped()) {
                return;
            }
            GlobalBarrier();
            if (in_rect) {
                const vk::BufferImageCopy region{
                    .bufferOffset = 0,
                    .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                      .mipLevel = 0,
                                      .baseArrayLayer = 0,
                                      .layerCount = 1},
                    .imageOffset = {s32(rect[0]), s32(rect[1]), 0},
                    .imageExtent = {rect[2], rect[3], 1},
                };
                stream.cmd.copyImageToBuffer(
                    vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())},
                    vk::ImageLayout::eGeneral,
                    vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())}, region);
                probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, size, frame,
                                  entry.write_version.load(), rect[2], rect[3], !entry.from_cpu,
                                  watched, true, 1, 1});
            } else if (packed) {
                // FIX-171: each mip with all its layers, untiled.
                std::vector<vk::BufferImageCopy> regions;
                u64 offset = 0;
                for (u32 mip = 0; mip < desc.mips; ++mip) {
                    const u32 w = std::max(desc.width >> mip, 1u);
                    const u32 h = std::max(desc.height >> mip, 1u);
                    regions.push_back(vk::BufferImageCopy{
                        .bufferOffset = offset,
                        .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                          .mipLevel = mip,
                                          .baseArrayLayer = 0,
                                          .layerCount = std::max(desc.layers, 1u)},
                        .imageExtent = {w, h, 1},
                    });
                    offset += u64(w) * h * std::max(desc.layers, 1u) * 4;
                }
                stream.cmd.copyImageToBuffer(
                    vk::Image{Handle<VkImage_T>(entry.image->NativeHandle())},
                    vk::ImageLayout::eGeneral,
                    vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())}, regions);
                probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, size, frame,
                                  entry.write_version.load(), desc.width, desc.height,
                                  !entry.from_cpu, watched, true, desc.mips,
                                  std::max(desc.layers, 1u)});
            } else if (registry->RecordDownload(*stream.list, entry, *buffer, 0)) {
                stream.pipeline = vk::Pipeline{};
                probes.push_back({std::move(buffer), r.scheduler.CurrentTick() + 1, size, frame,
                                  entry.write_version.load(), entry.info.size.width,
                                  entry.info.size.height, !entry.from_cpu, watched, false});
            }
        });
    }
}

void NativeRenderer::EvictImages() {
    ProbeWatchedPictures(); // FIX-156 (diagnosis)
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
    // FIX-157: a picture the GPU wrote and used for a second or more stays past its idle time
    // while nothing else writes its memory: its pixels exist nowhere else (the GPU's writes stay
    // in the picture, not in guest memory). GT Sport bakes its light probes (two cube arrays of
    // 322 32x32 cubes, at 0x107aee1400 and 0x108626d400) with compute shaders while a race
    // loads; nothing read them for about three seconds around the start, they were dropped, the
    // race's textures there uploaded the zeros of guest memory, and the scene's ambient light
    // went: a dark, blue cockpit and world, with the odd warm frame. It is kept when guest memory
    // still holds what it held when the picture was made (nothing but the GPU wrote there:
    // keeping pictures whose memory the CPU had reused drew stale pictures, and writing them
    // back overwrote shader code the game had loaded there), no newer picture overlaps it, and
    // GPU work wrote no part of its memory since; from then on its pages are watched, and a CPU
    // write, an unmap or GPU work there lets it go. Pictures in use for less than a second (the
    // scratch pool's) go as before; kept ones take at most 256 MB.
    constexpr u64 KeptBudget = 256ull << 20;
    u32 newly_kept = 0;
    const auto keep = [&](Native::ImageEntry& entry) {
        if (entry.from_cpu || !entry.image || entry.reused.load(std::memory_order_relaxed)) {
            return false;
        }
        const u64 version = entry.write_version.load(std::memory_order_relaxed);
        if (version == 0 || entry.foreign_clock.load(std::memory_order_relaxed) > version) {
            return false;
        }
        const VAddr address = entry.info.guest_address;
        const u64 size = std::max<u64>(entry.info.guest_size, 1);
        bool superseded = false;
        registry->ForEachOverlapping(address, address + size, [&](Native::ImageEntry& other) {
            superseded |= &other != &entry && !other.from_cpu &&
                          !other.reused.load(std::memory_order_relaxed) &&
                          other.write_version.load(std::memory_order_relaxed) > version;
        });
        if (superseded) {
            return false;
        }
        if (entry.kept.load(std::memory_order_relaxed)) {
            return true;
        }
        if (entry.last_use.load(std::memory_order_relaxed) < entry.made_frame + 60 ||
            kept_bytes + size > KeptBudget || !registry->MemoryUnchanged(entry)) {
            return false;
        }
        entry.kept.store(true, std::memory_order_relaxed);
        kept_bytes += size;
        WatchPages(address, size);
        ++newly_kept;
        if (Native::WatchedTexture(address, address + 1)) {
            LOG_WARNING(Render_Vulkan, "FIX-157 watch: {:#x} kept (version {}, frame {})",
                        address, version, frame);
        }
        return true;
    };
    registry->Evict(frame - KeepFrames, [&](Native::ImageEntry& entry) {
        if (entry.kept.load(std::memory_order_relaxed)) {
            kept_bytes -= std::min<u64>(kept_bytes, std::max<u64>(entry.info.guest_size, 1));
        }
        if (Native::WatchedTexture(entry.info.guest_address, entry.info.guest_address + 1)) {
            LOG_WARNING(Render_Vulkan, "FIX-157 watch: {} {:#x} {}x{} layers {} evicted (last "
                        "use frame {}, version {}, frame {})",
                        entry.from_cpu ? "texture" : "target", entry.info.guest_address,
                        entry.info.size.width, entry.info.size.height,
                        entry.info.resources.layers, entry.last_use.load(),
                        entry.write_version.load(), frame);
        }
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
        std::erase(dump_entries, &entry);
        for (auto it = stencil_owners.begin(); it != stencil_owners.end();) {
            auto& owners = it->second;
            owners.erase(std::remove(owners.begin(), owners.end(), &entry), owners.end());
            it = owners.empty() ? stencil_owners.erase(it) : std::next(it);
        }
    }, keep);
    if (newly_kept != 0) {
        static std::atomic<u32> logged{};
        if (const u32 n = logged++; n < 20 || n % 100 == 0) {
            LOG_WARNING(Render_Vulkan, "FIX-157: {} pictures the GPU wrote are kept past their idle "
                        "time ({:.1f} MB kept in all)", newly_kept, kept_bytes / 1048576.0);
        }
    }
}

void NativeRenderer::WatchPages(VAddr address, u64 size) {
    const VAddr begin = VideoCore::PageManager::GetPageAddr(address);
    const VAddr end = VideoCore::PageManager::GetNextPageAddr(address + size - 1);
    constexpr u64 PageSize = 4096;
    std::scoped_lock lock{watch_mutex};
    if (Native::WatchedTexture(begin, end)) {
        u32 fresh = 0;
        for (VAddr page = begin; page < end; page += PageSize) {
            fresh += watched_pages.contains(page) ? 0 : 1;
        }
        LOG_WARNING(Render_Vulkan, "FIX-153 watch: watch {:#x}-{:#x}, {} of {} pages not yet",
                    begin, end, fresh, (end - begin) / PageSize);
    }
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

void NativeRenderer::UnwatchPages(VAddr begin, VAddr end) {
    constexpr u64 PageSize = 4096;
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

void NativeRenderer::OnCpuWrite(VAddr address, u64 size, u64 exact) {
    // The written pages stop being watched (so they open up and the write goes on), and the
    // textures over them are uploaded again when next used (which watches them again).
    const VAddr begin = VideoCore::PageManager::GetPageAddr(address);
    const VAddr end = VideoCore::PageManager::GetNextPageAddr(address + std::max<u64>(size, 1) - 1);
    if (Native::WatchedTexture(begin, end)) {
        LOG_WARNING(Render_Vulkan, "FIX-153 watch: CPU write fault {:#x}+{:#x} (exact {:#x}); "
                    "memory {}", address, size, exact, Native::WatchedBytes(address));
    }
    // FIX-173: a region page holds what the GPU wrote there, which guest memory never got; a
    // CPU write to the page copies in only the bytes the CPU changed (RefreshRegion compares
    // with this copy of the page from before the write), not the whole page from guest memory.
    // Copying whole pages put stale guest bytes over the GPU's results: GT Sport computes the
    // cockpit's light from its interior capture into 64 bytes at 0x1000e317c0 every third
    // frame, and the CPU writes other data in that page every frame, so the cockpit was lit in
    // one frame of three and dark in the other two.
    if (end - begin <= 64_KB) {
        std::shared_lock map_lock{buffer_map_mutex};
        if (RegionOver(begin, end - begin)) {
            std::scoped_lock pending_lock{region_dirty_mutex};
            for (VAddr page = begin; page < end; page += 4096) {
                auto& slot = page_snapshots[page];
                if (slot) {
                    continue; // the first write since the page was copied in counts
                }
                auto copy = std::make_unique<std::array<u8, 4096>>();
                if (Core::RytyMemory::GuardedCopy(copy->data(),
                                                  reinterpret_cast<const void*>(page), 4096)) {
                    slot = std::move(copy);
                } else {
                    page_snapshots.erase(page);
                }
            }
        }
    }
    UnwatchPages(begin, end);
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
    if (InPlaceSampled()) {
        for (u32 i = 0; i < in_place.pages.size(); ++i) {
            in_place.total[i] += in_place.pages[i].size();
            in_place.pages[i].clear();
        }
        ++in_place.frames;
    }
    ++frame;
    shared_frame.store(frame, std::memory_order_relaxed); // OPT-010
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

void NativeRenderer::DumpFrame(const std::filesystem::path& dir) {
    std::scoped_lock lock{mutex};
    struct Readback {
        const Native::ImageEntry* entry;
        std::unique_ptr<Native::Rhi::Buffer> buffer;
        u64 size;
        u32 index;
    };
    std::vector<Readback> readbacks;
    Stream& s = Begin();
    GlobalBarrier();
    // NAT-123: the sampled GPU-written textures follow the targets (named tex_<key>).
    std::vector<std::pair<const Native::ImageEntry*, std::string>> jobs;
    {
        u32 order = 0;
        for (const auto* entry : dump_entries) {
            jobs.emplace_back(entry, fmt::format("{:03}_{:x}_{}x{}_{}{}.bin", order++,
                                                 entry->info.guest_address,
                                                 entry->image ? entry->image->Desc().width : 0,
                                                 entry->image ? entry->image->Desc().height : 0,
                                                 entry->image ? u32(entry->image->Desc().format)
                                                              : 0,
                                                 entry->info.props.is_depth ? "_depth" : ""));
        }
        std::string list;
        for (const auto& [key, entry] : dump_textures) {
            u32 sources;
            {
                std::scoped_lock lock{const_cast<Native::ImageEntry*>(entry)->sources_mutex};
                sources = u32(entry->sources.size());
            }
            const auto& info = entry->info;
            // FIX-153 (diagnosis): what guest memory holds there, and whether the GPU writes it
            // in a region.
            std::array<u32, 4> head{};
            const bool readable = Core::RytyMemory::GuardedCopy(
                head.data(), reinterpret_cast<const void*>(info.guest_address), sizeof(head));
            const Region* region =
                RegionOver(info.guest_address, std::max<u64>(info.guest_size, 1));
            // The bytes there now against the bytes last uploaded (NAT-109's hash).
            u64 hash_now = 0;
            const u64 size = info.guest_size;
            if (size != 0 && size <= (4ull << 20) && r.memory->IsValidMapping(info.guest_address) &&
                r.memory->ClampRangeSize(info.guest_address, size) == size &&
                Native::ReadablePrefix(reinterpret_cast<const void*>(info.guest_address), size) ==
                    size) {
                hash_now = Core::HashGuestBytes(info.guest_address, size) | 1;
            }
            list += fmt::format("{} -> {} {:#x} {}x{} mips {} layers {} {}, {} sources, "
                                "version {}{}; memory {}; {}; uploads {}, stale {}; "
                                "total uploads {} kind {}, tiled {} mode {} bits {} size {:#x}; "
                                "hash now {:016x} uploaded {:016x}{}\n",
                                key, entry->from_cpu ? "texture" : "target", info.guest_address,
                                info.size.width, info.size.height, info.resources.levels,
                                info.resources.layers, vk::to_string(info.pixel_format), sources,
                                entry->write_version.load(),
                                info.num_samples > 1 ? " (multisampled, not dumped)" : "",
                                readable ? fmt::format("{:08x} {:08x} {:08x} {:08x}", head[0],
                                                       head[1], head[2], head[3])
                                         : std::string{"unreadable"},
                                region ? fmt::format("in region {:#x}+{:#x}", region->begin,
                                                     region->end - region->begin)
                                       : std::string{"no region"},
                                entry->upload_count.load(), entry->needs_upload.load(),
                                entry->total_uploads.load(), entry->upload_kind,
                                info.props.is_tiled, u32(info.tile_mode), info.num_bits, size,
                                hash_now, entry->content_hash,
                                hash_now && entry->content_hash && hash_now != entry->content_hash
                                    ? " CHANGED"
                                    : "");
            jobs.emplace_back(entry, fmt::format("tex_{}{}.bin", key,
                                                 info.props.is_depth ? "_depth" : ""));
        }
        if (std::FILE* file = std::fopen((dir / "textures.txt").string().c_str(), "w")) {
            std::fwrite(list.data(), 1, list.size(), file);
            std::fclose(file);
        }
    }
    u32 index = 0;
    for (const auto& [entry, name] : jobs) {
        const u32 order = index++;
        if (!entry->image) {
            continue;
        }
        const auto& desc = entry->image->Desc();
        const u32 texel = entry->info.props.is_depth ? 4u : std::max(entry->info.num_bits / 8, 1u);
        // NAT-125: every layer of the first mip (cube faces, layered targets), one after another.
        const u32 layers = desc.dim == Native::Rhi::ImageDim::D3 ? 1u : std::max(desc.layers, 1u);
        // FIX-152 (diagnosis): block-compressed images give their blocks (4x4 texels each).
        const bool block = entry->info.props.is_block;
        const u32 row = block ? Common::AlignUp(desc.width, 4u) : desc.width;
        const u32 rows = block ? Common::AlignUp(desc.height, 4u) : desc.height;
        const u64 size = block ? u64(row / 4) * (rows / 4) * texel * layers
                               : u64(desc.width) * desc.height * texel * layers;
        auto buffer = device->CreateBuffer({.size = size,
                                            .usage = Native::Rhi::BufferUsageTransferDst,
                                            .memory = Native::Rhi::MemoryKind::Readback,
                                            .name = "native dump"});
        if (!buffer || !buffer->Mapped() || desc.samples != 1) {
            continue;
        }
        const vk::BufferImageCopy copy{
            .bufferOffset = 0,
            .bufferRowLength = row,
            .bufferImageHeight = rows,
            .imageSubresource{.aspectMask = entry->info.props.is_depth
                                               ? vk::ImageAspectFlagBits::eDepth
                                               : vk::ImageAspectFlagBits::eColor,
                              .mipLevel = 0,
                              .baseArrayLayer = 0,
                              .layerCount = layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {desc.width, desc.height, 1},
        };
        s.cmd.copyImageToBuffer(vk::Image{Handle<VkImage_T>(entry->image->NativeHandle())},
                                vk::ImageLayout::eGeneral,
                                vk::Buffer{Handle<VkBuffer_T>(buffer->NativeHandle())}, copy);
        readbacks.push_back({entry, std::move(buffer), size, order});
    }
    Splice();
    r.scheduler.Finish();
    for (const auto& readback : readbacks) {
        const auto& info = readback.entry->info;
        const auto& desc = readback.entry->image->Desc();
        const auto path = dir / jobs[readback.index].second;
        if (std::FILE* file = std::fopen(path.string().c_str(), "wb")) {
            std::fwrite(readback.buffer->Mapped(), 1, readback.size, file);
            std::fclose(file);
        }
    }
    LOG_WARNING(Render_Vulkan, "NAT-116: {} render targets and textures of the frame dumped to {}",
                readbacks.size(), dir.string());
    dump_entries.clear();
    dump_textures.clear();
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
    std::vector<std::tuple<u32, VAddr, u32, u32, u64, bool>> busy;
    registry->ForEachOverlapping(0, ~VAddr{0}, [&](Native::ImageEntry& entry) {
        if (const u32 count = entry.upload_count.exchange(0); count >= 4) {
            busy.emplace_back(count, entry.info.guest_address, entry.info.size.width,
                              entry.info.size.height, entry.info.guest_size, entry.from_cpu);
        }
    });
    std::ranges::sort(busy, std::greater{});
    std::string busiest;
    for (size_t i = 0; i < std::min<size_t>(busy.size(), 5); ++i) {
        const auto& [count, address, width, height, size, texture] = busy[i];
        busiest += fmt::format(" {} {:#x} {}x{} ({} KB) x{};", texture ? "texture" : "target",
                               address, width, height, size / 1024, count);
    }
    std::string crowded;
    for (const auto& [address, count] : registry->Crowded(4)) {
        crowded += fmt::format(" {:#x} x{};", address, count);
    }
    LOG_INFO(Render_Vulkan,
             "NAT-108 images (10 s): {} uploads ({:.1f} MB), {} skipped as unchanged (NAT-109), "
             "textures made stale by CPU writes {}, unmaps {}, GPU writes {} ({} targets so far, "
             "NAT-119); {} evicted, {} live, most at one address:{}; uploaded most:{}",
             uploads - report_base.uploads, (upload_bytes - report_base.upload_bytes) / 1048576.0,
             registry->GetUploadStats().unchanged.exchange(0),
             stale[0] - report_base.stale[0], stale[1] - report_base.stale[1],
             stale[2] - report_base.stale[2], registry_stats.targets_rewritten.load(),
             evicted - report_base.evicted, registry->Count(),
             crowded,
             busiest.empty() ? std::string{" none"} : busiest);
    report_base = {uploads, upload_bytes, evicted, stale};
    LOG_INFO(Render_Vulkan,
             "NAT-100 native (10 s): {} draws, {} dispatches, {} skipped, {} without pipeline; "
             "{} passes, {} barriers ({} pass barriers, {} passes unblocked, NAT-140), {} lists "
             "run; {} reads copied ({:.1f} MB), {} ranges bound "
             "empty, {} images bound empty; {} fills, {} copies, {} meta clears, {} inline "
             "writes, {} pictures copied from targets, {} written back, {} CPU writes to textures, "
             "{} ranges gathered, region binds/made/refreshes {}/{}/{} (NAT-114; {} regions, "
             "{:.1f} MB, {} pages refreshed in {:.1f} ms; {} guest buffers in {:.1f} ms), "
             "mirrors {} ({:.1f} MB; NAT-150: {} binds, {} made ({:.1f} MB), {} dropped as "
             "CPU-written, {} evicted, {} over budget, {} pages refreshed, {} granules read in "
             "place), "
             "memo hits textures {}/{} buffers {} (OPT-002/003), vertex ranges in the "
             "previous one's mirror {} (OPT-004), vertex records skipped {} (OPT-005), "
             "prepared draws (OPT-010): {} used, {} without one, rejected (pipeline/user "
             "data/unmap/images/buffers) {}, made on job threads {} ({:.2f} us each, {} could "
             "not, {} replaced, {} stale), vertices from them {} ({} not), pipelines from "
             "them {} (OPT-012), record parts "
             "barrier/begin/dynamic/vertex/pipeline/bindings/draw/notes {}, "
             "fill shaders {}, targets rewritten by clears/uploads/partial clears/fills "
             "{}/{}/{}/{} (NAT-120/124/132), "
             "on the GPU ({:.1f} MB), uploads from guest memory on the GPU/CPU {}/{}; images so far: "
             "{} lookups, {} textures and {} targets made, {} uploads ({:.1f} MB); {:.0f} draws/s, "
             "{:.2f} us a draw, {:.2f} us a dispatch on the recorder; images bound empty by "
             "reason (none/unmapped/format/geometry/outside/FIX-081/no image) {}; us per draw in "
             "pipeline/targets/buffers/textures(sources)/vertices/prepare/record {}",
             stats.draws, stats.dispatches, stats.skipped, stats.no_pipeline, stats.passes,
             stats.barriers, stats.pass_barriers, stats.passes_unblocked, stats.splices,
             stats.copied_reads, stats.copied_bytes / 1048576.0,
             stats.unbound, stats.null_images, stats.fills, stats.copies, stats.meta_clears,
             stats.inline_writes, stats.source_copies, stats.writebacks,
             stats.cpu_writes.exchange(0),
             stats.gathers, stats.region_binds, stats.regions_made, stats.region_refreshes,
             regions.size(),
             [&] {
                 u64 bytes = 0;
                 for (const auto& [base, region] : regions) {
                     bytes += region.end - region.begin;
                 }
                 return bytes / 1048576.0;
             }(),
             stats.region_refresh_pages, stats.region_refresh_ns / 1e6, stats.guest_buffer_calls,
             stats.guest_buffer_ns * 16 / 1e6, // OPT-001: timed on one draw in 16
             mirrors.size(), mirror_bytes / 1048576.0,
             stats.mirror_binds, stats.mirrors_made, stats.mirror_made_bytes / 1048576.0,
             stats.mirrors_dropped, stats.mirrors_evicted, stats.mirror_over_budget,
             stats.mirror_refresh_pages, dynamic_granules.size(), stats.texture_memo_hits,
             stats.texture_lookups, stats.buffer_memo_hits, stats.vertex_near,
             stats.vertex_skipped, stats.prepared_used, stats.prepared_missing,
             fmt::join(stats.prepared_rejected, "/"), prepare_stats.made.load(),
             prepare_stats.made.load() ? prepare_stats.ns.load() / 1000.0 / prepare_stats.made.load()
                                       : 0.0,
             prepare_stats.failed.load(), prepare_stats.replaced.load(),
             prepare_stats.stale.load(), stats.prepared_vertices,
             stats.prepared_vertices_rejected, stats.prepared_installed,
             [&] {
                 std::string text;
                 for (u32 i = 0; i < stats.record_ns.size(); ++i) {
                     text += fmt::format("{}{:.2f}", i == 0 ? "" : "/",
                                         stats.timed_draws
                                             ? stats.record_ns[i] / 1000.0 / stats.timed_draws
                                             : 0.0);
                 }
                 return text;
             }(),
             stats.fill_shaders, stats.rewrite_clears, stats.rewrite_uploads,
             stats.rewrite_partial, stats.rewrite_fills,
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
                         stats.timed_draws ? stats.phase_ns[i] / 1000.0 / stats.timed_draws
                                           : 0.0);
                 }
                 return text;
             }());
    if (in_place.frames != 0) {
        const auto mb = [&](u32 i) { return in_place.total[i] * 64.0 / 1024.0 / in_place.frames; };
        LOG_INFO(Render_Vulkan,
                 "NAT-142: guest memory used in place per frame (64 KB pages, MB): vertices "
                 "{:.1f}, indices {:.1f}, read-only storage {:.1f}, storage read {:.1f}, storage "
                 "written {:.1f}; from video memory instead: mirrors {:.1f}, regions {:.1f} "
                 "(NAT-150; {} frames sampled)",
                 mb(0), mb(1), mb(2), mb(3), mb(4), mb(5), mb(6), in_place.frames);
        in_place.total = {};
        in_place.frames = 0;
    }
    const u64 cpu_writes = stats.cpu_writes.load();
    stats.draws = stats.dispatches = stats.skipped = stats.no_pipeline = stats.passes = 0;
    stats.barriers = stats.splices = stats.copied_reads = stats.copied_bytes = stats.unbound = 0;
    stats.null_images = stats.fills = stats.copies = stats.meta_clears = stats.inline_writes = 0;
    stats.source_copies = stats.draw_ns = stats.dispatch_ns = 0;
    stats.timed_draws = 0;
    stats.texture_memo_hits = stats.texture_lookups = stats.buffer_memo_hits = 0;
    stats.vertex_near = stats.vertex_skipped = 0;
    stats.prepared_used = stats.prepared_missing = 0;
    stats.prepared_rejected = {};
    stats.prepared_vertices = stats.prepared_vertices_rejected = 0;
    stats.prepared_installed = 0;
    stats.record_ns = {};
    prepare_stats.made = 0;
    prepare_stats.failed = 0;
    prepare_stats.replaced = 0;
    prepare_stats.stale = 0;
    prepare_stats.ns = 0;
    stats.gathers = stats.gathered_bytes = stats.writebacks = 0;
    stats.region_binds = stats.regions_made = stats.region_refreshes = 0;
    stats.fill_shaders = stats.rewrite_clears = stats.rewrite_uploads = 0;
    stats.rewrite_partial = stats.rewrite_fills = 0;
    stats.pass_barriers = stats.passes_unblocked = 0;
    stats.region_refresh_pages = stats.region_refresh_ns = 0;
    stats.guest_buffer_calls = stats.guest_buffer_ns = 0;
    stats.mirror_binds = stats.mirrors_made = stats.mirror_made_bytes = stats.mirrors_dropped = 0;
    stats.mirrors_evicted = stats.mirror_over_budget = stats.mirror_refresh_pages = 0;
    stats.null_reasons = {};
    stats.phase_ns = {};
    (void)cpu_writes;
}

} // namespace Vulkan
