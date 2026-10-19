// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>

#include "common/logging/log.h"
#include "ryty_gpu/native/image_registry.h"
#include "video_core/amdgpu/tiling.h"

#include "video_core/host_shaders/tiling_macro_128_comp.h"
#include "video_core/host_shaders/tiling_macro_16_comp.h"
#include "video_core/host_shaders/tiling_macro_32_comp.h"
#include "video_core/host_shaders/tiling_macro_64_comp.h"
#include "video_core/host_shaders/tiling_macro_8_comp.h"
#include "video_core/host_shaders/tiling_macro_96_comp.h"
#include "video_core/host_shaders/tiling_micro_128_comp.h"
#include "video_core/host_shaders/tiling_micro_16_comp.h"
#include "video_core/host_shaders/tiling_micro_32_comp.h"
#include "video_core/host_shaders/tiling_micro_64_comp.h"
#include "video_core/host_shaders/tiling_micro_8_comp.h"
#include "video_core/host_shaders/tiling_micro_96_comp.h"

namespace Native {
namespace {

// The untiling shader's uniform (tiling.comp, scalar layout), as the texture cache fills it.
struct TilingParams {
    u32 bank_swizzle;
    u32 micro_tiled_mips;
    u32 num_mips;
    std::array<VideoCore::MipInfo, VideoCore::MAX_MIPS> mips;
};

Rhi::Format ToRhi(vk::Format format) {
    return Rhi::FormatFromVk(static_cast<u32>(format));
}

std::span<const u32> UntileCode(u32 num_bits, bool macro) {
    switch (num_bits) {
    case 8: return macro ? std::span<const u32>{TILING_MACRO_8_COMP} : std::span<const u32>{TILING_MICRO_8_COMP};
    case 16: return macro ? std::span<const u32>{TILING_MACRO_16_COMP} : std::span<const u32>{TILING_MICRO_16_COMP};
    case 32: return macro ? std::span<const u32>{TILING_MACRO_32_COMP} : std::span<const u32>{TILING_MICRO_32_COMP};
    case 64: return macro ? std::span<const u32>{TILING_MACRO_64_COMP} : std::span<const u32>{TILING_MICRO_64_COMP};
    case 96: return macro ? std::span<const u32>{TILING_MACRO_96_COMP} : std::span<const u32>{TILING_MICRO_96_COMP};
    case 128: return macro ? std::span<const u32>{TILING_MACRO_128_COMP} : std::span<const u32>{TILING_MICRO_128_COMP};
    default: return {};
    }
}

Rhi::ImageDesc DescOf(const VideoCore::ImageInfo& info, bool render_target, bool with_targets,
                      bool with_storage) {
    Rhi::ImageDesc desc{};
    desc.format = ToRhi(info.pixel_format);
    desc.width = std::max(info.size.width, 1u);
    desc.height = std::max(info.size.height, 1u);
    desc.mips = std::max<u32>(info.resources.levels, 1);
    desc.layers = std::max<u32>(info.resources.layers, 1);
    desc.samples = std::max(info.num_samples, 1u);
    // As in shadGT's texture cache, 1D images are 2D and cube maps are 2D arrays: the shader
    // translator declares them that way.
    switch (info.type) {
    case AmdGpu::ImageType::Color3D:
        desc.dim = Rhi::ImageDim::D3;
        desc.depth = std::max(info.size.depth, 1u);
        desc.layers = 1;
        break;
    default:
        desc.dim = Rhi::ImageDim::D2;
        break;
    }
    desc.usage = Rhi::ImageUsageSampled | Rhi::ImageUsageTransferSrc | Rhi::ImageUsageTransferDst;
    if (desc.dim == Rhi::ImageDim::D2 && desc.layers >= 6 && desc.width == desc.height) {
        desc.usage |= Rhi::ImageUsageCube;
    }
    if (info.props.is_depth) {
        desc.usage |= Rhi::ImageUsageDepthTarget;
    } else {
        // Views reinterpret colour images in other formats of the same texel size, as the game
        // does with GPU memory.
        desc.usage |= Rhi::ImageUsageMutableFormat;
        if (with_targets && !info.props.is_block) {
            desc.usage |= Rhi::ImageUsageColorTarget;
        }
        if (with_storage && !info.props.is_block && desc.samples == 1) {
            desc.usage |= Rhi::ImageUsageStorage; // compute shaders write pictures too
        }
    }
    (void)render_target;
    return desc;
}

} // namespace

bool SameLayout(const VideoCore::ImageInfo& a, const VideoCore::ImageInfo& b) {
    return a.guest_address == b.guest_address && a.num_bits == b.num_bits &&
           a.size == b.size && a.pitch == b.pitch && a.tile_mode == b.tile_mode &&
           a.resources == b.resources && a.num_samples == b.num_samples &&
           a.props.is_depth == b.props.is_depth && a.props.is_block == b.props.is_block &&
           a.type == b.type;
}

const Rhi::ImageView* ImageEntry::View(Rhi::Device& device, const ImageViewKey& key) {
    std::scoped_lock lock{views_mutex};
    for (const auto& [view_key, view] : views) {
        if (view_key == key) {
            return view.get();
        }
    }
    Rhi::ImageViewDesc desc{};
    desc.dim = key.dim;
    desc.format = key.format;
    desc.swizzle = key.swizzle;
    desc.base_mip = key.base_mip;
    desc.mips = key.mips;
    desc.base_layer = key.base_layer;
    desc.layers = key.layers;
    desc.aspect = key.aspect;
    desc.storage = key.storage;
    auto view = device.CreateImageView(image.get(), desc);
    const Rhi::ImageView* result = view.get();
    if (view) {
        views.emplace_back(key, std::move(view));
    }
    return result;
}

ImageRegistry::ImageRegistry(Rhi::Device& device) : device{device} {
    const std::array<Rhi::BindingSlot, 3> slots{{
        {0, Rhi::BindingType::StorageBuffer, Rhi::StageCompute},
        {1, Rhi::BindingType::StorageBuffer, Rhi::StageCompute},
        {2, Rhi::BindingType::UniformBuffer, Rhi::StageCompute},
    }};
    untile_layout = device.CreatePipelineLayout({.bindings = slots});
}

ImageRegistry::~ImageRegistry() = default;

ImageEntry* ImageRegistry::Find(const VideoCore::ImageInfo& info, bool render_target, u64 frame) {
    ++stats.lookups;
    Shard& shard = ShardOf(info.guest_address);
    {
        std::shared_lock lock{shard.mutex};
        const auto [first, last] = shard.images.equal_range(info.guest_address);
        for (auto it = first; it != last; ++it) {
            if (SameLayout(it->second->info, info)) {
                ++stats.hits;
                it->second->last_use.store(frame, std::memory_order_relaxed);
                return it->second.get();
            }
        }
    }
    if (!render_target) {
        // NAT-040: a depth buffer read as a texture (the T# names R32/R16, the image is D32/D16):
        // the depth image itself, viewed through its depth aspect.
        std::shared_lock lock{shard.mutex};
        const auto [first, last] = shard.images.equal_range(info.guest_address);
        for (auto it = first; it != last; ++it) {
            const auto& cached = it->second->info;
            if (cached.props.is_depth && !info.props.is_depth &&
                cached.size.width == info.size.width && cached.size.height == info.size.height &&
                cached.num_bits == info.num_bits) {
                ++stats.hits;
                it->second->last_use.store(frame, std::memory_order_relaxed);
                return it->second.get();
            }
        }
    }
    auto entry = std::make_unique<ImageEntry>();
    entry->info = info;
    entry->from_cpu = !render_target;
    if (image_factory) {
        MadeImage made = image_factory(info, render_target);
        entry->image = std::move(made.image);
        entry->backend_usage = made.usage;
        entry->backend_features = made.features;
    } else {
        entry->image = device.CreateImage(DescOf(info, render_target, true, true));
    }
    if (!entry->image && !image_factory) {
        // Some formats can't be written by shaders, or be render targets; such images are made
        // without those uses.
        entry->image = device.CreateImage(DescOf(info, render_target, true, false));
    }
    if (!entry->image && !image_factory) {
        entry->image = device.CreateImage(DescOf(info, render_target, false, false));
    }
    if (!entry->image) {
        LOG_WARNING(Render_Vulkan, "NAT-020: no image for {:#x} ({}x{}, format {})",
                    info.guest_address, info.size.width, info.size.height,
                    vk::to_string(info.pixel_format));
        return nullptr;
    }
    entry->last_use.store(frame, std::memory_order_relaxed);
    ImageEntry* result;
    {
        std::unique_lock lock{shard.mutex};
        // Another thread may have made it meanwhile.
        const auto [first, last] = shard.images.equal_range(info.guest_address);
        for (auto it = first; it != last; ++it) {
            if (SameLayout(it->second->info, info)) {
                ++stats.hits;
                return it->second.get();
            }
        }
        result = entry.get();
        shard.images.emplace(info.guest_address, std::move(entry));
    }
    if (render_target) {
        ++stats.created_targets;
        LinkTarget(*result);
    } else {
        ++stats.created_textures;
        LinkTexture(*result);
        result->needs_upload.store(true, std::memory_order_relaxed);
        std::scoped_lock lock{upload_mutex};
        pending_uploads.push_back(result);
    }
    return result;
}

bool ImageRegistry::HasImageAt(VAddr address) {
    Shard& shard = ShardOf(address);
    std::shared_lock lock{shard.mutex};
    return shard.images.contains(address);
}

ImageEntry* ImageRegistry::FindExisting(VAddr address,
                                        const std::function<bool(const ImageEntry&)>& accept) {
    Shard& shard = ShardOf(address);
    std::shared_lock lock{shard.mutex};
    const auto [first, last] = shard.images.equal_range(address);
    ImageEntry* best = nullptr;
    for (auto it = first; it != last; ++it) {
        // The most recently used one, when several layouts live at the address.
        if (accept(*it->second) &&
            (!best || it->second->last_use.load(std::memory_order_relaxed) >
                          best->last_use.load(std::memory_order_relaxed))) {
            best = it->second.get();
        }
    }
    return best;
}

SourcePlace PlaceOf(const VideoCore::ImageInfo& target, const VideoCore::ImageInfo& texture) {
    if (target.props.is_depth || texture.props.is_depth || target.props.is_block ||
        texture.props.is_block || target.num_bits != texture.num_bits ||
        target.num_samples != texture.num_samples) {
        return {};
    }
    // A crop: the texture is the corner of the target in the same layout (as IsSubrectOf, any
    // colour format of the texel size).
    if (target.guest_address == texture.guest_address && target.resources.levels == 1 &&
        target.resources.layers == 1 && texture.resources.levels == 1 &&
        texture.resources.layers == 1 && target.pitch == texture.pitch &&
        target.tile_mode == texture.tile_mode && target.bank_swizzle == texture.bank_swizzle &&
        target.alt_tile == texture.alt_tile &&
        target.mips_layout[0].pitch == texture.mips_layout[0].pitch &&
        texture.size.width <= target.size.width && texture.size.height <= target.size.height &&
        (texture.size.width < target.size.width || texture.size.height < target.size.height) &&
        target.size.depth == 1 && texture.size.depth == 1 &&
        target.type == AmdGpu::ImageType::Color2D && texture.type == target.type) {
        return {0, 0, texture.size.width, texture.size.height};
    }
    // A mip or layer of the texture drawn on its own.
    const s32 mip = target.MipOf(texture);
    if (mip < 0) {
        return {};
    }
    const s32 layer = target.SliceOf(texture, mip);
    if (layer < 0 || u32(layer) >= std::max<u32>(texture.resources.layers, 1)) {
        return {};
    }
    return {mip, layer, target.size.width, target.size.height};
}

void ImageRegistry::LinkTarget(ImageEntry& target) {
    // NAT-104: textures made before this target over its memory may be made of it.
    const VAddr begin = target.info.guest_address;
    const VAddr end = begin + std::max<u64>(target.info.guest_size, 1);
    for (Shard& other : shards) {
        std::shared_lock lock{other.mutex};
        for (const auto& [base, entry] : other.images) {
            if (!entry->from_cpu || base >= end || base + entry->info.guest_size <= begin) {
                continue;
            }
            if (PlaceOf(target.info, entry->info).mip >= 0) {
                std::scoped_lock sources_lock{entry->sources_mutex};
                entry->sources.push_back(&target);
            }
        }
    }
}

void ImageRegistry::LinkTexture(ImageEntry& texture) {
    // NAT-104: the render targets this texture is made of. NAT-041: a texture over a target
    // of another kind of layout reads what the CPU last wrote there, not the picture; logged
    // (a few times per layout pair) to learn which of these GT Sport has.
    const VAddr begin = texture.info.guest_address;
    const VAddr end = begin + std::max<u64>(texture.info.guest_size, 1);
    for (Shard& other : shards) {
        std::shared_lock lock{other.mutex};
        for (const auto& [base, entry] : other.images) {
            if (entry->from_cpu || base >= end || base + entry->info.guest_size <= begin) {
                continue;
            }
            if (PlaceOf(entry->info, texture.info).mip >= 0) {
                std::scoped_lock sources_lock{texture.sources_mutex};
                texture.sources.push_back(entry.get());
                continue;
            }
            const auto& a = texture.info;
            const auto& b = entry->info;
            const u64 key = (a.guest_address * 31 + b.guest_address) ^ (u64(a.size.width) << 40) ^
                            (u64(b.size.width) << 20) ^ (u64(a.resources.levels) << 8) ^
                            u64(b.resources.levels);
            std::scoped_lock log_lock{overlap_log_mutex};
            if (overlap_logged.size() < 4096 && overlap_logged.insert(key).second) {
                LOG_WARNING(Render_Vulkan,
                            "NAT-041: texture {:#x}+{:#x} {}x{} mips {} layers {} tile {} {} bits "
                            "over render target {:#x}+{:#x} {}x{} mips {} layers {} tile {} {} "
                            "bits{}",
                            a.guest_address, a.guest_size, a.size.width, a.size.height,
                            a.resources.levels, a.resources.layers, u32(a.tile_mode), a.num_bits,
                            b.guest_address, b.guest_size, b.size.width, b.size.height,
                            b.resources.levels, b.resources.layers, u32(b.tile_mode), b.num_bits,
                            b.props.is_depth ? " (depth)" : "");
            }
        }
    }
}

void ImageRegistry::ForEachOverlapping(VAddr begin, VAddr end,
                                       const std::function<void(ImageEntry&)>& fn) {
    for (Shard& shard : shards) {
        std::shared_lock lock{shard.mutex};
        for (auto& [base, entry] : shard.images) {
            if (base < end && base + std::max<u64>(entry->info.guest_size, 1) > begin) {
                fn(*entry);
            }
        }
    }
}

void ImageRegistry::Invalidate(VAddr address, u64 size) {
    const VAddr end = address + size;
    for (Shard& shard : shards) {
        std::shared_lock lock{shard.mutex};
        for (auto& [base, entry] : shard.images) {
            if (!entry->from_cpu || base >= end || base + entry->info.guest_size <= address) {
                continue;
            }
            if (!entry->needs_upload.exchange(true)) {
                std::scoped_lock upload_lock{upload_mutex};
                pending_uploads.push_back(entry.get());
            }
        }
    }
}

std::vector<Rhi::BufferImageCopy> MipCopies(const VideoCore::ImageInfo& info) {
    std::vector<Rhi::BufferImageCopy> copies;
    for (u32 m = 0; m < info.resources.levels; ++m) {
        const u32 width = std::max(info.size.width >> m, 1u);
        const u32 height = std::max(info.size.height >> m, 1u);
        const u32 depth = info.props.is_volume ? std::max(info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = info.mips_layout[m];
        copies.push_back({
            .buffer_offset = mip_offset,
            .buffer_row_texels = mip_pitch,
            .buffer_image_rows = mip_height,
            .mip = m,
            .base_layer = 0,
            .layers = info.resources.layers,
            .extent = {mip_pitch ? std::min<u32>(mip_pitch, width) : width,
                       mip_height ? std::min<u32>(mip_height, height) : height, depth},
            .aspect = static_cast<u8>(info.props.is_depth ? Rhi::AspectDepth : Rhi::AspectColor),
        });
    }
    return copies;
}

std::unique_ptr<Rhi::Buffer> ImageRegistry::TilingParamsBuffer(const VideoCore::ImageInfo& info) {
    auto buffer = device.CreateBuffer({.size = sizeof(TilingParams),
                                       .usage = Rhi::BufferUsageUniform,
                                       .memory = Rhi::MemoryKind::Upload});
    if (!buffer || !buffer->Mapped()) {
        return nullptr;
    }
    TilingParams params{};
    params.bank_swizzle = info.bank_swizzle;
    params.micro_tiled_mips = info.micro_tiled_mips;
    params.num_mips = info.resources.levels;
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max<u16>((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max<u16>((mip_info.height + 3) / 4, 1U);
        }
    }
    std::memcpy(buffer->Mapped(), &params, sizeof(params));
    return buffer;
}

const Rhi::Pipeline* ImageRegistry::TilingPipeline(const VideoCore::ImageInfo& info, bool tiler) {
    const u64 key = (u64(tiler) << 48) | (u64(info.tile_mode) << 40) | (u64(info.num_bits) << 8) |
                    info.num_samples;
    std::scoped_lock lock{untile_mutex};
    if (const auto it = untile_pipelines.find(key); it != untile_pipelines.end()) {
        return it->second.get();
    }
    const auto micro_tile_mode = AmdGpu::GetMicroTileMode(info.tile_mode);
    const bool macro = AmdGpu::IsMacroTiled(info.array_mode);
    std::array<Rhi::SpecConstant, 12> constants{};
    for (u32 i = 0; i < constants.size(); ++i) {
        constants[i].id = i;
    }
    constants[0].value = info.num_samples;
    constants[1].value = u32(micro_tile_mode);
    constants[2].value = AmdGpu::GetMicroTileThickness(info.array_mode);
    constants[3].value = tiler ? 1 : 0;
    if (macro) {
        const auto macro_tile_mode =
            AmdGpu::CalculateMacrotileMode(info.tile_mode, info.num_bits, info.num_samples);
        constants[4].value = u32(info.array_mode);
        constants[5].value = u32(AmdGpu::GetPipeConfig(info.tile_mode));
        constants[6].value = AmdGpu::GetBankWidth(macro_tile_mode);
        constants[7].value = AmdGpu::GetBankHeight(macro_tile_mode);
        constants[8].value = AmdGpu::GetNumBanks(macro_tile_mode);
        constants[9].value = std::bit_width(constants[8].value) - 1;
        constants[10].value = AmdGpu::CalculateTileSplit(info.tile_mode, info.array_mode,
                                                         micro_tile_mode, info.num_bits);
        constants[11].value = AmdGpu::GetMacrotileAspect(macro_tile_mode);
    }
    const auto code = UntileCode(info.num_bits, macro);
    if (code.empty() || !untile_layout) {
        return nullptr;
    }
    auto pipeline = device.CreateComputePipeline({
        .layout = untile_layout.get(),
        .shader = {Rhi::ShaderStage::Compute, code, "main",
                   std::span<const Rhi::SpecConstant>{constants.data(), macro ? 12u : 4u}},
    });
    const Rhi::Pipeline* result = pipeline.get();
    untile_pipelines.emplace(key, std::move(pipeline));
    return result;
}

u32 ImageRegistry::RecordUploads(Rhi::CommandList& list) {
    std::vector<ImageEntry*> pending;
    {
        std::scoped_lock lock{upload_mutex};
        pending.swap(pending_uploads);
    }
    u32 uploaded = 0;
    std::vector<std::unique_ptr<Rhi::Buffer>> staging; // freed after the GPU used them
    for (ImageEntry* entry : pending) {
        if (!entry->needs_upload.exchange(false)) {
            continue;
        }
        const VideoCore::ImageInfo& info = entry->info;
        const u64 size = info.guest_size;
        if (size == 0) {
            continue;
        }
        auto source = device.CreateBuffer({.size = size,
                                           .usage = Rhi::BufferUsageStorage |
                                                    Rhi::BufferUsageTransferSrc,
                                           .memory = Rhi::MemoryKind::Upload});
        if (!source || !source->Mapped()) {
            continue;
        }
        if (upload_hook) {
            upload_hook(*entry); // NAT-105: CPU writes from here on are seen
        }
        // Only what is mapped is read (a T# can reach past the end of a mapping); the rest of
        // the picture is zeros.
        const u64 readable = memory_clamp ? memory_clamp(info.guest_address, size) : size;
        if (readable < size) {
            std::memset(source->Mapped() + readable, 0, size - readable);
            static std::atomic<u32> logged{};
            if (logged++ < 16) {
                LOG_WARNING(Render_Vulkan,
                            "NAT-042: texture {:#x}+{:#x} ({}x{}, {} mips) is mapped for {:#x} "
                            "bytes only; the rest uploads as zeros",
                            info.guest_address, size, info.size.width, info.size.height,
                            info.resources.levels, readable);
            }
        }
        if (readable != 0) {
            std::memcpy(source->Mapped(), reinterpret_cast<const void*>(info.guest_address),
                        readable);
        }
        const Rhi::Buffer* linear = source.get();
        if (info.props.is_tiled) {
            const Rhi::Pipeline* untile = TilingPipeline(info, false);
            auto params_buffer = TilingParamsBuffer(info);
            auto out = device.CreateBuffer({.size = size,
                                            .usage = Rhi::BufferUsageStorage |
                                                     Rhi::BufferUsageTransferSrc,
                                            .memory = Rhi::MemoryKind::DeviceLocal});
            if (!untile || !params_buffer || !out) {
                continue;
            }
            list.BindPipeline(untile);
            const std::array<Rhi::Binding, 3> bindings{{
                {.slot = 0, .type = Rhi::BindingType::StorageBuffer, .buffer = source.get(), .size = size},
                {.slot = 1, .type = Rhi::BindingType::StorageBuffer, .buffer = out.get(), .size = size},
                {.slot = 2, .type = Rhi::BindingType::UniformBuffer, .buffer = params_buffer.get(),
                 .size = sizeof(TilingParams)},
            }};
            list.PushBindings(bindings);
            list.Dispatch(static_cast<u32>((size / (info.num_bits / 8)) / 64), 1, 1);
            list.Barrier(Rhi::ScopeCompute, Rhi::ScopeTransfer);
            linear = out.get();
            staging.push_back(std::move(params_buffer));
            staging.push_back(std::move(out));
        }
        list.CopyBufferToImage(linear, entry->image.get(), MipCopies(info));
        staging.push_back(std::move(source));
        stats.upload_bytes += size;
        ++uploaded;
    }
    if (uploaded) {
        list.Barrier(Rhi::ScopeTransfer, Rhi::ScopeDraw | Rhi::ScopeCompute);
        stats.uploads += uploaded;
    }
    // The device frees these once the submission that runs this list has finished.
    staging.clear();
    return uploaded;
}

bool ImageRegistry::RecordDownload(Rhi::CommandList& list, ImageEntry& entry, const Rhi::Buffer& out) {
    const VideoCore::ImageInfo& info = entry.info;
    const u64 size = info.guest_size;
    if (size == 0 || out.Size() < size) {
        return false;
    }
    if (!info.props.is_tiled) {
        list.CopyImageToBuffer(entry.image.get(), &out, MipCopies(info));
        return true;
    }
    const Rhi::Pipeline* tile = TilingPipeline(info, true);
    auto params_buffer = TilingParamsBuffer(info);
    auto linear = device.CreateBuffer({.size = size,
                                       .usage = Rhi::BufferUsageStorage |
                                                Rhi::BufferUsageTransferDst,
                                       .memory = Rhi::MemoryKind::DeviceLocal});
    if (!tile || !params_buffer || !linear) {
        return false;
    }
    list.CopyImageToBuffer(entry.image.get(), linear.get(), MipCopies(info));
    list.Barrier(Rhi::ScopeTransfer, Rhi::ScopeCompute);
    list.BindPipeline(tile);
    const std::array<Rhi::Binding, 3> bindings{{
        {.slot = 0, .type = Rhi::BindingType::StorageBuffer, .buffer = &out, .size = size},
        {.slot = 1, .type = Rhi::BindingType::StorageBuffer, .buffer = linear.get(), .size = size},
        {.slot = 2, .type = Rhi::BindingType::UniformBuffer, .buffer = params_buffer.get(),
         .size = sizeof(TilingParams)},
    }};
    list.PushBindings(bindings);
    list.Dispatch(static_cast<u32>((size / (info.num_bits / 8)) / 64), 1, 1);
    list.Barrier(Rhi::ScopeCompute, Rhi::ScopeTransfer | Rhi::ScopeHost);
    return true; // the device frees the temporary buffers after the GPU used them
}

} // namespace Native
