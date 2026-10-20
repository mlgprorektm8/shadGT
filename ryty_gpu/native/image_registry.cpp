// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>

#include "common/logging/log.h"
#include <xxhash.h>
#include "ryty_gpu/native/image_registry.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"

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

ImageEntry* ImageRegistry::Find(const VideoCore::ImageInfo& info, bool render_target, u64 frame,
                                bool create) {
    ++stats.lookups;
    Shard& shard = ShardOf(info.guest_address);
    {
        std::shared_lock lock{shard.mutex};
        const auto [first, last] = shard.images.equal_range(info.guest_address);
        ImageEntry* exact = nullptr;
        for (auto it = first; it != last; ++it) {
            if (SameLayout(it->second->info, info)) {
                exact = it->second.get();
                if (render_target || !exact->from_cpu) {
                    break;
                }
            }
        }
        // A target of the texture's own layout is read as it is.
        if (exact && (render_target || !exact->from_cpu)) {
            ++stats.hits;
            exact->last_use.store(frame, std::memory_order_relaxed);
            return exact;
        }
        if (!render_target && !info.props.is_depth &&
            Vulkan::LiverpoolToVK::IsFormatDepthCompatible(info.pixel_format)) {
            // NAT-040: a depth buffer read as a texture (the T# names R32/R16, the image is
            // D32/D16): the depth image itself, viewed through its depth aspect. NAT-121: before a
            // texture of the same layout (an upload of the memory under the depth buffer, made
            // before it existed, would hide it: GT Sport's shadow cascades, a 256x256 depth
            // target with 14 layers read with 13). NAT-126: only for depth-like formats (a
            // B10G11R11 bloom picture in the scratch pool matched a bigger depth buffer there).
            for (auto it = first; it != last; ++it) {
                const auto& cached = it->second->info;
                if (cached.props.is_depth && cached.num_bits == info.num_bits &&
                    cached.resources.layers >= info.resources.layers &&
                    ((cached.size.width == info.size.width &&
                      cached.size.height == info.size.height) ||
                     (cached.pitch == info.pitch && cached.size.width >= info.size.width &&
                      cached.size.height >= info.size.height))) { // NAT-113
                    ++stats.hits;
                    it->second->last_use.store(frame, std::memory_order_relaxed);
                    return it->second.get();
                }
            }
        }
        if (exact) {
            ++stats.hits;
            exact->last_use.store(frame, std::memory_order_relaxed);
            return exact;
        }
        if (render_target) {
            // NAT-113: a target drawn at the same address in the same layout (pitch, tiling,
            // texel size) that is at least as large is drawn into again: GT Sport's dynamic
            // resolution changes target sizes a little from frame to frame (depth 354x150, then
            // 336x150), and the next pass reads what the last one drew there.
            ImageEntry* best = nullptr;
            for (auto it = first; it != last; ++it) {
                const auto& c = it->second->info;
                if (!it->second->from_cpu && it->second->image && c.pitch == info.pitch &&
                    c.tile_mode == info.tile_mode && c.num_bits == info.num_bits &&
                    c.num_samples == info.num_samples && c.props.is_depth == info.props.is_depth &&
                    c.props.has_stencil == info.props.has_stencil && c.type == info.type &&
                    c.resources.levels >= info.resources.levels &&
                    c.resources.layers >= info.resources.layers &&
                    c.size.width >= info.size.width && c.size.height >= info.size.height &&
                    (!info.props.is_depth || c.pixel_format == info.pixel_format) &&
                    (!best || u64(c.size.width) * c.size.height <
                                  u64(best->info.size.width) * best->info.size.height)) {
                    best = it->second.get();
                }
            }
            if (best) {
                ++stats.hits;
                ++stats.reused_larger;
                best->last_use.store(frame, std::memory_order_relaxed);
                return best;
            }
        }
    }
    if (!create) {
        return nullptr; // OPT-010
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
        const std::pair<VAddr, VAddr> range{info.guest_address,
                                            info.guest_address + std::max<u64>(info.guest_size, 1)};
        NoteChanges({&range, 1}); // OPT-002, OPT-010
    }
    IndexEntry(*result);
    if (render_target) {
        ++stats.created_targets;
        LinkTarget(*result);
        // NAT-127: a new target starts with what memory holds there, as on the console (a pass
        // that draws part of it leaves the rest as it was). GT Sport draws its 16x16 brightness
        // grid into a 64x16 target the CPU reads back for the exposure: the undrawn part went
        // back to memory as uninitialized video memory, and the picture came out too dark.
        if (info.num_samples == 1) {
            result->needs_upload.store(true, std::memory_order_relaxed);
            result->force_upload.store(true, std::memory_order_relaxed);
            std::scoped_lock lock{upload_mutex};
            pending_uploads.push_back(result);
        }
    } else {
        ++stats.created_textures;
        LinkTexture(*result);
        result->needs_upload.store(true, std::memory_order_relaxed);
        std::scoped_lock lock{upload_mutex};
        pending_uploads.push_back(result);
    }
    return result;
}

void ImageRegistry::NoteChanges(std::span<const std::pair<VAddr, VAddr>> ranges) {
    std::scoped_lock lock{changes_mutex};
    const u64 next = generation.load(std::memory_order_relaxed) + 1;
    for (const auto& [begin, end] : ranges) {
        changes[change_count++ % changes.size()] = {next, begin, end};
    }
    generation.fetch_add(1, std::memory_order_acq_rel);
}

bool ImageRegistry::Untouched(u64 since, std::span<const std::pair<VAddr, VAddr>> ranges) {
    if (generation.load(std::memory_order_acquire) == since) {
        return true;
    }
    std::scoped_lock lock{changes_mutex};
    const u64 kept = std::min<u64>(change_count, changes.size());
    for (u64 i = 0; i < kept; ++i) {
        const Change& change = changes[(change_count - 1 - i) % changes.size()];
        if (change.generation <= since) {
            return true;
        }
        for (const auto& [begin, end] : ranges) {
            if (change.begin < end && change.end > begin) {
                return false;
            }
        }
    }
    // Every change kept is newer: the log reaches back far enough only if none was lost.
    return change_count <= changes.size();
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
    if (const s32 mip = target.MipOf(texture); mip >= 0) {
        const s32 layer = target.SliceOf(texture, mip);
        if (layer < 0 || u32(layer) >= std::max<u32>(texture.resources.layers, 1)) {
            return {};
        }
        return {mip, layer, target.size.width, target.size.height};
    }
    // NAT-131: the texture is one layer (or mip) of the target: a face of a cube drawn as a
    // layered target, read on its own (GT Sport's environment cube, whose faces the cockpit's
    // reflection cube is made of; only face 0, at the target's own address, was found).
    if (const s32 src_mip = texture.MipOf(target); src_mip >= 0) {
        const s32 src_layer = texture.SliceOf(target, src_mip);
        if (src_layer < 0 || u32(src_layer) >= std::max<u32>(target.resources.layers, 1)) {
            return {};
        }
        return {0, 0, texture.size.width, texture.size.height, src_mip, src_layer};
    }
    return {};
}

void ImageRegistry::IndexEntry(ImageEntry& entry) {
    const VAddr base = entry.info.guest_address;
    const u64 span = std::clamp<u64>(entry.info.guest_size, 1, MaxIndexedSize);
    std::unique_lock lock{granule_mutex};
    for (u64 g = base >> GranuleBits; g <= (base + span - 1) >> GranuleBits; ++g) {
        granules[g].push_back(&entry);
    }
}

void ImageRegistry::UnindexEntry(ImageEntry& entry) {
    const VAddr base = entry.info.guest_address;
    const u64 span = std::clamp<u64>(entry.info.guest_size, 1, MaxIndexedSize);
    for (u64 g = base >> GranuleBits; g <= (base + span - 1) >> GranuleBits; ++g) {
        const auto it = granules.find(g);
        if (it == granules.end()) {
            continue;
        }
        auto& list = it->second;
        if (const auto at = std::ranges::find(list, &entry); at != list.end()) {
            *at = list.back();
            list.pop_back();
        }
        if (list.empty()) {
            granules.erase(it);
        }
    }
}

void ImageRegistry::LinkTarget(ImageEntry& target) {
    // NAT-104: textures made before this target over its memory may be made of it.
    const VAddr begin = target.info.guest_address;
    const VAddr end = begin + std::max<u64>(target.info.guest_size, 1);
    ForEachOverlapping(begin, end, [&](ImageEntry& entry) {
        if (!entry.from_cpu) {
            return;
        }
        if (const auto place = PlaceOf(target.info, entry.info); place.mip >= 0) {
            {
                std::scoped_lock sources_lock{entry.sources_mutex};
                entry.sources.push_back({&target, place});
            }
            entry.sources_dirty.store(true, std::memory_order_relaxed);
            std::scoped_lock target_lock{target.sources_mutex};
            target.dependents.push_back(&entry);
        }
    });
}

void ImageRegistry::LinkTexture(ImageEntry& texture) {
    // NAT-104: the render targets this texture is made of. NAT-041: a texture over a target
    // of another kind of layout reads what the CPU last wrote there, not the picture; logged
    // (a few times per layout pair) to learn which of these GT Sport has.
    const VAddr begin = texture.info.guest_address;
    const VAddr end = begin + std::max<u64>(texture.info.guest_size, 1);
    ForEachOverlapping(begin, end, [&](ImageEntry& other) {
        const ImageEntry* entry = &other;
        {
            if (entry->from_cpu) {
                return;
            }
            if (const auto place = PlaceOf(entry->info, texture.info); place.mip >= 0) {
                {
                    std::scoped_lock sources_lock{texture.sources_mutex};
                    texture.sources.push_back({&other, place});
                }
                texture.sources_dirty.store(true, std::memory_order_relaxed);
                std::scoped_lock target_lock{other.sources_mutex};
                other.dependents.push_back(&texture);
                return;
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
    });
}

void ImageRegistry::ForEachOverlapping(VAddr begin, VAddr end,
                                       const std::function<void(ImageEntry&)>& fn) {
    if (end <= begin) {
        return;
    }
    // Each image once: in the first granule of the range it covers.
    const u64 first = begin >> GranuleBits;
    const u64 last = (end - 1) >> GranuleBits;
    const auto visit = [&](u64 g, const std::vector<ImageEntry*>& list) {
        for (ImageEntry* entry : list) {
            const VAddr base = entry->info.guest_address;
            const u64 span = std::clamp<u64>(entry->info.guest_size, 1, MaxIndexedSize);
            if (g == std::max(first, base >> GranuleBits) && base < end && base + span > begin) {
                fn(*entry);
            }
        }
    };
    std::shared_lock lock{granule_mutex};
    if (last - first >= granules.size()) {
        for (const auto& [g, list] : granules) {
            if (g >= first && g <= last) {
                visit(g, list);
            }
        }
        return;
    }
    for (u64 g = first; g <= last; ++g) {
        if (const auto it = granules.find(g); it != granules.end()) {
            visit(g, it->second);
        }
    }
}

std::vector<std::pair<VAddr, u32>> ImageRegistry::Crowded(u32 count) {
    std::unordered_map<VAddr, u32> per_address;
    for (Shard& shard : shards) {
        std::shared_lock lock{shard.mutex};
        for (const auto& [base, entry] : shard.images) {
            ++per_address[base];
        }
    }
    std::vector<std::pair<VAddr, u32>> list(per_address.begin(), per_address.end());
    std::ranges::sort(list, std::greater{}, [](const auto& item) { return item.second; });
    list.resize(std::min<size_t>(list.size(), count));
    return list;
}

u64 ImageRegistry::Count() {
    u64 count = 0;
    for (Shard& shard : shards) {
        std::shared_lock lock{shard.mutex};
        count += shard.images.size();
    }
    return count;
}

u32 ImageRegistry::Evict(u64 before, const std::function<void(ImageEntry&)>& on_evict) {
    std::unordered_set<const ImageEntry*> gone;
    for (Shard& shard : shards) {
        std::shared_lock lock{shard.mutex};
        for (const auto& [base, entry] : shard.images) {
            if (entry->last_use.load(std::memory_order_relaxed) < before) {
                gone.insert(entry.get());
            }
        }
    }
    if (gone.empty()) {
        return 0;
    }
    // Out of the shards first: from then on nothing finds them (a CPU write fault on another
    // thread queues the textures it finds for upload while holding a shard's lock).
    std::vector<std::unique_ptr<ImageEntry>> taken;
    for (Shard& shard : shards) {
        std::unique_lock lock{shard.mutex};
        for (auto it = shard.images.begin(); it != shard.images.end();) {
            // OPT-010: one a job thread found since the scan stays.
            if (gone.contains(it->second.get()) &&
                it->second->last_use.load(std::memory_order_relaxed) < before) {
                taken.push_back(std::move(it->second));
                it = shard.images.erase(it);
            } else {
                ++it;
            }
        }
    }
    gone.clear();
    for (const auto& entry : taken) {
        gone.insert(entry.get());
    }
    {
        std::unique_lock lock{granule_mutex};
        for (const auto& entry : taken) {
            UnindexEntry(*entry);
        }
    }
    {
        std::scoped_lock lock{upload_mutex};
        std::erase_if(pending_uploads, [&](ImageEntry* entry) { return gone.contains(entry); });
    }
    if (!taken.empty()) {
        std::vector<std::pair<VAddr, VAddr>> ranges;
        ranges.reserve(taken.size());
        for (const auto& entry : taken) {
            ranges.emplace_back(entry->info.guest_address,
                                entry->info.guest_address + std::max<u64>(entry->info.guest_size, 1));
        }
        NoteChanges(ranges); // OPT-002, OPT-010
    }
    // The links between textures and targets that went (NAT-111: both directions).
    for (const auto& entry : taken) {
        std::vector<ImageEntry*> others;
        {
            std::scoped_lock lock{entry->sources_mutex};
            for (const auto& link : entry->sources) {
                others.push_back(link.source);
            }
            others.insert(others.end(), entry->dependents.begin(), entry->dependents.end());
        }
        for (ImageEntry* other : others) {
            if (gone.contains(other)) {
                continue;
            }
            std::scoped_lock lock{other->sources_mutex};
            std::erase_if(other->sources,
                          [&](const SourceLink& link) { return link.source == entry.get(); });
            std::erase(other->dependents, entry.get());
        }
    }
    for (const auto& entry : taken) {
        on_evict(*entry);
    }
    stats.evicted += taken.size();
    return static_cast<u32>(taken.size());
}

void ImageRegistry::MarkStale(ImageEntry& entry, Stale why) {
    ++stats.stale[static_cast<u32>(why)];
    if (why != Stale::CpuWrite) {
        entry.force_upload.store(true, std::memory_order_relaxed);
    }
    if (!entry.needs_upload.exchange(true)) {
        std::scoped_lock upload_lock{upload_mutex};
        pending_uploads.push_back(&entry);
    }
}

void ImageRegistry::Invalidate(VAddr address, u64 size, Stale why) {
    ForEachOverlapping(address, address + size, [&](ImageEntry& entry) {
        if (entry.from_cpu) {
            MarkStale(entry, why);
        }
    });
}

void ImageRegistry::GpuWrote(VAddr address, u64 size, const std::array<u32, 2>* pattern,
                             bool exact, u64 clock) {
    ForEachOverlapping(address, address + std::max<u64>(size, 1), [&](ImageEntry& entry) {
        if (entry.from_cpu) {
            // NAT-129: a storage buffer as big as GT Sport's scratch pool would upload every
            // texture in it again at each dispatch (10-30 GB in 10 s): only buffers about the
            // texture's size count, as for targets below.
            if (!exact && size > 2 * std::max<u64>(entry.info.guest_size, 1)) {
                return;
            }
            // NAT-133: a fill that covers the whole texture fills it on the GPU at its next use
            // (GT Sport clears its environment cube and motion targets with fills, and textures
            // over that memory were uploaded again 60 times a second).
            if (pattern && address <= entry.info.guest_address &&
                address + size >= entry.info.guest_address + entry.info.guest_size) {
                entry.rewrite = ImageEntry::Rewrite::Clear;
                entry.rewrite_pattern = *pattern;
                entry.rewrite_bytes = entry.info.guest_size;
                entry.rewrite_clock = clock;
                return;
            }
            bool made_of_targets;
            {
                std::scoped_lock lock{entry.sources_mutex};
                made_of_targets = !entry.sources.empty();
            }
            if (!made_of_targets && entry.info.num_samples == 1) {
                MarkStale(entry, Stale::GpuWrite);
                static std::atomic<u32> logged{};
                if (const u32 n = logged++; n < 40 || n % 2000 == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "NAT-129: texture {:#x} {}x{} ({:#x} bytes) is uploaded again: a "
                                "{} wrote {:#x}+{:#x} ({} so far)",
                                entry.info.guest_address, entry.info.size.width,
                                entry.info.size.height, entry.info.guest_size,
                                exact ? (pattern ? "fill" : "copy") : "storage buffer", address,
                                size, n + 1);
                }
            }
            return;
        }
        if (entry.info.guest_address != address) {
            return;
        }
        if (pattern) {
            ++stats.targets_rewritten;
            entry.rewrite = ImageEntry::Rewrite::Clear;
            entry.rewrite_pattern = *pattern;
            entry.rewrite_bytes = std::min<u64>(size, entry.info.guest_size);
            entry.rewrite_clock = clock;
        } else if (size >= entry.info.guest_size &&
                   (exact || size <= 2 * entry.info.guest_size)) {
            ++stats.targets_rewritten;
            entry.rewrite = ImageEntry::Rewrite::Upload;
            entry.rewrite_bytes = entry.info.guest_size;
            entry.rewrite_clock = clock;
            static std::atomic<u32> logged{};
            if (const u32 n = logged++; n < 40 || n % 2000 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "NAT-129: target {:#x} {}x{} ({:#x} bytes) is uploaded again: a {} "
                            "wrote {:#x}+{:#x} ({} so far)",
                            entry.info.guest_address, entry.info.size.width,
                            entry.info.size.height, entry.info.guest_size,
                            exact ? "copy" : "storage buffer", address, size, n + 1);
            }
        }
    });
}

void ImageRegistry::CpuWrite(VAddr address, u64 size, u64 exact) {
    constexpr u64 HashedSize = 4ull << 20;
    const VAddr written_end = address + (exact != 0 ? exact : size);
    ForEachOverlapping(address, address + size, [&](ImageEntry& entry) {
        if (!entry.from_cpu) {
            return;
        }
        const VAddr base = entry.info.guest_address;
        const bool hit = base < written_end && base + entry.info.guest_size > address;
        // A small texture is checked by hash at its next use (its page is no longer watched,
        // so later writes there go unseen); a large one is uploaded again only when written.
        if (hit || entry.info.guest_size <= HashedSize) {
            MarkStale(entry, Stale::CpuWrite);
        }
    });
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
        if (info.num_samples > 1) {
            continue;
        }
        if (upload_hook && entry->from_cpu) {
            upload_hook(*entry); // NAT-105: CPU writes from here on are seen
        }
        // NAT-109: a small texture whose bytes did not change (a write elsewhere on its page)
        // is not uploaded again; GPU writes to its memory always upload.
        constexpr u64 HashedSize = 4ull << 20;
        const bool forced = entry->force_upload.exchange(false);
        if (size <= HashedSize && memory_clamp && memory_clamp(info.guest_address, size) == size) {
            const u64 hash =
                XXH3_64bits(reinterpret_cast<const void*>(info.guest_address), size) | 1;
            if (!forced && hash == entry->content_hash) {
                ++upload_stats.unchanged;
                continue;
            }
            entry->content_hash = hash;
        } else {
            entry->content_hash = 0;
        }
        // NAT-107: the bytes as the GPU sees guest memory, in GPU order (textures the game's
        // GPU work filled, in memory the CPU may not even read), else a copy the CPU makes now.
        UploadSource gpu_source;
        if (upload_source) {
            gpu_source = upload_source(*entry, list);
        }
        std::unique_ptr<Rhi::Buffer> source;
        const Rhi::Buffer* bytes = gpu_source.buffer;
        u64 bytes_offset = gpu_source.offset;
        for (auto& keep : gpu_source.keep) {
            staging.push_back(std::move(keep));
        }
        if (bytes) {
            ++upload_stats.gpu;
        } else {
            ++upload_stats.cpu;
            source = device.CreateBuffer({.size = size,
                                          .usage = Rhi::BufferUsageStorage |
                                                   Rhi::BufferUsageTransferSrc,
                                          .memory = Rhi::MemoryKind::Upload});
            if (!source || !source->Mapped()) {
                continue;
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
            bytes = source.get();
            bytes_offset = 0;
        }
        const Rhi::Buffer* linear = bytes;
        u64 linear_offset = bytes_offset;
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
                {.slot = 0, .type = Rhi::BindingType::StorageBuffer, .buffer = bytes,
                 .offset = bytes_offset, .size = size},
                {.slot = 1, .type = Rhi::BindingType::StorageBuffer, .buffer = out.get(), .size = size},
                {.slot = 2, .type = Rhi::BindingType::UniformBuffer, .buffer = params_buffer.get(),
                 .size = sizeof(TilingParams)},
            }};
            list.PushBindings(bindings);
            list.Dispatch(static_cast<u32>((size / (info.num_bits / 8)) / 64), 1, 1);
            list.Barrier(Rhi::ScopeCompute, Rhi::ScopeTransfer);
            linear = out.get();
            linear_offset = 0;
            staging.push_back(std::move(params_buffer));
            staging.push_back(std::move(out));
        }
        auto copies = MipCopies(info);
        for (auto& copy : copies) {
            copy.buffer_offset += linear_offset;
        }
        list.CopyBufferToImage(linear, entry->image.get(), copies);
        if (source) {
            staging.push_back(std::move(source));
        }
        stats.upload_bytes += size;
        ++entry->upload_count;
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

bool ImageRegistry::RecordDownload(Rhi::CommandList& list, ImageEntry& entry, const Rhi::Buffer& out,
                                   u64 out_offset) {
    const VideoCore::ImageInfo& info = entry.info;
    const u64 size = info.guest_size;
    if (size == 0 || out.Size() < out_offset + size) {
        return false;
    }
    if (!info.props.is_tiled) {
        auto copies = MipCopies(info);
        for (auto& copy : copies) {
            copy.buffer_offset += out_offset;
        }
        list.CopyImageToBuffer(entry.image.get(), &out, copies);
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
        {.slot = 0, .type = Rhi::BindingType::StorageBuffer, .buffer = &out, .offset = out_offset,
         .size = size},
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
