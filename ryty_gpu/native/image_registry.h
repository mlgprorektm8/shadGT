// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-020: the native renderer's images. GPU memory has no types on the PS4: a texture or render
// target is a guest address plus a layout the game describes in a T#, CB or DB. Every render job
// thread looks images up here at once, so lookups are shared-locked per address shard; creating
// an image takes the shard's lock briefly. Images made from CPU data (textures loaded from
// files) are uploaded, untiled on the GPU, by RecordUploads, which the executor runs before the
// frame's jobs; render targets start empty and are GPU-only.

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/types.h"
#include "ryty_gpu/native/rhi/rhi.h"
#include "video_core/texture_cache/image_info.h"

namespace Native {

/// How a view looks at an image (a T#'s format, dimensions, swizzle and subresources).
struct ImageViewKey {
    Rhi::ViewDim dim = Rhi::ViewDim::D2;
    Rhi::Format format = Rhi::Format::Undefined;
    std::array<Rhi::Swizzle, 4> swizzle{};
    u16 base_mip = 0;
    u16 mips = 1;
    u16 base_layer = 0;
    u16 layers = 1;
    u8 aspect = Rhi::AspectColor;
    bool storage = false;
    bool operator==(const ImageViewKey&) const = default;
};

/// NAT-100: what an image factory made: the image and the backend's usage and format features
/// for it (Vulkan: VkImageUsageFlags, VkFormatFeatureFlags2), which views are made with.
struct MadeImage {
    std::unique_ptr<Rhi::Image> image;
    u32 usage = 0;
    u64 features = 0;
};
using ImageFactory =
    std::function<MadeImage(const VideoCore::ImageInfo& info, bool render_target)>;

/// NAT-104: where a render target's picture goes in a texture made of it: the texture's mip and
/// layer and the size copied (a crop copies the texture's size from the target's corner).
struct SourcePlace {
    s32 mip = -1;
    s32 layer = -1;
    u32 width = 0;
    u32 height = 0;
};

class ImageEntry;
/// NAT-111: a render target a texture is made of, and where its picture goes.
struct SourceLink {
    ImageEntry* source;
    SourcePlace place;
};

class ImageEntry {
public:
    VideoCore::ImageInfo info;
    std::unique_ptr<Rhi::Image> image;
    u32 backend_usage = 0;    ///< from the image factory
    u64 backend_features = 0; ///< from the image factory
    /// NAT-104: when the renderer last wrote the picture (its own clock; 0: never).
    std::atomic<u64> write_version{0};
    /// NAT-104: render targets a texture's picture is made of: crops of a bigger target at
    /// the same address (the frosted-glass blur), or its mips and layers drawn one by one
    /// (environment cube maps). The renderer copies their newer pictures in before a read.
    std::mutex sources_mutex;
    std::vector<SourceLink> sources;
    /// NAT-111: on a render target, the textures made of it (guarded by its sources_mutex).
    std::vector<ImageEntry*> dependents;
    /// NAT-111: on a texture, a source was drawn since the renderer last copied them in.
    std::atomic<bool> sources_dirty{false};
    /// NAT-108: uploads since the renderer last looked (its report of busy textures).
    std::atomic<u32> upload_count{0};
    /// NAT-109: hash of the guest bytes last uploaded (small textures; 0: none), so a texture
    /// made stale by a write to its page that left its bytes alone is not uploaded again.
    u64 content_hash = 0;
    /// NAT-109: the GPU wrote its memory (the CPU can't tell what changed yet): upload.
    std::atomic<bool> force_upload{false};
    bool from_cpu = false; ///< texture data the CPU wrote; render targets are GPU-only
    std::atomic<bool> needs_upload{false};
    std::atomic<u64> last_use{0};

    /// A view, made on first use.
    const Rhi::ImageView* View(Rhi::Device& device, const ImageViewKey& key);

private:
    std::mutex views_mutex;
    std::vector<std::pair<ImageViewKey, std::unique_ptr<Rhi::ImageView>>> views;
};

class ImageRegistry {
public:
    explicit ImageRegistry(Rhi::Device& device);
    ~ImageRegistry();

    /// NAT-100: images are made by `factory` instead of from the RHI description (native mode
    /// makes them as the texture cache does: its formats and usage).
    void SetImageFactory(ImageFactory factory) {
        image_factory = std::move(factory);
    }
    /// NAT-100: how much of a guest range is mapped (uploads copy only that much).
    void SetMemoryClamp(std::function<u64(VAddr, u64)> clamp) {
        memory_clamp = std::move(clamp);
    }
    /// NAT-107: where an upload reads the picture's bytes on the GPU: guest memory itself
    /// (an imported buffer, or a copy gathered from several on the GPU), in GPU order. `buffer`
    /// null: the CPU copies them (memory the GPU can't see).
    struct UploadSource {
        const Rhi::Buffer* buffer = nullptr;
        u64 offset = 0;
        std::vector<std::unique_ptr<Rhi::Buffer>> keep; ///< alive until the GPU used them
    };
    void SetUploadSource(std::function<UploadSource(const ImageEntry&, Rhi::CommandList&)> source) {
        upload_source = std::move(source);
    }
    struct UploadStats {
        std::atomic<u64> gpu, cpu, unchanged;
    };
    UploadStats& GetUploadStats() {
        return upload_stats;
    }
    /// NAT-105: called for each texture right before its guest memory is read for an upload
    /// (native mode starts watching CPU writes to it there).
    void SetUploadHook(std::function<void(ImageEntry&)> hook) {
        upload_hook = std::move(hook);
    }
    /// NAT-105: every image overlapping [begin, end).
    void ForEachOverlapping(VAddr begin, VAddr end, const std::function<void(ImageEntry&)>& fn);

    /// The image for this layout: one already at the address with the same layout (whatever
    /// its colour format, if the texel size matches), or a new one. A texture found nowhere is
    /// queued for upload from guest memory; a render target is not. Thread-safe.
    ImageEntry* Find(const VideoCore::ImageInfo& info, bool render_target, u64 frame);

    /// NAT-100: whether uploads are queued (RecordUploads has work).
    bool HasPendingUploads() {
        std::scoped_lock lock{upload_mutex};
        return !pending_uploads.empty();
    }
    /// NAT-100: whether any image starts at the address.
    bool HasImageAt(VAddr address);
    /// NAT-100: an image already at the address that `accept` takes (no new image is made).
    ImageEntry* FindExisting(VAddr address, const std::function<bool(const ImageEntry&)>& accept);

    /// What made textures over a range stale.
    enum class Stale : u8 { CpuWrite, Unmap, GpuWrite, Count };
    /// CPU writes to guest memory (file loads, the game writing texture data): textures over the
    /// range are uploaded again before their next use.
    void Invalidate(VAddr address, u64 size, Stale why = Stale::CpuWrite);

    /// NAT-109: CPU writes to [address, address + size) (a page fault: the page stops being
    /// watched); `exact` is the bytes the instruction wrote there (0: unknown). Small textures
    /// on the page are checked by hash before their next use; large ones only when written.
    void CpuWrite(VAddr address, u64 size, u64 exact);
    /// NAT-109: the addresses holding the most images (address, count), for reports.
    std::vector<std::pair<VAddr, u32>> Crowded(u32 count);

    /// NAT-108: frees the images last used before frame `before`; `on_evict` sees each one
    /// first (the renderer drops what it keeps about it). The images' memory goes once the GPU
    /// has finished the work recorded so far. Returns how many went.
    u32 Evict(u64 before, const std::function<void(ImageEntry&)>& on_evict);
    u64 Count();

    /// Records the uploads queued since the last call (copy from guest memory, untile, copy
    /// into the images). Returns how many images were uploaded.
    u32 RecordUploads(Rhi::CommandList& list);

    /// Records copying an image back into a buffer in its guest layout (tiled again), for
    /// guest memory the CPU reads after the GPU wrote it.
    bool RecordDownload(Rhi::CommandList& list, ImageEntry& entry, const Rhi::Buffer& out,
                        u64 out_offset = 0);

    struct Stats {
        std::atomic<u64> lookups, hits, created_textures, created_targets, uploads, upload_bytes;
        std::atomic<u64> evicted;
        std::atomic<u64> reused_larger; ///< NAT-113
        std::array<std::atomic<u64>, 3> stale; ///< textures marked, by Stale
    };
    const Stats& GetStats() const {
        return stats;
    }

private:
    struct Shard {
        std::shared_mutex mutex;
        std::unordered_multimap<VAddr, std::unique_ptr<ImageEntry>> images; // by guest address
    };
    static constexpr u32 NumShards = 64;
    Shard& ShardOf(VAddr address) {
        return shards[(address >> 12) * 0x9E3779B97F4A7C15ULL >> 58];
    }

    const Rhi::Pipeline* TilingPipeline(const VideoCore::ImageInfo& info, bool tiler);
    void LinkTexture(ImageEntry& texture);
    void LinkTarget(ImageEntry& target);
    void MarkStale(ImageEntry& entry, Stale why);

    // NAT-109: images by the 64 KB granules of guest memory they cover (overlap queries look at
    // the granules of the range instead of every image).
    static constexpr u32 GranuleBits = 16;
    static constexpr u64 MaxIndexedSize = u64{1} << 30;
    std::shared_mutex granule_mutex;
    std::unordered_map<u64, std::vector<ImageEntry*>> granules;
    void IndexEntry(ImageEntry& entry);
    void UnindexEntry(ImageEntry& entry);
    std::unique_ptr<Rhi::Buffer> TilingParamsBuffer(const VideoCore::ImageInfo& info);

    Rhi::Device& device;
    ImageFactory image_factory;
    std::function<u64(VAddr, u64)> memory_clamp;
    std::function<void(ImageEntry&)> upload_hook;
    std::function<UploadSource(const ImageEntry&, Rhi::CommandList&)> upload_source;
    UploadStats upload_stats{};
    std::array<Shard, NumShards> shards;
    std::mutex upload_mutex;
    std::vector<ImageEntry*> pending_uploads;
    std::unique_ptr<Rhi::PipelineLayout> untile_layout;
    std::mutex untile_mutex;
    std::unordered_map<u64, std::unique_ptr<Rhi::Pipeline>> untile_pipelines;
    Stats stats{};
    std::mutex overlap_log_mutex;
    std::unordered_set<u64> overlap_logged;
};

/// The buffer-to-image copies of every mip of an image in its guest (untiled) layout.
std::vector<Rhi::BufferImageCopy> MipCopies(const VideoCore::ImageInfo& info);

/// Whether two layouts at the same address can share one image (the cached one's colour format
/// may differ from the request's when the texel size is the same).
bool SameLayout(const VideoCore::ImageInfo& a, const VideoCore::ImageInfo& b);

SourcePlace PlaceOf(const VideoCore::ImageInfo& target, const VideoCore::ImageInfo& texture);

} // namespace Native
