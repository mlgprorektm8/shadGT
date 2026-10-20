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
    /// NAT-131: where in the target the picture comes from (a layer of a layered target).
    s32 src_mip = 0;
    s32 src_layer = 0;
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
    /// FIX-153 (diagnosis): uploads in all, and where the last one read its bytes (0: none,
    /// 1: a region, 2: an imported buffer, 3: a gathered copy, 4: a CPU copy).
    std::atomic<u32> total_uploads{0};
    u8 upload_kind = 0;
    /// NAT-109: the GPU wrote its memory (the CPU can't tell what changed yet): upload.
    std::atomic<bool> force_upload{false};
    bool from_cpu = false; ///< texture data the CPU wrote; render targets are GPU-only
    std::atomic<bool> needs_upload{false};
    std::atomic<u64> last_use{0};
    /// FIX-157: the frame it was made (how long a picture has been in use). A picture the GPU
    /// wrote is `kept` past its idle time while nothing else wrote its memory (its pages are
    /// watched from then on); `reused` once something does: nothing finds it any more and it
    /// goes. `foreign_clock`: the renderer's write clock when GPU work last wrote its memory
    /// other than into the picture.
    u64 made_frame = 0;
    std::atomic<bool> kept{false};
    std::atomic<bool> reused{false};
    std::atomic<u64> foreign_clock{0};

    /// NAT-120: GPU work rewrote the guest memory of this render target from its first byte
    /// (GT Sport clears depth buffers and pictures with a fill shader). Applied at the
    /// target's next use: cleared with the fill pattern, or uploaded from memory. The stencil
    /// plane (its own memory) only takes clears. Recorder thread only.
    /// NAT-124: a fill of only the first bytes (a metadata clear in memory a picture shares)
    /// clears only the rows they hold; other writes of part of a picture are left alone (its
    /// picture is not in memory: uploading it wiped glows and the wet road's reflection).
    enum class Rewrite : u8 { None, Upload, Clear };
    Rewrite rewrite = Rewrite::None;
    std::array<u32, 2> rewrite_pattern{};
    u64 rewrite_bytes = 0; ///< how much of the picture's memory the fill wrote
    /// NAT-148: the renderer's write clock when the memory was rewritten: the picture's
    /// version once the rewrite is applied (later, at its next use), so targets drawn between
    /// the fill and that use still count as newer for the textures made of them.
    u64 rewrite_clock = 0;
    bool stencil_clear_pending = false;
    u8 stencil_clear_value = 0;

    /// A view, made on first use.
    const Rhi::ImageView* View(Rhi::Device& device, const ImageViewKey& key);

private:
    std::mutex views_mutex;
    std::vector<std::pair<ImageViewKey, std::unique_ptr<Rhi::ImageView>>> views;
};

/// FIX-153 (diagnosis): RYGT_NATIVE_WATCH_TEXTURE=<hex address>[,<hex address>...]: whether
/// [begin, end) holds one of them; what happens there (writes, unmaps, invalidations, uploads)
/// is logged.
bool WatchedTexture(VAddr begin, VAddr end);
/// FIX-156 (diagnosis): the watched addresses.
const std::vector<VAddr>& WatchedTextures();
/// FIX-156 (diagnosis): a fill of the watched address asks for a frame dump of the next frame.
extern std::atomic<bool> g_dump_request;
/// FIX-153 (diagnosis): the first 400 watched events of a kind, then one in 500.
bool WatchSample();
/// FIX-153 (diagnosis): on while a frame dump collects its frames: every watched event logs.
extern std::atomic<bool> g_watch_window;
/// FIX-153 (diagnosis): the first 16 bytes there, as text ("unreadable" when they are not).
std::string WatchedBytes(VAddr address);

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
        u8 kind = 0; ///< FIX-153 (diagnosis): see ImageEntry::upload_kind
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
    /// OPT-010: `create` false: none when there is none yet (game job threads look ahead).
    ImageEntry* Find(const VideoCore::ImageInfo& info, bool render_target, u64 frame,
                     bool create = true);

    /// OPT-010: whether no picture was made or dropped over any of the ranges since generation
    /// `since` (false when the log of changes no longer reaches back that far).
    bool Untouched(u64 since, std::span<const std::pair<VAddr, VAddr>> ranges);

    /// OPT-002: changes whenever a picture is made or dropped (what Find returns may change).
    u64 Generation() const {
        return generation.load(std::memory_order_acquire);
    }

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

    /// NAT-119/120: GPU work (a shader's storage buffer, a CP fill or copy) wrote [address,
    /// address + size), with `pattern` (two dwords, repeated) when it is a fill. As the old
    /// cache's InvalidateMemoryFromGPU: a render target that starts where the write starts gets
    /// a pending rewrite (ImageEntry::rewrite), textures the write overlaps are uploaded again
    /// (not those made of targets: their pictures come from the targets). A write inside a
    /// render target leaves it (its pictures are on the GPU only).
    /// NAT-129: `exact` when the size is what was written (a fill, a CP copy); a storage buffer's
    /// size is only what the shader may write (GT Sport binds its whole scratch pool), so such a
    /// write uploads a target again only when the buffer is about the target's size.
    void GpuWrote(VAddr address, u64 size, const std::array<u32, 2>* pattern, bool exact,
                  u64 clock);
    /// NAT-120: uploads the image from guest memory before its next use.
    void RequestUpload(ImageEntry& entry) {
        MarkStale(entry, Stale::GpuWrite);
    }

    /// NAT-109: CPU writes to [address, address + size) (a page fault: the page stops being
    /// watched); `exact` is the bytes the instruction wrote there (0: unknown). Small textures
    /// on the page are checked by hash before their next use; large ones only when written.
    void CpuWrite(VAddr address, u64 size, u64 exact);
    /// NAT-109: the addresses holding the most images (address, count), for reports.
    std::vector<std::pair<VAddr, u32>> Crowded(u32 count);

    /// NAT-108: frees the images last used before frame `before`; `on_evict` sees each one
    /// first (the renderer drops what it keeps about it). The images' memory goes once the GPU
    /// has finished the work recorded so far. Returns how many went.
    /// FIX-157: `keep` may keep one past its idle time (asked with no lock held).
    u32 Evict(u64 before, const std::function<void(ImageEntry&)>& on_evict,
              const std::function<bool(ImageEntry&)>& keep = {});
    /// FIX-157: whether guest memory under a picture the GPU wrote still holds what it held
    /// when the picture was last uploaded from it (its content hash).
    bool MemoryUnchanged(const ImageEntry& entry);
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
        std::atomic<u64> targets_rewritten; ///< NAT-119
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
    /// FIX-157: something else wrote a kept picture's memory.
    void MarkReused(ImageEntry& entry);
    /// FIX-157: GPU work wrote part of a picture's memory other than into the picture.
    void ForeignWrite(ImageEntry& entry, u64 size, bool exact, u64 clock);

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
    std::atomic<u64> generation{0}; ///< OPT-002
    /// OPT-010: the ranges of the pictures made or dropped, by the generation they made.
    struct Change {
        u64 generation = 0;
        VAddr begin = 0;
        VAddr end = 0;
    };
    std::mutex changes_mutex;
    std::array<Change, 1024> changes{};
    u64 change_count = 0;
    /// Logs the ranges, then moves the generation on.
    void NoteChanges(std::span<const std::pair<VAddr, VAddr>> ranges);
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
