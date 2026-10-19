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
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
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

class ImageEntry {
public:
    VideoCore::ImageInfo info;
    std::unique_ptr<Rhi::Image> image;
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

    /// The image for this layout: one already at the address with the same layout (whatever
    /// its colour format, if the texel size matches), or a new one. A texture found nowhere is
    /// queued for upload from guest memory; a render target is not. Thread-safe.
    ImageEntry* Find(const VideoCore::ImageInfo& info, bool render_target, u64 frame);

    /// CPU writes to guest memory (file loads, the game writing texture data): textures over the
    /// range are uploaded again before their next use.
    void Invalidate(VAddr address, u64 size);

    /// Records the uploads queued since the last call (copy from guest memory, untile, copy
    /// into the images). Returns how many images were uploaded.
    u32 RecordUploads(Rhi::CommandList& list);

    /// Records copying an image back into a buffer in its guest layout (tiled again), for
    /// guest memory the CPU reads after the GPU wrote it.
    bool RecordDownload(Rhi::CommandList& list, ImageEntry& entry, const Rhi::Buffer& out);

    struct Stats {
        std::atomic<u64> lookups, hits, created_textures, created_targets, uploads, upload_bytes;
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
    std::unique_ptr<Rhi::Buffer> TilingParamsBuffer(const VideoCore::ImageInfo& info);

    Rhi::Device& device;
    std::array<Shard, NumShards> shards;
    std::mutex upload_mutex;
    std::vector<ImageEntry*> pending_uploads;
    std::unique_ptr<Rhi::PipelineLayout> untile_layout;
    std::mutex untile_mutex;
    std::unordered_map<u64, std::unique_ptr<Rhi::Pipeline>> untile_pipelines;
    Stats stats{};
};

/// The buffer-to-image copies of every mip of an image in its guest (untiled) layout.
std::vector<Rhi::BufferImageCopy> MipCopies(const VideoCore::ImageInfo& info);

/// Whether two layouts at the same address can share one image (the cached one's colour format
/// may differ from the request's when the texel size is the same).
bool SameLayout(const VideoCore::ImageInfo& a, const VideoCore::ImageInfo& b);

} // namespace Native
