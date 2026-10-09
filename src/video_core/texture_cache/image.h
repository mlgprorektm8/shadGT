// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <vector>

#include "common/enum.h"
#include "common/incremental_id.h"
#include "common/lru_cache.h"
#include "common/small_vector.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_barriers.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"

namespace Vulkan {
class Instance;
class Runtime;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

namespace VideoCore {

enum ImageFlagBits : u32 {
    Empty = 0,
    MaybeCpuDirty = 1 << 0, ///< The page this image is in was touched before the image address
    CpuDirty = 1 << 1,      ///< Contents have been modified from the CPU
    GpuDirty = 1 << 2, ///< Contents have been modified from the GPU (valid data in buffer cache)
    Dirty = MaybeCpuDirty | CpuDirty | GpuDirty,
    GpuModified = 1 << 3,    ///< Contents have been modified from the GPU
    BufferCoherent = 1 << 4, ///< Full guest footprint has been preserved in the buffer cache
    Registered = 1 << 6,     ///< True when the image is registered
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

constexpr ImageFlagBits ImageFlagsAfterGpuWrite(
    ImageFlagBits flags, ImageFlagBits modified = ImageFlagBits::GpuModified) {
    return (flags | modified) & ~ImageFlagBits::BufferCoherent;
}

constexpr ImageFlagBits ImageFlagsAfterCpuWrite(ImageFlagBits flags, bool maybe = false) {
    return (flags | (maybe ? ImageFlagBits::MaybeCpuDirty : ImageFlagBits::CpuDirty)) &
           ~ImageFlagBits::BufferCoherent;
}

constexpr ImageFlagBits ImageFlagsAfterBufferUpload(ImageFlagBits flags,
                                                    bool preserve_buffer_coherence) {
    flags &= ~ImageFlagBits::Dirty;
    return preserve_buffer_coherence ? flags : flags & ~ImageFlagBits::BufferCoherent;
}

constexpr bool CanInvalidateImageFromGPU(ImageFlagBits flags, bool base_matches) {
    const bool safe_to_download =
        True(flags & ImageFlagBits::GpuModified) && False(flags & ImageFlagBits::Dirty);
    return base_matches || !safe_to_download || True(flags & ImageFlagBits::BufferCoherent);
}

struct UniqueImage {
    explicit UniqueImage() = default;
    explicit UniqueImage(vk::Device device, VmaAllocator allocator)
        : device{device}, allocator{allocator} {}
    ~UniqueImage();

    UniqueImage(const UniqueImage&) = delete;
    UniqueImage& operator=(const UniqueImage&) = delete;

    UniqueImage(UniqueImage&& other)
        : allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          image{std::exchange(other.image, VK_NULL_HANDLE)}, image_ci{std::move(other.image_ci)} {}
    UniqueImage& operator=(UniqueImage&& other) {
        image = std::exchange(other.image, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        image_ci = std::move(other.image_ci);
        return *this;
    }

    void Create(const vk::ImageCreateInfo& image_ci);

    void Destroy();

    operator vk::Image() const {
        return image;
    }

    operator bool() const {
        return image;
    }

public:
    vk::Device device{};
    VmaAllocator allocator{};
    VmaAllocation allocation{};
    vk::Image image{};
    vk::ImageCreateInfo image_ci{};
    vk::DeviceSize size_bytes{};
};

struct Image;

/// DIAG-042: an address window whose image modifications are logged with their caller (GT Sport's
/// car thumbnail source by default; SHADGT_WATCH_IMAGE=<hex address> picks another).
bool IsWatchedImageAddress(VAddr address, u64 size);
void NoteWatchedImageModification(const Image& image, std::source_location loc);

/// DIAG-044: every image modification, in a ring, to explain how a picture got its contents.
struct ImageModRecord {
    u64 uid;
    u64 version;
    const char* file;
    u32 line;
};
void NoteImageModification(u64 uid, u64 version, std::source_location loc);
/// The newest modifications of an image, oldest first.
std::vector<ImageModRecord> RecentImageModifications(u64 uid, size_t max_count);
/// The watched address windows (DIAG-043 dumps the images in them).
std::span<const VAddr> WatchedImageWindows();
constexpr u64 WatchedImageWindowSize = 2_MB;

/// DIAG-043: recent records of watched images, kept in memory and written to the log when the car
/// thumbnail is drawn, so the log shows the history that led to it without flooding.
void PushDiagRecord(std::string record);
void FlushDiagRecords(u32 then_live);

struct Image : public Common::LRUNode<> {
    explicit Image(const Vulkan::Instance& instance, Vulkan::Runtime& runtime,
                   Common::SlotVector<ImageView>& slot_image_views, const ImageInfo& info);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = delete;
    Image& operator=(Image&&) = delete;

    bool Overlaps(VAddr addr, size_t size) const noexcept {
        return info.guest_address < (addr + size) && addr < (info.guest_address + info.guest_size);
    }

    vk::Image GetImage() const {
        return backing->image.image;
    }

    bool IsUntracked() {
        return track_addr == 0 || track_addr_end == 0;
    }

    bool SafeToDownload() const {
        return True(flags & ImageFlagBits::GpuModified) && False(flags & ImageFlagBits::Dirty);
    }

    void MarkModified(std::source_location loc = std::source_location::current()) {
        contents_version = global_contents_version.Next();
        NoteImageModification(image_uid, contents_version, loc);
        if (IsWatchedImageAddress(info.guest_address, info.guest_size)) {
            NoteWatchedImageModification(*this, loc);
        }
    }

    void MarkGpuModified(ImageFlagBits modified = ImageFlagBits::GpuModified,
                         std::source_location loc = std::source_location::current()) {
        MarkModified(loc);
        flags = ImageFlagsAfterGpuWrite(flags, modified);
    }

    void AssociateDepth(ImageId depth_image_id, u64 depth_image_uid) {
        depth_id = depth_image_id;
        depth_uid = depth_image_uid;
    }

    void DisassociateDepth() {
        depth_id = {};
        depth_uid = {};
    }

    ImageView& FindView(const ImageViewInfo& view_info, bool ensure_guest_samples = true);

    using Barriers = SmallVector<vk::ImageMemoryBarrier2, 32>;
    void GetBarriers(Barriers& out_barriers, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                     vk::PipelineStageFlags2 dst_stage,
                     std::optional<SubresourceRange> subres_range = {});

public:
    Vulkan::Runtime* runtime;
    Common::SlotVector<ImageView>* slot_image_views;
    std::mutex mutex;
    ImageInfo info;
    vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eColor;
    vk::SampleCountFlags supported_samples = vk::SampleCountFlagBits::e1;
    ImageFlagBits flags = ImageFlagBits::Dirty;
    VAddr track_addr = 0;
    VAddr track_addr_end = 0;
    ImageId depth_id{};
    u64 depth_uid{};

    vk::ImageUsageFlags usage_flags;
    vk::FormatFeatureFlags2 format_features;
    using State = ImageBarrierState;
    struct BackingImage : ImageBarrierTracking {
        UniqueImage image;
        SmallVector<ImageViewInfo, 2> image_view_infos;
        SmallVector<ImageViewId, 2> image_view_ids;
        u32 num_samples;
    };
    SmallVector<BackingImage, 2> backing_images;
    BackingImage* backing{};
    u64 image_uid{};
    u64 contents_version{};
    // FIX-024: contents_version of the last readback recorded; guest memory already has it.
    u64 readback_version = ~0ULL;
    u64 lru_id{};
    u64 tick_accessed_last{};
    u64 hash{};

    struct {
        u32 texture : 1;
        u32 storage : 1;
        u32 render_target : 1;
        u32 depth_target : 1;
        u32 vo_surface : 1;
    } usage{};

    struct {
        u32 is_bound : 1;
        u32 is_target : 1;
        u32 needs_rebind : 1;
        u32 force_general : 1;
    } binding{};

private:
    static Common::IncrementalIdProvider<u64> global_image_uid;
    static Common::IncrementalIdProvider<u64> global_contents_version;
};

} // namespace VideoCore
