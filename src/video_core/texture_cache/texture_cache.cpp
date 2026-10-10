// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <unordered_set>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/read_capture.h"
#include "common/path_util.h"
#include "common/hash.h"
#include "common/perf_monitor.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

static constexpr u32 MAX_IMAGES = std::numeric_limits<u16>::max();
static constexpr u32 MAX_IMAGE_VIEWS = std::numeric_limits<u16>::max();
static constexpr u32 MAX_SAMPLERS = std::numeric_limits<u16>::max();

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                           BufferCache& buffer_cache_, PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, slot_images{MAX_IMAGES},
      slot_image_views{MAX_IMAGE_VIEWS}, slot_samplers{MAX_SAMPLERS},
      blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, runtime, buffer_cache.GetStreamBuffer()},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    const s64 min_pressure_floor =
        std::clamp<s64>(device_local_memory / 4, 256_MB, DEFAULT_PRESSURE_GC_MEMORY);
    const s64 min_critical_floor =
        std::clamp<s64>(device_local_memory / 2, 512_MB, DEFAULT_CRITICAL_GC_MEMORY);
    pressure_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      min_pressure_floor));
    critical_gc_memory = static_cast<u64>(
        std::max<s64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      min_critical_floor));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
}

TextureCache::~TextureCache() = default;

void TextureCache::ProcessDownloadImages() {
    auto readbacks = RecordPendingReadbacks();
    if (readbacks.empty()) {
        ReleaseFinishedReadbacks();
        return;
    }
    scheduler.Finish();
    CompleteReadbacks(readbacks);
    ReleaseFinishedReadbacks();
}

bool TextureCache::DumpImage(ImageId image_id, const std::filesystem::path& path) {
    Image& image = slot_images[image_id];
    const u32 bytes_per_texel = image.info.num_bits / 8;
    if (image.info.props.is_depth || image.info.props.is_block || bytes_per_texel == 0) {
        return false;
    }
    const u32 width = image.info.size.width;
    const u32 height = image.info.size.height;
    const u32 size = width * height * bytes_per_texel;
    const auto download = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16);
    const vk::BufferImageCopy copy = {
        .bufferOffset = download.offset,
        .bufferRowLength = width,
        .bufferImageHeight = height,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&copy, 1});
    scheduler.Finish();
    download.Invalidate();
    if (std::FILE* file = std::fopen(path.string().c_str(), "wb")) {
        std::fwrite(download.mapped, 1, size, file);
        std::fclose(file);
    }
    return true;
}

bool TextureCache::HasLargePendingReadbacks() {
    std::unique_lock lk{download_images_mutex};
    return std::ranges::any_of(download_images, [this](ImageId id) {
        const auto& info = slot_images[id].info;
        return info.pitch * info.size.height * (info.num_bits / 8) >= 64_KB;
    });
}

bool TextureCache::HasPendingReadbacks() {
    std::unique_lock lk{download_images_mutex};
    return !download_images.empty();
}

std::vector<TextureCache::PendingReadback> TextureCache::RecordPendingReadbacks() {
    std::unique_lock lk{download_images_mutex};
    std::vector<PendingReadback> readbacks;
    for (const ImageId image_id : download_images) {
        if (auto readback = RecordImageReadback(image_id)) {
            readbacks.push_back(*readback);
        }
    }
    download_images.clear();
    return readbacks;
}

// FIX-022: hash of guest memory read through its backing (no protection faults).
static u64 HashGuestBytes(VAddr address, u32 size) {
    thread_local std::vector<u8> bytes;
    bytes.resize(size);
    Core::Memory::Instance()->CopySparseMemory(address, bytes.data(), size);
    return XXH3_64bits(bytes.data(), size);
}

// FIX-029: guest memory read through its backing (no protection faults).
static std::shared_ptr<std::vector<u8>> CopyGuestBytes(VAddr address, u32 size) {
    auto bytes = std::make_shared<std::vector<u8>>(size);
    Core::Memory::Instance()->CopySparseMemory(address, bytes->data(), size);
    return bytes;
}

void TextureCache::CompleteReadbacks(std::span<const PendingReadback> readbacks) {
    // May run on the scheduler thread after the GPU work completed. The guest may have unmapped
    // the memory meanwhile (for example while loading a race); skip those writes.
    // FIX-022: skip it too when the guest changed the bytes after the readback was recorded.
    // GT Sport frees a small render target's memory and reuses it for its heap while the
    // readback is in flight; the stale image then overwrote a free-list link and the game
    // crashed in its allocator after buying a car or starting a race.
    // FIX-029: skipping the whole readback lost GPU results the game needed: GT Sport's 448x126
    // car thumbnail target lives in its heap too, and a skipped readback left the PNG encoder
    // without the picture (crash after buying a car). Only the 32-bit words the guest changed
    // keep the guest's value now; the rest gets the GPU result. -DisablePerf 38 skips whole.
    static const bool check_guest_bytes = Common::PerfFeatureEnabled(33);
    static const bool merge_words = Common::PerfFeatureEnabled(38);
    auto* memory = Core::Memory::Instance();
    for (const auto& readback : readbacks) {
        if (check_guest_bytes && merge_words && readback.guest_bytes &&
            memory->IsValidMapping(readback.address, readback.size)) {
            std::vector<u8> merged(readback.size);
            memory->CopySparseMemory(readback.address, merged.data(), readback.size);
            readback.download.Invalidate();
            const u8* gpu = readback.download.mapped;
            const u8* recorded = readback.guest_bytes->data();
            u32 kept = 0;
            for (u32 offset = 0; offset < readback.size; offset += 4) {
                const u32 n = std::min<u32>(4, readback.size - offset);
                if (std::memcmp(merged.data() + offset, recorded + offset, n) == 0) {
                    std::memcpy(merged.data() + offset, gpu + offset, n);
                } else {
                    ++kept;
                }
            }
            if (kept != 0) {
                static std::atomic<u32> merges{};
                if (const u32 n = ++merges; n <= 20 || n % 500 == 0 || readback.size >= 64_KB) {
                    LOG_WARNING(Render_Vulkan,
                                "FIX-029: image readback {} to {:#x}+{:#x} kept {} words the "
                                "guest changed after it was recorded",
                                n, readback.address, readback.size, kept);
                }
            }
            if (readback.size >= 64_KB) {
                // DIAG-038: large readbacks (GT Sport's car thumbnail) and what they carried.
                u32 nonzero = 0;
                for (u32 offset = 0; offset + 4 <= readback.size; offset += 4) {
                    u32 word;
                    std::memcpy(&word, gpu + offset, 4);
                    nonzero += word != 0;
                }
                LOG_WARNING(Render_Vulkan,
                            "DIAG-038: readback to {:#x}+{:#x} completed: {} of {} GPU words "
                            "nonzero, {} words kept from the guest",
                            readback.address, readback.size, nonzero, readback.size / 4, kept);
                // DIAG-040: the bytes themselves, to look at the picture offline.
                static std::atomic<u32> dumps{};
                if (const u32 n = dumps++; n < 16) {
                    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
                                      fmt::format("readback_{:02}_{:#x}.bin", n, readback.address);
                    if (std::FILE* file = std::fopen(path.string().c_str(), "wb")) {
                        std::fwrite(merged.data(), 1, readback.size, file);
                        std::fclose(file);
                    }
                }
            }
            memory->TryWriteBacking(std::bit_cast<u8*>(readback.address), merged.data(),
                                    readback.size);
            continue;
        }
        if (check_guest_bytes && memory->IsValidMapping(readback.address, readback.size) &&
            HashGuestBytes(readback.address, readback.size) != readback.guest_hash) {
            static std::atomic<u32> skipped{};
            if (const u32 n = ++skipped; n <= 20 || n % 500 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "FIX-022: image readback {} to {:#x}+{:#x} skipped: the guest changed "
                            "that memory after it was recorded",
                            n, readback.address, readback.size);
            }
            continue;
        }
        if (memory->IsValidMapping(readback.address, readback.size)) {
            readback.download.Invalidate();
            memory->TryWriteBacking(std::bit_cast<u8*>(readback.address), readback.download.mapped,
                                    readback.size);
        }
    }
    std::scoped_lock lk{finished_readbacks_mutex};
    for (const auto& readback : readbacks) {
        finished_readbacks.push_back(readback.download);
    }
}

void TextureCache::ReleaseFinishedReadbacks() {
    // Staging memory is owned by the GPU thread.
    std::scoped_lock lk{finished_readbacks_mutex};
    for (const auto& download : finished_readbacks) {
        runtime.GetStagingPool().FreeDeferred(download);
    }
    finished_readbacks.clear();
}

std::optional<TextureCache::PendingReadback> TextureCache::RecordImageReadback(ImageId image_id) {
    Image& image = slot_images[image_id];
    const bool large = image.info.pitch * image.info.size.height * (image.info.num_bits / 8) >= 64_KB;
    if (large && (False(image.flags & ImageFlagBits::GpuModified) ||
                  True(image.flags & ImageFlagBits::Dirty))) {
        LOG_WARNING(Render_Vulkan,
                    "DIAG-038: readback of image {} at {:#x} not recorded: flags {:#x}",
                    image_id.index, image.info.guest_address, u32(image.flags));
    }
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return std::nullopt;
    }
    // FIX-023: the CPU wrote the image's memory after the GPU drew into it (GT Sport frees these
    // small render targets back to its heap); the GPU copy is older than guest memory now, and
    // reading it back overwrote the heap's free-list links. -DisablePerf 33 reads back anyway.
    static const bool skip_cpu_written = Common::PerfFeatureEnabled(33);
    if (skip_cpu_written && True(image.flags & ImageFlagBits::Dirty)) {
        static std::atomic<u32> skipped{};
        if (const u32 n = ++skipped; n <= 20 || n % 500 == 0) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-023: readback {} of image at {:#x}+{:#x} skipped: the CPU wrote that "
                        "memory after the GPU drew into it",
                        n, image.info.guest_address, image.info.guest_size);
        }
        return std::nullopt;
    }
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    if (download_size == 0 || download_size > image.info.guest_size) {
        return std::nullopt;
    }
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, true);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    // DIAG-037: readback recorded, in the crash report's write history.
    static constexpr u32 NoData = 0;
    Core::MemoryManager::NoteEmulatorWrite(image.info.guest_address, 0, &NoData);
    image.readback_version = image.contents_version;
    if (download_size >= 64_KB) {
        LOG_WARNING(Render_Vulkan, "DIAG-038: readback of image {} at {:#x}+{:#x} recorded",
                    image_id.index, image.info.guest_address, download_size);
    }
    static const bool merge_words = Common::PerfFeatureEnabled(38);
    return PendingReadback{image.info.guest_address, download, download_size,
                           HashGuestBytes(image.info.guest_address, download_size),
                           merge_words ? CopyGuestBytes(image.info.guest_address, download_size)
                                       : nullptr};
}

static u32 ImageDownloadSize(const Image& image) {
    return image.info.pitch * image.info.size.height * image.info.size.depth *
           image.info.resources.layers * (image.info.num_bits / 8);
}

bool TextureCache::ShouldReadBack(const Image& image) {
    if (!readback_linear_images || (image.info.props.is_tiled && image.info.size.width > 8) ||
        image.info.guest_address == 0 || image.info.props.is_depth || image.info.size.depth > 1 ||
        image.info.num_samples > 1 || image.info.resources.layers > 1 ||
        ImageDownloadSize(image) > image.info.guest_size) {
        return false;
    }
    // CPU-read GPU results such as luminance chains are tiny; large linear targets are not
    // read back, which avoids heavy copies every submission. GT Sport renders car thumbnails
    // into 448x126 linear targets (225 KB) that its CPU PNG-encodes; with a 64 KB limit the
    // encoder saw empty memory and the game stopped with "BREAK! thumbnail_functions.ad:473".
    static constexpr u32 MaxReadbackSize = 256_KB;
    const u32 size = ImageDownloadSize(image);
    if (size <= MaxReadbackSize) {
        static std::unordered_set<VAddr> logged_readbacks;
        if (logged_readbacks.size() < 16 &&
            logged_readbacks.insert(image.info.guest_address).second) {
            LOG_WARNING(Render_Vulkan, "Reading back {}x{} linear image at {:#x} ({} bytes)",
                        image.info.size.width, image.info.size.height, image.info.guest_address,
                        size);
        }
        return true;
    }
    static std::unordered_set<u64> logged_sizes;
    const u64 key = u64(image.info.size.width) << 32 | image.info.size.height;
    if (logged_sizes.size() < 16 && logged_sizes.insert(key).second) {
        LOG_WARNING(Render_Vulkan, "Not reading back {}x{} linear image at {:#x} ({} bytes)",
                    image.info.size.width, image.info.size.height, image.info.guest_address, size);
    }
    return false;
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    // FIX-024: read-back images reach guest memory through their readbacks, which already skip
    // memory the CPU wrote after the GPU (FIX-022, FIX-023). Writing the old result again when the
    // cache evicts the image overwrote a GT Sport heap free-list link: the game frees these small
    // render targets to its heap (the dealership and race start crashes). -DisablePerf 34 writes
    // it anyway.
    static const bool skip_written_back = Common::PerfFeatureEnabled(34);
    if (skip_written_back && ShouldReadBack(image)) {
        // A GPU copy into the image after its last readback is not read back; log those.
        const bool read_back = image.readback_version == image.contents_version;
        static std::atomic<u32> skipped{};
        if (const u32 n = ++skipped; n <= 20 || n % 500 == 0 || !read_back) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-024: download {} of read-back image at {:#x}+{:#x} skipped ({})", n,
                        image.info.guest_address, image.info.guest_size,
                        read_back ? "its last GPU result was read back"
                                  : "a GPU copy after its last readback was not read back");
        }
        return;
    }
    static std::atomic<u32> downloads{};
    if (const u32 n = ++downloads; n <= 50 || n % 500 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "FIX-024: download {} of evicted image at {:#x}+{:#x}: {}x{} {} bits depth {}",
                    n, image.info.guest_address, image.info.guest_size, image.info.size.width,
                    image.info.size.height, image.info.num_bits, bool(image.info.props.is_depth));
    }
    const u32 download_size = ImageDownloadSize(image);
    ASSERT(download_size <= image.info.guest_size);
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, !sync);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    if (sync) {
        scheduler.Finish();
        download.Invalidate();
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image.info.guest_address),
                                                  download.mapped, download_size);
    } else {
        // FIX-024: the image is freed (and its pages unprotected) before this write lands, so
        // a guest write in between goes unseen; skip the write if the bytes changed meanwhile.
        const u64 guest_hash =
            skip_written_back ? HashGuestBytes(image.info.guest_address, download_size) : 0;
        scheduler.DeferPriorityOperation([this, device_addr = image.info.guest_address, download,
                                          download_size, guest_hash] {
            if (skip_written_back && HashGuestBytes(device_addr, download_size) != guest_hash) {
                LOG_WARNING(Render_Vulkan,
                            "FIX-024: download to {:#x}+{:#x} skipped: the guest changed that "
                            "memory while it was in flight",
                            device_addr, download_size);
            } else {
                download.Invalidate();
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr),
                                                          download.mapped, download_size);
            }
            runtime.GetStagingPool().FreeDeferred(download);
        });
    }
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags = ImageFlagsAfterCpuWrite(image.flags, true);
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);

    SmallVector<ImageId, 8> image_ids;
    ForEachImageInRegion(pages_start, pages_end - pages_start,
                         [&](ImageId image_id, Image&) { image_ids.push_back(image_id); });
    if (!image_ids.empty()) {
        // DIAG-037: a CPU write fault on image memory, in the crash report's write history.
        static constexpr u32 NoData = 0;
        Core::MemoryManager::NoteEmulatorWrite(addr, 0, &NoData);
    }

    for (const auto image_id : image_ids) {
        Image& image = slot_images[image_id];
        std::scoped_lock lk{image.mutex};

        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags = ImageFlagsAfterCpuWrite(image.flags);
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    }
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only full-image preservation proves an interior buffer write can invalidate a
        // clean rendered alias, including after a texture refresh cleared its dirty flag.
        if (!CanInvalidateImageFromGPU(image.flags, image.info.guest_address == address)) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    SmallVector<ImageId, 16> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        new_info.UpdateSize();
        const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;
        runtime.CopyColorAndDepth(&cache_image, &new_image);

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    static constexpr u64 NUM_FRAMES_BEFORE_REMOVAL = 32;

    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NUM_FRAMES_BEFORE_REMOVAL;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        // Cropped descriptors need their own image because views cannot change dimensions.
        if (image_info.IsSubrectOf(cache_image.info)) {
            ImageId result_id = merged_image_id;
            if (!result_id) {
                result_id = slot_images.Insert(instance, runtime, slot_image_views, image_info);
                RegisterImage(result_id);
            }
            auto& source = slot_images[cache_image_id];
            auto& destination = slot_images[result_id];
            if (source.contents_version > destination.contents_version && source.SafeToDownload()) {
                runtime.CopySubrect(&source, &destination);
            }
            return {result_id, -1, -1};
        }

        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // A full mip chain (or more layers) of the same memory under another color format of the
        // same texel size: expand and copy, as for the same format (FIX-040, perf id 42).
        static const bool extend_across_format = Common::PerfFeatureEnabled(42);
        if (extend_across_format && image_info.ExtendsAcrossFormat(cache_image.info)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool pow2_padding_only =
            image_info.props.is_pow2 != cache_image.info.props.is_pow2 &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            image_info.size == cache_image.info.size &&
            image_info.pitch == cache_image.info.pitch && image_info.resources.levels == 1 &&
            cache_image.info.resources.levels == 1 && image_info.resources.layers == 1 &&
            cache_image.info.resources.layers == 1;

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size || pow2_padding_only) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Same pow2:             {}\n"
                  "  Same alt tile:         {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,
                  bool(cache_image.info.props.is_pow2), cache_image.info.alt_tile,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,
                  bool(image_info.props.is_pow2), image_info.alt_tile,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (image_info.props.is_pow2 == cache_image.info.props.is_pow2),
                  (image_info.alt_tile == cache_image.info.alt_tile),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    runtime.CopyMip(&cache_image, &merged_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    runtime.CopyImage(&src_image, &new_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    ++Common::GetWorkCounters().find_image;
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    // A perfect match, and every crop of it, starts at the requested address, so look at those
    // images first. Only walk the whole range (every page of a large target) when there is no
    // perfect match and overlaps have to be resolved.
    const auto collect_same_address = [&](SmallVector<ImageId, 8>& ids) {
        ids.clear();
        ForEachImageInRegion(info.guest_address, 1, [&](ImageId id, Image& image) {
            if (image.info.guest_address == info.guest_address) {
                ids.push_back(id);
            }
        });
    };
    SmallVector<ImageId, 8> same_address;
    collect_same_address(same_address);

    ImageId image_id{};
    // PERF-041: whether overlap resolution or a new image may have changed the images at this
    // address; only then are they collected again below.
    bool registry_changed = false;

    // Check for a perfect match first
    for (const auto& cache_id : same_address) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        // Depth attachment formats must agree with the pipeline, even when their sampled
        // aspects are compatible. Let overlap resolution preserve and convert their contents.
        if (desc.type == BindingType::DepthTarget &&
            (cache_image.info.pixel_format != info.pixel_format ||
             cache_image.info.props.has_stencil != info.props.has_stencil)) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        SmallVector<ImageId, 8> image_ids;
        ForEachImageInRegion(info.guest_address, info.guest_size,
                             [&](ImageId id, Image&) { image_ids.push_back(id); });
        registry_changed |= !image_ids.empty();
        for (const auto& cache_id : image_ids) {
            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (!overlap_image_id) {
                continue;
            }
            // A later overlap that keeps the same image says nothing about the subresource;
            // keep the mip/slice found earlier. Otherwise a render target on mip 1 of a mip
            // chain falls back to mip 0 whenever another image overlaps the chain (GT Sport's
            // depth-of-field mips overwrite the top-left of mip 0).
            if (overlap_image_id != image_id || overlap_view_mip >= 0 || overlap_view_slice >= 0) {
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
            image_id = overlap_image_id;
        }
    } else {
        for (const auto& cache_id : same_address) {
            if (cache_id != image_id &&
                slot_images[image_id].info.IsSubrectOf(slot_images[cache_id].info)) {
                registry_changed = true;
                ResolveOverlap(slot_images[image_id].info, desc.type, cache_id, image_id);
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            registry_changed = true;
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        registry_changed = true;
        image_id = slot_images.Insert(instance, runtime, slot_image_views, info);
        RegisterImage(image_id);
    }

    // Cropped images share the full image's guest memory. When one was written after the full
    // image, its pixels are the current guest contents of that rectangle, so merge them back
    // (oldest first) before the full image is used. GT Sport draws its car into a 1200-wide
    // crop of a 1920-wide target, then blends over it through the full-width descriptor.
    // Overlap resolution may have freed or created images, so collect them again.
    static const bool reuse_collected = Common::PerfFeatureEnabled(53);
    if (registry_changed || !reuse_collected) {
        collect_same_address(same_address);
    }
    SmallVector<ImageId, 4> newer_subrects;
    for (const auto& cache_id : same_address) {
        if (cache_id == image_id ||
            !slot_images[cache_id].info.IsSubrectOf(slot_images[image_id].info)) {
            continue;
        }
        const Image& subrect = slot_images[cache_id];
        if (subrect.contents_version > slot_images[image_id].contents_version &&
            subrect.SafeToDownload()) {
            newer_subrects.push_back(cache_id);
        }
    }
    if (!newer_subrects.empty()) {
        std::ranges::sort(newer_subrects, {},
                          [&](ImageId id) { return slot_images[id].contents_version; });
        RefreshImage(slot_images[image_id]);
        for (const auto subrect_id : newer_subrects) {
            runtime.CopySubrect(&slot_images[subrect_id], &slot_images[image_id]);
        }
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    SmallVector<ImageId, 4> image_ids;
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

// DIAG-036: the first time each image is queued for readback, what it is and who writes it.
static void LogReadbackImage(ImageId image_id, const Image& image, const char* writer) {
    static std::mutex mutex;
    static std::unordered_set<u32> seen;
    std::scoped_lock lk{mutex};
    if (seen.size() >= 400 || !seen.insert(image_id.index).second) {
        return;
    }
    const auto& info = image.info;
    LOG_WARNING(Render_Vulkan,
                "DIAG-036: readback image {} ({}) at {:#x}+{:#x}: {}x{}x{} pitch {} {} bits {} "
                "tile {} array {} levels {} layers {} samples {}",
                image_id.index, writer, info.guest_address, info.guest_size, info.size.width,
                info.size.height, info.size.depth, info.pitch, info.num_bits,
                vk::to_string(info.pixel_format), magic_enum::enum_name(info.tile_mode),
                magic_enum::enum_name(info.array_mode), info.resources.levels,
                info.resources.layers, info.num_samples);
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.MarkGpuModified();
        if (ShouldReadBack(image)) {
            LogReadbackImage(image_id, image, "storage image");
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.MarkGpuModified();
    if (ShouldReadBack(image)) {
        LogReadbackImage(image_id, image, "render target");
        // DIAG-037: drawn into (queued for readback), in the crash report's write history.
        static constexpr u32 NoData = 0;
        Core::MemoryManager::NoteEmulatorWrite(image.info.guest_address, 0, &NoData);
        std::unique_lock lk{download_images_mutex};
        if (download_images.emplace(image_id).second && ImageDownloadSize(image) >= 64_KB) {
            // DIAG-038: each time a large read-back target is drawn into again.
            LOG_WARNING(Render_Vulkan, "DIAG-038: image {} at {:#x} drawn, readback queued",
                        image_id.index, image.info.guest_address);
        }
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaType::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaType::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.MarkGpuModified();
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaType::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        ImageId stencil_id{};
        ForEachImageInRegion(
            desc.info.stencil_addr, desc.info.stencil_size, [&](ImageId image_id, Image& image) {
                if (image.info.guest_address != desc.info.stencil_addr) {
                    return;
                }
                if (image.info.pixel_format == vk::Format::eUndefined ||
                    Vulkan::LiverpoolToVK::IsFormatStencilCompatible(image.info.pixel_format)) {
                    stencil_id = image_id;
                }
            });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id = slot_images.Insert(instance, runtime, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, image.image_uid);
    }

    return image.FindView(desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        // PERF-063: through the recorder's capture, like the upload that may follow.
        std::array<u8, 8 * 8 * 16> first_pixels;
        Common::CopyGuest(first_pixels.data(), addr, std::min<size_t>(size, first_pixels.size()));
        const u64 hash = XXH3_64bits(first_pixels.data(), std::min<size_t>(size, first_pixels.size()));
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    SmallVector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];
        const u32 extent_width = mip_pitch ? std::min<u32>(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min<u32>(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags = ImageFlagsAfterBufferUpload(image.flags, false);
        return;
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    const auto [buffer, offset] = tile_manager.DetileImage(in_buffer, in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    // Refresh covers every mip and layer from the canonical guest buffer footprint.
    runtime.UploadImage(&image, buffer, image_copies, true);
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sharp,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth) {
    // Compare and plain uses of one S# need separate samplers.
    const u64 hash = HashCombine(XXH3_64bits(&sharp, sizeof(sharp)), is_depth);
    const auto [it, new_sampler] = samplers.try_emplace(hash);
    if (new_sampler) {
        it->second = slot_samplers.Insert(instance, sharp, border_color_base, is_depth);
        Sampler& sampler = slot_samplers[it->second];
        sampler.hash = hash;
        sampler_lru_cache.Insert(sampler, gc_tick);
    }

    Sampler& sampler = slot_samplers[it->second];
    sampler_lru_cache.Touch(sampler, gc_tick);
    return sampler.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    ++registry_generation; // PERF-057
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image_lru_cache.Insert(image, gc_tick);
    const auto& info = image.info;
    ASSERT_MSG((info.guest_address & 0xff) == 0, "Trying to register an unaligned image");
    ForEachPage(info.guest_address, info.guest_size, [this, image_id, info](u64 page) {
        page_table[page].entries.emplace_back(BucketEntry{
            .key = u32(info.guest_address >> 8),
            .size = info.guest_size,
            .id = image_id,
        });
    });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    ++registry_generation; // PERF-057
    image_lru_cache.Free(image);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        ASSERT_MSG(page_it, "Unregistering unregistered page={:#x}", page << Traits::PAGE_BITS);
        auto& entries = page_it->entries;
        const auto vector_it = std::ranges::find(entries, image_id, &BucketEntry::id);
        ASSERT_MSG(vector_it != entries.end(), "Unregistering unregistered image in page={:#x}",
                   page << Traits::PAGE_BITS);
        entries.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    std::scoped_lock lk{image.mutex};
    if (image.IsUntracked()) {
        tracker.UpdatePageWatchers(image_begin, image.info.guest_size, PageOp::Track);
    } else {
        if (image_begin < image.track_addr) {
            tracker.UpdatePageWatchers(image_begin, image.track_addr - image_begin, PageOp::Track);
        }
        if (image.track_addr_end < image_end) {
            tracker.UpdatePageWatchers(image.track_addr_end, image_end - image.track_addr_end,
                                       PageOp::Track);
        }
    }
    image.track_addr = image_begin;
    image.track_addr_end = image_end;
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (image.IsUntracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    if (size != 0) {
        tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
    }
    image.track_addr = 0;
    image.track_addr_end = 0;
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (image.IsUntracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    tracker.UpdatePageWatchers(image_begin, size, PageOp::Untrack);

    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image.IsUntracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers(addr, size, PageOp::Untrack);
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](Image& image) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        const auto image_id = slot_images.GetSlotId(image);
        if (download) {
            DownloadImageMemory(image_id);
        }
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    image_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        image_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](Sampler& sampler) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        sampler_lru_cache.Free(sampler);
        samplers.erase(sampler.hash);
        const auto sampler_id = slot_samplers.GetSlotId(sampler);
        slot_samplers.Erase(sampler_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    GarbageCollectImages();
    GarbageCollectSamplers();
    ++gc_tick;
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(image.IsUntracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            const auto& image = slot_images[image_id];
            if (ImageDownloadSize(image) >= 64_KB) {
                LOG_WARNING(Render_Vulkan,
                            "DIAG-038: image {} at {:#x} freed before its readback was recorded",
                            image_id.index, image.info.guest_address);
            }
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.Erase(image_view_id);
            }
        }
        slot_images.Erase(image_id);
    });
}

} // namespace VideoCore
