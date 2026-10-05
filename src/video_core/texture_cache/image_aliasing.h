// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>
#include "common/types.h"
#include "video_core/texture_cache/image.h"

namespace VideoCore {

struct ImageAliasFootprint {
    VAddr address;
    u64 size;
    bool image_authoritative;
    bool buffer_authoritative;
    bool exportable;
};

constexpr ImageAliasFootprint DescribeImageAlias(VAddr address, u64 size, ImageFlagBits flags,
                                                 bool tracked_buffer_data, bool exportable) {
    const bool image_current =
        True(flags & ImageFlagBits::GpuModified) && False(flags & ImageFlagBits::Dirty);
    // A later render revokes BufferCoherent; older tracked buffer bytes must not block it.
    return {address, size, image_current && False(flags & ImageFlagBits::BufferCoherent),
            tracked_buffer_data && !image_current, exportable};
}

constexpr bool ImageAliasOverlaps(const ImageAliasFootprint& image, VAddr address, u64 size) {
    if (size == 0 || image.size == 0) {
        return false;
    }
    return image.address <= address ? address - image.address < image.size
                                    : image.address - address < size;
}

constexpr bool CompetingImageAliases(const ImageAliasFootprint& image,
                                     const ImageAliasFootprint& other) {
    if (!ImageAliasOverlaps(image, other.address, other.size)) {
        return false;
    }
    if (other.buffer_authoritative) {
        return true;
    }
    return other.image_authoritative;
}

struct ImageAliasExport {
    size_t index;
    VAddr address;
    u64 size;
};

inline std::vector<ImageAliasExport> PlanImageAliasExports(
    std::span<const ImageAliasFootprint> images, VAddr address, u64 size) {
    std::vector<ImageAliasExport> exports;
    for (size_t i = 0; i < images.size(); ++i) {
        const auto& image = images[i];
        if (!image.image_authoritative || image.buffer_authoritative || !image.exportable ||
            !ImageAliasOverlaps(image, address, size)) {
            continue;
        }
        bool competing = false;
        for (size_t j = 0; j < images.size(); ++j) {
            competing |= i != j && CompetingImageAliases(image, images[j]);
        }
        if (!competing) {
            // A partial buffer write still needs all untouched pixels preserved.
            exports.push_back({i, image.address, image.size});
        }
    }
    return exports;
}

constexpr std::optional<u64> ImageAliasExportOffset(const ImageAliasExport& image,
                                                    VAddr arena_address, u64 arena_size) {
    if (image.address < arena_address || image.size > arena_size ||
        image.address - arena_address > arena_size - image.size) {
        return std::nullopt;
    }
    return image.address - arena_address;
}

} // namespace VideoCore
