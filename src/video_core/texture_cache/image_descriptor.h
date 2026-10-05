// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <limits>
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

enum class ImageDescriptorGeometryError {
    None,
    InvalidType,
    UnrepresentableLayers,
    DeviceLayerLimit,
    DeviceExtentLimit,
    MipLevels,
    ViewMipLevels,
    SampleCount,
};

inline ImageDescriptorGeometryError CheckImageDescriptorGeometry(
    const AmdGpu::Image& image, const vk::PhysicalDeviceLimits& limits) {
    if (!image.Valid()) {
        return ImageDescriptorGeometryError::InvalidType;
    }
    const u32 layers = image.NumLayers();
    if (layers == 0 || layers > std::numeric_limits<u16>::max()) {
        // Check before ImageInfo narrows the count to SubresourceExtent::layers.
        return ImageDescriptorGeometryError::UnrepresentableLayers;
    }
    if (layers > limits.maxImageArrayLayers) {
        return ImageDescriptorGeometryError::DeviceLayerLimit;
    }
    const u32 width = image.width + 1;
    const u32 height = image.height + 1;
    const bool volume = image.GetBaseType() == AmdGpu::ImageType::Color3D;
    const u32 depth = volume ? image.depth + 1 : 1;
    if (volume) {
        if (std::max({width, height, depth}) > limits.maxImageDimension3D) {
            return ImageDescriptorGeometryError::DeviceExtentLimit;
        }
    } else {
        // The renderer represents guest 1D images using a Vulkan 2D backing too.
        if (std::max(width, height) > limits.maxImageDimension2D ||
            (image.GetBaseType() == AmdGpu::ImageType::Color1D &&
             width > limits.maxImageDimension1D)) {
            return ImageDescriptorGeometryError::DeviceExtentLimit;
        }
    }
    if (image.NumLevels() > std::bit_width(std::max({width, height, depth}))) {
        return ImageDescriptorGeometryError::MipLevels;
    }
    const bool multisampled = image.GetType() == AmdGpu::ImageType::Color2DMsaa ||
                              image.GetType() == AmdGpu::ImageType::Color2DMsaaArray;
    if (!multisampled && image.base_level > image.last_level) {
        return ImageDescriptorGeometryError::ViewMipLevels;
    }
    // RawNumSamples accepts the guest's 1/2/4/8/16 sample counts and deliberately
    // lowers them to the physical format's supported count later.
    if (image.NumSamples() > 16) {
        return ImageDescriptorGeometryError::SampleCount;
    }
    return ImageDescriptorGeometryError::None;
}

} // namespace VideoCore
