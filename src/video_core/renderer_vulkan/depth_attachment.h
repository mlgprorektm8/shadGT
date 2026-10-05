// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cmath>
#include <limits>
#include <vulkan/vulkan.hpp>
#include "video_core/amdgpu/regs_depth.h"

namespace Vulkan {

inline float ReadOnlyFarDepthMax(float minimum, float maximum,
                                 const AmdGpu::DepthControl& control) {
    // Precision workaround for background passes whose narrow viewport rounds an interior
    // depth up to the cleared 1.0 value. Ordinary and depth-writing passes stay unchanged.
    constexpr float far_range_min = 1.f - 4.f * std::numeric_limits<float>::epsilon();
    if (control.depth_enable && !control.depth_write_enable && !control.stencil_enable &&
        !control.depth_bounds_enable && control.depth_func == AmdGpu::CompareFunc::Less &&
        minimum >= far_range_min && minimum < 1.f && maximum == 1.f) {
        return std::nextafter(1.f, 0.f);
    }
    return maximum;
}

inline vk::ImageAspectFlags DepthStencilClearAspects(
    const AmdGpu::DepthBuffer& buffer, const AmdGpu::DepthControl& control,
    const AmdGpu::DepthRenderControl& render_control, bool metadata_clear) {
    vk::ImageAspectFlags aspects{};
    if (buffer.DepthValid() &&
        (metadata_clear || (render_control.depth_clear_enable && control.depth_enable &&
                            control.depth_write_enable))) {
        aspects |= vk::ImageAspectFlagBits::eDepth;
    }
    // A full HTile fast clear also resets stencil, even on a depth-only prepass.
    if (buffer.StencilValid() && (metadata_clear || render_control.stencil_clear_enable)) {
        aspects |= vk::ImageAspectFlagBits::eStencil;
    }
    return aspects;
}

inline vk::ImageLayout DepthAttachmentLayout(bool has_stencil, bool depth_write, bool stencil_write,
                                             bool sampled, bool feedback_layout_supported) {
    if (sampled && (depth_write || stencil_write)) {
        return feedback_layout_supported ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                         : vk::ImageLayout::eGeneral;
    }
    if (depth_write) {
        return has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                           : vk::ImageLayout::eDepthAttachmentOptimal;
    }
    if (stencil_write) {
        return vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal;
    }
    return has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                       : vk::ImageLayout::eDepthReadOnlyOptimal;
}

inline bool DepthCopyNeedsBuffer(vk::Format src, vk::Format dst, bool src_depth, bool dst_depth,
                                 bool maintenance8) {
    // Maintenance8 permits depth/color aspect copies, but different depth/stencil
    // formats still cannot be copied directly, even when only depth is requested.
    return src_depth && dst_depth ? src != dst : src_depth != dst_depth && !maintenance8;
}

} // namespace Vulkan
