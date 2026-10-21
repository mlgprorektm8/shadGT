// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vulkan/vulkan.hpp>

namespace Vulkan {

/// FIX-050: the Vulkan format of a vertex attribute. A fetch-shader attribute whose V# is empty
/// (address 0, invalid data format) has no format of its own; Vulkan does not allow
/// VK_FORMAT_UNDEFINED for an attribute, and Gran Turismo 7 lost the device on the draws after
/// one. Such an attribute reads the null vertex buffer (nullDescriptor), so it gets zeros in a
/// valid four-component float format instead.
constexpr vk::Format VertexAttributeFormat(vk::Format guest_format) {
    return guest_format == vk::Format::eUndefined ? vk::Format::eR32G32B32A32Sfloat
                                                  : guest_format;
}

} // namespace Vulkan
