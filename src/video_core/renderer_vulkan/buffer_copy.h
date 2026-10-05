// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <span>
#include <vulkan/vulkan.hpp>
#include "common/small_vector.h"

namespace Vulkan {

// Vulkan does not accept empty copy regions. Keep the original span when every region has work,
// and use the caller's storage only when empty regions need to be removed.
inline std::span<const vk::BufferCopy> NonEmptyBufferCopies(
    std::span<const vk::BufferCopy> copies, SmallVector<vk::BufferCopy, 8>& storage) {
    if (std::ranges::find(copies, vk::DeviceSize{}, &vk::BufferCopy::size) == copies.end()) {
        return copies;
    }
    storage.clear();
    storage.reserve(copies.size());
    for (const auto& copy : copies) {
        if (copy.size != 0) {
            storage.push_back(copy);
        }
    }
    return {storage.data(), storage.size()};
}

} // namespace Vulkan
