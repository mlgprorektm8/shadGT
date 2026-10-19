// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <string>

#include "ryty_gpu/native/rhi/rhi.h"

struct VkImageCreateInfo; // vulkan_core.h

namespace Vulkan {
class Instance;
}

namespace Native::Rhi {

/// NAT-011: the Vulkan backend, on the renderer's existing device (shared with the presenter, so
/// native images can be shown without copies between devices). Queue submissions take the
/// scheduler's submit mutex.
std::unique_ptr<Device> CreateVulkanDevice(const Vulkan::Instance& instance);

/// NAT-100: an image made from a full create info (flags, usage and format chosen by the
/// caller), on a device from CreateVulkanDevice; it starts in the general layout like the rest.
std::unique_ptr<Image> CreateVulkanImage(Device& device, const VkImageCreateInfo& image_ci,
                                         const ImageDesc& desc);

/// A device of its own (headless instance, first Vulkan 1.3 GPU), for tests and tools.
std::unique_ptr<Device> CreateStandaloneVulkanDevice(std::string* error = nullptr);

} // namespace Native::Rhi
