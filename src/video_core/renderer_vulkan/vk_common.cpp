// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_common.h"

// Implement vma functions
#define VMA_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include <vk_mem_alloc.h>
#pragma clang diagnostic pop

// Store the dispatch loader here
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <atomic>
namespace Vulkan {
static std::atomic<vk::DeviceSize> g_empty_vma_bytes_freed{};
vk::DeviceSize EmptyVmaBytesFreed() {
    return g_empty_vma_bytes_freed.load(std::memory_order_relaxed);
}
vk::DeviceSize TrimEmptyVmaBlocks(VmaAllocator allocator) {
    const VkPhysicalDeviceMemoryProperties* props = nullptr;
    vmaGetMemoryProperties(allocator, &props);
    vk::DeviceSize freed = 0;
    for (uint32_t i = 0; i < props->memoryTypeCount; ++i) {
        if (!(props->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
        if (auto* blocks = allocator->m_pBlockVectors[i]) freed += blocks->TrimEmptyBlocks();
    }
    g_empty_vma_bytes_freed.fetch_add(freed, std::memory_order_relaxed);
    return freed;
}
}
