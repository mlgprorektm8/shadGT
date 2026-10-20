// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

#include "common/logging/log.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/buffer_cache/proto_import.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace VideoCore {

namespace {

using MapAliasFn = void* (*)(std::uintptr_t, std::size_t);
using UnmapAliasFn = void (*)(void*);

MapAliasFn MapAlias() {
#ifdef _WIN32
    static const auto fn = [] {
        HMODULE libc = GetModuleHandleA("libc.prx");
        return libc ? reinterpret_cast<MapAliasFn>(GetProcAddress(libc, "TXQU8CpoEtI")) : nullptr;
    }();
    return fn;
#else
    return nullptr;
#endif
}

UnmapAliasFn UnmapAlias() {
#ifdef _WIN32
    static const auto fn = [] {
        HMODULE libc = GetModuleHandleA("libc.prx");
        return libc ? reinterpret_cast<UnmapAliasFn>(GetProcAddress(libc, "674-w6lwWXI"))
                    : nullptr;
    }();
    return fn;
#else
    return nullptr;
#endif
}

struct PendingRelease {
    u64 tick;
    void* alias;
    VkDeviceMemory memory;
    std::shared_ptr<Buffer> wrapper; // owns the VkBuffer (D6)
};
std::mutex g_pending_mutex;
std::vector<PendingRelease> g_pending;
std::atomic<bool> g_has_pending{false}; // checked on every bind without the lock
// D7: views a thread found lately, valid while no view was unmapped since (lock-free check).
std::atomic<u64> g_unmap_generation{1};
struct RecentView {
    u64 generation;
    VAddr begin;
    VAddr end;
};
thread_local std::array<RecentView, 8> t_recent_views{};
thread_local u32 t_recent_next = 0;

} // namespace

DirectMemory::DirectMemory(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    l1 = std::make_unique<Buffer>(instance, 0, L1Entries * sizeof(u64), MemoryType::Stream,
                                  "D1 page table L1");
    l2_pool = std::make_unique<Buffer>(instance, 0, MaxL2Tables * L2Entries * sizeof(u64),
                                       MemoryType::Stream, "D1 page table L2");
    zero_page = std::make_unique<Buffer>(instance, 0, 2 * PageSize, MemoryType::Stream,
                                         "D1 zero page");
    std::memset(zero_page->mapped_data.data(), 0, zero_page->mapped_data.size());
    std::memset(l1->mapped_data.data(), 0, l1->mapped_data.size());
    l2_index.assign(L1Entries, 0);
    volatile_bits = std::make_unique<std::atomic<u64>[]>(VolatileWords);
    LOG_WARNING(Render_Vulkan, "D1: direct memory on (page table L1 at {:#x})",
                l1->BufferDeviceAddress());
}

DirectMemory::~DirectMemory() = default;

bool DirectMemory::Enabled() {
    // D11: the direct renderer is the renderer (GPU.direct_renderer from older launchers is
    // ignored). SHADGT_DIRECT=0 still turns it off, for comparisons.
    static const bool enabled = [] {
        if (const char* env = std::getenv("SHADGT_DIRECT"); env && *env) {
            return env[0] == '1';
        }
        return true;
    }();
    return enabled;
}

void DirectMemory::MarkVolatile(VAddr address, u64 size) {
    const u64 first = address >> PageBits;
    const u64 last = (address + std::max<u64>(size, 1) - 1) >> PageBits;
    for (u64 page = first; page <= last && (page >> 6) < VolatileWords; ++page) {
        const u64 bit = u64{1} << (page & 63);
        if (!(volatile_bits[page >> 6].fetch_or(bit, std::memory_order_relaxed) & bit)) {
            if (const u64 n = ++volatile_pages; n <= 20 || n % 100 == 0) {
                LOG_WARNING(Render_Vulkan, "D4: page {:#x} is volatile ({} pages)",
                            page << PageBits, n);
            }
        }
    }
}

bool DirectMemory::IsVolatile(VAddr address, u64 size) const {
    const u64 first = address >> PageBits;
    const u64 last = (address + std::max<u64>(size, 1) - 1) >> PageBits;
    if ((last >> 6) >= VolatileWords) {
        return true;
    }
    // A word (64 pages, 1 MiB) at a time.
    for (u64 word = first >> 6; word <= (last >> 6); ++word) {
        u64 mask = ~u64{0};
        if (word == (first >> 6)) {
            mask &= ~u64{0} << (first & 63);
        }
        if (word == (last >> 6)) {
            mask &= ~u64{0} >> (63 - (last & 63));
        }
        if (volatile_bits[word].load(std::memory_order_relaxed) & mask) {
            return true;
        }
    }
    return false;
}

vk::DeviceAddress DirectMemory::ZeroPageAddress() const {
    return zero_page->BufferDeviceAddress();
}

vk::DeviceAddress DirectMemory::L1Address() const {
    return l1->BufferDeviceAddress();
}

u64* DirectMemory::L2For(VAddr address, bool create) {
    const u64 index = address >> (PageBits + L2Bits);
    if (index >= L1Entries) {
        return nullptr;
    }
    if (l2_index[index] == 0) {
        if (!create || l2_used >= MaxL2Tables) {
            return nullptr;
        }
        const u32 table = l2_used++;
        u64* entries = reinterpret_cast<u64*>(l2_pool->mapped_data.data()) + table * L2Entries;
        std::memset(entries, 0, L2Entries * sizeof(u64));
        const u64 table_address = l2_pool->BufferDeviceAddress() + table * L2Entries * sizeof(u64);
        reinterpret_cast<u64*>(l1->mapped_data.data())[index] = table_address;
        l2_index[index] = table + 1;
    }
    return reinterpret_cast<u64*>(l2_pool->mapped_data.data()) +
           (l2_index[index] - 1) * L2Entries;
}

void DirectMemory::MapPages(VAddr begin, VAddr end, vk::DeviceAddress device_address) {
    for (VAddr page = begin; page < end; page += PageSize) {
        if (u64* l2 = L2For(page, true)) {
            l2[(page >> PageBits) & (L2Entries - 1)] = device_address + (page - begin);
        }
    }
}

void DirectMemory::UnmapPages(VAddr begin, VAddr end) {
    for (VAddr page = begin; page < end; page += PageSize) {
        if (u64* l2 = L2For(page, false)) {
            l2[(page >> PageBits) & (L2Entries - 1)] = 0;
        }
    }
}

bool DirectMemory::Covered(VAddr address, u64 size) const {
    std::scoped_lock lk{mutex};
    const VAddr first = address & ~(PageSize - 1);
    for (VAddr page = first; page < address + size; page += PageSize) {
        const u64 index = page >> (PageBits + L2Bits);
        if (index >= L1Entries || l2_index[index] == 0) {
            return false;
        }
        const u64* l2 = reinterpret_cast<const u64*>(l2_pool->mapped_data.data()) +
                        (l2_index[index] - 1) * L2Entries;
        if (l2[(page >> PageBits) & (L2Entries - 1)] == 0) {
            return false;
        }
    }
    return true;
}

std::optional<std::pair<const Buffer*, u64>> DirectMemory::FindBuffer(VAddr address,
                                                                     u64 size) const {
    std::scoped_lock lk{mutex};
    auto it = views.upper_bound(address);
    if (it == views.begin()) {
        return std::nullopt;
    }
    const View& view = std::prev(it)->second;
    if (address < view.begin || address + size > view.end || !view.wrapper) {
        return std::nullopt;
    }
    return std::make_pair(static_cast<const Buffer*>(view.wrapper.get()), address - view.begin);
}

bool DirectMemory::ImportView(VAddr address) {
#ifdef _WIN32
    const VAddr base = GuestViewBase(address);
    if (base == 0) {
        return false;
    }
    if (std::ranges::binary_search(failed, base)) {
        return false;
    }
    const VAddr end = GuestViewEnd(base, 1ULL << 32);
    const auto fail = [&](const char* why) {
        failed.insert(std::ranges::upper_bound(failed, base), base);
        if (++failed_count <= 40) {
            LOG_WARNING(Render_Vulkan, "D1: view {:#x}+{:#x} not imported: {}", base, end - base,
                        why);
        }
        return false;
    };
    if (end <= base || (base & (PageSize - 1)) != 0 || ((end - base) & (PageSize - 1)) != 0) {
        return fail("not whole pages");
    }
    const auto map_alias = MapAlias();
    if (!map_alias) {
        return fail("no GuestArenaMapAlias");
    }
    void* alias = map_alias(base, end - base);
    if (!alias) {
        return fail("no alias (not shared memory)");
    }
    const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
    const VkDevice device = instance.GetDevice();
    VkMemoryHostPointerPropertiesEXT ptr_props{
        VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (d.vkGetMemoryHostPointerPropertiesEXT(device,
                                              VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                              alias, &ptr_props) != VK_SUCCESS ||
        ptr_props.memoryTypeBits == 0) {
        UnmapAlias()(alias);
        return fail("no memory type for the alias");
    }
    VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr,
                                         VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            &flags_info,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            alias};
    VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import, end - base,
                              static_cast<u32>(std::countr_zero(ptr_props.memoryTypeBits))};
    VkDeviceMemory memory{};
    if (const VkResult r = d.vkAllocateMemory(device, &info, nullptr, &memory); r != VK_SUCCESS) {
        UnmapAlias()(alias);
        return fail("import refused");
    }
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                              nullptr,
                                              VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external};
    buffer_info.size = end - base;
    buffer_info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT; // NAT-100: dispatch arguments
    VkBuffer buffer{};
    if (d.vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS ||
        d.vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        d.vkFreeMemory(device, memory, nullptr);
        UnmapAlias()(alias);
        return fail("buffer over the import failed");
    }
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr,
                                           buffer};
    const vk::DeviceAddress device_address = d.vkGetBufferDeviceAddress(device, &address_info);
    MapPages(base, end, device_address);
    auto wrapper = std::make_shared<Buffer>(instance, base, end - base, vk::Buffer{buffer});
    views.emplace(base, View{base, end, alias, memory, buffer, device_address, std::move(wrapper)});
    imported_bytes += end - base;
    if (views.size() <= 20 || views.size() % 100 == 0) {
        LOG_WARNING(Render_Vulkan, "D1: imported view {:#x}+{:#x} ({} views, {} MiB)", base,
                    end - base, views.size(), imported_bytes.load() >> 20);
    }
    return true;
#else
    return false;
#endif
}

bool DirectMemory::ForEachPart(VAddr address, u64 size,
                               const std::function<void(const Buffer&, u64, u64, u64)>& fn) const {
    std::scoped_lock lk{mutex};
    const VAddr end = address + size;
    for (VAddr cursor = address; cursor < end;) {
        auto it = views.upper_bound(cursor);
        if (it == views.begin()) {
            return false;
        }
        const View& view = std::prev(it)->second;
        if (cursor < view.begin || cursor >= view.end || !view.wrapper) {
            return false;
        }
        const u64 bytes = std::min(end, view.end) - cursor;
        fn(*view.wrapper, cursor - view.begin, cursor - address, bytes);
        cursor += bytes;
    }
    return true;
}

bool DirectMemory::EnsureImported(VAddr address, u64 size) {
    if (g_has_pending.load(std::memory_order_relaxed)) {
        // Aliases of unmapped views the GPU no longer uses.
        std::scoped_lock lk{g_pending_mutex};
        const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
        std::erase_if(g_pending, [&](PendingRelease& p) {
            if (!scheduler.IsFree(p.tick)) {
                return false;
            }
            p.wrapper.reset();
            d.vkFreeMemory(instance.GetDevice(), p.memory, nullptr);
            UnmapAlias()(p.alias);
            return true;
        });
        g_has_pending.store(!g_pending.empty(), std::memory_order_relaxed);
    }
    const VAddr end = address + size;
    const u64 generation = g_unmap_generation.load(std::memory_order_acquire);
    for (const RecentView& recent : t_recent_views) {
        if (recent.generation == generation && address >= recent.begin && end <= recent.end) {
            return true;
        }
    }
    std::scoped_lock lk{mutex};
    for (VAddr page = address & ~(PageSize - 1); page < end;) {
        auto it = views.upper_bound(page);
        if (it != views.begin()) {
            const auto& view = std::prev(it)->second;
            if (page >= view.begin && page < view.end) {
                if (address >= view.begin && end <= view.end) {
                    t_recent_views[t_recent_next++ % t_recent_views.size()] = {
                        generation, view.begin, view.end};
                }
                page = view.end;
                continue;
            }
        }
        if (!ImportView(page)) {
            return false;
        }
    }
    return true;
}

void DirectMemory::OnUnmap(VAddr address, u64 size) {
    std::scoped_lock lk{mutex};
    g_unmap_generation.fetch_add(1, std::memory_order_acq_rel); // D7
    const VAddr end = address + size;
    auto it = views.upper_bound(address);
    if (it != views.begin()) {
        --it;
    }
    const u64 tick = scheduler.CurrentTick();
    while (it != views.end() && it->first < end) {
        View& view = it->second;
        if (view.end <= address) {
            ++it;
            continue;
        }
        UnmapPages(view.begin, view.end);
        imported_bytes -= view.end - view.begin;
        // DIAG-132: views dropped while GPU work may still read them.
        static std::atomic<u32> dropped{};
        if (const u32 n = ++dropped; n <= 40 || n % 100 == 0) {
            LOG_WARNING(Render_Vulkan,
                        "DIAG-132: view {:#x}+{:#x} dropped (unmap {:#x}+{:#x}, tick {}, GPU at "
                        "{}, {} dropped)",
                        view.begin, view.end - view.begin, address, size, tick,
                        scheduler.CurrentTick(), n);
        }
        {
            std::scoped_lock pending_lk{g_pending_mutex};
            g_pending.push_back({tick, view.alias, view.memory, std::move(view.wrapper)});
            g_has_pending.store(true, std::memory_order_relaxed);
        }
        it = views.erase(it);
    }
    // A view base unmapped may be mapped again with other memory.
    std::erase_if(failed, [&](VAddr base) { return base >= address && base < end; });
}

DirectMemory::Stats DirectMemory::GetStats() const {
    std::scoped_lock lk{mutex};
    return {views.size(), imported_bytes.load(), failed_count.load()};
}

void DirectMemory::Release(View& view) {
    const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
    view.wrapper.reset();
    d.vkFreeMemory(instance.GetDevice(), view.memory, nullptr);
    UnmapAlias()(view.alias);
}

} // namespace VideoCore
