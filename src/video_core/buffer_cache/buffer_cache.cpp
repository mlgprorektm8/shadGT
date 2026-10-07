// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image_aliasing.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>
#ifdef _WIN32
#include <windows.h>
#endif

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}


void BufferCache::RecordGdsReadback(VAddr address, u32 gds_offset, u32 size) {
    auto readback = std::make_shared<AsyncReadback>();
    readback->address = address;
    readback->size = size;
    readback->download = staging_pool.Request(size, MemoryType::HostCached, 4, true);
    const vk::BufferCopy copy = {
        .srcOffset = gds_offset,
        .dstOffset = readback->download.offset,
        .size = size,
    };
    runtime.CopyBuffer(&gds_buffer, readback->download.buffer, std::span{&copy, 1});
    std::scoped_lock lk{async_readbacks_mutex};
    pending_async_readbacks.push_back(std::move(readback));
    num_pending_async_readbacks = static_cast<u32>(pending_async_readbacks.size());
    num_tracked_async_readbacks =
        static_cast<u32>(pending_async_readbacks.size() + inflight_async_readbacks.size());
}

std::vector<std::shared_ptr<BufferCache::AsyncReadback>> BufferCache::TakePendingAsyncReadbacks() {
    std::scoped_lock lk{async_readbacks_mutex};
    auto readbacks = std::move(pending_async_readbacks);
    pending_async_readbacks.clear();
    inflight_async_readbacks.insert(inflight_async_readbacks.end(), readbacks.begin(),
                                  readbacks.end());
    num_pending_async_readbacks = 0;
    return readbacks;
}

void BufferCache::NoteCpuReadFault(VAddr address, u64 size) {
    // PERF-010: a CPU access that needed GPU-written data drains the GPU. Pages that saw such an
    // access are read back after each later GPU write, so the next access finds current data.
    ++hot_page_stats.faults;
    constexpr u64 PageSize = 4_KB;
    const VAddr end = address + std::max<u64>(size, 1);
    for (VAddr page = Common::AlignDown(address, PageSize); page < end; page += PageSize) {
        if (!hot_pages.insert(page).second) {
            continue;
        }
        ++hot_page_stats.new_pages;
        hot_page_order.push_back(page);
        if (hot_page_order.size() > MaxHotPages) {
            hot_pages.erase(hot_page_order.front());
            hot_page_order.pop_front();
        }
    }
}

bool BufferCache::RecordHotPageReadbacks() {
    LogHotPageStats();
    if (hot_page_order.empty()) {
        return false;
    }
    constexpr u64 PageSize = 4_KB;
    std::unordered_map<VAddr, bool> inflight_pages;
    {
        std::scoped_lock lk{async_readbacks_mutex};
        for (const auto* list : {&pending_async_readbacks, &inflight_async_readbacks}) {
            for (const auto& readback : *list) {
                if (readback->valid) {
                    inflight_pages[Common::AlignDown(readback->address, PageSize)] = true;
                }
            }
        }
    }
    std::vector<std::shared_ptr<AsyncReadback>> recorded;
    for (const VAddr page : hot_page_order) {
        if (inflight_pages.contains(page) || !memory_tracker->IsRegionGpuModified(page, PageSize)) {
            continue;
        }
        const u64 block = page >> block_shift;
        const auto* arena = GetArena(block, block);
        // Same ranges as DownloadMemory: only bytes the GPU wrote, so CPU writes not yet uploaded
        // to the arena are never overwritten.
        memory_tracker->ForEachDownloadRange<false>(page, PageSize, [&](u64 address, u64 size) {
            gpu_modified_ranges.ForEachInRange(address, size, [&](VAddr start, VAddr end) {
                auto readback = std::make_shared<AsyncReadback>();
                readback->address = start;
                readback->size = static_cast<u32>(end - start);
                readback->download = staging_pool.Request(readback->size, MemoryType::HostCached);
                const vk::BufferCopy copy = {
                    .srcOffset = start - arena->cpu_addr,
                    .dstOffset = readback->download.offset,
                    .size = readback->size,
                };
                runtime.CopyBuffer(arena, readback->download.buffer, std::span{&copy, 1});
                hot_page_stats.recorded_bytes += readback->size;
                recorded.push_back(std::move(readback));
            });
        });
    }
    if (recorded.empty()) {
        return false;
    }
    hot_page_stats.recorded += static_cast<u32>(recorded.size());
    std::scoped_lock lk{async_readbacks_mutex};
    pending_async_readbacks.insert(pending_async_readbacks.end(), recorded.begin(),
                                   recorded.end());
    num_pending_async_readbacks = static_cast<u32>(pending_async_readbacks.size());
    num_tracked_async_readbacks =
        static_cast<u32>(pending_async_readbacks.size() + inflight_async_readbacks.size());
    return true;
}

void BufferCache::LogHotPageStats() {
    // PERF-DIAG-006: CPU faults on GPU-written data and the readbacks recorded for them.
    auto& stats = hot_page_stats;
    const auto now = std::chrono::steady_clock::now();
    if (stats.window_start == std::chrono::steady_clock::time_point{}) {
        stats.window_start = now;
        return;
    }
    if (now - stats.window_start < std::chrono::seconds{2}) {
        return;
    }
    LOG_WARNING(Render_Vulkan,
                "Readback pages in 2.0 s: {} CPU faults, {} new pages ({} tracked), {} readbacks "
                "({} KB), {} completed, {} superseded",
                stats.faults, stats.new_pages, hot_page_order.size(), stats.recorded,
                stats.recorded_bytes / 1024, stats.completed.exchange(0),
                stats.invalidated.exchange(0));
    stats.window_start = now;
    stats.faults = 0;
    stats.new_pages = 0;
    stats.recorded = 0;
    stats.recorded_bytes = 0;
}

void BufferCache::CompleteAsyncReadbacks(std::span<const std::shared_ptr<AsyncReadback>> readbacks) {
    std::scoped_lock lk{async_readbacks_mutex};
    for (const auto& readback : readbacks) {
        if (readback->valid && memory->IsValidMapping(readback->address, readback->size)) {
            readback->download.Invalidate();
            memory->TryWriteBacking(std::bit_cast<void*>(readback->address),
                                    readback->download.mapped, readback->size);
            // The guest copy is now current, so CPU accesses need no GPU drain. The arena keeps
            // the same bytes from the GPU-side copy.
            memory_tracker->UnmarkRegionAsGpuModified(readback->address, readback->size, false);
            ++hot_page_stats.completed;
        } else {
            ++hot_page_stats.invalidated;
        }
        std::erase(inflight_async_readbacks, readback);
        finished_async_downloads.push_back(readback->download);
    }
    num_tracked_async_readbacks =
        static_cast<u32>(pending_async_readbacks.size() + inflight_async_readbacks.size());
}

void BufferCache::ReleaseFinishedAsyncReadbacks() {
    std::vector<Vulkan::StagingBufferRef> finished;
    {
        std::scoped_lock lk{async_readbacks_mutex};
        finished.swap(finished_async_downloads);
    }
    for (const auto& download : finished) {
        staging_pool.FreeDeferred(download);
    }
}

void BufferCache::InvalidateAsyncReadbacks(VAddr address, u64 size) {
    // A later GPU write to the destination makes the read-back value stale; keep the
    // GPU-modified state so the exact path downloads the newer data.
    if (num_tracked_async_readbacks.load() == 0) {
        return;
    }
    // Completion clears GPU-modified state per 4 KB tracker page, so any readback sharing a page
    // with the write is stale too.
    constexpr u64 PageSize = 4_KB;
    const VAddr start = Common::AlignDown(address, PageSize);
    const VAddr end = Common::AlignUp(address + size, PageSize);
    std::scoped_lock lk{async_readbacks_mutex};
    const auto invalidate = [&](const std::shared_ptr<AsyncReadback>& readback) {
        if (readback->address < end && start < readback->address + readback->size) {
            readback->valid = false;
        }
    };
    std::ranges::for_each(pending_async_readbacks, invalidate);
    std::ranges::for_each(inflight_async_readbacks, invalidate);
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    const auto flush_request = [this, device_addr, size, is_write] {
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        DownloadMemory(arena, window_start, window_end - window_start);
        NoteCpuReadFault(device_addr, size);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
    });
    if (total_size_bytes == 0) {
        return;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    scheduler.Finish();
    // This download is current; a readback recorded earlier must not complete over it after the
    // CPU has written the range.
    InvalidateAsyncReadbacks(device_addr, size);

    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& copy : copies) {
        auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
        memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                copy.size);
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    if (is_written) {
        InvalidateAsyncReadbacks(device_addr, size);
    }
    SynchronizeMemoryFromImage(device_addr, size);
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size) &&
        !HasGpuImageAlias(device_addr, size)) {
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

bool BufferCache::HasGpuImageAlias(VAddr addr, size_t size) {
    if (size == 0) {
        return false;
    }
    bool found = false;
    texture_cache.ForEachImageInRegion(addr, size, [&](ImageId, Image& image) {
        found |= image.SafeToDownload();
        return found;
    });
    return found;
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }

    const vk::MemoryAllocateInfo alloc_info = {
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
        copies.emplace_back(total_size_bytes, addr, size);
        total_size_bytes += size;
    });
    if (!copies.empty()) {
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMetadata(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMetadata(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    return false;
}

void BufferCache::SynchronizeMemoryFromImage(VAddr device_addr, u32 size) {
    if (size == 0) {
        return;
    }
    std::vector<ImageId> image_ids;
    const auto collect = [&](ImageId id, Image&) {
        if (std::ranges::find(image_ids, id) == image_ids.end()) {
            image_ids.push_back(id);
        }
    };
    texture_cache.ForEachImageInRegion(device_addr, size, collect);
    const size_t requested_images = image_ids.size();
    // Full exports must also consider aliases outside the requested buffer slice.
    for (size_t i = 0; i < requested_images; ++i) {
        const auto& image = texture_cache.GetImage(image_ids[i]);
        if (image.SafeToDownload() && False(image.flags & ImageFlagBits::BufferCoherent)) {
            texture_cache.ForEachImageInRegion(image.info.guest_address, image.info.guest_size,
                                               collect);
        }
    }
    std::vector<ImageAliasFootprint> footprints;
    footprints.reserve(image_ids.size());
    for (const auto id : image_ids) {
        const auto& image = texture_cache.GetImage(id);
        const auto& info = image.info;
        const bool exportable = info.num_samples == 1 && image.backing->num_samples == 1 &&
                                !info.props.has_stencil && info.guest_size <= UINT32_MAX &&
                                info.pixel_format == image.backing->image.image_ci.format;
        footprints.push_back(DescribeImageAlias(
            info.guest_address, info.guest_size, image.flags,
            gpu_modified_ranges.Intersects(info.guest_address, info.guest_size), exportable));
    }
    for (const auto& export_image : PlanImageAliasExports(footprints, device_addr, size)) {
        Image& image = texture_cache.GetImage(image_ids[export_image.index]);
        const u64 first_block = export_image.address >> block_shift;
        const u64 last_block = (export_image.address + export_image.size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        EnsureResident(arena, first_block, last_block);
        const auto arena_offset =
            ImageAliasExportOffset(export_image, arena->cpu_addr, arena->size_bytes);
        ASSERT(arena_offset.has_value());
        SynchronizeMemory(arena, export_image.address, export_image.size, false, false);

        boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
        for (u32 mip = 0; mip < image.info.resources.levels; ++mip) {
            const auto& mip_info = image.info.mips_layout[mip];
            buffer_copies.push_back(vk::BufferImageCopy{
                .bufferOffset = mip_info.offset,
                .bufferRowLength = mip_info.pitch,
                .bufferImageHeight = mip_info.height,
                .imageSubresource{
                    .aspectMask = image.aspect_mask,
                    .mipLevel = mip,
                    .baseArrayLayer = 0,
                    .layerCount = image.info.resources.layers,
                },
                .imageOffset = {0, 0, 0},
                .imageExtent = {std::max(image.info.size.width >> mip, 1u),
                                std::max(image.info.size.height >> mip, 1u),
                                std::max(image.info.size.depth >> mip, 1u)},
            });
        }
        texture_cache.GetTileManager().TileImage(image, buffer_copies, arena, *arena_offset, true);
        InvalidateAsyncReadbacks(export_image.address, export_image.size);
        memory_tracker->MarkRegionAsGpuModified(export_image.address, export_image.size);
        gpu_modified_ranges.Add(export_image.address, export_image.size);
        image.flags |= ImageFlagBits::BufferCoherent;
        if (image_alias_exports_logged < 16) {
            ++image_alias_exports_logged;
            LOG_INFO(
                Render,
                "Preserved image alias base={:#x}, size={:#x}, extent={}x{}, request={:#x}+{:#x}",
                export_image.address, export_image.size, image.info.size.width,
                image.info.size.height, device_addr, size);
        }
    }
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
