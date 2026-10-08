// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "common/adaptive_mutex.h"
#include "common/types.h"
#include "core/emulator_settings.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

namespace VideoCore {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = std::mutex;
#endif

/**
 * Allows tracking CPU and GPU modification of pages in a contigious virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_},
          readbacks_mode{EmulatorSettings.GetReadbacksMode()} {
        cpu.Fill(~0ULL);
        gpu.Fill(0ULL);
    }
    explicit RegionManager() = default;

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
        hot_shadows.clear();
    }

    static constexpr Bounds GetBounds(u64 offset, u64 size) {
        const u64 end_address = offset + size - 1;
        return Bounds{
            .start_word = offset / BYTES_PER_WORD,
            .start_page = (offset) / BYTES_PER_PAGE,
            .end_word = end_address / BYTES_PER_WORD,
            .end_page = (end_address) / BYTES_PER_PAGE,
        };
    }

    static constexpr std::pair<u64, u64> GetMasks(u64 start_page, u64 end_page) {
        const u64 start_mask = ~u64{0} << (start_page & (PAGES_PER_WORD - 1));
        const u64 end_mask = ~u64{0} >> (63 - (end_page & (PAGES_PER_WORD - 1)));
        return std::make_pair(start_mask, end_mask);
    }

    static constexpr void IterateWords(Bounds bounds, auto&& func) {
        const auto [start_word, start_page, end_word, end_page] = bounds;
        const auto [start_mask, end_mask] = GetMasks(start_page, end_page);
        if (start_word == end_word) [[likely]] {
            func(start_word, start_mask & end_mask);
        } else {
            func(start_word, start_mask);
            for (s64 i = start_word + 1; i < end_word; ++i) {
                func(i, ~0ULL);
            }
            func(end_word, end_mask);
        }
    }

    static constexpr void IteratePages(u64 word, auto&& func) {
        u64 offset{};
        while (word != 0) {
            const u64 empty_bits = std::countr_zero(word);
            offset += empty_bits;
            word >>= empty_bits;
            const u64 set_bits = std::countr_one(word);
            func(offset, set_bits);
            word = set_bits < PAGES_PER_WORD ? (word >> set_bits) : 0;
            offset += set_bits;
        }
    }

    template <StateOp cpu_op, StateOp gpu_op, bool locked = true>
    void ChangeRegionState(u64 offset, u64 size) {
        RegionBits write_prot;
        RegionBits read_prot;
        auto bounds = GetBounds(offset, size);
        Bounds watcher_bounds;
        if constexpr (locked) {
            mutex.lock();
        }
        IterateWords(bounds, [&](u64 index, u64 mask) {
            UpdateStateAndProtection<cpu_op, gpu_op>(write_prot, read_prot, index, mask);
        });
        const auto write_op = GetPageOp<Type::CPU>(cpu_op);
        const auto read_op = GetPageOp<Type::GPU>(gpu_op);
        const bool update_watchers = write_op != PageOp::None || read_op != PageOp::None;
        if (update_watchers &&
            GetWatcherBounds<cpu_op, gpu_op>(bounds, write_prot, read_prot, watcher_bounds)) {
            tracker->UpdatePageWatchersForRegion(cpu_addr, watcher_bounds, write_prot, read_prot,
                                                 write_op, read_op);
        }
        if constexpr (locked) {
            mutex.unlock();
        }
    }

    template <Type type, StateOp cpu_op, StateOp gpu_op, bool locked = true>
    void ForEachModifiedRange(u64 offset, s64 size, auto&& func) {
        auto& state = GetRegionBits<type>();
        RegionBits write_prot;
        RegionBits read_prot;
        u64 start_page{};
        u64 end_page{};
        auto bounds = GetBounds(offset, size);
        Bounds watcher_bounds;
        if constexpr (locked) {
            mutex.lock();
        }
        if constexpr (type == Type::CPU && cpu_op == StateOp::Clear) {
            RefreshUploadEpoch();
        }
        IterateWords(bounds, [&](u64 index, u64 mask) {
            const u64 base_page = index * PAGES_PER_WORD;
            u64 word = state[index] & mask;
            if constexpr (type == Type::CPU && cpu_op == StateOp::Clear) {
                // PERF-012: a hot page uploaded earlier in this epoch is still current.
                word &= ~(hot[index] & uploaded[index]);
                const u64 hot_uploads = hot[index] & word;
                uploaded[index] |= hot_uploads;
                if (hot_shadows_enabled && hot_uploads != 0) {
                    word &= ~UnchangedHotPages(base_page, hot_uploads);
                }
            }
            UpdateStateAndProtection<cpu_op, gpu_op>(write_prot, read_prot, index, mask);
            IteratePages(word, [&](u64 pages_offset, u64 pages_size) {
                if (end_page == base_page + pages_offset) {
                    end_page += pages_size;
                    return;
                }
                if (end_page) {
                    func(cpu_addr + start_page * BYTES_PER_PAGE,
                         (end_page - start_page) * BYTES_PER_PAGE);
                }
                start_page = base_page + pages_offset;
                end_page = start_page + pages_size;
            });
        });
        if (end_page) {
            func(cpu_addr + start_page * BYTES_PER_PAGE, (end_page - start_page) * BYTES_PER_PAGE);
        }
        const auto write_op = GetPageOp<Type::CPU>(cpu_op);
        const auto read_op = GetPageOp<Type::GPU>(gpu_op);
        const bool update_watchers = write_op != PageOp::None || read_op != PageOp::None;
        if (update_watchers &&
            GetWatcherBounds<cpu_op, gpu_op>(bounds, write_prot, read_prot, watcher_bounds)) {
            tracker->UpdatePageWatchersForRegion(cpu_addr, watcher_bounds, write_prot, read_prot,
                                                 write_op, read_op);
        }
        if constexpr (locked) {
            mutex.unlock();
        }
    }

    template <Type type>
    bool IsRegionModified(u64 offset, u64 size) noexcept {
        auto& state = GetRegionBits<type>();
        const auto [start_word, start_page, end_word, end_page] = GetBounds(offset, size);
        const auto [start_mask, end_mask] = GetMasks(start_page, end_page);
        if (start_word == end_word) [[likely]] {
            return state[start_word] & (start_mask & end_mask);
        } else {
            if (state[start_word] & start_mask) {
                return true;
            }
            for (s64 i = start_word + 1; i < end_word; ++i) {
                if (state[i]) {
                    return true;
                }
            }
            return state[end_word] & end_mask;
        }
    }

    /// PERF-012: counts CPU write faults on pages that were just made writable (lock held);
    /// a page that keeps faulting becomes hot and stays unprotected.
    void NoteCpuWriteFault(u64 offset, u64 size) noexcept {
        const auto [start_word, start_page, end_word, end_page] = GetBounds(offset, size);
        for (u64 page = start_page; page <= end_page; ++page) {
            if (hot_pages_enabled && write_faults[page] < HotPageFaults &&
                ++write_faults[page] == HotPageFaults) {
                hot[page / PAGES_PER_WORD] |= 1ULL << (page % PAGES_PER_WORD);
            }
        }
    }

    /// PERF-012: hot pages in the range go back to normal tracking: their CPU-modified state is
    /// cleared and they are write-protected again. For GPU writes, which need CPU writes to the
    /// page to fault.
    void UntrackHotPages(u64 offset, u64 size) {
        RegionBits write_prot;
        RegionBits read_prot;
        auto bounds = GetBounds(offset, size);
        Bounds watcher_bounds;
        std::scoped_lock lk{mutex};
        bool any = false;
        IterateWords(bounds, [&](u64 index, u64 mask) {
            const u64 hot_bits = hot[index] & mask;
            if (hot_bits == 0) {
                return;
            }
            any = true;
            hot[index] &= ~hot_bits;
            DropHotShadows(index * PAGES_PER_WORD, hot_bits);
            const u64 prev = cpu[index];
            cpu[index] &= ~hot_bits;
            write_prot[index] = (cpu[index] ^ prev) & mask;
        });
        if (!any) {
            return;
        }
        for (u64 page = bounds.start_page; page <= bounds.end_page; ++page) {
            write_faults[page] = 0;
        }
        if (GetWatcherBounds<StateOp::Clear, StateOp::None>(bounds, write_prot, read_prot,
                                                            watcher_bounds)) {
            tracker->UpdatePageWatchersForRegion(cpu_addr, watcher_bounds, write_prot, read_prot,
                                                 PageOp::Track, PageOp::None);
        }
    }

    void Lock(const Bounds& bounds) noexcept {
        mutex.lock();
    }

    void Unlock(const Bounds& bounds) noexcept {
        mutex.unlock();
    }

private:
    template <StateOp cpu_op, StateOp gpu_op>
    void UpdateStateAndProtection(RegionBits& write_prot, RegionBits& read_prot, u64 index,
                                  u64 mask) {
        if constexpr (gpu_op == StateOp::Set) {
            // PERF-012: a GPU write needs CPU writes to the page to fault again.
            if (const u64 hot_bits = hot[index] & mask; hot_bits != 0) {
                hot[index] &= ~hot_bits;
                for (u64 bits = hot_bits; bits != 0; bits &= bits - 1) {
                    write_faults[index * PAGES_PER_WORD + std::countr_zero(bits)] = 0;
                }
                DropHotShadows(index * PAGES_PER_WORD, hot_bits);
            }
        }
        if constexpr (cpu_op != StateOp::None) {
            const u64 prev = cpu[index];
            if constexpr (cpu_op == StateOp::Clear) {
                // PERF-012: hot pages stay CPU-modified and unprotected.
                cpu[index] &= ~(mask & ~hot[index]);
            } else {
                cpu[index] |= mask;
            }
            write_prot[index] = (cpu[index] ^ prev) & mask;
        }
        if constexpr (gpu_op != StateOp::None) {
            const u64 prev = gpu[index];
            if constexpr (gpu_op == StateOp::Clear) {
                gpu[index] &= ~mask;
            } else {
                gpu[index] |= mask;
            }
            read_prot[index] = (gpu[index] ^ prev) & mask;
        }
    }

    template <StateOp cpu_op, StateOp gpu_op>
    static bool GetWatcherBounds(const Bounds& bounds, RegionBits& write_prot,
                                 RegionBits& read_prot, Bounds& watcher_bounds) {
        const auto prot = [&](u64 index) {
            u64 word{};
            if constexpr (cpu_op != StateOp::None) {
                word |= write_prot[index];
            }
            if constexpr (gpu_op != StateOp::None) {
                word |= read_prot[index];
            }
            return word;
        };
        u64 start_word = bounds.start_word;
        while (prot(start_word) == 0) {
            if (start_word == bounds.end_word) {
                return false;
            }
            ++start_word;
        }
        u64 end_word = bounds.end_word;
        while (prot(end_word) == 0) {
            --end_word;
        }
        const u64 start_prot = prot(start_word);
        const u64 end_prot = prot(end_word);
        watcher_bounds = Bounds{
            .start_word = start_word,
            .start_page = static_cast<u64>(std::countr_zero(start_prot)),
            .end_word = end_word,
            .end_page = PAGES_PER_WORD - std::countl_zero(end_prot) - 1,
        };
        return true;
    }

    template <Type type>
        requires(std::popcount(std::to_underlying(type)) == 1)
    constexpr PageOp GetPageOp(StateOp state_op) {
        if constexpr (type == Type::CPU) {
            if (state_op == StateOp::Set) {
                return PageOp::Untrack;
            } else if (state_op == StateOp::Clear) {
                return PageOp::Track;
            }
        } else if (type == Type::GPU && readbacks_mode == GpuReadbacksMode::Precise) {
            if (state_op == StateOp::Set) {
                return PageOp::Track;
            } else if (state_op == StateOp::Clear) {
                return PageOp::Untrack;
            }
        }
        return PageOp::None;
    }

    template <Type type>
        requires(std::popcount(std::to_underlying(type)) == 1)
    RegionBits& GetRegionBits() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    PageManager* tracker;
    VAddr cpu_addr{};
    u32 readbacks_mode;
    RegionBits cpu;
    RegionBits gpu;
    LockType mutex;

    // PERF-012: hot pages and their uploads in the current epoch.
    static constexpr u8 HotPageFaults = 8;
    inline static const bool hot_pages_enabled = Common::PerfFeatureEnabled(12);
    RegionBits hot{};
    RegionBits uploaded{};
    u32 uploaded_epoch{};
    std::array<u8, NUM_REGION_PAGES> write_faults{};

    // PERF-025: the bytes of each hot page as last uploaded. A hot page the CPU has not changed
    // since then is current in the buffer (GPU writes end hot tracking, so nothing else changes
    // it there) and is not uploaded again. The copy is taken before the upload reads the page,
    // so a CPU write in between only causes one more upload.
    inline static const bool hot_shadows_enabled = Common::PerfFeatureEnabled(25);
    std::unordered_map<u64, std::unique_ptr<std::array<u8, BYTES_PER_PAGE>>> hot_shadows;

    u64 UnchangedHotPages(u64 base_page, u64 pages) {
        u64 unchanged{};
        for (u64 bits = pages; bits != 0; bits &= bits - 1) {
            const u64 bit = std::countr_zero(bits);
            const u64 page = base_page + bit;
            const auto* guest = reinterpret_cast<const u8*>(cpu_addr + page * BYTES_PER_PAGE);
            auto& shadow = hot_shadows[page];
            if (shadow && std::memcmp(shadow->data(), guest, BYTES_PER_PAGE) == 0) {
                unchanged |= 1ULL << bit;
                continue;
            }
            if (!shadow) {
                shadow = std::make_unique<std::array<u8, BYTES_PER_PAGE>>();
            }
            std::memcpy(shadow->data(), guest, BYTES_PER_PAGE);
        }
        if (unchanged != 0) {
            Common::GetWorkCounters().hot_pages_unchanged.fetch_add(
                std::popcount(unchanged), std::memory_order_relaxed);
        }
        return unchanged;
    }

    void DropHotShadows(u64 base_page, u64 pages) {
        if (hot_shadows.empty()) {
            return;
        }
        for (u64 bits = pages; bits != 0; bits &= bits - 1) {
            hot_shadows.erase(base_page + std::countr_zero(bits));
        }
    }

    void RefreshUploadEpoch() {
        const u32 epoch = g_upload_epoch.load(std::memory_order_acquire);
        if (uploaded_epoch != epoch) {
            uploaded_epoch = epoch;
            uploaded.Fill(0ULL);
        }
    }
};

} // namespace VideoCore
