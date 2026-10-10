// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>

#include "common/types.h"

namespace VideoCore {

/// PERF-062: page hashes taken by one thread (the command thread, as it decodes a draw) and
/// looked up by another (the recorder, when it uploads the draw's pages). Direct-mapped; each
/// slot is a seqlock, so a lookup never sees a torn entry and never blocks the writer. An entry
/// carries the upload epoch it was taken in; it answers only for that epoch.
class PageHashTable {
public:
    explicit PageHashTable(size_t num_slots = size_t{1} << 16)
        : mask{num_slots - 1}, slots{std::make_unique<Slot[]>(num_slots)} {}

    /// Single writer.
    void Store(VAddr page, u32 epoch, u64 hash) noexcept {
        Slot& slot = slots[Index(page)];
        const u64 seq = slot.seq.load(std::memory_order_relaxed);
        slot.seq.store(seq + 1, std::memory_order_relaxed); // odd: being written
        std::atomic_thread_fence(std::memory_order_release);
        slot.page.store(page, std::memory_order_relaxed);
        slot.epoch.store(epoch, std::memory_order_relaxed);
        slot.hash.store(hash, std::memory_order_relaxed);
        slot.seq.store(seq + 2, std::memory_order_release);
    }

    /// Any thread: the hash of `page` taken in `epoch`, if the slot holds one.
    bool Lookup(VAddr page, u32 epoch, u64& hash) const noexcept {
        const Slot& slot = slots[Index(page)];
        const u64 seq = slot.seq.load(std::memory_order_acquire);
        if (seq & 1) {
            return false;
        }
        const VAddr stored_page = slot.page.load(std::memory_order_relaxed);
        const u32 stored_epoch = slot.epoch.load(std::memory_order_relaxed);
        const u64 stored_hash = slot.hash.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (slot.seq.load(std::memory_order_relaxed) != seq || stored_page != page ||
            stored_epoch != epoch || seq == 0) {
            return false;
        }
        hash = stored_hash;
        return true;
    }

    /// Writer side: whether the slot already holds `page` for `epoch`.
    bool Has(VAddr page, u32 epoch) const noexcept {
        u64 unused;
        return Lookup(page, epoch, unused);
    }

private:
    struct Slot {
        std::atomic<u64> seq{0};
        std::atomic<VAddr> page{0};
        std::atomic<u32> epoch{0};
        std::atomic<u64> hash{0};
    };

    size_t Index(VAddr page) const noexcept {
        const u64 p = page >> 12;
        return static_cast<size_t>((p ^ (p >> 16)) & mask);
    }

    size_t mask;
    std::unique_ptr<Slot[]> slots;
};

} // namespace VideoCore
