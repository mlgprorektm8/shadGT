// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstring>
#include <memory>
#include <vector>
#include "common/types.h"

namespace Common {

/// PERF-063: the guest pages a graphics submission reads that the CPU can write without a fault
/// (unprotected: hot, dirty or untracked pages), copied when the command thread decodes the
/// submission. The recorder reads these copies instead of guest memory, so the submission's
/// fence can be signaled once it is decoded: the game may then rewrite the pages while the
/// recorder still works on the submission, as it may on hardware once the fence is signaled.
///
/// One writer (the command thread) adds pages while readers (the recorder and its upload copy
/// workers) look them up; an added page is never changed or removed until the capture is reset
/// (when nothing reads it any more).
class ReadCapture {
public:
    static constexpr u64 PageBits = 12;
    static constexpr u64 PageSize = u64{1} << PageBits;

    explicit ReadCapture(size_t max_pages);
    ~ReadCapture();

    ReadCapture(const ReadCapture&) = delete;
    ReadCapture& operator=(const ReadCapture&) = delete;

    /// Writer: copies the page from `source` (its bytes as the guest sees them). False when the
    /// capture is full; the page is then not captured.
    bool Add(VAddr page, const u8* source);

    /// Writer: lists a page the work reads that stays protected (not copied: a CPU write to it
    /// faults, and the fault waits for the recorder). False when the list is full.
    bool AddWatched(VAddr page);

    /// Any thread: the copy of the page and its XXH3 hash, or null (also for a watched page).
    const u8* Find(VAddr page, u64* hash = nullptr) const noexcept;

    /// Any thread: whether the page is copied or watched.
    bool Lists(VAddr page) const noexcept;

    /// A number no other capture of this process has.
    u64 Id() const noexcept {
        return id;
    }

    bool Has(VAddr page) const noexcept {
        return Find(page) != nullptr;
    }

    size_t NumPages() const noexcept {
        return count.load(std::memory_order_acquire);
    }

    bool Full() const noexcept {
        return full.load(std::memory_order_relaxed);
    }

    /// Copies [address, address + size): captured pages from the capture, the rest with
    /// read_rest(address, dest, size).
    template <typename ReadRest>
    void Copy(VAddr address, u8* dest, u64 size, ReadRest&& read_rest) const {
        while (size != 0) {
            const VAddr page = address & ~(PageSize - 1);
            const u64 offset = address - page;
            const u64 n = std::min<u64>(PageSize - offset, size);
            if (const u8* copy = Find(page)) {
                std::memcpy(dest, copy + offset, n);
            } else {
                read_rest(address, dest, n);
            }
            address += n;
            dest += n;
            size -= n;
        }
    }

private:
    static constexpr size_t ChunkPages = 256; // 1 MB

    static constexpr u32 WatchedIndex = 0xFFFFFFFFu;
    bool Insert(VAddr page, u32 index);

    size_t Slot(VAddr page) const noexcept {
        const u64 p = page >> PageBits;
        return static_cast<size_t>((p * 0x9E3779B97F4A7C15ull) >> 20) & mask;
    }

    u64 id;
    size_t max_pages;
    size_t mask;
    size_t max_entries;
    size_t entries{};
    std::unique_ptr<std::atomic<VAddr>[]> keys;
    std::unique_ptr<u32[]> indices;
    std::unique_ptr<u64[]> hashes;
    std::vector<u8*> chunks; // sized at construction; entries filled by the writer
    std::atomic<size_t> count{};
    std::atomic<bool> full{};
};

/// PERF-063: the capture of the work the current thread records (the recorder sets it around
/// each job it takes from a captured submission), or null.
extern thread_local const ReadCapture* t_read_capture;

/// Copies guest memory, taking the bytes from the current thread's capture where it has them.
inline void CopyGuest(void* dest, const void* source, size_t size) {
    if (const ReadCapture* capture = t_read_capture) [[unlikely]] {
        capture->Copy(reinterpret_cast<VAddr>(source), static_cast<u8*>(dest), size,
                      [](VAddr address, u8* to, u64 n) {
                          std::memcpy(to, reinterpret_cast<const void*>(address), n);
                      });
        return;
    }
    std::memcpy(dest, source, size);
}

} // namespace Common
