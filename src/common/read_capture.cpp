// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <mutex>
#include <xxhash.h>
#include "common/read_capture.h"

namespace Common {

thread_local const ReadCapture* t_read_capture = nullptr;

namespace {
// Chunks are kept for the next captures; a capture is made per graphics submission.
constexpr size_t ChunkBytes = 256 * ReadCapture::PageSize;
std::mutex g_chunk_mutex;
std::vector<u8*> g_free_chunks;

u8* TakeChunk() {
    {
        std::scoped_lock lk{g_chunk_mutex};
        if (!g_free_chunks.empty()) {
            u8* chunk = g_free_chunks.back();
            g_free_chunks.pop_back();
            return chunk;
        }
    }
    return new u8[ChunkBytes];
}

void GiveChunk(u8* chunk) {
    std::scoped_lock lk{g_chunk_mutex};
    if (g_free_chunks.size() < 256) {
        g_free_chunks.push_back(chunk);
    } else {
        delete[] chunk;
    }
}
} // namespace

namespace {
std::atomic<u64> g_next_capture_id{1};
}

ReadCapture::ReadCapture(size_t max_pages_)
    : id{g_next_capture_id.fetch_add(1)}, max_pages{max_pages_},
      mask{std::bit_ceil(max_pages_ * 8) - 1}, max_entries{(mask + 1) / 2},
      keys{std::make_unique<std::atomic<VAddr>[]>(mask + 1)},
      indices{std::make_unique<u32[]>(mask + 1)},
      hashes{std::make_unique<u64[]>(max_pages_)},
      chunks((max_pages_ + ChunkPages - 1) / ChunkPages, nullptr) {}

ReadCapture::~ReadCapture() {
    for (u8* chunk : chunks) {
        if (chunk) {
            GiveChunk(chunk);
        }
    }
}

bool ReadCapture::Add(VAddr page, const u8* source) {
    const size_t n = count.load(std::memory_order_relaxed);
    if (n >= max_pages) {
        full.store(true, std::memory_order_relaxed);
        return false;
    }
    u8*& chunk = chunks[n / ChunkPages];
    if (!chunk) {
        chunk = TakeChunk();
    }
    if (entries >= max_entries) {
        full.store(true, std::memory_order_relaxed);
        return false;
    }
    u8* copy = chunk + (n % ChunkPages) * PageSize;
    std::memcpy(copy, source, PageSize);
    hashes[n] = XXH3_64bits(copy, PageSize);
    Insert(page, static_cast<u32>(n));
    count.store(n + 1, std::memory_order_release);
    return true;
}

bool ReadCapture::AddWatched(VAddr page) {
    if (entries >= max_entries) {
        full.store(true, std::memory_order_relaxed);
        return false;
    }
    return Insert(page, WatchedIndex);
}

bool ReadCapture::Insert(VAddr page, u32 index) {
    for (size_t slot = Slot(page);; slot = (slot + 1) & mask) {
        const VAddr key = keys[slot].load(std::memory_order_relaxed);
        if (key == page) {
            return true;
        }
        if (key == 0) {
            indices[slot] = index;
            // The copy, its hash and index are visible to a reader that sees the key.
            keys[slot].store(page, std::memory_order_release);
            ++entries;
            return true;
        }
    }
}

bool ReadCapture::Lists(VAddr page) const noexcept {
    for (size_t slot = Slot(page);; slot = (slot + 1) & mask) {
        const VAddr key = keys[slot].load(std::memory_order_acquire);
        if (key == 0) {
            return false;
        }
        if (key == page) {
            return true;
        }
    }
}

const u8* ReadCapture::Find(VAddr page, u64* hash) const noexcept {
    for (size_t slot = Slot(page);; slot = (slot + 1) & mask) {
        const VAddr key = keys[slot].load(std::memory_order_acquire);
        if (key == 0) {
            return nullptr;
        }
        if (key == page) {
            const u32 n = indices[slot];
            if (n == WatchedIndex) {
                return nullptr;
            }
            if (hash) {
                *hash = hashes[n];
            }
            return chunks[n / ChunkPages] + (n % ChunkPages) * PageSize;
        }
    }
}

} // namespace Common
