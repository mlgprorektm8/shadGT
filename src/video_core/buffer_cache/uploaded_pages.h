// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <tsl/robin_map.h>

#include "common/types.h"

namespace VideoCore {

/// PERF-033: what the GPU copy of each PERF-012 hot page holds. Hot pages stay writable, so CPU
/// stores to them are not seen and the page is uploaded again in every upload epoch (about 14 per
/// frame in GT Sport's races, 9-10 MB each). Most epochs find the same bytes as the last upload.
/// A page is left out of an upload only when it has stayed hot since then (the same hot
/// generation: leaving the hot set, as a GPU write makes it, starts a new one) and its bytes hash
/// the same as the bytes uploaded. A page whose bytes changed while they were being uploaded gets
/// no record, so the next epoch uploads it again.
class UploadedPageContents {
public:
    /// True when the page holds what its last upload copied.
    bool Unchanged(VAddr page, u16 generation, u64 hash) const {
        const auto it = pages.find(page);
        return it != pages.end() && it->second.generation == generation && it->second.hash == hash;
    }

    /// Records an upload of the page; `hash_before` and `hash_after` hash its bytes just before
    /// and just after they were copied.
    void Record(VAddr page, u16 generation, u64 hash_before, u64 hash_after) {
        if (hash_before != hash_after) {
            pages.erase(page);
            return;
        }
        pages[page] = Entry{hash_before, generation};
    }

    /// PERF-062: whether the page has an upload record.
    bool Contains(VAddr page) const {
        return pages.contains(page);
    }

    void Forget(VAddr page) {
        pages.erase(page);
    }

    /// Calls func(page) for each recorded page.
    template <typename Func>
    void ForEachPage(Func&& func) const {
        for (const auto& [page, entry] : pages) {
            func(page);
        }
    }

    size_t Size() const {
        return pages.size();
    }

private:
    struct Entry {
        u64 hash;
        u16 generation;
    };
    tsl::robin_map<VAddr, Entry> pages;
};

} // namespace VideoCore
