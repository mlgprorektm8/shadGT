// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <span>
#include <vector>

#include "common/types.h"

namespace AmdGpu {

/// PERF-031: the register blocks written since the last draw, so the recorder thread can bring
/// its own copy of the register file up to date with a draw's state instead of copying all of it.
/// A delta is a sequence of [block index, BlockDwords values] records.
template <u32 NumDwords, u32 BlockDwords = 32>
class RegsDelta {
public:
    static_assert(NumDwords % BlockDwords == 0);
    static constexpr u32 NumBlocks = NumDwords / BlockDwords;
    static constexpr u32 RecordDwords = BlockDwords + 1;

    /// Marks the dwords [first, first + count) as written.
    void Mark(u32 first, u32 count) {
        if (count == 0 || first >= NumDwords) {
            return;
        }
        const u32 last = std::min(first + count, NumDwords) - 1;
        for (u32 block = first / BlockDwords; block <= last / BlockDwords; ++block) {
            dirty[block / 64] |= u64{1} << (block % 64);
        }
    }

    void MarkAll() {
        dirty.fill(~u64{0});
        if constexpr (NumBlocks % 64 != 0) {
            dirty.back() = (u64{1} << (NumBlocks % 64)) - 1;
        }
    }

    bool Any() const {
        return std::ranges::any_of(dirty, [](u64 word) { return word != 0; });
    }

    /// Appends a record for each written block of `regs` to `out` and clears the marks.
    void Collect(std::span<const u32, NumDwords> regs, std::vector<u32>& out) {
        for (u32 word = 0; word < dirty.size(); ++word) {
            u64 bits = dirty[word];
            while (bits != 0) {
                const u32 block = word * 64 + std::countr_zero(bits);
                bits &= bits - 1;
                const size_t at = out.size();
                out.resize(at + RecordDwords);
                out[at] = block;
                std::memcpy(&out[at + 1], &regs[block * BlockDwords], BlockDwords * sizeof(u32));
            }
            dirty[word] = 0;
        }
    }

    /// Writes the blocks of a delta made by Collect into `regs`.
    static void Apply(std::span<const u32> delta, std::span<u32, NumDwords> regs) {
        for (size_t at = 0; at + RecordDwords <= delta.size(); at += RecordDwords) {
            const u32 block = delta[at];
            if (block < NumBlocks) {
                std::memcpy(&regs[block * BlockDwords], &delta[at + 1], BlockDwords * sizeof(u32));
            }
        }
    }

private:
    std::array<u64, (NumBlocks + 63) / 64> dirty{};
};

} // namespace AmdGpu
