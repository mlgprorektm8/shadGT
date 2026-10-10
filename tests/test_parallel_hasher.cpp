// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/parallel_hasher.h"

using VideoCore::ParallelPageHasher;

// PERF-046: hashing on several threads gives what one thread gives, batch after batch (batches of
// different sizes back to back exercise the hand-over between them).
TEST(ParallelPageHasher, MatchesSerialHashesAcrossBatches) {
    constexpr size_t PageSize = 4096;
    std::mt19937_64 rng{42};
    std::vector<u8> memory(PageSize * 600);
    for (auto& byte : memory) {
        byte = static_cast<u8>(rng());
    }
    ParallelPageHasher hasher{3};
    for (size_t round = 0; round < 300; ++round) {
        const size_t count = 1 + (rng() % 600);
        std::vector<const u8*> pages(count);
        for (size_t i = 0; i < count; ++i) {
            pages[i] = memory.data() + PageSize * (rng() % 600);
        }
        std::vector<u64> hashes(count, 0);
        hasher.Hash(pages, PageSize, hashes);
        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(hashes[i], XXH3_64bits(pages[i], PageSize)) << "round " << round;
        }
    }
}
