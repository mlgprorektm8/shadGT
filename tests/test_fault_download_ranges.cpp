// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/buffer_cache/fault_download_ranges.h"

using Range = std::pair<VAddr, VAddr>;
using VideoCore::PartsWithoutNewerWrites;
using VideoCore::TouchedByNewerWrite;

// PERF-052: with no newer write the whole downloaded range is released.
TEST(FaultDownloadRanges, NoNewerWriteReleasesEverything) {
    const auto parts = PartsWithoutNewerWrites(0x1000, 0x2000, {});
    ASSERT_EQ(parts.size(), 1u);
    EXPECT_EQ(parts[0], Range(0x1000, 0x2000));
}

// Newer writes, overlapping each other and the edges, are cut out exactly.
TEST(FaultDownloadRanges, NewerWritesAreCutOut) {
    const std::vector<Range> newer{{0x1800, 0x1900}, {0x0f00, 0x1100}, {0x1880, 0x1a00},
                                   {0x3000, 0x4000}};
    const auto parts = PartsWithoutNewerWrites(0x1000, 0x2000, newer);
    const std::vector<Range> expected{{0x1100, 0x1800}, {0x1a00, 0x2000}};
    EXPECT_EQ(parts, expected);
    EXPECT_TRUE(TouchedByNewerWrite(0x1000, 0x1001, newer));
    EXPECT_FALSE(TouchedByNewerWrite(0x2000, 0x3000, newer));
}

// Against a byte map: a byte is released exactly when no newer write covers it.
TEST(FaultDownloadRanges, MatchesByteByByteReference) {
    std::mt19937 rng{99};
    for (int round = 0; round < 2000; ++round) {
        const VAddr start = rng() % 256;
        const VAddr end = start + 1 + rng() % 256;
        std::vector<Range> newer;
        for (u32 i = 0, n = rng() % 6; i < n; ++i) {
            const VAddr s = rng() % 600;
            newer.emplace_back(s, s + 1 + rng() % 80);
        }
        std::vector<bool> expected(1024, false), got(1024, false);
        for (VAddr b = start; b < end; ++b) {
            expected[b] = std::none_of(newer.begin(), newer.end(),
                                       [&](const Range& w) { return w.first <= b && b < w.second; });
        }
        for (const auto& [s, e] : PartsWithoutNewerWrites(start, end, newer)) {
            ASSERT_LT(s, e);
            for (VAddr b = s; b < e; ++b) {
                ASSERT_FALSE(got[b]) << "overlapping parts";
                got[b] = true;
            }
        }
        ASSERT_EQ(got, expected) << "round " << round;
    }
}
