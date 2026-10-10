// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <xxhash.h>

#include "video_core/buffer_cache/hot_page_prehasher.h"

using VideoCore::HotPagePrehasher;

namespace {
constexpr size_t PageSize = 4096;

struct Fixture {
    std::vector<u8> memory = std::vector<u8>(PageSize * 64);
    std::vector<VAddr> source_pages;
    std::atomic<u32> epoch{1};
    std::atomic<u32> guard_calls{};
    std::vector<VAddr> invalid;

    HotPagePrehasher Make() {
        return HotPagePrehasher{
            1, PageSize, epoch, [this](std::vector<VAddr>& pages) { pages = source_pages; },
            [this](const std::function<void()>& func) {
                ++guard_calls;
                func();
            },
            [this](VAddr page) {
                return std::ranges::find(invalid, page) == invalid.end();
            }};
    }

    VAddr Page(size_t i) {
        return reinterpret_cast<VAddr>(memory.data() + PageSize * i);
    }
};
} // namespace

// PERF-049: hashes taken ahead equal XXH3 of the page and are found only in their own epoch.
TEST(HotPagePrehasher, HashesMatchAndBelongToTheirEpoch) {
    Fixture f;
    std::mt19937_64 rng{7};
    for (auto& byte : f.memory) {
        byte = static_cast<u8>(rng());
    }
    for (size_t i = 0; i < 40; ++i) {
        f.source_pages.push_back(f.Page(i));
    }
    auto hasher = f.Make();
    f.epoch = 2;
    hasher.OnEpoch(2);
    hasher.WaitIdle(2);
    const auto batch = hasher.Current();
    ASSERT_TRUE(batch);
    for (size_t i = 0; i < 40; ++i) {
        u64 hash = 0;
        ASSERT_TRUE(batch->Lookup(f.Page(i), 2, hash)) << i;
        EXPECT_EQ(hash, XXH3_64bits(reinterpret_cast<const void*>(f.Page(i)), PageSize));
        EXPECT_FALSE(batch->Lookup(f.Page(i), 3, hash)) << "another epoch's hash used";
    }
    u64 hash = 0;
    EXPECT_FALSE(batch->Lookup(f.Page(50), 2, hash)) << "a page that was not requested";
    EXPECT_GT(f.guard_calls.load(), 0u);
}

// Pages the mapping check rejects get no hash; the caller hashes them itself.
TEST(HotPagePrehasher, SkipsUnmappedPages) {
    Fixture f;
    for (size_t i = 0; i < 20; ++i) {
        f.source_pages.push_back(f.Page(i));
    }
    f.invalid = {f.Page(3), f.Page(17)};
    auto hasher = f.Make();
    f.epoch = 5;
    hasher.OnEpoch(5);
    hasher.WaitIdle(5);
    const auto batch = hasher.Current();
    u64 hash = 0;
    EXPECT_FALSE(batch->Lookup(f.Page(3), 5, hash));
    EXPECT_FALSE(batch->Lookup(f.Page(17), 5, hash));
    EXPECT_TRUE(batch->Lookup(f.Page(4), 5, hash));
}

// A newer epoch abandons the old batch; the new one hashes the bytes as they are then.
TEST(HotPagePrehasher, NewEpochRehashesCurrentBytes) {
    Fixture f;
    for (size_t i = 0; i < 64; ++i) {
        f.source_pages.push_back(f.Page(i));
    }
    auto hasher = f.Make();
    for (u32 round = 0; round < 200; ++round) {
        f.memory[PageSize * (round % 64)] = static_cast<u8>(round);
        const u32 epoch = 10 + round;
        f.epoch = epoch;
        hasher.OnEpoch(epoch);
        if (round % 10 != 9) {
            continue; // back-to-back epochs abandon batches
        }
        hasher.WaitIdle(epoch);
        const auto batch = hasher.Current();
        ASSERT_EQ(batch->epoch, epoch);
        for (size_t i = 0; i < 64; ++i) {
            u64 hash = 0;
            ASSERT_TRUE(batch->Lookup(f.Page(i), epoch, hash));
            ASSERT_EQ(hash, XXH3_64bits(reinterpret_cast<const void*>(f.Page(i)), PageSize));
        }
    }
}
