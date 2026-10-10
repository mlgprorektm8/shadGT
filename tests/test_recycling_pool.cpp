// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/recycling_pool.h"
#include "common/unique_function.h"

using Common::RecyclingPool;

// PERF-056: blocks of every class are usable for their full size and come back for reuse.
TEST(RecyclingPool, BlocksHoldTheirSizeAndAreReused) {
    for (const size_t size : {1u, 63u, 64u, 65u, 300u, 2048u, 5000u}) {
        void* block = RecyclingPool::Allocate(size);
        ASSERT_NE(block, nullptr);
        std::memset(block, 0xab, size);
        RecyclingPool::Free(block, size);
        if (size <= 2048) {
            void* again = RecyclingPool::Allocate(size);
            EXPECT_EQ(again, block) << size;
            RecyclingPool::Free(again, size);
        }
    }
}

// Made on one thread, destroyed on another, as the draw pipe does with its jobs.
TEST(RecyclingPool, UniqueFunctionsCrossThreads) {
    constexpr int Count = 20000;
    std::vector<Common::UniqueFunction<void>> made;
    std::atomic<long long> sum{0};
    for (int i = 0; i < Count; ++i) {
        std::vector<int> payload(i % 7, i);
        made.emplace_back([&sum, i, payload = std::move(payload)] { sum += i + long(payload.size()); });
    }
    std::thread other{[&] {
        for (auto& f : made) {
            f();
        }
        made.clear();
    }};
    other.join();
    long long expected = 0;
    for (int i = 0; i < Count; ++i) {
        expected += i + i % 7;
    }
    EXPECT_EQ(sum.load(), expected);
}

// Over-aligned callables keep their alignment.
TEST(RecyclingPool, OverAlignedCallables) {
    struct alignas(64) Wide {
        float v[16];
    };
    Wide wide{};
    wide.v[3] = 7.0f;
    Common::UniqueFunction<float> f{[wide] { return wide.v[3]; }};
    EXPECT_EQ(f(), 7.0f);
}

// Vectors handed back keep their capacity.
TEST(Recycler, KeepsCapacity) {
    Common::Recycler<std::vector<unsigned>> recycler;
    std::vector<unsigned> v(1000, 1);
    const auto* data = v.data();
    recycler.Give(std::move(v));
    auto again = recycler.Take();
    EXPECT_EQ(again.data(), data);
    EXPECT_GE(again.capacity(), 1000u);
    auto empty = recycler.Take();
    EXPECT_TRUE(empty.empty());
}
