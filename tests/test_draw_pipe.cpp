// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <random>
#include <vector>
#include <gtest/gtest.h>

#include "video_core/amdgpu/draw_pipe.h"
#include "video_core/amdgpu/regs_delta.h"

namespace {

constexpr u32 NumDwords = 0xD000; // the size of AmdGpu::Regs
using Delta = AmdGpu::RegsDelta<NumDwords>;

} // namespace

TEST(RegsDelta, ReproducesTheRegisterFileFromItsWrittenBlocks) {
    std::vector<u32> frontend(NumDwords, 0);
    std::vector<u32> recorder(NumDwords, 0);
    Delta delta;
    std::mt19937 rng{1234};
    for (int draw = 0; draw < 2000; ++draw) {
        // A few register writes of a few dwords, as SET_CONTEXT_REG / SET_SH_REG make them.
        const int writes = int(rng() % 12);
        for (int w = 0; w < writes; ++w) {
            const u32 count = 1 + rng() % 20;
            const u32 first = rng() % (NumDwords - count);
            for (u32 i = 0; i < count; ++i) {
                frontend[first + i] = rng();
            }
            delta.Mark(first, count);
        }
        std::vector<u32> packet;
        delta.Collect(std::span<const u32, NumDwords>{frontend.data(), NumDwords}, packet);
        EXPECT_FALSE(delta.Any());
        Delta::Apply(packet, std::span<u32, NumDwords>{recorder.data(), NumDwords});
        ASSERT_EQ(frontend, recorder) << "after draw " << draw;
    }
}

TEST(RegsDelta, SendsOnlyTheWrittenBlocks) {
    std::vector<u32> regs(NumDwords, 7);
    Delta delta;
    delta.Mark(31, 2); // straddles blocks 0 and 1
    delta.Mark(0x2C00 + 5, 1);
    std::vector<u32> packet;
    delta.Collect(std::span<const u32, NumDwords>{regs.data(), NumDwords}, packet);
    ASSERT_EQ(packet.size(), 3 * Delta::RecordDwords);
    EXPECT_EQ(packet[0], 0u);
    EXPECT_EQ(packet[Delta::RecordDwords], 1u);
    EXPECT_EQ(packet[2 * Delta::RecordDwords], 0x2C00u / 32);
    packet.clear();
    delta.Collect(std::span<const u32, NumDwords>{regs.data(), NumDwords}, packet);
    EXPECT_TRUE(packet.empty());
}

TEST(RegsDelta, MarkAllSendsEveryBlockAndIgnoresWritesPastTheEnd) {
    std::vector<u32> regs(NumDwords, 3);
    std::vector<u32> copy(NumDwords, 0);
    Delta delta;
    delta.Mark(NumDwords, 4);
    EXPECT_FALSE(delta.Any());
    delta.Mark(NumDwords - 2, 8); // clamped to the last block
    delta.MarkAll();
    std::vector<u32> packet;
    delta.Collect(std::span<const u32, NumDwords>{regs.data(), NumDwords}, packet);
    EXPECT_EQ(packet.size(), Delta::NumBlocks * Delta::RecordDwords);
    Delta::Apply(packet, std::span<u32, NumDwords>{copy.data(), NumDwords});
    EXPECT_EQ(copy, regs);
}

TEST(DrawPipe, RunsWorkInOrderOnTheRecorderThread) {
    std::thread::id recorder_seen{};
    AmdGpu::DrawPipe pipe{[&] { recorder_seen = std::this_thread::get_id(); }};
    std::vector<int> order;
    std::atomic<bool> off_thread{false};
    for (int i = 0; i < 5000; ++i) {
        pipe.Push([&, i] {
            order.push_back(i);
            if (!pipe.OnRecorderThread()) {
                off_thread = true;
            }
        });
    }
    pipe.Drain();
    ASSERT_EQ(order.size(), 5000u);
    for (int i = 0; i < 5000; ++i) {
        ASSERT_EQ(order[i], i);
    }
    EXPECT_FALSE(off_thread);
    EXPECT_EQ(pipe.RecorderThreadId(), recorder_seen);
    EXPECT_FALSE(pipe.OnRecorderThread());
}

TEST(DrawPipe, DrainWaitsForWorkAlreadyRunning) {
    AmdGpu::DrawPipe pipe{nullptr};
    std::atomic<int> done{0};
    for (int i = 0; i < 8; ++i) {
        pipe.Push([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            ++done;
        });
    }
    pipe.Drain();
    EXPECT_EQ(done.load(), 8);
    // Draining an idle pipe returns at once; a drain on the recorder thread is a no-op.
    pipe.Drain();
    pipe.Push([&] {
        pipe.Drain();
        ++done;
    });
    pipe.Drain();
    EXPECT_EQ(done.load(), 9);
}

TEST(DrawPipe, BoundsTheQueueAndKeepsOrderUnderBackpressure) {
    AmdGpu::DrawPipe pipe{nullptr, 16};
    std::vector<int> order;
    for (int i = 0; i < 1000; ++i) {
        pipe.Push([&, i] {
            if (i % 100 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            order.push_back(i);
        });
    }
    pipe.Drain();
    ASSERT_EQ(order.size(), 1000u);
    for (int i = 0; i < 1000; ++i) {
        ASSERT_EQ(order[i], i);
    }
    const auto stats = pipe.TakeStats();
    EXPECT_EQ(stats.pushed, 1000u);
    EXPECT_LE(stats.max_queued, 16u);
}

TEST(DrawPipe, StateWrittenBeforeADrainIsVisibleAfterIt) {
    // The command thread reads results (a GDS value, a fence) only after draining.
    AmdGpu::DrawPipe pipe{nullptr};
    std::mt19937 rng{7};
    u64 value = 0;
    for (int round = 0; round < 2000; ++round) {
        const u64 expected = rng();
        const int jobs = int(rng() % 4);
        for (int j = 0; j < jobs; ++j) {
            pipe.Push([] {});
        }
        pipe.Push([&value, expected] { value = expected; });
        pipe.Drain();
        ASSERT_EQ(value, expected);
    }
}
