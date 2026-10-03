// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>
#include "video_core/amdgpu/pm4_rewind.h"

TEST(PM4Rewind, WaitsForLivePublicationThenRefreshesSubsequentCommands) {
    std::array<u32, 6> live{0x80000000, 0xc0005900, 0, 0, 0, 0};
    auto snapshot = live;
    live[3] = 0xc0016900;
    live[4] = 0x10;
    live[5] = 0x1234;
    EXPECT_FALSE(AmdGpu::RefreshRewindTailIfReady(snapshot, live, 1));
    EXPECT_EQ(snapshot[3], 0);
    std::atomic_ref<u32>(live[2]).store(0x80000000, std::memory_order_release);
    ASSERT_TRUE(AmdGpu::RefreshRewindTailIfReady(snapshot, live, 1));
    EXPECT_EQ(snapshot, live);
}

TEST(PM4Rewind, PollsLiveBitEvenWhenSnapshotIsAlreadyValid) {
    std::array<u32, 2> live{0xc0005900, 0};
    auto snapshot = live;
    snapshot[1] = 0x80000000;
    EXPECT_FALSE(AmdGpu::RefreshRewindTailIfReady(snapshot, live, 0));
}

TEST(PM4Rewind, SupportsDirectBuffersAndRejectsInvalidPacketBounds) {
    std::array<u32, 2> live{0xc0005900, 0x80000000};
    EXPECT_TRUE(AmdGpu::RefreshRewindTailIfReady(live, live, 0));
    EXPECT_FALSE(AmdGpu::RefreshRewindTailIfReady(live, live, 1));
    EXPECT_FALSE(AmdGpu::RefreshRewindTailIfReady(live, live, 100));
    EXPECT_FALSE(AmdGpu::RefreshRewindTailIfReady(live, {}, 0));
}
