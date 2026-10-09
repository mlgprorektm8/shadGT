// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "common/guest_clock.h"

using Common::GuestClock::PausableCounter;

namespace {

u64 ReadAt(const PausableCounter& counter, u64 host) {
    return counter.Read([host] { return host; });
}

} // namespace

TEST(GuestClock, FollowsTheHostCounterWhileNotHeld) {
    PausableCounter counter;
    EXPECT_EQ(ReadAt(counter, 1000), 1000u);
    EXPECT_EQ(ReadAt(counter, 5000), 5000u);
    EXPECT_FALSE(counter.IsPaused());
}

TEST(GuestClock, StandsStillWhileHeldAndOmitsTheHeldTimeAfter) {
    PausableCounter counter;
    counter.Pause(1000);
    EXPECT_TRUE(counter.IsPaused());
    EXPECT_EQ(ReadAt(counter, 1000), 1000u);
    EXPECT_EQ(ReadAt(counter, 60000), 1000u); // a 59000-tick compile
    EXPECT_EQ(counter.Resume(60000), 59000u);
    EXPECT_FALSE(counter.IsPaused());
    EXPECT_EQ(ReadAt(counter, 60000), 1000u);
    EXPECT_EQ(ReadAt(counter, 61000), 2000u);

    // A second hold adds to the first.
    counter.Pause(70000);
    EXPECT_EQ(ReadAt(counter, 90000), 11000u);
    counter.Resume(90000);
    EXPECT_EQ(ReadAt(counter, 91000), 12000u);
    EXPECT_EQ(counter.PausedTotal(), 79000u);
}

TEST(GuestClock, NestedHoldsEndWithTheOutermost) {
    PausableCounter counter;
    counter.Pause(100);
    counter.Pause(200); // a shader translated inside a pipeline build
    EXPECT_EQ(counter.Resume(300), 0u);
    EXPECT_TRUE(counter.IsPaused());
    EXPECT_EQ(ReadAt(counter, 400), 100u);
    EXPECT_EQ(counter.Resume(500), 400u);
    EXPECT_EQ(ReadAt(counter, 600), 200u);
}

TEST(GuestClock, NeverRunsBackwardsAcrossAHold) {
    PausableCounter counter;
    u64 last = 0;
    u64 host = 0;
    for (int round = 0; round < 100; ++round) {
        host += 7;
        const u64 before = ReadAt(counter, host);
        EXPECT_GE(before, last);
        last = before;
        counter.Pause(host);
        host += 1000 + round;
        EXPECT_EQ(ReadAt(counter, host), last);
        counter.Resume(host);
        const u64 after = ReadAt(counter, host + 1);
        EXPECT_GE(after, last);
        last = after;
    }
}

TEST(GuestClock, ResumeWithoutAHoldChangesNothing) {
    PausableCounter counter;
    EXPECT_EQ(counter.Resume(500), 0u);
    EXPECT_EQ(ReadAt(counter, 800), 800u);
}
