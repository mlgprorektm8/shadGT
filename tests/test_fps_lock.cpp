// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "core/libraries/videoout/fps_lock.h"

namespace {

using Libraries::VideoOut::LockedFlipRate;

TEST(FpsLock, OffKeepsTheGameRate) {
    EXPECT_EQ(LockedFlipRate(0, 0, 60), 0);
    EXPECT_EQ(LockedFlipRate(1, 0, 60), 1);
}

TEST(FpsLock, ThirtyOnSixtyHertzFlipsEveryOtherVblank) {
    EXPECT_EQ(LockedFlipRate(0, 30, 60), 1);
}

TEST(FpsLock, NeverFasterThanTheGameAsks) {
    EXPECT_EQ(LockedFlipRate(2, 30, 60), 2); // the game's own 20 FPS
}

TEST(FpsLock, OtherRates) {
    EXPECT_EQ(LockedFlipRate(0, 20, 60), 2);
    EXPECT_EQ(LockedFlipRate(0, 60, 60), 0);  // no slower than the vblank: no lock
    EXPECT_EQ(LockedFlipRate(0, 120, 60), 0);
    EXPECT_EQ(LockedFlipRate(0, 45, 60), 1);  // rounds to whole vblanks, never faster
    EXPECT_EQ(LockedFlipRate(0, 30, 0), 0);   // no vblank rate known
}

} // namespace
