// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/amdgpu/ce_de_counter.h"

TEST(CeDeCounter, AllowsPrefetchBelowLimitInsteadOfUnderflowing) {
    EXPECT_FALSE(AmdGpu::ShouldWaitOnDeCounter(1, 0, 4));
    EXPECT_FALSE(AmdGpu::ShouldWaitOnDeCounter(3, 0, 4));
    EXPECT_TRUE(AmdGpu::ShouldWaitOnDeCounter(4, 0, 4));
}

TEST(CeDeCounter, ReleasesWhenDrawEngineCatchesUp) {
    EXPECT_TRUE(AmdGpu::ShouldWaitOnDeCounter(10, 6, 4));
    EXPECT_FALSE(AmdGpu::ShouldWaitOnDeCounter(10, 7, 4));
    EXPECT_FALSE(AmdGpu::ShouldWaitOnDeCounter(10, 10, 4));
}

TEST(CeDeCounter, PreservesCounterWraparound) {
    EXPECT_FALSE(AmdGpu::ShouldWaitOnDeCounter(1, 0xfffffffe, 4));
    EXPECT_TRUE(AmdGpu::ShouldWaitOnDeCounter(2, 0xfffffffe, 4));
}
