// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FIX-041: a fence becomes visible only after every readback issued before it has landed.

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/fence_order.h"

using Vulkan::FenceReadbackState;
using Vulkan::ShouldDeferFence;

namespace {
constexpr FenceReadbackState State(bool compute, bool queued_any, bool queued_large,
                                   bool in_flight, bool address_pending = false) {
    return {compute, queued_any, queued_large, in_flight, address_pending};
}
} // namespace

TEST(FenceOrder, NothingPendingSignalsAtOnce) {
    for (const bool compute : {false, true}) {
        EXPECT_FALSE(ShouldDeferFence(State(compute, false, false, false), true, true));
        EXPECT_FALSE(ShouldDeferFence(State(compute, false, false, false), false, true));
    }
}

TEST(FenceOrder, InFlightReadbackHoldsLaterFences) {
    // The gap: an earlier fence took the readback; this fence finds nothing queued.
    for (const bool compute : {false, true}) {
        EXPECT_TRUE(ShouldDeferFence(State(compute, false, false, true), true, true));
        EXPECT_FALSE(ShouldDeferFence(State(compute, false, false, true), false, true));
    }
}

TEST(FenceOrder, ComputeFencesWaitForSmallReadbacks) {
    // A small luminance readback queued before a compute-queue RELEASE_MEM.
    EXPECT_TRUE(ShouldDeferFence(State(true, true, false, false), true, true));
    EXPECT_FALSE(ShouldDeferFence(State(true, true, false, false), false, true));
    // Large ones were already covered by FIX-032.
    EXPECT_TRUE(ShouldDeferFence(State(true, true, true, false), false, true));
    EXPECT_FALSE(ShouldDeferFence(State(true, true, true, false), false, false));
}

TEST(FenceOrder, GraphicsRulesUnchangedForQueuedReadbacks) {
    EXPECT_TRUE(ShouldDeferFence(State(false, true, false, false), true, true));
    EXPECT_TRUE(ShouldDeferFence(State(false, true, false, false), false, true));
}

TEST(FenceOrder, PendingAddressKeepsItsOrder) {
    for (const bool order_all : {false, true}) {
        EXPECT_TRUE(ShouldDeferFence(State(true, false, false, false, true), order_all, false));
    }
}
