// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/stencil_reference.h"

using AmdGpu::CompareFunc;
using AmdGpu::StencilFunc;
using AmdGpu::StencilRefMask;
using Vulkan::ResolveStencilReference;

TEST(StencilReference, OnesWritesEveryUnmaskedBit) {
    const StencilRefMask ref{0x12, 0xff, 0x3c, 0x34};
    const auto result = ResolveStencilReference(StencilFunc::Keep, StencilFunc::Ones,
                                                StencilFunc::Keep, CompareFunc::Always, ref);
    EXPECT_TRUE(result.exact);
    for (u32 old = 0; old <= 0xff; ++old) {
        EXPECT_EQ((old & ~ref.stencil_write_mask) | (result.value & ref.stencil_write_mask),
                  old | ref.stencil_write_mask);
    }
}

TEST(StencilReference, KeepsTestReferenceWithoutSpecialReplacements) {
    const StencilRefMask ref{0x12, 0xff, 0xff, 0x34};
    const auto result = ResolveStencilReference(StencilFunc::Zero, StencilFunc::ReplaceTest,
                                                StencilFunc::AddWrap, CompareFunc::Equal, ref);
    EXPECT_EQ(result.value, 0x12);
    EXPECT_TRUE(result.exact);
}

TEST(StencilReference, CompareMaskAllowsDifferentOpReference) {
    const StencilRefMask ref{0x12, 0xf0, 0xff, 0x1f};
    const auto result = ResolveStencilReference(StencilFunc::Keep, StencilFunc::ReplaceOp,
                                                StencilFunc::Keep, CompareFunc::Equal, ref);
    EXPECT_EQ(result.value, 0x1f);
    EXPECT_TRUE(result.exact);
}

TEST(StencilReference, ReportsUnrepresentableComparison) {
    const StencilRefMask ref{0x12, 0xff, 0xff, 0x34};
    const auto result = ResolveStencilReference(StencilFunc::Keep, StencilFunc::Ones,
                                                StencilFunc::Keep, CompareFunc::Equal, ref);
    EXPECT_FALSE(result.exact);
}

TEST(StencilReference, ReportsDifferentReplacementValues) {
    const StencilRefMask ref{0x12, 0xff, 0xff, 0x34};
    const auto result = ResolveStencilReference(StencilFunc::ReplaceTest, StencilFunc::Ones,
                                                StencilFunc::ReplaceOp, CompareFunc::Always, ref);
    EXPECT_FALSE(result.exact);
}

TEST(StencilReference, WriteMaskCanMakeReplacementValuesEquivalent) {
    const StencilRefMask ref{0x3f, 0xff, 0x0f, 0x7f};
    const auto result = ResolveStencilReference(StencilFunc::ReplaceTest, StencilFunc::Ones,
                                                StencilFunc::ReplaceOp, CompareFunc::Always, ref);
    EXPECT_TRUE(result.exact);
}

TEST(StencilReference, IgnoresUnreachableReplacementOperations) {
    const StencilRefMask ref{0x12, 0xff, 0xff, 0x34};
    const auto always = ResolveStencilReference(StencilFunc::Ones, StencilFunc::ReplaceOp,
                                                StencilFunc::Keep, CompareFunc::Always, ref);
    EXPECT_EQ(always.value, 0x34);
    EXPECT_TRUE(always.exact);
    const auto never = ResolveStencilReference(StencilFunc::ReplaceTest, StencilFunc::Ones,
                                               StencilFunc::ReplaceOp, CompareFunc::Never, ref);
    EXPECT_EQ(never.value, 0x12);
    EXPECT_TRUE(never.exact);
}

TEST(StencilReference, IgnoresOperationsExcludedByDepthTest) {
    const StencilRefMask ref{0x12, 0xff, 0xff, 0x34};
    const auto depth_pass =
        ResolveStencilReference(StencilFunc::Keep, StencilFunc::ReplaceOp, StencilFunc::Ones,
                                CompareFunc::Always, ref, CompareFunc::Always);
    EXPECT_EQ(depth_pass.value, 0x34);
    EXPECT_TRUE(depth_pass.exact);
    const auto depth_fail =
        ResolveStencilReference(StencilFunc::Keep, StencilFunc::Ones, StencilFunc::ReplaceTest,
                                CompareFunc::Always, ref, CompareFunc::Never);
    EXPECT_EQ(depth_fail.value, 0x12);
    EXPECT_TRUE(depth_fail.exact);
}
