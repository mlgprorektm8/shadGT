// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/amdgpu/resource.h"

TEST(BufferResource, ComputesLargeStridedRangesWithoutOverflow) {
    AmdGpu::Buffer buffer{};
    buffer.stride = 64;
    buffer.num_records = 0xffffffff;
    EXPECT_EQ(buffer.GetSize(), 0x3fffffffc0ULL);
    EXPECT_EQ(buffer.NumDwords(), 0xfffffffF0ULL);
    buffer.num_records = 0x04000000;
    EXPECT_EQ(buffer.GetSize(), 0x100000000ULL);
}

TEST(BufferResource, RoundsRawBufferDwordCountWithoutOverflow) {
    AmdGpu::Buffer buffer{};
    buffer.num_records = 0xffffffff;
    EXPECT_EQ(buffer.GetSize(), 0xffffffffULL);
    EXPECT_EQ(buffer.NumDwords(), 0x40000000ULL);
    buffer.num_records = 5;
    EXPECT_EQ(buffer.NumDwords(), 2);
    buffer.num_records = 0;
    EXPECT_EQ(buffer.GetSize(), 0);
    EXPECT_EQ(buffer.NumDwords(), 0);
}
