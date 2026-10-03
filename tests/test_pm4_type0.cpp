// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>
#include "video_core/amdgpu/pm4_type0.h"

TEST(PM4Type0, DecodesConsecutiveRegisterWritesWithoutConsumingNextPacket) {
    const std::array<u32, 5> words{0x0002a000, 0x1234, 0x5678, 0x9abc, 0x80000000};
    const auto write = AmdGpu::DecodeType0RegisterWrite(words, 0xd000);
    ASSERT_TRUE(write.has_value());
    EXPECT_EQ(write->first_register, 0xa000);
    EXPECT_EQ(write->values.size(), 3);
    EXPECT_EQ(write->values.front(), 0x1234);
    EXPECT_EQ(write->values.back(), 0x9abc);
    EXPECT_EQ(words[write->values.size() + 1], 0x80000000);
}

TEST(PM4Type0, ZeroCountMeansOneRegisterIncludingLastRegister) {
    const std::array<u32, 2> words{0x0000cfff, 0xdeadbeef};
    const auto write = AmdGpu::DecodeType0RegisterWrite(words, 0xd000);
    ASSERT_TRUE(write.has_value());
    EXPECT_EQ(write->first_register, 0xcfff);
    EXPECT_EQ(write->values.size(), 1);
    EXPECT_EQ(write->values.front(), 0xdeadbeef);
}

TEST(PM4Type0, RejectsTruncatedAndOutOfRangePackets) {
    EXPECT_FALSE(AmdGpu::DecodeType0RegisterWrite({}, 0xd000));
    const std::array<u32, 2> truncated{0x00010003, 42};
    EXPECT_FALSE(AmdGpu::DecodeType0RegisterWrite(truncated, 0xd000));
    const std::array<u32, 3> overflow{0x0001cfff, 42, 43};
    EXPECT_FALSE(AmdGpu::DecodeType0RegisterWrite(overflow, 0xd000));
    const std::array<u32, 2> outside{0x0000d000, 42};
    EXPECT_FALSE(AmdGpu::DecodeType0RegisterWrite(outside, 0xd000));
    const std::array<u32, 2> wrong_type{0xc0001000, 42};
    EXPECT_FALSE(AmdGpu::DecodeType0RegisterWrite(wrong_type, 0xd000));
}
