// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/amdgpu/pixel_format.h"

using AmdGpu::DataFormat;
using AmdGpu::MapNumberConversion;
using AmdGpu::NumberConversion;
using AmdGpu::NumberFormat;

// FIX-045: garbage T#s with SNORM_NZ on formats it is not defined for stopped the emulator.
TEST(NumberConversion, SnormNzOnOtherFormatsIsNoConversion) {
    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat::FormatBc3),
              NumberConversion::None);
    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat(25)), NumberConversion::None);
    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat::Format8_8_8_8),
              NumberConversion::Sint8ToSnormNz);
    EXPECT_EQ(MapNumberConversion(NumberFormat::SnormNz, DataFormat::Format16),
              NumberConversion::Sint16ToSnormNz);
}
