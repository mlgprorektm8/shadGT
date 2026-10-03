// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "shader_recompiler/resource.h"

TEST(ImageResource, SeparatesComparisonAndOrdinarySamplingOfSameSharp) {
    Shader::ImageResource ordinary{};
    ordinary.sharp_fetch.summary = Shader::SharpFetch<AmdGpu::Image>::Summary::SingleLoad;
    ordinary.sharp_fetch.offsets[0] = 16;
    auto comparison = ordinary;
    comparison.is_depth = true;
    EXPECT_FALSE(ordinary.HasSameBinding(comparison));
    EXPECT_FALSE(comparison.HasSameBinding(ordinary));
}

TEST(ImageResource, SharesReadAndWriteUsageOfSameImageBinding) {
    Shader::ImageResource read{};
    auto write = read;
    write.is_written = true;
    write.is_atomic = true;
    EXPECT_TRUE(read.HasSameBinding(write));
}

TEST(ImageResource, SeparatesMipFallbackBindings) {
    Shader::ImageResource first{};
    first.mip_fallback_mode = Shader::MipStorageFallbackMode::ConstantIndex;
    auto second = first;
    second.constant_mip_index = 1;
    EXPECT_FALSE(first.HasSameBinding(second));
}
