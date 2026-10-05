// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/liverpool_to_vk.h"

using AmdGpu::ColorBuffer;
using AmdGpu::DataFormat;
using AmdGpu::NumberFormat;
using Vulkan::LiverpoolToVK::ColorBufferClearValue;

TEST(ColorClear, PackedUnormComponentsIgnoreShaderExportSwap) {
    ColorBuffer buffer{};
    buffer.info.format = u32(DataFormat::Format8_8_8_8);
    buffer.info.number_type = u32(NumberFormat::Unorm);
    buffer.clear_word0 = 0x44332211;
    for (const auto swap :
         {ColorBuffer::SwapMode::Standard, ColorBuffer::SwapMode::Alternate,
          ColorBuffer::SwapMode::StandardReverse, ColorBuffer::SwapMode::AlternateReverse}) {
        buffer.info.comp_swap = swap;
        const auto clear = ColorBufferClearValue(buffer).color.float32;
        for (u32 channel = 0; channel < 4; ++channel) {
            EXPECT_FLOAT_EQ(clear[channel], float(0x11 * (channel + 1)) / 255.0f);
        }
    }
}

TEST(ColorClear, ReversedMapClearKeepsAlphaInPhysicalFirstComponent) {
    ColorBuffer buffer{};
    buffer.info.format = u32(DataFormat::Format8_8_8_8);
    buffer.info.number_type = u32(NumberFormat::Unorm);
    buffer.info.comp_swap = ColorBuffer::SwapMode::StandardReverse;
    buffer.clear_word0 = 0xff;
    const auto clear = ColorBufferClearValue(buffer).color.float32;
    EXPECT_EQ(clear, (std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(ColorClear, PackedIntegerComponentsRemainBitExact) {
    ColorBuffer buffer{};
    buffer.info.format = u32(DataFormat::Format8_8_8_8);
    buffer.info.number_type = u32(NumberFormat::Uint);
    buffer.info.comp_swap = ColorBuffer::SwapMode::AlternateReverse;
    buffer.clear_word0 = 0x80402010;
    EXPECT_EQ(ColorBufferClearValue(buffer).color.uint32,
              (std::array<u32, 4>{0x10, 0x20, 0x40, 0x80}));
}

TEST(ColorClear, PackedHalfFloatComponentsKeepMemoryOrder) {
    ColorBuffer buffer{};
    buffer.info.format = u32(DataFormat::Format16_16_16_16);
    buffer.info.number_type = u32(NumberFormat::Float);
    buffer.info.comp_swap = ColorBuffer::SwapMode::StandardReverse;
    buffer.clear_word0 = 0x40003c00;
    buffer.clear_word1 = 0x44004200;
    EXPECT_EQ(ColorBufferClearValue(buffer).color.float32,
              (std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(ColorClear, SingleChannelClearDoesNotSelectUnusedExportComponent) {
    ColorBuffer buffer{};
    buffer.info.format = u32(DataFormat::Format8);
    buffer.info.number_type = u32(NumberFormat::Unorm);
    buffer.info.comp_swap = ColorBuffer::SwapMode::Alternate;
    buffer.clear_word0 = 0xff;
    EXPECT_FLOAT_EQ(ColorBufferClearValue(buffer).color.float32[0], 1.0f);
}

std::array<Common::Log::Level, Common::Log::NUM_LOG_CLASSES> Common::Log::g_class_levels{};

void Common::Log::VLog(Class, Level, const char* file, int line, const char*, fmt::string_view,
                       fmt::format_args) {
    ADD_FAILURE() << "Unexpected color-clear diagnostic at " << file << ':' << line;
}

void assert_fail_impl() {
    std::abort();
}

[[noreturn]] void unreachable_impl() {
    std::abort();
}
