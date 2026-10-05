// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/liverpool_to_vk.h"

namespace {

AmdGpu::BlendControl MapBlend() {
    using Factor = AmdGpu::BlendControl::BlendFactor;
    AmdGpu::BlendControl control{};
    control.enable = 1;
    control.separate_alpha_blend = 1;
    control.color_src_factor = Factor::SrcAlpha;
    control.color_dst_factor = Factor::OneMinusSrcAlpha;
    control.alpha_src_factor = Factor::Zero;
    control.alpha_dst_factor = Factor::OneMinusSrcAlpha;
    return control;
}

float Factor(vk::BlendFactor factor, const std::array<float, 4>& secondary, u32 channel) {
    switch (factor) {
    case vk::BlendFactor::eSrc1Color:
        return secondary[channel];
    case vk::BlendFactor::eSrc1Alpha:
        return secondary[3];
    case vk::BlendFactor::eOneMinusSrc1Alpha:
        return 1.f - secondary[3];
    default:
        ADD_FAILURE() << "Unexpected dual-source blend factor";
        return 0.f;
    }
}

} // namespace

TEST(ColorBlend, SwizzledSourceAlphaMatchesLogicalBlending) {
    using Swap = AmdGpu::ColorBuffer::SwapMode;
    for (const auto swap : {Swap::StandardReverse, Swap::AlternateReverse}) {
        AmdGpu::ColorBuffer buffer{};
        buffer.info.format = u32(AmdGpu::DataFormat::Format8_8_8_8);
        buffer.info.comp_swap = swap;
        const auto swizzle = buffer.Swizzle();
        ASSERT_TRUE(Vulkan::LiverpoolToVK::NeedsSwizzledAlphaBlend(swizzle, MapBlend()));
        vk::PipelineColorBlendAttachmentState attachment{};
        Vulkan::LiverpoolToVK::SetSwizzledAlphaBlend(attachment);
        EXPECT_EQ(attachment.colorBlendOp, vk::BlendOp::eAdd);
        EXPECT_EQ(attachment.alphaBlendOp, vk::BlendOp::eAdd);
        for (const auto alpha : {0.f, 0.25f, 0.8980392f, 1.f}) {
            for (const auto destination :
                 {std::array{0.f, 0.f, 0.f, 1.f}, std::array{0.15f, 0.3f, 0.7f, 0.4f}}) {
                const std::array source{1.f, 0.6f, 0.2f, alpha};
                const auto stored_source = swizzle.Apply(source);
                const auto stored_destination = swizzle.Apply(destination);
                const auto secondary = swizzle.Apply(std::array{alpha, alpha, alpha, 0.f});
                std::array<float, 4> logical_result{};
                for (u32 channel = 0; channel < 4; ++channel) {
                    logical_result[channel] = destination[channel] * (1.f - alpha);
                    if (channel != 3) {
                        logical_result[channel] += source[channel] * alpha;
                    }
                }
                const auto expected = swizzle.Apply(logical_result);
                for (u32 channel = 0; channel < 4; ++channel) {
                    const auto src = channel == 3 ? attachment.srcAlphaBlendFactor
                                                  : attachment.srcColorBlendFactor;
                    const auto dst = channel == 3 ? attachment.dstAlphaBlendFactor
                                                  : attachment.dstColorBlendFactor;
                    const auto actual =
                        stored_source[channel] * Factor(src, secondary, channel) +
                        stored_destination[channel] * Factor(dst, secondary, channel);
                    EXPECT_NEAR(actual, expected[channel], 1e-6f)
                        << "swap=" << u32(swap) << " alpha=" << alpha << " channel=" << channel;
                }
            }
        }
    }
}

TEST(ColorBlend, LeavesOtherSwizzlesAndBlendEquationsOnNativePath) {
    EXPECT_FALSE(
        Vulkan::LiverpoolToVK::NeedsSwizzledAlphaBlend(AmdGpu::IdentityMapping, MapBlend()));
    AmdGpu::ColorBuffer buffer{};
    buffer.info.format = u32(AmdGpu::DataFormat::Format8_8_8_8);
    buffer.info.comp_swap = AmdGpu::ColorBuffer::SwapMode::StandardReverse;
    auto control = MapBlend();
    control.alpha_dst_factor = AmdGpu::BlendControl::BlendFactor::One;
    EXPECT_FALSE(Vulkan::LiverpoolToVK::NeedsSwizzledAlphaBlend(buffer.Swizzle(), control));
    control = MapBlend();
    control.enable = 0;
    EXPECT_FALSE(Vulkan::LiverpoolToVK::NeedsSwizzledAlphaBlend(buffer.Swizzle(), control));
    control = MapBlend();
    control.color_src_factor = AmdGpu::BlendControl::BlendFactor::Src1Alpha;
    EXPECT_FALSE(Vulkan::LiverpoolToVK::NeedsSwizzledAlphaBlend(buffer.Swizzle(), control));
}
