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

namespace {

using BlendFactor = AmdGpu::BlendControl::BlendFactor;
using BlendFunc = AmdGpu::BlendControl::BlendFunc;

AmdGpu::BlendControl Blend(BlendFactor color_src, BlendFactor color_dst, BlendFunc color_func,
                           BlendFactor alpha_src, BlendFactor alpha_dst, BlendFunc alpha_func,
                           bool separate) {
    AmdGpu::BlendControl control{};
    control.enable = 1;
    control.separate_alpha_blend = separate;
    control.color_src_factor = color_src;
    control.color_dst_factor = color_dst;
    control.color_func = color_func;
    control.alpha_src_factor = alpha_src;
    control.alpha_dst_factor = alpha_dst;
    control.alpha_func = alpha_func;
    return control;
}

// GCN factor for one logical channel; SrcColor on the alpha channel reads source alpha.
float GuestFactor(BlendFactor factor, const std::array<float, 4>& source, u32 channel) {
    switch (factor) {
    case BlendFactor::Zero:
        return 0.f;
    case BlendFactor::One:
        return 1.f;
    case BlendFactor::SrcColor:
        return source[channel];
    case BlendFactor::OneMinusSrcColor:
        return 1.f - source[channel];
    case BlendFactor::SrcAlpha:
        return source[3];
    case BlendFactor::OneMinusSrcAlpha:
        return 1.f - source[3];
    default:
        ADD_FAILURE() << "Unexpected guest blend factor";
        return 0.f;
    }
}

float ApplyFunc(BlendFunc func, float source, float destination) {
    switch (func) {
    case BlendFunc::Add:
        return source + destination;
    case BlendFunc::Subtract:
        return source - destination;
    case BlendFunc::ReverseSubtract:
        return destination - source;
    default:
        ADD_FAILURE() << "Unexpected blend function";
        return 0.f;
    }
}

float HostFactor(vk::BlendFactor factor, const std::array<float, 4>& secondary, u32 channel) {
    switch (factor) {
    case vk::BlendFactor::eOne:
        return 1.f;
    case vk::BlendFactor::eOneMinusSrc1Color:
        return 1.f - secondary[channel];
    case vk::BlendFactor::eOneMinusSrc1Alpha:
        return 1.f - secondary[3];
    default:
        ADD_FAILURE() << "Unexpected host blend factor";
        return 0.f;
    }
}

float HostFunc(vk::BlendOp op, float source, float destination) {
    switch (op) {
    case vk::BlendOp::eAdd:
        return source + destination;
    case vk::BlendOp::eSubtract:
        return source - destination;
    case vk::BlendOp::eReverseSubtract:
        return destination - source;
    default:
        ADD_FAILURE() << "Unexpected host blend op";
        return 0.f;
    }
}

} // namespace

TEST(ColorBlend, SwizzledFactorBlendMatchesLogicalBlending) {
    using Swap = AmdGpu::ColorBuffer::SwapMode;
    const std::array controls{
        // Destination-alpha preservation over an alpha-blended color.
        Blend(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendFunc::Add,
              BlendFactor::Zero, BlendFactor::One, BlendFunc::Add, true),
        // Classic alpha blending shared by all channels.
        Blend(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendFunc::Add,
              BlendFactor::Zero, BlendFactor::Zero, BlendFunc::Add, false),
        // Premultiplied alpha.
        Blend(BlendFactor::One, BlendFactor::OneMinusSrcAlpha, BlendFunc::Add, BlendFactor::Zero,
              BlendFactor::Zero, BlendFunc::Add, false),
        // Additive color with preserved destination alpha.
        Blend(BlendFactor::SrcAlpha, BlendFactor::One, BlendFunc::Add, BlendFactor::Zero,
              BlendFactor::One, BlendFunc::Add, true),
        Blend(BlendFactor::SrcAlpha, BlendFactor::One, BlendFunc::Subtract, BlendFactor::Zero,
              BlendFactor::Zero, BlendFunc::Add, false),
        Blend(BlendFactor::OneMinusSrcColor, BlendFactor::SrcColor, BlendFunc::ReverseSubtract,
              BlendFactor::One, BlendFactor::OneMinusSrcAlpha, BlendFunc::ReverseSubtract, true),
    };
    for (const auto swap : {Swap::StandardReverse, Swap::AlternateReverse}) {
        AmdGpu::ColorBuffer buffer{};
        buffer.info.format = u32(AmdGpu::DataFormat::Format8_8_8_8);
        buffer.info.comp_swap = swap;
        const auto swizzle = buffer.Swizzle();
        for (u32 index = 0; index < controls.size(); ++index) {
            const auto& control = controls[index];
            ASSERT_TRUE(Vulkan::LiverpoolToVK::IsLaneDependentSwizzledBlend(swizzle, control))
                << "control=" << index;
            ASSERT_FALSE(Vulkan::LiverpoolToVK::NeedsSwizzledAlphaBlend(swizzle, control));
            const auto blend = Vulkan::LiverpoolToVK::GetSwizzledFactorBlend(swizzle, control);
            ASSERT_TRUE(blend.has_value()) << "control=" << index;
            vk::PipelineColorBlendAttachmentState attachment{};
            Vulkan::LiverpoolToVK::SetSwizzledFactorBlend(attachment, blend->color_func);
            for (const auto alpha : {0.f, 0.25f, 0.8980392f, 1.f}) {
                for (const auto destination :
                     {std::array{0.f, 0.f, 0.f, 1.f}, std::array{0.15f, 0.3f, 0.7f, 0.4f}}) {
                    const std::array source{1.f, 0.6f, 0.2f, alpha};
                    // Logical guest result, and the emulated shader outputs in logical order
                    // before the export swizzle.
                    std::array<float, 4> logical_result{};
                    std::array<float, 4> primary{};
                    std::array<float, 4> secondary{};
                    for (u32 channel = 0; channel < 4; ++channel) {
                        const bool is_alpha = channel == 3;
                        const auto src = is_alpha ? blend->alpha_src : blend->color_src;
                        const auto dst = is_alpha ? blend->alpha_dst : blend->color_dst;
                        const auto func = is_alpha ? blend->alpha_func : blend->color_func;
                        const float src_factor = GuestFactor(src, source, channel);
                        const float dst_factor = GuestFactor(dst, source, channel);
                        logical_result[channel] = ApplyFunc(func, source[channel] * src_factor,
                                                            destination[channel] * dst_factor);
                        primary[channel] = source[channel] * src_factor;
                        secondary[channel] = 1.f - dst_factor;
                    }
                    const auto stored_primary = swizzle.Apply(primary);
                    const auto stored_secondary = swizzle.Apply(secondary);
                    const auto stored_destination = swizzle.Apply(destination);
                    const auto expected = swizzle.Apply(logical_result);
                    for (u32 channel = 0; channel < 4; ++channel) {
                        const bool is_alpha = channel == 3;
                        const auto src = is_alpha ? attachment.srcAlphaBlendFactor
                                                  : attachment.srcColorBlendFactor;
                        const auto dst = is_alpha ? attachment.dstAlphaBlendFactor
                                                  : attachment.dstColorBlendFactor;
                        const auto op =
                            is_alpha ? attachment.alphaBlendOp : attachment.colorBlendOp;
                        const auto actual = HostFunc(
                            op, stored_primary[channel] * HostFactor(src, stored_secondary, channel),
                            stored_destination[channel] *
                                HostFactor(dst, stored_secondary, channel));
                        EXPECT_NEAR(actual, expected[channel], 1e-6f)
                            << "swap=" << u32(swap) << " control=" << index << " alpha=" << alpha
                            << " channel=" << channel;
                    }
                }
            }
        }
    }
}

TEST(ColorBlend, SwizzledFactorBlendSelection) {
    AmdGpu::ColorBuffer buffer{};
    buffer.info.format = u32(AmdGpu::DataFormat::Format8_8_8_8);
    buffer.info.comp_swap = AmdGpu::ColorBuffer::SwapMode::StandardReverse;
    const auto swizzle = buffer.Swizzle();
    using Vulkan::LiverpoolToVK::GetSwizzledFactorBlend;
    using Vulkan::LiverpoolToVK::IsLaneDependentSwizzledBlend;

    // The confirmed exact equation keeps its existing emulation.
    EXPECT_TRUE(IsLaneDependentSwizzledBlend(swizzle, MapBlend()));
    EXPECT_FALSE(GetSwizzledFactorBlend(swizzle, MapBlend()).has_value());

    // Identity swizzles and lane-independent equations are already exact natively.
    const auto alpha_blend =
        Blend(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendFunc::Add,
              BlendFactor::Zero, BlendFactor::Zero, BlendFunc::Add, false);
    EXPECT_FALSE(IsLaneDependentSwizzledBlend(AmdGpu::IdentityMapping, alpha_blend));
    const auto additive = Blend(BlendFactor::One, BlendFactor::One, BlendFunc::Add,
                                BlendFactor::Zero, BlendFactor::Zero, BlendFunc::Add, false);
    EXPECT_FALSE(IsLaneDependentSwizzledBlend(swizzle, additive));
    const auto modulate = Blend(BlendFactor::DstColor, BlendFactor::Zero, BlendFunc::Add,
                                BlendFactor::Zero, BlendFactor::Zero, BlendFunc::Add, false);
    EXPECT_FALSE(IsLaneDependentSwizzledBlend(swizzle, modulate));
    auto disabled = alpha_blend;
    disabled.enable = 0;
    EXPECT_FALSE(IsLaneDependentSwizzledBlend(swizzle, disabled));

    // Lane-dependent equations outside the exact factor set are reported, not emulated.
    const auto dest_alpha =
        Blend(BlendFactor::DstAlpha, BlendFactor::OneMinusDstAlpha, BlendFunc::Add,
              BlendFactor::Zero, BlendFactor::Zero, BlendFunc::Add, false);
    EXPECT_TRUE(IsLaneDependentSwizzledBlend(swizzle, dest_alpha));
    EXPECT_FALSE(GetSwizzledFactorBlend(swizzle, dest_alpha).has_value());
    const auto mixed_funcs =
        Blend(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendFunc::Add,
              BlendFactor::One, BlendFactor::One, BlendFunc::Max, true);
    EXPECT_TRUE(IsLaneDependentSwizzledBlend(swizzle, mixed_funcs));
    EXPECT_FALSE(GetSwizzledFactorBlend(swizzle, mixed_funcs).has_value());
    const auto min_blend = Blend(BlendFactor::SrcAlpha, BlendFactor::One, BlendFunc::Min,
                                 BlendFactor::Zero, BlendFactor::Zero, BlendFunc::Add, false);
    EXPECT_FALSE(GetSwizzledFactorBlend(swizzle, min_blend).has_value());
    const auto dual_source =
        Blend(BlendFactor::SrcAlpha, BlendFactor::InvSrc1Alpha, BlendFunc::Add,
              BlendFactor::Zero, BlendFactor::Zero, BlendFunc::Add, false);
    EXPECT_FALSE(GetSwizzledFactorBlend(swizzle, dual_source).has_value());
}
