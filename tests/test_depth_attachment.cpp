// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/amdgpu/regs_depth.h"
#include "video_core/renderer_vulkan/depth_attachment.h"

TEST(DepthAttachment, NarrowReadOnlyFarRangeDoesNotRoundSkyToClearDepth) {
    AmdGpu::DepthControl control{};
    control.depth_enable = 1;
    control.depth_func = AmdGpu::CompareFunc::Less;
    const float minimum = 1.f - 4.f * std::numeric_limits<float>::epsilon();
    const float maximum = Vulkan::ReadOnlyFarDepthMax(minimum, 1.f, control);
    EXPECT_FLOAT_EQ(maximum, std::nextafter(1.f, 0.f));
    const float sky_ndc = 0.99f;
    const auto mapped_depth = [minimum, sky_ndc](float far) {
        return (far + minimum) * 0.5f + sky_ndc * ((far - minimum) * 0.5f);
    };
    EXPECT_FLOAT_EQ(mapped_depth(1.f), 1.f);
    EXPECT_LT(mapped_depth(maximum), 1.f);
    EXPECT_GE(mapped_depth(maximum), minimum);
}

TEST(DepthAttachment, FarRangePrecisionWorkaroundLeavesOtherPassesUnchanged) {
    AmdGpu::DepthControl control{};
    control.depth_enable = 1;
    control.depth_func = AmdGpu::CompareFunc::Less;
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(0.f, 1.f, control), 1.f);
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(0.999f, 1.f, control), 1.f);
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(1.f, 1.f, control), 1.f);
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(1.f, 0.f, control), 0.f);
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(0.9999995f, 0.9f, control), 0.9f);
    control.depth_write_enable = 1;
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(0.9999995f, 1.f, control), 1.f);
    control.depth_write_enable = 0;
    control.depth_enable = 0;
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(0.9999995f, 1.f, control), 1.f);
    control.depth_enable = 1;
    control.stencil_enable = 1;
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(0.9999995f, 1.f, control), 1.f);
    control.stencil_enable = 0;
    control.depth_bounds_enable = 1;
    EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(0.9999995f, 1.f, control), 1.f);
    control.depth_bounds_enable = 0;
    for (const auto function : {AmdGpu::CompareFunc::LessEqual, AmdGpu::CompareFunc::Equal,
                                AmdGpu::CompareFunc::Greater, AmdGpu::CompareFunc::Always}) {
        control.depth_func = function;
        EXPECT_FLOAT_EQ(Vulkan::ReadOnlyFarDepthMax(0.9999995f, 1.f, control), 1.f);
    }
}

TEST(DepthAttachment, IgnoresFormatsWithoutBackingAddresses) {
    AmdGpu::DepthBuffer buffer{};
    buffer.z_info.format = AmdGpu::DepthBuffer::ZFormat::Z32Float;
    buffer.stencil_info.format = AmdGpu::DepthBuffer::StencilFormat::Stencil8;
    EXPECT_FALSE(buffer.DepthValid());
    EXPECT_FALSE(buffer.StencilValid());
    buffer.z_read_base = 0x100;
    EXPECT_TRUE(buffer.DepthValid());
    EXPECT_FALSE(buffer.StencilValid());
    buffer.stencil_read_base = 0x200;
    EXPECT_TRUE(buffer.StencilValid());
    EXPECT_FALSE(buffer.StencilWriteValid());
    buffer.stencil_write_base = 0x300;
    EXPECT_TRUE(buffer.StencilWriteValid());
}

TEST(DepthAttachment, MetadataClearIncludesStencilWithStencilTestingDisabled) {
    AmdGpu::DepthBuffer buffer{};
    buffer.z_info.format = AmdGpu::DepthBuffer::ZFormat::Z32Float;
    buffer.z_read_base = 0x100;
    buffer.stencil_info.format = AmdGpu::DepthBuffer::StencilFormat::Stencil8;
    buffer.stencil_read_base = 0x200;
    const AmdGpu::DepthControl control{};
    const AmdGpu::DepthRenderControl render_control{};
    EXPECT_EQ(Vulkan::DepthStencilClearAspects(buffer, control, render_control, true),
              vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);
    EXPECT_EQ(Vulkan::DepthStencilClearAspects(buffer, control, render_control, false),
              vk::ImageAspectFlags{});
}

TEST(DepthAttachment, ExplicitDepthClearPreservesStencil) {
    AmdGpu::DepthBuffer buffer{};
    buffer.z_info.format = AmdGpu::DepthBuffer::ZFormat::Z32Float;
    buffer.z_read_base = 0x100;
    buffer.stencil_info.format = AmdGpu::DepthBuffer::StencilFormat::Stencil8;
    buffer.stencil_read_base = 0x200;
    AmdGpu::DepthControl control{};
    control.depth_enable = 1;
    control.depth_write_enable = 1;
    AmdGpu::DepthRenderControl render_control{};
    render_control.depth_clear_enable = 1;
    EXPECT_EQ(Vulkan::DepthStencilClearAspects(buffer, control, render_control, false),
              vk::ImageAspectFlagBits::eDepth);
    control.depth_write_enable = 0;
    EXPECT_EQ(Vulkan::DepthStencilClearAspects(buffer, control, render_control, false),
              vk::ImageAspectFlags{});
    render_control.stencil_clear_enable = 1;
    EXPECT_EQ(Vulkan::DepthStencilClearAspects(buffer, control, render_control, false),
              vk::ImageAspectFlagBits::eStencil);
}

TEST(DepthAttachment, MetadataClearOnlyIncludesValidAspects) {
    AmdGpu::DepthBuffer buffer{};
    buffer.z_info.format = AmdGpu::DepthBuffer::ZFormat::Z32Float;
    buffer.stencil_info.format = AmdGpu::DepthBuffer::StencilFormat::Stencil8;
    EXPECT_EQ(Vulkan::DepthStencilClearAspects(buffer, {}, {}, true), vk::ImageAspectFlags{});
    buffer.z_read_base = 0x100;
    EXPECT_EQ(Vulkan::DepthStencilClearAspects(buffer, {}, {}, true),
              vk::ImageAspectFlagBits::eDepth);
    buffer.stencil_read_base = 0x200;
    buffer.z_info.format = AmdGpu::DepthBuffer::ZFormat::Invalid;
    EXPECT_EQ(Vulkan::DepthStencilClearAspects(buffer, {}, {}, true),
              vk::ImageAspectFlagBits::eStencil);
}

TEST(DepthAttachment, SamplingAndWritesShareFeedbackLayout) {
    for (const bool stencil : {false, true}) {
        EXPECT_EQ(Vulkan::DepthAttachmentLayout(stencil, true, false, true, true),
                  vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT);
        EXPECT_EQ(Vulkan::DepthAttachmentLayout(stencil, true, false, true, false),
                  vk::ImageLayout::eGeneral);
    }
    EXPECT_EQ(Vulkan::DepthAttachmentLayout(true, false, true, true, true),
              vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT);
    EXPECT_EQ(Vulkan::DepthAttachmentLayout(true, false, true, false, true),
              vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal);
    EXPECT_EQ(Vulkan::DepthAttachmentLayout(false, false, false, true, true),
              vk::ImageLayout::eDepthReadOnlyOptimal);
    EXPECT_EQ(Vulkan::DepthAttachmentLayout(true, false, false, true, true),
              vk::ImageLayout::eDepthStencilReadOnlyOptimal);
}

TEST(DepthAttachment, DepthFormatChangesRequireBufferEvenWithMaintenance8) {
    for (const bool maintenance8 : {false, true}) {
        EXPECT_TRUE(Vulkan::DepthCopyNeedsBuffer(
            vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, true, true, maintenance8));
        EXPECT_TRUE(Vulkan::DepthCopyNeedsBuffer(vk::Format::eD32SfloatS8Uint,
                                                 vk::Format::eD32Sfloat, true, true, maintenance8));
        EXPECT_FALSE(Vulkan::DepthCopyNeedsBuffer(vk::Format::eD32Sfloat, vk::Format::eD32Sfloat,
                                                  true, true, maintenance8));
    }
    EXPECT_TRUE(Vulkan::DepthCopyNeedsBuffer(vk::Format::eR32Sfloat, vk::Format::eD32Sfloat, false,
                                             true, false));
    EXPECT_FALSE(Vulkan::DepthCopyNeedsBuffer(vk::Format::eR32Sfloat, vk::Format::eD32Sfloat, false,
                                              true, true));
}
