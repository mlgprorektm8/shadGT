// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <vector>
#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/resource_binding.h"
#include "video_core/texture_cache/image_descriptor.h"

#include <cstdlib>
#include "common/assert.h"
#include "common/logging/log.h"

// pixel_format.cpp reports through the logger and assertion handlers.
void Common::Log::VLog(Class, Level, const char* file, int line, const char*, fmt::string_view,
                       fmt::format_args) {
    ADD_FAILURE() << "Unexpected diagnostic at " << file << ':' << line;
}

void assert_fail_impl() {
    std::abort();
}

[[noreturn]] void unreachable_impl() {
    std::abort();
}

TEST(ResourceBinding, MalformedMipRangeUsesOneBindingBeforeStageCompilation) {
    Shader::Info stage;
    stage.buffers.emplace_back();
    auto& image = stage.images.emplace_back();
    image.mip_fallback_mode = Shader::MipStorageFallbackMode::DynamicIndex;
    auto sharp = AmdGpu::Image::Null(false);
    sharp.base_level = 15;
    sharp.last_level = 0;
    image.sharp_fetch.immediates = std::bit_cast<std::array<u32, 8>>(sharp);
    stage.images.emplace_back();
    stage.samplers.emplace_back();
    EXPECT_EQ(image.NumBindings(stage), 1u);
    Shader::Backend::Bindings compiled{.unified = 7, .buffer = 3};
    stage.AddBindings(compiled);
    EXPECT_EQ(compiled, (Shader::Backend::Bindings{.unified = 11, .buffer = 4}));

    u32 emitted{};
    EXPECT_EQ(Vulkan::ForEachImageDescriptorBinding(stage, image,
                                                    [&](bool is_written) {
                                                        EXPECT_FALSE(is_written);
                                                        ++emitted;
                                                    }),
              1u);
    EXPECT_EQ(emitted, 1u);
}

TEST(ResourceBinding, RejectedMipArrayPreservesNullSlotsAndFollowingBindings) {
    Shader::Info stage;
    stage.buffers.emplace_back();
    auto& storage = stage.images.emplace_back();
    storage.is_written = true;
    storage.mip_fallback_mode = Shader::MipStorageFallbackMode::DynamicIndex;
    auto sharp = AmdGpu::Image::Null(false);
    sharp.type = u64(AmdGpu::ImageType::Cube);
    sharp.width = 1023;
    sharp.height = 1023;
    sharp.depth = 5461;
    sharp.pow2pad = true;
    sharp.base_level = 2;
    sharp.last_level = 5;
    storage.sharp_fetch.immediates = std::bit_cast<std::array<u32, 8>>(sharp);
    stage.images.emplace_back();
    stage.samplers.emplace_back();
    vk::PhysicalDeviceLimits limits{};
    limits.maxImageArrayLayers = 65536;
    EXPECT_EQ(VideoCore::CheckImageDescriptorGeometry(sharp, limits),
              VideoCore::ImageDescriptorGeometryError::UnrepresentableLayers);

    const auto plan = Vulkan::PlanStageResourceBindings(stage, {.unified = 7, .buffer = 3});
    auto runtime_binding = plan.textures;
    std::vector<bool> null_storage_types;
    for (const auto& image : stage.images) {
        runtime_binding.unified += Vulkan::ForEachImageDescriptorBinding(
            stage, image, [&](bool is_written) { null_storage_types.push_back(is_written); });
    }
    // Four rejected storage mip descriptors remain reserved before the ordinary image.
    EXPECT_EQ(null_storage_types, (std::vector<bool>{true, true, true, true, false}));
    EXPECT_EQ(runtime_binding.unified, 13u);
    runtime_binding.unified += stage.samplers.size();
    EXPECT_EQ(runtime_binding, plan.next);
    EXPECT_EQ(plan.next, (Shader::Backend::Bindings{.unified = 14, .buffer = 4}));
}

TEST(ResourceBinding, BufferFirstPassPreservesStageAndMipArrayBindingNumbers) {
    Shader::Info fragment;
    fragment.buffers.resize(2);
    fragment.images.emplace_back();
    auto& storage = fragment.images.emplace_back();
    storage.is_written = true;
    storage.mip_fallback_mode = Shader::MipStorageFallbackMode::DynamicIndex;
    auto sharp = AmdGpu::Image::Null(false);
    sharp.base_level = 1;
    sharp.last_level = 3;
    storage.sharp_fetch.immediates = std::bit_cast<std::array<u32, 8>>(sharp);
    fragment.samplers.emplace_back();

    const auto fs = Vulkan::PlanStageResourceBindings(fragment, {.unified = 9, .buffer = 4});
    EXPECT_EQ(fs.buffers, (Shader::Backend::Bindings{.unified = 9, .buffer = 4}));
    EXPECT_EQ(fs.textures, (Shader::Backend::Bindings{.unified = 11, .buffer = 6}));
    // Two buffers, one ordinary image, three mip-array slots, and one sampler.
    EXPECT_EQ(fs.next, (Shader::Backend::Bindings{.unified = 16, .buffer = 6}));

    Shader::Info vertex;
    vertex.buffers.emplace_back();
    vertex.images.emplace_back();
    const auto vs = Vulkan::PlanStageResourceBindings(vertex, fs.next);
    EXPECT_EQ(vs.buffers, fs.next);
    EXPECT_EQ(vs.textures, (Shader::Backend::Bindings{.unified = 17, .buffer = 7}));
    EXPECT_EQ(vs.next, (Shader::Backend::Bindings{.unified = 18, .buffer = 7}));
}

TEST(ResourceBinding, EmptyStageDoesNotAdvanceEitherBindingCounter) {
    Shader::Info empty;
    const Shader::Backend::Bindings start{.unified = 13, .buffer = 5};
    const auto bindings = Vulkan::PlanStageResourceBindings(empty, start);
    EXPECT_EQ(bindings.buffers, start);
    EXPECT_EQ(bindings.textures, start);
    EXPECT_EQ(bindings.next, start);
}

TEST(ResourceBinding, SharedSampledAndStorageViewsUseGeneralWithBothShaderAccesses) {
    // The fragment stage samples a whole mip chain and a later stage stores one mip.
    // The requirements are combined per physical backing before its full transition.
    const auto state = Vulkan::FinalImageBindingState({.is_storage = true});
    EXPECT_EQ(state.layout, vk::ImageLayout::eGeneral);
    EXPECT_TRUE(state.stages & vk::PipelineStageFlagBits2::eAllGraphics);
    EXPECT_TRUE(state.access & vk::AccessFlagBits2::eShaderRead);
    EXPECT_TRUE(state.access & vk::AccessFlagBits2::eShaderWrite);
}

TEST(ResourceBinding, SampledColorAttachmentKeepsShaderAndAttachmentAccess) {
    for (const bool supported : {false, true}) {
        const auto state = Vulkan::FinalImageBindingState({
            .color_attachment = true,
            .feedback_layout_supported = supported,
        });
        EXPECT_EQ(state.layout, supported ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                          : vk::ImageLayout::eGeneral);
        EXPECT_TRUE(state.stages & vk::PipelineStageFlagBits2::eAllGraphics);
        EXPECT_TRUE(state.stages & vk::PipelineStageFlagBits2::eColorAttachmentOutput);
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eShaderRead);
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eColorAttachmentRead);
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eColorAttachmentWrite);
    }
}

TEST(ResourceBinding, SampledDepthStencilWritesKeepShaderAndAttachmentAccess) {
    for (const bool depth_write : {false, true}) {
        const auto state = Vulkan::FinalImageBindingState({
            .is_depth = true,
            .depth_attachment = true,
            .has_stencil = true,
            .depth_write = depth_write,
            .stencil_write = !depth_write,
            .feedback_layout_supported = true,
        });
        EXPECT_EQ(state.layout, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT);
        EXPECT_TRUE(state.stages & vk::PipelineStageFlagBits2::eAllGraphics);
        EXPECT_TRUE(state.stages & vk::PipelineStageFlagBits2::eEarlyFragmentTests);
        EXPECT_TRUE(state.stages & vk::PipelineStageFlagBits2::eLateFragmentTests);
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eShaderRead);
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eDepthStencilAttachmentRead);
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
    }
}

TEST(ResourceBinding, ReadOnlyDepthAttachmentRetainsStoreWriteAccess) {
    for (const bool stencil : {false, true}) {
        const auto state = Vulkan::FinalImageBindingState({
            .is_depth = true,
            .depth_attachment = true,
            .has_stencil = stencil,
        });
        EXPECT_EQ(state.layout, stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                        : vk::ImageLayout::eDepthReadOnlyOptimal);
        EXPECT_TRUE(state.stages & vk::PipelineStageFlagBits2::eLateFragmentTests);
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eShaderRead);
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eDepthStencilAttachmentRead);
        // STORE is performed at EndRendering even when both guest write enables are off.
        EXPECT_TRUE(state.access & vk::AccessFlagBits2::eDepthStencilAttachmentWrite);
        EXPECT_FALSE(state.access & vk::AccessFlagBits2::eShaderWrite);
    }
}

TEST(ResourceBinding, ComputeStorageUsesComputeReadAndWriteAccess) {
    const auto state = Vulkan::FinalImageBindingState({.is_storage = true, .is_compute = true});
    EXPECT_EQ(state.layout, vk::ImageLayout::eGeneral);
    EXPECT_EQ(state.stages, vk::PipelineStageFlagBits2::eComputeShader);
    EXPECT_TRUE(state.access & vk::AccessFlagBits2::eShaderRead);
    EXPECT_TRUE(state.access & vk::AccessFlagBits2::eShaderWrite);
}
