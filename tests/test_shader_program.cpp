// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace {

Shader::StageSpecialization MakeSpec(const Shader::Info& info) {
    Shader::StageSpecialization spec;
    spec.info = &info;
    spec.runtime_info.Initialize(info.hw_stage, info.sw_stage);
    spec.buffers.resize(info.buffers.size());
    spec.images.resize(info.images.size());
    spec.samplers.resize(info.samplers.size());
    return spec;
}

std::unique_ptr<Shader::Info> MakeFragmentInfo(bool flatbuf) {
    auto info = std::make_unique<Shader::Info>();
    info->hw_stage = Shader::HwStage::Fragment;
    info->sw_stage = Shader::SwStage::Fragment;
    if (flatbuf) {
        info->buffers.push_back({.buffer_type = Shader::BufferType::Flatbuf});
    }
    info->images.emplace_back();
    info->samplers.emplace_back();
    return info;
}

} // namespace

TEST(ShaderProgram, KeepsDistinctResourceLayoutsAndStableMetadataForPermutations) {
    Vulkan::Program program;
    auto ordinary = MakeFragmentInfo(false);
    const auto* ordinary_ptr = ordinary.get();
    auto ordinary_spec = MakeSpec(*ordinary);
    program.AddPermut({}, std::move(ordinary_spec), std::move(ordinary));
    auto flatbuf = MakeFragmentInfo(true);
    const auto* flatbuf_ptr = flatbuf.get();
    auto flatbuf_spec = MakeSpec(*flatbuf);
    program.AddPermut({}, std::move(flatbuf_spec), std::move(flatbuf));

    // Force the permutation list out of its inline storage, as GT Sport does.
    for (size_t i = 0; i < Vulkan::Program::MaxPermutations; ++i) {
        auto info = MakeFragmentInfo(false);
        auto spec = MakeSpec(*info);
        program.AddPermut({}, std::move(spec), std::move(info));
    }
    EXPECT_EQ(program.modules[0].info.get(), ordinary_ptr);
    EXPECT_EQ(program.modules[1].info.get(), flatbuf_ptr);
    EXPECT_EQ(program.modules[0].spec.info, ordinary_ptr);
    EXPECT_EQ(program.modules[1].spec.info, flatbuf_ptr);

    Shader::Backend::Bindings ordinary_end;
    Shader::Backend::Bindings flatbuf_end;
    ordinary_ptr->AddBindings(ordinary_end);
    flatbuf_ptr->AddBindings(flatbuf_end);
    EXPECT_EQ(ordinary_end.unified, 2);
    EXPECT_EQ(flatbuf_end.unified, 3);
    EXPECT_EQ(flatbuf_end.buffer, 1);
}

TEST(ShaderProgram, RetainsEachPreloadedPermutationsMetadataAtItsOriginalIndex) {
    Vulkan::Program program;
    auto flatbuf = MakeFragmentInfo(true);
    const auto* flatbuf_ptr = flatbuf.get();
    auto flatbuf_spec = MakeSpec(*flatbuf);
    program.InsertPermut({}, std::move(flatbuf_spec), std::move(flatbuf), 5);
    auto ordinary = MakeFragmentInfo(false);
    const auto* ordinary_ptr = ordinary.get();
    auto ordinary_spec = MakeSpec(*ordinary);
    program.InsertPermut({}, std::move(ordinary_spec), std::move(ordinary), 0);

    EXPECT_EQ(program.modules[5].info.get(), flatbuf_ptr);
    EXPECT_EQ(program.modules[5].spec.info, flatbuf_ptr);
    EXPECT_EQ(program.modules[0].info.get(), ordinary_ptr);
    EXPECT_EQ(program.modules[0].spec.info, ordinary_ptr);
    EXPECT_FALSE(program.modules[1].spec.Valid());
    EXPECT_TRUE(program.modules[5].spec.Valid());
}

TEST(ShaderSpecialization, RejectsDifferentResourceCountsInEitherComparisonDirection) {
    auto ordinary = MakeFragmentInfo(false);
    auto flatbuf = MakeFragmentInfo(true);
    auto ordinary_spec = MakeSpec(*ordinary);
    auto flatbuf_spec = MakeSpec(*flatbuf);
    EXPECT_FALSE(ordinary_spec == flatbuf_spec);
    EXPECT_FALSE(flatbuf_spec == ordinary_spec);

    auto extra_image = ordinary_spec;
    extra_image.images.emplace_back();
    EXPECT_FALSE(ordinary_spec == extra_image);
    EXPECT_FALSE(extra_image == ordinary_spec);
    auto extra_sampler = ordinary_spec;
    extra_sampler.samplers.emplace_back();
    EXPECT_FALSE(ordinary_spec == extra_sampler);
    EXPECT_FALSE(extra_sampler == ordinary_spec);
    Shader::StageSpecialization unloaded;
    EXPECT_FALSE(ordinary_spec == unloaded);
    EXPECT_FALSE(unloaded == ordinary_spec);
}

TEST(ShaderSpecialization, NewlyActiveResourcesMustMatchTheCompiledPermutation) {
    auto info = MakeFragmentInfo(false);
    auto compiled = MakeSpec(*info);
    compiled.images[0].type = AmdGpu::ImageType::Color2D;
    auto current = compiled;
    current.bitset[0] = true;
    current.images[0].type = AmdGpu::ImageType::Color2DArray;
    EXPECT_FALSE(compiled == current);

    // An inactive current resource can retain the existing module's declaration.
    EXPECT_TRUE(current == compiled);
}

TEST(ShaderBindings, CountsMipFallbackDescriptorsWhenReusingAModule) {
    Shader::Info info;
    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.base_level = 2;
    image.last_level = 5;
    Shader::ImageResource resource;
    resource.mip_fallback_mode = Shader::MipStorageFallbackMode::DynamicIndex;
    std::memcpy(resource.sharp_fetch.immediates.data(), &image, sizeof(image));
    info.images.push_back(resource);
    info.images.emplace_back();
    info.buffers.emplace_back();
    info.samplers.emplace_back();
    Shader::Backend::Bindings binding{.unified = 7, .buffer = 3};
    info.AddBindings(binding);
    // A compiled module reserves four mip descriptors, one ordinary image, a buffer and sampler.
    EXPECT_EQ(binding.unified, 14);
    EXPECT_EQ(binding.buffer, 4);
}

// These focused header tests do not initialize the emulator's logging or fatal-error backend.
void Common::Log::VLog(Class, Level, const char* file, int line, const char*, fmt::string_view,
                       fmt::format_args) {
    ADD_FAILURE() << "Unexpected shader metadata assertion at " << file << ':' << line;
}

void assert_fail_impl() {
    std::abort();
}
