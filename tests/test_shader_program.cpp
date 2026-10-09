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

namespace {

// FIX-043: specializations built the way the pipeline cache builds them, from a translated
// shader's Info and the V#/T#s bound for a draw.
AmdGpu::Buffer BoundBuffer(u32 stride) {
    AmdGpu::Buffer buffer{};
    buffer.base_address = 0x10000;
    buffer.num_records = 256;
    buffer.stride = stride;
    return buffer;
}

void Bind(Shader::BufferResource& resource, const AmdGpu::Buffer& buffer) {
    std::memcpy(resource.sharp_fetch.immediates.data(), &buffer, sizeof(buffer));
}

void Bind(Shader::ImageResource& resource, const AmdGpu::Image& image) {
    std::memcpy(resource.sharp_fetch.immediates.data(), &image, sizeof(image));
}

Shader::StageSpecialization KeyFor(const Shader::Info& info) {
    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.hw_stage, info.sw_stage);
    return Shader::StageSpecialization(info, runtime_info, Shader::Profile{}, {});
}

Shader::Info TranslatedFragmentWithBuffer(bool usage_known) {
    Shader::Info info;
    info.hw_stage = Shader::HwStage::Fragment;
    info.sw_stage = Shader::SwStage::Fragment;
    info.buffers.emplace_back();
    info.resource_usage_known = usage_known;
    return info;
}

} // namespace

TEST(ShaderSpecialization, StrideOfABufferWithoutIndexedAddressingDoesNotSplitPermutations) {
    auto info = TranslatedFragmentWithBuffer(true);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto compiled = KeyFor(info);
    Bind(info.buffers[0], BoundBuffer(48));
    EXPECT_TRUE(compiled == KeyFor(info));

    // Indexed addressing multiplies by the stride, so it stays in the key.
    info.buffer_stride_used.set(0);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto indexed = KeyFor(info);
    Bind(info.buffers[0], BoundBuffer(48));
    EXPECT_FALSE(indexed == KeyFor(info));
}

TEST(ShaderSpecialization, KeepsEveryBufferPropertyWhenTranslationUsageIsUnknown) {
    // A shader loaded from storage carries no usage flags.
    auto info = TranslatedFragmentWithBuffer(false);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto compiled = KeyFor(info);
    Bind(info.buffers[0], BoundBuffer(48));
    EXPECT_FALSE(compiled == KeyFor(info));
}

TEST(ShaderSpecialization, ThreadIdAddressingAndCoherenceSplitPermutations) {
    auto info = TranslatedFragmentWithBuffer(true);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto compiled = KeyFor(info);

    auto add_tid = BoundBuffer(16);
    add_tid.add_tid_enable = 1;
    Bind(info.buffers[0], add_tid);
    EXPECT_FALSE(compiled == KeyFor(info));

    auto coherent = BoundBuffer(16);
    coherent.mtype = 3;
    Bind(info.buffers[0], coherent);
    EXPECT_FALSE(compiled == KeyFor(info));
}

TEST(ShaderSpecialization, UnboundBufferIsComparedWithTheDescriptorTranslationSaw) {
    auto info = TranslatedFragmentWithBuffer(true);
    // Translated while the slot held an empty, unswizzled V#.
    auto empty = BoundBuffer(16);
    empty.num_records = 0;
    Bind(info.buffers[0], empty);
    const auto compiled = KeyFor(info);
    EXPECT_FALSE(compiled.bitset[0]);
    Bind(info.buffers[0], BoundBuffer(64));
    EXPECT_TRUE(compiled == KeyFor(info));

    // Swizzled addressing in the empty V# went into the code, so a plain V# needs new code.
    auto swizzled = empty;
    swizzled.swizzle_enable = 1;
    Bind(info.buffers[0], swizzled);
    const auto compiled_swizzled = KeyFor(info);
    Bind(info.buffers[0], BoundBuffer(64));
    EXPECT_FALSE(compiled_swizzled == KeyFor(info));

    // A typed buffer read through a null V# returns zeros; a bound one needs new code.
    info.buffers[0].is_formatted = true;
    Bind(info.buffers[0], AmdGpu::Buffer::Null());
    const auto compiled_typed = KeyFor(info);
    auto typed = BoundBuffer(16);
    typed.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32);
    Bind(info.buffers[0], typed);
    EXPECT_FALSE(compiled_typed == KeyFor(info));
}

TEST(ShaderSpecialization, SrgbFormatSplitsPermutationsOnlyWhenItDecidesADegamma) {
    Shader::Info info;
    info.hw_stage = Shader::HwStage::Fragment;
    info.sw_stage = Shader::SwStage::Fragment;
    info.images.emplace_back();
    info.resource_usage_known = true;
    auto image = AmdGpu::Image::Null(false);
    image.base_address = 0x100;
    Bind(info.images[0], image);
    const auto compiled = KeyFor(info);
    ASSERT_TRUE(compiled.bitset[0]);
    auto srgb = image;
    srgb.num_format = static_cast<u64>(AmdGpu::NumberFormat::Srgb);
    Bind(info.images[0], srgb);
    EXPECT_TRUE(compiled == KeyFor(info));

    info.image_srgb_used.set(0);
    Bind(info.images[0], image);
    const auto degamma = KeyFor(info);
    Bind(info.images[0], srgb);
    EXPECT_FALSE(degamma == KeyFor(info));
}

// These focused header tests do not initialize the emulator's logging or fatal-error backend.
void Common::Log::VLog(Class, Level, const char* file, int line, const char*, fmt::string_view,
                       fmt::format_args) {
    ADD_FAILURE() << "Unexpected shader metadata assertion at " << file << ':' << line;
}

void assert_fail_impl() {
    std::abort();
}

void unreachable_impl() {
    std::abort();
}

// StageSpecialization parses a fetch shader only for vertex stages, which these tests do not use.
bool Shader::Gcn::ParseFetchShader(const Shader::Info&, Shader::Gcn::FetchShaderData&) {
    ADD_FAILURE() << "Unexpected fetch shader parse";
    return false;
}
