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

TEST(ImageDescriptor, NonArrayImagesIgnoreDepthAndArrayFields) {
    for (const auto type :
         {AmdGpu::ImageType::Color1D, AmdGpu::ImageType::Color2D, AmdGpu::ImageType::Color2DMsaa}) {
        AmdGpu::Image image{};
        image.type = u64(type);
        image.depth = 2047;
        image.base_array = 100;
        image.last_array = 200;
        for (const bool padded : {false, true}) {
            image.pow2pad = padded;
            EXPECT_EQ(image.NumLayers(), 1u) << AmdGpu::NameOf(type);
            EXPECT_EQ(image.NumViewLayers(false), 1u) << AmdGpu::NameOf(type);
        }
    }
}

TEST(ImageDescriptor, ArrayImagesKeepDepthAndLayerPadding) {
    for (const auto type : {AmdGpu::ImageType::Color1DArray, AmdGpu::ImageType::Color2DArray,
                            AmdGpu::ImageType::Color2DMsaaArray}) {
        AmdGpu::Image image{};
        image.type = u64(type);
        image.depth = 5;
        image.base_array = 2;
        image.last_array = 4;
        EXPECT_EQ(image.NumLayers(), 6u) << AmdGpu::NameOf(type);
        EXPECT_EQ(image.NumViewLayers(true), 3u) << AmdGpu::NameOf(type);
        EXPECT_EQ(image.NumViewLayers(false), 1u) << AmdGpu::NameOf(type);
        image.pow2pad = true;
        EXPECT_EQ(image.NumLayers(), 8u) << AmdGpu::NameOf(type);
        EXPECT_EQ(image.NumViewLayers(true), 3u) << AmdGpu::NameOf(type);
    }
}

TEST(ImageDescriptor, CubeImagesKeepSixFacesPerCube) {
    AmdGpu::Image image{};
    image.type = u64(AmdGpu::ImageType::Cube);
    image.depth = 2;
    EXPECT_EQ(image.NumLayers(), 18u);
    image.pow2pad = true;
    EXPECT_EQ(image.NumLayers(), 32u);
}

TEST(ImageDescriptor, VolumeDepthDoesNotBecomeArrayLayers) {
    AmdGpu::Image image{};
    image.type = u64(AmdGpu::ImageType::Color3D);
    image.depth = 2047;
    EXPECT_EQ(image.NumLayers(), 1u);
    image.pow2pad = true;
    EXPECT_EQ(image.NumLayers(), 1u);
}
