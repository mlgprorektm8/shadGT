// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>
#include "video_core/texture_cache/image_descriptor.h"

namespace {

vk::PhysicalDeviceLimits MakeLimits() {
    vk::PhysicalDeviceLimits limits{};
    limits.maxImageDimension1D = 8192;
    limits.maxImageDimension2D = 16384;
    limits.maxImageDimension3D = 2048;
    limits.maxImageArrayLayers = 2048;
    return limits;
}

AmdGpu::Image MakeImage(AmdGpu::ImageType type = AmdGpu::ImageType::Color2D) {
    AmdGpu::Image image{};
    image.type = u64(type);
    image.width = 1023;
    image.height = 511;
    image.pitch = 1023;
    image.last_level = 10;
    return image;
}

} // namespace

using VideoCore::CheckImageDescriptorGeometry;
using VideoCore::ImageDescriptorGeometryError;

TEST(ImageDescriptorGeometry, PaddedCubeLayerCountCannotWrapToZero) {
    auto image = MakeImage(AmdGpu::ImageType::Cube);
    image.depth = 5461;
    image.pow2pad = true;
    ASSERT_EQ(image.NumLayers(), 65536u);
    auto limits = MakeLimits();
    limits.maxImageArrayLayers = 65536;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, limits),
              ImageDescriptorGeometryError::UnrepresentableLayers);
}

TEST(ImageDescriptorGeometry, AcceptsValidArrayAndCubeGeometryAtDeviceLimits) {
    auto image = MakeImage(AmdGpu::ImageType::Color2DArray);
    image.depth = 2047;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::None);
    image.type = u64(AmdGpu::ImageType::Cube);
    image.depth = 255;
    image.pow2pad = true;
    ASSERT_EQ(image.NumLayers(), 2048u);
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::None);
}

TEST(ImageDescriptorGeometry, AppliesActualDeviceArrayLayerLimit) {
    auto image = MakeImage(AmdGpu::ImageType::Color2DArray);
    image.depth = 2048;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::DeviceLayerLimit);
    auto limits = MakeLimits();
    limits.maxImageArrayLayers = 4096;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, limits), ImageDescriptorGeometryError::None);
}

TEST(ImageDescriptorGeometry, ChecksDimensionsForTheCreatedBacking) {
    auto image = MakeImage(AmdGpu::ImageType::Color1D);
    image.width = 8191;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::None);
    image.width = 8192;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::DeviceExtentLimit);
    image.type = u64(AmdGpu::ImageType::Color2D);
    image.width = 16383;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::None);
    auto limits = MakeLimits();
    limits.maxImageDimension2D = 8192;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, limits),
              ImageDescriptorGeometryError::DeviceExtentLimit);
    image = MakeImage(AmdGpu::ImageType::Color3D);
    image.depth = 2047;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::None);
    image.depth = 2048;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::DeviceExtentLimit);
}

TEST(ImageDescriptorGeometry, RequiresLegalFullMipChainAndViewOrdering) {
    auto image = MakeImage();
    image.last_level = 11;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::MipLevels);
    image.last_level = 10;
    image.base_level = 11;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::ViewMipLevels);
}

TEST(ImageDescriptorGeometry, AcceptsSupportedGuestSamplesAndRejectsInvalidExponents) {
    auto image = MakeImage(AmdGpu::ImageType::Color2DMsaa);
    image.last_level = 4;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::None);
    image.last_level = 5;
    EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
              ImageDescriptorGeometryError::SampleCount);
}

TEST(ImageDescriptorGeometry, RejectsNonTextureDescriptorTypes) {
    auto image = MakeImage();
    for (u64 type = 0; type < 8; ++type) {
        image.type = type;
        EXPECT_EQ(CheckImageDescriptorGeometry(image, MakeLimits()),
                  ImageDescriptorGeometryError::InvalidType);
    }
}
