// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <limits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_aliasing.h"

using VideoCore::ImageAliasExportOffset;
using VideoCore::ImageAliasFootprint;
using VideoCore::ImageAliasOverlaps;
using VideoCore::PlanImageAliasExports;

TEST(ImageAliasing, NewImageWritesCanReplacePreviouslyPreservedBufferData) {
    using VideoCore::ImageFlagBits;
    const auto rendered =
        VideoCore::DescribeImageAlias(0x10000, 0x8000, ImageFlagBits::GpuModified, true, true);
    EXPECT_TRUE(rendered.image_authoritative);
    EXPECT_FALSE(rendered.buffer_authoritative);
    const std::array images{rendered};
    EXPECT_EQ(PlanImageAliasExports(images, 0x14000, 16).size(), 1);
}

TEST(ImageAliasing, SynchronizedImagesDoNotExportAgainUntilModified) {
    using VideoCore::ImageFlagBits;
    const auto flags = ImageFlagBits::GpuModified | ImageFlagBits::BufferCoherent;
    const std::array images{VideoCore::DescribeImageAlias(0x10000, 0x8000, flags, true, true)};
    EXPECT_TRUE(PlanImageAliasExports(images, 0x14000, 16).empty());
}

TEST(ImageAliasing, DirtyAliasesNeedTrackedBufferDataToBlockAnExport) {
    using VideoCore::ImageFlagBits;
    const auto dirty_flags = ImageFlagBits::GpuModified | ImageFlagBits::GpuDirty;
    const auto untracked = VideoCore::DescribeImageAlias(0x14000, 0x2000, dirty_flags, false, true);
    const auto tracked = VideoCore::DescribeImageAlias(0x14000, 0x2000, dirty_flags, true, true);
    EXPECT_FALSE(untracked.image_authoritative);
    EXPECT_FALSE(untracked.buffer_authoritative);
    EXPECT_TRUE(tracked.buffer_authoritative);
    const auto rendered =
        VideoCore::DescribeImageAlias(0x10000, 0x8000, ImageFlagBits::GpuModified, false, true);
    const std::array without_buffer{rendered, untracked};
    EXPECT_EQ(PlanImageAliasExports(without_buffer, 0x11000, 16).size(), 1);
    const std::array with_buffer{rendered, tracked};
    EXPECT_TRUE(PlanImageAliasExports(with_buffer, 0x11000, 16).empty());
}

TEST(ImageAliasing, DetectsOnlyOriginAlignedCompatible2DCrops) {
    VideoCore::ImageInfo full{};
    full.guest_address = 0x10000;
    full.type = AmdGpu::ImageType::Color2D;
    full.resources = {1, 1};
    full.props.is_depth = false;
    full.props.is_block = false;
    full.pixel_format = vk::Format::eR16G16B16A16Sfloat;
    full.num_bits = 64;
    full.num_samples = 1;
    full.pitch = 2048;
    full.tile_mode = AmdGpu::TileMode::DisplayLinearAligned;
    full.mips_layout[0].pitch = 2048;
    full.size = {1920, 1080, 1};

    auto crop = full;
    crop.size = {1280, 720, 1};
    EXPECT_TRUE(crop.IsSubrectOf(full));
    EXPECT_FALSE(full.IsSubrectOf(crop));

    auto incompatible = crop;
    incompatible.pitch -= 1;
    EXPECT_FALSE(incompatible.IsSubrectOf(full));
    incompatible = crop;
    incompatible.guest_address += 0x100;
    EXPECT_FALSE(incompatible.IsSubrectOf(full));
    incompatible = crop;
    incompatible.resources.layers = 2;
    EXPECT_FALSE(incompatible.IsSubrectOf(full));
}

TEST(ImageAliasing, InteriorWritePreservesTheEntireRenderedFootprint) {
    const std::array images{ImageAliasFootprint{0x12000, 0x8000, true, false, true}};
    const auto exports = PlanImageAliasExports(images, 0x17ff0, 16);
    ASSERT_EQ(exports.size(), 1);
    EXPECT_EQ(exports[0].index, 0);
    EXPECT_EQ(exports[0].address, 0x12000);
    EXPECT_EQ(exports[0].size, 0x8000);
    EXPECT_EQ(ImageAliasExportOffset(exports[0], 0x10000, 0x10000), 0x2000);
}

TEST(ImageAliasing, PreservesMultipleDisjointRenderedImages) {
    const std::array images{
        ImageAliasFootprint{0x10000, 0x2000, true, false, true},
        ImageAliasFootprint{0x12000, 0x3000, true, false, true},
        ImageAliasFootprint{0x18000, 0x1000, true, false, true},
    };
    const auto exports = PlanImageAliasExports(images, 0x11000, 0x2000);
    ASSERT_EQ(exports.size(), 2);
    EXPECT_EQ(exports[0].index, 0);
    EXPECT_EQ(exports[0].size, 0x2000);
    EXPECT_EQ(exports[1].index, 1);
    EXPECT_EQ(exports[1].address, 0x12000);
    EXPECT_EQ(exports[1].size, 0x3000);
}

TEST(ImageAliasing, RejectsCompetingRenderedImagesWithoutWriteProvenance) {
    const std::array images{
        ImageAliasFootprint{0x10000, 0x8000, true, false, true},
        ImageAliasFootprint{0x14000, 0x8000, true, false, true},
    };
    EXPECT_TRUE(PlanImageAliasExports(images, 0x15000, 4).empty());
}

TEST(ImageAliasing, BufferAuthoritativeCropsBlockSequentialImageExports) {
    constexpr VAddr base = 0x10000;
    const std::array images{
        ImageAliasFootprint{base, 0x4000, true, false, true},
        ImageAliasFootprint{base, 0x2000, true, true, true},
    };
    EXPECT_TRUE(PlanImageAliasExports(images, base, 0x1000).empty());

    const std::array self_authoritative{
        ImageAliasFootprint{base, 0x4000, true, true, true},
    };
    EXPECT_TRUE(PlanImageAliasExports(self_authoritative, base, 0x1000).empty());
}

TEST(ImageAliasing, RejectsCompetitorsOutsideTheRequestedWrite) {
    const std::array images{
        ImageAliasFootprint{0x10000, 0x8000, true, false, true},
        ImageAliasFootprint{0x17000, 0x1000, true, false, true},
    };
    // Exporting the first image in full would overwrite the second image's newer bytes,
    // even though that second image does not intersect this partial buffer write.
    EXPECT_FALSE(ImageAliasOverlaps(images[1], 0x11000, 16));
    EXPECT_TRUE(PlanImageAliasExports(images, 0x11000, 16).empty());
}

TEST(ImageAliasing, RejectsAnAliasWhoseNewerPixelsAreInTheBufferArena) {
    const std::array images{
        ImageAliasFootprint{0x10000, 0x8000, true, false, true},
        ImageAliasFootprint{0x14000, 0x2000, false, true, true},
    };
    EXPECT_TRUE(PlanImageAliasExports(images, 0x11000, 16).empty());
}

TEST(ImageAliasing, CpuOriginAliasesDoNotCompeteWithRenderedPixels) {
    const std::array images{
        ImageAliasFootprint{0x10000, 0x8000, true, false, true},
        ImageAliasFootprint{0x14000, 0x2000, false, false, true},
    };
    const auto exports = PlanImageAliasExports(images, 0x15000, 16);
    ASSERT_EQ(exports.size(), 1);
    EXPECT_EQ(exports[0].index, 0);
}

TEST(ImageAliasing, OnlyPendingBufferWritesMakeAnAliasAuthoritative) {
    constexpr VAddr base = 0x10000;
    VideoCore::RangeSet pending_buffer_writes;
    const auto make_images = [&] {
        return std::array{
            ImageAliasFootprint{base, 0x4000, true, pending_buffer_writes.Intersects(base, 0x4000),
                                true},
            ImageAliasFootprint{base, 0x2000, false, pending_buffer_writes.Intersects(base, 0x2000),
                                true},
        };
    };

    // A fresh alias can carry GpuDirty as an initial cache state without proving that
    // the buffer contains newer pixels. The clean rendered image can still be exported.
    auto images = make_images();
    EXPECT_EQ(PlanImageAliasExports(images, base, 0x2000).size(), 1);

    // A tracked GPU buffer write is concrete evidence that exporting the overlapping
    // image would overwrite newer buffer contents.
    pending_buffer_writes.Add(base, 0x2000);
    images = make_images();
    EXPECT_TRUE(PlanImageAliasExports(images, base, 0x2000).empty());
}

TEST(ImageAliasing, ExcludesMultisampledAndNonRenderedImages) {
    const std::array images{
        ImageAliasFootprint{0x10000, 0x1000, true, false, false},
        ImageAliasFootprint{0x12000, 0x1000, false, true, true},
        ImageAliasFootprint{0x14000, 0x1000, false, false, true},
    };
    EXPECT_TRUE(PlanImageAliasExports(images, 0x10000, 0x5000).empty());
}

TEST(ImageAliasing, EmptyAndAdjacentRangesDoNotOverlap) {
    const ImageAliasFootprint image{0x10000, 0x1000, true, false, true};
    EXPECT_FALSE(ImageAliasOverlaps(image, 0x10000, 0));
    EXPECT_FALSE(ImageAliasOverlaps(image, 0x11000, 4));
    EXPECT_FALSE(ImageAliasOverlaps(image, 0xfffc, 4));
    EXPECT_TRUE(ImageAliasOverlaps(image, 0x10fff, 1));
    const auto high_address = std::numeric_limits<VAddr>::max() - 15;
    EXPECT_TRUE(ImageAliasOverlaps({high_address, 16, true, false, true}, high_address + 8, 8));
}

TEST(ImageAliasing, FullExportMustFitTheResidentArenaAtTheImageBase) {
    const VideoCore::ImageAliasExport image{0, (1ULL << 32) + 0x8000, 0x4000};
    EXPECT_EQ(ImageAliasExportOffset(image, 1ULL << 32, 0x10000), 0x8000);
    EXPECT_EQ(ImageAliasExportOffset(image, 1ULL << 32, 0xc000), 0x8000);
    EXPECT_FALSE(ImageAliasExportOffset(image, 1ULL << 32, 0xbfff).has_value());
    EXPECT_FALSE(ImageAliasExportOffset(image, image.address + 1, 0x10000).has_value());
    EXPECT_FALSE(ImageAliasExportOffset(image, image.address, 0x3fff).has_value());
}

TEST(ImageAliasing, TracksUntouchedPixelsForLaterReadbackAndRefresh) {
    const std::array images{ImageAliasFootprint{0x10000, 0x8000, true, false, true}};
    const auto exports = PlanImageAliasExports(images, 0x14000, 16);
    ASSERT_EQ(exports.size(), 1);
    VideoCore::RangeSet gpu_ranges;
    for (const auto& image : exports) {
        gpu_ranges.Add(image.address, image.size);
    }
    gpu_ranges.Add(0x14000, 16);
    EXPECT_TRUE(gpu_ranges.Contains(0x10000, 0x8000));
    EXPECT_TRUE(gpu_ranges.Contains(0x17000, 0x1000));

    std::vector<std::pair<VAddr, VAddr>> downloads;
    gpu_ranges.ForEachInRange(0x17000, 0x1000,
                              [&](VAddr start, VAddr end) { downloads.emplace_back(start, end); });
    const std::vector<std::pair<VAddr, VAddr>> expected{{0x17000, 0x18000}};
    EXPECT_EQ(downloads, expected);

    // Reading back one untouched page must leave the partial write and all other
    // preserved bytes authoritative in the arena.
    gpu_ranges.Subtract(0x17000, 0x1000);
    EXPECT_TRUE(gpu_ranges.Contains(0x10000, 0x7000));
    EXPECT_FALSE(gpu_ranges.Intersects(0x17000, 0x1000));
}

TEST(ImageAliasing, FullPreservationProofSurvivesRefreshAndAllowsInteriorPostWriteInvalidation) {
    using VideoCore::ImageFlagBits;
    const auto preserved = ImageFlagBits::GpuModified | ImageFlagBits::GpuDirty |
                           ImageFlagBits::BufferCoherent | ImageFlagBits::Registered;
    const auto refreshed = VideoCore::ImageFlagsAfterBufferUpload(preserved, true);
    EXPECT_TRUE(False(refreshed & ImageFlagBits::Dirty));
    EXPECT_TRUE(True(refreshed & ImageFlagBits::BufferCoherent));
    EXPECT_TRUE(VideoCore::CanInvalidateImageFromGPU(refreshed, false));
    EXPECT_TRUE(True(refreshed & ImageFlagBits::Registered));
}

TEST(ImageAliasing, UnpreservedCleanRenderedImagesKeepTheInteriorWriteGuard) {
    using VideoCore::ImageFlagBits;
    EXPECT_FALSE(VideoCore::CanInvalidateImageFromGPU(ImageFlagBits::GpuModified, false));
    EXPECT_TRUE(VideoCore::CanInvalidateImageFromGPU(ImageFlagBits::GpuModified, true));
    EXPECT_TRUE(VideoCore::CanInvalidateImageFromGPU(ImageFlagBits::GpuDirty, false));
}

TEST(ImageAliasing, NewRenderStorageAndCopyWritesRevokeTheFullBufferProof) {
    using VideoCore::ImageFlagBits;
    const auto preserved = ImageFlagBits::GpuModified | ImageFlagBits::BufferCoherent;
    const auto written = VideoCore::ImageFlagsAfterGpuWrite(preserved);
    EXPECT_TRUE(True(written & ImageFlagBits::GpuModified));
    EXPECT_TRUE(False(written & ImageFlagBits::BufferCoherent));
    EXPECT_FALSE(VideoCore::CanInvalidateImageFromGPU(written, false));
    // A copy from a CPU-origin image must also revoke the destination proof,
    // even when it does not inherit a new GpuModified flag from the source.
    const auto copied = VideoCore::ImageFlagsAfterGpuWrite(preserved, ImageFlagBits::Empty);
    EXPECT_TRUE(False(copied & ImageFlagBits::BufferCoherent));
}

TEST(ImageAliasing, CpuWritesAndUnprovenUploadsCannotRetainBufferProof) {
    using VideoCore::ImageFlagBits;
    const auto preserved = ImageFlagBits::GpuModified | ImageFlagBits::BufferCoherent;
    for (const bool maybe : {false, true}) {
        const auto written = VideoCore::ImageFlagsAfterCpuWrite(preserved, maybe);
        EXPECT_TRUE(False(written & ImageFlagBits::BufferCoherent));
        EXPECT_TRUE(True(written & ImageFlagBits::Dirty));
    }
    const auto uploaded = VideoCore::ImageFlagsAfterBufferUpload(preserved, false);
    EXPECT_TRUE(False(uploaded & ImageFlagBits::BufferCoherent));
    EXPECT_FALSE(VideoCore::CanInvalidateImageFromGPU(uploaded, false));
    const auto unpreserved =
        VideoCore::ImageFlagsAfterBufferUpload(ImageFlagBits::GpuModified, true);
    EXPECT_TRUE(False(unpreserved & ImageFlagBits::BufferCoherent));
}
