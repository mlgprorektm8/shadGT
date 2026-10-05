// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/buffer_copy.h"

TEST(BufferCopy, EmptyBatchesHaveNoWork) {
    SmallVector<vk::BufferCopy, 8> storage;
    EXPECT_TRUE(Vulkan::NonEmptyBufferCopies({}, storage).empty());
    const std::array copies{vk::BufferCopy{16, 32, 0}, vk::BufferCopy{48, 64, 0}};
    EXPECT_TRUE(Vulkan::NonEmptyBufferCopies(copies, storage).empty());
}

TEST(BufferCopy, RemovesEmptyRegionsPreservingOffsetsAndOrder) {
    const std::array copies{vk::BufferCopy{16, 32, 0}, vk::BufferCopy{48, 64, 12},
                            vk::BufferCopy{80, 96, 0},
                            vk::BufferCopy{1ULL << 33, 1ULL << 34, 1ULL << 32}};
    SmallVector<vk::BufferCopy, 8> storage;
    const auto filtered = Vulkan::NonEmptyBufferCopies(copies, storage);
    ASSERT_EQ(filtered.size(), 2);
    EXPECT_EQ(filtered[0], copies[1]);
    EXPECT_EQ(filtered[1], copies[3]);
}

TEST(BufferCopy, NonEmptyBatchUsesOriginalStorage) {
    const std::array copies{vk::BufferCopy{16, 32, 4}, vk::BufferCopy{48, 64, 8}};
    SmallVector<vk::BufferCopy, 8> storage;
    const auto filtered = Vulkan::NonEmptyBufferCopies(copies, storage);
    EXPECT_EQ(filtered.data(), copies.data());
    EXPECT_EQ(filtered.size(), copies.size());
    EXPECT_TRUE(storage.empty());
}

TEST(BufferCopy, CacheDownloadIntersectionsExcludeEmptyRanges) {
    VideoCore::RangeSet ranges;
    ranges.Add(100, 100);
    ranges.Add(300, 100);
    size_t callbacks{};
    for (const VAddr address : {99, 100, 150, 200, 250, 300, 400}) {
        ranges.ForEachInRange(address, 0, [&](VAddr, VAddr) { ++callbacks; });
    }
    ranges.ForEachInRange(200, 100, [&](VAddr, VAddr) { ++callbacks; });
    EXPECT_EQ(callbacks, 0);

    std::vector<std::pair<VAddr, VAddr>> intersections;
    ranges.ForEachInRange(150, 200, [&](VAddr start, VAddr end) {
        intersections.emplace_back(start, end);
    });
    const std::vector<std::pair<VAddr, VAddr>> expected{{150, 200}, {300, 350}};
    EXPECT_EQ(intersections, expected);
}
