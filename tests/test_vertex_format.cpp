// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/vertex_format.h"

namespace {

using Vulkan::VertexAttributeFormat;

TEST(VertexFormat, EmptyVsharpGetsAValidFormat) {
    EXPECT_EQ(VertexAttributeFormat(vk::Format::eUndefined), vk::Format::eR32G32B32A32Sfloat);
}

TEST(VertexFormat, GuestFormatsAreKept) {
    for (const auto format : {vk::Format::eR32G32B32Sfloat, vk::Format::eR8G8B8A8Unorm,
                              vk::Format::eR16G16Sint, vk::Format::eR32Uint}) {
        EXPECT_EQ(VertexAttributeFormat(format), format);
    }
}

} // namespace
