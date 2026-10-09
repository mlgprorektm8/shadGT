// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FIX-039: render targets for the micro-tiled mips of a macro-tiled image. The values are the
// ones GT Sport programs for the environment map of the car purchase thumbnail (decoded from a
// diagnostic bundle's PM4 capture, tools/mcp/pm4_draws.py): a 2048x1024 B10G11R11 texture at
// 0x1009bc0000 with 12 mips and tile index 14, whose mips are rendered as separate targets, mips
// 1-4 with tile index 14 and mips 5-11 with tile index 13.

#include <gtest/gtest.h>
#include "core/libraries/kernel/process.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/image_info.h"

// image_info.cpp also holds constructors from guest descriptors that this test does not use.
namespace Libraries::Kernel {
s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    return 0;
}
} // namespace Libraries::Kernel
namespace Common::Log {
void VLog(Class, Level, const char*, int, const char*, fmt::string_view, fmt::format_args) {}
} // namespace Common::Log
namespace Vulkan::LiverpoolToVK {
const std::array<vk::Format, surface_format_table_size> surface_format_table{};
vk::Format DepthFormat(AmdGpu::DepthBuffer::ZFormat, AmdGpu::DepthBuffer::StencilFormat) {
    return vk::Format::eUndefined;
}
} // namespace Vulkan::LiverpoolToVK
void assert_fail_impl() {
    std::abort();
}
[[noreturn]] void unreachable_impl() {
    std::abort();
}

namespace {
using VideoCore::ImageInfo;

ImageInfo Color2D(VAddr address, u32 width, u32 height, u32 levels, AmdGpu::TileMode tile_mode) {
    ImageInfo info{};
    info.guest_address = address;
    info.type = AmdGpu::ImageType::Color2D;
    info.resources = {static_cast<u16>(levels), 1};
    info.props.is_block = false;
    info.props.is_pow2 = false;
    info.props.is_tiled = true;
    info.pixel_format = vk::Format::eB10G11R11UfloatPack32;
    info.num_bits = 32;
    info.num_samples = 1;
    info.size = {width, height, 1};
    info.pitch = std::max(width, 8u);
    info.tile_mode = tile_mode;
    info.array_mode = AmdGpu::GetArrayMode(tile_mode);
    info.UpdateSize();
    return info;
}

constexpr VAddr EnvMap = 0x1009bc0000;
constexpr auto Macro = AmdGpu::TileMode::Thin2DThin; // tile index 14
constexpr auto Micro = AmdGpu::TileMode::Thin1DThin; // tile index 13

// The render targets the game binds for mips 1-11, as (address, width, height, tile mode).
struct MipTarget {
    VAddr address;
    u32 width;
    u32 height;
    AmdGpu::TileMode tile_mode;
};
constexpr MipTarget Targets[] = {
    {0x100a3c0000, 1024, 512, Macro}, {0x100a5c0000, 512, 256, Macro},
    {0x100a640000, 256, 128, Macro},  {0x100a660000, 128, 64, Macro},
    {0x100a668000, 64, 32, Micro},    {0x100a66a000, 32, 16, Micro},
    {0x100a66a800, 16, 8, Micro},     {0x100a66aa00, 8, 4, Micro},
    {0x100a66ab00, 4, 2, Micro},      {0x100a66ac00, 2, 1, Micro},
    {0x100a66ad00, 1, 1, Micro},
};
} // namespace

TEST(ImageMips, LayoutMatchesTheGameMipAddresses) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    for (u32 mip = 1; mip < 12; ++mip) {
        EXPECT_EQ(texture.guest_address + texture.mips_layout[mip].offset, Targets[mip - 1].address)
            << "mip " << mip;
    }
    // Mips 5 and smaller are micro-tiled, matching the tile index the game uses for them.
    EXPECT_EQ(texture.micro_tiled_mips & 0x1E, 0u);
    EXPECT_EQ(texture.micro_tiled_mips & 0xFE0, 0xFE0u);
}

TEST(ImageMips, RenderTargetsResolveToTheTextureMips) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    for (u32 mip = 1; mip < 12; ++mip) {
        const auto& t = Targets[mip - 1];
        const auto target = Color2D(t.address, t.width, t.height, 1, t.tile_mode);
        EXPECT_EQ(target.MipOf(texture), static_cast<s32>(mip)) << "mip " << mip;
        EXPECT_EQ(target.SliceOf(texture, mip), 0) << "mip " << mip;
    }
}

TEST(ImageMips, MacroTiledLevelsStillNeedTheSameArrayMode) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    // Mip 1 is macro-tiled: a 1D target at its address is a different surface.
    const auto wrong = Color2D(0x100a3c0000, 1024, 512, 1, Micro);
    EXPECT_EQ(wrong.MipOf(texture), -1);
    // A depth (different micro tile mode) target at a micro-tiled mip is not that mip either.
    const auto depth = Color2D(0x100a668000, 64, 32, 1, AmdGpu::TileMode::Depth1DThin);
    EXPECT_EQ(depth.MipOf(texture), -1);
}
