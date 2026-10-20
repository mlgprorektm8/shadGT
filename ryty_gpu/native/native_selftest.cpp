// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-021: self test of the native image registry, outside the game. For a micro-tiled, a
// macro-tiled and a display-tiled layout with mips, a pattern is put in a render target, written
// out in the guest's tiled layout (RecordDownload), handed back as a texture in "guest memory"
// (RecordUploads untiles it) and compared texel by texel.

#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <fmt/format.h>

#include "ryty_gpu/native/image_registry.h"
#include "ryty_gpu/native/rhi/rhi_vulkan.h"
#include "video_core/amdgpu/tiling.h"

namespace {

using namespace Native;

VideoCore::ImageInfo MakeInfo(AmdGpu::TileMode tile_mode, u32 width, u32 height, u32 mips,
                              VAddr address) {
    VideoCore::ImageInfo info{};
    info.props = {};
    info.props.is_tiled = tile_mode != AmdGpu::TileMode::DisplayLinearAligned;
    info.props.is_pow2 = std::has_single_bit(width) && std::has_single_bit(height);
    info.tile_mode = tile_mode;
    info.array_mode = AmdGpu::GetArrayMode(tile_mode);
    info.pixel_format = vk::Format::eR8G8B8A8Unorm;
    info.num_samples = 1;
    info.num_bits = 32;
    info.type = AmdGpu::ImageType::Color2D;
    info.size = {width, height, 1};
    info.pitch = width;
    info.resources = {static_cast<u16>(mips), 1};
    info.guest_address = address;
    info.UpdateSize();
    return info;
}

u32 Texel(u32 x, u32 y, u32 mip) {
    return x | (y << 10) | (mip << 20) | 0xA0000000u;
}

bool RunCase(Rhi::Device& device, ImageRegistry& registry, Rhi::Fence& fence, u64& tick,
             AmdGpu::TileMode tile_mode, std::string& log) {
    constexpr u32 kWidth = 256, kHeight = 256, kMips = 4;
    const auto name = AmdGpu::NameOf(tile_mode);
    // 1. A render target filled with the pattern from a linear buffer.
    const VAddr target_address = 0x7000'0000'0000ULL + (u64(tile_mode) << 24);
    const auto target_info = MakeInfo(tile_mode, kWidth, kHeight, kMips, target_address);
    const u64 size = target_info.guest_size;
    ImageEntry* target = registry.Find(target_info, true, 1);
    auto pattern = device.CreateBuffer({.size = size,
                                        .usage = Rhi::BufferUsageTransferSrc,
                                        .memory = Rhi::MemoryKind::Upload});
    auto tiled = device.CreateBuffer({.size = size,
                                      .usage = Rhi::BufferUsageStorage | Rhi::BufferUsageTransferDst,
                                      .memory = Rhi::MemoryKind::Readback});
    if (!target || !pattern || !tiled) {
        log += fmt::format("{}: setup failed\n", name);
        return false;
    }
    std::memset(pattern->Mapped(), 0, size);
    for (u32 m = 0; m < kMips; ++m) {
        const auto& mip = target_info.mips_layout[m];
        const u32 width = std::max(kWidth >> m, 1u), height = std::max(kHeight >> m, 1u);
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                const u32 value = Texel(x, y, m);
                std::memcpy(pattern->Mapped() + mip.offset + (u64(y) * mip.pitch + x) * 4, &value, 4);
            }
        }
    }
    Rhi::CommandList* list = device.AcquireCommandList();
    list->Begin();
    list->CopyBufferToImage(pattern.get(), target->image.get(), MipCopies(target_info));
    list->Barrier(Rhi::ScopeTransfer, Rhi::ScopeTransfer);
    if (!registry.RecordDownload(*list, *target, *tiled)) {
        log += fmt::format("{}: download not recorded\n", name);
        return false;
    }
    list->Barrier(Rhi::ScopeTransfer | Rhi::ScopeCompute, Rhi::ScopeHost);
    list->End();
    Rhi::CommandList* lists[] = {list};
    device.GraphicsQueue().Submit(lists, {}, {&fence, ++tick});
    if (!fence.Wait(tick, 5'000'000'000ULL)) {
        log += fmt::format("{}: download did not finish\n", name);
        return false;
    }
    // 2. The tiled bytes as guest memory, found as a texture and untiled by the upload.
    std::vector<u8> guest(size + 4096);
    u8* guest_memory = reinterpret_cast<u8*>((reinterpret_cast<uintptr_t>(guest.data()) + 4095) & ~uintptr_t{4095});
    std::memcpy(guest_memory, tiled->Mapped(), size);
    const auto texture_info =
        MakeInfo(tile_mode, kWidth, kHeight, kMips, reinterpret_cast<VAddr>(guest_memory));
    ImageEntry* texture = registry.Find(texture_info, false, 2);
    auto readback = device.CreateBuffer({.size = size,
                                         .usage = Rhi::BufferUsageTransferDst,
                                         .memory = Rhi::MemoryKind::Readback});
    if (!texture || !readback) {
        log += fmt::format("{}: texture setup failed\n", name);
        return false;
    }
    std::memset(readback->Mapped(), 0, size);
    list = device.AcquireCommandList();
    list->Begin();
    const u32 uploads = registry.RecordUploads(*list);
    list->CopyImageToBuffer(texture->image.get(), readback.get(), MipCopies(texture_info));
    list->Barrier(Rhi::ScopeTransfer, Rhi::ScopeHost);
    list->End();
    lists[0] = list;
    device.GraphicsQueue().Submit(lists, {}, {&fence, ++tick});
    if (!fence.Wait(tick, 5'000'000'000ULL)) {
        log += fmt::format("{}: upload did not finish\n", name);
        return false;
    }
    u64 wrong = 0, total = 0;
    for (u32 m = 0; m < kMips; ++m) {
        const auto& mip = texture_info.mips_layout[m];
        const u32 width = std::max(kWidth >> m, 1u), height = std::max(kHeight >> m, 1u);
        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < width; ++x) {
                u32 value;
                std::memcpy(&value, readback->Mapped() + mip.offset + (u64(y) * mip.pitch + x) * 4, 4);
                wrong += value != Texel(x, y, m);
                ++total;
            }
        }
    }
    // The tiled bytes must differ from the linear ones (otherwise nothing was tiled).
    const bool changed = std::memcmp(tiled->Mapped(), pattern->Mapped(), size) != 0;
    log += fmt::format("{}: {} uploaded, guest size {} bytes, tiled layout {}, {} of {} texels {}\n",
                       name, uploads, size, changed ? "differs from linear" : "SAME AS LINEAR",
                       total - wrong, total, wrong == 0 ? "correct" : "WRONG");
    return wrong == 0 && uploads == 1 && changed;
}

bool RunImageSelfTest(std::string& log) {
    std::string error;
    auto device = Rhi::CreateStandaloneVulkanDevice(&error);
    if (!device) {
        log += "no device: " + error + "\n";
        return false;
    }
    auto fence = device->CreateFence(0);
    u64 tick = 0;
    bool ok = true;
    {
        ImageRegistry registry{*device};
        for (const auto mode : {AmdGpu::TileMode::Thin1DThin, AmdGpu::TileMode::Thin2DThin,
                                AmdGpu::TileMode::Display2DThin}) {
            ok = RunCase(*device, registry, *fence, tick, mode, log) && ok;
        }
        const auto& stats = registry.GetStats();
        log += fmt::format("registry: {} lookups, {} textures, {} render targets, {} uploads\n",
                           stats.lookups.load(), stats.created_textures.load(),
                           stats.created_targets.load(), stats.uploads.load());
    }
    std::ignore = fence->Wait(tick, 5'000'000'000ULL);
    device->Collect();
    return ok;
}

} // namespace

extern "C" __declspec(dllexport) int RytyNativeImageSelfTest(char* report, int report_size) {
    std::string log;
    const bool ok = RunImageSelfTest(log);
    log += ok ? "NAT-021 image self test passed\n" : "NAT-021 image self test FAILED\n";
    if (report && report_size > 0) {
        const size_t n = std::min<size_t>(log.size(), static_cast<size_t>(report_size) - 1);
        std::memcpy(report, log.data(), n);
        report[n] = '\0';
    }
    return ok ? 0 : 1;
}
