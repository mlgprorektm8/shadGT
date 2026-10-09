// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "video_core/diag_bundle.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace Vulkan {

/// A diagnostic bundle being captured (see video_core/diag_bundle.h).
struct Rasterizer::DiagCapture {
    VideoCore::DiagBundle::Request request;
    std::filesystem::path dir;
    u64 start_seq{};
    u32 frame{};
    u32 frames_done{};
    u32 snapshots{};
    /// Images touched since the last snapshot: uid -> slot.
    std::unordered_map<u64, u32> touched;
    std::unordered_set<u64> shaders;
    /// Buffer ranges bound during the capture -> written by a draw or dispatch.
    std::map<std::pair<VAddr, u32>, bool> buffers;
    std::ofstream images;
    u64 image_bytes{};
    u32 images_written{};
    u32 images_skipped{};
};

} // namespace Vulkan
