// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "common/types.h"

namespace Shader::Backend {

struct Bindings {
    u32 unified{};
    u32 buffer{};

    auto operator<=>(const Bindings&) const = default;
};

/// SHD-002: each graphics stage binds its resources in a descriptor set of its own, numbered
/// from 0, so a stage's translation and its compiled pipeline library don't depend on the other
/// stages of the pipeline (a vertex shader's bindings used to continue after the pixel
/// shader's, so it was translated and compiled again for pixel shaders with other resources).
/// The set is the stage's SwStage value (fragment 0, tessellation control 1, tessellation
/// evaluation 2, vertex 3, geometry 4); compute uses set 0. Push data's buffer offsets are one
/// array shared by the stages: each stage has its own range of it.
constexpr u32 MaxDescriptorSets = 5;
constexpr u32 StageSet(u32 sw_stage) {
    return sw_stage < MaxDescriptorSets ? sw_stage : 0;
}
constexpr std::array<u32, 6> StageBufferOffsetBase{0, 32, 44, 16, 56, 0};
constexpr std::array<u32, 6> StageBufferOffsetCount{16, 12, 12, 16, 8, 64};
constexpr Bindings StageStart(u32 sw_stage) {
    return {0, sw_stage < StageBufferOffsetBase.size() ? StageBufferOffsetBase[sw_stage] : 0};
}

} // namespace Shader::Backend
