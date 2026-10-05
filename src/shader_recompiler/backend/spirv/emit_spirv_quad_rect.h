// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include "common/types.h"

namespace Shader {
struct HwFragmentRuntimeInfo;
struct Info;
} // namespace Shader

namespace Shader::Backend::SPIRV {

// Reserve the first varying for layer transport and shift auxiliary stage parameters.
inline constexpr u32 AuxLayerLocation = 0;

enum class AuxShaderType : u32 {
    RectListTCS,
    QuadListTCS,
    PassthroughTES,
};

[[nodiscard]] std::vector<u32> EmitAuxilaryTessShader(AuxShaderType type,
                                                      const HwFragmentRuntimeInfo& fs_info,
                                                      const Info& vertex_info,
                                                      const Info* fragment_info);

} // namespace Shader::Backend::SPIRV
