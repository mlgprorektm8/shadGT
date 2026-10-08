// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "common/object_pool.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/program.h"

namespace Shader {

struct Profile;
struct RuntimeInfo;

struct Pools {
    static constexpr u32 InstPoolSize = 8192;
    static constexpr u32 BlockPoolSize = 32;

    Common::ObjectPool<IR::Inst> inst_pool;
    Common::ObjectPool<IR::Block> block_pool;

    explicit Pools() : inst_pool{InstPoolSize}, block_pool{BlockPoolSize} {}

    void ReleaseContents() {
        block_pool.ReleaseContents();
        inst_pool.ReleaseContents();
    }
};

/// FIX-018: true when the code has an EXEC scope opened right before an instruction that closes
/// it and opens the next one (an else after an empty if). Translations made before FIX-018 ran
/// that next scope with every invocation active.
[[nodiscard]] bool HasEmptyScopeBeforeElse(std::span<const u32> code);

/// DIAG-031: a readable listing of GCN code (pc, opcode, operands, memory offsets).
[[nodiscard]] std::string ListGcnCode(std::span<const u32> code);

[[nodiscard]] IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools,
                                           Info& info, RuntimeInfo& runtime_info,
                                           const Profile& profile);

} // namespace Shader
