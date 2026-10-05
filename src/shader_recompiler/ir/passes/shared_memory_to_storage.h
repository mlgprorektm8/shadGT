// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/resource.h"

namespace Shader::Optimization {

constexpr u32 SharedMemoryAccessIndexShift(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadSharedU16:
    case IR::Opcode::WriteSharedU16:
        return 1;
    case IR::Opcode::LoadSharedU64:
    case IR::Opcode::WriteSharedU64:
    case IR::Opcode::SharedAtomicIAdd64:
    case IR::Opcode::SharedAtomicISub64:
    case IR::Opcode::SharedAtomicSMin64:
    case IR::Opcode::SharedAtomicUMin64:
    case IR::Opcode::SharedAtomicSMax64:
    case IR::Opcode::SharedAtomicUMax64:
    case IR::Opcode::SharedAtomicInc64:
    case IR::Opcode::SharedAtomicDec64:
    case IR::Opcode::SharedAtomicAnd64:
    case IR::Opcode::SharedAtomicOr64:
    case IR::Opcode::SharedAtomicXor64:
    case IR::Opcode::SharedAtomicCmpSwap64:
        return 3;
    default:
        return 2;
    }
}

inline bool HasSharedMemoryStorageBuffer(const BufferResourceList& buffers) {
    return std::ranges::any_of(buffers, [](const BufferResource& buffer) {
        return buffer.buffer_type == BufferType::SharedMemory;
    });
}

} // namespace Shader::Optimization
