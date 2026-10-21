// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

Id EmitLoadSharedU16(EmitContext& ctx, Id offset) {
    const Id shift_id{ctx.ConstU32(1U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u16, ctx.shared_memory_u16, index);
    return ctx.OpLoad(ctx.U16, pointer);
}

Id EmitLoadSharedU32(EmitContext& ctx, Id offset) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
    return ctx.OpLoad(ctx.U32[1], pointer);
}

Id EmitLoadSharedU64(EmitContext& ctx, Id offset) {
    const Id shift_id{ctx.ConstU32(2U)};
    const Id index0{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    const Id index1{ctx.OpIAdd(ctx.U32[1], index0, ctx.ConstU32(1U))};
    const Id pointer0 = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index0);
    const Id pointer1 = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index1);
    const Id val0 = ctx.OpLoad(ctx.U32[1], pointer0);
    const Id val1 = ctx.OpLoad(ctx.U32[1], pointer1);
    const Id vec = ctx.OpCompositeConstruct(ctx.U32[2], val0, val1);
    return ctx.OpBitcast(ctx.U64, vec);
}

void EmitWriteSharedU16(EmitContext& ctx, Id offset, Id value) {
    const Id shift{ctx.ConstU32(1U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u16, ctx.shared_memory_u16, index);
    ctx.OpStore(pointer, value);
}

void EmitWriteSharedU32(EmitContext& ctx, Id offset, Id value) {
    const Id shift{ctx.ConstU32(2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift)};
    const Id pointer = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
    ctx.OpStore(pointer, value);
}

void EmitWriteSharedU64(EmitContext& ctx, Id offset, Id value) {
    const Id shift{ctx.ConstU32(2U)};
    const Id index0{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift)};
    const Id index1{ctx.OpIAdd(ctx.U32[1], index0, ctx.ConstU32(1U))};
    const Id vec = ctx.OpBitcast(ctx.U32[2], value);
    const Id val0 = ctx.OpCompositeExtract(ctx.U32[1], vec, 0U);
    const Id val1 = ctx.OpCompositeExtract(ctx.U32[1], vec, 1U);
    const Id pointer0 = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index0);
    const Id pointer1 = ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index1);
    ctx.OpStore(pointer0, val0);
    ctx.OpStore(pointer1, val1);
}

} // namespace Shader::Backend::SPIRV
