// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <map>
#include <gtest/gtest.h>
#include <spirv/unified1/spirv.hpp11>
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace {

using namespace Shader;

struct ExportResult {
    u32 mrt_mask{};
    std::array<std::array<float, 4>, 2> values{};
    std::array<bool, 2> stored{};
    std::vector<u32> spirv;
};

ExportResult CompileExport(bool swizzled_alpha) {
    Info info{};
    info.hw_stage = HwStage::Fragment;
    info.sw_stage = SwStage::Fragment;
    IR::Program program{info};
    Pools pools{};
    auto* block = pools.block_pool.Create(pools.inst_pool);
    program.blocks.push_back(block);
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Block;
    program.syntax_list.back().data.block = block;
    program.syntax_list.emplace_back();
    program.syntax_list.back().type = IR::AbstractSyntaxNode::Type::Return;
    program.post_order_blocks = IR::PostOrder(block);

    Profile profile{};
    profile.supported_spirv = 0x00010600;
    RuntimeInfo runtime{};
    runtime.Initialize(HwStage::Fragment, SwStage::Fragment);
    runtime.hw.fs.dual_source_blending = swizzled_alpha;
    auto& buffer = runtime.hw.fs.color_buffers[0];
    buffer.num_format = AmdGpu::NumberFormat::Unorm;
    buffer.export_format = AmdGpu::ShaderExportFormat::ABGR_32;
    buffer.swizzle = {AmdGpu::CompSwizzle::Alpha, AmdGpu::CompSwizzle::Blue,
                      AmdGpu::CompSwizzle::Green, AmdGpu::CompSwizzle::Red};
    buffer.blend_swizzled_alpha = swizzled_alpha;

    Gcn::Translator translator(program.info, runtime, profile);
    translator.EmitPrologue(block);
    IR::IREmitter ir{*block};
    const std::array source{1.f, 0.6f, 0.2f, 0.25f};
    Gcn::GcnInst instruction{};
    instruction.control.exp.target = u32(IR::Attribute::RenderTarget0);
    instruction.control.exp.en = 15;
    for (u32 channel = 0; channel < source.size(); ++channel) {
        ir.SetVectorReg(IR::VectorReg(channel), ir.Imm32(source[channel]));
        instruction.src[channel].code = channel;
        instruction.src[channel].field = Gcn::OperandField::VectorGPR;
    }
    translator.EmitExport(instruction);
    ir.Epilogue();
    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.blocks);
    Optimization::DeadCodeEliminationPass(program);
    Optimization::CollectShaderInfoPass(program, profile);

    ExportResult result;
    result.mrt_mask = program.info.mrt_mask;
    for (const auto& inst : block->Instructions()) {
        if (inst.GetOpcode() != IR::Opcode::SetAttribute) {
            continue;
        }
        const auto attribute = inst.Arg(0).Attribute();
        if (attribute != IR::Attribute::RenderTarget0 &&
            attribute != IR::Attribute::RenderTarget1) {
            continue;
        }
        const auto output = u32(attribute) - u32(IR::Attribute::RenderTarget0);
        result.stored[output] = true;
        const auto value = inst.Arg(1);
        EXPECT_TRUE(value.IsImmediate());
        if (value.IsImmediate()) {
            result.values[output][inst.Arg(2).U32()] = value.F32();
        }
    }
    Backend::Bindings bindings{};
    result.spirv = Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
    return result;
}

std::map<u32, u32> OutputIndices(const std::vector<u32>& spirv) {
    std::map<u32, u32> locations;
    std::map<u32, u32> indices;
    for (size_t offset = 5; offset < spirv.size();) {
        const auto count = spirv[offset] >> 16;
        if (count == 0 || offset + count > spirv.size()) {
            ADD_FAILURE() << "Malformed SPIR-V instruction";
            return {};
        }
        if (spv::Op(spirv[offset] & 0xffff) == spv::Op::OpDecorate && count == 4) {
            const auto decoration = spv::Decoration(spirv[offset + 2]);
            if (decoration == spv::Decoration::Location) {
                locations[spirv[offset + 1]] = spirv[offset + 3];
            } else if (decoration == spv::Decoration::Index) {
                indices[spirv[offset + 1]] = spirv[offset + 3];
            }
        }
        offset += count;
    }
    std::map<u32, u32> result;
    for (const auto& [variable, index] : indices) {
        EXPECT_TRUE(locations.contains(variable));
        if (locations.contains(variable)) {
            result[index] = locations.at(variable);
        }
    }
    return result;
}

} // namespace

TEST(ColorExport, SwizzledAlphaEmitsSecondSourceWithoutSecondAttachment) {
    const auto result = CompileExport(true);
    EXPECT_EQ(result.mrt_mask, 1U);
    EXPECT_TRUE(result.stored[0]);
    EXPECT_TRUE(result.stored[1]);
    EXPECT_EQ(result.values[0], (std::array{0.25f, 0.2f, 0.6f, 1.f}));
    EXPECT_EQ(result.values[1], (std::array{0.f, 0.25f, 0.25f, 0.25f}));
    const auto indices = OutputIndices(result.spirv);
    EXPECT_EQ(indices, (std::map<u32, u32>{{0, 0}, {1, 0}}));
}

TEST(ColorExport, NativeBlendDoesNotEmitSyntheticSource) {
    const auto result = CompileExport(false);
    EXPECT_EQ(result.mrt_mask, 1U);
    EXPECT_TRUE(result.stored[0]);
    EXPECT_FALSE(result.stored[1]);
    EXPECT_TRUE(OutputIndices(result.spirv).empty());
}
