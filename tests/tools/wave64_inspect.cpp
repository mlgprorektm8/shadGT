// SPDX-FileCopyrightText: Copyright 2026 shadGT
// SPDX-License-Identifier: GPL-2.0-or-later

// Offline check of the wave64 lowering decisions for a dumped GCN compute shader:
//   wave64_inspect <shader.bin> [workgroup threads] [subgroup size]
// Runs the translator's control-flow steps (no resource passes, so no guest memory is read)
// and prints, for every if and loop exit, whether LowerWave64BallotPass treats it as uniform,
// and which lane reads it leaves unlowered.

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common/object_pool.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"

namespace Common {
bool PerfFeatureEnabled(u32 id) {
    const char* env = std::getenv("SHADGT_DISABLE_PERF");
    if (!env) {
        return true;
    }
    const std::string list = std::string(",") + env + ",";
    return list.find("," + std::to_string(id) + ",") == std::string::npos;
}
} // namespace Common

namespace Shader::Optimization {
extern std::string* g_wave64_trace;
}

namespace Shader {
IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list);
void EmitControlFlowGraph(IR::Program& program, Pools& pools, Gcn::CFG& cfg,
                          RuntimeInfo& runtime_info, const Profile& profile);
} // namespace Shader

using namespace Shader;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: wave64_inspect <shader.bin> [threads] [subgroup]\n");
        return 1;
    }
    std::ifstream file{argv[1], std::ios::binary};
    std::vector<u32> code((std::istreambuf_iterator<char>(file)), {});
    {
        std::vector<char> bytes;
        file.clear();
        file.seekg(0);
        bytes.assign(std::istreambuf_iterator<char>(file), {});
        code.resize(bytes.size() / 4);
        std::memcpy(code.data(), bytes.data(), code.size() * 4);
    }
    const u32 threads = argc > 2 ? std::stoul(argv[2]) : 64;
    const u32 subgroup = argc > 3 ? std::stoul(argv[3]) : 32;

    Info info{};
    info.hw_stage = HwStage::Compute;
    info.sw_stage = SwStage::Compute;
    RuntimeInfo runtime_info{};
    runtime_info.Initialize(HwStage::Compute, SwStage::Compute);
    runtime_info.props.num_user_data = 16;
    runtime_info.hw.cs.workgroup_size = {threads, 1, 1};
    runtime_info.hw.cs.tgid_enable = {true, true, true};
    Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.subgroup_size = subgroup;
    profile.support_float64 = true;

    Pools pools;
    Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Gcn::GcnDecodeContext decoder;
    IR::Program program{info};
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};
    EmitControlFlowGraph(program, pools, cfg, runtime_info, profile);
    Optimization::SsaRewritePass(program);
    Optimization::ConstantPropagationPass(program.post_order_blocks);
    Optimization::ReadLaneEliminationPass(program);
    Optimization::PhiSimplificationPass(program);
    Optimization::InverseBallotEliminationPass(program);
    for (auto* block : program.blocks) {
        block->imm_predecessors.clear();
        block->imm_successors.clear();
        block->ssa_state.Reset();
    }
    Optimization::LowerPhisToRegsPass(program);
    program.syntax_list = Gcn::BuildASL(pools, cfg, info);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front().data.block);
    Optimization::SsaRepairPass(program);
    Optimization::SsaRewritePass(program);
    Optimization::DeadCodeEliminationPass(program);

    std::string trace;
    Optimization::g_wave64_trace = &trace;
    Optimization::LowerWave64BallotPass(program, runtime_info, profile);
    Optimization::g_wave64_trace = nullptr;
    for (char& c : trace) {
        if (c == ';') {
            c = '\n';
        }
    }
    std::cout << trace << '\n';
    return 0;
}
