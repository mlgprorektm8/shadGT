// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <string>
#include <unordered_set>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>
#include "common/logging/classes.h"
#include "common/perf_monitor.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

static bool IsDivergentCondition(const IR::U1& condition) {
    if (condition.IsImmediate()) {
        return false;
    }
    const IR::Inst* const condition_inst = condition.Inst();
    return IR::BreadthFirstSearch(condition_inst,
                                  [](const IR::Inst* inst) -> std::optional<bool> {
                                      switch (inst->GetOpcode()) {
                                      case IR::Opcode::LaneId:
                                          return true;
                                      case IR::Opcode::GetAttributeU32:
                                          if (inst->Arg(0).Attribute() ==
                                              IR::Attribute::LocalInvocationId) {
                                              return true;
                                          }
                                          break;
                                      default:
                                          break;
                                      }
                                      return std::nullopt;
                                  })
        .value_or(false);
}

// FIX-020: whether a loop condition can differ between the invocations of a GCN wave. GCN
// loops are often driven by scalar branches: SCC, or VCC/EXEC masks built from scalar values and
// EXEC itself. Those take the same path in every active invocation. Values that differ per lane
// (lane and invocation ids, shuffles, atomics, the bits of a ballot of a per-lane condition) make
// the loop divergent. Wave-wide reductions (lane reads, ballots) end the search: their result is
// the same everywhere.
// Offline inspection (tests/tools): when set, records why conditions are divergent.
std::string* g_wave64_trace = nullptr;

static bool TraceDivergence(const IR::Inst* inst) {
    if (g_wave64_trace) {
        *g_wave64_trace +=
            fmt::format(" [divergent at {}]", magic_enum::enum_name(inst->GetOpcode()));
    }
    return true;
}

static bool IsDivergentLoopCondition(const IR::U1& condition) {
    if (condition.IsImmediate()) {
        return false;
    }
    std::vector<const IR::Inst*> stack{condition.Inst()};
    std::unordered_set<const IR::Inst*> visited{condition.Inst()};
    bool in_inverse_ballot = false;
    const auto push_args = [&](const IR::Inst* inst) {
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            const IR::Value value{inst->Arg(arg)};
            if (!value.IsImmediate() && visited.insert(value.Inst()).second) {
                stack.push_back(value.Inst());
            }
        }
    };
    while (!stack.empty()) {
        const IR::Inst* inst = stack.back();
        stack.pop_back();
        switch (inst->GetOpcode()) {
        case IR::Opcode::LaneId:
        case IR::Opcode::MaskedBitCount32:
        case IR::Opcode::WriteLane:
        case IR::Opcode::Shuffle:
        case IR::Opcode::ShuffleXor:
        case IR::Opcode::QuadBroadcast:
        case IR::Opcode::DataAppend:
        case IR::Opcode::DataConsume:
            return TraceDivergence(inst);
        case IR::Opcode::GetAttributeU32: {
            const auto attribute = inst->Arg(0).Attribute();
            if (attribute == IR::Attribute::LocalInvocationId ||
                attribute == IR::Attribute::LocalInvocationIndex) {
                return TraceDivergence(inst);
            }
            break;
        }
        case IR::Opcode::GetExec:
            // EXEC by itself is a per-invocation condition (an EXEC scope).
            return TraceDivergence(inst);
        case IR::Opcode::ReadLane:
        case IR::Opcode::ReadFirstLane:
        case IR::Opcode::BallotFindLsb:
            continue;
        case IR::Opcode::ConditionRef:
            // Holds a branch condition so passes keep it; the value is its argument.
            push_args(inst);
            continue;
        case IR::Opcode::InverseBallot:
            // The invocation's bit of a mask: the same in every active invocation only when the
            // mask is made of EXEC and wave-wide values.
            in_inverse_ballot = true;
            break;
        case IR::Opcode::Ballot: {
            const IR::Value arg{inst->Arg(0)};
            const bool exec_ballot =
                !arg.IsImmediate() && arg.Inst()->GetOpcode() == IR::Opcode::GetExec;
            if (in_inverse_ballot && !exec_ballot) {
                return TraceDivergence(inst);
            }
            continue;
        }
        default:
            if (inst->MayHaveSideEffects()) {
                // Atomics and other memory operations return per-invocation values.
                return TraceDivergence(inst);
            }
            break;
        }
        push_args(inst);
    }
    return false;
}

static std::vector<IR::Block*> FindUniformBlocks(const IR::Program& program) {
    using Type = IR::AbstractSyntaxNode::Type;
    static const bool uniform_loops_enabled = Common::PerfFeatureEnabled(31);

    // FIX-020: loops whose exit conditions take the same path in every invocation, by merge block.
    // Ballots and lane reads in them used to be left as 32-wide operations on NVIDIA, so a wave64
    // scan in a loop (GT Sport's grass prefix sums, cs 0x766d0f18) read lane 63 from the wrong
    // subgroup and produced wrong offsets.
    std::unordered_set<const IR::Block*> divergent_loops;
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type != Type::Repeat && node.type != Type::Break) {
            continue;
        }
        const bool repeat = node.type == Type::Repeat;
        if (g_wave64_trace) {
            *g_wave64_trace += repeat ? "Repeat:" : "Break:";
        }
        const IR::U1 cond = repeat ? node.data.repeat.cond : node.data.break_node.cond;
        const bool divergent = !uniform_loops_enabled || IsDivergentLoopCondition(cond);
        if (g_wave64_trace) {
            *g_wave64_trace += divergent ? " divergent;" : " uniform;";
        }
        if (divergent) {
            divergent_loops.insert(repeat ? node.data.repeat.merge : node.data.break_node.merge);
        }
    }

    struct ConditionalScope {
        const IR::Block* merge;
        bool divergent;
    };

    std::vector<IR::Block*> blocks;
    std::vector<ConditionalScope> conditionals;
    std::vector<const IR::Block*> loops;
    u32 divergence_depth{};
    u32 divergent_loop_depth{};
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case Type::If: {
            // FIX-020: an if on wave-wide values (lane reads, SCC) is uniform too; the older
            // search followed a lane read's per-lane input and called it divergent.
            if (g_wave64_trace) {
                *g_wave64_trace += "If:";
            }
            const bool divergent = uniform_loops_enabled
                                       ? IsDivergentLoopCondition(node.data.if_node.cond)
                                       : IsDivergentCondition(node.data.if_node.cond);
            if (g_wave64_trace) {
                *g_wave64_trace += divergent ? " divergent;" : " uniform;";
            }
            conditionals.push_back({node.data.if_node.merge, divergent});
            divergence_depth += static_cast<u32>(divergent);
            break;
        }
        case Type::EndIf:
            ASSERT(!conditionals.empty() && conditionals.back().merge == node.data.end_if.merge);
            divergence_depth -= static_cast<u32>(conditionals.back().divergent);
            conditionals.pop_back();
            break;
        case Type::Loop:
            loops.push_back(node.data.loop.merge);
            divergent_loop_depth += divergent_loops.contains(node.data.loop.merge);
            break;
        case Type::Repeat:
            if (loops.empty() || loops.back() != node.data.repeat.merge) {
                return {};
            }
            divergent_loop_depth -= divergent_loops.contains(loops.back());
            loops.pop_back();
            break;
        case Type::Block:
            if (divergence_depth == 0 && divergent_loop_depth == 0) {
                blocks.push_back(node.data.block);
            }
            break;
        default:
            break;
        }
    }
    if (!conditionals.empty() || !loops.empty()) {
        return {};
    }
    return blocks;
}

static IR::Inst* FindBallotForMaskedBitCount(const IR::Inst& mbcnt) {
    IR::Value value = mbcnt.Arg(0);
    if (value.IsImmediate()) {
        return nullptr;
    }
    IR::Inst* inst = value.Inst();
    if (inst->GetOpcode() != IR::Opcode::CompositeExtractU32x2 || inst->Arg(0).IsImmediate()) {
        return nullptr;
    }
    inst = inst->Arg(0).Inst();
    if (inst->GetOpcode() != IR::Opcode::UnpackUint2x32 || inst->Arg(0).IsImmediate()) {
        return nullptr;
    }
    inst = inst->Arg(0).Inst();
    return inst->GetOpcode() == IR::Opcode::Ballot ? inst : nullptr;
}

void LowerWave64BallotPass(IR::Program& program, const RuntimeInfo& runtime_info,
                           const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute || profile.subgroup_size == 64) {
        return;
    }

    const auto [size_x, size_y, size_z] = runtime_info.hw.cs.workgroup_size;
    const u32 num_threads = size_x * size_y * size_z;
    if (num_threads <= 32) {
        return;
    }

    std::vector<IR::Inst*> worklist;
    const auto uniform_blocks = FindUniformBlocks(program);
    for (IR::Block* block : program.blocks) {
        const bool is_uniform = std::ranges::contains(uniform_blocks, block);
        const auto push_worklist = [&](IR::Inst& inst) {
            if (is_uniform) {
                worklist.push_back(&inst);
            } else {
                LOG_WARNING(Render_Recompiler, "{} instruction in non uniform control flow",
                            inst.GetOpcode());
                if (g_wave64_trace) {
                    *g_wave64_trace +=
                        fmt::format(" NOT LOWERED: {};", magic_enum::enum_name(inst.GetOpcode()));
                }
            }
        };
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::ReadLane && inst.Arg(1).IsImmediate()) {
                push_worklist(inst);
            } else if (inst.GetOpcode() == IR::Opcode::Ballot) {
                const auto is_unpack = [](const IR::Use& use) {
                    return use.user->GetOpcode() == IR::Opcode::UnpackUint2x32;
                };
                if (std::ranges::any_of(inst.Uses(), is_unpack)) {
                    push_worklist(inst);
                }
            } else if (inst.GetOpcode() == IR::Opcode::MaskedBitCount32) {
                IR::Inst* const ballot = FindBallotForMaskedBitCount(inst);
                if (ballot == nullptr ||
                    std::ranges::contains(uniform_blocks, ballot->GetParent())) {
                    worklist.push_back(&inst);
                }
            }
        }
    }
    if (worklist.empty()) {
        return;
    }

    const u32 scratch_base = Common::AlignUp(runtime_info.hw.cs.shared_memory_size, sizeof(u64));
    const u32 scratch_size =
        (Common::AlignUp(num_threads, 64) / profile.subgroup_size) * sizeof(u32);
    program.info.shared_memory_scratch_size =
        scratch_base + scratch_size - runtime_info.hw.cs.shared_memory_size;

    for (IR::Inst* inst : worklist) {
        LOG_INFO(Render_Recompiler, "Lowering {} instruction for wave64", inst->GetOpcode());
        IR::IREmitter ir{*inst->GetParent(), IR::Block::InstructionList::s_iterator_to(*inst)};
        const IR::U32 invocation_index = ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex);
        const IR::U32 subgroup_id = ir.ShiftRightLogical(invocation_index, ir.Imm32(5));
        if (inst->GetOpcode() == IR::Opcode::Ballot) {
            const IR::U32 mask_low =
                IR::U32{ir.CompositeExtract(ir.UnpackUint2x32(ir.Ballot(IR::U1{inst->Arg(0)})), 0)};
            const IR::U32 offset =
                ir.IAdd(ir.Imm32(scratch_base), ir.ShiftLeftLogical(subgroup_id, ir.Imm32(2u)));
            ir.WriteShared(32, mask_low, offset);
            ir.Barrier();
            const IR::U64 mask = IR::U64{ir.LoadShared(64, false, offset)};
            ir.Barrier();
            inst->ReplaceUsesWithAndRemove(mask);
        } else if (inst->GetOpcode() == IR::Opcode::ReadLane) {
            const IR::U32 lane32 = ir.BitwiseAnd(IR::U32{inst->Arg(1)}, ir.Imm32(31));
            const IR::U32 half = ir.ShiftRightLogical(IR::U32{inst->Arg(1)}, ir.Imm32(5u));
            const IR::U32 offset =
                ir.IAdd(ir.Imm32(scratch_base), ir.ShiftLeftLogical(subgroup_id, ir.Imm32(2u)));
            ir.WriteShared(32, ir.ReadLane(IR::U32{inst->Arg(0)}, lane32), offset);
            ir.Barrier();
            const IR::U32 value =
                IR::U32{ir.LoadShared(32, false,
                                      ir.IAdd(ir.BitwiseAnd(offset, ir.Imm32(~7u)),
                                              ir.ShiftLeftLogical(half, ir.Imm32(2u))))};
            ir.Barrier();
            inst->ReplaceUsesWithAndRemove(value);
        } else if (inst->GetOpcode() == IR::Opcode::MaskedBitCount32) {
            const IR::U32 subgroup_invocation_id = ir.BitwiseAnd(invocation_index, ir.Imm32(63));
            const IR::U64 mask = ir.ISub(
                ir.ShiftLeftLogical(ir.Imm64(u64{1}), subgroup_invocation_id), ir.Imm64(u64{1}));
            const IR::U32 thread_mask{
                ir.CompositeExtract(ir.UnpackUint2x32(mask), inst->Arg(2).U1() ? 1u : 0u)};
            const IR::U32 masked_value{
                ir.BitCount(ir.BitwiseAnd(IR::U32{inst->Arg(0)}, thread_mask))};
            inst->ReplaceUsesWithAndRemove(ir.IAdd(masked_value, IR::U32{inst->Arg(1)}));
        }
    }
}

} // namespace Shader::Optimization
