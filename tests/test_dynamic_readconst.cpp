// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/recompiler.h"

namespace {

class DynamicReadConst : public testing::Test {
protected:
    Shader::Pools pools;
    Shader::Info info{};
    Shader::IR::Program program{info};
    Shader::IR::Block* block = pools.block_pool.Create(pools.inst_pool);
    Shader::IR::IREmitter ir{*block};
    Shader::Optimization::ResourceDiscoveryList resources;

    DynamicReadConst() {
        program.blocks.push_back(block);
    }

    Shader::IR::Value Base() {
        return ir.CompositeConstruct(ir.GetUserData(Shader::IR::ScalarReg::S0),
                                     ir.GetUserData(Shader::IR::ScalarReg::S1));
    }
};

TEST_F(DynamicReadConst, PreservesWorkgroupIndexedDataRead) {
    const auto base = Base();
    const auto index =
        ir.IMul(ir.GetAttributeU32(Shader::IR::Attribute::WorkgroupId, 2), ir.Imm32(3U));
    const auto value = ir.ReadConst(base, index);
    const auto consumer = ir.IAdd(value, ir.Imm32(1U));

    Shader::Optimization::LowerDynamicReadConstPass(program, resources);

    ASSERT_EQ(resources.size(), 1);
    EXPECT_EQ(value.Inst()->GetOpcode(), Shader::IR::Opcode::ReadConstBuffer);
    EXPECT_EQ(value.Inst()->Arg(1), index);
    EXPECT_EQ(consumer.Inst()->Arg(0), value);
    EXPECT_EQ(resources[0].sharps[0].dwords[0], base.Inst()->Arg(0));
    EXPECT_EQ(resources[0].sharps[0].dwords[1], base.Inst()->Arg(1));
    EXPECT_EQ(resources[0].sharps[0].post_op, Shader::SharpFetchPostOp::ReadConstPointer);
    EXPECT_FALSE(value.Inst()->Flags<Shader::IR::BufferInstInfo>().sharp_source.Value());
}

TEST_F(DynamicReadConst, PreservesLoopPhiIndex) {
    auto& phi = *block->PrependNewInst(block->end(), Shader::IR::Opcode::Phi);
    phi.SetFlags(Shader::IR::Type::U32);
    phi.AddPhiOperand(block, ir.Imm32(0U));
    const auto next = ir.IAdd(Shader::IR::U32{&phi}, ir.Imm32(1U));
    phi.AddPhiOperand(block, next);
    const auto index = ir.IAdd(ir.IMul(Shader::IR::U32{&phi}, ir.Imm32(4U)), ir.Imm32(54U));
    const auto value = ir.ReadConst(Base(), index);

    Shader::Optimization::LowerDynamicReadConstPass(program, resources);

    ASSERT_EQ(resources.size(), 1);
    EXPECT_EQ(value.Inst()->GetOpcode(), Shader::IR::Opcode::ReadConstBuffer);
    EXPECT_EQ(value.Inst()->Arg(1), index);
    EXPECT_EQ(phi.GetOpcode(), Shader::IR::Opcode::Phi);
    EXPECT_EQ(phi.Arg(1), next);
}

TEST_F(DynamicReadConst, KeepsDescriptorProducingLoadsOnFlatteningPath) {
    const auto value = ir.ReadConst(Base(), ir.GetAttributeU32(Shader::IR::Attribute::WorkgroupId));
    auto& sharp = resources.emplace_back().sharps[0];
    sharp.num_dwords = 1;
    sharp.dwords[0] = value;

    Shader::Optimization::LowerDynamicReadConstPass(program, resources);

    EXPECT_EQ(resources.size(), 1);
    EXPECT_EQ(value.Inst()->GetOpcode(), Shader::IR::Opcode::ReadConst);
}

TEST_F(DynamicReadConst, KeepsStaticLoadsAndUnknownPointersOnFlatteningPath) {
    const auto base = Base();
    const auto pointer_lo = ir.ReadConst(base, ir.Imm32(8U));
    const auto pointer_hi = ir.ReadConst(base, ir.Imm32(9U));
    const auto nested = ir.CompositeConstruct(pointer_lo, pointer_hi);
    const auto value = ir.ReadConst(nested, ir.GetAttributeU32(Shader::IR::Attribute::WorkgroupId));

    Shader::Optimization::LowerDynamicReadConstPass(program, resources);

    EXPECT_TRUE(resources.empty());
    EXPECT_EQ(pointer_lo.Inst()->GetOpcode(), Shader::IR::Opcode::ReadConst);
    EXPECT_EQ(value.Inst()->GetOpcode(), Shader::IR::Opcode::ReadConst);
}

TEST_F(DynamicReadConst, KeepsResourceOffsetDependenciesOnFlatteningPath) {
    const auto base = Base();
    const auto index = ir.ReadConst(base, ir.GetAttributeU32(Shader::IR::Attribute::WorkgroupId));
    const auto descriptor = ir.ReadConst(base, index);
    auto& sharp = resources.emplace_back().sharps[0];
    sharp.num_dwords = 1;
    sharp.dwords[0] = descriptor;

    Shader::Optimization::LowerDynamicReadConstPass(program, resources);

    EXPECT_EQ(resources.size(), 1);
    EXPECT_EQ(index.Inst()->GetOpcode(), Shader::IR::Opcode::ReadConst);
    EXPECT_EQ(descriptor.Inst()->GetOpcode(), Shader::IR::Opcode::ReadConst);
}

TEST_F(DynamicReadConst, RawPointerCreatesAnUnstridedMappedRange) {
    info.flattened_ud_buf = {0x12340000U, 0xabcd00f4U};
    Shader::BufferResource resource{};
    resource.sharp_fetch.load_mask = 3;
    resource.sharp_fetch.offsets[0] = 0;
    resource.sharp_fetch.offsets[1] = 1;
    resource.post_op = Shader::SharpFetchPostOp::ReadConstPointer;

    const auto buffer = resource.GetSharp(info);

    EXPECT_EQ(buffer.base_address, 0xf412340000ULL);
    EXPECT_EQ(buffer.stride, 0);
    EXPECT_EQ(buffer.swizzle_enable, 0);
    EXPECT_EQ(buffer.num_records, UINT32_MAX);
    EXPECT_TRUE(buffer.Valid());
}

TEST_F(DynamicReadConst, RejectsPointerOutsideGuestBufferAddressSpace) {
    info.flattened_ud_buf = {0x12340000U, 0x100U};
    Shader::BufferResource resource{};
    resource.sharp_fetch.load_mask = 3;
    resource.sharp_fetch.offsets[0] = 0;
    resource.sharp_fetch.offsets[1] = 1;
    resource.post_op = Shader::SharpFetchPostOp::ReadConstPointer;

    const auto buffer = resource.GetSharp(info);

    EXPECT_EQ(buffer.num_records, 0);
}

TEST_F(DynamicReadConst, FlattensDescriptorIndexFromLaterNestedSiblingTable) {
    // GT Sport aa3822a3: root table[0] is the T# table, while root table[12]
    // points to a buffer table whose entry[4] points to the index data at dword 39.
    std::array<u32, 16> user_data{};
    std::array<u32, 16> root{};
    std::array<u32, 8> buffer_table{};
    std::array<u32, 48> index_data{};
    std::array<u32, 64> texture_table{};
    const auto set_pointer = [](u32* destination, const u32* source) {
        const auto address = reinterpret_cast<u64>(source);
        std::memcpy(destination, &address, sizeof(address));
    };
    set_pointer(user_data.data(), root.data());
    set_pointer(root.data(), texture_table.data());
    set_pointer(root.data() + 12, buffer_table.data());
    set_pointer(buffer_table.data() + 4, index_data.data());
    index_data[39] = 2;
    for (u32 i = 0; i < texture_table.size(); ++i) {
        texture_table[i] = 0x20000000U + i;
    }
    info.user_data = user_data;
    program.post_order_blocks.push_back(block);

    const auto root_base = Base();
    const auto texture_lo = ir.ReadConst(root_base, ir.Imm32(0U));
    const auto texture_hi = ir.ReadConst(root_base, ir.Imm32(1U));
    const auto table_lo = ir.ReadConst(root_base, ir.Imm32(12U));
    const auto table_hi = ir.ReadConst(root_base, ir.Imm32(13U));
    const auto table_base = ir.CompositeConstruct(table_lo, table_hi, ir.Imm32(0U), ir.Imm32(0U));
    Shader::IR::BufferInstInfo sharp_source{};
    sharp_source.sharp_source.Assign(1U);
    const auto index_lo = ir.ReadConstBuffer(table_base, ir.Imm32(4U), sharp_source);
    const auto index_hi = ir.ReadConstBuffer(table_base, ir.Imm32(5U), sharp_source);
    const auto index_base = ir.CompositeConstruct(index_lo, index_hi, ir.Imm32(0U), ir.Imm32(0U));
    const auto index = ir.ReadConstBuffer(index_base, ir.Imm32(39U), {});
    const auto offset =
        ir.ShiftRightLogical(ir.IMul(ir.IAdd(index, ir.Imm32(5U)), ir.Imm32(32U)), ir.Imm32(2U));
    const auto texture_base =
        ir.CompositeConstruct(texture_lo, texture_hi, ir.Imm32(0U), ir.Imm32(0U));
    std::array<Shader::IR::U32, 8> descriptor;
    for (u32 i = 0; i < descriptor.size(); ++i) {
        descriptor[i] =
            ir.ReadConstBuffer(texture_base, ir.IAdd(offset, ir.Imm32(i)), sharp_source);
    }

    Shader::Optimization::FlattenExtendedUserdataPass(program);

    const u32 first = descriptor[0].Inst()->Flags<Shader::IR::BufferInstInfo>().flatbuf_off_dw;
    ASSERT_NE(first, 0U);
    for (u32 i = 0; i < descriptor.size(); ++i) {
        const u32 location =
            descriptor[i].Inst()->Flags<Shader::IR::BufferInstInfo>().flatbuf_off_dw;
        EXPECT_EQ(location, first + i);
        EXPECT_EQ(info.flattened_ud_buf[location], texture_table[56 + i]);
    }
    // The generated walker must read the current index on each refresh instead of capturing it
    // when the shader is compiled.
    index_data[39] = 1;
    info.RefreshFlatBuf();
    for (u32 i = 0; i < descriptor.size(); ++i) {
        EXPECT_EQ(info.flattened_ud_buf[first + i], texture_table[48 + i]);
        // The recorded source follows the dynamic index, so GPU writes can be refreshed later.
        EXPECT_EQ(info.flattened_ud_src[first + i], reinterpret_cast<u64>(&texture_table[48 + i]));
    }
    ASSERT_EQ(info.flattened_ud_src.size(), info.flattened_ud_buf.size());
    for (u32 i = 0; i < Shader::NUM_USER_DATA_REGS; ++i) {
        EXPECT_EQ(info.flattened_ud_src[i], 0U);
    }
}

TEST_F(DynamicReadConst, FlattensOffsetDependencyFromLaterRoot) {
    std::array<u32, 16> user_data{};
    std::array<u32, 8> data{11, 22, 33, 44, 55, 66, 77, 88};
    std::array<u32, 1> index_data{3};
    const auto data_address = reinterpret_cast<u64>(data.data());
    const auto index_address = reinterpret_cast<u64>(index_data.data());
    std::memcpy(user_data.data(), &data_address, sizeof(data_address));
    std::memcpy(user_data.data() + 2, &index_address, sizeof(index_address));
    info.user_data = user_data;
    program.post_order_blocks.push_back(block);

    const auto index_base = ir.CompositeConstruct(ir.GetUserData(Shader::IR::ScalarReg::S2),
                                                  ir.GetUserData(Shader::IR::ScalarReg::S3));
    const auto index = ir.ReadConst(index_base, ir.Imm32(0U));
    const auto read = ir.ReadConst(Base(), index);
    const auto duplicate = ir.ReadConst(Base(), index);

    Shader::Optimization::FlattenExtendedUserdataPass(program);

    const auto location = read.Inst()->Flags<u16>();
    ASSERT_NE(location, 0U);
    EXPECT_EQ(duplicate.Inst()->Flags<u16>(), location);
    EXPECT_EQ(info.flattened_ud_buf[location], data[3]);
    EXPECT_EQ(info.flattened_ud_src[location], reinterpret_cast<u64>(&data[3]));
    // The index itself was read from guest memory through the other root.
    bool found_index_source = false;
    for (const u64 source : info.flattened_ud_src) {
        found_index_source |= source == reinterpret_cast<u64>(index_data.data());
    }
    EXPECT_TRUE(found_index_source);
}

TEST_F(DynamicReadConst, LeavesGpuDependentDescriptorOffsetsUnresolved) {
    std::array<u32, 16> user_data{};
    std::array<u32, 8> data{};
    const auto address = reinterpret_cast<u64>(data.data());
    std::memcpy(user_data.data(), &address, sizeof(address));
    info.user_data = user_data;
    program.post_order_blocks.push_back(block);
    const auto read = ir.ReadConst(Base(), ir.GetAttributeU32(Shader::IR::Attribute::WorkgroupId));

    Shader::Optimization::FlattenExtendedUserdataPass(program);

    EXPECT_EQ(read.Inst()->Flags<u16>(), 0U);
    EXPECT_EQ(info.srt_info.flattened_bufsize_dw, Shader::NUM_USER_DATA_REGS);
    // The unresolved group is logged, and the generated no-load walker is still safe to rerun.
    info.RefreshFlatBuf();
    EXPECT_EQ(info.flattened_ud_buf.size(), Shader::NUM_USER_DATA_REGS);
    EXPECT_EQ(info.flattened_ud_src, std::vector<u64>(Shader::NUM_USER_DATA_REGS, 0));
}

} // namespace
