// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/amdgpu/read_ahead_states.h"

using AmdGpu::ReadAheadStates;

namespace {

std::vector<u32> RegisterFile(u32 fill = 0) {
    return std::vector<u32>(ReadAheadStates::RegisterFileDwords, fill);
}

constexpr u32 ContextReg = ReadAheadStates::ContextFirst + 0x10;
constexpr u32 PsUserData = ReadAheadStates::ShGfxFirst + 0xC;   // PS user data 0
constexpr u32 PsProgram = ReadAheadStates::ShGfxFirst + 0x8;    // PS program address
constexpr u32 VsUserData = ReadAheadStates::ShGfxFirst + 0x4C;  // VS user data 0
constexpr u32 Primitive = ReadAheadStates::PrimitiveFirst + 0x42; // primitive type
constexpr u32 OutsideRanges = 0x1000;

} // namespace

TEST(ReadAheadStates, TakeWritesTheQueuedRangesOnly) {
    ReadAheadStates states;
    auto regs = RegisterFile();
    for (u32 i = 0; i < regs.size(); ++i) {
        regs[i] = i * 3 + 1;
    }
    ASSERT_TRUE(states.Offer(regs));
    auto out = RegisterFile(0xDEAD);
    ASSERT_TRUE(states.Take(out));
    for (const auto& range : ReadAheadStates::Ranges) {
        for (u32 reg = range.first; reg < range.first + range.count; ++reg) {
            ASSERT_EQ(out[reg], regs[reg]) << std::hex << reg;
        }
    }
    EXPECT_EQ(out[OutsideRanges], 0xDEADu);
    EXPECT_EQ(out[ReadAheadStates::ShGfxFirst + ReadAheadStates::ShGfxCount], 0xDEADu)
        << "compute shader registers are not part of a state";
    EXPECT_TRUE(states.Empty());
    EXPECT_FALSE(states.Take(out));
}

TEST(ReadAheadStates, UserDataDoesNotMakeANewState) {
    ReadAheadStates states;
    auto regs = RegisterFile();
    ASSERT_TRUE(states.Offer(regs));
    regs[PsUserData] = 0x1234;
    regs[VsUserData] = 0x5678;
    EXPECT_FALSE(states.Offer(regs));
    regs[OutsideRanges] = 7;
    EXPECT_FALSE(states.Offer(regs));
    regs[ContextReg] = 1;
    EXPECT_TRUE(states.Offer(regs));
    regs[PsProgram] = 0x100;
    EXPECT_TRUE(states.Offer(regs));
    regs[Primitive] = 4;
    EXPECT_TRUE(states.Offer(regs));
    // The same programs and state again.
    regs[ContextReg] = 0;
    regs[PsProgram] = 0;
    regs[Primitive] = 0;
    EXPECT_FALSE(states.Offer(regs));
}

TEST(ReadAheadStates, AffectsPipelineMatchesTheHash) {
    EXPECT_TRUE(ReadAheadStates::AffectsPipeline(ContextReg, 1));
    EXPECT_TRUE(ReadAheadStates::AffectsPipeline(PsProgram, 2));
    EXPECT_TRUE(ReadAheadStates::AffectsPipeline(Primitive, 1));
    EXPECT_FALSE(ReadAheadStates::AffectsPipeline(PsUserData, 16));
    EXPECT_FALSE(ReadAheadStates::AffectsPipeline(VsUserData, 4));
    EXPECT_FALSE(ReadAheadStates::AffectsPipeline(OutsideRanges, 1));
    EXPECT_FALSE(ReadAheadStates::AffectsPipeline(
        ReadAheadStates::ShGfxFirst + ReadAheadStates::ShGfxCount, 8));
    // A write that spans user data into the next program registers.
    EXPECT_TRUE(ReadAheadStates::AffectsPipeline(PsUserData + 15, 2));
    // Every register AffectsPipeline rejects leaves the hash as it is.
    auto regs = RegisterFile();
    const u64 base = ReadAheadStates::PipelineStateHash(regs);
    for (u32 reg = 0; reg < regs.size(); ++reg) {
        regs[reg] = 0xFFFFFFFF;
        const bool changed = ReadAheadStates::PipelineStateHash(regs) != base;
        regs[reg] = 0;
        if (changed) {
            ASSERT_TRUE(ReadAheadStates::AffectsPipeline(reg, 1)) << std::hex << reg;
        }
    }
}

TEST(ReadAheadStates, FullQueueDoesNotMarkTheStateSeen) {
    ReadAheadStates states{2};
    auto regs = RegisterFile();
    regs[ContextReg] = 1;
    ASSERT_TRUE(states.Offer(regs));
    regs[ContextReg] = 2;
    ASSERT_TRUE(states.Offer(regs));
    EXPECT_EQ(states.Space(), 0u);
    regs[ContextReg] = 3;
    EXPECT_FALSE(states.Offer(regs));
    auto out = RegisterFile();
    ASSERT_TRUE(states.Take(out));
    EXPECT_EQ(out[ContextReg], 1u);
    EXPECT_TRUE(states.Offer(regs)) << "the state that did not fit is offered again";
    ASSERT_TRUE(states.Take(out));
    EXPECT_EQ(out[ContextReg], 2u);
    ASSERT_TRUE(states.Take(out));
    EXPECT_EQ(out[ContextReg], 3u);
}

TEST(ReadAheadStates, MissesAreSeenOnce) {
    ReadAheadStates states;
    EXPECT_FALSE(states.TakeNewMiss());
    EXPECT_FALSE(states.MissWithin(std::chrono::seconds{5}));
    states.NoteMiss();
    states.NoteMiss();
    EXPECT_TRUE(states.TakeNewMiss());
    EXPECT_FALSE(states.TakeNewMiss());
    EXPECT_TRUE(states.MissWithin(std::chrono::seconds{5}));
}

TEST(ReadAheadStates, StatesArriveInOrderAcrossThreads) {
    constexpr u32 Count = 5000;
    ReadAheadStates states{64};
    std::thread producer{[&] {
        auto regs = RegisterFile();
        for (u32 i = 1; i <= Count; ++i) {
            regs[ContextReg] = i;
            regs[PsUserData] = i * 7; // carried along, not hashed
            while (!states.Offer(regs)) {
                std::this_thread::yield();
            }
        }
    }};
    auto out = RegisterFile();
    for (u32 expected = 1; expected <= Count;) {
        if (!states.Take(out)) {
            std::this_thread::yield();
            continue;
        }
        ASSERT_EQ(out[ContextReg], expected);
        ASSERT_EQ(out[PsUserData], expected * 7);
        ++expected;
    }
    producer.join();
    EXPECT_TRUE(states.Empty());
    EXPECT_EQ(states.TakeOffered(), Count);
}
