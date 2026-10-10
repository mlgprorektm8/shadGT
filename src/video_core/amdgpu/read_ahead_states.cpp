// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include <xxhash.h>

#include "video_core/amdgpu/read_ahead_states.h"

namespace AmdGpu {

u64 ReadAheadStates::PipelineStateHash(std::span<const u32> regs) {
    std::array<u32, ShGfxCount> sh;
    std::memcpy(sh.data(), regs.data() + ShGfxFirst, sizeof(sh));
    for (u32 stage_base = 0; stage_base < ShGfxCount; stage_base += 0x40) {
        std::fill_n(sh.begin() + stage_base + 0xC, 16, 0u);
    }
    std::array<u32, ContextCount> context;
    std::memcpy(context.data(), regs.data() + ContextFirst, sizeof(context));
    std::array<u32, PrimitiveCount> primitive;
    std::memcpy(primitive.data(), regs.data() + PrimitiveFirst, sizeof(primitive));
    for (const u32 reg : PerDrawRegisters) {
        if (reg >= ContextFirst && reg < ContextFirst + ContextCount) {
            context[reg - ContextFirst] = 0;
        } else if (reg >= PrimitiveFirst && reg < PrimitiveFirst + PrimitiveCount) {
            primitive[reg - PrimitiveFirst] = 0;
        }
    }
    u64 hash = XXH3_64bits(sh.data(), sizeof(sh));
    hash = XXH3_64bits_withSeed(context.data(), sizeof(context), hash);
    return XXH3_64bits_withSeed(primitive.data(), sizeof(primitive), hash);
}

ReadAheadStates::ReadAheadStates(size_t capacity_) : capacity{capacity_} {}

bool ReadAheadStates::Offer(std::span<const u32> regs) {
    if (Space() == 0 || regs.size() < RegisterFileDwords) {
        return false;
    }
    if (seen.size() >= MaxSeen) {
        seen.clear();
    }
    if (!seen.insert(PipelineStateHash(regs)).second) {
        return false;
    }
    std::vector<u32> state;
    {
        std::scoped_lock lk{mutex};
        if (!free_states.empty()) {
            state = std::move(free_states.back());
            free_states.pop_back();
        }
    }
    state.resize(StateDwords);
    u32 at = 0;
    for (const auto& range : Ranges) {
        std::memcpy(state.data() + at, regs.data() + range.first, range.count * sizeof(u32));
        at += range.count;
    }
    {
        std::scoped_lock lk{mutex};
        queue.push_back(std::move(state));
    }
    size.fetch_add(1, std::memory_order_release);
    offered.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool ReadAheadStates::Take(std::span<u32> regs) {
    if (regs.size() < RegisterFileDwords) {
        return false;
    }
    std::vector<u32> state;
    {
        std::scoped_lock lk{mutex};
        if (queue.empty()) {
            return false;
        }
        state = std::move(queue.front());
        queue.pop_front();
    }
    u32 at = 0;
    for (const auto& range : Ranges) {
        std::memcpy(regs.data() + range.first, state.data() + at, range.count * sizeof(u32));
        at += range.count;
    }
    size.fetch_sub(1, std::memory_order_release);
    std::scoped_lock lk{mutex};
    free_states.push_back(std::move(state));
    return true;
}

} // namespace AmdGpu
