// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "common/types.h"

namespace Vulkan {

/// PERF-032: whether a stored SPIR-V module is whole before it goes to the driver: the SPIR-V
/// magic, instructions whose word counts end exactly at the end of the data, and OpFunctionEnd
/// last (modules end with their last function). A file cut short when a session was killed while
/// storing it fails this.
inline bool IsCompleteSpirv(std::span<const u32> words) {
    constexpr u32 Magic = 0x07230203;
    constexpr u32 HeaderWords = 5;
    constexpr u32 OpFunctionEnd = 56;
    if (words.size() <= HeaderWords || words[0] != Magic) {
        return false;
    }
    size_t at = HeaderWords;
    u32 last_opcode = 0;
    while (at < words.size()) {
        const u32 count = words[at] >> 16;
        if (count == 0 || count > words.size() - at) {
            return false;
        }
        last_opcode = words[at] & 0xFFFF;
        at += count;
    }
    return last_opcode == OpFunctionEnd;
}

} // namespace Vulkan
