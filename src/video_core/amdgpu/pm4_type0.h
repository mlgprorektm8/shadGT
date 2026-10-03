// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>
#include "common/types.h"

namespace AmdGpu {

struct Type0RegisterWrite {
    u32 first_register;
    std::span<const u32> values;
};

// AMD SI programming guide, section 2.1.1: COUNT is N-1 and BASE_INDEX is in dwords.
inline std::optional<Type0RegisterWrite> DecodeType0RegisterWrite(std::span<const u32> packet,
                                                                  size_t register_count) {
    if (packet.empty() || (packet.front() >> 30) != 0) {
        return std::nullopt;
    }
    const u32 first = packet.front() & 0xffff;
    const u32 count = ((packet.front() >> 16) & 0x3fff) + 1;
    if (packet.size() <= count || first >= register_count || count > register_count - first) {
        return std::nullopt;
    }
    return Type0RegisterWrite{first, packet.subspan(1, count)};
}

} // namespace AmdGpu
