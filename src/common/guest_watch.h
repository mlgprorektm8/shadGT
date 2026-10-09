// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/types.h"

// Diagnostics: log guest 64-bit values once per presented frame for a while after an event
// (e.g. GT Sport's car thumbnail render start), to see how game-side counters advance against
// rendered frames. SHADGT_WATCH_GUEST lists the values as hex offsets from the executable's load
// base, comma-separated; nothing is logged without it.
namespace Common::GuestWatch {

inline std::atomic<u32> frames_left{0};

inline const std::vector<u64>& Offsets() {
    static const std::vector<u64> offsets = [] {
        std::vector<u64> out;
        if (const char* env = std::getenv("SHADGT_WATCH_GUEST")) {
            std::string list{env};
            size_t pos = 0;
            while (pos < list.size()) {
                const size_t end = std::min(list.find(',', pos), list.size());
                out.push_back(std::strtoull(list.substr(pos, end - pos).c_str(), nullptr, 16));
                pos = end + 1;
            }
        }
        return out;
    }();
    return offsets;
}

/// Starts (or extends) logging for the next `frames` presented frames.
inline void Arm(u32 frames) {
    if (!Offsets().empty()) {
        frames_left = std::max(frames_left.load(), frames);
    }
}

} // namespace Common::GuestWatch
