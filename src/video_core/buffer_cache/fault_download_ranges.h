// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <span>
#include <utility>
#include <vector>

#include "common/types.h"

namespace VideoCore {

/// PERF-052: the parts of a downloaded range [start, end) that no newer GPU write overlaps.
/// Only those may stop being GPU-written once the download reached guest memory; a newer write's
/// bytes stay GPU-written, with a stale guest copy, as after any GPU write.
inline std::vector<std::pair<VAddr, VAddr>> PartsWithoutNewerWrites(
    VAddr start, VAddr end, std::span<const std::pair<VAddr, VAddr>> newer) {
    std::vector<std::pair<VAddr, VAddr>> keep;
    for (const auto& [w_start, w_end] : newer) {
        const VAddr s = std::max(start, w_start);
        const VAddr e = std::min(end, w_end);
        if (s < e) {
            keep.emplace_back(s, e);
        }
    }
    std::ranges::sort(keep);
    std::vector<std::pair<VAddr, VAddr>> parts;
    VAddr cursor = start;
    for (const auto& [s, e] : keep) {
        if (s > cursor) {
            parts.emplace_back(cursor, s);
        }
        cursor = std::max(cursor, e);
    }
    if (cursor < end) {
        parts.emplace_back(cursor, end);
    }
    return parts;
}

/// PERF-052: whether [start, end) overlaps any newer GPU write.
inline bool TouchedByNewerWrite(VAddr start, VAddr end,
                                std::span<const std::pair<VAddr, VAddr>> newer) {
    return std::ranges::any_of(newer, [&](const auto& w) { return w.first < end && start < w.second; });
}

} // namespace VideoCore
