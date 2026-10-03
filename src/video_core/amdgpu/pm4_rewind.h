// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstring>
#include <span>
#include "common/types.h"

namespace AmdGpu {

// REWIND's payload valid bit publishes commands that the guest writes after submission.
// A snapshot must poll the original payload and refresh the remaining stream after publication.
inline bool RefreshRewindTailIfReady(std::span<u32> snapshot, std::span<const u32> live,
                                     size_t rewind_offset) {
    if (snapshot.size() != live.size() || rewind_offset >= live.size() ||
        live.size() - rewind_offset < 2) {
        return false;
    }
    const auto payload = std::atomic_ref<const u32>(live[rewind_offset + 1]);
    if ((payload.load(std::memory_order_acquire) & 0x80000000u) == 0) {
        return false;
    }
    if (snapshot.data() != live.data()) {
        std::memcpy(snapshot.data() + rewind_offset, live.data() + rewind_offset,
                    (live.size() - rewind_offset) * sizeof(u32));
    }
    return true;
}

} // namespace AmdGpu
