// SPDX-FileCopyrightText: Copyright 2026 shadGT
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>
#include <random>
#include "common/types.h"

namespace Common {

/// FIX-021: random bytes from the operating system for the guest's random sources. std::rand()
/// keeps its state per thread on Windows and was seeded on another thread, so every launch gave
/// the game the same "random" numbers (GT Sport played the same menu song each time).
inline void FillRandomBytes(void* buffer, u64 size) {
    thread_local std::random_device device;
    auto* out = static_cast<u8*>(buffer);
    while (size != 0) {
        const u32 value = device();
        const u64 count = size < sizeof(value) ? size : sizeof(value);
        std::memcpy(out, &value, count);
        out += count;
        size -= count;
    }
}

} // namespace Common
