// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/types.h"
#include <thread>

#if defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Common {

/// PERF-060: a lock held for a few microseconds at a time by two busy threads. A waiter spins
/// briefly, then yields, instead of sleeping in the kernel (a std::mutex waiter on Windows is
/// woken tens of microseconds after the release). BasicLockable.
class SpinMutex {
public:
    void lock() noexcept {
        for (u32 spins = 0; flag.exchange(true, std::memory_order_acquire); ++spins) {
            while (flag.load(std::memory_order_relaxed)) {
                if (spins < 4096) {
#if defined(_MSC_VER) || defined(__x86_64__)
                    _mm_pause();
#endif
                    ++spins;
                } else {
                    std::this_thread::yield();
                }
            }
        }
    }

    bool try_lock() noexcept {
        return !flag.load(std::memory_order_relaxed) &&
               !flag.exchange(true, std::memory_order_acquire);
    }

    void unlock() noexcept {
        flag.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> flag{false};
};

} // namespace Common
