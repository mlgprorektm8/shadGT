// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>

#include "common/types.h"

// FIX-044: the guest's monotonic clocks (TSC, process time counter, process time, monotonic
// clock_gettime) stand still while the GPU command thread is blocked translating shaders or
// building pipelines. A PS4 runs precompiled shaders, so these host pauses never happen there;
// without this the game sees frames hundreds of milliseconds long. GT Sport's car thumbnail
// script then polls a shot request after one 0.1 s sleep before the game loop could post it,
// and shows BREAK!. Real (wall) time and host waits are unchanged.
namespace Common::GuestClock {

/// A host counter as the guest sees it: frozen while paused, and minus all paused time after.
/// Readers are lock-free (a sequence lock); Pause/Resume may nest and come from any thread.
class PausableCounter {
public:
    /// The guest value of the host counter; `read_host` reads it inside the consistent window,
    /// so a value read just before a resume is never combined with the larger paused total.
    template <typename ReadHost>
    u64 Read(ReadHost&& read_host) const {
        for (;;) {
            const u32 begin = sequence.load(std::memory_order_acquire);
            if (begin & 1) {
                continue;
            }
            const u64 host = read_host();
            const u64 at = paused_at.load(std::memory_order_relaxed);
            const u64 total = paused_total.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (sequence.load(std::memory_order_relaxed) != begin) {
                continue;
            }
            const u64 value =
                at != 0 ? at : std::max(host, last_resume.load(std::memory_order_relaxed));
            return value - total;
        }
    }

    void Pause(u64 host) {
        std::scoped_lock lk{writer};
        if (depth++ != 0) {
            return;
        }
        Write([&] { paused_at.store(std::max<u64>(host, 1), std::memory_order_relaxed); });
    }

    /// Returns the length of the pause that ended, or 0 when an outer pause continues.
    u64 Resume(u64 host) {
        std::scoped_lock lk{writer};
        if (depth == 0 || --depth != 0) {
            return 0;
        }
        const u64 at = paused_at.load(std::memory_order_relaxed);
        const u64 end = std::max(host, at);
        Write([&] {
            paused_total.store(paused_total.load(std::memory_order_relaxed) + (end - at),
                               std::memory_order_relaxed);
            last_resume.store(end, std::memory_order_relaxed);
            paused_at.store(0, std::memory_order_relaxed);
        });
        return end - at;
    }

    bool IsPaused() const {
        return paused_at.load(std::memory_order_acquire) != 0;
    }

    u64 PausedTotal() const {
        return paused_total.load(std::memory_order_acquire);
    }

private:
    template <typename Func>
    void Write(Func&& func) {
        const u32 s = sequence.load(std::memory_order_relaxed);
        sequence.store(s + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        func();
        sequence.store(s + 2, std::memory_order_release);
    }

    std::atomic<u32> sequence{};
    std::atomic<u64> paused_at{};
    std::atomic<u64> paused_total{};
    std::atomic<u64> last_resume{};
    u32 depth{};
    std::mutex writer;
};

/// True when FIX-044 is on (perf id 45; -DisablePerf 45 turns it off).
bool Enabled();

/// The guest's TSC: the host TSC, held while the GPU command thread compiles.
u64 ReadTsc();

/// True while the guest clocks are held.
bool IsPaused();

/// Marks the calling thread as the GPU command thread, whose compile waits hold the clocks.
void MarkGpuCommandThread();

/// Holds the guest clocks for its lifetime when created on the GPU command thread.
class CompilePause {
public:
    CompilePause();
    ~CompilePause();
    CompilePause(const CompilePause&) = delete;
    CompilePause& operator=(const CompilePause&) = delete;

private:
    bool active{};
    std::chrono::steady_clock::time_point start{};
};

} // namespace Common::GuestClock
