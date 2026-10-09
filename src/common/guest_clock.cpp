// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>

#include "common/guest_clock.h"
#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "common/rdtsc.h"

namespace Common::GuestClock {

namespace {

PausableCounter& Counter() {
    static PausableCounter counter;
    return counter;
}

thread_local bool is_gpu_command_thread = false;

std::atomic<u64> long_holds{};
std::atomic<double> total_ms{};

} // namespace

bool Enabled() {
    static const bool enabled = PerfFeatureEnabled(45);
    return enabled;
}

u64 ReadTsc() {
    if (!Enabled()) {
        return FencedRDTSC();
    }
    return Counter().Read([] { return FencedRDTSC(); });
}

bool IsPaused() {
    return Enabled() && Counter().IsPaused();
}

void MarkGpuCommandThread() {
    is_gpu_command_thread = true;
}

CompilePause::CompilePause() : active{is_gpu_command_thread && Enabled()} {
    if (active) {
        start = std::chrono::steady_clock::now();
        Counter().Pause(FencedRDTSC());
    }
}

CompilePause::~CompilePause() {
    if (!active) {
        return;
    }
    if (Counter().Resume(FencedRDTSC()) == 0) {
        return; // An outer hold continues.
    }
    // The longer holds are the ones the game would otherwise have seen as long frames.
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    total_ms += ms;
    if (ms >= 50.0) {
        const u64 count = ++long_holds;
        if (count <= 40 || count % 100 == 0) {
            LOG_WARNING(Common,
                        "FIX-044: guest clocks held {:.0f} ms while the GPU thread compiled "
                        "({} holds of 50 ms or more, {:.0f} ms held in all)",
                        ms, count, total_ms.load());
        }
    }
}

} // namespace Common::GuestClock
