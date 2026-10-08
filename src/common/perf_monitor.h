// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <string>
#if defined(_WIN32) || defined(__x86_64__)
#include <intrin.h>
#endif
#include "common/types.h"

namespace Common {

/// PERF-DIAG-007: CPU time used by each thread of this process since the previous call,
/// as "name pct%" entries for the busiest threads (percent of one core), plus the process total.
/// Returns an empty string where unsupported.
std::string SampleThreadCpuUsage(size_t max_threads);

/// PERF-DIAG-010: work done by the emulator's GPU frontend, read and reset by the monitor.
struct WorkCounters {
    std::atomic<u64> pm4_packets{};
    std::atomic<u64> draws{};
    std::atomic<u64> dispatches{};
    std::atomic<u64> find_image{};
    std::atomic<u64> obtain_buffer{};
    std::atomic<u64> obtain_stream{};
    std::atomic<u64> uploads{};
    std::atomic<u64> upload_bytes{};
    std::atomic<u64> protects{};
    std::atomic<u64> protect_bytes{};
    std::atomic<u64> shaders_compiled{};
    std::atomic<u64> pipelines_compiled{};
    std::atomic<u64> upload_epochs{};
    std::atomic<u64> waits_skipped{};
    // DIAG-033: where the time of a long frame went (cumulative, like the counters above).
    std::atomic<u64> gpu_waits{};
    std::atomic<u64> gpu_wait_us{};
    std::atomic<u64> frontend_waits{};
    std::atomic<u64> frontend_wait_us{};
    std::atomic<u64> pipeline_waits{};
    std::atomic<u64> pipeline_wait_us{};
    std::atomic<u64> file_reads{};
    std::atomic<u64> file_read_bytes{};
    std::atomic<u64> file_read_us{};
};
WorkCounters& GetWorkCounters();

/// Experimental-branch switch: false when the PERF id (e.g. 14 for PERF-014) is listed in the
/// SHADGT_DISABLE_PERF environment variable (comma separated), for A/B runs without rebuilding.
bool PerfFeatureEnabled(u32 id);

/// PERF-DIAG-011: command thread time per step of a draw, in TSC ticks, reported every 2 s.
enum class Phase : u32 {
    DrawSetup,
    Pipeline,
    RenderTargets,
    VertexIndex,
    Buffers,
    Textures,
    TextureRebind,
    BeginRendering,
    Descriptors,
    DynamicState,
    Record,
    DrawTotal,
    DispatchTotal,
    Submit,
    Count,
};
std::array<std::atomic<u64>, size_t(Phase::Count)>& GetPhaseTicks();

/// DIAG-033: called for every presented game frame. A frame that took longer than
/// SHADGT_STALL_MS (default 100) is logged with what the emulator did during it.
void NoteGameFrame();

// PERF-026: the draw-step timers run about 14 times per draw, so the counter read and the
// tick array are inline instead of function calls.
inline std::array<std::atomic<u64>, size_t(Phase::Count)> g_phase_ticks{};

class PhaseTimer {
public:
    explicit PhaseTimer(Phase phase_) : phase{phase_}, start{ReadTsc()} {}
    ~PhaseTimer() {
        g_phase_ticks[size_t(phase)].fetch_add(ReadTsc() - start, std::memory_order_relaxed);
    }
    static u64 ReadTsc() {
#if defined(_WIN32) || defined(__x86_64__)
        return __rdtsc();
#else
        return ReadTscFallback();
#endif
    }
    static u64 ReadTscFallback();

private:
    Phase phase;
    u64 start;
};
std::string TakeWorkCounters();

} // namespace Common
