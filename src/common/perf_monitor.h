// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <string>
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
};
WorkCounters& GetWorkCounters();

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

class PhaseTimer {
public:
    explicit PhaseTimer(Phase phase_) : phase{phase_}, start{ReadTsc()} {}
    ~PhaseTimer() {
        GetPhaseTicks()[size_t(phase)].fetch_add(ReadTsc() - start, std::memory_order_relaxed);
    }
    static u64 ReadTsc();

private:
    Phase phase;
    u64 start;
};
std::string TakeWorkCounters();

} // namespace Common
