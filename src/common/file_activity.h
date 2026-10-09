// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>

#include "common/types.h"

// Guest file activity for diagnostics: the reads and opens in progress, and per-window counts,
// so stalls and late results (e.g. a streamed asset not loaded when GT Sport renders the car
// thumbnail) can be lined up with the file system in Warning-level logs.
namespace Common::FileActivity {

using Clock = std::chrono::steady_clock;

struct Operation {
    const char* kind;
    std::string path;
    u64 bytes;
    Clock::time_point start;
};

struct State {
    std::mutex mutex;
    std::unordered_map<u64, Operation> in_flight;
    u64 next_id = 0;
    u64 reads = 0;
    u64 opens = 0;
    u64 read_bytes = 0;
    double slowest_ms = 0.0;
    std::string slowest;
};

inline State& GetState() {
    static State state;
    return state;
}

/// Milliseconds since the first call, to line up log lines that have no timestamps.
inline u64 NowMs() {
    static const auto origin = Clock::now();
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - origin).count());
}

class Scope {
public:
    Scope(const char* kind, std::string_view path, u64 bytes) {
        auto& state = GetState();
        std::scoped_lock lk{state.mutex};
        id = ++state.next_id;
        state.in_flight.emplace(id, Operation{kind, std::string(path), bytes, Clock::now()});
    }
    ~Scope() {
        auto& state = GetState();
        std::scoped_lock lk{state.mutex};
        const auto it = state.in_flight.find(id);
        if (it == state.in_flight.end()) {
            return;
        }
        const auto& op = it->second;
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - op.start).count();
        if (op.kind[0] == 'o') {
            ++state.opens;
        } else {
            ++state.reads;
            state.read_bytes += op.bytes;
        }
        if (ms > state.slowest_ms) {
            state.slowest_ms = ms;
            state.slowest = fmt::format("{} {}", op.kind, op.path);
        }
        state.in_flight.erase(it);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    u64 id;
};

/// The operations in progress (oldest first, at most 8).
inline std::string InFlight() {
    auto& state = GetState();
    std::scoped_lock lk{state.mutex};
    std::vector<const Operation*> ops;
    for (const auto& [id, op] : state.in_flight) {
        ops.push_back(&op);
    }
    std::ranges::sort(ops, [](const Operation* a, const Operation* b) { return a->start < b->start; });
    const auto now = Clock::now();
    std::string out = fmt::format("{} in flight", ops.size());
    for (size_t i = 0; i < ops.size() && i < 8; ++i) {
        out += fmt::format(
            "; {} {} ({} bytes, {:.0f} ms)", ops[i]->kind, ops[i]->path, ops[i]->bytes,
            std::chrono::duration<double, std::milli>(now - ops[i]->start).count());
    }
    return out;
}

/// Counts since the last report, plus the operations in progress; resets the counts.
inline std::string Report() {
    std::string counts;
    {
        auto& state = GetState();
        std::scoped_lock lk{state.mutex};
        counts = fmt::format("{} opens, {} reads ({} KB), slowest {:.1f} ms ({})", state.opens,
                             state.reads, state.read_bytes / 1024, state.slowest_ms,
                             state.slowest.empty() ? "none" : state.slowest);
        state.opens = state.reads = state.read_bytes = 0;
        state.slowest_ms = 0.0;
        state.slowest.clear();
    }
    return fmt::format("t={} ms: {}; {}", NowMs(), counts, InFlight());
}

} // namespace Common::FileActivity
