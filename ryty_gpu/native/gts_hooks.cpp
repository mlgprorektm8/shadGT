// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-002: where GT Sport's graphics reach the GPU (static analysis of eboot.bin v1.69). The game
// draws through PGL, Polyphony's OpenGL-like layer over Sony's statically linked Gnm library:
// - every draw goes through one of five PGL functions, which end in Gnm's drawIndexAuto or
//   drawIndex wrapper (nothing else calls those wrappers);
// - no game code writes GPU commands itself, apart from PGL's links between command chunks;
// - compute dispatches go through four Gnm wrappers, and every graphics frame is submitted by one
//   PGL function.
// With RYGT_NATIVE_HOOKS=1 these functions are hooked and counted. Every 10 seconds the counts are
// logged next to the draws and dispatches the renderer received (NAT-002 lines), to check that the
// hooks see everything before a native renderer relies on them. Nothing else changes.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <windows.h>
#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "ryty_gpu/native/entry_hook.h"
#include "ryty_gpu/native/gts_hooks.h"

namespace Native {
namespace {

// The first instructions of the hooked functions (moved to their trampolines).
constexpr std::array<u8, 6> kPushR15 = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57}; // push rbp; mov rbp,rsp; push r15
constexpr std::array<u8, 6> kPushR14 = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56}; // ...; push r14
constexpr std::array<u8, 5> kPushRbx = {0x55, 0x48, 0x89, 0xe5, 0x53};       // ...; push rbx

enum DrawKind : int { Arrays, ArraysInstanced, Triangles32, Elements, Immediate, kDrawKinds };

struct DrawEntry {
    u64 eboot_va;
    const char* name;
    const char* label;
};
constexpr DrawEntry kDraws[kDrawKinds] = {
    {0x1c5a750, "PGL drawArrays(mode, first, count)", "arrays"},
    {0x1c5a890, "PGL drawArraysInstanced(mode, first, count, instances)", "instanced"},
    {0x1c5a9e0, "PGL drawTriangles32(count, indices)", "triangles32"},
    {0x1c5ab00, "PGL drawElements(mode, count, type, indices, offset, instances)", "elements"},
    {0x1c621b0, "PGL immediate-mode draw", "immediate"},
};

// Draw counts per game thread: which threads record draws, and how many.
struct ThreadCounts {
    std::atomic<u64> draws[kDrawKinds]{};
    u32 tid = 0;
    std::string name;
    u64 reported = 0; // total at the previous report (reporter thread only)
};
std::mutex g_threads_mutex;
std::vector<ThreadCounts*> g_threads;
thread_local ThreadCounts* t_counts = nullptr;

ThreadCounts& Counts() {
    if (t_counts == nullptr) {
        auto* counts = new ThreadCounts;
        counts->tid = GetCurrentThreadId();
        PWSTR description = nullptr;
        if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &description)) && description) {
            for (const wchar_t* c = description; *c; ++c) {
                counts->name += *c < 128 ? static_cast<char>(*c) : '?';
            }
            LocalFree(description);
        }
        std::scoped_lock lock{g_threads_mutex};
        g_threads.push_back(counts);
        t_counts = counts;
    }
    return *t_counts;
}

// The PGL draw functions take at most six integer arguments; all six registers are passed on.
using DrawFn = void(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64);
DrawFn g_draw_original[kDrawKinds]{};

// The renderer's log is ready once the game draws, so the reporter starts with the first draw.
std::atomic<bool> g_reporter_started{false};
void StartReporter();

template <int Kind>
void PS4_SYSV_ABI DrawHook(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    Counts().draws[Kind].fetch_add(1, std::memory_order_relaxed);
    if (!g_reporter_started.load(std::memory_order_relaxed)) {
        StartReporter();
    }
    g_draw_original[Kind](a, b, c, d, e, f);
}

struct CountEntryInfo {
    u64 eboot_va;
    std::span<const u8> prologue;
    const char* name;
    std::atomic<u64>* counter = nullptr;
};
enum CountKind : int {
    GnmDrawIndexAuto,
    GnmDrawIndex,
    GnmDispatchGfx,
    GnmDispatchAsync,
    GnmDispatchB,
    GnmDispatchC,
    PglSubmit,
    PglJobBegin,
    PglJobEnd,
    kCountKinds
};
CountEntryInfo g_counts[kCountKinds] = {
    {0x1b4890, kPushR14, "Gnm drawIndexAuto wrapper"},
    {0x1b4aa0, kPushR15, "Gnm drawIndex wrapper"},
    {0x1b53c0, kPushR15, "Gnm dispatch (graphics ring)"},
    {0x1d97d0, kPushR15, "Gnm dispatch (compute rings)"},
    {0x1cafa0, kPushRbx, "Gnm dispatch (b)"},
    {0x1db650, kPushRbx, "Gnm dispatch (c)"},
    {0x1c56610, kPushR14, "PGL submit"},
    // A render job takes a PGL context from a pool, makes it current on its thread with a fresh
    // command chunk and state from a template, and links its chunk after the previous job's.
    {0x1c572c0, kPushR15, "PGL job begin"},
    {0x1c57860, kPushR15, "PGL job end"},
};

u64 Count(CountKind kind) {
    return g_counts[kind].counter ? g_counts[kind].counter->load(std::memory_order_relaxed) : 0;
}

void ReporterLoop() {
    using namespace std::chrono_literals;
    const auto& work = Common::GetWorkCounters();
    u64 last_draws[kDrawKinds]{}, last_gnm = 0, last_dispatch = 0, last_submits = 0, last_jobs = 0;
    u64 last_renderer_draws = work.draws.load(), last_renderer_dispatches = work.dispatches.load();
    u64 start_renderer_draws = last_renderer_draws, start_renderer_dispatches = last_renderer_dispatches;
    while (true) {
        std::this_thread::sleep_for(10s);
        u64 draws[kDrawKinds]{};
        std::vector<std::pair<u64, std::string>> threads;
        {
            std::scoped_lock lock{g_threads_mutex};
            for (ThreadCounts* counts : g_threads) {
                u64 total = 0;
                for (int kind = 0; kind < kDrawKinds; ++kind) {
                    const u64 value = counts->draws[kind].load(std::memory_order_relaxed);
                    draws[kind] += value;
                    total += value;
                }
                if (total > counts->reported) {
                    threads.emplace_back(total - counts->reported,
                                         fmt::format("{} {}", counts->tid,
                                                     counts->name.empty() ? "?" : counts->name));
                }
                counts->reported = total;
            }
        }
        u64 game_draws = 0, window_draws = 0;
        std::string by_kind;
        for (int kind = 0; kind < kDrawKinds; ++kind) {
            game_draws += draws[kind];
            window_draws += draws[kind] - last_draws[kind];
            by_kind += fmt::format(" {} {}", kDraws[kind].label, draws[kind] - last_draws[kind]);
            last_draws[kind] = draws[kind];
        }
        const u64 gnm = Count(GnmDrawIndexAuto) + Count(GnmDrawIndex);
        const u64 dispatch = Count(GnmDispatchGfx) + Count(GnmDispatchAsync) + Count(GnmDispatchB) +
                             Count(GnmDispatchC);
        const u64 submits = Count(PglSubmit);
        const u64 jobs = Count(PglJobBegin);
        const u64 renderer_draws = work.draws.load(std::memory_order_relaxed);
        const u64 renderer_dispatches = work.dispatches.load(std::memory_order_relaxed);
        // Totals since the hooks' first report window: the renderer gets a frame's draws after
        // the game submits it, so a small, steady difference is work in flight. A growing one
        // means draws reach the GPU without passing the hooks.
        LOG_INFO(Lib_GnmDriver,
                 "NAT-002 hooks (10 s): game draws {} ({}), Gnm draw wrappers {}, renderer draws "
                 "{}; dispatches game {} renderer {}; submits {}, render jobs {}. Since start: "
                 "game draws {}, "
                 "renderer draws {} (diff {}), game dispatches {}, renderer dispatches {} (diff {})",
                 window_draws, by_kind.substr(1), gnm - last_gnm,
                 renderer_draws - last_renderer_draws, dispatch - last_dispatch,
                 renderer_dispatches - last_renderer_dispatches, submits - last_submits,
                 jobs - last_jobs, game_draws, renderer_draws - start_renderer_draws,
                 static_cast<s64>(renderer_draws - start_renderer_draws) - static_cast<s64>(game_draws),
                 dispatch, renderer_dispatches - start_renderer_dispatches,
                 static_cast<s64>(renderer_dispatches - start_renderer_dispatches) -
                     static_cast<s64>(dispatch));
        std::ranges::sort(threads, [](const auto& a, const auto& b) { return a.first > b.first; });
        std::string line;
        for (size_t i = 0; i < std::min<size_t>(threads.size(), 12); ++i) {
            line += fmt::format(" [{}: {}]", threads[i].second, threads[i].first);
        }
        LOG_INFO(Lib_GnmDriver, "NAT-002 draw threads (10 s, {} threads):{}", threads.size(), line);
        last_gnm = gnm;
        last_dispatch = dispatch;
        last_submits = submits;
        last_jobs = jobs;
        last_renderer_draws = renderer_draws;
        last_renderer_dispatches = renderer_dispatches;
    }
}

void StartReporter() {
    if (!g_reporter_started.exchange(true)) {
        std::thread{ReporterLoop}.detach();
    }
}

} // namespace

void InstallGtsHooks(u8* exe_base) {
    const char* env = std::getenv("RYGT_NATIVE_HOOKS");
    if (!env || env[0] != '1') {
        return;
    }
    static NearCode code{exe_base};
    void* const hooks[kDrawKinds] = {
        reinterpret_cast<void*>(&DrawHook<Arrays>),
        reinterpret_cast<void*>(&DrawHook<ArraysInstanced>),
        reinterpret_cast<void*>(&DrawHook<Triangles32>),
        reinterpret_cast<void*>(&DrawHook<Elements>),
        reinterpret_cast<void*>(&DrawHook<Immediate>),
    };
    int installed = 0;
    for (int kind = 0; kind < kDrawKinds; ++kind) {
        g_draw_original[kind] = reinterpret_cast<DrawFn>(
            HookEntry(code, exe_base, kDraws[kind].eboot_va, kPushR15, hooks[kind], kDraws[kind].name));
        installed += g_draw_original[kind] != nullptr;
    }
    for (CountEntryInfo& entry : g_counts) {
        entry.counter = CountEntry(code, exe_base, entry.eboot_va, entry.prologue, entry.name);
        installed += entry.counter != nullptr;
    }
    std::fprintf(stderr, "[native] NAT-002: %d of %d hooks installed\n", installed,
                 static_cast<int>(kDrawKinds) + static_cast<int>(kCountKinds));
}

} // namespace Native
