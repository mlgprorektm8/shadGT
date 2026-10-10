// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>

#include "common/sampling_profiler.h"

#ifdef _WIN32

#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/thread.h"

namespace Common::SamplingProfiler {
namespace {

bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_PROFILE");
        return env && env[0] == '1';
    }();
    return enabled;
}

struct SampledThread {
    std::string name;
    HANDLE handle{};
    ULONG_PTR stack_low{};
    ULONG_PTR stack_high{};
    u64 samples{};
    std::unordered_map<u64, u64> self;      // address -> samples it was the current address
    std::unordered_map<u64, u64> inclusive; // function start -> samples it was on the stack
};

class Sampler {
public:
    static Sampler& Instance() {
        static Sampler sampler;
        return sampler;
    }

    void Register(const char* name) {
        HANDLE handle{};
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                             &handle,
                             THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                             FALSE, 0)) {
            return;
        }
        ULONG_PTR low{};
        ULONG_PTR high{};
        GetCurrentThreadStackLimits(&low, &high);
        std::scoped_lock lk{mutex};
        auto& thread = threads.emplace_back(std::make_unique<SampledThread>());
        thread->name = name;
        thread->handle = handle;
        thread->stack_low = low;
        thread->stack_high = high;
        if (!worker.joinable()) {
            worker = std::jthread{[this](std::stop_token stop) { Run(stop); }};
        }
    }

private:
    void Run(std::stop_token stop) {
        SetCurrentThreadName("shadGT:Profiler");
        SetCurrentThreadPriority(ThreadPriority::High);
        timeBeginPeriod(1);
        auto report_at = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (!stop.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
            {
                std::scoped_lock lk{mutex};
                for (auto& thread : threads) {
                    Sample(*thread);
                }
            }
            if (std::chrono::steady_clock::now() >= report_at) {
                report_at += std::chrono::seconds{10};
                Report();
            }
        }
        timeEndPeriod(1);
    }

    static void Sample(SampledThread& thread) {
        // Nothing that could take a lock the suspended thread holds (the heap's included)
        // runs until it is resumed: the frames go to a local array first.
        u64 frames[16];
        u32 depth = 0;
        if (SuspendThread(thread.handle) == static_cast<DWORD>(-1)) {
            return;
        }
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        if (GetThreadContext(thread.handle, &context)) {
            frames[depth++] = context.Rip;
            for (u32 frame = 0; frame < 15 && context.Rip != 0; ++frame) {
                // Only the thread's own stack is read.
                if (context.Rsp < thread.stack_low || context.Rsp + 8 > thread.stack_high) {
                    break;
                }
                DWORD64 image_base{};
                const auto* entry = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
                if (!entry) {
                    // A leaf without unwind data: its return address is on top of the stack.
                    if (frame != 0) {
                        break;
                    }
                    context.Rip = *reinterpret_cast<const u64*>(context.Rsp);
                    context.Rsp += 8;
                } else {
                    void* handler_data{};
                    DWORD64 establisher_frame{};
                    RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip,
                                     const_cast<PRUNTIME_FUNCTION>(entry), &context,
                                     &handler_data, &establisher_frame, nullptr);
                }
                if (context.Rip != 0) {
                    frames[depth++] = context.Rip;
                }
            }
        }
        ResumeThread(thread.handle);
        if (depth == 0) {
            return;
        }
        ++thread.samples;
        ++thread.self[frames[0]];
        u64 seen[16];
        u32 unique = 0;
        for (u32 i = 0; i < depth; ++i) {
            DWORD64 image_base{};
            const auto* entry = RtlLookupFunctionEntry(frames[i], &image_base, nullptr);
            const u64 function = entry ? image_base + entry->BeginAddress : frames[i];
            if (std::find(seen, seen + unique, function) == seen + unique) {
                seen[unique++] = function;
                ++thread.inclusive[function];
            }
        }
    }

    static std::string Describe(u64 address) {
        HMODULE module{};
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                    GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(address), &module)) {
            return fmt::format("{:#x}", address);
        }
        char path[MAX_PATH]{};
        GetModuleFileNameA(module, path, MAX_PATH);
        std::string name{path};
        if (const auto slash = name.find_last_of("\\/"); slash != std::string::npos) {
            name = name.substr(slash + 1);
        }
        return fmt::format("{}+{:#x}", name, address - reinterpret_cast<u64>(module));
    }

    static std::string Top(const std::unordered_map<u64, u64>& counts, u64 total, size_t count) {
        std::vector<std::pair<u64, u64>> sorted(counts.begin(), counts.end());
        std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string out;
        for (size_t i = 0; i < std::min(count, sorted.size()); ++i) {
            out += fmt::format(" {}={:.1f}%", Describe(sorted[i].first),
                               100.0 * sorted[i].second / std::max<u64>(total, 1));
        }
        return out;
    }

    void Report() {
        std::scoped_lock lk{mutex};
        for (auto& thread : threads) {
            if (thread->samples == 0) {
                continue;
            }
            LOG_WARNING(Debug, "DIAG-048 profile {} ({} samples), self:{}", thread->name,
                        thread->samples, Top(thread->self, thread->samples, 40));
            LOG_WARNING(Debug, "DIAG-048 profile {} ({} samples), inclusive:{}", thread->name,
                        thread->samples, Top(thread->inclusive, thread->samples, 60));
            thread->samples = 0;
            thread->self.clear();
            thread->inclusive.clear();
        }
    }

    std::mutex mutex;
    std::vector<std::unique_ptr<SampledThread>> threads;
    std::jthread worker;
};

} // namespace

void RegisterCurrentThread(const char* name) {
    if (Enabled()) {
        Sampler::Instance().Register(name);
    }
}

} // namespace Common::SamplingProfiler

#else

namespace Common::SamplingProfiler {
void RegisterCurrentThread(const char*) {}
} // namespace Common::SamplingProfiler

#endif
