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
#include <map>
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
        return env && (env[0] == '1' || env[0] == '2');
    }();
    return enabled;
}

struct SampledThread {
    std::string name;
    HANDLE handle{};
    ULONG_PTR stack_low{};
    ULONG_PTR stack_high{};
    u64 samples{};
    u32 period{1}; // sampled every this many ticks (1 ms)
    std::unordered_map<u64, u64> self;      // address -> samples it was the current address
    std::unordered_map<u64, u64> inclusive; // function start -> samples it was on the stack
    std::map<std::vector<u64>, u64> stacks;  // function starts, innermost first -> samples
};

class Sampler {
public:
    static Sampler& Instance() {
        static Sampler sampler;
        return sampler;
    }

    void Register(const char* name, ULONG_PTR low, ULONG_PTR high, u32 period) {
        HANDLE handle{};
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                             &handle,
                             THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                             FALSE, 0)) {
            return;
        }
        std::scoped_lock lk{mutex};
        auto& thread = threads.emplace_back(std::make_unique<SampledThread>());
        thread->name = name;
        thread->handle = handle;
        thread->stack_low = low;
        thread->stack_high = high;
        thread->period = period;
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
        u64 tick = 0;
        while (!stop.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
            ++tick;
            {
                std::scoped_lock lk{mutex};
                for (auto& thread : threads) {
                    if (tick % thread->period == 0) {
                        Sample(*thread);
                    }
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
        constexpr u32 MaxFrames = 16;
        u64 frames[MaxFrames];
        u32 depth = 0;
        if (SuspendThread(thread.handle) == static_cast<DWORD>(-1)) {
            return; // exited
        }
        CONTEXT context{};
        context.ContextFlags = CONTEXT_FULL;
        if (GetThreadContext(thread.handle, &context)) {
            frames[depth++] = context.Rip;
            u64 last_rsp = 0;
            for (u32 frame = 0; frame < MaxFrames - 1 && context.Rip != 0; ++frame) {
                // Only the thread's own stack is read.
                if (context.Rsp < thread.stack_low || context.Rsp + 8 > thread.stack_high ||
                    context.Rsp <= last_rsp) {
                    break;
                }
                last_rsp = context.Rsp;
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
        u64 seen[MaxFrames];
        u32 unique = 0;
        std::vector<u64> stack;
        stack.reserve(depth);
        for (u32 i = 0; i < depth; ++i) {
            DWORD64 image_base{};
            const auto* entry = RtlLookupFunctionEntry(frames[i], &image_base, nullptr);
            const u64 function = entry ? image_base + entry->BeginAddress : frames[i];
            stack.push_back(function);
            if (std::find(seen, seen + unique, function) == seen + unique) {
                seen[unique++] = function;
                ++thread.inclusive[function];
            }
        }
        // Call paths: the innermost 12 functions.
        if (stack.size() > 12) {
            stack.resize(12);
        }
        ++thread.stacks[std::move(stack)];
    }

public:
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

private:
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
            const bool guest = thread->period != 1;
            LOG_WARNING(Debug, "DIAG-048 profile {} ({} samples), self:{}", thread->name,
                        thread->samples, Top(thread->self, thread->samples, guest ? 12 : 40));
            LOG_WARNING(Debug, "DIAG-048 profile {} ({} samples), inclusive:{}", thread->name,
                        thread->samples, Top(thread->inclusive, thread->samples, guest ? 20 : 60));
            if (!guest) {
                std::vector<std::pair<u64, const std::vector<u64>*>> sorted;
                for (const auto& [stack, count] : thread->stacks) {
                    sorted.emplace_back(count, &stack);
                }
                std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.first > b.first; });
                for (size_t i = 0; i < std::min<size_t>(sorted.size(), 60); ++i) {
                    std::string path;
                    for (const u64 function : *sorted[i].second) {
                        path += ' ' + Describe(function);
                    }
                    LOG_WARNING(Debug, "DIAG-048 stack {} ({} samples) {:.1f}%:{}", thread->name,
                                thread->samples,
                                100.0 * sorted[i].first / std::max<u64>(thread->samples, 1), path);
                }
            }
            thread->samples = 0;
            thread->self.clear();
            thread->inclusive.clear();
            thread->stacks.clear();
        }
    }

    std::mutex mutex;
    std::vector<std::unique_ptr<SampledThread>> threads;
    std::jthread worker;
};

} // namespace

void RegisterCurrentThread(const char* name) {
    if (Enabled()) {
        ULONG_PTR low{};
        ULONG_PTR high{};
        GetCurrentThreadStackLimits(&low, &high);
        Sampler::Instance().Register(name, low, high, 1);
    }
}

void RegisterGuestThread(const char* name, const void* stack_low, const void* stack_high) {
    // Game threads only with SHADGT_PROFILE=2: sampling them costs time.
    static const bool guests = [] {
        const char* env = std::getenv("SHADGT_PROFILE");
        return env && env[0] == '2';
    }();
    if (Enabled() && guests) {
        // Game threads run on their own stacks, and there are many: sampled every 4 ms.
        ULONG_PTR low = reinterpret_cast<ULONG_PTR>(stack_low);
        ULONG_PTR high = reinterpret_cast<ULONG_PTR>(stack_high);
        if (!stack_low) {
            GetCurrentThreadStackLimits(&low, &high);
        }
        Sampler::Instance().Register(fmt::format("guest:{}", name).c_str(), low, high, 4);
    }
}

std::string DescribeCode(u64 address) {
    return Sampler::Describe(address);
}

std::string DescribeStack() {
    void* frames[24];
    const USHORT count = RtlCaptureStackBackTrace(1, 24, frames, nullptr);
    std::string out;
    for (USHORT i = 0; i < count; ++i) {
        out += ' ' + Sampler::Describe(reinterpret_cast<u64>(frames[i]));
    }
    return out;
}

} // namespace Common::SamplingProfiler

#else

#include <fmt/format.h>

namespace Common::SamplingProfiler {
void RegisterCurrentThread(const char*) {}
void RegisterGuestThread(const char*, const void*, const void*) {}
std::string DescribeCode(u64 address) {
    return fmt::format("{:#x}", address);
}
std::string DescribeStack() {
    return {};
}
} // namespace Common::SamplingProfiler

#endif
