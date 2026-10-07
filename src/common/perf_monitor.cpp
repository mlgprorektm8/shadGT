// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>

#include "common/perf_monitor.h"
#include "common/types.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace Common {

#ifdef _WIN32
namespace {

struct ThreadSample {
    HANDLE handle{};
    u64 cpu_100ns{};
    bool seen{};
};

u64 FileTimeTo100ns(const FILETIME& time) {
    return (u64(time.dwHighDateTime) << 32) | time.dwLowDateTime;
}

std::string ThreadName(HANDLE handle, DWORD tid) {
    PWSTR description = nullptr;
    std::string name;
    if (SUCCEEDED(GetThreadDescription(handle, &description)) && description) {
        for (const wchar_t* c = description; *c; ++c) {
            name.push_back(*c < 0x80 ? static_cast<char>(*c) : '?');
        }
        LocalFree(description);
    }
    if (name.empty()) {
        name = fmt::format("tid{}", tid);
    }
    return name;
}

} // namespace

std::string SampleThreadCpuUsage(size_t max_threads) {
    static std::mutex mutex;
    std::scoped_lock lk{mutex};
    static std::unordered_map<DWORD, ThreadSample> threads;
    static auto last_time = std::chrono::steady_clock::now();

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_100ns =
        std::chrono::duration<double>(now - last_time).count() * 10'000'000.0;
    last_time = now;

    for (auto& [tid, sample] : threads) {
        sample.seen = false;
    }
    struct Usage {
        double percent;
        DWORD tid;
        HANDLE handle;
    };
    std::vector<Usage> usages;
    double total_percent = 0.0;

    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return {};
    }
    const DWORD pid = GetCurrentProcessId();
    THREADENTRY32 entry{.dwSize = sizeof(THREADENTRY32)};
    for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != pid) {
            continue;
        }
        auto [it, inserted] = threads.try_emplace(entry.th32ThreadID);
        auto& sample = it->second;
        if (inserted) {
            sample.handle =
                OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
        }
        if (!sample.handle) {
            continue;
        }
        FILETIME creation, exit, kernel, user;
        if (!GetThreadTimes(sample.handle, &creation, &exit, &kernel, &user)) {
            continue;
        }
        sample.seen = true;
        const u64 cpu = FileTimeTo100ns(kernel) + FileTimeTo100ns(user);
        if (!inserted && elapsed_100ns > 0.0 && cpu >= sample.cpu_100ns) {
            const double percent = double(cpu - sample.cpu_100ns) * 100.0 / elapsed_100ns;
            total_percent += percent;
            usages.push_back({percent, entry.th32ThreadID, sample.handle});
        }
        sample.cpu_100ns = cpu;
    }
    CloseHandle(snapshot);

    std::erase_if(threads, [](const auto& pair) {
        if (!pair.second.seen && pair.second.handle) {
            CloseHandle(pair.second.handle);
        }
        return !pair.second.seen;
    });

    std::ranges::sort(usages, std::greater{}, &Usage::percent);
    std::string summary = fmt::format("process {:.0f}% of {}%;", total_percent,
                                      std::max(1u, std::thread::hardware_concurrency()) * 100);
    for (size_t i = 0; i < std::min(max_threads, usages.size()); ++i) {
        summary += fmt::format(" {} {:.0f}%", ThreadName(usages[i].handle, usages[i].tid),
                               usages[i].percent);
    }
    return summary;
}
#else
std::string SampleThreadCpuUsage(size_t) {
    return {};
}
#endif

WorkCounters& GetWorkCounters() {
    static WorkCounters counters;
    return counters;
}

std::string TakeWorkCounters() {
    auto& c = GetWorkCounters();
    return fmt::format("{} PM4 packets, {} draws, {} dispatches, {} image lookups, {} buffer "
                       "binds ({} streamed), {} uploads ({} KB), {} protection calls ({} KB)",
                       c.pm4_packets.exchange(0), c.draws.exchange(0), c.dispatches.exchange(0),
                       c.find_image.exchange(0), c.obtain_buffer.exchange(0),
                       c.obtain_stream.exchange(0), c.uploads.exchange(0),
                       c.upload_bytes.exchange(0) / 1024, c.protects.exchange(0),
                       c.protect_bytes.exchange(0) / 1024);
}

} // namespace Common
