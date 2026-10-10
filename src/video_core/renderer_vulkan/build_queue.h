// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <stop_token>

#include "common/types.h"

namespace Vulkan {

/// The pipeline build workers' jobs (PERF-019). Urgent jobs (a draw waits) run first, then
/// normal ones (read-ahead) in order. PERF-038: background jobs (stored pipelines built while
/// the game runs) run only when neither waits, on at most `max_background` workers at once,
/// and not while paused: each pipeline a draw needs pauses them for a while, so they never
/// compete with the builds the game is waiting for.
class BuildQueue {
public:
    using Job = std::function<void()>;
    using Clock = std::chrono::steady_clock;
    enum class Kind { Urgent, Normal, Background };

    explicit BuildQueue(u32 max_background_) : max_background{max_background_} {}

    void Push(Job&& job, Kind kind) {
        {
            std::scoped_lock lk{mutex};
            switch (kind) {
            case Kind::Urgent:
                jobs.push_front(std::move(job));
                break;
            case Kind::Normal:
                jobs.push_back(std::move(job));
                break;
            case Kind::Background:
                background.push_back(std::move(job));
                break;
            }
        }
        cv.notify_one();
    }

    /// Waits for the next job. False once stopped. `is_background` tells which queue it came
    /// from; a background job must be followed by FinishedBackground.
    bool Pop(Job& job, bool& is_background, std::stop_token stop) {
        std::unique_lock lk{mutex};
        while (true) {
            if (stop.stop_requested()) {
                return false;
            }
            if (!jobs.empty()) {
                job = std::move(jobs.front());
                jobs.pop_front();
                is_background = false;
                return true;
            }
            const auto now = Clock::now();
            const bool paused = now < pause_until;
            if (!background.empty() && !paused && running_background < max_background) {
                job = std::move(background.front());
                background.pop_front();
                ++running_background;
                is_background = true;
                return true;
            }
            if (!background.empty() && paused) {
                cv.wait_until(lk, stop, pause_until, [&] {
                    return !jobs.empty() || Clock::now() >= pause_until;
                });
            } else {
                cv.wait(lk, stop, [&] {
                    return !jobs.empty() ||
                           (!background.empty() && running_background < max_background);
                });
            }
        }
    }

    void FinishedBackground() {
        {
            std::scoped_lock lk{mutex};
            --running_background;
        }
        cv.notify_one();
    }

    /// No background job starts before `until` (jobs already running finish).
    void PauseBackground(Clock::time_point until) {
        std::scoped_lock lk{mutex};
        pause_until = std::max(pause_until, until);
    }

    size_t BackgroundQueued() {
        std::scoped_lock lk{mutex};
        return background.size();
    }

    /// Wakes the workers waiting for jobs or a pause, so they can see a stop request.
    void NotifyAll() {
        cv.notify_all();
    }

private:
    const u32 max_background;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<Job> jobs;
    std::deque<Job> background;
    u32 running_background{};
    Clock::time_point pause_until{};
};

} // namespace Vulkan
