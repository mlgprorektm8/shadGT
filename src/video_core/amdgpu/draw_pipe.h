// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "common/types.h"
#include "common/unique_function.h"

namespace AmdGpu {

/// PERF-031: the second stage of the GPU command processor. The command thread (stage A) keeps
/// decoding PM4 and the register file; the work that records Vulkan commands and touches the
/// buffer and texture caches (draws, dispatches, copies, fences) runs here, in submission
/// order, on the recorder thread (stage B). Anything else that needs the rasterizer drains the
/// pipe first, so the caches only ever have one user.
///
/// Design after the two-stage pipeline of the Bloodborne PC port (bbport, GPL-2.0-or-later,
/// https://github.com/deadinside28/bloodborne_pc, docs/parallel_gpu.md); written for shadGT.
class DrawPipe {
public:
    using Work = Common::UniqueFunction<void>;

    /// `thread_init` runs first on the recorder thread (thread name, thread-local state).
    explicit DrawPipe(std::function<void()> thread_init, size_t max_pending = 4096);
    ~DrawPipe();

    DrawPipe(const DrawPipe&) = delete;
    DrawPipe& operator=(const DrawPipe&) = delete;

    /// Queues work behind everything pushed before it. Waits while `max_pending` jobs are queued.
    void Push(Work&& work);

    /// Returns once every job pushed so far has finished. A no-op on the recorder thread.
    /// PERF-050: the caller (the command thread) then owns the caches until its next Push or
    /// ReleaseExclusive; urgent work waits for it or is run by it (RunUrgentIfExclusive).
    void Drain();

    /// PERF-050: runs `work` as soon as the caches are free: on the recorder between two jobs
    /// (not behind the queued ones), or on the command thread while it owns them after a drain.
    /// Any thread; does not wait.
    void PushUrgent(Work&& work);

    /// Recorder thread, at a point where no job is half done: runs the queued urgent work.
    void RunUrgent();

    /// Command thread: runs the queued urgent work if it owns the caches after a drain.
    void RunUrgentIfExclusive();

    /// Command thread: gives up ownership of the caches taken by Drain (before it goes idle).
    void ReleaseExclusive();

    bool OnRecorderThread() const {
        return std::this_thread::get_id() == recorder_id.load(std::memory_order_acquire);
    }

    std::thread::id RecorderThreadId() const {
        return recorder_id.load(std::memory_order_acquire);
    }

    struct Stats {
        u64 pushed;
        u64 drains;
        u64 drains_that_waited;
        u64 drain_wait_us;
        u64 push_waits;
        u64 recorder_busy_us;
        size_t max_queued;
    };
    /// The counters since the previous call (resets them).
    Stats TakeStats();

private:
    void Run(std::stop_token stop);

    std::mutex mutex;
    std::condition_variable_any work_cv;
    std::condition_variable_any done_cv;
    std::deque<Work> queue;
    // PERF-050
    std::deque<Work> urgent;
    std::atomic<bool> has_urgent{};
    bool exclusive{};
    bool urgent_running{};
    u64 pushed{};
    u64 finished{};
    bool recorder_waiting{};
    const size_t max_pending;
    std::atomic<std::thread::id> recorder_id{};
    std::function<void()> thread_init;
    Stats stats{};
    std::jthread recorder;
};

} // namespace AmdGpu
