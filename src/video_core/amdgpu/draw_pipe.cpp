// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>

#include <immintrin.h>
#include "common/nvtx.h"
#include "common/perf_monitor.h"
#include "common/thread.h"
#include "video_core/amdgpu/draw_pipe.h"

namespace AmdGpu {

namespace {
u64 MicrosecondsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}
} // namespace

DrawPipe::DrawPipe(std::function<void()> thread_init_, size_t max_pending_)
    : max_pending{max_pending_}, thread_init{std::move(thread_init_)} {
    recorder = std::jthread{[this](std::stop_token stop) { Run(stop); }};
    // Push and Drain may need the recorder's id (OnRecorderThread) as soon as this returns.
    while (recorder_id.load(std::memory_order_acquire) == std::thread::id{}) {
        std::this_thread::yield();
    }
    // PERF-067: the buffer stage thread does each draw's buffer work ahead of the recorder.
    // -DisablePerf 74 keeps it off.
    if (const char* env = std::getenv("SHADGT_BUFFER_STAGE");
        env && env[0] == '1' && Common::PerfFeatureEnabled(74)) {
        stage_enabled = true;
        stage = std::jthread{[this](std::stop_token stop) { RunStage(stop); }};
    }
}

DrawPipe::~DrawPipe() {
    Drain();
    if (stage_enabled) {
        stage.request_stop();
        {
            std::scoped_lock lk{stage_mutex};
            stage_cv.notify_all();
        }
        stage.join();
    }
    recorder.request_stop();
    {
        std::scoped_lock lk{mutex};
        work_cv.notify_all();
    }
    recorder.join();
}

void DrawPipe::Push(Work&& work) {
    Push(std::move(work), JobKind::Barrier);
}

void DrawPipe::Push(Work&& work, JobKind kind, Prepare&& prepare) {
    if (stage_enabled) {
        {
            std::scoped_lock lk{mutex};
            exclusive = false;
            ++pushed;
            pushed_jobs.store(pushed, std::memory_order_release);
            ++stats.pushed;
        }
        std::scoped_lock lk{stage_mutex};
        stage_queue.push_back(StageJob{std::move(work), std::move(prepare), kind});
        stage_cv.notify_one();
        return;
    }
    std::unique_lock lk{mutex};
    exclusive = false;
    ++pushed;
    pushed_jobs.store(pushed, std::memory_order_release);
    ++stats.pushed;
    if (queue.size() >= max_pending) {
        ++stats.push_waits;
        done_cv.wait(lk, [this] { return queue.size() < max_pending; });
    }
    queue.push_back(std::move(work));
    stats.max_queued = std::max(stats.max_queued, queue.size());
    if (recorder_waiting) {
        work_cv.notify_one();
    }
}

void DrawPipe::Forward(Work&& work) {
    std::unique_lock lk{mutex};
    if (queue.size() >= max_pending) {
        ++stats.push_waits;
        done_cv.wait(lk, [this] { return queue.size() < max_pending; });
    }
    queue.push_back(std::move(work));
    stats.max_queued = std::max(stats.max_queued, queue.size());
    if (recorder_waiting) {
        work_cv.notify_one();
    }
}

void DrawPipe::WaitFinished(u64 jobs) {
    for (u32 spins = 0; finished_jobs.load(std::memory_order_acquire) < jobs; ++spins) {
        if (spins < 64) {
            _mm_pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void DrawPipe::RunStage(std::stop_token stop) {
    Common::SetCurrentThreadName("shadGT:GpuBufferStage");
    // The stage keeps at most this many jobs ahead of the recorder: what it prepares is
    // committed to rings that are reused once the GPU passes the tick they were taken in.
    constexpr u64 MaxLead = 256;
    while (true) {
        StageJob job;
        {
            std::unique_lock lk{stage_mutex};
            stage_cv.wait(lk, stop, [this] { return !stage_queue.empty(); });
            if (stage_queue.empty()) {
                return; // stopped
            }
            job = std::move(stage_queue.front());
            stage_queue.pop_front();
        }
        if (forwarded > MaxLead) {
            WaitFinished(forwarded - MaxLead);
        }
        bool sync = job.kind == JobKind::Barrier;
        if (sync) {
            stage_barriers.fetch_add(1, std::memory_order_relaxed);
        }
        if (job.prepare) {
            Common::Nvtx::Scope nvtx{"buffer stage"}; // DIAG-056
            sync |= !job.prepare();
            stage_prepared.fetch_add(1, std::memory_order_relaxed);
        }
        Forward(std::move(job.run));
        ++forwarded;
        if (sync) {
            // The jobs after this one see what it changes only once the recorder ran it.
            stage_syncs.fetch_add(1, std::memory_order_relaxed);
            WaitFinished(forwarded);
        }
    }
}

void DrawPipe::Drain() {
    if (OnRecorderThread()) {
        return;
    }
    std::unique_lock lk{mutex};
    ++stats.drains;
    if (finished == pushed && !urgent_running) {
        exclusive = true;
        return;
    }
    ++stats.drains_that_waited;
    const auto start = std::chrono::steady_clock::now();
    const u64 target = pushed;
    done_cv.wait(lk, [this, target] { return finished >= target && !urgent_running; });
    exclusive = true;
    stats.drain_wait_us += MicrosecondsSince(start);
}

void DrawPipe::PushUrgent(Work&& work) {
    std::scoped_lock lk{mutex};
    urgent.push_back(std::move(work));
    has_urgent.store(true, std::memory_order_release);
    work_cv.notify_all();
}

void DrawPipe::RunUrgent() {
    if (!has_urgent.load(std::memory_order_acquire)) {
        return;
    }
    std::deque<Work> jobs;
    {
        std::scoped_lock lk{mutex};
        if (exclusive || urgent.empty()) {
            return;
        }
        jobs.swap(urgent);
        has_urgent.store(false, std::memory_order_relaxed);
        urgent_running = true;
    }
    for (auto& job : jobs) {
        job();
    }
    std::scoped_lock lk{mutex};
    urgent_running = false;
    done_cv.notify_all();
}

void DrawPipe::RunUrgentIfExclusive() {
    if (!has_urgent.load(std::memory_order_acquire)) {
        return;
    }
    std::deque<Work> jobs;
    {
        std::scoped_lock lk{mutex};
        if (!exclusive || urgent.empty()) {
            return;
        }
        jobs.swap(urgent);
        has_urgent.store(false, std::memory_order_relaxed);
    }
    for (auto& job : jobs) {
        job();
    }
}

void DrawPipe::ReleaseExclusive() {
    std::scoped_lock lk{mutex};
    exclusive = false;
    if (!urgent.empty()) {
        work_cv.notify_all();
    }
}

DrawPipe::Stats DrawPipe::TakeStats() {
    std::scoped_lock lk{mutex};
    Stats taken = stats;
    taken.stage_prepared = stage_prepared.exchange(0);
    taken.stage_syncs = stage_syncs.exchange(0);
    taken.stage_barriers = stage_barriers.exchange(0);
    stats = {};
    return taken;
}

void DrawPipe::Run(std::stop_token stop) {
    if (thread_init) {
        thread_init();
    }
    recorder_id.store(std::this_thread::get_id(), std::memory_order_release);

    std::deque<Work> batch;
    while (true) {
        RunUrgent();
        {
            std::unique_lock lk{mutex};
            if (queue.empty()) {
                recorder_waiting = true;
                work_cv.wait(lk, [&] {
                    return !queue.empty() || (!urgent.empty() && !exclusive) ||
                           stop.stop_requested();
                });
                recorder_waiting = false;
            }
            if (queue.empty()) {
                if (!urgent.empty() && !stop.stop_requested()) {
                    continue; // urgent work only
                }
                return; // Stopped with nothing left to do.
            }
            // Take what is queued now; new work keeps queueing behind it meanwhile.
            batch.swap(queue);
            if (batch.size() >= max_pending) {
                done_cv.notify_all(); // A producer may wait for space.
            }
        }
        const auto start = std::chrono::steady_clock::now();
        u64 ran = 0;
        Common::Nvtx::Scope nvtx{"recording"}; // DIAG-056
        while (!batch.empty()) {
            Work work = std::move(batch.front());
            batch.pop_front();
            work();
            ++ran;
            finished_jobs.fetch_add(1, std::memory_order_release);
            RunUrgent();
        }
        std::scoped_lock lk{mutex};
        finished += ran;
        stats.recorder_busy_us += MicrosecondsSince(start);
        done_cv.notify_all();
    }
}

} // namespace AmdGpu
