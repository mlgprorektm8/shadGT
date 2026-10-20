// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>

#include <immintrin.h>
#include <string>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/nvtx.h"
#include "common/perf_monitor.h"
#include "common/thread.h"
#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/amdgpu/draw_pipe.h"

#ifdef RYTY_GPU
namespace Vulkan {
bool NativeModeRequested(); // NAT-100 (vk_rasterizer_native.cpp)
}
#endif

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
    // D9: on by default with the direct renderer (its recorder obtains far fewer buffers, so the
    // stage's plans are rarely refused: Tokyo rain 45.2 fps vs 43.4); =0 turns it off.
    const char* stage_env = std::getenv("SHADGT_BUFFER_STAGE");
    bool stage_wanted = stage_env && *stage_env ? stage_env[0] == '1'
                                                : VideoCore::DirectMemory::Enabled();
#ifdef RYTY_GPU
    stage_wanted = stage_wanted && !Vulkan::NativeModeRequested(); // NAT-100
#endif
    if (stage_wanted && Common::PerfFeatureEnabled(74)) {
        stage_enabled = true;
        if (const char* workers_env = std::getenv("SHADGT_STAGE_WORKERS")) {
            worker_count = static_cast<u32>(std::clamp(std::atoi(workers_env), 1, 4));
        }
        if (worker_count > 1) {
            for (u32 i = 0; i < worker_count; ++i) {
                workers.emplace_back([this, i](std::stop_token stop) { RunWorker(stop, i); });
            }
            stage = std::jthread{[this](std::stop_token stop) { RunDispatcher(stop); }};
        } else {
            stage = std::jthread{[this](std::stop_token stop) { RunStage(stop); }};
        }
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
        for (auto& worker : workers) {
            worker.request_stop();
        }
        {
            std::scoped_lock lk{work_mutex};
            work_items_cv.notify_all();
        }
        workers.clear();
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

void DrawPipe::Push(Work&& work, JobKind kind, Prepare&& prepare, Work&& order) {
    if (stage_enabled) {
        {
            std::scoped_lock lk{mutex};
            exclusive = false;
            ++pushed;
            pushed_jobs.store(pushed, std::memory_order_release);
            ++stats.pushed;
        }
        std::scoped_lock lk{stage_mutex};
        if (kind == JobKind::Barrier) {
            // DIAG-069: which jobs make the stage wait for the recorder, every 2 s.
            ++barrier_tags[push_tag ? push_tag : "untagged"];
            const auto now = std::chrono::steady_clock::now();
            if (barrier_tags_since == std::chrono::steady_clock::time_point{}) {
                barrier_tags_since = now;
            } else if (now - barrier_tags_since >= std::chrono::seconds{2}) {
                std::string line;
                for (const auto& [tag, count] : barrier_tags) {
                    line += fmt::format(" {}={}", tag, count);
                }
                LOG_WARNING(Render, "DIAG-069: buffer stage barriers in 2 s:{}", line);
                barrier_tags.clear();
                barrier_tags_since = now;
            }
        }
        push_tag = nullptr;
        stage_queue.push_back(
            StageJob{std::move(work), std::move(prepare), kind, std::move(order)});
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
    // A short spin (the recorder is often about to finish), then the stage sleeps until the
    // recorder has finished the job (it wakes the stage when one waits).
    for (u32 spins = 0; spins < 256; ++spins) {
        if (finished_jobs.load(std::memory_order_acquire) >= jobs) {
            return;
        }
        _mm_pause();
    }
    std::unique_lock lk{stage_wait_mutex};
    stage_wait_target.store(jobs); // seq_cst: against the recorder's count and load
    stage_wait_cv.wait(lk, [&] { return finished_jobs.load() >= jobs; });
    stage_wait_target.store(0, std::memory_order_release);
}

void DrawPipe::RunStage(std::stop_token stop) {
    Common::SetCurrentThreadName("shadGT:GpuBufferStage");
    Common::PreferPerformanceCores(); // PERF-088
    // The stage keeps at most this many jobs ahead of the recorder: what it prepares is
    // committed to rings that are reused once the GPU passes the tick they were taken in.
    constexpr u64 MaxLead = 256;
    while (true) {
        StageJob job;
        {
            const auto idle_start = std::chrono::steady_clock::now();
            std::unique_lock lk{stage_mutex};
            stage_cv.wait(lk, stop, [this] { return !stage_queue.empty(); });
            stage_idle_us.fetch_add(MicrosecondsSince(idle_start), std::memory_order_relaxed);
            if (stage_queue.empty()) {
                return; // stopped
            }
            job = std::move(stage_queue.front());
            stage_queue.pop_front();
        }
        if (forwarded > MaxLead) {
            const auto lead_start = std::chrono::steady_clock::now();
            WaitFinished(forwarded - MaxLead);
            stage_lead_us.fetch_add(MicrosecondsSince(lead_start), std::memory_order_relaxed);
        }
        bool sync = job.kind == JobKind::Barrier;
        if (sync) {
            stage_barriers.fetch_add(1, std::memory_order_relaxed);
        }
        if (job.prepare) {
            Common::Nvtx::Scope nvtx{"buffer stage"}; // DIAG-056
            const auto prepare_start = std::chrono::steady_clock::now();
            if (job.order) {
                job.order();
            }
            sync |= !job.prepare();
            stage_prepare_us.fetch_add(MicrosecondsSince(prepare_start), std::memory_order_relaxed);
            stage_prepared.fetch_add(1, std::memory_order_relaxed);
        }
        Forward(std::move(job.run));
        ++forwarded;
        if (sync) {
            // The jobs after this one see what it changes only once the recorder ran it.
            stage_syncs.fetch_add(1, std::memory_order_relaxed);
            const auto sync_start = std::chrono::steady_clock::now();
            WaitFinished(forwarded);
            stage_sync_us.fetch_add(MicrosecondsSince(sync_start), std::memory_order_relaxed);
        }
    }
}

std::vector<DrawPipe::Idle> DrawPipe::RecentIdle() {
    std::scoped_lock lk{idle_mutex};
    std::vector<Idle> out;
    const u32 count = std::min<u32>(idle_next, idle_ring.size());
    for (u32 i = idle_next - count; i != idle_next; ++i) {
        out.push_back(idle_ring[i % idle_ring.size()]);
    }
    return out;
}

void DrawPipe::RunWorker(std::stop_token stop, u32 index) {
    Common::SetCurrentThreadName(fmt::format("shadGT:GpuStageWorker{}", index).c_str());
    Common::PreferPerformanceCores();
    bool initialized = false;
    while (true) {
        InFlight* item;
        {
            std::unique_lock lk{work_mutex};
            work_items_cv.wait(lk, stop, [this] { return !work_items.empty(); });
            if (work_items.empty()) {
                return; // stopped
            }
            item = work_items.front();
            work_items.pop_front();
        }
        if (!initialized && worker_init) {
            worker_init(index); // set once the rasterizer exists
            initialized = true;
        }
        const auto prepare_start = std::chrono::steady_clock::now();
        item->ok = item->job.prepare();
        stage_prepare_us.fetch_add(MicrosecondsSince(prepare_start), std::memory_order_relaxed);
        stage_prepared.fetch_add(1, std::memory_order_relaxed);
        {
            std::scoped_lock lk{stage_mutex};
            item->done.store(true, std::memory_order_release);
            stage_cv.notify_all();
        }
    }
}

void DrawPipe::ForwardReady() {
    // Dispatcher: the prepared jobs at the front go to the recorder, in order.
    while (!window.empty() && window.front()->done.load(std::memory_order_acquire)) {
        auto item = std::move(window.front());
        window.pop_front();
        Forward(std::move(item->job.run));
        ++forwarded;
        if (!item->ok) {
            // The jobs after a refused plan see what the recorder does for it only once it ran
            // it: their plans are made again then.
            stage_syncs.fetch_add(1, std::memory_order_relaxed);
            const auto sync_start = std::chrono::steady_clock::now();
            WaitFinished(forwarded);
            stage_sync_us.fetch_add(MicrosecondsSince(sync_start), std::memory_order_relaxed);
            RedoWindow();
        }
    }
}

void DrawPipe::RedoWindow() {
    if (worker_init) {
        worker_init(0); // the dispatcher redoes plans with worker 0's rings (idle meanwhile)
    }
    for (auto& item : window) {
        if (!item->done.load(std::memory_order_acquire)) {
            std::unique_lock lk{stage_mutex};
            stage_cv.wait(lk, [&] { return item->done.load(std::memory_order_acquire); });
        }
    }
    for (auto& item : window) {
        const auto prepare_start = std::chrono::steady_clock::now();
        item->ok = item->job.prepare();
        stage_prepare_us.fetch_add(MicrosecondsSince(prepare_start), std::memory_order_relaxed);
        if (!item->ok) {
            break; // the jobs after it are made again once this one is recorded
        }
    }
}

void DrawPipe::RunDispatcher(std::stop_token stop) {
    Common::SetCurrentThreadName("shadGT:GpuBufferStage");
    Common::PreferPerformanceCores();
    constexpr u64 MaxLead = 256;
    const size_t max_window = 16 * worker_count;
    while (true) {
        ForwardReady();
        StageJob job;
        {
            const auto idle_start = std::chrono::steady_clock::now();
            std::unique_lock lk{stage_mutex};
            stage_cv.wait(lk, stop, [&] {
                return (!stage_queue.empty() && window.size() < max_window) ||
                       (!window.empty() && window.front()->done.load(std::memory_order_acquire));
            });
            stage_idle_us.fetch_add(MicrosecondsSince(idle_start), std::memory_order_relaxed);
            if (stop.stop_requested() && stage_queue.empty() && window.empty()) {
                return;
            }
            if (stage_queue.empty() || window.size() >= max_window) {
                continue; // a prepared job to forward
            }
            job = std::move(stage_queue.front());
            stage_queue.pop_front();
        }
        if (job.kind == JobKind::Barrier || !job.prepare) {
            // Everything before it goes to the recorder first; a barrier is then waited for.
            while (!window.empty()) {
                if (!window.front()->done.load(std::memory_order_acquire)) {
                    std::unique_lock lk{stage_mutex};
                    stage_cv.wait(lk, [&] {
                        return window.front()->done.load(std::memory_order_acquire);
                    });
                }
                ForwardReady();
            }
            const bool sync = job.kind == JobKind::Barrier;
            if (sync) {
                stage_barriers.fetch_add(1, std::memory_order_relaxed);
            }
            Forward(std::move(job.run));
            ++forwarded;
            if (sync) {
                stage_syncs.fetch_add(1, std::memory_order_relaxed);
                const auto sync_start = std::chrono::steady_clock::now();
                WaitFinished(forwarded);
                stage_sync_us.fetch_add(MicrosecondsSince(sync_start), std::memory_order_relaxed);
            }
            continue;
        }
        if (forwarded > MaxLead) {
            const auto lead_start = std::chrono::steady_clock::now();
            WaitFinished(forwarded - MaxLead);
            stage_lead_us.fetch_add(MicrosecondsSince(lead_start), std::memory_order_relaxed);
        }
        if (job.order) {
            job.order();
        }
        auto item = std::make_unique<InFlight>();
        item->job = std::move(job);
        InFlight* raw = item.get();
        window.push_back(std::move(item));
        {
            std::scoped_lock lk{work_mutex};
            work_items.push_back(raw);
        }
        work_items_cv.notify_one();
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
    taken.stage_idle_us = stage_idle_us.exchange(0);
    taken.stage_prepare_us = stage_prepare_us.exchange(0);
    taken.stage_lead_us = stage_lead_us.exchange(0);
    taken.stage_sync_us = stage_sync_us.exchange(0);
    stats = {};
    return taken;
}

void DrawPipe::Run(std::stop_token stop) {
    if (thread_init) {
        thread_init();
    }
    Common::PreferPerformanceCores(); // PERF-088
    recorder_id.store(std::this_thread::get_id(), std::memory_order_release);
    // PERF-088: on which kind of core the recorder ran its jobs, every 10 s.
    std::array<u64, 2> core_jobs{};
    auto core_since = std::chrono::steady_clock::now();

    std::deque<Work> batch;
    while (true) {
        RunUrgent();
        if (idle_hook) {
            bool empty;
            {
                std::scoped_lock lk{mutex};
                empty = queue.empty();
            }
            if (empty) {
                idle_hook(); // D5
            }
        }
        {
            std::unique_lock lk{mutex};
            if (queue.empty()) {
                recorder_waiting = true;
                const auto idle_from = std::chrono::steady_clock::now();
                work_cv.wait(lk, [&] {
                    return !queue.empty() || (!urgent.empty() && !exclusive) ||
                           stop.stop_requested();
                });
                recorder_waiting = false;
                const auto idle_to = std::chrono::steady_clock::now();
                if (idle_to - idle_from > std::chrono::microseconds{50}) {
                    std::scoped_lock idle_lk{idle_mutex};
                    idle_ring[idle_next++ % idle_ring.size()] = {idle_from, idle_to};
                }
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
            if (const int fast = Common::CurrentCoreIsFast(); fast >= 0) {
                ++core_jobs[fast];
            }
            const u64 done = finished_jobs.fetch_add(1) + 1;
            if (const u64 target = stage_wait_target.load();
                target != 0 && done >= target) {
                std::scoped_lock lk{stage_wait_mutex};
                stage_wait_cv.notify_all();
            }
            RunUrgent();
        }
        if (const auto now = std::chrono::steady_clock::now();
            now - core_since >= std::chrono::seconds{10} && core_jobs[0] + core_jobs[1] != 0) {
            LOG_WARNING(Render, "PERF-088: recorder jobs in 10 s: {} on fast cores, {} on slow cores",
                        core_jobs[1], core_jobs[0]);
            core_jobs = {};
            core_since = now;
        }
        std::scoped_lock lk{mutex};
        finished += ran;
        stats.recorder_busy_us += MicrosecondsSince(start);
        done_cv.notify_all();
    }
}

} // namespace AmdGpu
