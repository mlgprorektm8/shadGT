// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <immintrin.h>
#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "common/nvtx.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "ryty_gpu/ryty_diag.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

// PERF-DIAG-001: where threads block on the GPU (scheduler Wait/Finish), reported every 2 s.
static void RecordGpuWait(const std::source_location& loc, std::chrono::steady_clock::duration d) {
    auto& counters = Common::GetWorkCounters();
    ++counters.gpu_waits;
    counters.gpu_wait_us += u64(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
    RyDiag::RecordElapsed(RyDiag::Zone::GpuWait, d, loc.line());
    static std::mutex mutex;
    static std::map<std::string, std::pair<u32, double>> sites;
    static auto window_start = std::chrono::steady_clock::now();
    std::scoped_lock lk{mutex};
    std::string_view file = loc.file_name();
    if (const auto slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
        file.remove_prefix(slash + 1);
    }
    auto& site = sites[fmt::format("{}:{}", file, loc.line())];
    ++site.first;
    site.second += std::chrono::duration<double, std::milli>(d).count();
    const auto now = std::chrono::steady_clock::now();
    if (now - window_start < std::chrono::seconds{2}) {
        return;
    }
    std::string summary;
    for (const auto& [name, stat] : sites) {
        summary += fmt::format(" {}={}/{:.1f}ms", name, stat.first, stat.second);
    }
    LOG_WARNING(Render_Vulkan, "GPU waits in {:.1f} s (count/total):{}",
                std::chrono::duration<double>(now - window_start).count(), summary);
    sites.clear();
    window_start = now;
}

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    BeginSession();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

void Scheduler::StartPerfMonitor() {
    perf_monitor_thread = std::jthread(std::bind_front(&Scheduler::PerfMonitorThread, this));
}

Scheduler::~Scheduler() {
    if (command_thread.joinable()) {
        command_thread.request_stop();
        ring->Wake();
        command_thread.join();
    }
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::EnableCommandThread() {
    if (ring) {
        return;
    }
    ring = std::make_unique<CommandRing>();
    command_thread = std::jthread(std::bind_front(&Scheduler::CommandThread, this));
    LOG_WARNING(Render_Vulkan, "PERF-084: Vulkan commands are made on a command thread");
}

vk::CommandBuffer ReplayCommandBuffer(Scheduler& scheduler, CmdTarget target) {
    if (target == CmdTarget::Upload) {
        return scheduler.OpenUploadCommandBuffer();
    }
    return scheduler.sessions.back().primary;
}

namespace {
thread_local CommandRing* t_thread_ring = nullptr; // MT-002
}

void Scheduler::SetThreadRing(CommandRing* worker_ring) noexcept {
    t_thread_ring = worker_ring;
}

CommandRing* Scheduler::ThreadRing() noexcept {
    return t_thread_ring;
}

std::unique_ptr<CommandRing> Scheduler::NewWorkerRing() const {
    return std::make_unique<CommandRing>();
}

void Scheduler::RecordChunk(CommandRing* worker_ring, u64 start, u64 end) {
    if (start == end) {
        return;
    }
    ASSERT_MSG(ring, "MT-002: chunks need the command thread");
    ring->RecordControl([worker_ring, start, end](Scheduler& scheduler) {
        scheduler.ReplayChunk(*worker_ring, start, end);
    });
}

void Scheduler::ReplayChunk(CommandRing& worker_ring, u64 start, u64 end) {
    while (worker_ring.Published() < end) {
        _mm_pause();
    }
    u64 position = start;
    while (position < end) {
        auto* header = worker_ring.At(position);
        if (header->exec != nullptr) {
            header->exec(*this, reinterpret_cast<u8*>(header) + header->fn_offset);
        }
        position += header->size;
    }
    worker_ring.SetReplayed(end);
}

void CommandRing::WaitForSpace(size_t needed) {
    ++full_waits;
    while (head_local + needed - tail.load(std::memory_order_acquire) > Capacity) {
        std::this_thread::yield();
    }
}

void CommandRing::WaitForWork(u64 seen, const std::stop_token& stoken) {
    // The recorder hands over commands steadily while it works; spin briefly before sleeping.
    // PERF-125: SHADGT_CMD_SPIN pauses (default 2048). The spin was most of this thread's
    // CPU time (replay itself ~23% busy).
    static const u32 spin = [] {
        const char* env = std::getenv("SHADGT_CMD_SPIN");
        return env && *env ? u32(std::strtoul(env, nullptr, 10)) : 2048u;
    }();
    for (u32 i = 0; i < spin; ++i) {
        if (head.load(std::memory_order_acquire) != seen) {
            return;
        }
        _mm_pause();
    }
    const u32 wake_value = wake.load(std::memory_order_acquire);
    sleeping.store(true, std::memory_order_seq_cst);
    if (head.load(std::memory_order_seq_cst) != seen || stoken.stop_requested()) {
        sleeping.store(false, std::memory_order_relaxed);
        return;
    }
    wake.wait(wake_value, std::memory_order_acquire);
}

void Scheduler::SyncCommandThread() {
    if (!ring) {
        return;
    }
    const u64 target = ring->Recorded();
    if (ring->Replayed() >= target) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    u32 spins = 0;
    while (ring->Replayed() < target) {
        if (++spins < 512) {
            _mm_pause();
        } else {
            std::this_thread::yield();
        }
    }
    sync_waits.fetch_add(1, std::memory_order_relaxed);
    sync_wait_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - start)
                                   .count()),
                           std::memory_order_relaxed);
}

void Scheduler::WaitSubmitted(u64 tick) {
    if (!ring) {
        return;
    }
    u64 submitted = submitted_tick.load(std::memory_order_acquire);
    while (submitted < tick) {
        submitted_tick.wait(submitted, std::memory_order_acquire);
        submitted = submitted_tick.load(std::memory_order_acquire);
    }
}

vk::CommandBuffer Scheduler::DirectCommandBuffer() {
    SyncCommandThread();
    return sessions.back().primary;
}

void Scheduler::CommandThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadGT:VkCommands");
    Common::PreferPerformanceCores(); // PERF-088
    auto& commands = *ring;
    u64 position = commands.Replayed();
    u64 window_commands = 0;
    u64 window_busy_ns = 0;
    auto window_start = std::chrono::steady_clock::now();
    while (!stoken.stop_requested()) {
        const u64 end = commands.Published();
        if (position == end) {
            commands.SetReplayed(position);
            const auto now = std::chrono::steady_clock::now();
            if (now - window_start >= std::chrono::seconds{2}) {
                const double seconds = std::chrono::duration<double>(now - window_start).count();
                LOG_WARNING(Render_Vulkan,
                            "PERF-084 command thread in {:.1f} s: {} commands, busy {:.0f}%; "
                            "recorder waited {} times for ring space, {} times for the thread "
                            "({:.1f} ms)",
                            seconds, window_commands, window_busy_ns / (seconds * 1e7),
                            commands.full_waits, sync_waits.exchange(0),
                            sync_wait_ns.exchange(0) / 1e6);
                commands.full_waits = 0;
                window_commands = 0;
                window_busy_ns = 0;
                window_start = now;
            }
            commands.WaitForWork(position, stoken);
            continue;
        }
        const auto start = std::chrono::steady_clock::now();
        u32 since_progress = 0;
        while (position != end) {
            auto* header = commands.At(position);
            if (header->exec != nullptr) {
                header->exec(*this, reinterpret_cast<u8*>(header) + header->fn_offset);
                ++window_commands;
            }
            position += header->size;
            if (++since_progress == 64) {
                commands.SetReplayed(position);
                since_progress = 0;
            }
        }
        commands.SetReplayed(position);
        window_busy_ns += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - start)
                                  .count());
    }
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    CommandBuffer().beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    CommandBuffer().endRendering();
}

CmdBuf Scheduler::UploadCommandBuffer() {
    if (ring) {
        if (CommandRing* worker = ThreadRing()) {
            return CmdBuf{worker, CmdTarget::Upload}; // MT-002
        }
        return CmdBuf{ring.get(), CmdTarget::Upload};
    }
    return CmdBuf{OpenUploadCommandBuffer()};
}

vk::CommandBuffer Scheduler::OpenUploadCommandBuffer() {
    auto& upload_cmdbuf = sessions.back().upload;
    if (upload_cmdbuf) {
        return upload_cmdbuf;
    }
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    upload_cmdbuf = command_pool.Commit();
    Check(upload_cmdbuf.begin(begin_info));
    // PERF-066: the uploads come after all work recorded before this session.
    const vk::MemoryBarrier2 before{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eTransferRead,
    };
    upload_cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &before,
    });
    return upload_cmdbuf;
}

// DIAG-118: where submits come from, every 2 s.
static void NoteFlush(const std::source_location& loc) {
    static std::mutex diag_mutex;
    static std::map<std::pair<std::string_view, u32>, u64> sites;
    static auto since = std::chrono::steady_clock::now();
    std::scoped_lock lk{diag_mutex};
    std::string_view file = loc.file_name();
    if (const auto slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
        file.remove_prefix(slash + 1);
    }
    ++sites[{file, loc.line()}];
    const auto now = std::chrono::steady_clock::now();
    if (now - since >= std::chrono::seconds{2}) {
        std::string text;
        for (const auto& [site, count] : sites) {
            text += fmt::format(" {}:{}={}", site.first, site.second, count);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-118: submits in 2 s by caller:{}", text);
        sites.clear();
        since = now;
    }
}

void Scheduler::Flush(SubmitInfo& info, std::source_location loc) {
    NoteFlush(loc);
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush(std::source_location loc) {
    SubmitInfo info{};
    Flush(info, loc);
}

void Scheduler::Finish(std::source_location loc) {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick, loc);
}

void Scheduler::Wait(u64 tick, std::source_location loc) {
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    if (work_semaphore.IsFree(tick)) {
        return;
    }
    Common::Nvtx::Scope nvtx{"wait for GPU"}; // DIAG-056
    const auto start = std::chrono::steady_clock::now();
    work_semaphore.Wait(tick);
    RecordGpuWait(loc, std::chrono::steady_clock::now() - start);
}

void Scheduler::PopPendingOperations() {
    std::unique_lock lk(pending_ops_mutex);
    // PERF-045: called before every draw; reading the timeline semaphore is a driver call, so
    // only when an operation waits for the GPU. Submits and Scheduler::IsFree refresh it too.
    // -DisablePerf 57 reads it every time.
    static const bool skip_idle_refresh = Common::PerfFeatureEnabled(57);
    if (skip_idle_refresh && pending_ops.empty()) {
        return;
    }
    work_semaphore.Refresh();
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::BeginSession() {
    EndSession();

    session_open = true;
    ++session_id;
    if (ring) {
        ring->RecordControl([](Scheduler& scheduler) { scheduler.OpenSession(); });
    } else {
        OpenSession();
    }

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::OpenSession() {
    auto& session = sessions.emplace_back();
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    session.primary = command_pool.Commit();
    Check(session.primary.begin(begin_info));
}

void Scheduler::EndSession() {
    if (!session_open) {
        return;
    }

    if (on_session) {
        on_session();
    }

    EndRendering();
    session_open = false;
    if (ring) {
        ring->RecordControl([](Scheduler& scheduler) { scheduler.CloseSession(); });
    } else {
        CloseSession();
    }
}

void Scheduler::CloseSession() {
    const auto& session = sessions.back();
    if (session.upload) {
        // PERF-066: the session's work sees the uploads.
        const vk::MemoryBarrier2 after{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        session.upload.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &after,
        });
        Check(session.upload.end());
    }
    Check(session.primary.end());
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    Common::PhaseTimer submit_timer{Common::Phase::Submit};
    RYDIAG_SCOPE(Submit);
    // PERF-084: the queue lock is taken only around queue calls (the submit callback's sparse
    // binds take it themselves), so the command thread can submit while the recorder works.
    const u64 signal_value = work_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }

    EndSession();

    info.AddSignal(work_semaphore.Handle(), signal_value);
    if (ring) {
        ring->RecordControl([info, signal_value](Scheduler& scheduler) mutable {
            scheduler.SubmitSessions(info, signal_value);
        });
    } else {
        SubmitSessions(info, signal_value);
    }

    BeginSession();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::SubmitSessions(SubmitInfo& info, u64 signal_value) {
    boost::container::small_vector<vk::CommandBuffer, 4> cmd_buffers;
    for (const auto& session : sessions) {
        if (session.upload) {
            cmd_buffers.push_back(session.upload);
        }
        cmd_buffers.push_back(session.primary);
    }
    sessions.clear();

    // A ready frame is sampled by the fragment shader, before color attachment output.
    // Include every supported wait semaphore and protect uploads and other shader accesses too.
    std::array<vk::PipelineStageFlags, std::tuple_size_v<decltype(info.wait_semas)>>
        wait_stage_masks;
    wait_stage_masks.fill(vk::PipelineStageFlagBits::eAllCommands);

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = static_cast<u32>(cmd_buffers.size()),
        .pCommandBuffers = cmd_buffers.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    {
        std::scoped_lock lk{submit_mutex};
        ImGui::Core::TextureManager::Submit();
        auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
        ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    }

    work_semaphore.Refresh();
    if (ring) {
        submitted_tick.store(signal_value, std::memory_order_release);
        submitted_tick.notify_all();
    }
}

void Scheduler::PerfMonitorThread(std::stop_token stoken) {
    // PERF-DIAG-007: every 2 s, how busy the GPU was (sampled: submitted work not yet complete)
    // and which threads used the CPU, to tell GPU-bound, CPU-bound and serialized frames apart.
    Common::SetCurrentThreadName("shadGT:PerfMonitor");
    Common::SampleThreadCpuUsage(0);
    auto window_start = std::chrono::steady_clock::now();
    u64 samples = 0;
    u64 busy_samples = 0;
    u64 window_tick = CurrentTick();
    while (!stoken.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        work_semaphore.Refresh();
        const u64 submitted = work_semaphore.CurrentTick() - 1;
        ++samples;
        busy_samples += work_semaphore.KnownGpuTick() < submitted ? 1 : 0;
        const auto now = std::chrono::steady_clock::now();
        if (now - window_start < std::chrono::seconds{2}) {
            continue;
        }
        const u64 tick = work_semaphore.CurrentTick();
        LOG_WARNING(Render_Vulkan, "Perf monitor {:.1f} s: GPU busy {:.0f}%, {} submits; CPU {}",
                    std::chrono::duration<double>(now - window_start).count(),
                    samples ? busy_samples * 100.0 / samples : 0.0, tick - window_tick,
                    Common::SampleThreadCpuUsage(10));
        LOG_WARNING(Render_Vulkan, "Frontend work in {:.1f} s: {}",
                    std::chrono::duration<double>(now - window_start).count(),
                    Common::TakeWorkCounters());
        // DIAG-103: the GPU busy with nothing new submitted for 10 s: the GPU (or a wait on it) is
        // stuck; once per run, every thread's stack goes to user\log\gpu-stuck-stacks.txt.
        static u32 stuck_windows = 0;
        static bool dumped = false;
        stuck_windows = (samples && busy_samples == samples && tick == window_tick)
                            ? stuck_windows + 1
                            : 0;
        if (stuck_windows >= 5 && !dumped) {
            dumped = true;
            LOG_WARNING(Render_Vulkan, "DIAG-103: GPU work unfinished for 10 s with nothing new "
                        "submitted (GPU tick {} of {})", work_semaphore.KnownGpuTick(),
                        tick - 1);
            DumpThreadStacks(L"gpu-stuck-stacks.txt");
        }
        window_start = now;
        window_tick = tick;
        samples = 0;
        busy_samples = 0;
    }
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadGT:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        work_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const CmdBuf& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled);
    }
}

} // namespace Vulkan
