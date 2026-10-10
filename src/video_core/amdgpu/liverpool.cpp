// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/preprocessor/stringize.hpp>
#include <fmt/ranges.h>

#include "common/assert.h"
#include "common/sampling_profiler.h"
#include "common/scope_exit.h"
#include "common/debug.h"
#include "common/guest_clock.h"
#include "common/nvtx.h"
#include "common/perf_monitor.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/ce_de_counter.h"
#include "video_core/amdgpu/draw_pipe.h"
#include <xxhash.h>

#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/pm4_rewind.h"
#include "video_core/amdgpu/pm4_type0.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu {

// A fence signaled after the GPU completes may target memory the guest unmapped meanwhile.
// PERF-DIAG-012: when deferred fences last wrote each address, to tell waits on GPU completion
// from waits on the CPU or another queue.
static std::mutex g_deferred_fence_writes_mutex;
static std::unordered_map<uintptr_t, std::chrono::steady_clock::time_point> g_deferred_fence_writes;

// DIAG-055: graphics end-of-pipe interrupts and the game's next submission.
static std::atomic<s64> g_last_eop_ns{0};
static std::atomic<u64> g_eops_deferred{0};
static std::atomic<u64> g_eops_immediate{0};

// EXP-063 (measurement only, not safe for play): graphics EOP/EOS fences are signaled when the
// command thread decodes them instead of after the recorder has recorded the work before them.
// It measures how fast races get when the game no longer waits for the recorder. The game may
// then rewrite memory the recorder has not read yet, and reads GPU results before they exist,
// so rendering can be wrong. On while the file named by SHADGT_EARLY_FENCES_FLAG exists (checked
// every 250 ms), so a benchmark can turn it on after the menus. Command thread only.
static std::atomic<u64> g_fences_early{0};
static bool EarlyFencesActive() {
    static const char* const flag = std::getenv("SHADGT_EARLY_FENCES_FLAG");
    if (!flag || !*flag) {
        return false;
    }
    static std::chrono::steady_clock::time_point next_check{};
    static bool active = false;
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_check) {
        next_check = now + std::chrono::milliseconds(250);
        std::error_code ec;
        const bool on = std::filesystem::exists(flag, ec);
        if (on != active) {
            LOG_WARNING(Render, "EXP-063: early fences {} (measurement only)", on ? "on" : "off");
        }
        active = on;
    }
    return active;
}
static s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
static void NoteEop() {
    g_last_eop_ns.store(NowNs());
    Common::Nvtx::Mark("EOP interrupt"); // DIAG-056
}

static void WriteDeferredFence(void* address, const void* data, u32 num_bytes) {
    {
        std::scoped_lock lk{g_deferred_fence_writes_mutex};
        if (g_deferred_fence_writes.size() > 4096) {
            g_deferred_fence_writes.clear();
        }
        g_deferred_fence_writes[reinterpret_cast<uintptr_t>(address)] =
            std::chrono::steady_clock::now();
    }
    auto* memory = Core::Memory::Instance();
    if (memory->IsValidMapping(reinterpret_cast<VAddr>(address), num_bytes)) {
        memory->TryWriteBacking(address, data, num_bytes);
    }
}

// PERF-DIAG-009: how long each queue's command processing is blocked in wait packets, and on
// which addresses, reported every 2 s. Only the command processor thread updates it.
enum class FrontendWait : u32 {
    GfxWaitRegMem,
    GfxVoLabel,
    GfxMemSemaphore,
    GfxRewind,
    AscWaitRegMem,
    AscMemSemaphore,
    AscRewind,
    Count
};
static void RecordFrontendWait(FrontendWait kind, uintptr_t address,
                               std::chrono::steady_clock::time_point start) {
    // PERF-012: whoever the queue waited for may have written memory later draws read.
    VideoCore::BumpUploadEpoch();
    static constexpr std::array<const char*, size_t(FrontendWait::Count)> Names = {
        "gfx WAIT_REG_MEM", "gfx VO label",      "gfx MEM_SEMAPHORE", "gfx REWIND",
        "asc WAIT_REG_MEM", "asc MEM_SEMAPHORE", "asc REWIND"};
    static std::array<std::pair<u32, double>, size_t(FrontendWait::Count)> totals{};
    static std::pair<u32, double> gfx_on_fence{};
    static std::unordered_map<uintptr_t, std::pair<FrontendWait, double>> addresses;
    static auto window_start = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - start).count();
    ++Common::GetWorkCounters().frontend_waits;
    Common::GetWorkCounters().frontend_wait_us += u64(ms * 1000.0);
    auto& total = totals[size_t(kind)];
    ++total.first;
    total.second += ms;
    if (kind == FrontendWait::GfxWaitRegMem) {
        std::scoped_lock lk{g_deferred_fence_writes_mutex};
        const auto it = g_deferred_fence_writes.find(address);
        if (it != g_deferred_fence_writes.end() && it->second >= start) {
            ++gfx_on_fence.first;
            gfx_on_fence.second += ms;
        }
    }
    auto& entry = addresses[address];
    entry.first = kind;
    entry.second += ms;
    if (now - window_start < std::chrono::seconds{2}) {
        return;
    }
    std::string summary;
    for (size_t i = 0; i < Names.size(); ++i) {
        if (totals[i].first != 0) {
            summary += fmt::format(" {}={}/{:.1f}ms", Names[i], totals[i].first, totals[i].second);
        }
    }
    summary += fmt::format(" (gfx WAIT_REG_MEM ended by a deferred fence: {}/{:.1f}ms)",
                           gfx_on_fence.first, gfx_on_fence.second);
    gfx_on_fence = {};
    std::vector<std::pair<uintptr_t, std::pair<FrontendWait, double>>> top(addresses.begin(),
                                                                           addresses.end());
    std::ranges::sort(top, std::greater{}, [](const auto& e) { return e.second.second; });
    summary += ";";
    for (size_t i = 0; i < std::min<size_t>(5, top.size()); ++i) {
        summary += fmt::format(" {} {:#x} {:.1f}ms", Names[size_t(top[i].second.first)],
                               top[i].first, top[i].second.second);
    }
    LOG_WARNING(Render, "Command processor waits in {:.1f} s (count/total):{}",
                std::chrono::duration<double>(now - window_start).count(), summary);
    totals = {};
    addresses.clear();
    window_start = now;
}

// PERF-014: the bytes a fence packet writes, when they are known when it is processed.
static std::vector<u8> FenceValueBytes(DataSelect data_sel, u32 low, u64 qword) {
    std::vector<u8> bytes;
    if (data_sel == DataSelect::Data32Low) {
        bytes.resize(sizeof(u32));
        std::memcpy(bytes.data(), &low, sizeof(u32));
    } else if (data_sel == DataSelect::Data64) {
        bytes.resize(sizeof(u64));
        std::memcpy(bytes.data(), &qword, sizeof(u64));
    }
    return bytes;
}

// PERF-014: whether a WAIT_REG_MEM on memory is satisfied by the value a fence the command
// thread already processed will write once the GPU reaches it.
static bool SatisfiedByPendingFence(Vulkan::Rasterizer* rasterizer,
                                    const PM4CmdWaitRegMem* wait_reg_mem) {
    static const bool enabled = Common::PerfFeatureEnabled(14);
    // PERF-014b (id 1402, off by default): also wait while a deferred fence still brings GPU
    // data back to guest memory. In GT Sport one always does, so it never went early. The
    // exploded vertices first blamed on PERF-014 came from PERF-015.
    static const bool wait_for_readbacks = !Common::PerfFeatureEnabled(1402);
    if (!enabled || !rasterizer || wait_reg_mem->mem_space != PM4CmdWaitRegMem::MemSpace::Memory ||
        (wait_for_readbacks && rasterizer->HasReadbackFences())) {
        return false;
    }
    const auto value =
        rasterizer->PendingFenceDword(reinterpret_cast<VAddr>(wait_reg_mem->Address<u32*>()));
    if (!value || !wait_reg_mem->TestValue(*value)) {
        return false;
    }
    ++Common::GetWorkCounters().waits_skipped;
    return true;
}

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

class GpuWaitDiagnostics {
public:
    bool Ready() {
        const auto now = std::chrono::steady_clock::now();
        if (now < next_report) {
            return false;
        }
        next_report = now + std::chrono::seconds(5);
        return true;
    }

private:
    std::chrono::steady_clock::time_point next_report =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

// FIX-035: DMA_DATA with a register source or destination address space (SAS/DAS = 1). GT
// Sport issues it on the graphics queue in the main menu; its exact effect is undocumented in
// the sources available, so the packet is reported once per destination and skipped rather
// than stopping the emulator.
static bool SkipRegisterSpaceDmaData(const PM4DmaData* dma_data, const char* queue) {
    if (dma_data->command.das == 0 && dma_data->command.sas == 0) {
        return false;
    }
    static std::mutex mutex;
    static std::unordered_set<u64> reported;
    std::scoped_lock lk{mutex};
    if (reported
            .insert(u64(dma_data->dst_addr_lo) << 8 | u64(dma_data->src_sel) << 4 |
                    u64(dma_data->dst_sel))
            .second) {
        LOG_ERROR(Render_Vulkan,
                  "FIX-035: {} DMA_DATA with register address space skipped: sas {} das {} "
                  "src_sel {} dst_sel {} src {:#x} dst {:#x} bytes {} saic {} daic {}",
                  queue, u32(dma_data->command.sas), u32(dma_data->command.das),
                  u32(dma_data->src_sel), u32(dma_data->dst_sel), dma_data->SrcAddress<u64>(),
                  dma_data->DstAddress<u64>(), dma_data->NumBytes(), u32(dma_data->command.saic),
                  u32(dma_data->command.daic));
    }
    return true;
}

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(Lib_GnmDriver,
                  "Packet exceeds submission at {:#x}: packet dwords={}, remaining dwords={}, "
                  "remaining words={:#010x}",
                  reinterpret_cast<uintptr_t>(span.data()), offset, span.size(),
                  fmt::join(span.first(std::min<size_t>(span.size(), 16)), " "));
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
    draw_pipe.reset();
}

// PERF-031: the draw pipe is opt-in (SHADGT_DRAW_PIPE=1) while it is measured;
// -DisablePerf 46 keeps it off regardless.
static bool DrawPipeRequested() {
    static const bool requested = [] {
        const char* env = std::getenv("SHADGT_DRAW_PIPE");
        return env && env[0] == '1' && Common::PerfFeatureEnabled(46);
    }();
    return requested;
}

void Liverpool::StartDrawPipe() {
    recorder = std::make_unique<RecorderState>();
    recorder->regs = regs;
    // Dispatches carry their queue's compute registers; there is no current queue here.
    recorder->cb_extent = last_cb_extent;
    recorder->db_extent = last_db_extent;
    regs_dirty = {};
    // PERF-034: the pipeline read-ahead goes on with the draw pipe; -DisablePerf 49 (or 19)
    // turns it off.
    if (Common::PerfFeatureEnabled(49) && Common::PerfFeatureEnabled(19)) {
        read_ahead_states = std::make_unique<ReadAheadStates>();
        pipeline_regs_written = true;
    }
    if (const char* env = std::getenv("SHADGT_DRAW_PIPE_VERIFY")) {
        verify_interval = static_cast<u32>(std::strtoul(env, nullptr, 10));
    }
    draw_pipe = std::make_unique<DrawPipe>([this] {
        Common::SetCurrentThreadName("shadGT:GpuRecorder");
        Common::SamplingProfiler::RegisterCurrentThread("GpuRecorder"); // DIAG-048
        // FIX-044: shader and pipeline compiles on this thread hold the guest clocks too.
        Common::GuestClock::MarkGpuCommandThread();
        recorder_state = recorder.get();
#ifdef __linux__
        recorder_tid = gettid();
#endif
    });
    pipe_report_start = std::chrono::steady_clock::now();
    pipelined.store(true, std::memory_order_release);
    // PERF-063: SHADGT_EARLY_FENCES=1 signals graphics fences at decode when the work before
    // them reads guest memory only through a capture (=2: also right after fences that waited
    // for readbacks). Needs the command thread's selections as they are (PERF-061).
    if (const char* env = std::getenv("SHADGT_EARLY_FENCES");
        env && (env[0] == '1' || env[0] == '2') && Common::PerfFeatureEnabled(71) &&
        Common::PerfFeatureEnabled(69) && rasterizer && rasterizer->SelectsPipelinesAhead()) {
        early_fences_mode = env[0] - '0';
        LOG_WARNING(Render, "PERF-063: early fences on (mode {})", early_fences_mode);
    }
    LOG_WARNING(
        Render, "PERF-031: draws are recorded on a second thread (draw pipe){}{}",
        read_ahead_states ? ", pipelines read ahead (PERF-034)" : "",
        verify_interval ? fmt::format(", registers verified every {} draws", verify_interval) : "");
}

bool Liverpool::WaitTurnsEnabled() {
    // PERF-048: -DisablePerf 60 drains at once as before.
    static const bool enabled = Common::PerfFeatureEnabled(60);
    return enabled;
}

void Liverpool::Record(Common::UniqueFunction<void>&& work, const char* reason) {
    if (early_fences_mode) {
        UnsafeWindow(reason);
    }
    if (draw_pipe) {
        draw_pipe->Push(std::move(work));
    } else {
        work();
    }
}

void Liverpool::RecordSafe(Common::UniqueFunction<void>&& work) {
    if (draw_pipe) {
        draw_pipe->Push(std::move(work), DrawPipe::JobKind::Neutral);
    } else {
        work();
    }
}

// Captures go back to the pool when the last job reading them is done. The pool outlives
// everything (left to the process exit), so a capture released late still has it.
static std::mutex* const g_capture_pool_mutex = new std::mutex;
static std::vector<Common::ReadCapture*>* const g_capture_pool =
    new std::vector<Common::ReadCapture*>;

std::shared_ptr<Common::ReadCapture> Liverpool::TakeCapture() {
    Common::ReadCapture* capture = nullptr;
    {
        std::scoped_lock lk{*g_capture_pool_mutex};
        if (!g_capture_pool->empty()) {
            capture = g_capture_pool->back();
            g_capture_pool->pop_back();
        }
    }
    if (capture) {
        capture->Reset();
    } else {
        capture = new Common::ReadCapture(8192);
    }
    return std::shared_ptr<Common::ReadCapture>(capture, [](Common::ReadCapture* c) {
        std::scoped_lock lk{*g_capture_pool_mutex};
        if (g_capture_pool->size() < 64) {
            g_capture_pool->push_back(c);
        } else {
            delete c;
        }
    });
}

void Liverpool::UnsafeWindow(const char* reason) {
    if (window_safe) {
        window_safe = false;
        ++early_stats.unsafe[reason];
    }
}

void Liverpool::NewWindow() {
    if (window_capture) {
        ++early_stats.windows_captured;
        early_stats.pages_captured += window_capture->NumPages();
        early_stats.full += window_capture->Full();
    }
    window_capture.reset();
    window_safe = true;
}

bool Liverpool::EarlyFenceAllowed() {
    if (!early_fences_mode || !draw_pipe || !rasterizer) {
        return false;
    }
    if (!window_safe) {
        return false; // counted by UnsafeWindow
    }
    if (fences_delivered.load(std::memory_order_acquire) !=
        fences_decoded.load(std::memory_order_relaxed)) {
        // Fences reach memory in order (FIX-049): an earlier one is still to be recorded.
        ++early_stats.earlier_fence_pending;
        return false;
    }
    if (rasterizer->DeferredFencesPending()) {
        ++early_stats.deferred_pending;
        return false;
    }
    if (early_fences_mode == 1) {
        // The game read GPU results with a recent fence (image readbacks): keep fences in
        // step with the recorder for a while, so it does not read them before they exist.
        const s64 last = rasterizer->LastReadbackFenceNs();
        if (last != 0 && NowNs() - last < 2'000'000'000) {
            ++early_stats.recent_readbacks;
            return false;
        }
    }
    return true;
}

void Liverpool::NoteEarlyFence() {
    ++early_stats.early;
    fences_decoded.fetch_add(1, std::memory_order_relaxed);
    fences_delivered.fetch_add(1, std::memory_order_release);
    if (window_capture) {
        std::scoped_lock lk{early_windows_mutex};
        // Windows the recorder has finished are not guarded any more (their captures go back
        // to the pool).
        const u64 done = draw_pipe->FinishedJobs();
        while (!early_windows.empty() && early_windows.front().end_job <= done) {
            early_windows.pop_front();
        }
        early_windows.push_back({window_capture, draw_pipe->PushedJobs()});
    }
    NewWindow();
}

void Liverpool::WaitForEarlyFences(VAddr page) {
    if (!early_fences_mode || !draw_pipe) {
        return;
    }
    u64 wait_until = 0;
    {
        std::scoped_lock lk{early_windows_mutex};
        const u64 done = draw_pipe->FinishedJobs();
        while (!early_windows.empty() && early_windows.front().end_job <= done) {
            early_windows.pop_front();
        }
        for (const auto& window : early_windows) {
            if (window.capture->Lists(page)) {
                wait_until = std::max(wait_until, window.end_job);
            }
        }
    }
    if (wait_until == 0 || draw_pipe->FinishedJobs() >= wait_until) {
        return;
    }
    Common::Nvtx::Scope nvtx{"CPU fault: wait for early-fenced work"}; // DIAG-056
    const auto start = std::chrono::steady_clock::now();
    while (draw_pipe->FinishedJobs() < wait_until) {
        if (recorder_in_wait.load(std::memory_order_acquire)) {
            // The recorder waits for memory a CPU thread writes; it may be this one.
            ++guard_skipped_in_wait;
            break;
        }
        std::this_thread::yield();
    }
    ++guard_waits;
    guard_wait_us += static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - start)
                                          .count());
}

void Liverpool::ReportEarlyFences() {
    if (!early_fences_mode) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (early_report == std::chrono::steady_clock::time_point{}) {
        early_report = now;
        return;
    }
    if (now - early_report < std::chrono::seconds{2}) {
        return;
    }
    auto& st = early_stats;
    std::string unsafe;
    for (const auto& [reason, n] : st.unsafe) {
        unsafe += fmt::format(" {}={}", reason, n);
    }
    LOG_WARNING(Render,
                "PERF-063 early fences in {:.1f} s: {} signaled at decode, {} in step with the "
                "recorder (windows not covered:{}; earlier fence pending {}, deferred fences {}, "
                "recent readbacks {}); {} windows captured, {:.1f} pages each, {} full; CPU "
                "faults waited {} times, {:.1f} ms ({} not waited: recorder in a wait)",
                std::chrono::duration<double>(now - early_report).count(), st.early, st.normal,
                unsafe.empty() ? " none" : unsafe, st.earlier_fence_pending, st.deferred_pending,
                st.recent_readbacks, st.windows_captured,
                st.windows_captured ? double(st.pages_captured) / st.windows_captured : 0.0,
                st.full, guard_waits.exchange(0), guard_wait_us.exchange(0) / 1000.0,
                guard_skipped_in_wait.exchange(0));
    st = {};
    early_report = now;
}

void Liverpool::RecordDraw(Common::UniqueFunction<void>&& draw, bool compute, bool direct_draw) {
    if (!draw_pipe) {
        draw();
        return;
    }
    // PERF-056: the delta vectors go back and forth between the threads with their capacity.
    std::vector<u32> delta = delta_recycler.Take();
    delta.clear();
    regs_dirty.Collect(std::span<const u32, Regs::NumRegs>{regs.reg_array}, delta);
    std::optional<ComputeProgram> cs;
    if (compute) {
        cs = GetCsRegs();
    }
    std::unique_ptr<Regs> expected;
    if (verify_interval != 0 && ++verify_count % verify_interval == 0) {
        expected = std::make_unique<Regs>(regs);
    }
    // PERF-047: the pipeline is selected now, on this thread, from these registers; the
    // recorder checks the selection against the memory it sees before using it.
    std::optional<Vulkan::Rasterizer::SelectedPipeline> selected;
    if (direct_draw && rasterizer->SelectsPipelinesAhead()) {
        selected = compute ? rasterizer->SelectComputePipelineAhead()
                           : rasterizer->SelectPipelineAhead();
    }
    // PERF-062: the upload epoch this draw is decoded in; its pages are hashed now, on this
    // thread, and the recorder uses those hashes for this draw.
    const u32 decode_epoch = VideoCore::g_upload_epoch.load(std::memory_order_acquire);
    if (selected) {
        rasterizer->HashDrawPagesAhead(*selected, decode_epoch);
    }
    // PERF-063: the pages the draw reads that the CPU can rewrite without a fault are copied
    // now; the recorder reads the copies.
    std::shared_ptr<Common::ReadCapture> capture;
    if (early_fences_mode) {
        if (selected) {
            if (!window_capture) {
                window_capture = TakeCapture();
            }
            VAddr index_address = 0;
            u64 index_size = 0;
            if (!compute && regs.index_base_address.Address<VAddr>() != 0) {
                const u32 index_bytes =
                    regs.index_buffer_type.index_type == IndexType::Index16 ? 2 : 4;
                index_address = regs.index_base_address.Address<VAddr>();
                index_size = u64(regs.max_index_size) * index_bytes;
            }
            if (rasterizer->CaptureDrawReads(*selected, *window_capture, index_address,
                                             index_size)) {
                capture = window_capture;
            } else {
                UnsafeWindow("draw not captured");
            }
        } else {
            UnsafeWindow(!direct_draw ? (compute ? "indirect dispatch" : "indirect draw")
                         : compute ? "dispatch not selected ahead"
                                   : "draw not selected ahead (quad list)");
        }
    }
    // PERF-067: with the buffer stage, the selection and the buffers obtained for it are shared
    // by the stage's work and the recorder's.
    struct StagedDraw {
        std::optional<Vulkan::Rasterizer::SelectedPipeline> selected;
        Vulkan::Rasterizer::BufferPlan plan;
    };
    std::shared_ptr<StagedDraw> staged;
    DrawPipe::Prepare prepare;
    if (draw_pipe->HasStage() && selected) {
        staged = std::make_shared<StagedDraw>();
        staged->selected = std::move(selected);
        selected.reset();
        prepare = [this, staged] {
            return rasterizer->PrepareBuffersAhead(*staged->selected, staged->plan);
        };
    }
    // A draw whose buffers the stage does not obtain is obtained by the recorder: the stage
    // must not run past it.
    const auto kind = staged ? DrawPipe::JobKind::Neutral : DrawPipe::JobKind::Barrier;
    draw_pipe->Push([this, delta = std::move(delta), cs, cb_extent = last_cb_extent,
                     db_extent = last_db_extent, expected = std::move(expected),
                     selected = std::move(selected), draw = std::move(draw),
                     decode_epoch, capture = std::move(capture), staged]() mutable {
        auto& sel = staged ? staged->selected : selected;
        auto& state = *recorder;
        RegsDelta<Regs::NumRegs>::Apply(delta, std::span<u32, Regs::NumRegs>{state.regs.reg_array});
        if (cs) {
            state.cs = *cs;
        }
        state.cb_extent = cb_extent;
        state.db_extent = db_extent;
        if (expected) {
            VerifyRecorderRegs(*expected);
        }
        rasterizer->SetSelectedPipeline(sel ? &*sel : nullptr);
        rasterizer->SetDrawEpoch(sel ? decode_epoch : 0);
        rasterizer->SetReadCapture(capture.get());
        rasterizer->SetBufferPlan(staged ? &staged->plan : nullptr);
        draw();
        rasterizer->SetBufferPlan(nullptr);
        rasterizer->SetReadCapture(nullptr);
        rasterizer->SetDrawEpoch(0);
        rasterizer->SetSelectedPipeline(nullptr);
        delta_recycler.Give(std::move(delta));
        if (sel) {
            rasterizer->RecycleSelectedPipeline(std::move(*sel));
        }
    }, kind, std::move(prepare));
    if (read_ahead_states && !compute) {
        OfferDrawState();
    }
    ReportDrawPipe();
    ReportEarlyFences();
}

// PERF-037: the registers the draw packets write, left out of the read-ahead state hash.
#define REG_DWORD(field) (offsetof(Regs, field) / sizeof(u32))
static_assert(REG_DWORD(index_base_address) == ReadAheadStates::PerDrawRegisters[0] &&
              REG_DWORD(draw_initiator) == ReadAheadStates::PerDrawRegisters[2] &&
              REG_DWORD(max_index_size) == ReadAheadStates::PerDrawRegisters[3] &&
              REG_DWORD(num_indices) == ReadAheadStates::PerDrawRegisters[4] &&
              sizeof(Regs::index_base_address) == 2 * sizeof(u32));
#undef REG_DWORD

// FIX-014: draws the rasterizer filters out (Rasterizer::FilterDraw) never get a pipeline.
static bool FilteredDraw(const Regs& state) {
    using OperationMode = ColorControl::OperationMode;
    const auto mode = state.color_control.mode;
    return state.primitive_type == PrimitiveType::None ||
           mode == OperationMode::EliminateFastClear || mode == OperationMode::FmaskDecompress ||
           mode == OperationMode::Resolve;
}

void Liverpool::OfferDrawState() {
    // The draw being decoded is still ahead of the recorder. Its state is hashed only when a
    // register its pipeline key comes from was written since the previous draw.
    if (pipeline_regs_written && read_ahead_states->Space() > 0) {
        pipeline_regs_written = false;
        if (!FilteredDraw(regs)) {
            read_ahead_states->Offer(regs.reg_array);
        }
    }
    // A pipeline miss on the recorder: new pipelines are appearing, so the draws after this
    // one are read now, not only from the next command buffer on.
    // A read-ahead that stopped on a full queue goes on once a quarter of it is free again.
    if (read_ahead_states->TakeNewMiss() ||
        (read_ahead_stopped_early && read_ahead_states->Space() >= 128)) {
        PipeReadAhead();
    }
}

void Liverpool::PipeReadAhead() {
    const size_t space = read_ahead_states->Space();
    if (space == 0) {
        read_ahead_stopped_early = true; // Again at the next draw.
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    read_ahead_stopped_early = ScanAheadForPipelines(
        [&](const Regs& state) {
            return !FilteredDraw(state) && read_ahead_states->Offer(state.reg_array);
        },
        4096, static_cast<u32>(std::min<size_t>(space, 256)));
    ++pipe_read_ahead_stats.scans;
    pipe_read_ahead_stats.ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void Liverpool::SyncRecorder(std::string_view reason) {
    if (!draw_pipe) {
        return;
    }
    Common::Nvtx::Scope nvtx{"drain recorder"}; // DIAG-056
    const auto start = std::chrono::steady_clock::now();
    draw_pipe->Drain();
    auto& entry = sync_reasons[reason];
    ++entry.first;
    entry.second +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void Liverpool::RecordLabelWrite(VAddr address, std::vector<u8> value,
                                 Common::UniqueFunction<void>&& work) {
    if (!draw_pipe || value.empty() || address == 0) {
        Record(std::move(work));
        return;
    }
    const u64 job = ++label_jobs;
    if (early_fences_mode) {
        // PERF-063: a fence written by the recorder; the next window starts after it.
        ++early_stats.normal;
        fences_decoded.fetch_add(1, std::memory_order_relaxed);
        NewWindow();
    }
    if (waited_label != 0 && address <= waited_label && waited_label < address + value.size()) {
        ++labels_written_while_waiting; // DIAG-047
    }
    {
        std::scoped_lock lk{pending_labels_mutex};
        pending_labels[address] = PendingLabel{std::move(value), job};
    }
    draw_pipe->Push([this, address, job, work = std::move(work)] {
        work();
        if (early_fences_mode) {
            fences_delivered.fetch_add(1, std::memory_order_release);
        }
        std::scoped_lock lk{pending_labels_mutex};
        if (const auto it = pending_labels.find(address);
            it != pending_labels.end() && it->second.job == job) {
            pending_labels.erase(it);
        }
    });
}

bool Liverpool::PendingLabelSatisfies(const PM4CmdWaitRegMem& wait) {
    if (!draw_pipe || wait.mem_space != PM4CmdWaitRegMem::MemSpace::Memory) {
        return false;
    }
    const auto address = reinterpret_cast<VAddr>(wait.Address<u32*>());
    std::scoped_lock lk{pending_labels_mutex};
    for (const VAddr start : {address, address - sizeof(u32)}) {
        const auto it = pending_labels.find(start);
        if (it == pending_labels.end() || address + sizeof(u32) > start + it->second.bytes.size()) {
            continue;
        }
        u32 value;
        std::memcpy(&value, it->second.bytes.data() + (address - start), sizeof(u32));
        return wait.TestValue(value);
    }
    return false;
}

void Liverpool::RecorderWaitRegMem(const PM4CmdWaitRegMem& wait) {
    // PERF-063: a CPU fault does not wait for the recorder while it may wait for that CPU.
    recorder_in_wait.store(true, std::memory_order_release);
    SCOPE_EXIT {
        recorder_in_wait.store(false, std::memory_order_release);
    };
    // The job that writes the label ran before this one. If its fence was deferred, the value
    // reaches memory when the GPU finishes, written by the scheduler's completion thread.
    const auto& state_regs = recorder->regs.reg_array;
    if (wait.Test(state_regs) || SatisfiedByPendingFence(rasterizer, &wait)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    rasterizer->FlushForDeferredFences();
    while (!wait.Test(state_regs) && !SatisfiedByPendingFence(rasterizer, &wait)) {
        draw_pipe->RunUrgent(); // PERF-050: no job is half done here
        std::this_thread::yield();
    }
    VideoCore::BumpUploadEpoch();
    static u32 reports = 0;
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (ms > 100.0 && reports++ < 10) {
        LOG_WARNING(Render, "PERF-031: the recorder waited {:.0f} ms for label {:#x}", ms,
                    reinterpret_cast<uintptr_t>(wait.Address<u32*>()));
    }
}

void Liverpool::VerifyRecorderRegs(const Regs& expected) {
    // SHADGT_DRAW_PIPE_VERIFY=N: a register write the command thread did not mark would leave
    // the recorder's copy stale; name the first dword that differs.
    static u32 reports = 0;
    const auto& actual = recorder->regs.reg_array;
    for (u32 i = 0; i < Regs::NumRegs; ++i) {
        if (actual[i] != expected.reg_array[i]) {
            if (reports++ < 20) {
                LOG_ERROR(Render, "PERF-031: recorder register {:#x} is {:#x}, expected {:#x}", i,
                          actual[i], expected.reg_array[i]);
            }
            recorder->regs = expected;
            return;
        }
    }
}

void Liverpool::ReportDrawPipe() {
    const auto now = std::chrono::steady_clock::now();
    if (now - pipe_report_start < std::chrono::seconds{2}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - pipe_report_start).count();
    pipe_report_start = now;
    const auto stats = draw_pipe->TakeStats();
    std::string reasons;
    for (const auto& [reason, entry] : sync_reasons) {
        reasons += fmt::format(" {}={}/{:.1f}ms", reason, entry.first, entry.second);
    }
    sync_reasons.clear();
    std::string read_ahead;
    if (read_ahead_states) {
        read_ahead = fmt::format("; read-ahead: {} draw states queued, {} reads ahead ({:.1f} ms)",
                                 read_ahead_states->TakeOffered(), pipe_read_ahead_stats.scans,
                                 pipe_read_ahead_stats.ms);
        pipe_read_ahead_stats = {};
    }
    if (commands_recorded != 0) {
        read_ahead += fmt::format("; {} commands run on the recorder", commands_recorded);
        commands_recorded = 0;
    }
    if (labels_written_while_waiting != 0) {
        read_ahead += fmt::format("; {} waited-on labels written by queued jobs during the wait",
                                  labels_written_while_waiting);
        labels_written_while_waiting = 0;
    }
    if (const u64 checked = diag_moved_waits_checked.exchange(0); checked != 0) {
        read_ahead += fmt::format("; DIAG-054: {} moved waits checked, {} with changed commands",
                                  checked, diag_moved_waits_changed.exchange(0));
    }
    if (waits_moved_after_turns != 0) {
        read_ahead += fmt::format("; {} waits moved after the other queues decoded",
                                  waits_moved_after_turns);
        waits_moved_after_turns = 0;
    }
    {
        std::scoped_lock lk{submit_timing_mutex};
        const auto& t = submit_stats;
        if (t.count != 0) {
            read_ahead += fmt::format(
                "; DIAG-051: {} gfx submits, after the game's submit: decoded {:.2f} ms, recorded "
                "{:.2f} ms, GPU done {:.2f} ms (max {:.1f}); game submits every {:.2f} ms; "
                "DIAG-055: submits come {:.2f} ms after the last EOP interrupt; EOPs {} deferred, "
                "{} at once, {} EXP-063 early fences",
                t.count, t.decode_us / 1000.0 / t.count, t.record_us / 1000.0 / t.count,
                t.gpu_count ? t.gpu_us / 1000.0 / t.gpu_count : 0.0, t.max_gpu_us / 1000.0,
                t.gaps ? t.gap_us / 1000.0 / t.gaps : 0.0,
                t.eop_to_submit_count ? t.eop_to_submit_us / 1000.0 / t.eop_to_submit_count
                                      : 0.0,
                g_eops_deferred.exchange(0), g_eops_immediate.exchange(0),
                g_fences_early.exchange(0));
        }
        submit_stats = {};
    }
    if (const u64 flushes = fault_flushes.exchange(0); flushes != 0) {
        const u64 wait_us = fault_flush_wait_us.exchange(0);
        read_ahead += fmt::format(
            "; DIAG-050: {} CPU fault flushes, game threads waited {:.1f} ms (avg {:.2f}, max "
            "{:.1f})",
            flushes, wait_us / 1000.0, wait_us / 1000.0 / flushes,
            fault_flush_max_us.exchange(0) / 1000.0);
    }
    LOG_WARNING(Render,
                "PERF-031 draw pipe in {:.1f} s: {} jobs, recorder busy {:.0f}%, max queued {}, "
                "{} full-queue waits; {} waits moved to the recorder; drains {} ({} waited, "
                "{:.1f} ms):{}{}",
                seconds, stats.pushed, stats.recorder_busy_us / (seconds * 1e4), stats.max_queued,
                stats.push_waits, waits_moved, stats.drains, stats.drains_that_waited,
                stats.drain_wait_us / 1000.0, reasons, read_ahead);
    if (draw_pipe->HasStage()) {
        LOG_WARNING(Render,
                    "PERF-067 buffer stage in {:.1f} s: {} draws prepared ahead, {} waits for the "
                    "recorder to catch up ({} at barrier jobs)",
                    seconds, stats.stage_prepared, stats.stage_syncs, stats.stage_barriers);
    }
    waits_moved = 0;
}

bool Liverpool::WritesLiveCommands(VAddr address, u64 size) const {
    for (const auto& span : live_cmd_buffers) {
        const auto begin = reinterpret_cast<VAddr>(span.data());
        const auto end = begin + span.size_bytes();
        if (address < end && address + size > begin) {
            return true;
        }
    }
    return false;
}

bool Liverpool::IsGpuThread(std::thread::id id) const {
    return id == gpu_id || (Pipelined() && id == draw_pipe->RecorderThreadId());
}

#ifdef __linux__
bool Liverpool::IsGpuThreadTid(u32 tid) const {
    return tid == gpu_tid || (Pipelined() && tid == recorder_tid.load());
}
#endif

void Liverpool::OnGpuThreadFault() {
    if (Pipelined() && std::this_thread::get_id() == gpu_id) {
        draw_pipe->Drain();
    }
}

void Liverpool::SendFaultCommand(Common::UniqueFunction<void>&& func) {
    static const bool urgent = Common::PerfFeatureEnabled(62);
    Common::Nvtx::Scope nvtx{"CPU fault: wait for GPU data"}; // DIAG-056
    const auto start = std::chrono::steady_clock::now();
    if (draw_pipe && urgent) {
        std::binary_semaphore sem{0};
        draw_pipe->PushUrgent([&sem, &func] {
            func();
            sem.release();
        });
        sem.acquire();
    } else {
        SendCommand<true>([&func] { func(); });
    }
    const u64 us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - start)
                                        .count());
    fault_flushes.fetch_add(1, std::memory_order_relaxed);
    fault_flush_wait_us.fetch_add(us, std::memory_order_relaxed);
    u64 max = fault_flush_max_us.load(std::memory_order_relaxed);
    while (us > max && !fault_flush_max_us.compare_exchange_weak(max, us)) {
    }
}

bool Liverpool::CommandThreadFaultIsUrgent() {
    static const bool urgent = Common::PerfFeatureEnabled(62) && Common::PerfFeatureEnabled(65);
    return urgent && Pipelined() && std::this_thread::get_id() == gpu_id &&
           !draw_pipe->IsExclusive();
}

void Liverpool::ProcessCommands() {
    if (draw_pipe) {
        draw_pipe->RunUrgentIfExclusive(); // PERF-050
    }
    if (!num_commands) {
        return;
    }
    // PERF-031: commands from other threads (flips, cache flushes, unmaps) use the caches.
    // PERF-035: with the draw pipe they run on the recorder, in order behind the work decoded so
    // far (where a drain would have run them here), and decoding goes on meanwhile. A sender
    // that waits for its command still returns only once it ran. -DisablePerf 50 drains instead.
    static const bool commands_on_recorder = Common::PerfFeatureEnabled(50);
    const bool record = draw_pipe && commands_on_recorder;
    if (!record) {
        SyncRecorder("commands");
    }
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        if (record) {
            ++commands_recorded;
            Record(std::move(callback), "queued command (flip, fault)");
        } else {
            callback();
        }
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadGT:GpuCommandProcessor");
    Common::GuestClock::MarkGpuCommandThread(); // FIX-044
    Common::SamplingProfiler::RegisterCurrentThread("GpuCommandProcessor"); // DIAG-048
    gpu_id = std::this_thread::get_id();
#ifdef __linux__
    gpu_tid = gettid();
#endif

    while (!stoken.stop_requested()) {
        if (draw_pipe) {
            draw_pipe->ReleaseExclusive(); // PERF-050: idle; urgent work may run on the recorder
        }
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_tasks || submit_done; });
        }
        if (stoken.stop_requested()) {
            break;
        }
        if (!draw_pipe && rasterizer && DrawPipeRequested()) {
            StartDrawPipe();
        }

        VideoCore::StartCapture();

        curr_qid = -1;

        while (num_tasks || num_commands) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            task.resume();

            if (task.done()) {
                task.destroy();

                {
                    std::scoped_lock lock{queue.m_access};
                    queue.submits.pop();
                }
                --num_tasks;
                // DIAG-051: a graphics submission is decoded; time it through the pipeline.
                std::optional<std::chrono::steady_clock::time_point> submitted;
                std::optional<SubmitRanges> ranges;
                if (curr_qid == GfxQueueId) {
                    std::scoped_lock lk{submit_timing_mutex};
                    if (!gfx_submit_times.empty()) {
                        submitted = gfx_submit_times.front();
                        gfx_submit_times.pop_front();
                    }
                    if (!gfx_submit_ranges.empty()) {
                        ranges = gfx_submit_ranges.front();
                        gfx_submit_ranges.pop_front();
                        Common::Nvtx::End(ranges->to_decoded);
                    }
                }
                const auto decoded = std::chrono::steady_clock::now();
                // PERF-031: the submission is finished once its recorded work is.
                RecordSafe([this, submitted, decoded, ranges] {
                    --num_submits;
                    u64 gpu_range = 0;
                    if (ranges) {
                        Common::Nvtx::End(ranges->to_recorded);
                        gpu_range = Common::Nvtx::Start(
                            fmt::format("gfx submit {}: recorded until GPU done", ranges->number));
                        if (!rasterizer) {
                            Common::Nvtx::End(gpu_range);
                        }
                    }
                    if (!submitted && rasterizer && gpu_range) {
                        rasterizer->WhenGpuDone([gpu_range] { Common::Nvtx::End(gpu_range); });
                    }
                    if (submitted && rasterizer) {
                        const auto recorded = std::chrono::steady_clock::now();
                        const auto us = [&](auto d) {
                            return static_cast<u64>(
                                std::chrono::duration_cast<std::chrono::microseconds>(d).count());
                        };
                        {
                            std::scoped_lock lk{submit_timing_mutex};
                            ++submit_stats.count;
                            submit_stats.decode_us += us(decoded - *submitted);
                            submit_stats.record_us += us(recorded - *submitted);
                        }
                        rasterizer->WhenGpuDone([this, start = *submitted, us, gpu_range] {
                            Common::Nvtx::End(gpu_range);
                            const u64 gpu = us(std::chrono::steady_clock::now() - start);
                            std::scoped_lock lk{submit_timing_mutex};
                            submit_stats.gpu_us += gpu;
                            ++submit_stats.gpu_count;
                            submit_stats.max_gpu_us = std::max(submit_stats.max_gpu_us, gpu);
                        });
                    }
                    std::scoped_lock lock2{submit_mutex};
                    submit_cv.notify_all();
                });
            }
        }

        // PERF-031: the GPU is idle once everything decoded so far is recorded.
        if (submit_done) {
            submit_done = false;
            // PERF-063: submits the recorded work; reads and writes no guest memory.
            RecordSafe([this] {
                VideoCore::EndCapture();
                if (rasterizer) {
                    rasterizer->OnSubmit();
                    rasterizer->Flush();
                }
            });
        }
        RecordSafe([] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle); });
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const u32 packet_words = header->type3.NumWords() + 1;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            // PERF-031: queued draws may still read the memory this overwrites.
            SyncRecorder("ce dump");
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            // DIAG-009: remember recent dump destinations for the tonemap trace.
            recent_const_dumps[const_dump_sequence % recent_const_dumps.size()] = {
                .address = dump_const->Address<VAddr>(),
                .size = dump_const->Size(),
                .ce_count = cblock.ce_count,
                .de_count = cblock.de_count,
                .sequence = ++const_dump_sequence,
            };
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            GpuWaitDiagnostics diagnostics;
            while (ShouldWaitOnDeCounter(cblock.ce_count, cblock.de_count, diff)) {
                if (diagnostics.Ready()) {
                    LOG_WARNING(Render, "GPU CE wait stalled: CE={} DE={} limit={}",
                                cblock.ce_count, cblock.de_count, diff);
                }
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, packet_words);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                                           std::span<const u32> original_dcb) {
    FIBER_ENTER(dcb_task_name);
    if (!original_dcb.empty()) {
        RecordCmdBuffer(original_dcb.data(), original_dcb.size_bytes(), false);
    }

    cblock.Reset();
    // PERF-019: a command buffer can be reused with other contents; never continue an earlier
    // read-ahead into this one.
    scan_buffer_end = nullptr;
    if (lookahead_outer.empty()) {
        scanned_packets.clear();
    }
    // PERF-023: a nested command buffer an earlier read-ahead already read needs no new one.
    // PERF-031: the read-ahead shares the pipeline cache with the recorder thread; it is off
    // while the draw pipe runs.
    if (on_command_buffer_start && !dcb.empty() && !draw_pipe &&
        !(Common::PerfFeatureEnabled(23) && scanned_packets.contains(dcb.data()))) {
        lookahead_dcb = dcb;
        on_command_buffer_start();
    }
    // PERF-034: with the draw pipe the command thread reads ahead itself, from each command
    // buffer's start while new pipelines keep appearing (PERF-020).
    if (read_ahead_states && !dcb.empty() &&
        read_ahead_states->MissWithin(std::chrono::seconds{5}) &&
        !(Common::PerfFeatureEnabled(23) && scanned_packets.contains(dcb.data()))) {
        lookahead_dcb = dcb;
        PipeReadAhead();
    }

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    const auto submitted_dcb = dcb;
    const auto live_dcb = original_dcb.empty() ? submitted_dcb : original_dcb;
    live_cmd_buffers.push_back(live_dcb);
    // DIAG-052: the last packets decoded, for an invalid-packet report.
    std::array<std::pair<u32, u32>, 16> recent_packets{};
    u32 num_recent = 0;
    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        recent_packets[num_recent++ % recent_packets.size()] = {
            static_cast<u32>(dcb.data() - submitted_dcb.data()), header->raw};

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0: {
            const auto write = DecodeType0RegisterWrite(dcb, regs.reg_array.size());
            if (!write) {
                const size_t word_offset = dcb.data() - submitted_dcb.data();
                const size_t start = word_offset > 16 ? word_offset - 16 : 0;
                const auto context = submitted_dcb.subspan(
                    start, std::min<size_t>(32, submitted_dcb.size() - start));
                LOG_CRITICAL(Render,
                             "Invalid PM4 submission base={:#x}, dword offset={}, context starts "
                             "at dword {}: {:#010x}",
                             base_addr, word_offset, start, fmt::join(context, " "));
                // DIAG-052: the packets before it (dword offset:header), the submission's size,
                // whether the live buffer still holds the same words, and a wider context.
                std::string recent;
                for (u32 i = 0; i < std::min<u32>(num_recent, 16); ++i) {
                    const auto& [offset, raw] =
                        recent_packets[(num_recent - 1 - i) % recent_packets.size()];
                    recent += fmt::format(" {}:{:#010x}", offset, raw);
                }
                const size_t wide_start = word_offset > 96 ? word_offset - 96 : 0;
                const auto wide = submitted_dcb.subspan(
                    wide_start, std::min<size_t>(112, submitted_dcb.size() - wide_start));
                const bool same = live_dcb.size() == submitted_dcb.size() &&
                                  std::equal(live_dcb.begin(), live_dcb.end(), submitted_dcb.begin());
                LOG_CRITICAL(Render,
                             "DIAG-052: submission of {} dwords (copied: {}, live buffer still "
                             "equal: {}); last packets (newest first):{}; dwords from {}: {:#010x}",
                             submitted_dcb.size(), !original_dcb.empty(), same, recent, wide_start,
                             fmt::join(wide, " "));
                // DIAG-052: the emulator's own recent writes near the packets before it.
                const VAddr bad = reinterpret_cast<VAddr>(dcb.data());
                LOG_CRITICAL(Render, "DIAG-052: emulator writes within 256 bytes of {:#x}:{}",
                             bad, Core::Memory::Instance()->DescribeBackingWrites(bad - 128, 256));
            }
            ASSERT_MSG(write.has_value(),
                       "Invalid PM4 type 0 at {:#x}: header={:#x}, remaining dwords={}",
                       reinterpret_cast<uintptr_t>(dcb.data()), header->raw, dcb.size());
            std::memcpy(&regs.reg_array[write->first_register], write->values.data(),
                        write->values.size_bytes());
            MarkRegs(write->first_register, static_cast<u32>(write->values.size()));

            // The graphics queue keeps compute shader state separately from the graphics regs.
            constexpr u32 cs_first = Regs::ShRegWordOffset + 0x200;
            constexpr u32 cs_end = cs_first + sizeof(ComputeProgram) / sizeof(u32);
            const u32 first = std::max(write->first_register, cs_first);
            const u32 end = std::min<u32>(write->first_register + write->values.size(), cs_end);
            if (first < end) {
                auto* cs = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state);
                std::memcpy(cs + first - cs_first, &regs.reg_array[first],
                            (end - first) * sizeof(u32));
            }
            dcb = NextPacket(dcb, write->values.size() + 1);
            continue;
        }
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            ++Common::GetWorkCounters().pm4_packets;
            lookahead_dcb =
                dcb.size() > count + 1 ? dcb.subspan(count + 1) : std::span<const u32>{};
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    RecordSafe(
                        [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip); });
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        std::string label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                          marker_sz};
                        RecordSafe([this, label = std::move(label)] {
                            rasterizer->ScopeMarkerBegin(label, true);
                        });
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        std::string label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                          marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        RecordSafe([this, label = std::move(label), color] {
                            rasterizer->ScopedMarkerInsertColor(label, color, true);
                        });
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        RecordSafe([this] { rasterizer->ScopeMarkerEnd(true); });
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                regs_dirty.MarkAll();
                pipeline_regs_written = true;
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                MarkRegs(reg_addr, count - 1);
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                MarkRegs(reg_addr, count - 1);

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                    MarkRegs(Regs::ShRegWordOffset + set_data->reg_offset, count - 1);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                MarkRegs(Regs::UconfigRegWordOffset + set_data->reg_offset, count - 1);
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                MarkReg(regs.index_buffer_type);
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                MarkReg(regs.max_index_size);
                MarkReg(regs.index_base_address);
                MarkReg(regs.num_indices);
                MarkReg(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                RecordDraw(
                    [this, cmd_address] {
                        rasterizer->ScopeMarker("gfx:{}:DrawIndex2",
                                                fmt::make_format_args(cmd_address),
                                                [&] { rasterizer->Draw(true); });
                    },
                    false, true);
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                MarkReg(regs.max_index_size);
                MarkReg(regs.num_indices);
                MarkReg(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const u32 index_offset = draw_index_off->index_offset;
                RecordDraw(
                    [this, cmd_address, index_offset] {
                        rasterizer->ScopeMarker("gfx:{}:DrawIndexOffset2",
                                                fmt::make_format_args(cmd_address),
                                                [&] { rasterizer->Draw(true, index_offset); });
                    },
                    false, true);
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                MarkReg(regs.num_indices);
                MarkReg(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                RecordDraw(
                    [this, cmd_address] {
                        rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto",
                                                fmt::make_format_args(cmd_address),
                                                [&] { rasterizer->Draw(false); });
                    },
                    false, true);
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const VAddr args_addr = indirect_args_addr;
                const u16 base_vtx_loc = draw_indirect->base_vtx_loc;
                const u16 start_inst_loc = draw_indirect->start_inst_loc;
                RecordDraw(
                    [this, cmd_address, args_addr, offset, stride, base_vtx_loc, start_inst_loc] {
                        rasterizer->ScopeMarker(
                            "gfx:{}:DrawIndirect", fmt::make_format_args(cmd_address), [&] {
                                rasterizer->DrawIndirect(false, args_addr, offset, stride, 1, 0,
                                                         base_vtx_loc, start_inst_loc);
                            });
                    },
                    false);
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const VAddr args_addr = indirect_args_addr;
                const u32 stride = draw_indirect->stride;
                const u32 max_count = draw_indirect->count;
                const u16 base_vtx_loc = draw_indirect->base_vtx_loc;
                const u16 start_inst_loc = draw_indirect->start_inst_loc;
                RecordDraw(
                    [this, cmd_address, args_addr, offset, stride, max_count, base_vtx_loc,
                     start_inst_loc] {
                        rasterizer->ScopeMarker(
                            "gfx:{}:DrawIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                                rasterizer->DrawIndirect(false, args_addr, offset, stride,
                                                         max_count, 0, base_vtx_loc,
                                                         start_inst_loc);
                            });
                    },
                    false);
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const VAddr args_addr = indirect_args_addr;
                const u16 base_vtx_loc = draw_index_indirect->base_vtx_loc;
                const u16 start_inst_loc = draw_index_indirect->start_inst_loc;
                RecordDraw(
                    [this, cmd_address, args_addr, offset, stride, base_vtx_loc, start_inst_loc] {
                        rasterizer->ScopeMarker(
                            "gfx:{}:DrawIndexIndirect", fmt::make_format_args(cmd_address), [&] {
                                rasterizer->DrawIndirect(true, args_addr, offset, stride, 1, 0,
                                                         base_vtx_loc, start_inst_loc);
                            });
                    },
                    false);
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const VAddr args_addr = indirect_args_addr;
                const u32 stride = draw_index_indirect->stride;
                const u32 max_count = draw_index_indirect->count;
                const u16 base_vtx_loc = draw_index_indirect->base_vtx_loc;
                const u16 start_inst_loc = draw_index_indirect->start_inst_loc;
                RecordDraw(
                    [this, cmd_address, args_addr, offset, stride, max_count, base_vtx_loc,
                     start_inst_loc] {
                        rasterizer->ScopeMarker("gfx:{}:DrawIndexIndirectMulti",
                                                fmt::make_format_args(cmd_address), [&] {
                                                    rasterizer->DrawIndirect(
                                                        true, args_addr, offset, stride, max_count,
                                                        0, base_vtx_loc, start_inst_loc);
                                                });
                    },
                    false);
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const VAddr args_addr = indirect_args_addr;
                const u32 stride = draw_index_indirect->stride;
                const u32 max_count = draw_index_indirect->count;
                const VAddr count_addr = draw_index_indirect->count_indirect_enable.Value()
                                             ? draw_index_indirect->count_addr
                                             : 0;
                const u16 base_vtx_loc = draw_index_indirect->base_vtx_loc;
                const u16 start_inst_loc = draw_index_indirect->start_inst_loc;
                RecordDraw(
                    [this, cmd_address, args_addr, offset, stride, max_count, count_addr,
                     base_vtx_loc, start_inst_loc] {
                        rasterizer->ScopeMarker("gfx:{}:DrawIndexIndirectCountMulti",
                                                fmt::make_format_args(cmd_address), [&] {
                                                    rasterizer->DrawIndirect(
                                                        true, args_addr, offset, stride, max_count,
                                                        count_addr, base_vtx_loc, start_inst_loc);
                                                });
                    },
                    false);
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                RecordDraw(
                    [this, cmd_address] {
                        rasterizer->ScopeMarker("gfx:{}:DispatchDirect",
                                                fmt::make_format_args(cmd_address),
                                                [&] { rasterizer->DispatchDirect(); });
                    },
                    true, true);
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const VAddr args_addr = indirect_args_addr;
                RecordDraw(
                    [this, cmd_address, args_addr, offset, size] {
                        rasterizer->ScopeMarker(
                            "gfx:{}:DispatchIndirect", fmt::make_format_args(cmd_address),
                            [&] { rasterizer->DispatchIndirect(args_addr, offset, size); });
                    },
                    true);
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                MarkReg(regs.num_instances);
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                MarkReg(regs.index_base_address);
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                MarkReg(regs.num_indices);
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                    MarkReg(regs.cp_strmout_cntl);
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        const u64 counter = pixel_counter;
                        pixel_counter += OcclusionCounterStep;
                        // PERF-031: written in order with the draws before it.
                        const auto write_results = [results, counter, pairs = num_counter_pairs] {
                            u64* result = results;
                            for (s32 i = 0; i < s32(pairs); ++i, result += 2) {
                                *result = counter | OcclusionCounterValidMask;
                            }
                        };
                        if (early_fences_mode) {
                            // PERF-063: the values do not depend on the draws; written now, so
                            // the window's fence may be signaled at decode.
                            write_results();
                        } else {
                            Record(write_results);
                        }
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto event = *reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (draw_pipe && rasterizer &&
                    event.command != PM4CmdEventWriteEos::Command::GdsStore &&
                    (EarlyFencesActive() || EarlyFenceAllowed())) {
                    // EXP-063: signaled now; the recorder only submits at this point.
                    event.SignalFence([](void* address, u64 data, u32 num_bytes) {
                        Core::Memory::Instance()->TryWriteBacking(address, &data, num_bytes);
                    });
                    ++g_fences_early;
                    RecordSafe(
                        [this] { rasterizer->OnFence(Vulkan::Rasterizer::DrainSource::GfxEos); });
                    if (early_fences_mode) {
                        NoteEarlyFence();
                    }
                    break;
                }
                // PERF-031: fences are signaled in order behind the recorded work.
                RecordLabelWrite(
                    event.command != PM4CmdEventWriteEos::Command::GdsStore ? event.Address<VAddr>()
                                                                            : 0,
                    FenceValueBytes(DataSelect::Data32Low, event.DataDWord(), 0), [this, event] {
                        const auto* event_eos = &event;
                        const bool deferred =
                            rasterizer &&
                            event_eos->command != PM4CmdEventWriteEos::Command::GdsStore &&
                            rasterizer->DeferFenceSignal(
                                event_eos->Address<VAddr>(),
                                [event] {
                                    event.SignalFence([](void* address, u64 data, u32 num_bytes) {
                                        WriteDeferredFence(address, &data, num_bytes);
                                    });
                                },
                                false,
                                FenceValueBytes(DataSelect::Data32Low, event_eos->DataDWord(), 0));
                        if (!deferred) {
                            if (rasterizer) {
                                rasterizer->OnFence(Vulkan::Rasterizer::DrainSource::GfxEos);
                            }
                            event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                                auto* memory = Core::Memory::Instance();
                                ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                            });
                        }
                        if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                            ASSERT(event_eos->size == 1);
                            if (rasterizer) {
                                rasterizer->FinishForGds();
                                const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                                *event_eos->Address() = value;
                            }
                        }
                    });
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto event = *reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (draw_pipe && rasterizer && (EarlyFencesActive() || EarlyFenceAllowed())) {
                    // EXP-063: signaled now; the recorder only submits at this point.
                    event.SignalFence(
                        [](void* address, u64 data, u32 num_bytes) {
                            Core::Memory::Instance()->TryWriteBacking(address, &data, num_bytes);
                        },
                        [] {
                            NoteEop();
                            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                        });
                    ++g_fences_early;
                    RecordSafe(
                        [this] { rasterizer->OnFence(Vulkan::Rasterizer::DrainSource::GfxEop); });
                    if (early_fences_mode) {
                        NoteEarlyFence();
                    }
                    break;
                }
                // PERF-031: fences are signaled in order behind the recorded work.
                RecordLabelWrite(
                    reinterpret_cast<VAddr>(event.Address<u32>()),
                    FenceValueBytes(event.data_sel.Value(), event.DataDWord(), event.DataQWord()),
                    [this, event] {
                        const auto* event_eop = &event;
                        const bool deferred =
                            rasterizer &&
                            rasterizer->DeferFenceSignal(
                                reinterpret_cast<VAddr>(event_eop->Address<u32>()),
                                [event] {
                                    event.SignalFence(
                                        [](void* address, u64 data, u32 num_bytes) {
                                            WriteDeferredFence(address, &data, num_bytes);
                                        },
                                        [] {
                                            NoteEop();
                                            ++g_eops_deferred;
                                            Platform::IrqC::Instance()->Signal(
                                                Platform::InterruptId::GfxEop);
                                        });
                                },
                                false,
                                FenceValueBytes(event_eop->data_sel.Value(), event_eop->DataDWord(),
                                                event_eop->DataQWord()));
                        if (!deferred) {
                            if (rasterizer) {
                                rasterizer->OnFence(Vulkan::Rasterizer::DrainSource::GfxEop);
                            }
                            event_eop->SignalFence(
                                [](void* address, u64 data, u32 num_bytes) {
                                    auto* memory = Core::Memory::Instance();
                                    ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                                },
                                [] {
                                    NoteEop();
                                    ++g_eops_immediate;
                                    Platform::IrqC::Instance()->Signal(
                                        Platform::InterruptId::GfxEop);
                                });
                        }
                    });
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_packet = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_packet->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                if (SkipRegisterSpaceDmaData(dma_packet, "gfx")) {
                    break;
                }
                // PERF-031: copies run in order with the draws around them. One that writes
                // a command buffer still being decoded runs before decoding goes on.
                const auto dma = *dma_packet;
                const bool to_memory =
                    dma.dst_sel == DmaDataDst::Memory || dma.dst_sel == DmaDataDst::MemoryUsingL2;
                RecordDma([this, dma] {
                    const auto* dma_data = &dma;
                    if (dma_data->src_sel == DmaDataSrc::Data &&
                        dma_data->dst_sel == DmaDataDst::Gds) {
                        rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                               dma_data->data, true);
                    } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                                dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                               dma_data->dst_sel == DmaDataDst::Gds) {
                        rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                               dma_data->NumBytes(), true, false);
                    } else if (dma_data->src_sel == DmaDataSrc::Data &&
                               (dma_data->dst_sel == DmaDataDst::Memory ||
                                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                               dma_data->data, false);
                    } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                               (dma_data->dst_sel == DmaDataDst::Memory ||
                                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                               dma_data->NumBytes(), false, true);
                    } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                                dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                               (dma_data->dst_sel == DmaDataDst::Memory ||
                                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                        rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                               dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                               false, false);
                    } else {
                        UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                        u32(dma_data->src_sel), u32(dma_data->dst_sel));
                    }
                });
                if (to_memory && WritesLiveCommands(dma.DstAddress<VAddr>(), dma.NumBytes())) {
                    SyncRecorder("dma into commands");
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (!write_data->wr_one_addr.Value()) {
                    // Keep guest-visible writes in order behind deferred fences.
                    std::vector<u8> data(data_size);
                    std::memcpy(data.data(), write_data->data, data_size);
                    // PERF-031: in order with the recorded work; before decoding goes on when it
                    // writes a command buffer being decoded.
                    const bool into_commands =
                        WritesLiveCommands(reinterpret_cast<VAddr>(address), data_size);
                    std::vector<u8> label = data;
                    RecordLabelWrite(
                        reinterpret_cast<VAddr>(address), std::move(label),
                        [this, address, data = std::move(data)]() mutable {
                            const u32 data_size = static_cast<u32>(data.size());
                            const std::vector<u8> value = data;
                            const bool deferred =
                                rasterizer &&
                                rasterizer->DeferFenceSignal(
                                    reinterpret_cast<VAddr>(address),
                                    [address, data = std::move(data)] {
                                        WriteDeferredFence(address, data.data(), u32(data.size()));
                                    },
                                    false, value);
                            if (!deferred) {
                                if (rasterizer) {
                                    rasterizer->OnFence(
                                        Vulkan::Rasterizer::DrainSource::GfxWriteData);
                                }
                                Core::MemoryManager::NoteEmulatorWrite(
                                    reinterpret_cast<VAddr>(address), data_size, value.data());
                                std::memcpy(address, value.data(), data_size);
                                VideoCore::BumpUploadEpoch();
                            } else {
                                rasterizer->InlineDeferredWrite(reinterpret_cast<VAddr>(address),
                                                                value);
                            }
                        });
                    if (into_commands) {
                        SyncRecorder("write into commands");
                    }
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    Record([semaphore = *mem_semaphore] { semaphore.Signal(); }, "memory semaphore");
                } else {
                    GpuWaitDiagnostics diagnostics;
                    const auto wait_start = std::chrono::steady_clock::now();
                    if (!mem_semaphore->Signaled()) {
                        SyncRecorder("mem semaphore");
                    }
                    const bool waited = !mem_semaphore->Signaled();
                    while (!mem_semaphore->Signaled()) {
                        if (diagnostics.Ready()) {
                            LOG_WARNING(Render,
                                        "GPU semaphore wait stalled: address={:#x} value={}",
                                        mem_semaphore->Address<uintptr_t>(),
                                        *mem_semaphore->Address<u64*>());
                        }
                        YIELD_GFX();
                    }
                    if (waited) {
                        RecordFrontendWait(FrontendWait::GfxMemSemaphore,
                                           mem_semaphore->Address<uintptr_t>(), wait_start);
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const size_t rewind_offset = dcb.data() - submitted_dcb.data();
                ASSERT_MSG(live_dcb.size() == submitted_dcb.size() && dcb.size() >= 2,
                           "Invalid REWIND packet or original command-buffer size");
                GpuWaitDiagnostics diagnostics;
                const auto snapshot =
                    std::span{const_cast<u32*>(submitted_dcb.data()), submitted_dcb.size()};
                const auto rewind_start = std::chrono::steady_clock::now();
                const bool rewind_waited =
                    !RefreshRewindTailIfReady(snapshot, live_dcb, rewind_offset);
                while (rewind_waited &&
                       !RefreshRewindTailIfReady(snapshot, live_dcb, rewind_offset)) {
                    if (diagnostics.Ready()) {
                        LOG_WARNING(Render, "GPU REWIND stalled: snapshot={:#x} live={:#x}",
                                    reinterpret_cast<uintptr_t>(header),
                                    reinterpret_cast<uintptr_t>(live_dcb.data() + rewind_offset));
                    }
                    YIELD_GFX();
                }
                if (rewind_waited) {
                    RecordFrontendWait(FrontendWait::GfxRewind, reinterpret_cast<uintptr_t>(header),
                                       rewind_start);
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                const auto report_wait = [&] {
                    const bool memory =
                        wait_reg_mem->mem_space == PM4CmdWaitRegMem::MemSpace::Memory;
                    const u32 value =
                        memory ? *wait_reg_mem->Address() : regs.reg_array[wait_reg_mem->Reg()];
                    LOG_WARNING(Render,
                                "GPU WAIT_REG_MEM stalled: memory={} address={:#x} value={:#x} "
                                "reference={:#x} mask={:#x} function={}",
                                memory, reinterpret_cast<uintptr_t>(wait_addr), value,
                                wait_reg_mem->ref, wait_reg_mem->mask,
                                u32(wait_reg_mem->function.Value()));
                };
                const auto wait_start = std::chrono::steady_clock::now();
                // PERF-031: what is waited on may be written by recorded work (a fence, a copy, a
                // flip); let the recorder finish before waiting.
                if (!wait_reg_mem->Test(regs.reg_array)) {
                    // PERF-048: the label may come from another queue's packets not decoded
                    // yet. This queue stays at its wait while the others take a few turns;
                    // a write they queue moves the wait to the recorder.
                    bool yielded = false;
                    for (u32 turn = 0; draw_pipe && WaitTurnsEnabled() && turn < 4 &&
                                       !wait_reg_mem->Test(regs.reg_array) &&
                                       !PendingLabelSatisfies(*wait_reg_mem);
                         ++turn) {
                        yielded = true;
                        YIELD_GFX();
                    }
                    if (!wait_reg_mem->Test(regs.reg_array) &&
                        PendingLabelSatisfies(*wait_reg_mem)) {
                        // A queued fence writes the label: the recorder waits for it in order
                        // and decoding goes on.
                        ++waits_moved;
                        waits_moved_after_turns += yielded;
                        // DIAG-054: the commands decoded after the wait, as read now; checked
                        // again once the wait is really satisfied.
                        const u32* after = dcb.data() + header->type3.NumWords() + 1;
                        const size_t after_dwords = std::min<size_t>(
                            512, dcb.size() > header->type3.NumWords() + 1
                                     ? dcb.size() - header->type3.NumWords() - 1
                                     : 0);
                        const u64 after_hash = XXH3_64bits(after, after_dwords * sizeof(u32));
                        RecordSafe([this, wait = *wait_reg_mem, after, after_dwords,
                                    after_hash] {
                            RecorderWaitRegMem(wait);
                            ++diag_moved_waits_checked;
                            if (XXH3_64bits(after, after_dwords * sizeof(u32)) != after_hash) {
                                static std::atomic<u32> logged{};
                                ++diag_moved_waits_changed;
                                if (logged.fetch_add(1) < 20) {
                                    LOG_ERROR(Render,
                                              "DIAG-054: commands after a moved WAIT_REG_MEM "
                                              "(label {:#x}) changed between decoding and the "
                                              "label being written: {:#x}+{} dwords",
                                              reinterpret_cast<uintptr_t>(wait.Address<u32*>()),
                                              reinterpret_cast<uintptr_t>(after), after_dwords);
                                }
                            }
                        });
                        break;
                    }
                    if (!wait_reg_mem->Test(regs.reg_array)) {
                        SyncRecorder("wait reg mem");
                    }
                }
                const bool waited = !wait_reg_mem->Test(regs.reg_array);
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_tasks == mapped_queues[GfxQueueId].submits.size()) {
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); },
                                         report_wait);
                    if (waited) {
                        RecordFrontendWait(FrontendWait::GfxVoLabel,
                                           reinterpret_cast<uintptr_t>(wait_addr), wait_start);
                    }
                    break;
                }
                GpuWaitDiagnostics diagnostics;
                if (waited && SatisfiedByPendingFence(rasterizer, wait_reg_mem)) {
                    break;
                }
                if (rasterizer && !wait_reg_mem->Test(regs.reg_array)) {
                    rasterizer->FlushForDeferredFences();
                }
                // Other queues run while this one yields and may record more work.
                // DIAG-047: a queue's job writing this label meanwhile is counted.
                waited_label = reinterpret_cast<VAddr>(wait_reg_mem->Address<u32*>());
                while (!wait_reg_mem->Test(regs.reg_array) &&
                       (SyncRecorder("wait reg mem"),
                        !SatisfiedByPendingFence(rasterizer, wait_reg_mem))) {
                    if (diagnostics.Ready()) {
                        report_wait();
                        if (rasterizer) {
                            rasterizer->FlushForDeferredFences();
                        }
                    }
                    YIELD_GFX();
                }
                waited_label = 0;
                if (waited) {
                    RecordFrontendWait(vo_port->IsVoLabel(wait_addr) ? FrontendWait::GfxVoLabel
                                                                     : FrontendWait::GfxWaitRegMem,
                                       reinterpret_cast<uintptr_t>(wait_addr), wait_start);
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                RecordCmdBuffer(indirect_buffer->Address<const u32>(),
                                u64(indirect_buffer->ib_size) * sizeof(u32), true);
                lookahead_outer.push_back(lookahead_dcb);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                lookahead_outer.pop_back();
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                GpuWaitDiagnostics diagnostics;
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    if (diagnostics.Ready()) {
                        LOG_WARNING(Render, "GPU DE wait stalled: CE={} DE={}", cblock.ce_count,
                                    cblock.de_count);
                    }
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                // PERF-031: the condition may be written by recorded work.
                SyncRecorder("cond exec");
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            // A fence in this packet or a nested IB can let the guest recycle the buffer.
            // Its header may have changed while we yielded; advance using the captured count.
            dcb = NextPacket(dcb, count + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }
    std::erase_if(live_cmd_buffers,
                  [&](const auto& span) { return span.data() == live_dcb.data(); });

    FIBER_EXIT;
}

bool Liverpool::ScanPackets(std::span<const u32> dcb,
                            const std::function<bool(const Regs&)>& on_draw, u32 depth,
                            u32& draws_left, u32& builds_left, const u32** stopped_at) {
    auto& shadow = *scan_regs;
    const auto set_regs = [&](u32 first, const u32* values, u32 num) {
        if (first + num <= shadow.reg_array.size()) {
            std::memcpy(&shadow.reg_array[first], values, num * sizeof(u32));
        }
    };
    while (!dcb.empty()) {
        if (draws_left == 0 || builds_left == 0) {
            if (stopped_at) {
                *stopped_at = dcb.data();
            }
            return false;
        }
        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        if (type == 0) {
            const auto write = DecodeType0RegisterWrite(dcb, shadow.reg_array.size());
            if (!write) {
                return true;
            }
            set_regs(write->first_register, write->values.data(),
                     static_cast<u32>(write->values.size()));
            dcb = dcb.subspan(std::min<size_t>(dcb.size(), write->values.size() + 1));
            continue;
        }
        if (type == 2) {
            dcb = dcb.subspan(1);
            continue;
        }
        if (type != 3) {
            // Not a command stream (or past its end); stop reading ahead.
            return true;
        }
        const u32 count = header->type3.NumWords();
        if (count + 1 > dcb.size()) {
            return true;
        }
        const auto* payload = reinterpret_cast<const u32*>(header + 2);
        const u32 num_values = count > 0 ? count - 1 : 0;
        switch (header->type3.opcode) {
        case PM4ItOpcode::ClearState:
            shadow.SetDefaults();
            break;
        case PM4ItOpcode::SetConfigReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            set_regs(Regs::ConfigRegWordOffset + set_data->reg_offset, payload, num_values);
            break;
        }
        case PM4ItOpcode::SetContextReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            set_regs(Regs::ContextRegWordOffset + set_data->reg_offset, payload, num_values);
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            // Compute shader registers are kept apart from the graphics state.
            if (set_data->reg_offset < 0x200 ||
                set_data->reg_offset > 0x200 + sizeof(ComputeProgram) / 4) {
                set_regs(Regs::ShRegWordOffset + set_data->reg_offset, payload, num_values);
            }
            break;
        }
        case PM4ItOpcode::SetUconfigReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            set_regs(Regs::UconfigRegWordOffset + set_data->reg_offset, payload, num_values);
            break;
        }
        case PM4ItOpcode::IndexType:
            shadow.index_buffer_type.raw =
                reinterpret_cast<const PM4CmdDrawIndexType*>(header)->raw;
            break;
        case PM4ItOpcode::NumInstances:
            shadow.num_instances.num_instances =
                reinterpret_cast<const PM4CmdDrawNumInstances*>(header)->num_instances;
            break;
        case PM4ItOpcode::DrawIndex2:
        case PM4ItOpcode::DrawIndexOffset2:
        case PM4ItOpcode::DrawIndexAuto:
        case PM4ItOpcode::DrawIndirect:
        case PM4ItOpcode::DrawIndirectMulti:
        case PM4ItOpcode::DrawIndexIndirect:
        case PM4ItOpcode::DrawIndexIndirectMulti:
        case PM4ItOpcode::DrawIndexIndirectCountMulti:
            if (!Common::PerfFeatureEnabled(22) || scanned_packets.insert(header).second) {
                --draws_left;
                if (on_draw(shadow)) {
                    --builds_left;
                }
            }
            break;
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            const auto* ib = indirect_buffer->Address<const u32>();
            const u32 ib_size = indirect_buffer->ib_size;
            const bool new_ib =
                !Common::PerfFeatureEnabled(22) || scanned_packets.insert(ib).second;
            if (new_ib && depth < 4 && ib && ib_size > 0 && ib_size < (1u << 22)) {
                ScanPackets({ib, ib_size}, on_draw, depth + 1, draws_left, builds_left, nullptr);
            }
            break;
        }
        default:
            break;
        }
        dcb = dcb.subspan(count + 1);
    }
    return true;
}

bool Liverpool::ScanAheadForPipelines(const std::function<bool(const Regs&)>& on_draw,
                                      u32 max_draws, u32 max_builds) {
    if (lookahead_dcb.empty()) {
        return false;
    }
    const u32* buffer_end = lookahead_dcb.data() + lookahead_dcb.size();
    std::span<const u32> start = lookahead_dcb;
    if (scan_regs && scan_buffer_end == buffer_end && scan_resume &&
        scan_resume >= lookahead_dcb.data() && scan_resume <= buffer_end) {
        // An earlier scan of this command buffer got past this point; continue it with the
        // registers it had there.
        if (scan_finished) {
            return false;
        }
        start = {scan_resume, buffer_end};
    } else {
        if (!scan_regs) {
            scan_regs = std::make_unique<Regs>();
            *scan_regs = regs;
        } else if (Common::PerfFeatureEnabled(23)) {
            // PERF-023: copy only the register blocks pipeline state comes from (config,
            // graphics shader, context and uconfig), about 40 KB instead of all 208 KB.
            const auto copy = [&](u32 first, u32 num) {
                std::memcpy(&scan_regs->reg_array[first], &regs.reg_array[first],
                            num * sizeof(u32));
            };
            copy(Regs::ConfigRegWordOffset, Regs::ShRegWordOffset - Regs::ConfigRegWordOffset);
            copy(Regs::ShRegWordOffset, 0x200);
            copy(Regs::ContextRegWordOffset, 0x400);
            copy(Regs::UconfigRegWordOffset, Regs::NumRegs - Regs::UconfigRegWordOffset);
        } else {
            *scan_regs = regs;
        }
        scan_buffer_end = buffer_end;
    }
    // Enough to keep every build worker busy, without reading a whole frame ahead each time.
    u32 draws_left = max_draws;
    u32 builds_left = max_builds;
    const u32* stopped_at = nullptr;
    scan_finished = ScanPackets(start, on_draw, 0, draws_left, builds_left, &stopped_at);
    // The command buffers this one was called from continue after it.
    for (auto it = lookahead_outer.rbegin(); scan_finished && it != lookahead_outer.rend(); ++it) {
        if (!it->empty()) {
            ScanPackets(*it, on_draw, 0, draws_left, builds_left, nullptr);
        }
    }
    scan_resume = scan_finished ? buffer_end : stopped_at;
    return !scan_finished;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        ++Common::GetWorkCounters().pm4_packets;

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (SkipRegisterSpaceDmaData(dma_data, "compute")) {
                break;
            }
            // PERF-031: copies run on the recorder thread in order with the dispatches around them.
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                RecordDma(
                    [this, dst = dma_data->dst_addr_lo, bytes = dma_data->NumBytes(),
                     value = dma_data->data] { rasterizer->FillBuffer(dst, bytes, value, true); });
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                RecordDma([this, dst = dma_data->dst_addr_lo, src = dma_data->SrcAddress<VAddr>(),
                        bytes = dma_data->NumBytes()] {
                    rasterizer->CopyBuffer(dst, src, bytes, true, false);
                });
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                RecordDma(
                    [this, dst = dma_data->DstAddress<VAddr>(), bytes = dma_data->NumBytes(),
                     value = dma_data->data] { rasterizer->FillBuffer(dst, bytes, value, false); });
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                RecordDma([this, dst = dma_data->DstAddress<VAddr>(), src = dma_data->src_addr_lo,
                        bytes = dma_data->NumBytes()] {
                    rasterizer->CopyBuffer(dst, src, bytes, false, true);
                });
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    RecordDma([this, dst_addr, src_addr, num_bytes] {
                        rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                    });
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            const auto wait_start = std::chrono::steady_clock::now();
            const bool waited = !rewind->Valid();
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            if (waited) {
                RecordFrontendWait(FrontendWait::AscRewind, reinterpret_cast<uintptr_t>(header),
                                   wait_start);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
                MarkRegs(Regs::ShRegWordOffset + set_data->reg_offset,
                         static_cast<u32>(set_size / sizeof(u32)));
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                RecordDraw([this, args = it->indirect_addr,
                            size] { rasterizer->DispatchIndirect(args, 0, size); },
                           true);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            RecordDraw(
                [this, vqid, cmd_address] {
                    rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect",
                                            fmt::make_format_args(vqid, cmd_address),
                                            [&] { rasterizer->DispatchDirect(); });
                },
                true, true);
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            RecordDraw(
                [this, vqid, cmd_address, ib_address, size] {
                    rasterizer->ScopeMarker(
                        "asc[{}]:{}:DispatchIndirect", fmt::make_format_args(vqid, cmd_address),
                        [&] { rasterizer->DispatchIndirect(ib_address, 0, size); });
                },
                true);
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                std::vector<u8> data(data_size);
                std::memcpy(data.data(), write_data->data, data_size);
                std::vector<u8> label = data;
                RecordLabelWrite(
                    write_data->Address<VAddr>(), std::move(label),
                    [this, address = write_data->Address<VAddr>(), data = std::move(data)] {
                        if (rasterizer) {
                            rasterizer->OnFence(Vulkan::Rasterizer::DrainSource::AscWriteData);
                        }
                        Core::MemoryManager::NoteEmulatorWrite(address, data.size(), data.data());
                        std::memcpy(reinterpret_cast<void*>(address), data.data(), data.size());
                        VideoCore::BumpUploadEpoch();
                    });
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                Record([semaphore = *mem_semaphore] { semaphore.Signal(); }, "memory semaphore");
            } else {
                const auto wait_start = std::chrono::steady_clock::now();
                if (!mem_semaphore->Signaled()) {
                    SyncRecorder("mem semaphore");
                }
                const bool waited = !mem_semaphore->Signaled();
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                if (waited) {
                    RecordFrontendWait(FrontendWait::AscMemSemaphore,
                                       mem_semaphore->Address<uintptr_t>(), wait_start);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            const auto wait_start = std::chrono::steady_clock::now();
            // PERF-048: as for graphics waits, the other queues decode first.
            bool yielded = false;
            for (u32 turn = 0; draw_pipe && WaitTurnsEnabled() && turn < 4 &&
                               !wait_reg_mem->Test(regs.reg_array) &&
                               !PendingLabelSatisfies(*wait_reg_mem);
                 ++turn) {
                yielded = true;
                YIELD_ASC(vqid);
            }
            if (!wait_reg_mem->Test(regs.reg_array) && PendingLabelSatisfies(*wait_reg_mem)) {
                // PERF-031: a queued fence writes the label; the recorder waits for it in order.
                ++waits_moved;
                waits_moved_after_turns += yielded;
                RecordSafe([this, wait = *wait_reg_mem] { RecorderWaitRegMem(wait); });
                break;
            }
            // PERF-031: what is waited on may be written by recorded work; the pending-fence
            // check reads the rasterizer, which only the drained recorder lets the command use.
            const auto unmet = [&] {
                if (wait_reg_mem->Test(regs.reg_array)) {
                    return false;
                }
                SyncRecorder("wait reg mem");
                return !wait_reg_mem->Test(regs.reg_array) &&
                       !SatisfiedByPendingFence(rasterizer, wait_reg_mem);
            };
            const bool waited = unmet();
            while (waited && unmet()) {
                YIELD_ASC(vqid);
            }
            if (waited) {
                RecordFrontendWait(FrontendWait::AscWaitRegMem,
                                   reinterpret_cast<uintptr_t>(wait_reg_mem->Address<u64*>()),
                                   wait_start);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            // Signaling the fence can allow the guest to reuse the containing command buffer.
            const auto release = *reinterpret_cast<const PM4CmdReleaseMem*>(header);
            // PERF-010: when GPU-written pages the CPU read before are being read back, write
            // this fence once the GPU has executed the work and the data is in guest memory,
            // instead of letting the CPU access drain the GPU.
            const auto pipe_id = queue.pipe_id;
            // PERF-031: signaled in order behind the recorded work.
            RecordLabelWrite(
                release.data_sel != DataSelect::GdsMemStore ? release.Address<VAddr>() : 0,
                FenceValueBytes(release.data_sel.Value(), release.DataDWord(), release.DataQWord()),
                [this, release, pipe_id] {
                    const auto* release_mem = &release;
                    const bool deferred =
                        rasterizer && release.data_sel != DataSelect::GdsMemStore &&
                        rasterizer->DeferFenceSignal(
                            release.Address<VAddr>(),
                            [release, pipe_id] {
                                u64 value{};
                                u32 num_bytes = sizeof(u64);
                                switch (release.data_sel.Value()) {
                                case DataSelect::None:
                                    num_bytes = 0;
                                    break;
                                case DataSelect::Data32Low:
                                    value = release.DataDWord();
                                    num_bytes = sizeof(u32);
                                    break;
                                case DataSelect::Data64:
                                    value = release.DataQWord();
                                    break;
                                case DataSelect::GpuClock64:
                                    value = GetGpuClock64();
                                    break;
                                case DataSelect::PerfCounter:
                                    value = GetGpuPerfCounter();
                                    break;
                                default:
                                    UNREACHABLE();
                                }
                                if (num_bytes != 0) {
                                    WriteDeferredFence(release.Address<void*>(), &value, num_bytes);
                                }
                                if (release.int_sel != InterruptSelect::None) {
                                    Platform::IrqC::Instance()->Signal(
                                        static_cast<Platform::InterruptId>(pipe_id));
                                }
                            },
                            true,
                            FenceValueBytes(release.data_sel.Value(), release.DataDWord(),
                                            release.DataQWord()));
                    if (deferred) {
                        return;
                    }
                    if (rasterizer) {
                        rasterizer->OnFence(Vulkan::Rasterizer::DrainSource::AscReleaseMem);
                    }
                    // PERF-051: the value goes to guest memory through the backing, as graphics
                    // EOP/EOS fences do, instead of a store that faults on a tracked page and
                    // makes the recorder wait for the whole GPU. -DisablePerf 63 stores directly.
                    static const bool write_backing = Common::PerfFeatureEnabled(63);
                    release_mem->SignalFence(
                        [pipe_id] {
                            Platform::IrqC::Instance()->Signal(
                                static_cast<Platform::InterruptId>(pipe_id));
                        },
                        [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                            rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false,
                                                   true);
                        },
                        [](void* address, u64 data, u32 num_bytes) {
                            auto* memory = Core::Memory::Instance();
                            if (!write_backing ||
                                !memory->TryWriteBacking(address, &data, num_bytes)) {
                                std::memcpy(address, &data, num_bytes);
                            }
                        });
                });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    const auto original_dcb = dcb;

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    // PERF-012: memory written before this submission must be uploaded again.
    VideoCore::BumpUploadEpoch();
    {
        // DIAG-051
        const auto now = std::chrono::steady_clock::now();
        std::scoped_lock lk{submit_timing_mutex};
        gfx_submit_times.push_back(now);
        if constexpr (Common::Nvtx::Enabled) {
            const u64 n = ++gfx_submit_number;
            gfx_submit_ranges.push_back(
                {n, Common::Nvtx::Start(fmt::format("gfx submit {}: decode", n)),
                 Common::Nvtx::Start(fmt::format("gfx submit {}: until recorded", n))});
        }
        if (const s64 eop = g_last_eop_ns.load(); eop != 0) {
            const s64 since = NowNs() - eop;
            if (since >= 0 && since < 100'000'000) {
                submit_stats.eop_to_submit_us += static_cast<u64>(since / 1000);
                ++submit_stats.eop_to_submit_count;
            }
        }
        if (last_gfx_submit != std::chrono::steady_clock::time_point{}) {
            submit_stats.gap_us += static_cast<u64>(
                std::chrono::duration_cast<std::chrono::microseconds>(now - last_gfx_submit)
                    .count());
            ++submit_stats.gaps;
        }
        last_gfx_submit = now;
    }
    auto task = ProcessGraphics(dcb, ccb, original_dcb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    ++num_tasks;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    VideoCore::BumpUploadEpoch();
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    ++num_tasks;
    submit_cv.notify_one();
}

} // namespace AmdGpu
