// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/preprocessor/stringize.hpp>
#include <fmt/ranges.h>

#include "common/assert.h"
#include "common/debug.h"
#include "common/guest_clock.h"
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
    if (const char* env = std::getenv("SHADGT_DRAW_PIPE_VERIFY")) {
        verify_interval = static_cast<u32>(std::strtoul(env, nullptr, 10));
    }
    draw_pipe = std::make_unique<DrawPipe>([this] {
        Common::SetCurrentThreadName("shadGT:GpuRecorder");
        // FIX-044: shader and pipeline compiles on this thread hold the guest clocks too.
        Common::GuestClock::MarkGpuCommandThread();
        recorder_state = recorder.get();
#ifdef __linux__
        recorder_tid = gettid();
#endif
    });
    pipe_report_start = std::chrono::steady_clock::now();
    pipelined.store(true, std::memory_order_release);
    LOG_WARNING(
        Render, "PERF-031: draws are recorded on a second thread (draw pipe){}",
        verify_interval ? fmt::format(", registers verified every {} draws", verify_interval) : "");
}

void Liverpool::Record(Common::UniqueFunction<void>&& work) {
    if (draw_pipe) {
        draw_pipe->Push(std::move(work));
    } else {
        work();
    }
}

void Liverpool::RecordDraw(Common::UniqueFunction<void>&& draw, bool compute) {
    if (!draw_pipe) {
        draw();
        return;
    }
    std::vector<u32> delta;
    regs_dirty.Collect(std::span<const u32, Regs::NumRegs>{regs.reg_array}, delta);
    std::optional<ComputeProgram> cs;
    if (compute) {
        cs = GetCsRegs();
    }
    std::unique_ptr<Regs> expected;
    if (verify_interval != 0 && ++verify_count % verify_interval == 0) {
        expected = std::make_unique<Regs>(regs);
    }
    draw_pipe->Push([this, delta = std::move(delta), cs, cb_extent = last_cb_extent,
                     db_extent = last_db_extent, expected = std::move(expected),
                     draw = std::move(draw)] {
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
        draw();
    });
    ReportDrawPipe();
}

void Liverpool::SyncRecorder(std::string_view reason) {
    if (!draw_pipe) {
        return;
    }
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
    {
        std::scoped_lock lk{pending_labels_mutex};
        pending_labels[address] = PendingLabel{std::move(value), job};
    }
    draw_pipe->Push([this, address, job, work = std::move(work)] {
        work();
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
    // The job that writes the label ran before this one. If its fence was deferred, the value
    // reaches memory when the GPU finishes, written by the scheduler's completion thread.
    const auto& state_regs = recorder->regs.reg_array;
    if (wait.Test(state_regs) || SatisfiedByPendingFence(rasterizer, &wait)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    rasterizer->FlushForDeferredFences();
    while (!wait.Test(state_regs) && !SatisfiedByPendingFence(rasterizer, &wait)) {
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
    LOG_WARNING(Render,
                "PERF-031 draw pipe in {:.1f} s: {} jobs, recorder busy {:.0f}%, max queued {}, "
                "{} full-queue waits; {} waits moved to the recorder; drains {} ({} waited, "
                "{:.1f} ms):{}",
                seconds, stats.pushed, stats.recorder_busy_us / (seconds * 1e4), stats.max_queued,
                stats.push_waits, waits_moved, stats.drains, stats.drains_that_waited,
                stats.drain_wait_us / 1000.0, reasons);
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

void Liverpool::ProcessCommands() {
    if (!num_commands) {
        return;
    }
    // PERF-031: commands from other threads (flips, cache flushes, unmaps) use the caches.
    SyncRecorder("commands");
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadGT:GpuCommandProcessor");
    Common::GuestClock::MarkGpuCommandThread(); // FIX-044
    gpu_id = std::this_thread::get_id();
#ifdef __linux__
    gpu_tid = gettid();
#endif

    while (!stoken.stop_requested()) {
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
                // PERF-031: the submission is finished once its recorded work is.
                Record([this] {
                    --num_submits;
                    std::scoped_lock lock2{submit_mutex};
                    submit_cv.notify_all();
                });
            }
        }

        // PERF-031: the GPU is idle once everything decoded so far is recorded.
        if (submit_done) {
            submit_done = false;
            Record([this] {
                VideoCore::EndCapture();
                if (rasterizer) {
                    rasterizer->OnSubmit();
                    rasterizer->Flush();
                }
            });
        }
        Record([] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle); });
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
    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

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
                    Record(
                        [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip); });
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        std::string label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                          marker_sz};
                        Record([this, label = std::move(label)] {
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
                        Record([this, label = std::move(label), color] {
                            rasterizer->ScopedMarkerInsertColor(label, color, true);
                        });
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        Record([this] { rasterizer->ScopeMarkerEnd(true); });
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
                    false);
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
                    false);
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
                    false);
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
                    true);
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
                        Record([results, counter, pairs = num_counter_pairs] {
                            u64* result = results;
                            for (s32 i = 0; i < s32(pairs); ++i, result += 2) {
                                *result = counter | OcclusionCounterValidMask;
                            }
                        });
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto event = *reinterpret_cast<const PM4CmdEventWriteEos*>(header);
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
                Record([this, dma] {
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
                    Record([semaphore = *mem_semaphore] { semaphore.Signal(); });
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
                    if (PendingLabelSatisfies(*wait_reg_mem)) {
                        // A queued fence writes the label: the recorder waits for it in order
                        // and decoding goes on.
                        ++waits_moved;
                        Record([this, wait = *wait_reg_mem] { RecorderWaitRegMem(wait); });
                        break;
                    }
                    SyncRecorder("wait reg mem");
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

void Liverpool::ScanAheadForPipelines(const std::function<bool(const Regs&)>& on_draw) {
    if (lookahead_dcb.empty()) {
        return;
    }
    const u32* buffer_end = lookahead_dcb.data() + lookahead_dcb.size();
    std::span<const u32> start = lookahead_dcb;
    if (scan_regs && scan_buffer_end == buffer_end && scan_resume &&
        scan_resume >= lookahead_dcb.data() && scan_resume <= buffer_end) {
        // An earlier scan of this command buffer got past this point; continue it with the
        // registers it had there.
        if (scan_finished) {
            return;
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
    u32 draws_left = 2048;
    u32 builds_left = 48;
    const u32* stopped_at = nullptr;
    scan_finished = ScanPackets(start, on_draw, 0, draws_left, builds_left, &stopped_at);
    // The command buffers this one was called from continue after it.
    for (auto it = lookahead_outer.rbegin(); scan_finished && it != lookahead_outer.rend(); ++it) {
        if (!it->empty()) {
            ScanPackets(*it, on_draw, 0, draws_left, builds_left, nullptr);
        }
    }
    scan_resume = scan_finished ? buffer_end : stopped_at;
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
                Record(
                    [this, dst = dma_data->dst_addr_lo, bytes = dma_data->NumBytes(),
                     value = dma_data->data] { rasterizer->FillBuffer(dst, bytes, value, true); });
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                Record([this, dst = dma_data->dst_addr_lo, src = dma_data->SrcAddress<VAddr>(),
                        bytes = dma_data->NumBytes()] {
                    rasterizer->CopyBuffer(dst, src, bytes, true, false);
                });
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                Record(
                    [this, dst = dma_data->DstAddress<VAddr>(), bytes = dma_data->NumBytes(),
                     value = dma_data->data] { rasterizer->FillBuffer(dst, bytes, value, false); });
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                Record([this, dst = dma_data->DstAddress<VAddr>(), src = dma_data->src_addr_lo,
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
                    Record([this, dst_addr, src_addr, num_bytes] {
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
                true);
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
                Record([semaphore = *mem_semaphore] { semaphore.Signal(); });
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
            if (!wait_reg_mem->Test(regs.reg_array) && PendingLabelSatisfies(*wait_reg_mem)) {
                // PERF-031: a queued fence writes the label; the recorder waits for it in order.
                ++waits_moved;
                Record([this, wait = *wait_reg_mem] { RecorderWaitRegMem(wait); });
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
                    release_mem->SignalFence(
                        [pipe_id] {
                            Platform::IrqC::Instance()->Signal(
                                static_cast<Platform::InterruptId>(pipe_id));
                        },
                        [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                            rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false,
                                                   true);
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
