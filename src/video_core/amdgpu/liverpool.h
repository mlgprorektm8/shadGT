// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <optional>

#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/recycling_pool.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/read_ahead_states.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_delta.h"

namespace Vulkan {
class Rasterizer;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

class DrawPipe;
struct PM4CmdWaitRegMem;

struct Liverpool {
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this] { return num_submits == 0; });
    }

    /// PERF-031: the registers of the draw or dispatch being recorded. On the recorder thread
    /// of the draw pipe these are its copy, up to date with that draw; otherwise the live ones.
    const Regs& DrawRegs() const {
        return recorder_state ? recorder_state->regs : regs;
    }
    const ComputeProgram& DrawCsRegs() {
        return recorder_state ? recorder_state->cs : GetCsRegs();
    }
    const CbDbExtent& DrawCbExtent(u32 cb) const {
        return recorder_state ? recorder_state->cb_extent[cb] : last_cb_extent[cb];
    }
    const CbDbExtent& DrawDbExtent() const {
        return recorder_state ? recorder_state->db_extent : last_db_extent;
    }

    /// PERF-031: whether draws are recorded on a second thread (SHADGT_DRAW_PIPE=1).
    bool Pipelined() const {
        return pipelined.load(std::memory_order_acquire);
    }
    /// The command processor thread, or the draw pipe's recorder thread.
    bool IsGpuThread(std::thread::id id) const;
#ifdef __linux__
    bool IsGpuThreadTid(u32 tid) const;
#endif
    /// A fault on the command processor thread: its handling uses the caches, so the recorder
    /// finishes first.
    void OnGpuThreadFault();

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    /// PERF-050: a CPU fault's flush from a game thread; returns once it ran. With the draw pipe
    /// it runs as soon as the caches are free instead of behind all decoded work
    /// (-DisablePerf 62 queues it like any command). DIAG-050 times the wait.
    void SendFaultCommand(Common::UniqueFunction<void>&& func);

    /// PERF-054: a fault on the command thread is flushed like a game thread's (urgent work on
    /// the recorder) instead of draining every decoded job first, unless the command thread owns
    /// the caches already. -DisablePerf 65 drains.
    bool CommandThreadFaultIsUrgent();

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func] {
                    func();
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            sem.acquire();
        } else {
            std::scoped_lock lk{submit_mutex};
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    /// Diagnostic (DIAG-009): most recent CE constant dump covering a range, if any.
    struct ConstDumpRecord {
        VAddr address{};
        u32 size{};
        u32 ce_count{};
        u32 de_count{};
        u64 sequence{};
    };
    std::optional<ConstDumpRecord> FindRecentConstDump(VAddr address, u64 size) const {
        std::optional<ConstDumpRecord> found;
        for (const auto& dump : recent_const_dumps) {
            if (dump.size != 0 && address >= dump.address &&
                address + size <= dump.address + dump.size &&
                (!found || dump.sequence > found->sequence)) {
                found = dump;
            }
        }
        return found;
    }
    u64 ConstDumpSequence() const {
        return const_dump_sequence;
    }
    /// Diagnostic (DIAG-010): most recent graphics command buffer covering a range.
    std::optional<ConstDumpRecord> FindRecentCmdBuffer(VAddr address, u64 size) const {
        std::optional<ConstDumpRecord> found;
        for (const auto& range : recent_cmd_buffers) {
            if (range.size != 0 && address >= range.address &&
                address + size <= range.address + range.size &&
                (!found || range.sequence > found->sequence)) {
                found = range;
            }
        }
        return found;
    }
    /// DIAG-053: a recent graphics command buffer overlapping a range (racy; diagnostics only).
    std::optional<ConstDumpRecord> OverlappingRecentCmdBuffer(VAddr address, u64 size) const {
        std::optional<ConstDumpRecord> found;
        for (const auto& range : recent_cmd_buffers) {
            if (range.size != 0 && address < range.address + range.size &&
                range.address < address + size && (!found || range.sequence > found->sequence)) {
                found = range;
            }
        }
        return found;
    }
    u64 CmdBufferSequence() const {
        return cmd_buffer_sequence;
    }
    void RecordCmdBuffer(const void* data, u64 size_bytes, bool indirect) {
        recent_cmd_buffers[cmd_buffer_sequence % recent_cmd_buffers.size()] = {
            .address = reinterpret_cast<VAddr>(data),
            .size = static_cast<u32>(size_bytes),
            .ce_count = indirect,
            .sequence = ++cmd_buffer_sequence,
        };
    }
    std::pair<u32, u32> CeDeCounters() const {
        return {cblock.ce_count, cblock.de_count};
    }

    /// Diagnostic: submissions queued but not yet fully processed.
    u32 PendingSubmits() const {
        return num_submits;
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    /// PERF-019: reads the graphics commands after the current packet without executing them,
    /// tracking register writes in a copy of the registers, and calls on_draw with that copy at
    /// each draw. on_draw returns true when it started a pipeline build. A later call for the
    /// same command buffer continues where the previous one stopped. Returns true when it
    /// stopped early (after `max_draws` new draws or `max_builds` builds).
    bool ScanAheadForPipelines(const std::function<bool(const Regs&)>& on_draw,
                               u32 max_draws = 2048, u32 max_builds = 48);

    /// PERF-034: upcoming draw states the command thread read ahead for the pipeline cache;
    /// null without the draw pipe.
    ReadAheadStates* PipeReadAheadStates() const {
        return read_ahead_states.get();
    }
    /// PERF-020: called on the command thread when a graphics command buffer starts, with the
    /// read-ahead positioned at its first packet.
    std::function<void()> on_command_buffer_start;

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{64};

    std::thread::id GetGpuCommandProcessorThread() {
        return gpu_id;
    }

#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId() {
        return gpu_tid;
    }
#endif

private:
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    // PERF-019 read-ahead state: the commands after the packet being processed (and those of
    // the command buffers that called into it), and the scan position with its registers.
    std::span<const u32> lookahead_dcb{};
    std::vector<std::span<const u32>> lookahead_outer;
    std::unique_ptr<Regs> scan_regs;
    const u32* scan_buffer_end{};
    const u32* scan_resume{};
    bool scan_finished{};
    // PERF-022: draw packets and nested command buffers the read-ahead has already read in the
    // current submission; each is read once however often a read-ahead passes over it.
    std::unordered_set<const void*> scanned_packets;
    bool ScanPackets(std::span<const u32> dcb, const std::function<bool(const Regs&)>& on_draw,
                     u32 depth, u32& draws_left, u32& builds_left, const u32** stopped_at);

    // PERF-031: the draw pipe. The command thread decodes and keeps `regs`; draws, dispatches
    // and everything that touches the caches or writes guest memory run on the recorder thread
    // in submission order, with the registers brought up to date from regs_dirty.
    struct RecorderState {
        Regs regs{};
        ComputeProgram cs{};
        std::array<CbDbExtent, NUM_COLOR_BUFFERS> cb_extent{};
        CbDbExtent db_extent{};
    };
    static inline thread_local RecorderState* recorder_state = nullptr;
    std::unique_ptr<RecorderState> recorder;
    std::unique_ptr<DrawPipe> draw_pipe;
    std::atomic<bool> pipelined{};
    RegsDelta<Regs::NumRegs> regs_dirty;
    u32 verify_interval{};
    u64 verify_count{};
#ifdef __linux__
    std::atomic<u32> recorder_tid{};
#endif
    // Command buffers being decoded, so writes into them are not reordered after their decoding.
    std::vector<std::span<const u32>> live_cmd_buffers;
    std::unordered_map<std::string_view, std::pair<u64, double>> sync_reasons;
    std::chrono::steady_clock::time_point pipe_report_start{};

    // PERF-031: labels that fence jobs queued to the recorder will write, so a wait on one does
    // not drain the pipe: the recorder waits for it in order instead.
    struct PendingLabel {
        std::vector<u8> bytes;
        u64 job;
    };
    std::mutex pending_labels_mutex;
    std::unordered_map<VAddr, PendingLabel> pending_labels;
    u64 label_jobs{};
    u64 waits_moved{};
    // PERF-035: commands from other threads run on the recorder instead of after a drain.
    u64 commands_recorded{};
    // PERF-048: waits on labels another queue writes, found after letting it decode first.
    u64 waits_moved_after_turns{};
    std::atomic<u64> diag_moved_waits_checked{}; // DIAG-054
    std::atomic<u64> diag_moved_waits_changed{};
    Common::Recycler<std::vector<u32>> delta_recycler; // PERF-056
    // DIAG-051: graphics submissions through the pipeline: game submit, decoded, recorded,
    // GPU done (microseconds after the submit), and the gap between game submits.
    struct SubmitTiming {
        std::chrono::steady_clock::time_point submitted;
        std::chrono::steady_clock::time_point decoded;
    };
    std::deque<std::chrono::steady_clock::time_point> gfx_submit_times;
    std::chrono::steady_clock::time_point last_gfx_submit{};
    std::mutex submit_timing_mutex;
    struct SubmitStats {
        u64 count{};
        u64 decode_us{};
        u64 record_us{};
        u64 gpu_us{};
        u64 gpu_count{};
        u64 gap_us{};
        u64 gaps{};
        u64 max_gpu_us{};
    } submit_stats;
    // DIAG-050: CPU fault flushes and how long the faulting threads waited for them.
    std::atomic<u64> fault_flushes{};
    std::atomic<u64> fault_flush_wait_us{};
    std::atomic<u64> fault_flush_max_us{};
    // DIAG-047: the label a graphics wait drains for, and how often a queue's job writes it then.
    VAddr waited_label{};
    u64 labels_written_while_waiting{};
    static bool WaitTurnsEnabled();
    /// Records a job that writes `value` at `address` (a fence or WRITE_DATA).
    void RecordLabelWrite(VAddr address, std::vector<u8> value,
                          Common::UniqueFunction<void>&& work);
    /// Whether a queued job will write a value that satisfies the memory wait.
    bool PendingLabelSatisfies(const PM4CmdWaitRegMem& wait);
    /// Runs on the recorder, in order: waits as the command thread would have.
    void RecorderWaitRegMem(const PM4CmdWaitRegMem& wait);

    // PERF-034: the pipeline read-ahead with the draw pipe, on the command thread.
    std::unique_ptr<ReadAheadStates> read_ahead_states;
    bool pipeline_regs_written{};
    bool read_ahead_stopped_early{};
    struct {
        u64 scans;
        double ms;
    } pipe_read_ahead_stats{};
    /// Queues the state of the graphics draw being decoded when its pipeline registers
    /// changed, and reads ahead while new pipelines keep appearing.
    void OfferDrawState();
    /// Reads the commands ahead of the command thread and queues the new draw states found.
    void PipeReadAhead();

    void StartDrawPipe();
    void ReportDrawPipe();
    void VerifyRecorderRegs(const Regs& expected);
    /// Runs work on the recorder thread behind everything queued before it (inline without one).
    void Record(Common::UniqueFunction<void>&& work);
    /// Records a draw or dispatch together with the registers written since the previous one.
    /// `direct_draw`: a draw packet with its counts in the packet (PERF-047 selects its pipeline
    /// here, while decoding).
    void RecordDraw(Common::UniqueFunction<void>&& draw, bool compute, bool direct_draw = false);
    /// Waits for the recorder thread to finish, before the command thread uses the rasterizer.
    void SyncRecorder(std::string_view reason);
    /// Whether [address, address + size) overlaps a command buffer being decoded.
    bool WritesLiveCommands(VAddr address, u64 size) const;
    void MarkRegs(u32 first, u32 count) {
        regs_dirty.Mark(first, count);
        if (read_ahead_states && ReadAheadStates::AffectsPipeline(first, count)) {
            pipeline_regs_written = true; // PERF-034
        }
    }
    template <typename T>
    void MarkReg(const T& field) {
        const auto offset = static_cast<u32>(reinterpret_cast<const u8*>(&field) -
                                             reinterpret_cast<const u8*>(regs.reg_array.data()));
        MarkRegs(offset / sizeof(u32), static_cast<u32>((sizeof(T) + 3) / sizeof(u32)));
    }

    std::array<ConstDumpRecord, 256> recent_const_dumps{};
    u64 const_dump_sequence{};
    std::array<ConstDumpRecord, 256> recent_cmd_buffers{};
    u64 cmd_buffer_sequence{};

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                         std::span<const u32> original_dcb = {});
    Task ProcessCeUpdate(std::span<const u32> ccb);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);

    void ProcessCommands();
    void Process(std::stop_token stoken);

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    u32 num_mapped_queues{1u}; // GFX is always available

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    const bool guest_markers_enabled;
    std::jthread process_thread{};
    // Submissions the guest made that are not finished: the recorder thread finishes them when
    // the draw pipe runs. num_tasks counts the ones the command thread has not decoded yet.
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_tasks{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
#ifdef __linux__
    u32 gpu_tid;
#endif
    s32 curr_qid{-1};
};

} // namespace AmdGpu
