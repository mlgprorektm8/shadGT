// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <windows.h>
#include <fmt/format.h>
#include <xxhash.h>

#include "common/logging/log.h"
#include "ryty_gpu/native/job_walker.h"
#include "ryty_gpu/native/shadow.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/regs.h"

namespace Native {
namespace {

using AmdGpu::PM4ItOpcode;
using AmdGpu::Regs;

const bool g_enabled = [] {
    const char* env = std::getenv("RYGT_NATIVE_SHADOW");
    return env && env[0] == '1';
}();

// The state compared at each draw (dword indices into Regs::reg_array): the graphics stages' SH
// registers (programs, user data; compute registers from 0x2E00 are kept apart by the command
// processor), all context registers, and the draw registers outside those (uconfig).
constexpr u32 kShFirst = Regs::ShRegWordOffset;
constexpr u32 kShCount = 0x200;
constexpr u32 kCtxFirst = Regs::ContextRegWordOffset;
constexpr u32 kCtxCount = 0x400;
#define NAT_FIELD(field) static_cast<u32>(offsetof(Regs, field) / sizeof(u32))
constexpr std::array<u32, 3> kDrawFields = {NAT_FIELD(primitive_type), NAT_FIELD(num_indices),
                                        NAT_FIELD(num_instances)};
#undef NAT_FIELD

constexpr int kShGroups = kShCount / 64;
constexpr int kCtxGroups = kCtxCount / 64;
constexpr int kGroups = kShGroups + kCtxGroups + 1;
constexpr u32 kFullDwords = kShCount + kCtxCount + 3;

std::string GroupName(int group) {
    if (group < kShGroups) {
        return fmt::format("sh {:#x}", kShFirst + group * 64);
    }
    if (group < kShGroups + kCtxGroups) {
        return fmt::format("ctx {:#x}", kCtxFirst + (group - kShGroups) * 64);
    }
    return "draw";
}

u32 FullIndexToRegister(u32 index) {
    if (index < kShCount) {
        return kShFirst + index;
    }
    if (index < kShCount + kCtxCount) {
        return kCtxFirst + index - kShCount;
    }
    return kDrawFields[index - kShCount - kCtxCount];
}

using Hashes = std::array<u64, kGroups>;
using Full = std::array<u32, kFullDwords>;

void HashRegs(const u32* regs, Hashes& out) {
    for (int g = 0; g < kShGroups; ++g) {
        out[g] = XXH3_64bits(regs + kShFirst + g * 64, 64 * sizeof(u32));
    }
    for (int g = 0; g < kCtxGroups; ++g) {
        out[kShGroups + g] = XXH3_64bits(regs + kCtxFirst + g * 64, 64 * sizeof(u32));
    }
    u32 fields[kDrawFields.size()];
    for (size_t i = 0; i < kDrawFields.size(); ++i) {
        fields[i] = regs[kDrawFields[i]];
    }
    out[kGroups - 1] = XXH3_64bits(fields, sizeof(fields));
}

void CaptureFull(const u32* regs, Full& out) {
    for (u32 i = 0; i < kFullDwords; ++i) {
        out[i] = regs[FullIndexToRegister(i)];
    }
}

// NAT-007: the registers a draw depends on. Registers left over from earlier work (another job's,
// or ones the draw's own settings switch off) can hold anything without changing what is drawn,
// so jobs need not agree with the GPU on them. The mask keeps the enabled stages' programs and
// the user data they read, the render targets written, and the rest of the context state except
// parts the draw's settings switch off: interpolants past the pixel shader's count, disabled
// color targets and their blending, clear values without a clear, the depth surface when depth,
// stencil and clears are all off, depth bounds and stencil state when off, sample positions
// without MSAA, viewports past the first, disabled clip planes, polygon offset when off,
// tessellation state without a hull shader and the restart index without primitive restart.
constexpr u32 kMaskWords = (kFullDwords + 63) / 64;
using Mask = std::array<u64, kMaskWords>;

constexpr u32 FullIndex(u32 reg) {
    return reg < kCtxFirst ? reg - kShFirst : kShCount + reg - kCtxFirst;
}
void SetRegs(Mask& mask, u32 first_reg, u32 count, bool value = true) {
    for (u32 i = 0; i < count; ++i) {
        const u32 bit = FullIndex(first_reg + i);
        if (value) {
            mask[bit / 64] |= u64{1} << (bit % 64);
        } else {
            mask[bit / 64] &= ~(u64{1} << (bit % 64));
        }
    }
}
bool MaskTest(const Mask& mask, u32 bit) {
    return (mask[bit / 64] >> (bit % 64)) & 1;
}

#define NAT_REG(field) static_cast<u32>(offsetof(Regs, field) / sizeof(u32))
constexpr u32 kAaCentroidPriority = 0xA2F5; // PA_SC_CENTROID_PRIORITY_0/1
constexpr u32 kAaSampleLocations = 0xA2FE;  // PA_SC_AA_SAMPLE_LOCS_PIXEL_*, 16 registers
static_assert(NAT_REG(aa_config) == 0xA2F8);
constexpr u32 kColorBufferDwords = sizeof(AmdGpu::ColorBuffer) / sizeof(u32);
// VGT_HOS_CNTL, VGT_HOS_MAX/MIN_TESS_LEVEL, VGT_HOS_REUSE_DEPTH (Regs::hs_clamp sits one
// register late, at 0xA287).
constexpr u32 kHosFirst = 0xA285;
static_assert(NAT_REG(index_base_address) == 0xA1F9 && NAT_REG(index_size) == 0xA29D);
// Depth surface details the renderer doesn't use: DB_DEPTH_INFO (legacy tiling; the tile mode
// comes from DB_Z_INFO) and DB_HTILE_SURFACE (HTILE layout), and of DB_RENDER_CONTROL only the
// clear, copy and resummarize bits count (the rest switch HTILE compression).
constexpr u32 kDbDepthInfo = 0xA00F;
constexpr u32 kDbHtileSurface = 0xA2AF;
constexpr u32 kDbEqaa = 0xA201;          // EQAA settings: not used by the renderer
constexpr u32 kDbCountControl = 0xA001; // occlusion counting: not draw state
constexpr u32 kPaPointFirst = 0xA280;    // PA_SU_POINT_SIZE, PA_SU_POINT_MINMAX: points only
constexpr u32 kPaLineFirst = 0xA282;     // PA_SU_LINE_CNTL, PA_SC_LINE_STIPPLE: lines only
constexpr u32 kVgtGsMaxVertOut = 0xA2CE; // geometry shader state (the field is a bit field)
constexpr u32 kDbRenderControlBits = 0x1f;

u32 ValueMask(u32 full_index) {
    return full_index == FullIndex(NAT_REG(depth_render_control)) ? kDbRenderControlBits
                                                                   : 0xffffffffu;
}

void SetGsRegs(Mask& mask, bool value) {
    SetRegs(mask, NAT_REG(vgt_gs_onchip_control), 1, value);
    SetRegs(mask, NAT_REG(vgt_gsvs_ring_offset_1), 4, value); // ring offsets 1-3, output primitive
    SetRegs(mask, NAT_REG(vgt_esgs_ring_itemsize), 2, value);
    SetRegs(mask, kVgtGsMaxVertOut, 1, value);
    SetRegs(mask, NAT_REG(vgt_gs_vert_itemsize), 4, value);
    SetRegs(mask, NAT_REG(vgt_gs_instance_cnt), 1, value);
}

const Mask kBaseMask = [] {
    Mask mask{};
    SetRegs(mask, kCtxFirst, kCtxCount);
    for (u32 i = 0; i < kDrawFields.size(); ++i) {
        const u32 bit = kShCount + kCtxCount + i;
        mask[bit / 64] |= u64{1} << (bit % 64);
    }
    SetRegs(mask, NAT_REG(ps_inputs), 32, false);
    SetRegs(mask, NAT_REG(color_buffers), AmdGpu::NUM_COLOR_BUFFERS * kColorBufferDwords, false);
    SetRegs(mask, NAT_REG(blend_control), AmdGpu::NUM_COLOR_BUFFERS, false);
    SetRegs(mask, NAT_REG(blend_constants), sizeof(AmdGpu::BlendConstants) / sizeof(u32), false);
    SetRegs(mask, NAT_REG(depth_view), 1, false);
    SetRegs(mask, NAT_REG(depth_htile_data_base), 1, false);
    SetRegs(mask, NAT_REG(depth_buffer), sizeof(AmdGpu::DepthBuffer) / sizeof(u32), false);
    SetRegs(mask, NAT_REG(depth_clear), 1, false);
    SetRegs(mask, NAT_REG(stencil_clear), 1, false);
    SetRegs(mask, NAT_REG(depth_bounds_min), 2, false);
    SetRegs(mask, NAT_REG(stencil_control), 3, false); // stencil ops, front and back refs
    SetRegs(mask, kAaCentroidPriority, 2, false);
    SetRegs(mask, kAaSampleLocations, 16, false);
    constexpr u32 kViewport = sizeof(AmdGpu::ViewportBounds) / sizeof(u32);
    constexpr u32 kScissor = sizeof(AmdGpu::ViewportScissor) / sizeof(u32);
    constexpr u32 kDepthRange = sizeof(AmdGpu::ViewportDepth) / sizeof(u32);
    SetRegs(mask, NAT_REG(viewports) + kViewport, (AmdGpu::NUM_VIEWPORTS - 1) * kViewport, false);
    SetRegs(mask, NAT_REG(viewport_scissors) + kScissor, (AmdGpu::NUM_VIEWPORTS - 1) * kScissor,
            false);
    SetRegs(mask, NAT_REG(viewport_depths) + kDepthRange,
            (AmdGpu::NUM_VIEWPORTS - 1) * kDepthRange, false);
    SetRegs(mask, NAT_REG(clip_user_data),
            AmdGpu::NUM_CLIP_PLANES * sizeof(AmdGpu::ClipUserData) / sizeof(u32), false);
    SetRegs(mask, NAT_REG(poly_offset), sizeof(AmdGpu::PolygonOffset) / sizeof(u32), false);
    SetRegs(mask, NAT_REG(ls_hs_config), 1, false);
    SetRegs(mask, NAT_REG(tess_config), 1, false);
    SetRegs(mask, kHosFirst, 4, false);
    SetRegs(mask, NAT_REG(primitive_restart_index), 1, false);
    SetRegs(mask, kDbDepthInfo, 1, false);
    SetRegs(mask, kDbHtileSurface, 1, false);
    SetRegs(mask, kDbEqaa, 1, false);
    SetRegs(mask, kDbCountControl, 1, false);
    SetRegs(mask, kPaPointFirst, 2, false);
    SetRegs(mask, kPaLineFirst, 2, false);
    SetRegs(mask, NAT_REG(aa_config), 1, false);
    SetGsRegs(mask, false);
    // index buffer: base, size, type
    SetRegs(mask, NAT_REG(index_base_address), 2, false);
    SetRegs(mask, NAT_REG(index_size), 3, false);
    return mask;
}();

void BuildMask(const Regs& regs, Mask& mask) {
    mask = kBaseMask;
    const u32* raw = regs.reg_array.data();
    for (u32 stage = 0; stage < 6; ++stage) {
        if (!regs.stage_enable.IsStageEnabled(stage)) {
            continue;
        }
        const AmdGpu::ShaderProgram* program = regs.ProgramForStage(stage);
        const u32 first = static_cast<u32>(reinterpret_cast<const u32*>(program) - raw);
        // address, settings and the user data the program reads
        SetRegs(mask, first, 4 + std::min<u32>(program->settings.num_user_regs, 16));
    }
    SetRegs(mask, NAT_REG(ps_inputs), std::min<u32>(regs.num_interp, 32));
    if (regs.color_control.mode != AmdGpu::ColorControl::OperationMode::Disable) {
        bool blending = false;
        for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
            if (regs.color_target_mask.GetMask(i) == 0 || !regs.color_buffers[i]) {
                continue;
            }
            SetRegs(mask, NAT_REG(color_buffers) + i * kColorBufferDwords, kColorBufferDwords);
            SetRegs(mask, NAT_REG(blend_control) + i, 1);
            blending |= regs.blend_control[i].enable != 0;
        }
        if (blending) {
            SetRegs(mask, NAT_REG(blend_constants), sizeof(AmdGpu::BlendConstants) / sizeof(u32));
        }
    }
    const auto& depth = regs.depth_control;
    const auto& render = regs.depth_render_control;
    if (depth.depth_enable || depth.stencil_enable || render.depth_clear_enable ||
        render.stencil_clear_enable) {
        SetRegs(mask, NAT_REG(depth_view), 1);
        SetRegs(mask, NAT_REG(depth_htile_data_base), 1);
        SetRegs(mask, NAT_REG(depth_buffer), sizeof(AmdGpu::DepthBuffer) / sizeof(u32));
    }
    if (render.depth_clear_enable) {
        SetRegs(mask, NAT_REG(depth_clear), 1);
    }
    if (render.stencil_clear_enable) {
        SetRegs(mask, NAT_REG(stencil_clear), 1);
    }
    if (depth.depth_bounds_enable) {
        SetRegs(mask, NAT_REG(depth_bounds_min), 2);
    }
    if (depth.stencil_enable) {
        SetRegs(mask, NAT_REG(stencil_control), 3);
    }
    // The renderer takes sample counts from the render targets, and the MSAA config only for
    // pixel shaders that read the sample coverage; it keeps the standard sample positions.
    if (regs.ps_input_ena.sample_coverage_ena && regs.ps_input_addr.sample_coverage_ena) {
        SetRegs(mask, NAT_REG(aa_config), 1);
    }
    switch (regs.primitive_type) {
    case AmdGpu::PrimitiveType::PointList:
        SetRegs(mask, kPaPointFirst, 2);
        break;
    case AmdGpu::PrimitiveType::LineList:
    case AmdGpu::PrimitiveType::LineStrip:
        SetRegs(mask, kPaLineFirst, 2);
        break;
    default:
        break;
    }
    if (regs.stage_enable.gs_en) {
        SetGsRegs(mask, true);
    }
    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
        if ((regs.clipper_control.user_clip_plane_enable >> i) & 1) {
            constexpr u32 kPlane = sizeof(AmdGpu::ClipUserData) / sizeof(u32);
            SetRegs(mask, NAT_REG(clip_user_data) + i * kPlane, kPlane);
        }
    }
    if (regs.polygon_control.NeedsBias()) {
        SetRegs(mask, NAT_REG(poly_offset), sizeof(AmdGpu::PolygonOffset) / sizeof(u32));
    }
    if (regs.stage_enable.hs_en) {
        SetRegs(mask, NAT_REG(ls_hs_config), 1);
        SetRegs(mask, NAT_REG(tess_config), 1);
        SetRegs(mask, kHosFirst, 4);
    }
    if ((regs.draw_initiator & 3) == 0) { // indices read from memory
        SetRegs(mask, NAT_REG(index_base_address), 2);
        SetRegs(mask, NAT_REG(index_size), 3);
    }
    if (regs.enable_primitive_restart & 1) {
        SetRegs(mask, NAT_REG(primitive_restart_index), 1);
    }
}
#undef NAT_REG

/// A hash of the registers the draw depends on (and of which ones those are).
u64 RelevantHash(const Regs& regs, Mask* mask_out = nullptr) {
    Mask mask;
    BuildMask(regs, mask);
    std::array<u32, kFullDwords> values;
    u32 n = 0;
    for (u32 w = 0; w < kMaskWords; ++w) {
        for (u64 bits = mask[w]; bits != 0; bits &= bits - 1) {
            const u32 bit = w * 64 + static_cast<u32>(std::countr_zero(bits));
            values[n++] = regs.reg_array[FullIndexToRegister(bit)] & ValueMask(bit);
        }
    }
    if (mask_out) {
        *mask_out = mask;
    }
    return XXH3_64bits_withSeed(values.data(), n * sizeof(u32), XXH3_64bits(&mask, sizeof(mask)));
}

// What a job worked out at one of its draws, keyed by the draw packet's address.
struct Sample {
    Full values;
    Mask mask;
};
struct Entry {
    Hashes hashes;     // all compared registers, in groups
    u64 relevant;      // the registers the draw depends on (NAT-007)
    std::unique_ptr<Sample> sample; // one draw in 32 keeps every value, to name what differs
    u32 epoch;
    bool from_parent; // the job started from its parent's state at the fork (NAT-006)
};
struct Shard {
    std::mutex mutex;
    std::unordered_map<u64, Entry> map;
};
std::array<Shard, 64> g_shards;
std::atomic<u32> g_epoch{0}; // graphics submissions seen by the command processor

Shard& ShardOf(u64 key) {
    return g_shards[(key * 0x9E3779B97F4A7C15ULL) >> 58];
}

struct Stats {
    std::atomic<u64> jobs_begun, jobs_decoded, jobs_unmatched_end, jobs_without_base;
    std::atomic<u64> draws, packets, chain_links, call_links, bad_packets, overruns;
    std::atomic<u64> decode_ns;
    std::atomic<u64> checked, identical, different, not_decoded;
    std::atomic<u64> group_different[kGroups];
    // NAT-006: forks, and where jobs' start states came from (0: submission, 1: parent).
    std::atomic<u64> forks, forks_walked, forks_failed, forks_other_context, fork_states_dropped;
    std::atomic<u64> jobs_from[2], checked_from[2], identical_from[2];
    // NAT-007: draws whose relevant registers match, and sampled draws whose masks differ.
    std::atomic<u64> relevant_same, masks_differ, relevant_group_different[kGroups];
    std::atomic<u64> forks_other_logged;
    // NAT-008: forks from the frame's root context, followed like a job's.
    std::atomic<u64> forks_root, root_restarts, root_walks_failed;
    // NAT-009: jobs that move between threads, or end out of order.
    std::atomic<u64> forks_job_elsewhere, ends_out_of_order, ends_job_elsewhere, ends_unknown;
    std::atomic<u64> clear_states;
};
Stats g_stats{};

std::mutex g_register_mutex;
struct RegisterDiff {
    u64 count = 0;
    u32 cp_value = 0;
    u32 job_value = 0;
};
std::unordered_map<u32, RegisterDiff> g_register_diffs; // sampled draws only
std::unordered_map<u32, RegisterDiff> g_relevant_diffs; // sampled draws, registers that matter

constexpr u32 kContextDcb = 0x780; // a PGL context's Gnm DrawCommandBuffer

const u32* DcbBegin(const u8* dcb) {
    return *reinterpret_cast<const u32* const*>(dcb); // Gnm CommandBuffer::m_beginptr
}
const u32* DcbEnd(const u8* dcb) {
    return *reinterpret_cast<const u32* const*>(dcb + 0x8); // m_endptr
}
const u32* DcbCmdPtr(const u8* dcb) {
    return *reinterpret_cast<const u32* const*>(dcb + 0x10); // m_cmdptr
}

// A job open on this thread: its walk goes as far as the job has recorded (to each fork, then to
// its end), with the state at that point.
struct Job {
    const u8* dcb = nullptr;
    const u32* end = nullptr; // where the job's commands end (set at job end)
    std::unique_ptr<Regs> regs;
    JobExtents extents{};
    JobCursor cursor;
    WalkStats walk{};
    bool from_parent = false;
    u64 walk_ns = 0;
};
thread_local std::vector<Job> t_jobs;
// NAT-009: open jobs of all threads (context's command buffer -> thread), to see whether a job's
// context forks or ends on another thread than the one that began it.
std::mutex g_open_mutex;
std::unordered_map<const u8*, u32> g_open_jobs;
bool OpenElsewhere(const u8* dcb) {
    std::scoped_lock lock{g_open_mutex};
    const auto it = g_open_jobs.find(dcb);
    return it != g_open_jobs.end() && it->second != GetCurrentThreadId();
}
thread_local Job t_ended{};
thread_local bool t_has_ended = false;
thread_local std::vector<std::unique_ptr<Regs>> t_free_regs; // Regs is large: reused
thread_local u32 t_sample = 0;

std::unique_ptr<Regs> TakeRegs() {
    if (t_free_regs.empty()) {
        return std::make_unique<Regs>();
    }
    std::unique_ptr<Regs> regs = std::move(t_free_regs.back());
    t_free_regs.pop_back();
    return regs;
}

// NAT-006: a child job starts with the GPU state its parent had at the fork (the GPU runs the
// child right there), handed over from the parent's thread, keyed by the child's context.
struct ForkState {
    std::array<u32, kShCount> sh;
    std::array<u32, kCtxCount> ctx;
    std::array<u32, kDrawFields.size()> draw;
    JobExtents extents;
};
std::mutex g_fork_mutex;
std::unordered_map<const u8*, std::unique_ptr<ForkState>> g_fork_states;

void CopyCompared(const u32* from, u32* to) {
    std::memcpy(to + kShFirst, from + kShFirst, kShCount * sizeof(u32));
    std::memcpy(to + kCtxFirst, from + kCtxFirst, kCtxCount * sizeof(u32));
    for (const u32 field : kDrawFields) {
        to[field] = from[field];
    }
}

// The registers at the start of the latest graphics submission: the state other jobs start from.
std::shared_mutex g_base_mutex;
std::unique_ptr<Regs> g_base;

/// The state a job starts from: its parent's at the fork, or the latest submission's.
bool InitJobState(Job& job) {
    const u8* context = job.dcb - kContextDcb;
    std::unique_ptr<ForkState> fork;
    {
        std::scoped_lock lock{g_fork_mutex};
        if (const auto it = g_fork_states.find(context); it != g_fork_states.end()) {
            fork = std::move(it->second);
            g_fork_states.erase(it);
        }
    }
    u32* regs = job.regs->reg_array.data();
    if (fork) {
        std::memcpy(regs + kShFirst, fork->sh.data(), kShCount * sizeof(u32));
        std::memcpy(regs + kCtxFirst, fork->ctx.data(), kCtxCount * sizeof(u32));
        for (size_t i = 0; i < kDrawFields.size(); ++i) {
            regs[kDrawFields[i]] = fork->draw[i];
        }
        job.extents = fork->extents;
        job.from_parent = true;
        return true;
    }
    std::shared_lock lock{g_base_mutex};
    if (!g_base) {
        return false;
    }
    CopyCompared(g_base->reg_array.data(), regs);
    job.extents = {};
    return true;
}

void Snapshot(const u32* packet, const Regs& regs, u32 epoch, bool from_parent) {
    Entry entry;
    entry.from_parent = from_parent;
    HashRegs(regs.reg_array.data(), entry.hashes);
    if ((t_sample++ & 31) == 0) {
        entry.sample = std::make_unique<Sample>();
        CaptureFull(regs.reg_array.data(), entry.sample->values);
        entry.relevant = RelevantHash(regs, &entry.sample->mask);
    } else {
        entry.relevant = RelevantHash(regs);
    }
    entry.epoch = epoch;
    const u64 key = reinterpret_cast<u64>(packet);
    Shard& shard = ShardOf(key);
    std::scoped_lock lock{shard.mutex};
    if (shard.map.size() > 20000) {
        // Draws the command processor never checked (not submitted, or not reached yet after a
        // long stall): forget the ones from more than a few submissions ago.
        std::erase_if(shard.map, [epoch](const auto& item) { return epoch - item.second.epoch > 8; });
    }
    shard.map.insert_or_assign(key, std::move(entry));
}

// Records a snapshot of the state at each draw of a job.
class SnapshotVisitor final : public JobVisitor {
public:
    SnapshotVisitor(u32 epoch, bool from_parent) : epoch{epoch}, from_parent{from_parent} {}
    void OnDraw(const u32* packet, PM4ItOpcode, const Regs& regs, const JobExtents&) override {
        Snapshot(packet, regs, epoch, from_parent);
    }

private:
    u32 epoch;
    bool from_parent;
};

// Walks a job's commands up to `until` like the command processor does (NAT-004), recording a
// snapshot of the state at each draw.
void WalkJobTo(Job& job, const u32* until) {
    const auto start = std::chrono::steady_clock::now();
    SnapshotVisitor visitor{g_epoch.load(std::memory_order_relaxed), job.from_parent};
    job.cursor.WalkTo(until, *job.regs, job.extents, visitor, job.walk);
    job.walk_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count();
}

std::mutex g_failure_mutex;
std::unordered_map<std::string, u64> g_failures;         // reason -> walks, this window
std::unordered_map<std::string, u32> g_failures_logged;  // reason -> walks logged in detail

void LogFailedWalk(const Job& job) {
    // NAT-005: failed walks by reason, and the first few of each reason in detail: the job's
    // range, where the walk stopped and the chunks it went through.
    const WalkStats& walk = job.walk;
    const std::string reason = walk.reason ? walk.reason : "?";
    u32 n;
    {
        std::scoped_lock lock{g_failure_mutex};
        ++g_failures[reason];
        n = ++g_failures_logged[reason];
    }
    if (n > 4) {
        return;
    }
    std::string chunks;
    bool end_inside = false;
    for (u32 i = 0; i < std::min<u32>(walk.num_chunks, walk.chunks.size()); ++i) {
        const auto [begin, limit] = walk.chunks[i];
        chunks += fmt::format(" [{}..{}]", fmt::ptr(begin), fmt::ptr(limit));
        end_inside |= job.end >= begin && job.end <= limit;
    }
    std::string around;
    if (walk.stop != nullptr && Readable(walk.stop - 4, 12 * sizeof(u32))) {
        for (int i = -4; i < 8; ++i) {
            around += fmt::format("{}{:08x}", i == 0 ? " >" : " ", walk.stop[i]);
        }
    }
    LOG_WARNING(Lib_GnmDriver,
                "NAT-005 walk failed ({} of this kind): {} at {} (header {:#010x}) after {} "
                "packets, {} draws, {} forks; job end {} ({} in a visited chunk); dwords around "
                "the stop:{}; {} chunks:{}",
                n, reason, fmt::ptr(walk.stop), walk.stop_header,
                walk.packets, walk.draws, walk.forks, fmt::ptr(job.end),
                end_inside ? "it lies" : "NOT", around.empty() ? " unreadable" : around,
                walk.num_chunks, chunks);
}

// The job's walk is over: account for it and give its registers back.
void FinishJob(Job& job) {
    const WalkStats& walk = job.walk;
    if (walk.bad_packet || walk.overrun) {
        LogFailedWalk(job);
    }
    g_stats.clear_states += walk.clear_states;
    g_stats.packets += walk.packets;
    g_stats.draws += walk.draws;
    g_stats.chain_links += walk.chain_links;
    g_stats.call_links += walk.call_links;
    g_stats.bad_packets += walk.bad_packet;
    g_stats.overruns += walk.overrun;
    ++g_stats.jobs_decoded;
    g_stats.decode_ns += job.walk_ns;
    t_free_regs.push_back(std::move(job.regs));
}

void ReporterLoop() {
    using namespace std::chrono_literals;
    Stats last{};
    const auto take = [](const std::atomic<u64>& value, std::atomic<u64>& previous) {
        const u64 now = value.load(std::memory_order_relaxed);
        const u64 delta = now - previous.load(std::memory_order_relaxed);
        previous.store(now, std::memory_order_relaxed);
        return delta;
    };
    while (true) {
        std::this_thread::sleep_for(10s);
        const u64 jobs = take(g_stats.jobs_decoded, last.jobs_decoded);
        const u64 draws = take(g_stats.draws, last.draws);
        const u64 packets = take(g_stats.packets, last.packets);
        const u64 ns = take(g_stats.decode_ns, last.decode_ns);
        LOG_INFO(Lib_GnmDriver,
                 "NAT-003 jobs (10 s): {} begun, {} decoded on their threads ({} draws, {} "
                 "packets, {} chunk links, {} CLEAR_STATEs, {:.1f} us per job, {:.0f} ns per draw); "
                 "problems: {} bad "
                 "packets, {} overruns, {} calls not followed, {} without base state, {} unmatched "
                 "ends",
                 take(g_stats.jobs_begun, last.jobs_begun), jobs, draws, packets,
                 take(g_stats.chain_links, last.chain_links),
                 take(g_stats.clear_states, last.clear_states), jobs ? ns / 1000.0 / jobs : 0.0,
                 draws ? double(ns) / draws : 0.0, take(g_stats.bad_packets, last.bad_packets),
                 take(g_stats.overruns, last.overruns), take(g_stats.call_links, last.call_links),
                 take(g_stats.jobs_without_base, last.jobs_without_base),
                 take(g_stats.jobs_unmatched_end, last.jobs_unmatched_end));
        const u64 checked_parent = take(g_stats.checked_from[1], last.checked_from[1]);
        const u64 checked_base = take(g_stats.checked_from[0], last.checked_from[0]);
        const u64 identical_parent = take(g_stats.identical_from[1], last.identical_from[1]);
        const u64 identical_base = take(g_stats.identical_from[0], last.identical_from[0]);
        LOG_INFO(Lib_GnmDriver,
                 "NAT-006 forks (10 s): {} forks, {} walked to on the parent's thread ({} walks "
                 "failed there), {} from the frame's root context ({} restarts, {} walks failed), "
                 "{} from jobs open on another thread, {} from other contexts, {} handovers "
                 "dropped; job ends: {} out of order, {} of jobs open on another thread, {} of "
                 "contexts that are not jobs; jobs started from their parent's "
                 "state {} (draws checked {}, the same where it matters {:.2f}%), from the "
                 "submission's state {} (draws checked {}, the same {:.2f}%)",
                 take(g_stats.forks, last.forks), take(g_stats.forks_walked, last.forks_walked),
                 take(g_stats.forks_failed, last.forks_failed),
                 take(g_stats.forks_root, last.forks_root),
                 take(g_stats.root_restarts, last.root_restarts),
                 take(g_stats.root_walks_failed, last.root_walks_failed),
                 take(g_stats.forks_job_elsewhere, last.forks_job_elsewhere),
                 take(g_stats.forks_other_context, last.forks_other_context),
                 take(g_stats.fork_states_dropped, last.fork_states_dropped),
                 take(g_stats.ends_out_of_order, last.ends_out_of_order),
                 take(g_stats.ends_job_elsewhere, last.ends_job_elsewhere),
                 take(g_stats.ends_unknown, last.ends_unknown),
                 take(g_stats.jobs_from[1], last.jobs_from[1]), checked_parent,
                 checked_parent ? 100.0 * identical_parent / checked_parent : 0.0,
                 take(g_stats.jobs_from[0], last.jobs_from[0]), checked_base,
                 checked_base ? 100.0 * identical_base / checked_base : 0.0);
        {
            std::string failures;
            std::scoped_lock lock{g_failure_mutex};
            for (const auto& [reason, count] : g_failures) {
                failures += fmt::format(" [{}: {}]", reason, count);
            }
            g_failures.clear();
            if (!failures.empty()) {
                LOG_INFO(Lib_GnmDriver, "NAT-005 failed walks by reason (10 s):{}", failures);
            }
        }
        const u64 checked = take(g_stats.checked, last.checked);
        const u64 relevant_same = take(g_stats.relevant_same, last.relevant_same);
        const u64 identical = take(g_stats.identical, last.identical);
        std::string groups;
        for (int g = 0; g < kGroups; ++g) {
            if (const u64 n = take(g_stats.relevant_group_different[g],
                                   last.relevant_group_different[g])) {
                groups += fmt::format(" [{}: {}]", GroupName(g), n);
            }
            take(g_stats.group_different[g], last.group_different[g]);
        }
        LOG_INFO(Lib_GnmDriver,
                 "NAT-007 compare (10 s): {} draws checked by the command processor; the same where "
                 "it matters {} ({:.2f}%), different {}, not decoded by a job {}; every register "
                 "the same {} ({:.2f}%); sampled draws where the job and the GPU disagree on "
                 "which registers matter {}; groups with relevant differences (sampled):{}",
                 checked, relevant_same, checked ? 100.0 * relevant_same / checked : 0.0,
                 take(g_stats.different, last.different),
                 take(g_stats.not_decoded, last.not_decoded), identical,
                 checked ? 100.0 * identical / checked : 0.0,
                 take(g_stats.masks_differ, last.masks_differ), groups.empty() ? " none" : groups);
        const auto report = [](std::unordered_map<u32, RegisterDiff>& table, const char* what) {
            std::vector<std::pair<u32, RegisterDiff>> diffs;
            {
                std::scoped_lock lock{g_register_mutex};
                diffs.assign(table.begin(), table.end());
                table.clear();
            }
            if (diffs.empty()) {
                return;
            }
            std::ranges::sort(diffs, [](const auto& a, const auto& b) {
                return a.second.count > b.second.count;
            });
            std::string line;
            for (size_t i = 0; i < std::min<size_t>(diffs.size(), 24); ++i) {
                line += fmt::format(" [{:#x}: {} (cp {:#x}, job {:#x})]", diffs[i].first,
                                    diffs[i].second.count, diffs[i].second.cp_value,
                                    diffs[i].second.job_value);
            }
            LOG_INFO(Lib_GnmDriver, "NAT-007 {} (sampled draws, 10 s, {} registers):{}", what,
                     diffs.size(), line);
        };
        report(g_relevant_diffs, "relevant registers that differ");
        report(g_register_diffs, "registers that differ (any)");
    }
}

std::atomic<bool> g_reporter_started{false};

} // namespace

bool ShadowEnabled() {
    return g_enabled;
}

void ShadowJobBegin(const u8* dcb) {
    ++g_stats.jobs_begun;
    Job job;
    job.dcb = dcb;
    job.regs = TakeRegs();
    if (!InitJobState(job)) {
        ++g_stats.jobs_without_base;
        t_free_regs.push_back(std::move(job.regs));
    } else {
        ++g_stats.jobs_from[job.from_parent];
        job.cursor.Start(DcbBegin(dcb), DcbEnd(dcb), job.walk);
    }
    if (t_jobs.size() >= 64) {
        // Jobs that ended on another thread stay open here: don't let them pile up.
        std::scoped_lock lock{g_open_mutex};
        for (auto it = t_jobs.begin(); it != t_jobs.begin() + 32; ++it) {
            g_open_jobs.erase(it->dcb);
            if (it->regs) {
                t_free_regs.push_back(std::move(it->regs));
            }
        }
        t_jobs.erase(t_jobs.begin(), t_jobs.begin() + 32);
    }
    t_jobs.push_back(std::move(job));
    std::scoped_lock lock{g_open_mutex};
    g_open_jobs[dcb] = GetCurrentThreadId();
}

namespace {

// Hands the parent's state at the fork to the child job.
void HandOver(const Job& parent, const u8* child_context) {
    auto state = std::make_unique<ForkState>();
    const u32* regs = parent.regs->reg_array.data();
    std::memcpy(state->sh.data(), regs + kShFirst, kShCount * sizeof(u32));
    std::memcpy(state->ctx.data(), regs + kCtxFirst, kCtxCount * sizeof(u32));
    for (size_t i = 0; i < kDrawFields.size(); ++i) {
        state->draw[i] = regs[kDrawFields[i]];
    }
    state->extents = parent.extents;
    std::scoped_lock lock{g_fork_mutex};
    if (g_fork_states.size() > 20000) {
        // children that never began (should not happen): don't let them pile up
        g_stats.fork_states_dropped += g_fork_states.size();
        g_fork_states.clear();
    }
    g_fork_states.insert_or_assign(child_context, std::move(state));
}

// NAT-008: the frame's root context is not a job. PGL's frame code makes it current, writes the
// frame's setup (render targets, MSAA, depth surface) and forks the frame's jobs from it, so its
// commands are followed like a job's: from where its current run began (the frame's start, with
// the submission's state), up to each fork.
std::mutex g_roots_mutex;
std::unordered_map<const u8*, std::unique_ptr<Job>> g_roots;

Job* RootFor(const u8* context, const u32* slot, const u32* run_begin, const u32* run_limit) {
    std::unique_ptr<Job>& root = g_roots[context];
    if (root && root->cursor.Ok() && slot >= root->cursor.Position() &&
        slot < root->cursor.Limit()) {
        return root.get(); // the fork lies ahead in the run the walk is in
    }
    // A new frame (the root's command buffer starts over), or the walk lost its way.
    if (!root) {
        root = std::make_unique<Job>();
        root->regs = std::make_unique<Regs>();
    }
    {
        std::shared_lock lock{g_base_mutex};
        if (!g_base) {
            return nullptr;
        }
        CopyCompared(g_base->reg_array.data(), root->regs->reg_array.data());
    }
    ++g_stats.root_restarts;
    root->dcb = context + kContextDcb;
    root->end = nullptr;
    root->extents = {};
    root->walk = {};
    root->from_parent = false;
    root->walk_ns = 0;
    root->cursor.Start(run_begin, run_limit, root->walk);
    return root.get();
}

} // namespace

void ShadowFork(const u8* parent_context, const u32* slot, const u32* run_begin,
                const u32* run_limit, const u8* child_context, u64 caller) {
    ++g_stats.forks;
    if (parent_context == nullptr || child_context == nullptr) {
        ++g_stats.forks_other_context;
        return;
    }
    const u8* parent_dcb = parent_context + kContextDcb;
    Job* parent = nullptr;
    for (auto it = t_jobs.rbegin(); it != t_jobs.rend(); ++it) {
        if (it->dcb == parent_dcb) {
            parent = &*it;
            break;
        }
    }
    if (parent == nullptr && OpenElsewhere(parent_dcb)) {
        ++g_stats.forks_job_elsewhere; // its walk lives on the other thread: not followed
        return;
    }
    if (parent == nullptr) {
        if (g_stats.forks_other_logged.fetch_add(1) < 4) {
            LOG_INFO(Lib_GnmDriver,
                     "NAT-008 fork from a context that is not a job open on this thread "
                     "(followed as a root context): context {}, run {}..{}, link slot {}, child {}, "
                     "called from {:#x}; {} jobs open here",
                     fmt::ptr(parent_context), fmt::ptr(run_begin), fmt::ptr(run_limit),
                     fmt::ptr(slot), fmt::ptr(child_context), caller, t_jobs.size());
        }
        std::scoped_lock lock{g_roots_mutex};
        Job* root = RootFor(parent_context, slot, run_begin, run_limit);
        if (root == nullptr) {
            ++g_stats.forks_other_context;
            return;
        }
        WalkJobTo(*root, slot);
        root->cursor.SkipFork(DcbBegin(parent_dcb), DcbEnd(parent_dcb), root->walk);
        if (!root->cursor.Ok()) {
            ++g_stats.root_walks_failed;
            LogFailedWalk(*root);
            return;
        }
        ++g_stats.forks_root;
        HandOver(*root, child_context);
        return;
    }
    if (!parent->regs) {
        ++g_stats.forks_other_context;
        return;
    }
    // The parent's commands up to the link slot, then on where its own commands continue.
    WalkJobTo(*parent, slot);
    parent->cursor.SkipFork(DcbBegin(parent_dcb), DcbEnd(parent_dcb), parent->walk);
    if (!parent->cursor.Ok()) {
        ++g_stats.forks_failed;
        return;
    }
    ++g_stats.forks_walked;
    HandOver(*parent, child_context);
}

void ShadowJobEnd(const u8* current_context) {
    t_has_ended = false;
    if (t_jobs.empty()) {
        ++g_stats.jobs_unmatched_end;
        return;
    }
    // Job end ends the context current on this thread: normally the innermost job open here.
    auto it = std::prev(t_jobs.end());
    if (current_context != nullptr && it->dcb != current_context + kContextDcb) {
        const u8* dcb = current_context + kContextDcb;
        const auto found = std::find_if(t_jobs.begin(), t_jobs.end(),
                                        [dcb](const Job& job) { return job.dcb == dcb; });
        if (found == t_jobs.end()) {
            ++(OpenElsewhere(dcb) ? g_stats.ends_job_elsewhere : g_stats.ends_unknown);
            return; // not a job begun on this thread: leave this thread's jobs open
        }
        ++g_stats.ends_out_of_order;
        it = found;
    }
    t_ended = std::move(*it);
    t_jobs.erase(it);
    t_ended.end = DcbCmdPtr(t_ended.dcb);
    t_has_ended = true;
    std::scoped_lock lock{g_open_mutex};
    g_open_jobs.erase(t_ended.dcb);
}

void ShadowJobDecode() {
    if (!t_has_ended) {
        return;
    }
    t_has_ended = false;
    if (!t_ended.regs) {
        return; // no start state
    }
    WalkJobTo(t_ended, t_ended.end);
    FinishJob(t_ended);
}

void ShadowSubmitBase(const Regs& regs) {
    ++g_epoch;
    std::unique_lock lock{g_base_mutex};
    if (!g_base) {
        g_base = std::make_unique<Regs>();
    }
    std::memcpy(&g_base->reg_array[kShFirst], &regs.reg_array[kShFirst], kShCount * sizeof(u32));
    std::memcpy(&g_base->reg_array[kCtxFirst], &regs.reg_array[kCtxFirst], kCtxCount * sizeof(u32));
    for (const u32 field : kDrawFields) {
        g_base->reg_array[field] = regs.reg_array[field];
    }
}

void ShadowCheckDraw(const void* packet, const Regs& regs) {
    if (!g_reporter_started.exchange(true)) {
        std::thread{ReporterLoop}.detach();
    }
    ++g_stats.checked;
    const u64 key = reinterpret_cast<u64>(packet);
    Entry entry;
    {
        Shard& shard = ShardOf(key);
        std::scoped_lock lock{shard.mutex};
        const auto it = shard.map.find(key);
        if (it == shard.map.end()) {
            ++g_stats.not_decoded;
            return;
        }
        entry = std::move(it->second);
        shard.map.erase(it);
    }
    ++g_stats.checked_from[entry.from_parent];
    Hashes cp;
    HashRegs(regs.reg_array.data(), cp);
    bool same = true;
    for (int g = 0; g < kGroups; ++g) {
        if (cp[g] != entry.hashes[g]) {
            same = false;
            ++g_stats.group_different[g];
        }
    }
    if (same) {
        ++g_stats.identical;
    }
    Mask cp_mask;
    const bool relevant_same = RelevantHash(regs, &cp_mask) == entry.relevant;
    if (relevant_same) {
        ++g_stats.relevant_same;
        ++g_stats.identical_from[entry.from_parent];
    } else {
        ++g_stats.different;
    }
    if (!entry.sample || (same && relevant_same)) {
        return;
    }
    Full now;
    CaptureFull(regs.reg_array.data(), now);
    const Full& job = entry.sample->values;
    if (entry.sample->mask != cp_mask) {
        ++g_stats.masks_differ;
    }
    std::scoped_lock lock{g_register_mutex};
    std::array<bool, kGroups> groups{};
    for (u32 i = 0; i < kFullDwords; ++i) {
        if (now[i] == job[i]) {
            continue;
        }
        const u32 reg = FullIndexToRegister(i);
        auto& diff = g_register_diffs[reg];
        ++diff.count;
        diff.cp_value = now[i];
        diff.job_value = job[i];
        if (MaskTest(cp_mask, i) && ((now[i] ^ job[i]) & ValueMask(i)) != 0) {
            auto& relevant = g_relevant_diffs[reg];
            ++relevant.count;
            relevant.cp_value = now[i];
            relevant.job_value = job[i];
            groups[i < kShCount              ? i / 64
                   : i < kShCount + kCtxCount ? kShGroups + (i - kShCount) / 64
                                              : kGroups - 1] = true;
        }
    }
    for (int g = 0; g < kGroups; ++g) {
        g_stats.relevant_group_different[g] += groups[g];
    }
}

} // namespace Native
