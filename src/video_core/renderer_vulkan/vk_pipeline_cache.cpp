// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <array>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <mutex>
#include <ranges>
#include <sstream>
#include <chrono>
#include <thread>
#include <string>
#include <unordered_map>

#include "common/elf_info.h"
#include "common/guest_clock.h"
#include "common/nvtx.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/perf_monitor.h"
#include "common/recoverable.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/build_queue.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/spirv_check.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "ryty_gpu/ryty_diag.h"

namespace Shader::Optimization {
extern thread_local std::span<const u32> g_diag_compiling_code;
extern thread_local u64 g_diag_compiling_hash;
} // namespace Shader::Optimization
namespace Vulkan {

namespace {
std::atomic<u32> g_shader_translations_running{};
struct TranslationActivity {
    TranslationActivity() { g_shader_translations_running.fetch_add(1, std::memory_order_relaxed); }
    ~TranslationActivity() { g_shader_translations_running.fetch_sub(1, std::memory_order_relaxed); }
    TranslationActivity(const TranslationActivity&) = delete;
    TranslationActivity& operator=(const TranslationActivity&) = delete;
};
} // namespace

bool CompilationInProgress() {
    return PipelineBuildsRunning() != 0 ||
           g_shader_translations_running.load(std::memory_order_relaxed) != 0;
}

class PipelineCache::TranslateWorkers {
public:
    TranslateWorkers(const Shader::Profile& profile_, vk::Device device_, u32 count)
        : profile{profile_}, device{device_} {
        for (u32 i = 0; i < count; ++i) {
            threads.emplace_back([this, i](std::stop_token stop) { Run(stop, i); });
        }
    }
    ~TranslateWorkers() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        cv.notify_all();
    }

    void Queue(std::shared_ptr<TranslateJob> job) {
        {
            std::scoped_lock lk{mutex};
            queue.push_back(std::move(job));
        }
        cv.notify_one();
    }

private:
    void Run(std::stop_token stop, u32 index) {
        Common::SetCurrentThreadName(fmt::format("shadGT:ShaderTranslate{}", index).c_str());
        Common::PreferEfficiencyCores(); // PERF-097
        Shader::Pools pools;
        while (!stop.stop_requested()) {
            std::shared_ptr<TranslateJob> job;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, stop, [this] { return !queue.empty(); });
                if (queue.empty()) {
                    continue;
                }
                job = std::move(queue.front());
                queue.pop_front();
            }
            Translate(*job, pools);
            {
                std::scoped_lock lk{job->mutex};
                job->done = true;
            }
            job->cv.notify_all();
        }
    }

    void Translate(TranslateJob& job, Shader::Pools& pools) {
        TranslationActivity activity;
        const Shader::ShaderParams params{
            .user_data = std::span<const u32, Shader::ShaderParams::NumShaderUserData>{job.user_data},
            .code = job.code,
            .hash = job.hash,
        };
        auto info = std::make_unique<Shader::Info>(job.hw_stage, job.sw_stage, params);
        auto runtime_info = job.runtime_info;
        auto binding = job.start;
        const auto start = std::chrono::steady_clock::now();
        Shader::Optimization::g_diag_compiling_code = job.code;
        Shader::Optimization::g_diag_compiling_hash = job.hash;
        try {
            Common::RecoverableScope recoverable;
            const auto program = Shader::TranslateProgram(job.code, pools, *info, runtime_info,
                                                          profile);
            job.spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, binding);
        } catch (const Common::RecoverableFailure&) {
            // Translated again on the command thread when needed, which reports the failure.
            return;
        }
        if (!Common::PerfFeatureEnabled(44)) {
            info->resource_usage_known = false;
        }
        job.module = CompileSPV(job.spv, device);
        job.info = std::move(info);
        job.translate_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
    }

    const Shader::Profile& profile;
    vk::Device device;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::shared_ptr<TranslateJob>> queue;
    std::vector<std::jthread> threads;
};


static bool SaneTessConstants(const Shader::Info& hull, const AmdGpu::Liverpool* liverpool,
                              bool read_ahead); // FIX-058
#ifdef RYTY_GPU
void RytyWatchWrites(std::uint64_t address, std::uint32_t ms, std::string label); // DIAG-067
#endif

// PERF-098: precompile progress.
static std::atomic<u64> g_precompile_total{};
static std::atomic<u64> g_precompile_done{};
static std::atomic<bool> g_precompile_reading{};

// PERF-102: while the precompile runs at start, the game waits on the splash screen (its clocks
// held, so it sees no time pass), as if the splash screen were frozen until it is done.
static std::atomic<bool> g_precompile_hold{};

bool PrecompileHoldsGame() {
    return g_precompile_hold.load(std::memory_order_relaxed);
}

static void ReleasePrecompileHold() {
    if (g_precompile_hold.exchange(false)) {
        Common::GuestClock::ReleaseHold();
        LOG_WARNING(Render_Vulkan, "PERF-102: precompile done; the game continues");
    }
}

// PERF-103: compile slow motion. While draws are skipped because their pipeline is still being
// built, the guest clocks run at a quarter speed: held 3 ms of every 4 (no vblank passes while
// held, so the game's frame rate drops with it), so a long compile looks like slow motion.
static std::atomic<s64> g_last_skip_ns{};

static u32 CompileMode() {
#ifdef RYTY_GPU
    static const u32 mode = static_cast<u32>(std::clamp(EmulatorSettings.GetCompileMode(), 0, 2));
    return mode;
#else
    return 2;
#endif
}

static void NoteSkippedDraw() {
    g_last_skip_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                         std::memory_order_relaxed);
}

#ifdef RYTY_GPU
bool RytyGtsInRace(); // FIX-110, ryty_game_patch.cpp
#endif

bool CompileSlowMotion() {
    const s64 last = g_last_skip_ns.load(std::memory_order_relaxed);
#ifdef RYTY_GPU
    // FIX-113: a race keeps going at full speed while its new pipelines build (objects show a
    // moment later instead of the race slowing down). -DisablePerf 136 slows races down too.
    static const bool race_keeps_going = Common::PerfFeatureEnabled(136);
    if (race_keeps_going && RytyGtsInRace()) {
        return false;
    }
#endif
    return CompileMode() == 1 && last != 0 &&
           std::chrono::steady_clock::now().time_since_epoch().count() - last < 50'000'000;
}

static void SlowMotionThread(std::stop_token stop) {
    Common::SetCurrentThreadName("shadGT:CompileSlowMotion");
    using namespace std::chrono_literals;
    while (!stop.stop_requested()) {
        if (!CompileSlowMotion()) {
            std::this_thread::sleep_for(5ms);
            continue;
        }
        Common::GuestClock::TakeHold();
        std::this_thread::sleep_for(3ms);
        Common::GuestClock::ReleaseHold();
        std::this_thread::sleep_for(1ms);
    }
}

PrecompileProgress GetPrecompileProgress() {
    const u64 total = g_precompile_total.load(std::memory_order_relaxed);
    const u64 done = std::min(total, g_precompile_done.load(std::memory_order_relaxed));
    return {total, done, total != 0 && done < total};
}

// PERF-019: worker threads that build graphics pipelines ahead of the draws that need them.
// PERF-038: stored pipelines are built in the background on a few of them, at low priority.
class PipelineCache::PipelineBuildWorkers {
public:
    PipelineBuildWorkers()
        : queue{CompileConcurrency(), std::max(1u, NumWorkers() / 4), CompileConcurrency()} {
        // PERF-102: extra workers precompile on the splash screen (the queue's boost), but keep
        // concurrent driver pipeline compilations bounded. Large bursts can overwhelm the driver
        // while the game and its pipeline linker are already using the GPU.
        for (u32 i = 0; i < AllCores(); ++i) {
            threads.emplace_back([this](std::stop_token stop) { Work(stop); });
        }
    }
    ~PipelineBuildWorkers() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        queue.NotifyAll();
    }

    void Push(std::function<void()>&& job, bool urgent) {
        queue.Push(std::move(job), urgent ? BuildQueue::Kind::Urgent : BuildQueue::Kind::Normal);
    }
    void PushBackground(std::function<void()>&& job) {
        queue.Push(std::move(job), BuildQueue::Kind::Background);
    }
    void PauseBackground(BuildQueue::Clock::time_point until) {
        queue.PauseBackground(until);
    }
    void SetBoost(bool on) {
        queue.SetBoost(on);
    }
    size_t BackgroundQueued() {
        return queue.BackgroundQueued();
    }
    bool ForegroundIdle() {
        return queue.ForegroundIdle();
    }

private:
    static u32 AllCores() {
        return std::max(std::thread::hardware_concurrency(), NumWorkers());
    }

    static u32 NumWorkers() {
        const u32 cores = std::max(std::thread::hardware_concurrency(), 4u);
        // PERF-021: all but two cores (the command thread and the game's own main thread).
        return std::clamp(cores - 2, 2u, 14u);
    }

    static u32 CompileConcurrency() {
        // Keep driver pipeline creation to a small burst even on large CPUs. Four concurrent
        // builds still use the parallel compiler without sending a dozen expensive requests to
        // the graphics driver at once.
        return std::clamp(AllCores() / 3, 2u, 4u);
    }

    void Work(std::stop_token stop) {
        Common::SetCurrentThreadName("shadGT:PipelineBuild");
        // PERF-097: driver compiles stay off the cores the game and the recorder run on.
        Common::PreferEfficiencyCores();
        bool low_priority = false;
        bool all_cores = false;
        bool backlog = false;
        // PERF-160: when new pipelines pile up (a new menu, the first seconds on a new track), the
        // workers may use every core, at low priority so the game's own threads keep theirs; on
        // Intel hybrid CPUs they otherwise stay on the efficiency cores (PERF-097), where a pile of
        // compiles took several times longer. -DisablePerf 160 keeps them there.
        static const bool backlog_cores = Common::PerfFeatureEnabled(160);
        while (true) {
            BuildQueue::Job job;
            bool background = false;
            if (!queue.Pop(job, background, stop)) {
                return;
            }
            // PERF-102: while the game waits on the splash screen for the precompile, every core
            // builds (and at normal priority); afterwards only the efficiency cores.
            const bool boosted = queue.Boosted();
            if (backlog_cores) {
                const size_t pending = queue.Pending();
                if (pending >= 16) {
                    backlog = true;
                } else if (pending <= 2) {
                    backlog = false;
                }
            }
            const bool wide = boosted || backlog;
            if (wide != all_cores) {
                all_cores = wide;
                if (all_cores) {
                    Common::AllowAllCores();
                } else {
                    Common::PreferEfficiencyCores();
                }
            }
            const bool low = !boosted && (background || backlog);
            if (low != low_priority) {
                low_priority = low;
                Common::SetCurrentThreadPriority(low_priority ? Common::ThreadPriority::Low
                                                              : Common::ThreadPriority::Normal);
            }
            job();
            if (background) {
                queue.FinishedBackground();
            } else {
                queue.FinishedForeground();
            }
        }
    }

    BuildQueue queue;
    std::vector<std::jthread> threads;
};

// PERF-019: what a worker needs to build one graphics pipeline. The shader infos are copied
// (with their user data) because the command thread keeps changing the cached ones.
struct PipelineCache::PipelineBuild {
    GraphicsPipelineKey key{};
    std::array<Shader::Info, MaxShaderStages> info_copies{};
    std::array<std::vector<u32>, MaxShaderStages> user_data{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    GraphicsPipeline::SerializationSupport sdata{};
    std::unique_ptr<GraphicsPipeline> pipeline;
    std::promise<void> promise;
    std::shared_future<void> done;
    std::chrono::steady_clock::time_point queued;
    bool urgent{};
    // PERF-038: a stored pipeline built in the background (no runtime shader state needed).
    bool preloading{};
    // PERF-076: a new pipeline its draws skip until a worker has built it.
    bool async{};
    // Whoever sets this first builds the pipeline: a worker, or the command thread when a
    // draw needs it before a worker got to it.
    std::atomic<bool> started{};
    // The cached infos the copies were made from; the pipeline points at them once built.
    std::array<const Shader::Info*, MaxShaderStages> canonical{};
};

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

/// SHD-004: a color target as far as a pixel shader's translation reads it: the export format,
/// the number class of the output, the conversion, which source channel each channel takes (a
/// constant channel, as formats without alpha have, takes its own, as the translation does) and
/// the blend emulation. A target the draw doesn't bind is translated as a float target with the
/// usual packed export (its writes go nowhere: the pass has no attachment there), so a pixel
/// shader translates the same into passes with one target or two and into float targets of other
/// formats (two in five of GT Sport's pixel shader translations differed only in that).
static Shader::PsColorBuffer TranslationColorBuffer(const Shader::PsColorBuffer& cb,
                                                    bool unorm_fixup) {
    if (unorm_fixup || cb.blend_self_scale || cb.blend_swizzled_alpha ||
        cb.blend_swizzled_factors) {
        return cb; // these read the target's format too
    }
    Shader::PsColorBuffer out{};
    out.data_format = AmdGpu::DataFormat::Format16_16_16_16;
    if (cb.data_format == AmdGpu::DataFormat::FormatInvalid) {
        out.num_format = AmdGpu::NumberFormat::Float;
        out.export_format = AmdGpu::ShaderExportFormat::ABGR_FP16;
        out.swizzle = AmdGpu::IdentityMapping;
        return out;
    }
    switch (AmdGpu::GetNumberClass(cb.num_format)) {
    case AmdGpu::NumberClass::Uint:
        out.num_format = AmdGpu::NumberFormat::Uint;
        break;
    case AmdGpu::NumberClass::Sint:
        out.num_format = AmdGpu::NumberFormat::Sint;
        break;
    default:
        out.num_format = AmdGpu::NumberFormat::Float;
        break;
    }
    out.num_conversion = cb.num_conversion;
    out.export_format = cb.export_format;
    for (u32 i = 0; i < 4; ++i) {
        out.swizzle.array[i] =
            AmdGpu::CompSwizzle(u32(AmdGpu::CompSwizzle::Red) + cb.swizzle.Map(i));
    }
    return out;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(HwStage stage, SwStage l_stage) {
    auto& info = S().runtime_infos[u32(l_stage)];
    const auto& regs = Regs();
    const auto BuildCommon = [&](const auto& program) {
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = S().infos[u32(SwStage::TessellationControl)];
        if (hull_info) { // FIX-058: no hull info when its draw is skipped
            hull_info->ReadTessConstantBuffer(tess_constants);
        }
        info.hw.ls.ls_stride = tess_constants.ls_stride;
        break;
    }
    case HwStage::Hull:
        BuildCommon(regs.hs_program);
        break;
    case HwStage::Export:
        BuildCommon(regs.es_program);
        info.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        break;
    case HwStage::Geometry: {
        BuildCommon(regs.gs_program);
        info.hw.gs.num_outputs = MapOutputs(info.hw.gs.outputs, regs.vs_output_control);
        info.hw.gs.output_vertices = regs.vgt_gs_max_vert_out;
        info.hw.gs.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            info.hw.gs.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            info.hw.gs.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            info.hw.gs.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        info.hw.gs.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        info.hw.gs.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        info.hw.gs.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        info.hw.gs.vs_copy = params_vc.code;
        info.hw.gs.vs_copy_hash = params_vc.hash;
        DumpShader(info.hw.gs.vs_copy, info.hw.gs.vs_copy_hash, Shader::HwStage::Vertex, 0,
                   "copy.bin");
        break;
    }
    case HwStage::Vertex: {
        BuildCommon(regs.vs_program);
        info.hw.vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.hw.vs.num_outputs = MapOutputs(info.hw.vs.outputs, regs.vs_output_control);
        info.hw.vs.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.hw.vs.clip_disable = regs.IsClipDisabled();
        break;
    }
    case HwStage::Fragment: {
        BuildCommon(regs.ps_program);
        info.hw.fs.en_flags = regs.ps_input_ena;
        info.hw.fs.addr_flags = regs.ps_input_addr;
        info.hw.fs.num_inputs = regs.num_interp;
        info.hw.fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
        info.hw.fs.quad_sample_mask =
            regs.reg_array[0xa30e] != ~0u || regs.reg_array[0xa30f] != ~0u;
        info.hw.fs.num_samples =
            regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
                ? regs.aa_config.NumSamples()
                : 1;
        info.hw.fs.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.hw.fs.mrtz_mask = regs.depth_shader_control.z_export_enable |
                               (stencil_ref_export_enable << 1) |
                               (regs.depth_shader_control.mask_export_enable << 2) |
                               (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.hw.fs.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.hw.fs.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.hw.fs.dual_source_blending = false;
        }
        info.hw.fs.dual_source_blending |= S().graphics_key.color_buffers[0].blend_swizzled_alpha ||
                                           S().graphics_key.color_buffers[0].blend_swizzled_factors;
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.hw.fs.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.hw.fs.color_buffers[i] = TranslationColorBuffer(
                S().graphics_key.color_buffers[i], profile.needs_unorm_fixup); // SHD-004
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry));
        info.hw.fs.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case HwStage::Compute: {
        const auto& cs_pgm = liverpool->DrawCsRegs();
        info.props.num_user_data = cs_pgm.settings.num_user_regs;
        info.props.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs_pgm.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs_pgm.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs_pgm.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs_pgm.settings.fp_round_mode64;
        info.hw.cs.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                     cs_pgm.num_thread_z.full};
        info.hw.cs.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                  cs_pgm.IsTgidEnabled(2)};
        info.hw.cs.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    switch (l_stage) {
    case SwStage::Vertex:
        info.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.sw.vs.vertex_sgpr_offset = S().draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = S().draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            (regs.primitive_type == AmdGpu::PrimitiveType::QuadList && !QuadListAsTriangles());
        break;
    case SwStage::TessellationControl: {
        info.sw.tcs.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.sw.tcs.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.sw.tcs.tess_type = regs.tess_config.type;
        info.sw.tcs.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        break;
    }
    case SwStage::TessellationEval: {
        info.sw.tes.tess_type = regs.tess_config.type;
        info.sw.tes.tess_topology = regs.tess_config.topology;
        info.sw.tes.tess_partitioning = regs.tess_config.partitioning;
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes} {
    if (liverpool) {
        liverpool->on_command_buffer_start = [this] { ReadAheadAtBufferStart(); };
    }
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance_.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    // D2: read-only guest buffers are read in place through the direct-memory page table.
    profile.direct_buffers =
        VideoCore::DirectMemory::Enabled() && instance.IsExternalMemoryHostSupported();
    CreateDriverCache();
    LoadElseScopeFixed();
    // PERF-032: the shader store is used again, without the startup precompile Lance had removed:
    // a program's stored translations are read when a draw first uses it, and new translations
    // are stored. -DisablePerf 47 leaves the store closed.
    if (EmulatorSettings.IsPipelineCacheEnabled() && Common::PerfFeatureEnabled(47)) {
        if (EmulatorSettings.IsPipelineCacheArchived()) {
            LOG_WARNING(Render_Vulkan, "PERF-032: the shader store is not used with an archived "
                                       "pipeline cache (pipeline_cache_archived)");
        } else {
            Storage::DataBase::Instance().Open();
            use_stored_shaders = true;
            // D10: the old renderer's translations stand in while the new renderer's are built.
            // SHADGT_LEGACY_FALLBACK=0 turns it off.
            // SHD-002: not with stage descriptor sets: the old translations bind everything
            // in set 0.
            if (profile.direct_buffers && false) {
                const char* fallback_env = std::getenv("SHADGT_LEGACY_FALLBACK");
                if (!fallback_env || fallback_env[0] != '0') {
                    Storage::DataBase::Legacy().OpenLegacy();
                    legacy_fallback = Storage::DataBase::Legacy().IsOpened();
                    legacy_stats.since = std::chrono::steady_clock::now();
                    LOG_WARNING(Render_Vulkan, "D10: old-renderer shader fallback {}",
                                legacy_fallback ? "on" : "off (no old shader store)");
                }
            }
            LOG_WARNING(Render_Vulkan,
                        "PERF-032: shader translations are loaded from the store on first use");
            // PERF-038: opt-in (SHADGT_PREWARM=1): the stored pipelines are built in the
            // background while the game runs, most recently used first. -DisablePerf 52. ryGT
            // turns it on unless SHADGT_PREWARM=0, so GTSport.exe runs without a launcher.
            const char* prewarm = std::getenv("SHADGT_PREWARM");
#ifdef RYTY_GPU
            // PERF-098: GPU.precompile_shaders (on by default) unless the variable says otherwise.
            const bool prewarm_on =
                prewarm ? prewarm[0] != '0' : EmulatorSettings.IsPrecompileShaders();
#else
            const bool prewarm_on = prewarm && prewarm[0] == '1';
#endif
            if (prewarm_on && Common::PerfFeatureEnabled(52)) {
                if (!instance.IsVertexInputDynamicState()) {
                    LOG_WARNING(Render_Vulkan, "PERF-038: background pipeline builds need dynamic "
                                               "vertex input; not available here");
                } else {
                    prewarm_enabled = true;
                    prewarm_stats.since = std::chrono::steady_clock::now();
                    prewarm_reader =
                        std::jthread{[this](std::stop_token stop) { PrewarmRead(stop); }};
                }
            }
        }
    }
    if (CompileMode() == 1 && AsyncPipelines()) {
        static std::jthread slow_motion{SlowMotionThread}; // PERF-103
    }
    // PB-1 (SHADGT_REPLAY_TEST=<folder of <hash>.bin programs>): translates every stored shader
    // again from its program and its stored record, and counts how many match the stored SPIR-V.
    if (const char* codes = std::getenv("SHADGT_REPLAY_TEST"); codes && *codes) {
        static std::jthread replay_test{[this, dir = std::string{codes}] { ReplayTest(dir); }};
    }
    if (const char* dir = std::getenv("SHADGT_TEMPLATE_TEST"); dir && *dir) {
        static std::jthread template_test{[this, d = std::string{dir}] { TemplateTest(d); }};
    }
    if (const char* dir = std::getenv("SHADGT_DUMP_STORE"); dir && *dir) {
        static std::jthread dump_store{[this, d = std::string{dir}] { DumpStore(d); }};
    }
    // FIX-078: a quarter of the hardware threads (at least 2) translate read-ahead shaders.
    if (Common::PerfFeatureEnabled(102)) {
        const u32 count = std::max(2u, std::thread::hardware_concurrency() / 4);
        translate_workers =
            std::make_unique<TranslateWorkers>(profile, instance.GetDevice(), count);
        translate_stats.since = std::chrono::steady_clock::now();
    }
}

PipelineCache::~PipelineCache() {
    if (liverpool) {
        liverpool->on_command_buffer_start = nullptr;
    }
    SaveDriverCache(true);
}

void PipelineCache::CreateDriverCache() {
    std::vector<u8> data;
    if (EmulatorSettings.IsPipelineCacheEnabled()) {
        const auto& serial = Common::Singleton<Common::ElfInfo>::Instance()->GameSerial();
        driver_cache_path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                            fmt::format("{}.vkpipelinecache", serial);
        const Common::FS::IOFile file{driver_cache_path, Common::FS::FileAccessMode::Read};
        if (file.IsOpen()) {
            data.resize(file.GetSize());
            if (file.Read(data) != data.size()) {
                data.clear();
            }
        }
        // Header (Vulkan spec, VkPipelineCacheHeaderVersionOne): size, version, vendor,
        // device, UUID. Data from another GPU or driver starts an empty cache.
        const auto props = instance.GetPhysicalDevice().getProperties();
        struct Header {
            u32 size;
            u32 version;
            u32 vendor_id;
            u32 device_id;
            std::array<u8, VK_UUID_SIZE> uuid;
        } header{};
        if (data.size() < sizeof(Header)) {
            data.clear();
        } else {
            std::memcpy(&header, data.data(), sizeof(Header));
            if (header.version != static_cast<u32>(vk::PipelineCacheHeaderVersion::eOne) ||
                header.vendor_id != props.vendorID || header.device_id != props.deviceID ||
                std::memcmp(header.uuid.data(), props.pipelineCacheUUID.data(), VK_UUID_SIZE)) {
                LOG_INFO(Render_Vulkan, "Driver pipeline cache is from another GPU or driver");
                data.clear();
            }
        }
    }
    auto [cache_result, cache] =
        instance.GetDevice().createPipelineCacheUnique(vk::PipelineCacheCreateInfo{
            .initialDataSize = data.size(),
            .pInitialData = data.data(),
        });
    if (cache_result != vk::Result::eSuccess && !data.empty()) {
        LOG_WARNING(Render_Vulkan, "Driver pipeline cache rejected ({}), starting empty",
                    vk::to_string(cache_result));
        auto [empty_result, empty_cache] = instance.GetDevice().createPipelineCacheUnique({});
        cache_result = empty_result;
        cache = std::move(empty_cache);
    }
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
    pipelines_at_driver_save = num_new_pipelines;
    driver_cache_saved_at = std::chrono::steady_clock::now();
    LOG_INFO(Render_Vulkan, "Driver pipeline cache: loaded {} bytes", data.size());
}

void PipelineCache::SaveDriverCache(bool wait) {
    if (driver_cache_path.empty() || !pipeline_cache) {
        return;
    }
    // PERF-036: GT Sport's driver cache grows to several GB (3.4 GB on October 9). Reading it out
    // with vkGetPipelineCacheData took about 1.1 s on the GPU thread every 30 s, a frozen frame
    // each time. The read and the write now run on their own thread; a save still running is
    // not waited for, the next one comes later. Pipeline caches are internally synchronized, so
    // builds go on meanwhile. -DisablePerf 51 reads it here as before.
    static const bool off_thread = Common::PerfFeatureEnabled(51);
    if (!wait && driver_cache_writing.load()) {
        return;
    }
    if (driver_cache_writer.joinable()) {
        driver_cache_writer.join();
    }
    pipelines_at_driver_save = num_new_pipelines;
    driver_cache_saved_at = std::chrono::steady_clock::now();
    const auto write = [path = driver_cache_path](std::vector<u8> data) {
        // Write beside the file and rename, so an interrupted write never leaves a broken cache.
        auto temp = path;
        temp += ".tmp";
        {
            const Common::FS::IOFile file{temp, Common::FS::FileAccessMode::Create};
            if (!file.IsOpen() || file.Write(data) != data.size()) {
                return;
            }
        }
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
    };
    const vk::Device device = instance.GetDevice();
    const vk::PipelineCache cache = *pipeline_cache;
    if (off_thread) {
        driver_cache_writing = true;
        driver_cache_writer = std::jthread([this, device, cache, write] {
            Common::SetCurrentThreadName("shadGT:DriverCacheSave");
            const auto start = std::chrono::steady_clock::now();
            auto [result, data] = device.getPipelineCacheData(cache);
            const auto read_end = std::chrono::steady_clock::now();
            if (result == vk::Result::eSuccess && !data.empty()) {
                const size_t size = data.size();
                write(std::move(data));
                LOG_WARNING(Render_Vulkan,
                            "PERF-036: driver pipeline cache saved off the GPU thread: {} MB, "
                            "read {:.0f} ms, written {:.0f} ms",
                            size >> 20,
                            std::chrono::duration<double, std::milli>(read_end - start).count(),
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - read_end)
                                .count());
            }
            driver_cache_writing = false;
        });
    } else {
        auto [result, data] = device.getPipelineCacheData(cache);
        if (result != vk::Result::eSuccess || data.empty()) {
            return;
        }
        driver_cache_writing = true;
        driver_cache_writer = std::jthread([this, write, data = std::move(data)]() mutable {
            write(std::move(data));
            driver_cache_writing = false;
        });
    }
    if (wait) {
        driver_cache_writer.join();
    }
}

void PipelineCache::MaybeSaveDriverCache() {
    // New pipelines are compiled during play; save them in batches without blocking the frame.
    using namespace std::chrono_literals;
    if (num_new_pipelines - pipelines_at_driver_save >= 64 &&
        std::chrono::steady_clock::now() - driver_cache_saved_at >= 30s) {
        SaveDriverCache(false);
    }
}

const AmdGpu::Regs& PipelineCache::Regs() const {
    return regs_override ? *regs_override : liverpool->DrawRegs();
}

bool PipelineCache::HasSupportedColorTargets() const {
    if (S().graphics_key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        return false;
    }
    for (u32 cb = 0; cb < S().graphics_key.num_color_attachments; ++cb) {
        const auto& color_buffer = S().graphics_key.color_buffers[cb];
        if (color_buffer.data_format != AmdGpu::DataFormat::FormatInvalid &&
            Vulkan::LiverpoolToVK::TrySurfaceFormat(
                color_buffer.data_format, color_buffer.num_format) == vk::Format::eUndefined) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<PipelineCache::PipelineBuild> PipelineCache::MakePipelineBuild() {
    auto build = std::make_shared<PipelineBuild>();
    build->key = S().graphics_key;
    for (size_t stage = 0; stage < MaxShaderStages; ++stage) {
        if (!S().infos[stage]) {
            continue;
        }
        auto& copy = build->info_copies[stage];
        copy = *S().infos[stage];
        build->user_data[stage].assign(copy.user_data.begin(), copy.user_data.end());
        copy.user_data = build->user_data[stage];
        build->infos[stage] = &copy;
        build->canonical[stage] = S().infos[stage];
    }
    build->runtime_infos = S().runtime_infos;
    if (S().fetch_shader) {
        build->fetch = *S().fetch_shader;
    }
    build->modules = S().modules;
    build->done = build->promise.get_future().share();
    build->queued = std::chrono::steady_clock::now();
    return build;
}

void PipelineCache::FinishBuild(PipelineBuild& build) {
    // FIX-044: the GPU thread waiting for (or doing) a pipeline build holds the guest clocks.
    Common::GuestClock::CompilePause hold;
    if (!build.started.exchange(true)) {
        build.pipeline = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, build.key, *pipeline_cache, build.infos,
            build.runtime_infos, build.fetch ? &*build.fetch : nullptr, build.modules, build.sdata,
            build.preloading);
        // The copies were only needed while building; draws use the cached infos.
        build.pipeline->SetStageInfos(build.canonical);
        build.info_copies = {};
        build.user_data = {};
        build.promise.set_value();
    }
    build.done.wait();
}

void PipelineCache::QueueBuild(const std::shared_ptr<PipelineBuild>& build, bool urgent) {
    if (!build_workers) {
        build_workers = std::make_unique<PipelineBuildWorkers>();
    }
    auto job = [this, build] {
        if (!build->started.load()) {
            FinishBuild(*build);
        }
    };
    build_workers->Push(std::move(job), urgent);
}

std::shared_ptr<PipelineCache::PipelineBuild> PipelineCache::StartPipelineBuild(bool urgent) {
    auto build = MakePipelineBuild();
    build->urgent = urgent;
    QueueBuild(build, urgent);
    return build;
}

// PB-1: one stored shader translated again outside the game, from its program's bytes and the
// shader record. False when it could not be translated.
bool ReplayTranslate(const Shader::Profile& profile, Shader::Pools& pools,
                     const std::vector<u8>& program, const Shader::Info& stored,
                     const Shader::StageSpecialization& spec, std::vector<u32>& spv,
                     const std::function<void(const Shader::Info&)>& inspect = {}) {
    if (program.size() < 64 || stored.flattened_ud_buf.size() < 16) {
        return false;
    }
    const u8* footer = program.data() + program.size() - 32;
    u32 bits;
    std::memcpy(&bits, footer + 8, 4);
    const u32 length = bits >> 8;
    if (length == 0 || length > program.size() - 32) {
        return false;
    }
    std::vector<u32> code((program.size() + 3) / 4);
    std::memcpy(code.data(), program.data(), program.size());
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    std::copy_n(stored.flattened_ud_buf.begin(), user_data.size(), user_data.begin());
    const Shader::ShaderParams params{
        .user_data = std::span<const u32, Shader::ShaderParams::NumShaderUserData>{user_data},
        .code = std::span<const u32>{code.data(), length / sizeof(u32)},
        .hash = stored.pgm_hash,
    };
    Shader::Info info(stored.hw_stage, stored.sw_stage, params);
    // The vertex attribute V#s, from what the record kept of them (number class, swizzle).
    std::vector<std::array<u32, 4>> sharps;
    for (const auto& attrib : spec.vs_attribs) {
        AmdGpu::Buffer buffer{};
        buffer.base_address = 0x1000;
        buffer.num_records = 1;
        buffer.data_format = 14; // 32_32_32_32
        buffer.num_format = attrib.num_class == AmdGpu::NumberClass::Sint   ? 5
                            : attrib.num_class == AmdGpu::NumberClass::Uint ? 4
                                                                            : 7;
        buffer.dst_sel_x = u32(attrib.dst_select.r);
        buffer.dst_sel_y = u32(attrib.dst_select.g);
        buffer.dst_sel_z = u32(attrib.dst_select.b);
        buffer.dst_sel_w = u32(attrib.dst_select.a);
        auto& out = sharps.emplace_back();
        std::memcpy(out.data(), &buffer, sizeof(buffer));
    }
    // Tessellation constants: the parts translation uses are in the stored runtime info.
    Shader::TessellationDataConstantBuffer tess{};
    if (stored.sw_stage == Shader::SwStage::TessellationControl) {
        tess.ls_stride = spec.runtime_info.sw.tcs.ls_stride;
        tess.hs_cp_stride = spec.runtime_info.sw.tcs.hs_output_cp_stride;
        tess.hs_output_base = spec.runtime_info.sw.tcs.hs_output_base;
    } else if (stored.sw_stage == Shader::SwStage::TessellationEval) {
        tess.hs_cp_stride = spec.runtime_info.sw.tes.hs_output_cp_stride;
    }
    const Shader::ReplayInputs replay{
        .flat = &stored.flattened_ud_buf,
        .fetch = spec.fetch_shader_data.Empty() ? nullptr : &spec.fetch_shader_data,
        .attrib_sharps = &sharps,
        .tess_constants = &tess,
    };
    Shader::t_replay = &replay;
    struct Reset {
        ~Reset() {
            Shader::t_replay = nullptr;
        }
    } reset;
    auto runtime_info = spec.runtime_info;
    auto binding = spec.start;
    try {
        Common::RecoverableScope recoverable;
        const auto ir = Shader::TranslateProgram(params.code, pools, info, runtime_info, profile);
        spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, binding);
        if (inspect) {
            // PB-4: the translation's resources (their sharps come from the record's user data).
            inspect(info);
        }
    } catch (...) {
        return false;
    }
    return true;
}

// PB-1: shader records without their SPIR-V (put in the store by RyGT's Set up, with the
// programs they were made from in <cache>\<serial>-programs) are translated here, on this PC,
// before the stored pipelines are built.
void PipelineCache::RebuildFromPrograms() {
    const auto& serial = Common::ElfInfo::Instance().GameSerial();
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                     (std::string{serial} + "-programs");
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return;
    }
    auto& store = Storage::DataBase::Instance();
    const auto start = std::chrono::steady_clock::now();
    const auto metas = store.ListBlobs(Storage::BlobType::ShaderMeta);
    std::atomic<u64> rebuilt = 0, failed = 0, no_program = 0;
    std::atomic<size_t> next = 0;
    // PERF-161: translated on several threads (a first boot with RyGT's shader pack translated
    // 12,590 shaders in 36 s on one). -DisablePerf 161 uses one.
    static const bool parallel = Common::PerfFeatureEnabled(161);
    const u32 threads =
        parallel ? std::clamp(std::thread::hardware_concurrency() - 2, 1u, 12u) : 1u;
    const auto work = [&] {
        Shader::Pools pools;
        for (size_t i = next++; i < metas.size(); i = next++) {
            const auto& name = metas[i].first;
            std::vector<u8> blob;
            store.Load(Storage::BlobType::ShaderMeta, name, blob);
            Shader::Info info{};
            Shader::StageSpecialization spec{};
            spec.info = &info;
            size_t perm_idx{};
            Serialization::Archive ar{std::move(blob)};
            try {
                Common::RecoverableScope recoverable;
                if (ar.SizeBytes() < 24 || !LoadShaderMeta(ar, info, spec, perm_idx)) {
                    continue;
                }
            } catch (...) {
                continue;
            }
            const auto spv_name = fmt::format("{:#018x}_{}", info.pgm_hash, perm_idx);
            if (store.Exists(Storage::BlobType::ShaderBinary, spv_name)) {
                continue;
            }
            std::vector<u8> program;
            if (Common::FS::IOFile file{dir / fmt::format("{:016x}.bin", info.pgm_hash),
                                        Common::FS::FileAccessMode::Read};
                file.IsOpen()) {
                program.resize(file.GetSize());
                file.ReadSpan(std::span{program});
            }
            if (program.empty()) {
                ++no_program;
                continue;
            }
            std::vector<u32> spv;
            if (!ReplayTranslate(profile, pools, program, info, spec, spv)) {
                ++failed;
                continue;
            }
            RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);
            ++rebuilt;
        }
    };
    {
        std::vector<std::jthread> workers;
        for (u32 t = 1; t < threads; ++t) {
            workers.emplace_back([&] {
                Common::SetCurrentThreadName("shadGT:PackTranslate");
                work();
            });
        }
        work();
    }
    if (rebuilt + failed + no_program != 0) {
        LOG_WARNING(Render_Vulkan,
                    "PB-1: {} shaders translated from this PC's game files in {:.1f} s on {} "
                    "threads ({} failed, {} without their program)",
                    rebuilt.load(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                    threads, failed.load(), no_program.load());
    }
}

// PB-2 (SHADGT_TEMPLATE_TEST=<folder>): for the programs in <folder>\unseen.txt, how many of
// their stored records have the same specialization as a record of another program (a
// "template" an unseen program could borrow), and for pipelines of two unseen programs, how far
// apart the vertex and fragment programs sit in their file (<folder>\slots.tsv: hash, path,
// slot). Results in the log.
void PipelineCache::TemplateTest(const std::string& dir) {
    Common::SetCurrentThreadName("shadGT:TemplateTest");
    std::unordered_set<u64> unseen;
    {
        std::ifstream in{std::filesystem::path{dir} / "unseen.txt"};
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) unseen.insert(std::stoull(line, nullptr, 16));
        }
    }
    std::unordered_map<u64, std::pair<std::string, int>> slots;
    {
        std::ifstream in{std::filesystem::path{dir} / "slots.tsv"};
        std::string h, path, slot;
        while (std::getline(in, h, '\t') && std::getline(in, path, '\t') && std::getline(in, slot)) {
            slots[std::stoull(h, nullptr, 16)] = {path, std::stoi(slot)};
        }
    }
    struct Record {
        u64 pgm;
        std::unique_ptr<Shader::Info> info;
        Shader::StageSpecialization spec;
    };
    std::vector<Record> records;
    std::unordered_map<u64, u64> perm_to_pgm;
    auto& store = Storage::DataBase::Instance();
    for (const auto& [name, time] : store.ListBlobs(Storage::BlobType::ShaderMeta)) {
        std::vector<u8> blob;
        store.Load(Storage::BlobType::ShaderMeta, name, blob);
        Record r;
        r.info = std::make_unique<Shader::Info>();
        r.spec.info = r.info.get();
        size_t perm_idx{};
        Serialization::Archive ar{std::move(blob)};
        try {
            Common::RecoverableScope recoverable;
            if (ar.SizeBytes() < 24 || !LoadShaderMeta(ar, *r.info, r.spec, perm_idx)) continue;
        } catch (...) {
            continue;
        }
        r.spec.info = r.info.get();
        r.pgm = r.info->pgm_hash;
        perm_to_pgm[std::stoull(name, nullptr, 16)] = r.pgm;
        records.push_back(std::move(r));
    }
    u64 tried = 0, matched = 0, by_stage_tried[8]{}, by_stage_matched[8]{};
    std::map<std::string, u64> reasons;
    for (const auto& p : records) {
        if (!unseen.contains(p.pgm)) continue;
        ++tried;
        const u32 stage = std::min<u32>(u32(p.info->sw_stage), 7);
        ++by_stage_tried[stage];
        bool found = false;
        const char* best_reason = "no record of that stage";
        for (const auto& t : records) {
            if (unseen.contains(t.pgm) || t.info->sw_stage != p.info->sw_stage) continue;
            if (t.spec == p.spec) {
                found = true;
                break;
            }
            u32 index{};
            if (const char* why = t.spec.FirstDifference(p.spec, index)) best_reason = why;
        }
        if (found) {
            ++matched;
            ++by_stage_matched[stage];
        } else {
            ++reasons[best_reason];
        }
    }
    std::string why_text;
    for (const auto& [why, n] : reasons) why_text += fmt::format(" {}={}", why, n);
    LOG_WARNING(Render_Vulkan,
                "PB-2 templates: {} records of unseen programs, {} have the specialization of a "
                "record of another program; by stage (same/all) fs {}/{} vs {}/{} cs {}/{}; last "
                "difference of the others:{}",
                tried, matched, by_stage_matched[0], by_stage_tried[0], by_stage_matched[3],
                by_stage_tried[3], by_stage_matched[5], by_stage_tried[5], why_text);
    // Vertex/fragment pairs of unseen programs: their distance in the file.
    std::map<int, u64> distance;
    u64 pairs = 0, other_file = 0, no_slot = 0, fs_unseen_pipelines = 0, fs_unseen_vs_known = 0;
    std::unordered_map<u64, u64> known_vs;
    for (const auto& [name, time] : store.ListBlobs(Storage::BlobType::PipelineKey)) {
        if (!name.starts_with("g_")) continue;
        std::vector<u8> blob;
        store.Load(Storage::BlobType::PipelineKey, name, blob);
        GraphicsPipelineKey key{};
        GraphicsPipeline::SerializationSupport sdata{};
        try {
            Common::RecoverableScope recoverable;
            if (!LoadStoredGraphicsPipeline(std::move(blob), key, sdata)) continue;
        } catch (...) {
            continue;
        }
        const auto vs = perm_to_pgm.find(key.stage_hashes[u32(Shader::SwStage::Vertex)]);
        const auto fs = perm_to_pgm.find(key.stage_hashes[u32(Shader::SwStage::Fragment)]);
        if (vs != perm_to_pgm.end() && fs != perm_to_pgm.end() && unseen.contains(fs->second)) {
            ++fs_unseen_pipelines;
            if (!unseen.contains(vs->second)) {
                ++fs_unseen_vs_known;
                known_vs[vs->second]++;
            }
        }
        if (vs == perm_to_pgm.end() || fs == perm_to_pgm.end() || !unseen.contains(vs->second) ||
            !unseen.contains(fs->second)) {
            continue;
        }
        ++pairs;
        const auto a = slots.find(vs->second);
        const auto b = slots.find(fs->second);
        if (a == slots.end() || b == slots.end()) {
            ++no_slot;
        } else if (a->second.first != b->second.first) {
            ++other_file;
        } else {
            ++distance[std::clamp(b->second.second - a->second.second, -9, 9)];
        }
    }
    std::string dist_text;
    for (const auto& [d, n] : distance) dist_text += fmt::format(" {}:{}", d, n);
    LOG_WARNING(Render_Vulkan,
                "PB-2 pairs: {} pipelines of two unseen programs; {} in different files, {} "
                "without a slot; fragment slot minus vertex slot (count):{}",
                pairs, other_file, no_slot, dist_text);
    // Simulation of the loading-screen rule: a pipeline of an unseen fragment program P and a
    // known vertex permutation V is predicted when a known pipeline with the same V and the same
    // state (everything but the fragment stage) has a fragment record whose specialization is
    // P's. Counts how many of the real pipelines would have been predicted, and how many
    // candidates the rule makes.
    {
        std::unordered_map<u64, const Record*> rec_by_perm;
        for (const auto& [perm, pgm] : perm_to_pgm) {
            (void)pgm;
        }
        // perm hash -> record (records are in ListBlobs order; rebuild a map from names)
        std::unordered_map<u64, size_t> perm_index;
        {
            size_t i = 0;
            for (const auto& [name, time] : store.ListBlobs(Storage::BlobType::ShaderMeta)) {
                (void)time;
                if (i < records.size()) {
                    perm_index[std::stoull(name, nullptr, 16)] = i;
                }
                ++i;
            }
        }
        struct Known {
            GraphicsPipelineKey key;
        };
        std::unordered_map<u64, std::vector<GraphicsPipelineKey>> known_by_vs;
        std::vector<GraphicsPipelineKey> wanted;
        const u32 vs_i = u32(Shader::SwStage::Vertex), fs_i = u32(Shader::SwStage::Fragment);
        for (const auto& [name, time] : store.ListBlobs(Storage::BlobType::PipelineKey)) {
            if (!name.starts_with("g_")) continue;
            std::vector<u8> blob;
            store.Load(Storage::BlobType::PipelineKey, name, blob);
            GraphicsPipelineKey key{};
            GraphicsPipeline::SerializationSupport sdata{};
            try {
                Common::RecoverableScope recoverable;
                if (!LoadStoredGraphicsPipeline(std::move(blob), key, sdata)) continue;
            } catch (...) {
                continue;
            }
            const auto vs = perm_to_pgm.find(key.stage_hashes[vs_i]);
            const auto fs = perm_to_pgm.find(key.stage_hashes[fs_i]);
            if (vs == perm_to_pgm.end() || fs == perm_to_pgm.end() || unseen.contains(vs->second)) {
                continue;
            }
            if (unseen.contains(fs->second)) {
                wanted.push_back(key);
            } else {
                known_by_vs[key.stage_hashes[vs_i]].push_back(key);
            }
        }
        u64 predicted = 0, candidates = 0;
        for (const auto& want : wanted) {
            const auto it = known_by_vs.find(want.stage_hashes[vs_i]);
            if (it == known_by_vs.end()) continue;
            const auto pi = perm_index.find(want.stage_hashes[fs_i]);
            if (pi == perm_index.end() || pi->second >= records.size()) continue;
            const auto& p_spec = records[pi->second].spec;
            bool hit = false;
            for (const auto& known : it->second) {
                GraphicsPipelineKey a = want, b = known;
                a.stage_hashes[fs_i] = 0;
                b.stage_hashes[fs_i] = 0;
                if (!(a == b)) continue;
                ++candidates;
                const auto ti = perm_index.find(known.stage_hashes[fs_i]);
                if (ti != perm_index.end() && ti->second < records.size() &&
                    records[ti->second].spec == p_spec) {
                    hit = true;
                }
            }
            predicted += hit ? 1 : 0;
        }
        LOG_WARNING(Render_Vulkan,
                    "PB-2 prediction: {} pipelines of unseen fragment and known vertex programs; "
                    "{} predicted by a known pipeline with the same vertex permutation and state "
                    "and an equal fragment specialization ({} same-state candidates in all)",
                    wanted.size(), predicted, candidates);
    }
    std::vector<std::pair<u64, u64>> top(known_vs.begin(), known_vs.end());
    std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string top_text;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 8); ++i) {
        const auto s = slots.find(top[i].first);
        top_text += fmt::format(" {:#x}x{} ({})", top[i].first, top[i].second,
                                s != slots.end() ? s->second.first : "not in the files");
    }
    LOG_WARNING(Render_Vulkan,
                "PB-2 unseen fragment shaders: {} pipelines, {} with a vertex shader seen before "
                "({} different ones); most used:{}",
                fs_unseen_pipelines, fs_unseen_vs_known, known_vs.size(), top_text);
}

void PipelineCache::ReplayTest(const std::string& dir) {
    Common::SetCurrentThreadName("shadGT:ReplayTest");
    auto& store = Storage::DataBase::Instance();
    const auto metas = store.ListBlobs(Storage::BlobType::ShaderMeta);
    Shader::Pools pools;
    u64 tested = 0, same = 0, differ = 0, failed = 0, no_program = 0, by_stage_same[8]{}, by_stage_all[8]{};
    std::vector<std::string> identical, not_identical;
    for (const auto& [name, time] : metas) {
        std::vector<u8> blob;
        store.Load(Storage::BlobType::ShaderMeta, name, blob);
        Shader::Info info{};
        Shader::StageSpecialization spec{};
        spec.info = &info;
        size_t perm_idx{};
        Serialization::Archive ar{std::move(blob)};
        try {
            Common::RecoverableScope recoverable;
            if (ar.SizeBytes() < 24 || !LoadShaderMeta(ar, info, spec, perm_idx)) {
                continue;
            }
        } catch (...) {
            continue;
        }
        const auto path = std::filesystem::path{dir} / fmt::format("{:016x}.bin", info.pgm_hash);
        std::vector<u8> program;
        if (Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read}; file.IsOpen()) {
            program.resize(file.GetSize());
            file.ReadSpan(std::span{program});
        }
        if (program.empty()) {
            ++no_program;
            not_identical.push_back(fmt::format("{}\t{:016x}", name, info.pgm_hash));
            continue;
        }
        std::vector<u32> stored_spv;
        store.Load(Storage::BlobType::ShaderBinary, fmt::format("{:#018x}_{}", info.pgm_hash, perm_idx),
                   stored_spv);
        std::vector<u32> spv;
        ++tested;
        const u32 stage = std::min<u32>(u32(info.sw_stage), 7);
        ++by_stage_all[stage];
        if (!ReplayTranslate(profile, pools, program, info, spec, spv)) {
            ++failed;
            not_identical.push_back(fmt::format("{}\t{:016x}", name, info.pgm_hash));
        } else if (spv == stored_spv) {
            ++same;
            ++by_stage_same[stage];
            identical.push_back(fmt::format("{}\t{:016x}", name, info.pgm_hash));
        } else {
            ++differ;
            not_identical.push_back(fmt::format("{}\t{:016x}", name, info.pgm_hash));
            if (differ <= 10) {
                LOG_WARNING(Render_Vulkan, "PB-1: {} differs (stage {}, {} vs {} words)", name,
                            stage, spv.size(), stored_spv.size());
            }
        }
    }
    LOG_WARNING(Render_Vulkan,
                "PB-1 replay test: {} shaders tested, {} identical, {} different, {} failed, {} "
                "without a program file; identical by stage (sw stage: same/all): {}/{} {}/{} {}/{} "
                "{}/{} {}/{} {}/{}",
                tested, same, differ, failed, no_program, by_stage_same[0], by_stage_all[0],
                by_stage_same[1], by_stage_all[1], by_stage_same[2], by_stage_all[2],
                by_stage_same[3], by_stage_all[3], by_stage_same[4], by_stage_all[4],
                by_stage_same[5], by_stage_all[5]);
    // The records that translate back exactly, for RyGT's shader pack.
    std::ofstream list{Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "replay_identical.txt"};
    for (const auto& name : identical) {
        list << name << '\n';
    }
    // The others: a program is shipped only with all of its permutations (their indices are
    // the store's, and a gap made the renderer build one pipeline again and again).
    std::ofstream other{Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "replay_other.txt"};
    for (const auto& name : not_identical) {
        other << name << '\n';
    }
}

// PB-4 (SHADGT_DUMP_STORE=<folder>): writes the store's shader records and pipeline keys as
// text, for research on predicting them from the game's files: <folder>\metas.tsv (each record
// translated again from <folder>\programs\<hash>.bin, with every resource's sharp read from the
// record's user data) and <folder>\keys.tsv.
void PipelineCache::DumpStore(const std::string& dir) {
    Common::SetCurrentThreadName("shadGT:DumpStore");
    const auto hex = [](const void* data, size_t size) {
        std::string out;
        const auto* bytes = static_cast<const u8*>(data);
        for (size_t i = 0; i < size; ++i) {
            out += fmt::format("{:02x}", bytes[i]);
        }
        return out;
    };
    auto& store = Storage::DataBase::Instance();
    Shader::Pools pools;
    u64 written = 0, no_program = 0, failed = 0;
    {
        std::ofstream out{std::filesystem::path{dir} / "metas.tsv"};
        out << "name\tpgm\tperm\tstage\tflat_dw\truntime\tvs_attribs\tfetch\tbuffers\timages\tsamplers"
               "\tspec_buffers\tspec_images\tspec_samplers\n";
        for (const auto& [name, time] : store.ListBlobs(Storage::BlobType::ShaderMeta)) {
            std::vector<u8> blob;
            store.Load(Storage::BlobType::ShaderMeta, name, blob);
            Shader::Info stored{};
            Shader::StageSpecialization spec{};
            spec.info = &stored;
            size_t perm_idx{};
            Serialization::Archive ar{std::move(blob)};
            try {
                Common::RecoverableScope recoverable;
                if (ar.SizeBytes() < 24 || !LoadShaderMeta(ar, stored, spec, perm_idx)) continue;
            } catch (...) {
                continue;
            }
            std::string runtime;
            const auto& rt = spec.runtime_info;
            if (stored.sw_stage == Shader::SwStage::Fragment) {
                const auto& fs = rt.hw.fs;
                runtime = fmt::format("fs en={} addr={} inputs={} [",
                                      hex(&fs.en_flags, sizeof(fs.en_flags)),
                                      hex(&fs.addr_flags, sizeof(fs.addr_flags)), fs.num_inputs);
                for (u32 i = 0; i < fs.num_inputs && i < fs.inputs.size(); ++i) {
                    const auto& in = fs.inputs[i];
                    runtime += fmt::format("{}{}{}{},", in.param_index, in.is_default ? "d" : "",
                                           in.is_flat ? "f" : "", in.is_default ? in.default_value : 0);
                }
                runtime += "] cb=[";
                for (const auto& cb : fs.color_buffers) {
                    runtime += fmt::format("{}/{}/{}/{}/{},", u32(cb.data_format), u32(cb.num_format),
                                           u32(cb.num_conversion), u32(cb.export_format),
                                           hex(&cb.swizzle, sizeof(cb.swizzle)));
                }
                runtime += fmt::format("] z={} samples={} mrtz={} dual={}", u32(fs.z_export_format),
                                       fs.num_samples, fs.mrtz_mask, fs.dual_source_blending);
            } else if (stored.hw_stage == Shader::HwStage::Vertex) {
                const auto& vs = rt.hw.vs;
                runtime = fmt::format("vs outputs={} map={} clip_disable={} ucp={}", vs.num_outputs,
                                      hex(vs.outputs.data(), sizeof(vs.outputs)), vs.clip_disable,
                                      vs.user_clip_plane_mask);
            } else {
                runtime = fmt::format("stage{} {}", u32(stored.hw_stage), hex(&rt, sizeof(rt)));
            }
            runtime += fmt::format(" ud={} vgprs={}/{}", rt.props.num_user_data,
                                   rt.props.num_input_vgprs, rt.props.num_allocated_vgprs);
            std::string vs_attribs;
            for (const auto& a : spec.vs_attribs) {
                vs_attribs += fmt::format("{}/{}/{},", a.divisor, u32(a.num_class),
                                          hex(&a.dst_select, sizeof(a.dst_select)));
            }
            std::string fetch;
            for (const auto& a : spec.fetch_shader_data.attributes) {
                fetch += hex(&a, sizeof(a)) + ",";
            }
            std::string spec_buffers, spec_images, spec_samplers;
            for (const auto& b : spec.buffers) {
                spec_buffers += fmt::format("{}/{}/{}/{}/{},", u32(b.stride), u32(b.is_formatted),
                                            u32(b.data_format), u32(b.num_format),
                                            u32(b.swizzle_enable));
            }
            for (const auto& i : spec.images) {
                spec_images += fmt::format("{}/{}/{}/{}/{}/{},", u32(i.type), i.is_integer,
                                           i.is_storage, i.is_srgb, u32(i.num_conversion),
                                           i.num_bindings);
            }
            for (const auto& s : spec.samplers) {
                spec_samplers += fmt::format("{}/{},", u32(s.force_unnormalized), u32(s.force_degamma));
            }
            std::string buffers = "-", images = "-", samplers = "-";
            std::vector<u8> program;
            if (Common::FS::IOFile file{std::filesystem::path{dir} / "programs" /
                                            fmt::format("{:016x}.bin", stored.pgm_hash),
                                        Common::FS::FileAccessMode::Read};
                file.IsOpen()) {
                program.resize(file.GetSize());
                file.ReadSpan(std::span{program});
            }
            if (program.empty()) {
                ++no_program;
            } else {
                std::vector<u32> spv;
                const bool ok = ReplayTranslate(
                    profile, pools, program, stored, spec, spv, [&](const Shader::Info& info) {
                        buffers.clear();
                        images.clear();
                        samplers.clear();
                        for (const auto& b : info.buffers) {
                            const auto sharp = b.GetSharp(info);
                            buffers += fmt::format("{}/{}/{}/{}:{},", u32(b.buffer_type), b.is_written,
                                                   b.is_formatted, u32(b.used_types),
                                                   hex(&sharp, sizeof(sharp)));
                        }
                        for (const auto& i : info.images) {
                            const auto sharp = i.GetSharp(info);
                            images += fmt::format("{}{}{}{}:{},", i.is_depth ? "d" : "",
                                                  i.is_array ? "a" : "", i.is_written ? "w" : "",
                                                  i.is_r128 ? "r" : "", hex(&sharp, sizeof(sharp)));
                        }
                        for (const auto& s : info.samplers) {
                            const auto sharp = s.GetSharp(info);
                            samplers += hex(&sharp, sizeof(sharp)) + ",";
                        }
                    });
                if (!ok) {
                    ++failed;
                }
            }
            out << name << '\t' << fmt::format("{:016x}", stored.pgm_hash) << '\t' << perm_idx << '\t'
                << u32(stored.sw_stage) << '\t' << stored.flattened_ud_buf.size() << '\t' << runtime
                << '\t' << vs_attribs << '\t' << fetch << '\t' << buffers << '\t' << images << '\t'
                << samplers << '\t' << spec_buffers << '\t' << spec_images << '\t' << spec_samplers
                << '\n';
            ++written;
        }
    }
    u64 keys = 0;
    {
        std::ofstream out{std::filesystem::path{dir} / "keys.tsv"};
        out << "name\tstages\tvertex_formats\tcolor\tblend\twrite_masks\tmisc\n";
        for (const auto& [name, time] : store.ListBlobs(Storage::BlobType::PipelineKey)) {
            if (!name.starts_with("g_")) continue;
            std::vector<u8> blob;
            store.Load(Storage::BlobType::PipelineKey, name, blob);
            GraphicsPipelineKey key{};
            GraphicsPipeline::SerializationSupport sdata{};
            try {
                Common::RecoverableScope recoverable;
                if (!LoadStoredGraphicsPipeline(std::move(blob), key, sdata)) continue;
            } catch (...) {
                continue;
            }
            std::string stages, formats, color, blend, masks;
            for (const auto h : key.stage_hashes) stages += fmt::format("{:x},", h);
            for (const auto f : key.vertex_buffer_formats) {
                if (f != vk::Format::eUndefined) formats += fmt::format("{},", u32(f));
            }
            for (u32 i = 0; i < key.num_color_attachments && i < key.color_buffers.size(); ++i) {
                const auto& cb = key.color_buffers[i];
                color += fmt::format("{}/{}/{}/{},", u32(cb.data_format), u32(cb.num_format),
                                     u32(cb.num_conversion), u32(cb.export_format));
                blend += hex(&key.blend_controls[i], sizeof(key.blend_controls[i])) + ",";
                masks += fmt::format("{},", u32(key.write_masks[i]));
            }
            out << name << '\t' << stages << '\t' << formats << '\t' << color << '\t' << blend << '\t'
                << masks << '\t'
                << fmt::format("cbmask={} samples={}/{} mrt={:x} z={} stencil={} prim={} poly={} "
                               "patch={}",
                               hex(&key.cb_shader_mask, sizeof(key.cb_shader_mask)),
                               key.num_samples, key.depth_samples, key.mrt_mask, u32(key.z_format),
                               u32(key.stencil_format), u32(key.prim_type), u32(key.polygon_mode),
                               key.patch_control_points)
                << '\n';
            ++keys;
        }
    }
    LOG_WARNING(Render_Vulkan,
                "PB-4 store dump: {} shader records ({} without a program, {} failed), {} pipeline "
                "keys in {}",
                written, no_program, failed, keys, dir);
}

bool PipelineCache::LegacyWaits() const {
    // Keep going: rendering must not wait for a cosmetic pipeline.
    if (CompileMode() == 2) return false;
    // FIX-119: a pipeline of the old renderer's translations (D10) is built at once, as the
    // driver usually has it already. Not in a race: there one the driver did not have stopped
    // the game for up to seconds (19.7 s of such waits in one test race); it builds in the
    // background like the rest. -DisablePerf 141 waits for them in races too.
    if (!S().key_uses_legacy) {
        return false;
    }
#ifdef RYTY_GPU
    static const bool skip_in_races = Common::PerfFeatureEnabled(141);
    if (skip_in_races && RytyGtsInRace()) {
        return false;
    }
#endif
    return true;
}

std::chrono::microseconds PipelineCache::QuickWait() {
    if (CompileMode() == 2) return std::chrono::microseconds{0};
    // FIX-114: a draw whose new pipeline is building waits for it a little, so the pipelines
    // that build quickly (the driver had them, or small ones) are drawn at once and only the
    // slow ones pop in a moment later. At most 4 ms of waiting in any 16 ms, 2 ms per draw.
    // -DisablePerf 137 skips without waiting.
    static const bool enabled = Common::PerfFeatureEnabled(137);
    if (!enabled) {
        return std::chrono::microseconds{0};
    }
    using namespace std::chrono;
    static thread_local steady_clock::time_point window{};
    static thread_local microseconds used{};
    const auto now = steady_clock::now();
    if (now - window > milliseconds{16}) {
        window = now;
        used = microseconds{0};
    }
    const auto wait = std::clamp(microseconds{4000} - used, microseconds{0}, microseconds{2000});
    used += wait; // charged up front: simple, and errs on the side of fewer waits
    return wait;
}

bool PipelineCache::MenusWait() {
    // Keep going: rendering must not wait for a cosmetic pipeline.
    if (CompileMode() == 2) return false;
    // FIX-110: in GT Sport's menus a draw waits for its new pipeline (the guest clocks are held
    // meanwhile, as in the "pause" compile mode); only races skip draws. Some menu scenes draw
    // once into a texture they keep showing: the GR86 slide of the home screen played on black
    // instead of the Fuji pit straight when its scenery draws were skipped while they built.
    // -DisablePerf 134 skips in the menus too.
#ifdef RYTY_GPU
    static const bool enabled = Common::PerfFeatureEnabled(134);
    return enabled && !RytyGtsInRace();
#else
    return false;
#endif
}

bool PipelineCache::CancelBuild(PipelineBuild& build) {
    // A worker may already have taken the job and wait in FinishBuild for whoever claimed the
    // build; release it, the build stays empty.
    if (build.started.exchange(true)) {
        return false;
    }
    build.promise.set_value();
    return true;
}

void PipelineCache::QueueBackgroundBuild(const std::shared_ptr<PipelineBuild>& build) {
    if (!build_workers) {
        build_workers = std::make_unique<PipelineBuildWorkers>();
    }
    build_workers->PushBackground([this, build] {
        if (!build->started.load()) {
            FinishBuild(*build);
            ++prewarm_built;
        }
        ++g_precompile_done; // PERF-098
    });
}

void PipelineCache::ReadAheadAtBufferStart() {
    using namespace std::chrono_literals;
    if (!build_workers || !Common::PerfFeatureEnabled(20) || !Common::PerfFeatureEnabled(19) ||
        std::chrono::steady_clock::now() - last_pipeline_miss > 2s) {
        return;
    }
    // No draw is being processed, so nothing needs restoring afterwards.
    ReadAhead(S().draw_indirect_params, false);
}

void PipelineCache::ReadAhead(const DrawIndirectParams params, bool restore) {
    const auto start = std::chrono::steady_clock::now();
    u32 draws = 0;
    u32 started = 0;
    u32 evaluated = 0;
    const bool dedupe = Common::PerfFeatureEnabled(21);
    liverpool->ScanAheadForPipelines([&](const AmdGpu::Regs& regs) {
        ++draws;
        // FIX-014: draws the rasterizer filters out (Rasterizer::FilterDraw) never get a
        // pipeline; building one for them can fail (primitive type 'None' asserted on a build
        // worker when selecting a car).
        using OperationMode = AmdGpu::ColorControl::OperationMode;
        const auto mode = regs.color_control.mode;
        if (regs.primitive_type == AmdGpu::PrimitiveType::None ||
            mode == OperationMode::EliminateFastClear || mode == OperationMode::FmaskDecompress ||
            mode == OperationMode::Resolve) {
            return false;
        }
        if (dedupe) {
            // Everything the pipeline key is built from that the commands can change: graphics
            // shader registers (programs and user data), context registers (targets, blending,
            // depth, rasterizer state) and the primitive type.
            const auto& r = regs.reg_array;
            constexpr u32 ShGfxRegs = 0x200;
            constexpr u32 NumContextRegs = 0x400;
            constexpr u32 PrimitiveRegs = 0x100;
            std::array<u32, ShGfxRegs> sh;
            std::memcpy(sh.data(), &r[AmdGpu::Regs::ShRegWordOffset], sizeof(sh));
            if (Common::PerfFeatureEnabled(22)) {
                // PERF-022: user data (per-draw constants and resource pointers) changes on
                // nearly every draw; the programs and the other state decide the pipeline.
                // Each stage's 16 user data registers start 0xC after its program registers.
                for (u32 stage_base = 0; stage_base < ShGfxRegs; stage_base += 0x40) {
                    std::fill_n(sh.begin() + stage_base + 0xC, 16, 0u);
                }
            }
            u64 hash = XXH3_64bits(sh.data(), sizeof(sh));
            hash = XXH3_64bits_withSeed(&r[AmdGpu::Regs::ContextRegWordOffset],
                                        NumContextRegs * sizeof(u32), hash);
            hash = XXH3_64bits_withSeed(&r[AmdGpu::Regs::UconfigRegWordOffset + 0x200],
                                        PrimitiveRegs * sizeof(u32), hash);
            if (!read_ahead_states.insert(hash).second) {
                return false;
            }
            if (read_ahead_states.size() > 1u << 16) {
                read_ahead_states.clear();
            }
        }
        ++evaluated;
        regs_override = &regs;
        S().draw_indirect_params = {};
        const bool valid = RefreshGraphicsKey();
        regs_override = nullptr;
        if (!valid || !HasSupportedColorTargets() || graphics_pipelines.contains(S().graphics_key)) {
            return false;
        }
        if (pending_builds.contains(S().graphics_key)) {
            return false;
        }
        pending_builds.emplace(S().graphics_key, StartPipelineBuild(false));
        ++started;
        return true;
    });
    if (draws > 0) {
        last_scan_draws = draws;
        draws_since_scan = 0;
        // Back to the draw being processed.
        S().draw_indirect_params = params;
        if (restore && evaluated > 0) {
            RefreshGraphicsKey();
        }
    }
    // One summary every 2 s instead of a line per read.
    auto& stats = read_ahead_stats;
    const auto now = std::chrono::steady_clock::now();
    ++stats.scans;
    stats.draws += draws;
    stats.evaluated += evaluated;
    stats.started += started;
    stats.ms += std::chrono::duration<double, std::milli>(now - start).count();
    if (now - stats.since >= std::chrono::seconds{2}) {
        if (stats.draws > 0) {
            LOG_WARNING(Render_Vulkan,
                        "Pipeline read-ahead in {:.1f} s: {} reads, {} draws read, {} evaluated, "
                        "{} builds started, {:.1f} ms, {} pending",
                        std::chrono::duration<double>(now - stats.since).count(), stats.scans,
                        stats.draws, stats.evaluated, stats.started, stats.ms,
                        pending_builds.size());
        }
        stats = {};
        stats.since = now;
    }
}

void PipelineCache::BuildQueuedStates(const DrawIndirectParams params, bool restore) {
    auto* states = liverpool->PipeReadAheadStates();
    if (!states || states->Empty()) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!queued_state_regs) {
        // A queued state holds every register a key is built from; the others keep these values.
        queued_state_regs = std::make_unique<AmdGpu::Regs>(liverpool->DrawRegs());
    }
    Common::PhaseTimer eval_timer{Common::Phase::ReadAheadEval};
    const auto key = S().graphics_key;
    const bool key_was_legacy = S().key_uses_legacy; // D10: made again the same way below
    u32 evaluated = 0;
    u32 started = 0;
    // Before a draw, 2 ms at most, and while waiting for a build, 64 states between checks on
    // it; the rest comes next time.
    const auto budget = start + std::chrono::milliseconds{2};
    while ((restore ? evaluated < 64 : std::chrono::steady_clock::now() < budget) &&
           states->Take(queued_state_regs->reg_array)) {
        ++evaluated;
        regs_override = queued_state_regs.get();
        S().draw_indirect_params = {};
        const bool valid = RefreshGraphicsKey();
        regs_override = nullptr;
        if (!valid || !HasSupportedColorTargets() || graphics_pipelines.contains(S().graphics_key)) {
            continue;
        }
        if (const auto pending = pending_builds.find(S().graphics_key);
            pending != pending_builds.end()) {
            // PERF-038: a stored pipeline still waiting in the background queue is needed soon;
            // build it now from this draw's state instead.
            if (pending->second->preloading && CancelBuild(*pending->second)) {
                pending.value() = StartPipelineBuild(false);
                ++prewarm_stats.promoted;
                ++started;
            }
            continue;
        }
        pending_builds.emplace(S().graphics_key, StartPipelineBuild(false));
        ++started;
    }
    if (started > 0 && build_workers) {
        // PERF-038: the game needs pipelines; background builds wait.
        build_workers->PauseBackground(std::chrono::steady_clock::now() + std::chrono::seconds{2});
    }
    S().draw_indirect_params = params;
    if (restore && evaluated > 0) {
        const bool was_forced = force_legacy;
        force_legacy = key_was_legacy && legacy_fallback;
        RefreshGraphicsKey();
        force_legacy = was_forced;
        ASSERT_MSG(S().graphics_key == key, "Read-ahead changed the current pipeline key");
    }
    auto& stats = queued_state_stats;
    const auto now = std::chrono::steady_clock::now();
    stats.evaluated += evaluated;
    stats.started += started;
    stats.ms += std::chrono::duration<double, std::milli>(now - start).count();
    if (now - stats.since >= std::chrono::seconds{2}) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-034 pipeline read-ahead in {:.1f} s: {} draw states from the command "
                    "thread evaluated, {} builds started, {:.1f} ms; {} draws waited for a build "
                    "({:.1f} ms); {} pending",
                    std::chrono::duration<double>(now - stats.since).count(), stats.evaluated,
                    stats.started, stats.ms, stats.waits, stats.wait_ms, pending_builds.size());
        stats = {};
        stats.since = now;
    }
}

void PipelineCache::WaitForBuild(PipelineBuild& build, const DrawIndirectParams params) {
    using namespace std::chrono_literals;
    BuildQueuedStates(params, true);
    // An urgent build is at the front of the workers' queue, and an idle worker takes it at
    // once; with every worker busy, this thread builds it (FinishBuild). So does a read-ahead
    // build no worker has reached yet.
    if (build.urgent) {
        const auto grace = std::chrono::steady_clock::now() + 200us;
        while (!build.started.load() && std::chrono::steady_clock::now() < grace) {
            std::this_thread::yield();
        }
    }
    while (build.started.load() && build.done.wait_for(500us) != std::future_status::ready) {
        BuildQueuedStates(params, true);
    }
#ifdef RYTY_GPU
    // PERF-101: GPU.compile_batch_ms (launcher "Shader compile batching"): with the game paused
    // for a compile anyway, the builds of the upcoming draws the command thread read ahead are
    // finished in the same pause, up to that long, so later draws pause less often.
    static const u32 batch_ms = static_cast<u32>(std::max(0, EmulatorSettings.GetCompileBatchMs()));
    if (batch_ms != 0 && !AsyncPipelines() && build_workers) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{batch_ms};
        BuildQueuedStates(params, true);
        while (!build_workers->ForegroundIdle() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(500us);
            BuildQueuedStates(params, true);
        }
    }
#endif
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params) {
    // PERF-034: builds for the upcoming draws the command thread read ahead start first.
    BuildQueuedStates(params, false);
    // PERF-038: then, while no draw waits for pipelines, stored ones go to the background.
    InstallPrewarmed();
    S().draw_indirect_params = params;
    ++draws_since_scan;
    // FIX-078: missing translations of this draw's stages are queued together first; then the
    // draws after it are read and their shaders queued too, and the key is made again, waiting
    // for the workers' results (the slowest one, not their sum).
    defer_misses = AsyncTranslation();
    deferred_miss = false;
    fallback_miss = false;
    translation_skipped = false; // SHD-005
    bool key_valid = RefreshGraphicsKey();
    // D10: a new translation this draw needs is being made; the old renderer's draw it.
    if (!key_valid && fallback_miss) {
        fallback_nested = true;
        force_legacy = true;
        const bool legacy_valid = RefreshGraphicsKey() && !S().legacy_failed;
        force_legacy = false;
        const GraphicsPipeline* legacy = legacy_valid ? PipelineForCurrentKey(params) : nullptr;
        ++(legacy ? legacy_stats.fallback_ok : legacy_stats.fallback_none);
        ReportLegacy();
        if (legacy) {
            fallback_nested = false;
            return legacy;
        }
        // No old translations: the usual way (the queued translations are taken when ready).
        deferred_miss = false;
        key_valid = RefreshGraphicsKey();
    }
    struct NestedReset {
        bool& flag;
        ~NestedReset() {
            flag = false;
        }
    } nested_reset{fallback_nested};
    // One stage is queued per pass (a stage's start bindings follow from the stages before it);
    // the read-ahead runs once, after the first.
    for (u32 pass = 0; !key_valid && deferred_miss && pass < 6; ++pass) {
        if (pass == 0) {
            liverpool->ReadAheadNow();
            BuildQueuedStates(params, false);
            S().draw_indirect_params = params;
        }
        deferred_miss = false;
        key_valid = RefreshGraphicsKey();
        if (!key_valid && translation_skipped) {
            break; // SHD-005
        }
    }
    defer_misses = false;
    if (!key_valid && translation_skipped) {
        // SHD-005: skipped while its shaders translate; a later draw takes them.
        deferred_miss = false;
        translation_skipped = false;
        ++async_stats.skipped_translating;
        NoteSkippedDraw();
        return nullptr;
    }
    if (!key_valid && deferred_miss) {
        deferred_miss = false;
        key_valid = RefreshGraphicsKey();
    }
    if (!key_valid) {
        return nullptr;
    }
    // D10: a key of new translations whose pipeline is not built yet. It is built in the
    // background; meanwhile the draw uses the key made of the old renderer's translations.
    if (legacy_fallback && !fallback_nested && !S().key_uses_legacy && regs_override == nullptr &&
        !graphics_pipelines.contains(S().graphics_key)) {
        const auto pending = pending_builds.find(S().graphics_key);
        const bool ready = pending != pending_builds.end() &&
                           pending->second->done.wait_for(std::chrono::seconds{0}) ==
                               std::future_status::ready;
        if (!ready && S().graphics_key.num_color_attachments <= AmdGpu::NUM_COLOR_BUFFERS) {
            if (pending == pending_builds.end()) {
                auto build = StartPipelineBuild(false);
                build->async = true;
                pending_builds[S().graphics_key] = build;
            }
            fallback_nested = true;
            force_legacy = true;
            const bool legacy_valid = RefreshGraphicsKey() && !S().legacy_failed;
            force_legacy = false;
            const GraphicsPipeline* legacy =
                legacy_valid ? PipelineForCurrentKey(params) : nullptr;
            fallback_nested = false;
            ++(legacy ? legacy_stats.fallback_ok
                      : legacy_valid ? legacy_stats.fallback_unbuilt : legacy_stats.fallback_none);
            ReportLegacy();
            if (legacy) {
                return legacy;
            }
            // No old translations for every stage: the new key again, the usual way.
            if (!RefreshGraphicsKey()) {
                return nullptr;
            }
        }
    }
    return PipelineForCurrentKey(params);
}

const GraphicsPipeline* PipelineCache::LookupBuiltPipeline(const DrawIndirectParams params) {
    // NAT-031: the key is computed in this thread's own scratch; the cache's own (the threads
    // above's last selection) is not touched.
    static thread_local KeyScratch scratch;
    KeyScratch* const saved = std::exchange(t_scratch, &scratch);
    t_lookup_only = true;
    scratch.draw_indirect_params = params;
    scratch.lookup_miss = nullptr;
    const GraphicsPipeline* found = nullptr;
    if (RefreshGraphicsKey()) {
        if (const auto it = graphics_pipelines.find(scratch.graphics_key);
            it != graphics_pipelines.end()) {
            found = it->second.get();
        }
        if (!found) {
            scratch.lookup_miss = "pipeline not built";
        }
    } else if (!scratch.lookup_miss) {
        scratch.lookup_miss = "no key (unsupported state or hull constants)";
    }
    t_lookup_only = false;
    t_scratch = saved;
    t_job_infos = &scratch.infos;
    // The miss reason is read through LookupMiss() on this thread right after: keep it.
    t_last_miss = scratch.lookup_miss;
    return found;
}

Shader::Info& PipelineCache::MatchInfo(const Shader::Info& cached) {
    struct Copy {
        u64 identity{};
        std::unique_ptr<Shader::Info> info;
    };
    static thread_local std::unordered_map<const Shader::Info*, Copy> copies;
    if (copies.size() > 8192) {
        copies.clear(); // programs come and go: don't keep copies of retired ones forever
    }
    auto& copy = copies[&cached];
    if (!copy.info || copy.identity != cached.identity) {
        copy.info = std::make_unique<Shader::Info>(cached);
        copy.identity = cached.identity;
    }
    return *copy.info;
}

const GraphicsPipeline* PipelineCache::PipelineForCurrentKey(const DrawIndirectParams params) {
    if (S().graphics_key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        LOG_WARNING(Render_Vulkan, "Skipping draw with {} color attachments (MRT mask={:#x})",
                    S().graphics_key.num_color_attachments, S().graphics_key.mrt_mask);
        return nullptr;
    }
    for (u32 cb = 0; cb < S().graphics_key.num_color_attachments; ++cb) {
        const auto& color_buffer = S().graphics_key.color_buffers[cb];
        if (color_buffer.data_format != AmdGpu::DataFormat::FormatInvalid &&
            Vulkan::LiverpoolToVK::TrySurfaceFormat(
                color_buffer.data_format, color_buffer.num_format) == vk::Format::eUndefined) {
            LOG_WARNING(Render_Vulkan,
                        "Skipping draw with unsupported color target {} format: data={}, number={}",
                        cb, static_cast<u32>(color_buffer.data_format),
                        static_cast<u32>(color_buffer.num_format));
            return nullptr;
        }
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(S().graphics_key);
    if (is_new) {
        Common::GuestClock::CompilePause hold; // FIX-044
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(S().graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
        // DIAG-050: the parts of new keys of tessellated pipelines (GT Sport's race summary built
        // hundreds of keys for the same few programs).
        // DIAG-051: which part of a new key differs from the last key with the same stages
        // (GT Sport's race summary asks for thousands of new pipelines).
        {
            static std::unordered_map<u64, GraphicsPipelineKey> last_by_stages;
            static std::unordered_map<std::string, u32> diff_counts;
            static u32 keys_seen = 0;
            const u64 stages_hash = XXH3_64bits(S().graphics_key.stage_hashes.data(),
                                                sizeof(S().graphics_key.stage_hashes));
            auto [prev, first] = last_by_stages.try_emplace(stages_hash, S().graphics_key);
            std::string fields = first ? "new stages" : "";
            if (!first) {
                const auto& a = prev->second;
                const auto& b = S().graphics_key;
                const auto differs = [&](const auto& x, const auto& y) {
                    return std::memcmp(&x, &y, sizeof(x)) != 0;
                };
                const auto add = [&](const char* name) {
                    fields += fields.empty() ? name : std::string(",") + name;
                };
                if (differs(a.vertex_buffer_formats, b.vertex_buffer_formats)) add("vtx");
                if (a.patch_control_points != b.patch_control_points) add("pcp");
                if (a.num_color_attachments != b.num_color_attachments) add("ncb");
                if (differs(a.color_buffers, b.color_buffers)) add("cb");
                if (differs(a.blend_controls, b.blend_controls)) add("blend");
                if (differs(a.write_masks, b.write_masks)) add("mask");
                if (differs(a.cb_shader_mask, b.cb_shader_mask)) add("shmask");
                if (a.logic_op != b.logic_op) add("logic");
                if (a.num_samples != b.num_samples || a.depth_samples != b.depth_samples ||
                    differs(a.color_samples, b.color_samples)) add("samples");
                if (a.mrt_mask != b.mrt_mask) add("mrt");
                if (a.z_format != b.z_format || a.stencil_format != b.stencil_format ||
                    a.depth_clamp_enable != b.depth_clamp_enable) add("depth");
                if (a.prim_type != b.prim_type || a.polygon_mode != b.polygon_mode ||
                    a.clip_space != b.clip_space || a.provoking_vtx_last != b.provoking_vtx_last ||
                    a.depth_clip_enable != b.depth_clip_enable) add("raster");
                if (fields.empty()) {
                    add("padding");
                    static u32 logged = 0;
                    if (logged++ < 10) {
                        const auto* x = reinterpret_cast<const u8*>(&a);
                        const auto* y = reinterpret_cast<const u8*>(&b);
                        std::string offsets;
                        for (size_t i = 0; i < sizeof(GraphicsPipelineKey); ++i) {
                            if (x[i] != y[i]) {
                                offsets += fmt::format(" {}:{:02x}/{:02x}", i, x[i], y[i]);
                            }
                        }
                        LOG_WARNING(Render_Vulkan,
                                    "DIAG-051: keys differ only outside named fields (size {}, "
                                    "color_buffers at {}, mrt_mask at {}):{}",
                                    sizeof(GraphicsPipelineKey),
                                    offsetof(GraphicsPipelineKey, color_buffers),
                                    offsetof(GraphicsPipelineKey, mrt_mask), offsets);
                    }
                }
                prev->second = S().graphics_key;
            }
            ++diff_counts[fields];
            if (++keys_seen % 250 == 0) {
                std::string summary;
                for (const auto& [name, count] : diff_counts) {
                    summary += fmt::format(" {}={}", name, count);
                }
                LOG_WARNING(Render_Vulkan, "DIAG-051: {} new keys; differences from the last "
                            "key with the same stages:{}", keys_seen, summary);
            }
        }
        // SHADGT_LOG_KEYS=1 logs every new key the same way (up to 5000).
        static const bool log_all_keys = [] {
            const char* env = std::getenv("SHADGT_LOG_KEYS");
            return env && env[0] == '1';
        }();
        if (log_all_keys ||
            S().graphics_key.stage_hashes[u32(Shader::SwStage::TessellationControl)] != 0) {
            static std::atomic<u32> logged{};
            if (const u32 n = ++logged; n <= (log_all_keys ? 5000u : 80u)) {
                const auto& k = S().graphics_key;
                const auto part = [](const auto& field) {
                    return XXH3_64bits(&field, sizeof(field)) & 0xffff;
                };
                LOG_WARNING(Render_Vulkan,
                            "DIAG-050: new key {} {:#x}: stages {:#x} {:#x} {:#x} {:#x} {:#x}; "
                            "pcp {} prim {} vtx {:#x} cb {:#x} blend {:#x} mask {:#x} "
                            "shmask {:#x} samples {} {} mrt {:#x} z {} clip {}; all {:#x}",
                            n, pipeline_hash, k.stage_hashes[0], k.stage_hashes[1],
                            k.stage_hashes[2], k.stage_hashes[3], k.stage_hashes[4],
                            k.patch_control_points,
                            u32(k.prim_type), part(k.vertex_buffer_formats), part(k.color_buffers),
                            part(k.blend_controls), part(k.write_masks), part(k.cb_shader_mask),
                            k.num_samples, k.depth_samples, k.mrt_mask, u32(k.z_format),
                            u32(k.depth_clip_enable),
                            XXH3_64bits(&k, sizeof(k)) & 0xffffff);
            }
        }

        GraphicsPipeline::SerializationSupport sdata{};
        // PERF-019: the pipeline may already be building from an earlier read-ahead. If not,
        // it starts building now, and the draws after it are read for more new pipelines to
        // build at the same time on other cores.
        std::shared_ptr<PipelineBuild> build;
        bool predicted = false;
        auto pending = pending_builds.find(S().graphics_key);
        if (pending != pending_builds.end() &&
            (pending->second->preloading || pending->second->async)) {
            auto stored = pending->second;
            if (stored->done.wait_for(stored->async ? QuickWait() : std::chrono::microseconds{0}) ==
                std::future_status::ready) {
                // PERF-038: built in the background before the draw needed it; no miss.
                pending_builds.erase(pending);
                stored->pipeline->SetStageInfos(S().infos);
                sdata = stored->sdata;
                it.value() = std::move(stored->pipeline);
                ++(stored->async ? async_stats.installed : prewarm_stats.hits);
                // Stored again, so the next session builds it among the first.
                if (!S().key_uses_legacy) {
                    RegisterPipelineData(S().graphics_key, pipeline_hash, sdata);
                }
                return it->second.get();
            }
            if ((stored->async || (CompileMode() == 2 && AsyncPipelines())) &&
                !LegacyWaits() && !WritesMemory(S().infos) && !NeverSkipped(S().infos) &&
                !MenusWait()) {
                // PERF-076: still building; this draw is skipped, the next ones look again.
                graphics_pipelines.erase(it);
                ++async_stats.skipped_draws;
                NoteSkippedDraw(); // PERF-103
                return nullptr;
            }
            if (CancelBuild(*stored)) {
                // Not started in the background yet: built now the usual way.
                pending_builds.erase(pending);
                pending = pending_builds.end();
                ++prewarm_stats.cancelled;
            }
        }
        if (build_workers && CompileMode() != 2) {
            build_workers->PauseBackground(std::chrono::steady_clock::now() +
                                           std::chrono::seconds{2});
        }
        ++Common::GetWorkCounters().pipelines_compiled;
        last_pipeline_miss = std::chrono::steady_clock::now();
        GraphicsPipeline::NotePipelineMiss();
        auto* const queued_states = liverpool->PipeReadAheadStates();
        if (queued_states) {
            queued_states->NoteMiss();
        }
        if (pending != pending_builds.end()) {
            build = pending->second;
            pending_builds.erase(pending);
            predicted = true;
            // PERF-031: the read-ahead reads the command thread's state; not with the draw pipe.
            if (Common::PerfFeatureEnabled(19) && !liverpool->Pipelined() &&
                draws_since_scan >= last_scan_draws) {
                // Past the last read-ahead: keep reading from where it stopped.
                const auto key = S().graphics_key;
                ReadAhead(params);
                ASSERT_MSG(S().graphics_key == key, "Read-ahead changed the current pipeline key");
            }
        } else if (Common::PerfFeatureEnabled(19) && !liverpool->Pipelined()) {
            const auto key = S().graphics_key;
            build = StartPipelineBuild(true);
            ReadAhead(params);
            ASSERT_MSG(S().graphics_key == key, "Read-ahead changed the current pipeline key");
        } else if (queued_states) {
            // PERF-034: a worker builds it while this thread starts the builds the command
            // thread's read-ahead queues meanwhile.
            build = StartPipelineBuild(true);
        } else if (CompileMode() == 2 && AsyncPipelines()) {
            // Queue even when read-ahead/draw-pipe helpers are disabled. Retain the job
            // in pending_builds below so a later draw installs its completed pipeline.
            build = StartPipelineBuild(true);
        }
        // PERF-076: a draw does not wait for a new pipeline. Its build goes on on a worker, the
        // draws that need it are skipped until it is done (a frame or two of a missing object),
        // and the game does not stall for the 60-140 ms a build takes. SHADGT_ASYNC_PIPELINES=0
        // or -DisablePerf 76 waits as before.
        // PERF-103: in slow-motion mode every new pipeline goes on in the background at once
        // and the game goes on in slow motion until it is built.
        // D10: a pipeline of old translations is built right away (the driver usually has it
        // from the old renderer's sessions, so it takes milliseconds).
        if (build && AsyncPipelines() && !LegacyWaits() && !WritesMemory(S().infos) &&
            !NeverSkipped(S().infos) && !MenusWait() &&
            build->done.wait_for(QuickWait()) != std::future_status::ready) {
            build->async = true;
            pending_builds[S().graphics_key] = build;
            graphics_pipelines.erase(it);
            ++async_stats.deferred;
            ++async_stats.skipped_draws;
            NoteSkippedDraw(); // PERF-103
            return nullptr;
        }
        if (build) {
            const auto wait_start = std::chrono::steady_clock::now();
            if (queued_states) {
                Common::Nvtx::Scope nvtx{"wait for pipeline build"}; // DIAG-056
                WaitForBuild(*build, params);
            }
            FinishBuild(*build);
            const auto now = std::chrono::steady_clock::now();
            if (queued_states) {
                ++queued_state_stats.waits;
                queued_state_stats.wait_ms +=
                    std::chrono::duration<double, std::milli>(now - wait_start).count();
            }
            ++Common::GetWorkCounters().pipeline_waits;
            Common::GetWorkCounters().pipeline_wait_us += u64(
                std::chrono::duration_cast<std::chrono::microseconds>(now - wait_start).count());
            RyDiag::RecordElapsed(RyDiag::Zone::PipelineWait, now - wait_start, pipeline_hash);
            // DIAG-124: why this draw waited instead of being skipped.
            std::string why;
            if (!AsyncPipelines()) why += " no-async";
            if (S().key_uses_legacy) why += " legacy";
            if (WritesMemory(S().infos)) why += " writes-memory";
            if (NeverSkipped(S().infos)) why += " tessellation";
            if (MenusWait()) why += " menus";
            LOG_WARNING(Render_Vulkan,
                        "Pipeline {:#x} {}: built {:.1f} ms after it was queued, draw waited "
                        "{:.1f} ms (DIAG-124:{})",
                        pipeline_hash, predicted ? "read ahead" : "on demand",
                        std::chrono::duration<double, std::milli>(now - build->queued).count(),
                        std::chrono::duration<double, std::milli>(now - wait_start).count(),
                        why.empty() ? " ready within the quick wait" : why);
            build->pipeline->SetStageInfos(S().infos);
            sdata = build->sdata;
            it.value() = std::move(build->pipeline);
        } else {
            Common::Nvtx::Scope nvtx{"build pipeline here"}; // DIAG-056
            RYDIAG_SCOPE(PipelineBuild, pipeline_hash);
            it.value() = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, S().graphics_key, *pipeline_cache, S().infos,
                S().runtime_infos, S().fetch_shader, S().modules, sdata);
        }

        if (!S().key_uses_legacy) { // D10: old translations are not in this store
            RegisterPipelineData(S().graphics_key, pipeline_hash, sdata);
        } else {
            ++legacy_stats.pipelines;
        }
        ++num_new_pipelines;
        MaybeSaveDriverCache();

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (S().infos[stage]) {
                    auto& m = S().modules[stage];
                    module_related_pipelines[m].emplace_back(S().graphics_key);
                }
            }
        }
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    fallback_miss = false;
    bool key_valid = RefreshComputeKey();
    if (!key_valid && fallback_miss) {
        // D10: as for graphics, the old translation while the new one is made.
        fallback_nested = true;
        force_legacy = true;
        key_valid = RefreshComputeKey() && !S().legacy_failed;
        force_legacy = false;
        ++(key_valid ? legacy_stats.fallback_ok : legacy_stats.fallback_none);
        if (!key_valid) {
            key_valid = RefreshComputeKey();
        }
        fallback_nested = false;
    }
    if (!key_valid) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        Common::GuestClock::CompilePause hold; // FIX-044
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        ++Common::GetWorkCounters().pipelines_compiled;
        RYDIAG_SCOPE(PipelineBuild, pipeline_hash);
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *S().infos[0],
                                                       S().modules[0], sdata);
        if (!S().key_uses_legacy) { // D10
            RegisterPipelineData(compute_key, sdata);
        }
        ++num_new_pipelines;
        MaybeSaveDriverCache();

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = S().modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

void PipelineCache::RefreshSwizzledBlend(u32 cb, Shader::PsColorBuffer& color_buffer,
                                         const AmdGpu::BlendControl& bc) {
    const auto& regs = Regs();
    // Dual-source blending needs attachment 0 as the only written color target. Hardware writes
    // another MRT only when it is bound, unmasked, and has a shader export format.
    const auto writes_other_mrt = [&] {
        for (u32 other = 1; other < AmdGpu::NUM_COLOR_BUFFERS; ++other) {
            if (regs.color_buffers[other] && regs.color_target_mask.GetMask(other) != 0 &&
                regs.color_export_format.GetFormat(other) != AmdGpu::ShaderExportFormat::Zero) {
                return true;
            }
        }
        return false;
    };
    const char* rejection = nullptr;
    if (cb != 0) {
        rejection = "not attachment 0";
    } else if (!instance.IsDualSourceBlendSupported()) {
        rejection = "dual-source blending unsupported";
    } else if (writes_other_mrt()) {
        rejection = "other MRT exports";
    } else if ((regs.color_shader_mask.GetMask(cb) & AmdGpu::ColorBufferMask::ComponentA) == 0) {
        rejection = "alpha not exported";
    }
    if (!rejection && LiverpoolToVK::NeedsSwizzledAlphaBlend(color_buffer.swizzle, bc)) {
        color_buffer.blend_swizzled_alpha = 1;
        return;
    }

    using NumberFormat = AmdGpu::NumberFormat;
    const auto num_format = color_buffer.num_format;
    const bool blendable_format =
        num_format == NumberFormat::Unorm || num_format == NumberFormat::Snorm ||
        num_format == NumberFormat::Srgb || num_format == NumberFormat::Float;
    if (!rejection && color_buffer.num_conversion != AmdGpu::NumberConversion::None) {
        rejection = "number conversion";
    } else if (!rejection && !blendable_format) {
        rejection = "number format";
    }
    const auto general = LiverpoolToVK::GetSwizzledFactorBlend(color_buffer.swizzle, bc);
    if (!rejection && general) {
        color_buffer.blend_swizzled_factors = 1;
        color_buffer.swizzled_color_src = general->color_src;
        color_buffer.swizzled_color_dst = general->color_dst;
        color_buffer.swizzled_alpha_src = general->alpha_src;
        color_buffer.swizzled_alpha_dst = general->alpha_dst;
        return;
    }
    if (!rejection) {
        rejection = "unsupported equation";
    }

    // Report each remaining configuration once; it still blends with the wrong lane order.
    const auto& swizzle = color_buffer.swizzle;
    const u64 config =
        u64(std::bit_cast<u32>(bc)) | u64(cb) << 32 | u64(u32(num_format)) << 36 |
        u64(swizzle.r) << 40 | u64(swizzle.g) << 44 | u64(swizzle.b) << 48 | u64(swizzle.a) << 52 |
        u64(writes_other_mrt()) << 56 |
        u64((regs.color_shader_mask.GetMask(cb) & AmdGpu::ColorBufferMask::ComponentA) != 0) << 57;
    static constexpr size_t MaxLoggedSwizzledBlends = 32;
    if (t_lookup_only || // NAT-031: job threads leave the log to the threads above
        logged_swizzled_blends.size() >= MaxLoggedSwizzledBlends ||
        !logged_swizzled_blends.insert(config).second) {
        return;
    }
    const auto blend = LiverpoolToVK::EffectiveBlend(bc);
    LOG_WARNING(Render_Vulkan,
                "Unhandled swizzled blend ({}): cb={} swizzle={},{},{},{} format={}/{} "
                "color={}*src {} {}*dst alpha={}*src {} {}*dst separate={} export_formats={:#x} "
                "shader_mask={:#x} target_mask={:#x}",
                rejection, cb, u32(swizzle.r), u32(swizzle.g), u32(swizzle.b), u32(swizzle.a),
                u32(color_buffer.data_format), u32(num_format), u32(blend.color_src),
                u32(blend.color_func), u32(blend.color_dst), u32(blend.alpha_src),
                u32(blend.alpha_func), u32(blend.alpha_dst), u32(bc.separate_alpha_blend),
                regs.color_export_format.raw, regs.color_shader_mask.raw,
                regs.color_target_mask.raw);
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&S().graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = Regs();
    auto& key = S().graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type =
        QuadListAsTriangles() ? AmdGpu::PrimitiveType::TriangleList : regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        const auto data_format = col_buf.GetDataFmt();
        const auto number_format = col_buf.GetNumberFmt();
        if (Vulkan::LiverpoolToVK::TrySurfaceFormat(data_format, number_format) ==
            vk::Format::eUndefined) {
            LOG_WARNING(Render_Vulkan,
                        "Skipping draw with unsupported color target {} format: data={}, number={}",
                        cb, static_cast<u32>(data_format), static_cast<u32>(number_format));
            return false;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = data_format;
        color_buffer.num_format = number_format;
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();

        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == AmdGpu::BlendControl::BlendFunc::Min ||
             bc.color_func == AmdGpu::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == AmdGpu::BlendControl::BlendFactor::DstColor;
        if (!col_buf.info.blend_bypass &&
            LiverpoolToVK::IsLaneDependentSwizzledBlend(color_buffer.swizzle, bc)) {
            RefreshSwizzledBlend(cb, color_buffer, bc);
        }
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = Regs();
    auto& key = S().graphics_key;
    S().fetch_shader = nullptr;
    S().stage_deferred = false; // FIX-078
    S().key_uses_legacy = false; // D10
    S().legacy_failed = false;

    // FIX-027: a shader program register pointing outside guest memory (GT Sport's dealership had
    // a pixel shader at 0x100) crashed the emulator reading the program. Skip such a draw, like
    // the GPU would fault on it rather than take the system down. Checked again only when an
    // address changes. -DisablePerf 36 reads it anyway.
    static const bool check_programs = Common::PerfFeatureEnabled(36);
    if (check_programs) {
        static std::array<VAddr, 6> checked{};
        for (u32 stage = 0; stage < checked.size(); ++stage) {
            const auto* pgm = regs.ProgramForStage(stage);
            const VAddr address = pgm ? pgm->Address<VAddr>() : 0;
            if (!regs.stage_enable.IsStageEnabled(stage) || address == 0 ||
                address == checked[stage]) {
                continue;
            }
            auto* memory = Core::Memory::Instance();
            if (!memory->IsValidMapping(address, 8) || !memory->IsMappedAddress(address)) {
                static std::atomic<u32> skipped{};
                if (const u32 n = ++skipped; n <= 20 || n % 500 == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "FIX-027: draw {} skipped: hardware stage {} program at {:#x} is "
                                "not in guest memory ({})",
                                n, stage, address, regs_override ? "read-ahead" : "draw");
                }
                return false;
            }
            if (!t_lookup_only) { // NAT-031: job threads only read the list
                checked[stage] = address;
            }
        }
    }

    Shader::Backend::Bindings binding{};
    bool stage_failed = false;
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            S().infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            S().infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::tie(S().infos[stage_out_idx], S().modules[stage_out_idx], key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        if (!S().modules[stage_out_idx]) {
            stage_failed = true; // FIX-034: the shader could not be translated
        }
        return true;
    };

    S().infos.fill(nullptr);
    S().modules.fill(nullptr);

    bind_stage(HwStage::Fragment, SwStage::Fragment);

    const auto* fs_info = S().infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);
    if (key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        LOG_WARNING(Render_Vulkan, "Skipping draw with invalid MRT mask {:#x}", key.mrt_mask);
        return false;
    }

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (const auto* hull = S().infos[u32(SwStage::TessellationControl)];
            !hull || !SaneTessConstants(*hull, liverpool, regs_override != nullptr)) {
            return false; // FIX-058: the local stage reads the hull's constants
        }
        if (!bind_stage(HwStage::Vertex, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (const auto* hull = S().infos[u32(SwStage::TessellationControl)];
            !hull || !SaneTessConstants(*hull, liverpool, regs_override != nullptr)) {
            return false; // FIX-058: the local stage reads the hull's constants
        }
        if (!bind_stage(HwStage::Export, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    default:
        LOG_WARNING(Render_Vulkan, "unimplemented shader stage {}", (u32)regs.stage_enable.raw);
        return false;
    }
    if (stage_failed) {
        return false;
    }

    const auto* vs_info = S().infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && S().fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : S().fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            const auto data_format = buffer.GetDataFmt();
            const auto number_format = buffer.GetNumberFmt();
            const auto format = Vulkan::LiverpoolToVK::TrySurfaceFormat(data_format, number_format);
            if (format == vk::Format::eUndefined) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping draw with unsupported vertex attribute {} format: data={}, "
                            "number={}",
                            vertex_binding, static_cast<u32>(data_format),
                            static_cast<u32>(number_format));
                return false;
            }
            key.vertex_buffer_formats[vertex_binding++] = format;
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    S().key_uses_legacy = false; // D10
    S().legacy_failed = false;
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->DrawCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(S().infos[0], S().modules[0], compute_key.value) =
        GetProgram(HwStage::Compute, SwStage::Compute, cs_params, binding);
    return S().modules[0] != nullptr; // FIX-034: false when the shader could not be translated
}

} // namespace Vulkan
namespace Shader::Optimization {
extern thread_local std::span<const u32> g_diag_compiling_code;
extern thread_local u64 g_diag_compiling_hash;
} // namespace Shader::Optimization
namespace Vulkan {

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    // FIX-044: translating a shader on the GPU thread holds the guest clocks; a PS4 runs
    // precompiled shaders, so the game never sees this time pass.
    Common::GuestClock::CompilePause hold;
    ++Common::GetWorkCounters().shaders_compiled;
    RYDIAG_SCOPE(ShaderCompile, info.pgm_hash);
    last_spv_hash = 0;
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");

    Shader::Optimization::g_diag_compiling_code = code;
    Shader::Optimization::g_diag_compiling_hash = info.pgm_hash;
    const auto translate_start = std::chrono::steady_clock::now();
    std::vector<u32> spv;
    try {
        // FIX-034: a shader the translator cannot handle no longer stops the emulator. Its
        // draws are skipped and it is reported once, so one run lists every such shader.
        Common::RecoverableScope recoverable;
        const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
        spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    } catch (const Common::RecoverableFailure& e) {
        LOG_ERROR(Render_Vulkan,
                  "SHADER-FAIL: {} shader {:#x} (permutation {}, {} dwords) could not be "
                  "translated: {}; draws that use it are skipped",
                  info.hw_stage, info.pgm_hash, perm_idx, code.size(), e.what());
        DumpFailedShader(code, info.pgm_hash, info.hw_stage);
        return {};
    }
    const auto translate_end = std::chrono::steady_clock::now();
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");
    last_spv_hash = XXH3_64bits(spv.data(), spv.size() * sizeof(u32));
    if (!Common::PerfFeatureEnabled(44)) {
        // FIX-043 off (-DisablePerf 44): permutations keyed on every resource property.
        info.resource_usage_known = false;
    }

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.hw_stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }
    // PERF-DIAG-013: where runtime shader compile time goes.
    LOG_WARNING(
        Render_Vulkan, "Shader {} {:#x}: translate {:.1f} ms, module {:.1f} ms, {} words",
        info.hw_stage, info.pgm_hash,
        std::chrono::duration<double, std::milli>(translate_end - translate_start).count(),
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - translate_end)
            .count(),
        spv.size());

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.sw_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::StoredListing PipelineCache::ReadStoredListing(u64 pgm_hash) {
    return ReadStoredListing(pgm_hash, Storage::DataBase::Instance());
}

PipelineCache::StoredListing PipelineCache::ReadStoredListing(u64 pgm_hash,
                                                              Storage::DataBase& store) {
    // Each permutation is stored as <perm hash>.meta (key and shader info) and
    // <program hash>_<index>.spv; indices are dense from 0 (PERF-013 numbers runtime ones after
    // the stored ones), so the listing stops after a run of missing indices. Only file reads, so
    // the PERF-038 reader thread can do it too.
    StoredListing listing{.pgm_hash = pgm_hash};
    u32 missing = 0;
    for (size_t perm_idx = 0; perm_idx < 512 && missing < 8; ++perm_idx) {
        const auto name = fmt::format("{:#018x}", HashCombine(pgm_hash, perm_idx));
        if (!store.Exists(Storage::BlobType::ShaderMeta, name)) {
            ++missing;
            continue;
        }
        missing = 0;
        auto& entry = listing.entries.emplace_back();
        entry.perm_idx = perm_idx;
        store.Load(Storage::BlobType::ShaderMeta, name, entry.meta);
        entry.has_spv = store.Exists(Storage::BlobType::ShaderBinary,
                                     fmt::format("{:#018x}_{}", pgm_hash, perm_idx));
    }
    return listing;
}

void PipelineCache::ApplyStoredListing(StoredListing&& listing, Program& program) {
    static u64 programs = 0;
    static u64 listed = 0;
    static u64 stale = 0;
    const u64 pgm_hash = listing.pgm_hash;
    for (auto& entry : listing.entries) {
        const size_t perm_idx = entry.perm_idx;
        auto info = std::make_unique<Shader::Info>();
        Shader::StageSpecialization spec{};
        spec.info = info.get();
        size_t stored_idx{};
        Serialization::Archive ar{std::move(entry.meta)};
        // Entries of another format version, cut short, or for another program are skipped.
        bool valid = false;
        try {
            Common::RecoverableScope recoverable;
            valid = ar.SizeBytes() >= 24 && LoadShaderMeta(ar, *info, spec, stored_idx);
        } catch (const Common::RecoverableFailure&) {
            valid = false;
        }
        if (!valid || stored_idx != perm_idx || info->pgm_hash != pgm_hash) {
            ++stale;
            continue;
        }
        // D10: stored before D2, the byte uses_direct takes was padding.
        info->uses_direct = std::ranges::any_of(info->buffers, [](const auto& buffer) {
            return buffer.buffer_type == Shader::BufferType::DirectTable;
        });
        auto& end = stored_perm_end[pgm_hash];
        end = std::max(end, perm_idx + 1);
        // FIX-018/020: translated before a translator fix this program needs.
        if (const auto fixed = else_scope_fixed.find(pgm_hash);
            fixed != else_scope_fixed.end() && perm_idx < fixed->second.boundary) {
            continue;
        }
        if (!entry.has_spv) {
            continue;
        }
        if (perm_idx < program.modules.size() && program.modules[perm_idx].info) {
            continue;
        }
        program.InsertPermut({}, std::move(spec), std::move(info), perm_idx);
        program.modules[perm_idx].stored = true;
        ++listed;
    }
    if (++programs % 500 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-032: shader store: {} programs looked up, {} stored permutations "
                    "listed, {} entries of another version skipped",
                    programs, listed, stale);
    }
}

void PipelineCache::ListStoredPermutations(u64 pgm_hash, Program& program) {
    ApplyStoredListing(ReadStoredListing(pgm_hash), program);
}

// PERF-038: one stored pipeline, read and parsed by the reader thread.
struct PipelineCache::PrewarmRecord {
    GraphicsPipelineKey key{};
    GraphicsPipeline::SerializationSupport sdata{};
    std::array<u64, MaxShaderStages> pgm{};
    std::array<size_t, MaxShaderStages> perm{};
    std::array<std::vector<u32>, MaxShaderStages> spv{};
    std::vector<StoredListing> listings;
};

void PipelineCache::PrewarmRead(std::stop_token stop) {
    Common::SetCurrentThreadName("shadGT:PrewarmRead");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
#ifdef RYTY_GPU
    // PERF-102: the game waits on the splash screen until the precompile is done.
    if (!g_precompile_hold.exchange(true)) {
        Common::GuestClock::TakeHold();
        LOG_WARNING(Render_Vulkan, "PERF-102: the game waits on the splash screen for the precompile");
    }
#endif
    auto& store = Storage::DataBase::Instance();
    RebuildFromPrograms(); // PB-1
    // The profile file is written only when missing (by the removed precompile), so it can be
    // older than most entries: GT Sport's is from October 6, the store's entries from October 6
    // to 9. A draw's first use loads the same stored SPIR-V without checking it (PERF-032), so
    // a background build makes the pipeline that draw would make; differences are only logged.
    std::vector<u8> stored_profile;
    if (store.Exists(Storage::BlobType::ShaderProfile, "profile")) {
        store.Load(Storage::BlobType::ShaderProfile, "profile", stored_profile);
    }
    if (!stored_profile.empty()) {
        std::string differences;
        const auto* current = reinterpret_cast<const u8*>(&profile);
        for (size_t i = 0; i < std::min(stored_profile.size(), sizeof(profile)); ++i) {
            if (stored_profile[i] != current[i]) {
                differences += fmt::format(" [{}] {:#x}->{:#x}", i, stored_profile[i], current[i]);
            }
        }
        if (!differences.empty() || stored_profile.size() != sizeof(profile)) {
            LOG_WARNING(Render_Vulkan,
                        "PERF-038: the store's profile file ({} bytes) differs from this "
                        "profile ({} bytes):{}",
                        stored_profile.size(), sizeof(profile), differences);
        }
    }
    auto keys = store.ListBlobs(Storage::BlobType::PipelineKey);
    std::erase_if(keys, [](const auto& key) { return !key.first.starts_with("g_"); });
    // Most recently used first: a pipeline's key is stored again each session that uses it.
    std::ranges::sort(keys, [](const auto& a, const auto& b) { return a.second > b.second; });
    if (const char* limit = std::getenv("SHADGT_PREWARM_LIMIT"); limit && *limit) {
        keys.resize(std::min<size_t>(keys.size(), std::strtoull(limit, nullptr, 10)));
    }
    prewarm_total = keys.size();
    g_precompile_total = keys.size(); // PERF-098
    g_precompile_done = 0;
    LOG_WARNING(Render_Vulkan, "PERF-038: building {} stored pipelines in the background",
                keys.size());
    std::unordered_set<u64> listed;
    u64 skipped = 0;
    for (const auto& [name, time] : keys) {
        if (stop.stop_requested()) {
            return;
        }
        ++prewarm_read;
        auto record = std::make_unique<PrewarmRecord>();
        bool valid = false;
        try {
            Common::RecoverableScope recoverable;
            std::vector<u8> blob;
            store.Load(Storage::BlobType::PipelineKey, name, blob);
            valid = LoadStoredGraphicsPipeline(std::move(blob), record->key, record->sdata);
            // Only pipelines whose every stage input is stored (see InstallPrewarmRecord).
            valid = valid && record->key.stage_hashes[u32(SwStage::Vertex)] != 0 &&
                    record->key.stage_hashes[u32(SwStage::Fragment)] != 0;
            for (u32 stage = 0; valid && stage < MaxShaderStages; ++stage) {
                const u64 perm_hash = record->key.stage_hashes[stage];
                if (perm_hash == 0) {
                    continue;
                }
                std::vector<u8> meta_blob;
                store.Load(Storage::BlobType::ShaderMeta, fmt::format("{:#018x}", perm_hash),
                           meta_blob);
                // Only the program hash and index: the full decode is the GPU thread's
                // (ApplyStoredListing), as it registers SRT walker code.
                u64 pgm_hash{};
                size_t perm_idx{};
                if (meta_blob.size() < 24 ||
                    !PeekShaderMeta(std::move(meta_blob), pgm_hash, perm_idx) ||
                    HashCombine(pgm_hash, perm_idx) != perm_hash) {
                    valid = false;
                    break;
                }
                record->pgm[stage] = pgm_hash;
                record->perm[stage] = perm_idx;
                if (listed.insert(pgm_hash).second) {
                    record->listings.push_back(ReadStoredListing(pgm_hash));
                }
                store.Load(Storage::BlobType::ShaderBinary,
                           fmt::format("{:#018x}_{}", pgm_hash, perm_idx), record->spv[stage]);
            }
        } catch (const std::exception&) {
            // RecoverableFailure, or a corrupt entry (std::bitset rejects bad text).
            valid = false;
        }
        if (!valid) {
            ++skipped;
            ++g_precompile_done; // PERF-098
            continue;
        }
        std::unique_lock lk{prewarm_mutex};
        prewarm_cv.wait(lk, stop, [this] { return prewarm_records.size() < 32; });
        if (stop.stop_requested()) {
            return;
        }
        prewarm_records.push_back(std::move(record));
        ++prewarm_queued;
    }
    LOG_WARNING(Render_Vulkan,
                "PERF-038: reader done: {} stored pipelines read, {} skipped (other key version, "
                "missing stages or files)",
                keys.size(), skipped);
    prewarm_reader_done = true;
    if (keys.empty()) {
        ReleasePrecompileHold(); // nothing stored yet (first start)
    }
}

void PipelineCache::IdlePrewarm(bool splash) {
    if (!prewarm_enabled) {
        ReleasePrecompileHold();
        return;
    }
    // PERF-102: released once every stored pipeline is built (or after 15 minutes at most).
    static const auto first_call = std::chrono::steady_clock::now();
    if (PrecompileHoldsGame() &&
        ((prewarm_reader_done && !GetPrecompileProgress().active) ||
         std::chrono::steady_clock::now() - first_call > std::chrono::minutes{15})) {
        ReleasePrecompileHold();
    }
    if (build_workers) {
        build_workers->SetBoost(splash);
    } else if (splash) {
        build_workers = std::make_unique<PipelineBuildWorkers>();
        build_workers->SetBoost(true);
    }
    std::scoped_lock lk{lookup_mutex};
    InstallPrewarmed(splash);
}

void PipelineCache::InstallPrewarmed(bool force) {
    if (!prewarm_enabled) {
        return;
    }
    using namespace std::chrono_literals;
    const auto start = std::chrono::steady_clock::now();
    ReportPrewarm(start);
    // Not while new pipelines are being needed: the draws come first (PERF-098: unless forced,
    // while the splash screen shows).
    if (prewarm_queued.load(std::memory_order_relaxed) == 0 ||
        (!force && start - last_pipeline_miss < 2s)) {
        return;
    }
    // The members a key is computed in are restored, so the draw sees what it would have.
    const auto saved_key = S().graphics_key;
    const auto saved_infos = S().infos;
    const auto saved_modules = S().modules;
    const auto saved_runtime_infos = S().runtime_infos;
    auto* const saved_fetch = S().fetch_shader;
    const auto budget = start + (force ? 20000us : 300us);
    do {
        std::unique_ptr<PrewarmRecord> record;
        {
            std::scoped_lock lk{prewarm_mutex};
            if (prewarm_records.empty()) {
                break;
            }
            record = std::move(prewarm_records.front());
            prewarm_records.pop_front();
        }
        --prewarm_queued;
        prewarm_cv.notify_one();
        if (InstallPrewarmRecord(*record)) {
            ++prewarm_stats.installed;
        } else {
            ++prewarm_stats.skipped;
            ++g_precompile_done; // PERF-098
        }
    } while (std::chrono::steady_clock::now() < budget);
    S().graphics_key = saved_key;
    S().infos = saved_infos;
    S().modules = saved_modules;
    S().runtime_infos = saved_runtime_infos;
    S().fetch_shader = saved_fetch;
    prewarm_stats.ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

bool PipelineCache::InstallPrewarmRecord(PrewarmRecord& record) {
    for (auto& listing : record.listings) {
        auto [it, is_new] = program_cache.try_emplace(listing.pgm_hash);
        if (is_new) {
            it.value() = std::make_unique<Program>();
            ApplyStoredListing(std::move(listing), *it.value());
        }
    }
    if (graphics_pipelines.contains(record.key) || pending_builds.contains(record.key)) {
        return false;
    }
    const bool tess_emulated = record.key.prim_type == AmdGpu::PrimitiveType::RectList ||
                               record.key.prim_type == AmdGpu::PrimitiveType::QuadList;
    S().infos.fill(nullptr);
    S().modules.fill(nullptr);
    S().fetch_shader = nullptr;
    for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
        if (record.pgm[stage] == 0) {
            continue;
        }
        const auto it = program_cache.find(record.pgm[stage]);
        if (it == program_cache.end() || record.perm[stage] >= it->second->modules.size()) {
            return false;
        }
        auto& permutation = it->second->modules[record.perm[stage]];
        if (!permutation.info) {
            return false;
        }
        if (permutation.stored) {
            // The same SPIR-V a draw's first use would load (PERF-032), read on the reader thread.
            if (!IsCompleteSpirv(record.spv[stage])) {
                return false;
            }
            permutation.module = CompileSPV(record.spv[stage], instance.GetDevice());
            permutation.stored = false;
        }
        if (!permutation.module) {
            return false;
        }
        // Tessellation helpers are made from the vertex outputs, which old metadata lacks; a draw
        // recovers them from the guest code, a background build cannot.
        if (tess_emulated && stage == u32(SwStage::Vertex) &&
            !permutation.info->attribute_flags_known) {
            return false;
        }
        // A mip fallback indexed at runtime takes its descriptor count from the draw's T#.
        if (std::ranges::any_of(permutation.info->images, [](const auto& image) {
                return image.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex;
            })) {
            return false;
        }
        S().infos[stage] = permutation.info.get();
        S().modules[stage] = permutation.module;
        // The runtime info a draw passes is the one the permutation was specialized with.
        S().runtime_infos[stage] = permutation.spec.runtime_info;
        if (auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
            S().fetch_shader = &fetch;
        }
    }
    S().graphics_key = record.key;
    auto build = MakePipelineBuild();
    build->preloading = true;
    build->sdata = record.sdata;
    pending_builds.emplace(record.key, build);
    QueueBackgroundBuild(build);
    return true;
}

bool PipelineCache::NeverSkipped(const std::array<const Shader::Info*, MaxShaderStages>& infos) {
    // Keep going: rendering must not wait for a cosmetic pipeline.
    if (CompileMode() == 2) return false;
    // FIX-064: tessellated draws wait for their pipeline. On GT Sport's race summary they are
    // the driver figure; skipped while each of its pipelines built, parts of it blinked out
    // frame after frame. -DisablePerf 88 skips them like other draws.
    // FIX-115: not during a race, though: there a tessellated car or road part that builds for
    // 0.3-2 s stopped the race that long. It shows a moment later instead (the quick wait still
    // draws the ones that build fast). -DisablePerf 138 waits in races too.
    static const bool wait_for_tessellation = Common::PerfFeatureEnabled(88);
    static const bool skip_in_races = Common::PerfFeatureEnabled(138);
#ifdef RYTY_GPU
    if (skip_in_races && RytyGtsInRace()) {
        return false;
    }
#endif
    return wait_for_tessellation && infos[u32(Shader::SwStage::TessellationControl)] != nullptr;
}

bool PipelineCache::WritesMemory(const std::array<const Shader::Info*, MaxShaderStages>& infos) {
    // PERF-076: a draw that writes buffers or images is never skipped. GT Sport's results screen
    // writes data later draws read as vertex descriptors; skipping those draws left garbage
    // there and the renderer ran out of memory on it.
    for (const auto* info : infos) {
        if (!info) {
            continue;
        }
        if (info->has_storage_images) {
            return true;
        }
        for (const auto& buffer : info->buffers) {
            if (buffer.is_written) {
                return true;
            }
        }
        for (const auto& image : info->images) {
            if (image.is_written) {
                return true;
            }
        }
    }
    return false;
}

bool PipelineCache::AsyncPipelines() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_ASYNC_PIPELINES");
#ifdef RYTY_GPU
        // PERF-100: GPU.compile_freeze (on by default) pauses the game while a new pipeline
        // compiles instead of skipping its draws.
        // PERF-103: GPU.compile_mode 0 (pause) waits; slow motion and keep going build in
        // the background.
        const bool by_default = CompileMode() != 0;
#else
        const bool by_default = false;
#endif
        const bool on = env ? env[0] != '0' : by_default;
        return on && Common::PerfFeatureEnabled(76);
    }();
    return enabled;
}

void PipelineCache::ReportPrewarm(std::chrono::steady_clock::time_point now) {
    auto& stats = prewarm_stats;
    if (now - stats.since < std::chrono::seconds{2}) {
        return;
    }
    if (async_stats.deferred + async_stats.installed + async_stats.skipped_draws +
            async_stats.skipped_translating !=
        0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-076 background pipelines in {:.1f} s: {} new ones built without a wait, "
                    "{} ready and installed, {} draws skipped meanwhile; SHD-005: {} draws "
                    "skipped while their shaders translated",
                    std::chrono::duration<double>(now - stats.since).count(),
                    async_stats.deferred, async_stats.installed, async_stats.skipped_draws,
                    async_stats.skipped_translating);
        async_stats = {};
    }
    const u64 built = prewarm_built.load();
    if (stats.installed + stats.hits + stats.cancelled + stats.promoted != 0 ||
        built != stats.built_before) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-038 background builds in {:.1f} s: {} queued ({} skipped, {:.1f} ms on "
                    "this thread), {} built ({} in all), {} used by draws without waiting, {} "
                    "built for a draw instead, {} moved ahead by the read-ahead; {} waiting; "
                    "reader {}/{}{}",
                    std::chrono::duration<double>(now - stats.since).count(), stats.installed,
                    stats.skipped, stats.ms, built - stats.built_before, built, stats.hits,
                    stats.cancelled, stats.promoted,
                    build_workers ? build_workers->BackgroundQueued() : 0, prewarm_read.load(),
                    prewarm_total.load(), prewarm_reader_done.load() ? " (done)" : "");
    }
    const u64 built_now = built;
    stats = {};
    stats.built_before = built_now;
    stats.since = now;
}

vk::ShaderModule PipelineCache::LoadStoredModule(u64 pgm_hash, size_t perm_idx) {
    return LoadStoredModule(pgm_hash, perm_idx, Storage::DataBase::Instance());
}

vk::ShaderModule PipelineCache::LoadStoredModule(u64 pgm_hash, size_t perm_idx,
                                                 Storage::DataBase& store) {
    static u64 loaded = 0;
    std::vector<u32> spv;
    store.Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx), spv);
    if (!IsCompleteSpirv(spv)) {
        static u32 reports = 0;
        if (reports++ < 20) {
            LOG_WARNING(Render_Vulkan,
                        "PERF-032: stored shader {:#x} permutation {} is incomplete; translating "
                        "it again",
                        pgm_hash, perm_idx);
        }
        return {};
    }
    if (++loaded % 500 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-032: {} shader S().modules loaded from the store instead of translated",
                    loaded);
    }
    return CompileSPV(spv, instance.GetDevice());
}

void PipelineCache::DumpFailedShader(std::span<const u32> code, u64 hash, Shader::HwStage stage) {
    // FIX-034: the guest code of a shader that could not be translated, raw and as a listing,
    // in the log folder (once per program), so it can be studied and tested offline.
    if (!failed_shaders.insert(hash).second) {
        return;
    }
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "failed_shaders";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto name = fmt::format("{}_{:#x}", stage, hash);
    Common::FS::IOFile{dir / (name + ".bin"), Common::FS::FileAccessMode::Create}.WriteSpan(code);
    std::ofstream{dir / (name + ".txt")} << Shader::ListGcnCode(code);
}

bool IsWatchedShader(u64 hash) {
    static const std::vector<u64> watched = [] {
        // The Nurburgring grass: vertex generator, blade generator, and the pass before them.
        // DIAG-039: the dealership shader whose buffer V#s read as float constants.
        std::vector<u64> hashes{0xab6a2d10, 0xaa3822a3, 0x766d0f18, 0xae32f77f};
        if (const char* env = std::getenv("SHADGT_WATCH_SHADERS"); env && *env) {
            hashes.clear();
            const std::string list{env};
            for (size_t start = 0; start < list.size();) {
                const size_t end = std::min(list.find(',', start), list.size());
                hashes.push_back(std::stoull(list.substr(start, end - start), nullptr, 16));
                start = end + 1;
            }
        }
        return hashes;
    }();
    return std::ranges::find(watched, hash) != watched.end();
}

void PipelineCache::LoadElseScopeFixed() {
    const auto& serial = Common::Singleton<Common::ElfInfo>::Instance()->GameSerial();
    else_scope_fixed_path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                            fmt::format("{}.else-scope-fix", serial);
    std::ifstream file{else_scope_fixed_path};
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream fields{line};
        u64 hash{};
        size_t boundary{};
        u32 fixes = 1; // Lines written before FIX-020 have no mask.
        if (fields >> std::hex >> hash >> std::dec >> boundary) {
            fields >> fixes;
            else_scope_fixed[hash] = {boundary, fixes};
        }
    }
    if (!else_scope_fixed.empty()) {
        LOG_INFO(Render_Vulkan, "FIX-018: {} programs already translated with the else-scope fix",
                 else_scope_fixed.size());
    }
}

void PipelineCache::SaveElseScopeFixed() const {
    if (else_scope_fixed_path.empty()) {
        return;
    }
    std::ofstream file{else_scope_fixed_path, std::ios::trunc};
    for (const auto& [hash, program] : else_scope_fixed) {
        file << std::hex << hash << ' ' << std::dec << program.boundary << ' ' << program.fixes
             << '\n';
    }
}

// FIX-058: GT Sport's race summary draws tessellated objects whose tessellation constants are
// garbage (strides of hundreds of MB). Hull shaders translated from them declared arrays of 200
// million entries, and the driver ran out of memory compiling their pipelines (the process went
// from 11 GB to 100 GB in seconds; the driver figure flickered while they built). Constants are
// sane when they fit the 64 KB of LDS they describe; draws with others are skipped.
// -DisablePerf 82 draws them.
// DIAG-063: the tessellation-constant V#s (HS user data 7..10) an indirect buffer sets, read
// from a copy of it.
static std::vector<VAddr> TessBasesInIb(const u32* data, u32 dwords) {
    std::vector<u32> ib(dwords);
    Common::CopyGuest(ib.data(), data, ib.size() * sizeof(u32));
    std::vector<VAddr> bases;
    std::array<u32, 16> hs{};
    constexpr u32 HsUserData = 0x2D0C;
    const auto apply = [&](u32 first, std::span<const u32> values) {
        bool touched = false;
        for (u32 i = 0; i < values.size(); ++i) {
            if (first + i >= HsUserData + 7 && first + i < HsUserData + 11) {
                hs[first + i - HsUserData] = values[i];
                touched = true;
            }
        }
        if (touched && hs[9] == 3) {
            bases.push_back((VAddr(hs[8] & 0xffff) << 32) | hs[7]);
        }
    };
    std::span<const u32> rest{ib};
    while (!rest.empty()) {
        const u32 type = rest.front() >> 30;
        if (type == 2) {
            rest = rest.subspan(1);
        } else if (type == 0) {
            const u32 first = rest.front() & 0xffff;
            const u32 count = ((rest.front() >> 16) & 0x3fff) + 1;
            if (count + 1 > rest.size()) {
                break;
            }
            apply(first, rest.subspan(1, count));
            rest = rest.subspan(count + 1);
        } else if (type == 3) {
            const u32 count = ((rest.front() >> 16) & 0x3fff) + 1;
            const u32 opcode = (rest.front() >> 8) & 0xff;
            if (count + 1 > rest.size()) {
                break;
            }
            if (opcode == 0x76 && count >= 1) { // SET_SH_REG
                apply(0x2C00 + (rest[1] & 0xffff), rest.subspan(2, count - 1));
            }
            rest = rest.subspan(count + 1);
        } else {
            break;
        }
    }
    return bases;
}

// DIAG-063: whether the indirect buffer a draw came from changes after it was decoded, and the
// constant V#s it sets before and after.
static void WatchIbAfterDecode(u32 n, bool sane, VAddr base) {
    const auto [ib, dwords] = AmdGpu::Liverpool::CurrentDecodeIb();
    if (!ib || dwords == 0 || dwords > 4'000'000) {
        LOG_WARNING(Render_Vulkan, "DIAG-063: {} draw {}: not inside an indirect buffer",
                    sane ? "sane" : "garbage", n);
        return;
    }
    std::vector<u32> before(dwords);
    Common::CopyGuest(before.data(), ib, before.size() * sizeof(u32));
    const auto bases_before = TessBasesInIb(ib, dwords);
    static std::atomic<u32> dumps{};
    if ((!sane || n == 5000) && dumps.fetch_add(1) < 6) {
        // DIAG-064: the indirect buffer and 64 KB around the constants, for offline study.
        const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
        const auto name = fmt::format("tess_{}_{}", sane ? "sane" : "garbage", n);
        if (std::ofstream f{dir / (name + "_ib.bin"), std::ios::binary}) {
            f.write(reinterpret_cast<const char*>(before.data()), before.size() * sizeof(u32));
        }
        std::vector<u8> around(64_KB);
        const VAddr from = base - 32_KB;
        Common::CopyGuest(around.data(), reinterpret_cast<const void*>(from), around.size());
        if (std::ofstream f{dir / (name + "_mem.bin"), std::ios::binary}) {
            f.write(reinterpret_cast<const char*>(around.data()), around.size());
        }
        LOG_WARNING(Render_Vulkan, "DIAG-064: {} draw {}: indirect buffer {:#x} ({} dwords) and "
                                   "memory from {:#x} saved as {}_*.bin",
                    sane ? "sane" : "garbage", n, reinterpret_cast<VAddr>(ib), dwords, from,
                    name);
    }
    std::thread([n, sane, base, ib, dwords, before = std::move(before), bases_before] {
        std::string timeline;
        for (const u32 ms : {1u, 3u, 10u, 30u}) {
            std::this_thread::sleep_until(std::chrono::steady_clock::now() +
                                          std::chrono::milliseconds(ms));
            std::vector<u32> now(dwords);
            Common::CopyGuest(now.data(), ib, now.size() * sizeof(u32));
            size_t first_diff = dwords;
            for (size_t i = 0; i < dwords; ++i) {
                if (now[i] != before[i]) {
                    first_diff = i;
                    break;
                }
            }
            std::array<u32, 7> c{};
            Common::CopyGuest(c.data(), reinterpret_cast<const void*>(base), sizeof(c));
            const bool constants_sane =
                c[2] != 0 && std::ranges::all_of(c, [](u32 x) { return x <= 64_KB; });
            timeline += fmt::format(" | +{} ms: ib {}, constants {}", ms,
                                    first_diff == dwords
                                        ? std::string("same")
                                        : fmt::format("changed from dword {}", first_diff),
                                    constants_sane ? "sane" : "garbage");
        }
        const auto bases_after = TessBasesInIb(ib, dwords);
        const auto list = [](const std::vector<VAddr>& v) {
            std::string out;
            for (size_t i = 0; i < std::min<size_t>(v.size(), 8); ++i) {
                out += fmt::format(" {:#x}", v[i]);
            }
            return out + (v.size() > 8 ? fmt::format(" (+{})", v.size() - 8) : "");
        };
        LOG_WARNING(Render_Vulkan,
                    "DIAG-063: {} draw {} (constants {:#x}) in indirect buffer {:#x} ({} "
                    "dwords){}; constant V#s in it at decoding:{}; 30 ms later:{}",
                    sane ? "sane" : "garbage", n, base, reinterpret_cast<VAddr>(ib), dwords,
                    timeline, list(bases_before), list(bases_after));
    }).detach();
}

static bool SaneTessConstants(const Shader::Info& hull, const AmdGpu::Liverpool* liverpool,
                              bool read_ahead) {
    static const bool check = Common::PerfFeatureEnabled(82);
    if (!check || hull.tess_consts_dword_offset < 0) {
        return true;
    }
    const auto buf = hull.ReadUdReg<AmdGpu::Buffer>(static_cast<u32>(hull.tess_consts_ptr_base),
                                                    static_cast<u32>(hull.tess_consts_dword_offset));
    const VAddr base = buf.base_address;
    Shader::TessellationDataConstantBuffer c{};
#ifdef RYTY_GPU
    if (base == 0 || !Core::RytyMemory::IsGuestReadable(base, sizeof(c))) {
        return false;
    }
#endif
    constexpr u32 Lds = 64_KB;
    const auto is_sane = [&] {
        return c.ls_stride <= Lds && c.hs_cp_stride <= Lds && c.num_patches != 0 &&
               c.num_patches <= Lds && c.hs_output_base <= Lds && c.patch_const_size <= Lds &&
               c.patch_const_base <= Lds && c.patch_output_size <= Lds;
    };
    Common::CopyGuest(&c, reinterpret_cast<const void*>(base), sizeof(c));
    const bool sane = is_sane();
    {
        // DIAG-060: sane and garbage constants per hull program, every 5 s.
        static std::mutex stats_mutex;
        static std::unordered_map<u64, std::pair<u32, u32>> per_hull;
        static u32 garbage_drawn{};
        static u32 garbage_read_ahead{};
        if (!sane) {
            ++(read_ahead ? garbage_read_ahead : garbage_drawn);
        }
        static auto last = std::chrono::steady_clock::now();
        std::scoped_lock lk{stats_mutex};
        auto& [good, bad] = per_hull[hull.pgm_hash];
        ++(sane ? good : bad);
        if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{5}) {
            last = now;
            std::string line;
            for (const auto& [hash, counts] : per_hull) {
                if (counts.second != 0) {
                    line += fmt::format(" {:#x}={}/{}", hash, counts.first, counts.second);
                }
            }
            if (!line.empty()) {
                LOG_WARNING(Render_Vulkan,
                            "DIAG-060: hulls with garbage constants (sane/garbage):{}; decoding "
                            "submission {} of {}; garbage in {} draws, {} read-ahead states",
                            line, liverpool ? liverpool->DecodingSubmit() : 0,
                            liverpool ? liverpool->SubmittedCount() : 0, garbage_drawn,
                            garbage_read_ahead);
                garbage_drawn = 0;
                garbage_read_ahead = 0;
            }
            per_hull.clear();
        }
    }
#ifdef RYTY_GPU
    if (sane && !read_ahead) {
        // DIAG-067: who writes these constants next, and when (SHADGT_WATCH_TESS=<ms>).
        static const u32 watch_ms = [] {
            const char* env = std::getenv("SHADGT_WATCH_TESS");
            return env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;
        }();
        static auto next_watch = std::chrono::steady_clock::now();
        if (watch_ms != 0 && std::chrono::steady_clock::now() >= next_watch) {
            next_watch = std::chrono::steady_clock::now() + std::chrono::milliseconds{watch_ms + 50};
            RytyWatchWrites(base, watch_ms,
                            fmt::format("tessellation constants of hs {:#x} decoded in submission "
                                        "{}; {}",
                                        hull.pgm_hash, liverpool ? liverpool->DecodingSubmit() : 0,
                                        AmdGpu::Liverpool::DescribeEops()));
        }
    }
#endif
    if (sane) {
        // DIAG-059: what sane constants and their V#s look like, to compare with garbage ones.
        static std::atomic<u32> sane_logged{};
        if (const u32 n = ++sane_logged; n <= 40 || n % 5000 == 0) {
            if (n % 5000 == 0 && std::getenv("SHADGT_TESS_SUBMIT_CHECK")) {
                WatchIbAfterDecode(n, true, base);
            }
            const auto dump = liverpool ? liverpool->FindRecentConstDump(base, sizeof(c))
                                        : std::nullopt;
            const auto rec = AmdGpu::Liverpool::FindTessSubmit(base);
            LOG_WARNING(Render_Vulkan, "DIAG-059: sane constants {}: {}; DIAG-061: {} at "
                                       "submission {}, decoding {} of {}; DIAG-066: {}",
                        n,
                        dump ? fmt::format("CE dump {} of {}", dump->sequence,
                                           liverpool->ConstDumpSequence())
                             : std::string("not from a recent CE dump"),
                        rec ? (rec->sane ? "sane" : "garbage") : "unseen", rec ? rec->submit : 0,
                        liverpool ? liverpool->DecodingSubmit() : 0,
                        liverpool ? liverpool->SubmittedCount() : 0,
                        AmdGpu::Liverpool::DescribeEops());
            LOG_WARNING(Render_Vulkan,
                        "DIAG-059: sane tessellation constants {} of hs {:#x} at {:#x}: ls stride "
                        "{:#x}, cp stride {:#x}, patches {:#x}, output base {:#x}, patch output "
                        "size {:#x}; V# {:08x} {:08x} {:08x} {:08x}",
                        n, hull.pgm_hash, base, c.ls_stride, c.hs_cp_stride, c.num_patches,
                        c.hs_output_base, c.patch_output_size,
                        reinterpret_cast<const u32*>(&buf)[0], reinterpret_cast<const u32*>(&buf)[1],
                        reinterpret_cast<const u32*>(&buf)[2], reinterpret_cast<const u32*>(&buf)[3]);
        }
    }
    if (!sane) {
        static std::atomic<u32> logged{};
        if (const u32 n = ++logged; n <= 60 || n % 100 == 0) {
            if (n <= 40 && std::getenv("SHADGT_TESS_SUBMIT_CHECK")) {
                WatchIbAfterDecode(n, false, base);
            }
            auto* rasterizer = Core::Memory::Instance()->GetRasterizer();
            const bool gpu_written =
                rasterizer && rasterizer->GetBufferCache().IsRegionGpuModified(base, sizeof(c));
            // Where the V# itself was read: a table in memory the user data points to, or the
            // user data registers.
            VAddr table = 0;
            if (u32(hull.tess_consts_ptr_base) != Shader::IR::NumScalarRegs &&
                u32(hull.tess_consts_ptr_base) + 1 < hull.user_data.size()) {
                std::memcpy(&table, &hull.user_data[u32(hull.tess_consts_ptr_base)], sizeof(table));
                table &= 0xFFFFFFFFFFFFULL;
                table += u64(hull.tess_consts_dword_offset) * 4;
            }
            const bool table_gpu_written =
                rasterizer && table != 0 &&
                rasterizer->GetBufferCache().IsRegionGpuModified(table, 16);
            const auto dump = liverpool ? liverpool->FindRecentConstDump(base, sizeof(c))
                                        : std::nullopt;
            LOG_WARNING(Render_Vulkan,
                        "FIX-058: constants of draw {}: {}", n,
                        dump ? fmt::format("CE dump {} of {} (ce {} de {}) at {:#x}+{:#x}",
                                           dump->sequence, liverpool->ConstDumpSequence(),
                                           dump->ce_count, dump->de_count, dump->address,
                                           dump->size)
                             : std::string("not from a recent CE dump"));
            if (liverpool) {
                // DIAG-061: the same constants when the game submitted the draw.
                const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
                const auto cmd = liverpool->FindRecentCmdBuffer(base, sizeof(c));
                const auto rec = AmdGpu::Liverpool::FindTessSubmit(base);
                LOG_WARNING(Render_Vulkan,
                            "DIAG-061: constants of draw {} {}; decoding submission {} of {}; "
                            "inside a recent command buffer: {}; DIAG-062: its indirect buffer "
                            "changed after submission: {}",
                            n,
                            rec ? fmt::format("were {} at submission {} ({:.1f} ms ago, ls stride "
                                              "{:#x}{})",
                                              rec->sane ? "sane" : "garbage", rec->submit,
                                              (now - rec->time_ns) / 1e6, rec->ls_stride,
                                              rec->in_cmd_buffer ? ", in its command buffer" : "")
                                : std::string("were not seen at any submission"),
                            liverpool->DecodingSubmit(), liverpool->SubmittedCount(),
                            cmd ? fmt::format("yes ({} of {})", cmd->sequence,
                                              liverpool->CmdBufferSequence())
                                : std::string("no"),
                            AmdGpu::Liverpool::CurrentIbChangedSinceSubmit());
            }
            // DIAG-065: the renderer's own recent writes over the constants (readbacks, fault
            // flushes, CP writes), newest last.
            LOG_WARNING(Render_Vulkan, "DIAG-065: emulator writes over the constants of draw {}:{}",
                        n, Core::Memory::Instance()->DescribeBackingWrites(base, 0));
            LOG_WARNING(Render_Vulkan, "DIAG-066: draw {} decoded in submission {} of {}; {}", n,
                        liverpool ? liverpool->DecodingSubmit() : 0,
                        liverpool ? liverpool->SubmittedCount() : 0,
                        AmdGpu::Liverpool::DescribeEops());
            LOG_WARNING(Render_Vulkan,
                        "FIX-058: V# of draw {} read from {} {:#x} (GPU-written: {}); user data "
                        "regs {} (ptr base {}, dword {})",
                        n, table ? "memory" : "user data", table, table_gpu_written,
                        hull.user_data.size(), u32(hull.tess_consts_ptr_base),
                        hull.tess_consts_dword_offset);
            LOG_WARNING(Render_Vulkan,
                        "FIX-058: skipping tessellated {} {} of hs {:#x}: constants at {:#x} "
                        "are garbage (ls stride {:#x}, cp stride {:#x}, patches {:#x}, output "
                        "base {:#x}, patch output size {:#x}); GPU-written: {}; V# {:08x} "
                        "{:08x} {:08x} {:08x}",
                        read_ahead ? "read-ahead state" : "draw", n, hull.pgm_hash, base,
                        c.ls_stride, c.hs_cp_stride, c.num_patches,
                        c.hs_output_base, c.patch_output_size, gpu_written,
                        reinterpret_cast<const u32*>(&buf)[0], reinterpret_cast<const u32*>(&buf)[1],
                        reinterpret_cast<const u32*>(&buf)[2], reinterpret_cast<const u32*>(&buf)[3]);
        }
    }
    return sane;
}

bool PipelineCache::SkipTranslationWait(SwStage sw_stage, const Shader::ShaderParams& params) {
    // SHD-005: as a race's draws are skipped while their new pipeline builds (PERF-076), a draw
    // whose new shader still translates on a worker is skipped too, instead of the command thread
    // waiting for it (GT Sport's first race on a track stopped for 100-300 ms whenever a few
    // dozen new shaders came at once). Not compute, and not a shader that writes memory (later
    // work reads what it writes). -DisablePerf 171 waits as before.
    static const bool enabled = Common::PerfFeatureEnabled(171);
    if (!enabled || regs_override != nullptr || sw_stage == SwStage::Compute ||
        !AsyncTranslation() || !AsyncPipelines() || MenusWait()) {
        return false;
    }
    auto [it, inserted] = code_writes_memory.try_emplace(params.hash, false);
    if (inserted) {
        it.value() = Shader::HasMemoryWrites(params.code);
    }
    return !it->second;
}

bool PipelineCache::AsyncTranslation() const {
    // FIX-078: -DisablePerf 102 translates on the command thread as before. Dumped, collected or
    // patched shaders need their permutation index while they are translated.
    static const bool enabled = Common::PerfFeatureEnabled(102) &&
                                !EmulatorSettings.IsDumpShaders() &&
                                !EmulatorSettings.IsShaderCollect() &&
                                !EmulatorSettings.IsPatchShaders();
    return enabled && translate_workers != nullptr;
}

bool PipelineCache::QueueTranslation(HwStage hw_stage, SwStage sw_stage,
                                     const Shader::ShaderParams& params,
                                     const Shader::RuntimeInfo& runtime_info,
                                     const Shader::Backend::Bindings& start) {
    auto& jobs = translate_jobs[params.hash];
    for (const auto& job : jobs) {
        if (job->hw_stage == hw_stage && job->runtime_info == runtime_info && job->start == start) {
            ++translate_stats.duplicate;
            return false;
        }
    }
    auto job = std::make_shared<TranslateJob>(runtime_info);
    job->hw_stage = hw_stage;
    job->sw_stage = sw_stage;
    job->hash = params.hash;
    std::ranges::copy(params.user_data, job->user_data.begin());
    job->code = params.code;
    job->start = start;
    jobs.push_back(job);
    ++translate_stats.queued;
    translate_workers->Queue(std::move(job));
    return true;
}

bool PipelineCache::TakeReadyTranslation(HwStage hw_stage, SwStage sw_stage, u64 hash,
                                         Program& program,
                                         const Shader::ShaderParams& params,
                                         const Shader::RuntimeInfo& runtime_info,
                                         const Shader::Backend::Bindings& binding, bool insert) {
    const auto it = ready_translations.find(hash);
    if (it == ready_translations.end()) {
        return false;
    }
    auto& ready = it.value();
    for (auto job_it = ready.begin(); job_it != ready.end(); ++job_it) {
        auto& job = *job_it;
        // Matched the way GetProgram matches a permutation: its specialization for this lookup.
        auto& info = *job->info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        { Common::PhaseTimer flat_timer{Common::Phase::SelFlatten}; info.RefreshFlatBuf(); }
        if (job->hw_stage != hw_stage || job->sw_stage != sw_stage ||
            Shader::StageSpecialization(info, runtime_info, profile, binding) != job->spec) {
            continue;
        }
        if (!insert) {
            return true;
        }
        size_t perm_idx = program.modules.size();
        if (const auto end = stored_perm_end.find(hash); end != stored_perm_end.end()) {
            perm_idx = std::max(perm_idx, end->second);
        }
        const u64 perm_hash = HashCombine(hash, perm_idx);
        RegisterShaderMeta(info, job->spec.fetch_shader_data, job->spec, perm_hash, perm_idx);
        const u64 spv_hash = XXH3_64bits(job->spv.data(), job->spv.size() * sizeof(u32));
        Vulkan::SetObjectName(instance.GetDevice(), job->module,
                              GetShaderName(job->hw_stage, hash, perm_idx));
        program.InsertPermut(job->module, std::move(job->spec), std::move(job->info), perm_idx);
        program.modules[perm_idx].spv_hash = spv_hash;
        RegisterShaderBinary(std::move(job->spv), hash, perm_idx);
        ++Common::GetWorkCounters().shaders_compiled;
        ++translate_stats.used;
        ready.erase(job_it);
        if (ready.empty()) {
            ready_translations.erase(it);
        }
        return true;
    }
    return false;
}

void PipelineCache::HarvestTranslations(u64 hash, bool wait) {
    const auto it = translate_jobs.find(hash);
    if (it == translate_jobs.end()) {
        return;
    }
    auto& jobs = it.value();
    if (wait) {
        const auto wait_start = std::chrono::steady_clock::now();
        bool waited = false;
        for (const auto& job : jobs) {
            if (!job->done) {
                // As CompileModule: the game never sees a PS4 wait for a shader.
                Common::GuestClock::CompilePause hold;
                RYDIAG_SCOPE(ShaderCompile, hash);
                std::unique_lock lk{job->mutex};
                job->cv.wait(lk, [&] { return job->done.load(); });
                waited = true;
            }
        }
        if (waited) {
            ++translate_stats.waited;
            translate_stats.wait_ms += std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - wait_start)
                                           .count();
        }
    }
    auto& ready = ready_translations[hash];
    const auto now = std::chrono::steady_clock::now();
    std::erase_if(jobs, [&](const std::shared_ptr<TranslateJob>& job) {
        if (!job->done) {
            return false;
        }
        if (!job->info || !job->module) {
            ++translate_stats.failed;
            return true;
        }
        translate_stats.translate_ms += job->translate_ms;
        job->spec = Shader::StageSpecialization(*job->info, job->runtime_info, profile, job->start);
        job->ready_since = now;
        ready.push_back(job);
        return true;
    });
    if (jobs.empty()) {
        translate_jobs.erase(it);
    }
    // Translations no draw took within 30 s came from states that were not drawn.
    std::erase_if(ready, [&](const std::shared_ptr<TranslateJob>& job) {
        if (now - job->ready_since < std::chrono::seconds{30} && ready.size() <= 32) {
            return false;
        }
        instance.GetDevice().destroyShaderModule(job->module);
        ++translate_stats.expired;
        return true;
    });
    auto& stats = translate_stats;
    if (now - stats.since >= std::chrono::seconds{2}) {
        if (stats.queued + stats.used != 0) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-078 shader translation in {:.1f} s: {} queued, {} used by draws "
                        "({:.1f} ms of translation done off this thread), {} never drawn, {} "
                        "duplicates, {} failed; {} draws waited for one ({:.1f} ms)",
                        std::chrono::duration<double>(now - stats.since).count(), stats.queued,
                        stats.used, stats.translate_ms, stats.expired, stats.duplicate,
                        stats.failed, stats.waited, stats.wait_ms);
        }
        stats = {};
        stats.since = now;
    }
}

std::optional<PipelineCache::Result> PipelineCache::TryLegacyProgram(
    HwStage hw_stage, const Shader::ShaderParams& params, const Shader::RuntimeInfo& runtime_info,
    Shader::Backend::Bindings& binding) {
    auto& store = Storage::DataBase::Legacy();
    auto [it, is_new] = legacy_programs.try_emplace(params.hash);
    if (is_new) {
        it.value() = std::make_unique<Program>();
        // Like ApplyStoredListing, without its bookkeeping of this store's indices.
        auto listing = ReadStoredListing(params.hash, store);
        for (auto& entry : listing.entries) {
            if (!entry.has_spv) {
                continue;
            }
            auto info = std::make_unique<Shader::Info>();
            Shader::StageSpecialization spec{};
            spec.info = info.get();
            size_t stored_idx{};
            Serialization::Archive ar{std::move(entry.meta)};
            bool valid = false;
            try {
                Common::RecoverableScope recoverable;
                valid = ar.SizeBytes() >= 24 && LoadShaderMeta(ar, *info, spec, stored_idx);
            } catch (const Common::RecoverableFailure&) {
                valid = false;
            }
            if (!valid || stored_idx != entry.perm_idx || info->pgm_hash != params.hash ||
                std::ranges::any_of(info->buffers, [](const auto& buffer) {
                    return buffer.buffer_type == Shader::BufferType::DirectTable;
                })) {
                continue;
            }
            info->uses_direct = false; // the byte was padding in old metas
            it.value()->InsertPermut({}, std::move(spec), std::move(info), entry.perm_idx);
            it.value()->modules[entry.perm_idx].stored = true;
        }
        ++legacy_stats.programs;
    }
    auto& program = *it.value();
    for (size_t perm_idx = 0; perm_idx < program.modules.size(); ++perm_idx) {
        auto& permutation = program.modules[perm_idx];
        if (!permutation.info) {
            continue;
        }
        auto& info = *permutation.info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        info.RefreshFlatBuf();
        const auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
        if (permutation.spec != spec) {
            static std::atomic<u32> logged{};
            if (logged++ < 40) {
                u32 index{};
                const char* reason = permutation.spec.FirstDifference(spec, index);
                LOG_WARNING(Render_Vulkan, "D10: old translation {} of {}_{:#x} differs: {}[{}]",
                            perm_idx, hw_stage, params.hash, reason ? reason : "none", index);
            }
            continue;
        }
        if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
            continue; // FIX-012 would store it again: left to the new translation
        }
        if (permutation.stored) {
            permutation.stored = false;
            permutation.module = LoadStoredModule(params.hash, perm_idx, store);
            if (!permutation.module) {
                permutation.info.reset();
                continue;
            }
        }
        info.AddBindings(binding);
        if (auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
            S().fetch_shader = &fetch;
        }
        ++legacy_stats.draws;
        ReportLegacy();
        // Its own pipeline keys: the module hashes of old translations differ from new ones.
        constexpr u64 LegacyTag = 0x4c45474143590000ULL;
        return std::make_tuple(&info, permutation.module,
                               HashCombine(params.hash, perm_idx) ^ LegacyTag);
    }
    return std::nullopt;
}

void PipelineCache::ReportLegacy() {
    const auto now = std::chrono::steady_clock::now();
    if (now - legacy_stats.since < std::chrono::seconds{2}) {
        return;
    }
    LOG_WARNING(Render_Vulkan,
                "D10: old-renderer shaders in 2 s: {} stage lookups used one, {} new translations "
                "queued for them, {} pipelines of old shaders, {} misses without one ({} programs "
                "read from the old store); pipelines not built yet: {} drawn with old shaders, "
                "{} old ones not built either, {} without old shaders",
                legacy_stats.draws, legacy_stats.queued, legacy_stats.pipelines,
                legacy_stats.no_legacy, legacy_stats.programs, legacy_stats.fallback_ok,
                legacy_stats.fallback_unbuilt, legacy_stats.fallback_none);
    const u64 programs = legacy_stats.programs;
    legacy_stats = {};
    legacy_stats.programs = programs;
    legacy_stats.since = now;
}

PipelineCache::Result PipelineCache::GetProgramForJob(HwStage hw_stage,
                                                      const Shader::ShaderParams& params,
                                                      const Shader::RuntimeInfo& runtime_info,
                                                      Shader::Backend::Bindings& binding) {
    // NAT-031: GetProgram for a game job thread (shared lock): only the programs and
    // permutations the threads above made are matched, in this thread's own copies of their
    // infos; nothing is created, loaded, harvested or remembered.
    const auto none = [] { return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0}); };
    const auto it_pgm = program_cache.find(params.hash);
    if (it_pgm == program_cache.end() || !else_scope_checked.contains(params.hash)) {
        S().lookup_miss = "program not used yet";
        return none();
    }
    const Program& program = *it_pgm->second;
    bool tess_checked = false;
    // The permutation the last search found first (R7), then all of them in order.
    const size_t first = program.last_match;
    for (size_t step = 0; step <= program.modules.size(); ++step) {
        const size_t perm_idx = step == 0 ? first : step - 1;
        if (perm_idx >= program.modules.size() || (step != 0 && perm_idx == first)) {
            continue;
        }
        const auto& permutation = program.modules[perm_idx];
        if (!permutation.info) {
            continue;
        }
        auto& info = MatchInfo(*permutation.info);
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        info.RefreshFlatBuf();
        if (hw_stage == HwStage::Hull && !tess_checked) {
            tess_checked = true; // FIX-058, as GetProgram
            if (!SaneTessConstants(info, liverpool, false)) {
                S().lookup_miss = "hull constants";
                return none();
            }
        }
        if (permutation.spec != Shader::StageSpecialization(info, runtime_info, profile, binding)) {
            continue;
        }
        if (permutation.stored) {
            S().lookup_miss = "stored translation not loaded";
            return none();
        }
        if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
            S().lookup_miss = "rect/quad list attribute flags";
            return none();
        }
        info.AddBindings(binding);
        if (const auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
            S().fetch_shader = const_cast<Shader::Gcn::FetchShaderData*>(&fetch);
        }
        return std::make_tuple(&info, permutation.module, HashCombine(params.hash, perm_idx));
    }
    S().lookup_miss = hw_stage == HwStage::Fragment ? "no fragment translation for these inputs"
                      : hw_stage == HwStage::Vertex ? "no vertex translation for these inputs"
                                                     : "no translation for these inputs";
    return none();
}

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    // SHD-002: a stage's bindings start where its own descriptor set does, whatever the stages
    // before it bound.
    binding = Shader::Backend::StageStart(u32(sw_stage));
    auto runtime_info = BuildRuntimeInfo(hw_stage, sw_stage);
    if (t_lookup_only) {
        return GetProgramForJob(hw_stage, params, runtime_info, binding); // NAT-031
    }
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>();
        if (use_stored_shaders) {
            ListStoredPermutations(params.hash, *it_pgm.value());
        }
    }
    // FIX-018: stored permutations of a program with an else after an empty if were translated
    // with the else running for every invocation. Translate those programs again, once; the new
    // permutations start after the stored ones, so their pipelines are new as well.
    const bool first_use = else_scope_checked.insert(params.hash).second;
    if (first_use && IsWatchedShader(params.hash)) {
        // DIAG-031: the guest code of watched shaders, raw and as a listing, in the log folder.
        const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
        const auto name = fmt::format("gcn_{:#x}", params.hash);
        Common::FS::IOFile{dir / (name + ".bin"), Common::FS::FileAccessMode::Create}.WriteSpan(
            params.code);
        std::ofstream{dir / (name + ".txt")} << Shader::ListGcnCode(params.code);
        LOG_WARNING(Render_Vulkan, "DIAG-031: wrote the code of {}_{:#x} ({} dwords) to {}",
                    hw_stage, params.hash, params.code.size(), (dir / (name + ".txt")).string());
    }
    // FIX-018/FIX-020: stored translations made before a translator fix that changes them.
    u32 needed_fixes = 0;
    if (first_use) {
        needed_fixes |= Shader::HasEmptyScopeBeforeElse(params.code) ? 1u : 0u;
        if (Shader::HasLegacyMad(params.code)) {
            needed_fixes |= 16u; // FIX-101
            LOG_INFO(Render_Vulkan, "FIX-101: {}_{:#x} uses legacy multiply-add", hw_stage,
                     params.hash);
        }
        const auto& wg = runtime_info.hw.cs.workgroup_size;
        if (hw_stage == HwStage::Compute && profile.subgroup_size < 64 &&
            wg[0] * wg[1] * wg[2] > 32 && Common::PerfFeatureEnabled(31) &&
            Shader::HasLaneReadsAndLoop(params.code)) {
            needed_fixes |= 8u; // FIX-020 v3 (conditions behind ConditionRef); v1/v2 were 2/4
        }
    }
    const auto known = else_scope_fixed.find(params.hash);
    const u32 applied_fixes = known != else_scope_fixed.end() ? known->second.fixes : 0u;
    if ((needed_fixes & ~applied_fixes) != 0) {
        auto& boundary = stored_perm_end[params.hash];
        boundary = std::max(boundary, it_pgm.value()->modules.size());
        if (boundary != 0) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-018/020: translating {}_{:#x} again (fixes {:#x}, {} stored "
                        "permutations)",
                        hw_stage, params.hash, needed_fixes, boundary);
            else_scope_fixed[params.hash] = {boundary, applied_fixes | needed_fixes};
            SaveElseScopeFixed();
            if (!it_pgm.value()->modules.empty()) {
                // Pipelines already built keep the old modules alive.
                retired_programs.push_back(std::move(it_pgm.value()));
                it_pgm.value() = std::make_unique<Program>();
            }
        }
    }

    auto& program = it_pgm.value();
    // FIX-078: translations made on the workers become permutations; a draw (not a read-ahead)
    // first waits for any still running, as one of them is likely the permutation it needs.
    // D10: with the fallback a draw does not wait for translations in progress (the old
    // translations draw meanwhile); only when there are none (fallback_nested).
    const bool skip_wait = SkipTranslationWait(sw_stage, params); // SHD-005
    HarvestTranslations(params.hash, regs_override == nullptr && !force_legacy && !skip_wait &&
                                         (!legacy_fallback || fallback_nested));
    // D10: a key is made of new translations only, or (forced) of old ones only.
    if (force_legacy) {
        if (auto legacy = TryLegacyProgram(hw_stage, params, runtime_info, binding)) {
            S().key_uses_legacy = true;
            return *legacy;
        }
        S().legacy_failed = true;
        return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
    }
    std::optional<Common::PhaseTimer> match_timer{std::in_place, Common::Phase::ProgramMatch};
    // Diagnostic for runaway permutations: why each existing permutation was rejected.
    std::string mismatch_reasons;
    // PERF-044: replay the last search when its inputs are the same. -DisablePerf 56.
    static const bool memo_enabled = Common::PerfFeatureEnabled(56);
    auto& memo = program->match_memo;
    const bool memo_stage = memo_enabled && (hw_stage == HwStage::Fragment ||
                                             hw_stage == HwStage::Compute);
    bool memo_usable = memo_stage && memo.valid &&
                       memo.modules_size == program->modules.size() &&
                       memo.runtime_info == runtime_info && memo.start == binding &&
                       std::ranges::equal(memo.user_data, params.user_data);
    const auto start_binding = binding;
    boost::container::small_vector<std::vector<u32>, 4> flats;
    // Leaves the memo's replay: the buffers of the permutations before `end` were the memo's.
    const auto stop_replay = [&](size_t end) {
        memo_usable = false;
        flats.clear();
        for (size_t i = 0; i < end; ++i) {
            flats.push_back(i < memo.flats.size() ? memo.flats[i] : std::vector<u32>{});
        }
    };
    // R7: the permutation the last search found is tried first; a program has up to dozens and
    // each one tried builds a specialization (the command thread's largest cost per draw).
    // -DisablePerf 133 searches from the first every time.
    static const bool last_first = Common::PerfFeatureEnabled(133);
    if (last_first && !memo_usable && hw_stage != HwStage::Hull &&
        program->last_match < program->modules.size()) {
        auto& permutation = program->modules[program->last_match];
        if (permutation.info && !permutation.stored) {
            auto& info = *permutation.info;
            info.pgm_base = params.Base();
            info.user_data = params.user_data;
            { Common::PhaseTimer flat_timer{Common::Phase::SelFlatten}; info.RefreshFlatBuf(); }
            if (permutation.spec == [&] {
                    Common::PhaseTimer spec_timer{Common::Phase::SelSpec};
                    return Shader::StageSpecialization(info, runtime_info, profile, binding);
                }()) {
                if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
                    RecoverAttributeFlags(info, permutation.spec, params, runtime_info,
                                          program->last_match);
                }
                info.AddBindings(binding);
                if (auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
                    S().fetch_shader = &fetch;
                }
                return std::make_tuple(&info, permutation.module,
                                       HashCombine(params.hash, program->last_match));
            }
        }
    }
    bool tess_checked = false;
    for (size_t perm_idx = 0; perm_idx < program->modules.size(); ++perm_idx) {
        auto& permutation = program->modules[perm_idx];
        if (!permutation.info) {
            if (memo_stage && !memo_usable) {
                flats.emplace_back();
            }
            continue;
        }
        auto& info = *permutation.info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        { Common::PhaseTimer flat_timer{Common::Phase::SelFlatten}; info.RefreshFlatBuf(); }
        if (hw_stage == HwStage::Hull && !tess_checked) {
            // FIX-058: checked with the first translation's info, before a new permutation
            // could be made from garbage constants.
            tess_checked = true;
            if (!SaneTessConstants(info, liverpool, regs_override != nullptr)) {
                return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
            }
        }
        if (memo_usable && (perm_idx > memo.perm || memo.flats[perm_idx] != info.flattened_ud_buf)) {
            // The inputs differ from here on; the search goes on the usual way.
            stop_replay(perm_idx);
        }
        if (memo_stage && !memo_usable) {
            flats.push_back(info.flattened_ud_buf);
        }
        bool matches;
        if (memo_usable) {
            matches = perm_idx == memo.perm;
        } else {
            std::optional<Common::PhaseTimer> spec_timer{std::in_place, Common::Phase::SelSpec};
            const auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
            spec_timer.reset();
            matches = permutation.spec == spec;
            if (!matches && program->modules.size() >= 4 && mismatch_reasons.size() < 600) {
                u32 index{};
                const char* reason = permutation.spec.FirstDifference(spec, index);
                mismatch_reasons +=
                    fmt::format(" {}:{}[{}]", perm_idx, reason ? reason : "none", index);
            }
        }
        // Newly active resources must match the compiled specialization; inactive ones can reuse
        // it.
        if (!matches) {
            continue;
        }
        if (permutation.stored) {
            // PERF-032: a stored translation matched; its SPIR-V is read now.
            permutation.stored = false;
            permutation.module = LoadStoredModule(params.hash, perm_idx);
            if (!permutation.module) {
                permutation.info.reset();
                if (memo_usable) {
                    stop_replay(perm_idx + 1);
                }
                memo.valid = false;
                continue;
            }
        }
        if (memo_stage && !memo_usable) {
            memo.valid = true;
            memo.perm = perm_idx;
            memo.modules_size = program->modules.size();
            memo.runtime_info = runtime_info;
            memo.start = start_binding;
            memo.user_data.assign(params.user_data.begin(), params.user_data.end());
            memo.flats.assign(flats.begin(), flats.end());
        }
        program->last_match = perm_idx; // R7
        if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
            RecoverAttributeFlags(info, permutation.spec, params, runtime_info, perm_idx);
        }
        info.AddBindings(binding);
        if (auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
            S().fetch_shader = &fetch;
        }
        return std::make_tuple(&info, permutation.module, HashCombine(params.hash, perm_idx));
    }
    match_timer.reset();
    if (AsyncTranslation()) {
        // FIX-078: a translation the workers made for these inputs.
        const bool read_ahead = regs_override != nullptr;
        if (TakeReadyTranslation(hw_stage, sw_stage, params.hash, *program, params, runtime_info,
                                 binding,
                                 !read_ahead)) {
            if (read_ahead) {
                return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
            }
            return GetProgram(hw_stage, sw_stage, params, binding);
        }
        // D10: the new translation is made on a worker; meanwhile the draw's key is made of the
        // old renderer's translations (GetGraphicsPipeline / GetComputePipeline). Without them,
        // the usual way (fallback_nested).
        if (legacy_fallback && !read_ahead && !fallback_nested) {
            if (QueueTranslation(hw_stage, sw_stage, params, runtime_info, binding)) {
                ++legacy_stats.queued;
            }
            fallback_miss = true;
            return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
        }
        if (read_ahead || defer_misses || skip_wait) {
            // A shader an upcoming draw needs (or this draw, which looks again once the next
            // draws' shaders are queued too); translated on a worker while this thread reads
            // on. No key without it: the state is skipped for now. SHD-005: a stage's bindings
            // no longer follow from the stages before it, so a race queues every missing stage
            // of the draw at once and skips it.
            if (!S().stage_deferred || skip_wait) {
                S().stage_deferred = true;
                QueueTranslation(hw_stage, sw_stage, params, runtime_info, binding);
            }
            if (!read_ahead) {
                deferred_miss = true;
                translation_skipped |= skip_wait;
            }
            return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
        }
    }

    // PERF-013: runtime permutations used to take the next free slot of this session, which
    // depends on which stored permutations were preloaded. A slot already used in the stored
    // cache then got a different specialization, overwriting the stored shader and leaving
    // every pipeline that used it unloadable ("stale"; about 1,085 per GT Sport run, compiled
    // again in every race). Start after the stored indices instead.
    size_t perm_idx = program->modules.size();
    if (const auto it = stored_perm_end.find(params.hash); it != stored_perm_end.end()) {
        perm_idx = std::max(perm_idx, it->second);
    }
    const u64 perm_hash = HashCombine(params.hash, perm_idx);
    if (!mismatch_reasons.empty()) {
        static u64 runaway_events = 0;
        if (++runaway_events <= 40 || runaway_events % 500 == 0) {
            LOG_WARNING(Render_Vulkan, "New permutation {} of {}_{:#x} (event {}); rejected:{}",
                        perm_idx, hw_stage, params.hash, runaway_events, mismatch_reasons);
        }
    }
    const auto start = binding;
    auto info = std::make_unique<Shader::Info>(hw_stage, sw_stage, params);
    // Compilation passes may adjust their runtime info (clip-distance emulation adds a fragment
    // input). The stored key must use the unmodified runtime info, or it never matches the next
    // lookup and the same module is recompiled for every draw.
    auto compile_runtime_info = runtime_info;
    const auto module = CompileModule(*info, compile_runtime_info, params.code, perm_idx, binding);
    auto spec = Shader::StageSpecialization(*info, runtime_info, profile, start);
    RegisterShaderMeta(*info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
    const auto* info_ptr = info.get();
    program->InsertPermut(module, std::move(spec), std::move(info), perm_idx);
    program->modules[perm_idx].spv_hash = last_spv_hash;
    LogRepeatedPermutation(*program, perm_idx, hw_stage, params.hash);
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        S().fetch_shader = &fetch;
    }
    return std::make_tuple(info_ptr, module, perm_hash);
}

void PipelineCache::LogRepeatedPermutation(const Program& program, size_t perm_idx,
                                           Shader::HwStage hw_stage, u64 pgm_hash) {
    // FIX-043 diagnostic: a new permutation whose SPIR-V equals an existing one's was translated
    // only because their keys differ in a property its code does not read.
    const auto& added = program.modules[perm_idx];
    if (added.spv_hash == 0) {
        return;
    }
    for (size_t i = 0; i < program.modules.size(); ++i) {
        const auto& other = program.modules[i];
        if (i == perm_idx || !other.info || other.spv_hash != added.spv_hash) {
            continue;
        }
        static u64 repeats = 0;
        ++repeats;
        if (repeats <= 40 || repeats % 200 == 0) {
            u32 index{};
            const char* reason = other.spec.FirstDifference(added.spec, index);
            LOG_WARNING(Render_Vulkan,
                        "FIX-043: permutation {} of {}_{:#x} has the same SPIR-V as permutation "
                        "{} (keys differ first in {}[{}]); {} such repeats so far",
                        perm_idx, hw_stage, pgm_hash, i, reason ? reason : "none", index, repeats);
        }
        return;
    }
}

bool PipelineCache::IsTessEmulatedDraw() const {
    const auto prim = Regs().primitive_type;
    return prim == AmdGpu::PrimitiveType::RectList ||
           (prim == AmdGpu::PrimitiveType::QuadList && !QuadListAsTriangles());
}

bool PipelineCache::QuadListAsTriangles() const {
    // PERF-028: the rasterizer draws quad lists as triangle lists through generated indices
    // (Rasterizer::BindQuadListIndices). Their pipeline is a plain triangle-list pipeline, and
    // the vertex shader writes the fragment shader's inputs directly.
    return Regs().primitive_type == AmdGpu::PrimitiveType::QuadList &&
           !S().draw_indirect_params.tessellate_quads && Common::PerfFeatureEnabled(28);
}

void PipelineCache::RecoverAttributeFlags(Shader::Info& info,
                                          const Shader::StageSpecialization& spec,
                                          const Shader::ShaderParams& params,
                                          const Shader::RuntimeInfo& runtime_info,
                                          size_t perm_idx) {
    // FIX-012: a shader from a cache entry stored without its attribute loads/stores is
    // translated again for them (its stored SPIR-V is kept), because the rect/quad-list helper
    // shaders forward exactly the outputs these flags name. The entry is stored again with them.
    Common::GuestClock::CompilePause hold; // FIX-044
    Shader::Info translated{info.hw_stage, info.sw_stage, params};
    auto translate_runtime_info = runtime_info;
    [[maybe_unused]] const auto program =
        Shader::TranslateProgram(params.code, pools, translated, translate_runtime_info, profile);
    info.loads = translated.loads;
    info.stores = translated.stores;
    info.attribute_flags_known = true;
    RegisterShaderMeta(info, spec.fetch_shader_data, spec, HashCombine(params.hash, perm_idx),
                       perm_idx);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::HwStage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::HwStage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
