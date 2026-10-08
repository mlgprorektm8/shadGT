// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <ranges>

#include "common/perf_monitor.h"
#include "common/elf_info.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

// PERF-019: worker threads that build graphics pipelines ahead of the draws that need them.
class PipelineCache::PipelineBuildWorkers {
public:
    PipelineBuildWorkers() {
        const u32 cores = std::max(std::thread::hardware_concurrency(), 4u);
        // PERF-021: all but two cores (the command thread and the game's own main thread).
        const u32 count = std::clamp(cores - 2, 2u, 14u);
        for (u32 i = 0; i < count; ++i) {
            threads.emplace_back([this](std::stop_token stop) { Work(stop); });
        }
    }
    ~PipelineBuildWorkers() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        cv.notify_all();
    }

    void Push(std::function<void()>&& job, bool urgent) {
        {
            std::scoped_lock lk{mutex};
            if (urgent) {
                jobs.push_front(std::move(job));
            } else {
                jobs.push_back(std::move(job));
            }
        }
        cv.notify_one();
    }

private:
    void Work(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:PipelineBuild");
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, stop, [this] { return !jobs.empty(); });
                if (stop.stop_requested()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            job();
        }
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::function<void()>> jobs;
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

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(HwStage stage, SwStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
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
        const auto* hull_info = infos[u32(SwStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
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
        info.hw.fs.dual_source_blending |= graphics_key.color_buffers[0].blend_swizzled_alpha ||
                                           graphics_key.color_buffers[0].blend_swizzled_factors;
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
            info.hw.fs.color_buffers[i] = graphics_key.color_buffers[i];
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
        const auto& cs_pgm = liverpool->GetCsRegs();
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
        info.sw.vs.vertex_sgpr_offset = draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
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
    // The driver cache must exist before WarmUp so preloaded pipelines can use it.
    CreateDriverCache();
    WarmUp();
    SaveDriverCache(false);
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
    auto [result, data] = instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if (result != vk::Result::eSuccess || data.empty()) {
        return;
    }
    pipelines_at_driver_save = num_new_pipelines;
    driver_cache_saved_at = std::chrono::steady_clock::now();
    if (driver_cache_writer.joinable()) {
        driver_cache_writer.join();
    }
    // Write beside the file and rename, so an interrupted write never leaves a broken cache.
    driver_cache_writer = std::jthread([path = driver_cache_path, data = std::move(data)] {
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
    });
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
    return regs_override ? *regs_override : liverpool->regs;
}

bool PipelineCache::HasSupportedColorTargets() const {
    if (graphics_key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        return false;
    }
    for (u32 cb = 0; cb < graphics_key.num_color_attachments; ++cb) {
        const auto& color_buffer = graphics_key.color_buffers[cb];
        if (color_buffer.data_format != AmdGpu::DataFormat::FormatInvalid &&
            Vulkan::LiverpoolToVK::TrySurfaceFormat(
                color_buffer.data_format, color_buffer.num_format) == vk::Format::eUndefined) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<PipelineCache::PipelineBuild> PipelineCache::StartPipelineBuild(bool urgent) {
    if (!build_workers) {
        build_workers = std::make_unique<PipelineBuildWorkers>();
    }
    auto build = std::make_shared<PipelineBuild>();
    build->key = graphics_key;
    for (size_t stage = 0; stage < MaxShaderStages; ++stage) {
        if (!infos[stage]) {
            continue;
        }
        auto& copy = build->info_copies[stage];
        copy = *infos[stage];
        build->user_data[stage].assign(copy.user_data.begin(), copy.user_data.end());
        copy.user_data = build->user_data[stage];
        build->infos[stage] = &copy;
    }
    build->runtime_infos = runtime_infos;
    if (fetch_shader) {
        build->fetch = *fetch_shader;
    }
    build->modules = modules;
    build->done = build->promise.get_future().share();
    build->queued = std::chrono::steady_clock::now();
    build->urgent = urgent;
    build_workers->Push(
        [this, build] {
            build->pipeline = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, build->key, *pipeline_cache,
                build->infos, build->runtime_infos, build->fetch ? &*build->fetch : nullptr,
                build->modules, build->sdata, false);
            build->promise.set_value();
        },
        urgent);
    return build;
}

void PipelineCache::ReadAheadAtBufferStart() {
    using namespace std::chrono_literals;
    if (!build_workers || !Common::PerfFeatureEnabled(20) || !Common::PerfFeatureEnabled(19) ||
        std::chrono::steady_clock::now() - last_pipeline_miss > 2s) {
        return;
    }
    // No draw is being processed, so nothing needs restoring afterwards.
    ReadAhead(draw_indirect_params, false);
}

void PipelineCache::ReadAhead(const DrawIndirectParams params, bool restore) {
    const auto start = std::chrono::steady_clock::now();
    u32 draws = 0;
    u32 started = 0;
    u32 evaluated = 0;
    const bool dedupe = Common::PerfFeatureEnabled(21);
    liverpool->ScanAheadForPipelines([&](const AmdGpu::Regs& regs) {
        ++draws;
        if (dedupe) {
            // Everything the pipeline key is built from that the commands can change: graphics
            // shader registers (programs and user data), context registers (targets, blending,
            // depth, rasterizer state) and the primitive type.
            const auto& r = regs.reg_array;
            constexpr u32 ShGfxRegs = 0x200;
            constexpr u32 NumContextRegs = 0x400;
            constexpr u32 PrimitiveRegs = 0x100;
            u64 hash = XXH3_64bits(&r[AmdGpu::Regs::ShRegWordOffset], ShGfxRegs * sizeof(u32));
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
        draw_indirect_params = {};
        const bool valid = RefreshGraphicsKey();
        regs_override = nullptr;
        if (!valid || !HasSupportedColorTargets() || graphics_pipelines.contains(graphics_key) ||
            pending_builds.contains(graphics_key)) {
            return false;
        }
        pending_builds.emplace(graphics_key, StartPipelineBuild(false));
        ++started;
        return true;
    });
    if (draws > 0) {
        last_scan_draws = draws;
        draws_since_scan = 0;
        // Back to the draw being processed.
        draw_indirect_params = params;
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

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params) {
    draw_indirect_params = params;
    ++draws_since_scan;
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    if (graphics_key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        LOG_WARNING(Render_Vulkan, "Skipping draw with {} color attachments (MRT mask={:#x})",
                    graphics_key.num_color_attachments, graphics_key.mrt_mask);
        return nullptr;
    }
    for (u32 cb = 0; cb < graphics_key.num_color_attachments; ++cb) {
        const auto& color_buffer = graphics_key.color_buffers[cb];
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
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        ++Common::GetWorkCounters().pipelines_compiled;
        // PERF-019: the pipeline may already be building from an earlier read-ahead. If not,
        // it starts building now, and the draws after it are read for more new pipelines to
        // build at the same time on other cores.
        std::shared_ptr<PipelineBuild> build;
        bool predicted = false;
        last_pipeline_miss = std::chrono::steady_clock::now();
        GraphicsPipeline::NotePipelineMiss();
        if (const auto pending = pending_builds.find(graphics_key);
            pending != pending_builds.end()) {
            build = pending->second;
            pending_builds.erase(pending);
            predicted = true;
            if (Common::PerfFeatureEnabled(19) && draws_since_scan >= last_scan_draws) {
                // Past the last read-ahead: keep reading from where it stopped.
                const auto key = graphics_key;
                ReadAhead(params);
                ASSERT_MSG(graphics_key == key, "Read-ahead changed the current pipeline key");
            }
        } else if (Common::PerfFeatureEnabled(19)) {
            const auto key = graphics_key;
            build = StartPipelineBuild(true);
            ReadAhead(params);
            ASSERT_MSG(graphics_key == key, "Read-ahead changed the current pipeline key");
        }
        if (build) {
            const auto wait_start = std::chrono::steady_clock::now();
            build->done.wait();
            const auto now = std::chrono::steady_clock::now();
            LOG_WARNING(Render_Vulkan,
                        "Pipeline {:#x} {}: built {:.1f} ms after it was queued, draw waited "
                        "{:.1f} ms",
                        pipeline_hash, predicted ? "read ahead" : "on demand",
                        std::chrono::duration<double, std::milli>(now - build->queued).count(),
                        std::chrono::duration<double, std::milli>(now - wait_start).count());
            build->pipeline->SetStageInfos(infos);
            sdata = build->sdata;
            it.value() = std::move(build->pipeline);
        } else {
            it.value() = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
                runtime_infos, fetch_shader, modules, sdata, false);
        }

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;
        MaybeSaveDriverCache();

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        ++Common::GetWorkCounters().pipelines_compiled;
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;
        MaybeSaveDriverCache();

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
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
    const bool blendable_format = num_format == NumberFormat::Unorm ||
                                  num_format == NumberFormat::Snorm ||
                                  num_format == NumberFormat::Srgb ||
                                  num_format == NumberFormat::Float;
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
    const u64 config = u64(std::bit_cast<u32>(bc)) | u64(cb) << 32 |
                       u64(u32(num_format)) << 36 | u64(swizzle.r) << 40 |
                       u64(swizzle.g) << 44 | u64(swizzle.b) << 48 | u64(swizzle.a) << 52 |
                       u64(writes_other_mrt()) << 56 |
                       u64((regs.color_shader_mask.GetMask(cb) &
                            AmdGpu::ColorBufferMask::ComponentA) != 0)
                           << 57;
    static constexpr size_t MaxLoggedSwizzledBlends = 32;
    if (logged_swizzled_blends.size() >= MaxLoggedSwizzledBlends ||
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
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = Regs();
    auto& key = graphics_key;

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
    key.prim_type = regs.primitive_type;
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
    auto& key = graphics_key;
    fetch_shader = nullptr;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::tie(infos[stage_out_idx], modules[stage_out_idx], key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);

    bind_stage(HwStage::Fragment, SwStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(SwStage::Fragment)];
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

    const auto* vs_info = infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
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
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(infos[0], modules[0], compute_key.value) =
        GetProgram(HwStage::Compute, SwStage::Compute, cs_params, binding);
    return true;
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
    ++Common::GetWorkCounters().shaders_compiled;
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");

    Shader::Optimization::g_diag_compiling_code = code;
    Shader::Optimization::g_diag_compiling_hash = info.pgm_hash;
    const auto translate_start = std::chrono::steady_clock::now();
    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    const auto translate_end = std::chrono::steady_clock::now();
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");

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
    LOG_WARNING(Render_Vulkan, "Shader {} {:#x}: translate {:.1f} ms, module {:.1f} ms, {} words",
                info.hw_stage, info.pgm_hash,
                std::chrono::duration<double, std::milli>(translate_end - translate_start).count(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                          translate_end)
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

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(hw_stage, sw_stage);
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>();
    }

    auto& program = it_pgm.value();
    // Diagnostic for runaway permutations: why each existing permutation was rejected.
    std::string mismatch_reasons;
    for (size_t perm_idx = 0; perm_idx < program->modules.size(); ++perm_idx) {
        auto& permutation = program->modules[perm_idx];
        if (!permutation.info) {
            continue;
        }
        auto& info = *permutation.info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        info.RefreshFlatBuf();
        const auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
        // Newly active resources must match the compiled specialization; inactive ones can reuse
        // it.
        if (permutation.spec != spec) {
            if (program->modules.size() >= 4 && mismatch_reasons.size() < 600) {
                u32 index{};
                const char* reason = permutation.spec.FirstDifference(spec, index);
                mismatch_reasons += fmt::format(" {}:{}[{}]", perm_idx, reason ? reason : "none",
                                                index);
            }
            continue;
        }
        if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
            RecoverAttributeFlags(info, permutation.spec, params, runtime_info, perm_idx);
        }
        info.AddBindings(binding);
        if (auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
            fetch_shader = &fetch;
        }
        return std::make_tuple(&info, permutation.module, HashCombine(params.hash, perm_idx));
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
    const auto module =
        CompileModule(*info, compile_runtime_info, params.code, perm_idx, binding);
    auto spec = Shader::StageSpecialization(*info, runtime_info, profile, start);
    RegisterShaderMeta(*info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
    const auto* info_ptr = info.get();
    program->InsertPermut(module, std::move(spec), std::move(info), perm_idx);
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }
    return std::make_tuple(info_ptr, module, perm_hash);
}

bool PipelineCache::IsTessEmulatedDraw() const {
    const auto prim = Regs().primitive_type;
    return prim == AmdGpu::PrimitiveType::RectList || prim == AmdGpu::PrimitiveType::QuadList;
}

void PipelineCache::RecoverAttributeFlags(Shader::Info& info,
                                          const Shader::StageSpecialization& spec,
                                          const Shader::ShaderParams& params,
                                          const Shader::RuntimeInfo& runtime_info,
                                          size_t perm_idx) {
    // FIX-012: a shader from a cache entry stored without its attribute loads/stores is
    // translated again for them (its stored SPIR-V is kept), because the rect/quad-list helper
    // shaders forward exactly the outputs these flags name. The entry is stored again with them.
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
