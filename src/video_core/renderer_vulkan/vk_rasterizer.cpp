// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <filesystem>
#include <array>
#include <map>
#include <mutex>
#include <deque>
#include <fstream>
#include <optional>
#include <xxhash.h>
#include <unordered_map>
#include <unordered_set>
#include <boost/container/small_vector.hpp>
#include <fmt/ranges.h>
#include "common/debug.h"
#include "common/path_util.h"
#include "common/perf_monitor.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/guest_hash.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/depth_attachment.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/resource_binding.h"
#include "video_core/renderer_vulkan/stencil_reference.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#ifdef RYTY_GPU
#include "ryty_gpu/native/shadow.h"
#include "video_core/renderer_vulkan/vk_rasterizer_native.h"
#endif
#include "video_core/amdgpu/draw_pipe.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_descriptor.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    // R5: buffer stage workers use their own rings.
    AmdGpu::DrawPipe::worker_init = [](u32 index) { VideoCore::BufferCache::SetStageWorker(index); };
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    scheduler.StartPerfMonitor();

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
#ifdef RYTY_GPU
        if (native) {
            native->Splice(); // NAT-100: native work recorded so far runs in this submission
        }
#endif
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });

    // PERF-084: Vulkan calls on a command thread. -DisablePerf 97 makes them here as before.
    if (Common::PerfFeatureEnabled(97)) {
        scheduler.EnableCommandThread();
    }
#ifdef RYTY_GPU
    if (NativeModeRequested()) {
        native = std::make_unique<NativeRenderer>(*this); // NAT-100
    }
#endif
}

Rasterizer::~Rasterizer() = default;

namespace {
/// What FilterDraw does with a draw instead of drawing it, decided from the registers alone
/// (PERF-047: the command thread selects pipelines only for draws that are drawn).
enum class DrawFilter { Draw, EliminateFastClear, FmaskDecompress, Resolve, NoPrimitive, Copy };

DrawFilter ClassifyDraw(const AmdGpu::Regs& regs, bool& depth_copy, bool& stencil_copy) {
    depth_copy = false;
    stencil_copy = false;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        return DrawFilter::EliminateFastClear;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        return DrawFilter::FmaskDecompress;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        return DrawFilter::Resolve;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return DrawFilter::NoPrimitive;
    }
    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        return DrawFilter::Copy;
    }
    return DrawFilter::Draw;
}
} // namespace

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->DrawRegs();
    bool depth_copy;
    bool stencil_copy;
    switch (ClassifyDraw(regs, depth_copy, stencil_copy)) {
    case DrawFilter::EliminateFastClear:
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    case DrawFilter::FmaskDecompress:
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    case DrawFilter::Resolve:
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    case DrawFilter::NoPrimitive:
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    case DrawFilter::Copy:
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    case DrawFilter::Draw:
        break;
    }
    return true;
}

bool Rasterizer::SelectsPipelinesAhead() const {
    // PERF-047: -DisablePerf 59 selects every pipeline on the recorder thread as before.
    static const bool enabled = Common::PerfFeatureEnabled(59);
    return enabled && liverpool->Pipelined();
}

void Rasterizer::SeedStage(SelectedStage& selected, const Shader::Info* info) {
    // FIX-077: the recorder's ShadowOf copied a new info under the lookup lock, which the
    // command thread holds while it builds pipelines or waits for one; the recorder sat in the
    // lock (most of its time in GT Sport's hitches) with earlier draws still to record.
    static const bool enabled = Common::PerfFeatureEnabled(101);
    if (!enabled) {
        return;
    }
    auto& seeded = seeded_infos[info];
    if (seeded != info->identity) {
        seeded = info->identity;
        selected.seed = std::make_shared<const Shader::Info>(*info);
    }
}

void Rasterizer::CopySelectedStages(std::array<SelectedStage, Shader::MaxStageTypes>& out) {
    const auto& infos = pipeline_cache.SelectedInfos();
    for (u32 stage = 0; stage < Shader::MaxStageTypes; ++stage) {
        auto& selected = out[stage];
        const Shader::Info* info = stage < infos.size() ? infos[stage] : nullptr;
        selected.present = info != nullptr;
        selected.seed.reset();
        if (!info) {
            continue;
        }
        SeedStage(selected, info);
        selected.pgm_base = info->pgm_base;
        selected.num_user_data = static_cast<u32>(info->user_data.size());
        ASSERT(selected.num_user_data <= selected.user_data.size());
        std::ranges::copy(info->user_data, selected.user_data.begin());
        selected.flattened = info->flattened_ud_buf;
    }
}

std::optional<Rasterizer::SelectedPipeline> Rasterizer::SelectPipelineAhead() {
    // The command thread's registers are those of the draw it is decoding.
    const auto& regs = liverpool->DrawRegs();
    bool depth_copy;
    bool stencil_copy;
    const bool quad_list = regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
    // PERF-068: a quad list's pipeline depends on whether its indices are GPU-written, which
    // only the recorder's buffer cache knows: both are selected, the recorder takes its one.
    // -DisablePerf 75 leaves quad lists to the recorder.
    static const bool select_quads = Common::PerfFeatureEnabled(75);
    if (ClassifyDraw(regs, depth_copy, stencil_copy) != DrawFilter::Draw ||
        (quad_list && !select_quads)) {
        return std::nullopt;
    }
    Common::PhaseTimer timer{Common::Phase::SelectAhead};
    SelectedPipeline selected = selected_recycler.Take(); // PERF-056
    selected.is_compute = false;
    selected.compute = nullptr;
    selected.is_quad_list = quad_list;
    selected.quad_tess_pipeline = nullptr;
    std::scoped_lock lk{pipeline_cache.LookupMutex()};
    selected.pipeline = pipeline_cache.GetGraphicsPipeline({
        .vertex_sgpr_offset = 0,
        .instance_sgpr_offset = 0,
        .tessellate_quads = false,
    });
    if (selected.pipeline) {
        CopySelectedStages(selected.stages);
    }
    if (quad_list) {
        selected.quad_tess_pipeline = pipeline_cache.GetGraphicsPipeline({
            .vertex_sgpr_offset = 0,
            .instance_sgpr_offset = 0,
            .tessellate_quads = true,
        });
        if (selected.quad_tess_pipeline) {
            CopySelectedStages(selected.quad_tess_stages);
        }
    }
    return selected;
}

void Rasterizer::CopyComputeStage(std::array<SelectedStage, Shader::MaxStageTypes>& out) {
    // The cache's compute lookup leaves the program's info in its first slot; the pipeline keeps
    // it in the compute slot.
    const auto& infos = pipeline_cache.SelectedInfos();
    for (auto& stage : out) {
        stage.present = false;
    }
    const Shader::Info* info = infos[0];
    auto& selected = out[u32(Shader::SwStage::Compute)];
    selected.present = info != nullptr;
    selected.seed.reset();
    if (!info) {
        return;
    }
    SeedStage(selected, info);
    selected.pgm_base = info->pgm_base;
    selected.num_user_data = static_cast<u32>(info->user_data.size());
    ASSERT(selected.num_user_data <= selected.user_data.size());
    std::ranges::copy(info->user_data, selected.user_data.begin());
    selected.flattened = info->flattened_ud_buf;
}

const GraphicsPipeline* Rasterizer::LookupJobPipeline(const char** miss) {
    const auto& regs = liverpool->DrawRegs();
    bool depth_copy;
    bool stencil_copy;
    if (ClassifyDraw(regs, depth_copy, stencil_copy) != DrawFilter::Draw) {
        if (miss) {
            *miss = "special draw";
        }
        return nullptr;
    }
    // A quad list's pipeline here is the triangle-list one (PERF-068: the command thread selects
    // both and the recorder picks); its tessellated twin is looked up the same way when needed.
    std::shared_lock lk{pipeline_cache.LookupMutex()}; // NAT-031: job threads at once
    const auto* pipeline = pipeline_cache.LookupBuiltPipeline({});
    if (!pipeline && miss) {
        *miss = pipeline_cache.LookupMiss();
    }
    return pipeline;
}

u64 Rasterizer::SharpsHash(std::span<const Shader::Info* const> stages) {
    static thread_local std::vector<u32> words;
    words.clear();
    const auto add = [](const auto& sharp) {
        const auto* data = reinterpret_cast<const u32*>(&sharp);
        words.insert(words.end(), data, data + sizeof(sharp) / sizeof(u32));
    };
    for (u32 stage = 0; stage < stages.size(); ++stage) {
        const Shader::Info* info = stages[stage];
        if (!info) {
            continue;
        }
        words.push_back(stage);
        for (const auto& desc : info->buffers) {
            if (!desc.IsSpecial()) {
                add(desc.GetSharp(*info));
            }
        }
        for (const auto& desc : info->images) {
            add(desc.GetSharp(*info));
        }
        for (const auto& desc : info->samplers) {
            add(desc.GetSharp(*info));
        }
        for (const auto& desc : info->fmasks) {
            add(desc.GetSharp(*info));
        }
    }
    return XXH3_64bits(words.data(), words.size() * sizeof(u32));
}

u64 Rasterizer::JobSharpsHash() {
    return SharpsHash(PipelineCache::JobInfos());
}

std::optional<Rasterizer::SelectedPipeline> Rasterizer::SelectComputePipelineAhead() {
    // PERF-055: as SelectPipelineAhead, for a direct dispatch. -DisablePerf 66 looks every
    // compute pipeline up on the recorder, under the lookup lock, as before.
    static const bool enabled = Common::PerfFeatureEnabled(66);
    if (!enabled) {
        return std::nullopt;
    }
    Common::PhaseTimer timer{Common::Phase::SelectAhead};
    SelectedPipeline selected = selected_recycler.Take(); // PERF-056
    selected.pipeline = nullptr;
    selected.is_compute = true;
    std::scoped_lock lk{pipeline_cache.LookupMutex()};
    selected.compute = pipeline_cache.GetComputePipeline();
    if (selected.compute) {
        CopyComputeStage(selected.stages);
    }
    return selected;
}

const ComputePipeline* Rasterizer::AcquireComputePipeline() {
    if (selected_pipeline && selected_pipeline->is_compute) {
        const auto* pipeline = selected_pipeline->compute;
        if (!pipeline) {
            ++select_stats.compute_ahead;
            return nullptr;
        }
        if (InstallDrawStages(pipeline, selected_pipeline->stages, true)) {
            ++select_stats.compute_ahead;
            static const bool verify = !Common::LeanRun();
            if (!verify || read_capture || ++select_verify_count % 64 != 0) {
                return pipeline;
            }
            ++select_stats.verified;
            const ComputePipeline* own;
            {
                // FIX-077: a check, not needed to draw: skipped while the command thread holds
                // the lock (it may be building a pipeline) instead of waiting for it.
                std::unique_lock lk{pipeline_cache.LookupMutex(), std::try_to_lock};
                if (!lk.owns_lock()) {
                    return pipeline;
                }
                own = pipeline_cache.GetComputePipeline();
            }
            if (own == pipeline) {
                return pipeline;
            }
            ++select_stats.verify_failed;
            static u32 reports = 0;
            if (reports++ < 20) {
                LOG_ERROR(Render_Vulkan,
                          "PERF-055: compute pipeline selected on the command thread differs "
                          "from the one selected here");
            }
            // Fall through: select here and use that one.
        } else {
            ++select_stats.mismatched;
        }
    }
    // Copies are installed only when dispatches may have installed some before (PERF-055 on);
    // otherwise the pipeline keeps using the cache's infos, as before.
    static const bool copies = Common::PerfFeatureEnabled(66);
    const bool install = copies && SelectsPipelinesAhead();
    std::array<SelectedStage, Shader::MaxStageTypes> stages{};
    const ComputePipeline* pipeline;
    {
        std::scoped_lock lk{pipeline_cache.LookupMutex()};
        pipeline = pipeline_cache.GetComputePipeline();
        if (!pipeline) {
            return nullptr;
        }
        if (install) {
            CopyComputeStage(stages);
        }
    }
    if (install) {
        // The pipeline may still point at copies installed for an earlier dispatch.
        InstallDrawStages(pipeline, stages, false);
    }
    return pipeline;
}

void Rasterizer::HashDrawPagesAhead(const SelectedPipeline& selected, u32 epoch) {
    // Opt-in (SHADGT_HASH_AHEAD=1): in the October 10 Nurburgring benchmark it hashed about
    // 600,000 pages per 2 s ahead (a new upload epoch starts about 10 times a frame) of which a
    // quarter were used, and decoding a submission went from 5 to 14 ms (27 FPS against 35).
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_HASH_AHEAD");
        return env && env[0] == '1';
    }() && Common::PerfFeatureEnabled(70) &&
                                EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Precise;
    if (!enabled) {
        return;
    }
    const Pipeline* pipeline = selected.is_compute
                                   ? static_cast<const Pipeline*>(selected.compute)
                                   : static_cast<const Pipeline*>(selected.pipeline);
    if (!pipeline) {
        return;
    }
    struct FlatView {
        const std::vector<u32>& flattened_ud_buf;
        VAddr pgm_base;
    };
    boost::container::small_vector<std::pair<VAddr, u64>, 16> ranges;
    const auto& canonical = pipeline->CanonicalStages();
    for (u32 stage = 0; stage < Shader::MaxStageTypes; ++stage) {
        const auto& sel = selected.stages[stage];
        const Shader::Info* info = canonical[stage];
        if (!info || !sel.present) {
            continue;
        }
        const FlatView view{sel.flattened, sel.pgm_base};
        for (const auto& desc : info->buffers) {
            if (desc.IsSpecial() || desc.is_written) {
                continue;
            }
            if (!desc.sharp_fetch.FitsIn(sel.flattened.size())) {
                continue;
            }
            const auto vsharp = desc.GetSharp(view);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0 ||
                !IsMappedStart(vsharp.base_address)) {
                continue;
            }
            // Open-ended (scalar pointer) ranges are clamped to their mapping, as BindBuffers
            // does before binding them.
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            if (size == 0 || size > 64_MB) {
                continue;
            }
            ranges.emplace_back(vsharp.base_address, size);
        }
    }
    if (!ranges.empty()) {
        buffer_cache.HashPagesAhead(std::span{ranges.data(), ranges.size()}, epoch);
    }
}

bool Rasterizer::CaptureGuestRange(Common::ReadCapture& capture, VAddr address, u64 size) {
    if (address == 0 || size == 0) {
        return true;
    }
    if (size > 64_MB || !IsMappedStart(address)) {
        return size <= 64_MB; // An unmapped source reads nothing from guest memory.
    }
    size = memory->ClampRangeSize(address, size);
    thread_local std::array<u8, Common::ReadCapture::PageSize> bytes;
    for (VAddr page = Common::AlignDown(address, Common::ReadCapture::PageSize);
         page < address + size; page += Common::ReadCapture::PageSize) {
        if (!page_manager.IsUnwatched(page)) {
            if (!capture.AddWatched(page)) {
                return false;
            }
            continue;
        }
        if (capture.Has(page)) {
            continue;
        }
        memory->CopySparseMemory(page, bytes.data(), bytes.size());
        if (!capture.Add(page, bytes.data())) {
            return false;
        }
    }
    return true;
}

bool Rasterizer::CaptureDrawReads(const SelectedPipeline& selected,
                                  Common::ReadCapture& capture, VAddr index_address,
                                  u64 index_size) {
    const Pipeline* pipeline = selected.is_compute
                                   ? static_cast<const Pipeline*>(selected.compute)
                                   : static_cast<const Pipeline*>(selected.pipeline);
    capture_failure = nullptr;
    capture_writes.clear();
    if (!pipeline) {
        capture_failure = "no pipeline";
        return false;
    }
    auto& scan = capture_scan;
    if (scan.capture_id != capture.Id()) {
        scan.capture_id = capture.Id();
        scan.ranges.clear();
    }
    thread_local std::array<u8, Common::ReadCapture::PageSize> page_bytes;
    bool complete = true;
    // Copies the unprotected pages of a range (each range once per capture).
    const auto capture_range = [&](VAddr address, u64 size) {
        if (address == 0 || size == 0) {
            return;
        }
        if (!IsMappedStart(address)) {
            return; // Not read by the recorder either (bound null).
        }
        size = memory->ClampRangeSize(address, size);
        if (size > 64_MB) {
            complete = false;
            capture_failure = "range over 64 MB";
            return;
        }
        const u64 key = (address * 0x9E3779B97F4A7C15ull) ^ size;
        if (!scan.ranges.insert(key).second) {
            return;
        }
        for (VAddr page = Common::AlignDown(address, Common::ReadCapture::PageSize);
             page < address + size; page += Common::ReadCapture::PageSize) {
            ++scan.pages_checked;
            if (!page_manager.IsUnwatched(page)) {
                // Protected: a CPU write faults; the fault waits while this work is pending.
                // Pages of large ranges (shadow maps, big textures the game does not rewrite
                // during play) are not listed, to keep the list small.
                if (size <= 8_MB && !capture.AddWatched(page)) {
                    complete = false;
                    capture_failure = "capture full";
                    return;
                }
                continue;
            }
            if (capture.Has(page)) {
                continue;
            }
            memory->CopySparseMemory(page, page_bytes.data(), page_bytes.size());
            if (!capture.Add(page, page_bytes.data())) {
                complete = false;
                capture_failure = "capture full";
                return;
            }
        }
    };
    struct FlatView {
        const std::vector<u32>& flattened_ud_buf;
        VAddr pgm_base;
    };
    const auto& canonical = pipeline->CanonicalStages();
    for (u32 stage = 0; stage < Shader::MaxStageTypes && complete; ++stage) {
        const auto& sel = selected.stages[stage];
        const Shader::Info* info = canonical[stage];
        if (!info || !sel.present) {
            continue;
        }
        if (info->hw_stage == Shader::HwStage::Hull) {
            capture_failure = "tessellation";
            return false; // Tessellation constants are read through a pointer; not covered.
        }
        const FlatView view{sel.flattened, sel.pgm_base};
        for (const auto& desc : info->buffers) {
            if (desc.IsSpecial()) {
                continue;
            }
            if (!desc.sharp_fetch.FitsIn(sel.flattened.size())) {
                capture_failure = "sharp outside the flattened data";
                return false;
            }
            const auto vsharp = desc.GetSharp(view);
            if (vsharp.num_records == UINT32_MAX) {
                // An open-ended V# is bound up to the end of its mapping: copying that for
                // every window cost more than the early fence saved (October 10 benchmark).
                capture_failure = "open-ended buffer";
                static std::atomic<u32> logged{};
                if (vsharp.base_address != 0 && IsMappedStart(vsharp.base_address) &&
                    logged.fetch_add(1) < 40) {
                    const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                    u64 unwatched = 0;
                    for (VAddr page = Common::AlignDown(vsharp.base_address, 4096);
                         page < vsharp.base_address + std::min<u64>(size, 256_MB); page += 4096) {
                        unwatched += page_manager.IsUnwatched(page);
                    }
                    LOG_WARNING(Render_Vulkan,
                                "PERF-063: open-ended V# shader {}_{:#x} binding {} base {:#x} "
                                "stride {} bound {:#x} bytes, {} of its pages unprotected",
                                info->hw_stage, info->pgm_hash, &desc - info->buffers.data(),
                                u64(vsharp.base_address), vsharp.GetStride(), size, unwatched);
                }
                return false;
            }
            capture_range(vsharp.base_address, vsharp.GetSize());
            if (desc.is_written && vsharp.base_address != 0 && vsharp.GetSize() != 0 &&
                IsMappedStart(vsharp.base_address)) {
                // PERF-070: guarded until recorded, so the CPU does not read it before.
                capture_writes.emplace_back(
                    vsharp.base_address,
                    memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize()));
            }
        }
        for (const auto& desc : info->images) {
            if (!desc.sharp_fetch.FitsIn(sel.flattened.size())) {
                capture_failure = "sharp outside the flattened data";
                return false;
            }
            const auto tsharp = desc.GetSharp(view);
            if (tsharp.Address() == 0 ||
                tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid ||
                !magic_enum::enum_contains(tsharp.GetDataFmt()) ||
                !magic_enum::enum_contains(tsharp.GetNumberFmt()) ||
                LiverpoolToVK::TrySurfaceFormat(tsharp.GetDataFmt(), tsharp.GetNumberFmt()) ==
                    vk::Format::eUndefined ||
                VideoCore::CheckImageDescriptorGeometry(tsharp, instance.GetImageLimits()) !=
                    VideoCore::ImageDescriptorGeometryError::None) {
                continue; // Bound null by the recorder (the same checks as BindTextures).
            }
            const VideoCore::TextureCache::ImageDesc image{tsharp, desc};
            capture_range(image.info.guest_address, image.info.guest_size);
        }
    }
    if (!complete) {
        return false;
    }
    if (!selected.is_compute) {
        // Vertex buffers: their V#s are read through a pointer in the vertex shader's user
        // data; the V# table entries are captured as well as the buffers.
        const auto& fetch = selected.pipeline->GetFetchShader();
        const auto& vs = selected.stages[u32(Shader::SwStage::Vertex)];
        if (!fetch.attributes.empty() && !vs.present) {
            capture_failure = "vertex stage missing";
            return false;
        }
        for (const auto& attrib : fetch.attributes) {
            if (attrib.sgpr_base == Shader::IR::NumScalarRegs ||
                attrib.sgpr_base + 1u >= vs.num_user_data) {
                capture_failure = "vertex table outside user data";
                return false;
            }
            VAddr table;
            std::memcpy(&table, &vs.user_data[attrib.sgpr_base], sizeof(table));
            table &= 0xFFFFFFFFFFFFULL;
            const VAddr entry = table + attrib.dword_offset * sizeof(u32);
            capture_range(entry, sizeof(AmdGpu::Buffer));
            if (!complete || !IsMappedStart(entry)) {
                if (complete) {
                    capture_failure = "vertex table unmapped";
                }
                return false;
            }
            AmdGpu::Buffer vsharp;
            capture.Copy(entry, reinterpret_cast<u8*>(&vsharp), sizeof(vsharp),
                         [this](VAddr address, u8* to, u64 n) {
                             memory->CopySparseMemory(address, to, n);
                         });
            vsharp.base_address += attrib.inst_offset;
            capture_range(vsharp.base_address, vsharp.GetSize());
        }
        capture_range(index_address, std::min<u64>(index_size, 16_MB));
    }
    return complete;
}

void Rasterizer::OrderBuffersAhead(const SelectedPipeline& selected, BufferPlan& plan) {
    // R5 (stage dispatcher, in job order): the job's number and the ranges it writes, before
    // any worker plans a later job.
    static const bool track_writes = Common::PerfFeatureEnabled(129);
    if (!track_writes) {
        return;
    }
    plan.seq = ++stage_seq;
    PrepareBuffersAhead(selected, plan, 0, 0, true);
}

// FIX-111: set while PrepareBuffersAhead only asks IsComputeImageCopy/MetaClear/ImageClear.
static thread_local bool t_special_dry_run = false;

bool Rasterizer::PrepareBuffersAhead(const SelectedPipeline& selected, BufferPlan& plan,
                                     VAddr index_address, u32 index_bytes, bool order_only) {
    const Pipeline* pipeline = selected.is_compute
                                   ? static_cast<const Pipeline*>(selected.compute)
                                   : static_cast<const Pipeline*>(selected.pipeline);
    // FIX-111: only asks whether the dispatch is a copy or clear the renderer does itself; doing
    // it here, ahead of the draws before it, copied a picture before it was drawn (GT Sport's top
    // bar blur came out black on some frames, or on all of them).
    t_special_dry_run = true;
    const bool special = pipeline && (IsComputeImageCopy(pipeline) ||
                                      IsComputeMetaClear(pipeline) || IsComputeImageClear(pipeline));
    t_special_dry_run = false;
    if (!pipeline || special) {
        return true; // Nothing to obtain, or BindResources is not reached.
    }
    // MT-007: reads over a range an earlier planned job writes are left to the recorder.
    static const bool track_writes = Common::PerfFeatureEnabled(129);
    boost::container::small_vector<PendingWrite, 4> job_writes;
    if (order_only) {
        const u64 recorded = recorded_seq.load(std::memory_order_acquire);
        std::scoped_lock lk{pending_writes_mutex};
        while (!pending_writes.empty() && pending_writes.front().seq <= recorded) {
            pending_writes.pop_front();
        }
    } else {
        // R5: a plan redone after a refusal starts over (its uploads stay: they are recorded
        // with the job).
        plan.entries.clear();
        plan.next = 0;
        plan.vertices = false;
        plan.index = false;
        plan.generation = texture_cache.RegistryGeneration();
        plan.labels_done = liverpool->LabelsDone(); // PERF-082
        plan.labels_pushed = liverpool->LabelsPushed();
    }
    const auto pending_overlap = [&](VAddr begin, u64 size) {
        std::scoped_lock lk{pending_writes_mutex};
        for (const auto& write : pending_writes) {
            if (write.seq < plan.seq && write.begin < begin + size && begin < write.end) {
                return true;
            }
        }
        return false;
    };
    // MT-004: the vertex buffers too, merged as BindVertexBuffers merges them (it takes these
    // entries first). Quad lists refresh their pages first and are left to the recorder.
    static const bool plan_vertices = Common::PerfFeatureEnabled(126);
    if (!order_only && plan_vertices && !selected.is_compute && !selected.is_quad_list && selected.pipeline) {
        const auto& fetch = selected.pipeline->GetFetchShader();
        const auto& vs = selected.stages[u32(Shader::SwStage::Vertex)];
        if (!fetch.Empty() && !fetch.attributes.empty()) {
            if (!vs.present) {
                return false;
            }
            struct Range {
                VAddr begin;
                VAddr end;
            };
            boost::container::small_vector<Range, 16> ranges;
            for (const auto& attrib : fetch.attributes) {
                if (attrib.sgpr_base == Shader::IR::NumScalarRegs ||
                    attrib.sgpr_base + 1u >= vs.num_user_data) {
                    return false;
                }
                VAddr table;
                std::memcpy(&table, &vs.user_data[attrib.sgpr_base], sizeof(table));
                table &= 0xFFFFFFFFFFFFULL;
                const VAddr entry = table + attrib.dword_offset * sizeof(u32);
                if (!IsMappedStart(entry)) {
                    return false;
                }
                AmdGpu::Buffer vsharp;
                memory->CopySparseMemory(entry, reinterpret_cast<u8*>(&vsharp), sizeof(vsharp));
                vsharp.base_address += attrib.inst_offset;
                if (vsharp.base_address == 0 || vsharp.GetSize() == 0 ||
                    memory->MappedAreaEnd(vsharp.base_address, vsharp.base_address + 1) == 0) {
                    continue; // Bound empty by the recorder (FIX-053).
                }
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                ranges.push_back({vsharp.base_address, vsharp.base_address + size});
            }
            std::ranges::sort(ranges, {}, &Range::begin);
            boost::container::small_vector<Range, 16> merged;
            for (const auto& range : ranges) {
                if (merged.empty() || merged.back().end < range.begin) {
                    merged.push_back(range);
                } else {
                    merged.back().end = std::max(merged.back().end, range.end);
                }
            }
            for (const auto& range : merged) {
                const u32 size =
                    static_cast<u32>(memory->ClampRangeSize(range.begin, range.end - range.begin));
                if (track_writes && pending_overlap(range.begin, size)) {
                    plan.entries.push_back({range.begin, 0, false, false, nullptr, 0, false});
                    continue;
                }
                bool overlapped = false;
                const auto result = buffer_cache.ObtainBufferAhead(
                    range.begin, size, false, false, plan.uploads, &overlapped);
                if (!result) {
                    ++plan_refusals[2];
                    return false;
                }
                plan.entries.push_back(
                    {range.begin, size, false, false, result->first, result->second, overlapped});
            }
            plan.vertices = true;
        }
    }
    static const bool plan_index = Common::PerfFeatureEnabled(127);
    if (!order_only && plan_index && index_bytes != 0 && index_address != 0 && !selected.is_compute &&
        !selected.is_quad_list) {
        bool overlapped = false;
        if (track_writes && pending_overlap(index_address, index_bytes)) {
            plan.entries.push_back({index_address, 0, false, false, nullptr, 0, false});
            plan.index = true;
        } else {
            const auto result = buffer_cache.ObtainBufferAhead(
                index_address, index_bytes, false, false, plan.uploads, &overlapped);
            if (!result) {
                ++plan_refusals[2];
                return false;
            }
            plan.entries.push_back({index_address, index_bytes, false, false, result->first,
                                    result->second, overlapped});
            plan.index = true;
        }
    }
    struct FlatView {
        const std::vector<u32>& flattened_ud_buf;
        VAddr pgm_base;
    };
    static const bool null_impossible = Common::PerfFeatureEnabled(39);
    constexpr u64 GpuAddressLimit = 1ULL << 40;
    bool barrier_after = false;
    const auto& canonical = pipeline->CanonicalStages();
    for (u32 stage = 0; stage < Shader::MaxStageTypes; ++stage) {
        const auto& sel = selected.stages[stage];
        const Shader::Info* info = canonical[stage];
        if (!info || !sel.present) {
            continue;
        }
        if (!order_only && IsWatchedShader(info->pgm_hash)) {
            ++plan_refusals[0];
            return false; // Its read pages are checked again before each use (FIX-017).
        }
        const FlatView view{sel.flattened, sel.pgm_base};
        // FIX-083: a buffer the draw writes is left to the recorder, and the stage waits until
        // the recorder has recorded the draw. Obtained ahead, the write was registered while
        // earlier draws were still to be recorded: those then read the range from the GPU copy
        // without the CPU's newer bytes, and a CPU fault there wrote the GPU copy back over
        // them (cockpit shadows flickered). -DisablePerf 105 plans written buffers again.
        static const bool written_to_recorder = Common::PerfFeatureEnabled(105);
        static const bool skip_written = Common::PerfFeatureEnabled(128);
        for (const auto& desc : info->buffers) {
            if (desc.IsSpecial()) {
                continue;
            }
            if (order_only && !desc.is_written) {
                continue;
            }
            // D9: read in place by the recorder (BindBuffers), not obtained.
            if (info->uses_direct && desc.IsDirectEligible() && buffer_cache.GetDirectMemory()) {
                continue;
            }
            if (!order_only && written_to_recorder && desc.is_written) {
                ++plan_refusals[3];
                if (!skip_written) {
                    return false;
                }
            }
            if (!desc.sharp_fetch.FitsIn(sel.flattened.size())) {
                if (order_only) {
                    continue;
                }
                ++plan_refusals[1];
                return false;
            }
            const auto vsharp = desc.GetSharp(view);
            // The same rejections as BindBuffers, which then obtains nothing.
            const bool impossible =
                null_impossible && vsharp.num_records != UINT32_MAX &&
                (vsharp.GetSize() >= GpuAddressLimit ||
                 u64(vsharp.base_address) + vsharp.GetSize() > GpuAddressLimit ||
                 (vsharp.GetSize() >= 1_GB && vsharp.base_address != 0 &&
                  !memory->IsValidMapping(vsharp.base_address)));
            const bool unmapped = !impossible && vsharp.base_address != 0 &&
                                  vsharp.GetSize() != 0 && !IsMappedStart(vsharp.base_address);
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0 || impossible || unmapped) {
                continue;
            }
            const u32 size =
                static_cast<u32>(memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize()));
            if (order_only) {
                job_writes.push_back({plan.seq, vsharp.base_address, vsharp.base_address + size});
                continue;
            }
            if (written_to_recorder && desc.is_written) {
                // MT-006: the buffer is left to the recorder (an entry without a buffer), the
                // draw's other buffers are still planned, and the stage waits after the draw
                // (MT-007: or remembers the range and goes on).
                plan.entries.push_back(
                    {vsharp.base_address, 0, true, desc.is_formatted, nullptr, 0, false});
                if (!track_writes) {
                    barrier_after = true;
                }
                continue;
            }
            if (track_writes && pending_overlap(vsharp.base_address, size)) {
                plan.entries.push_back(
                    {vsharp.base_address, 0, false, desc.is_formatted, nullptr, 0, false});
                continue;
            }
            bool overlapped = false;
            const auto result = buffer_cache.ObtainBufferAhead(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, plan.uploads,
                &overlapped);
            if (!result) {
                ++plan_refusals[2];
                return false;
            }
            plan.entries.push_back({vsharp.base_address, size, desc.is_written, desc.is_formatted,
                                    result->first, result->second, overlapped});
        }
    }
    if (order_only && !job_writes.empty()) {
        std::scoped_lock lk{pending_writes_mutex};
        pending_writes.insert(pending_writes.end(), job_writes.begin(), job_writes.end());
    }
    return !barrier_after;
}

std::optional<std::pair<const VideoCore::Buffer*, u64>> Rasterizer::TakePlannedBuffer(
    VAddr address, u32 size, bool written, bool texel) {
    auto* plan = buffer_plan;
    if (!plan || plan->next >= plan->entries.size()) {
        return std::nullopt;
    }
    const auto& entry = plan->entries[plan->next];
    if (!entry.buffer && entry.address == address && entry.written == written &&
        entry.texel == texel) {
        ++plan->next; // MT-006: left to the recorder
        return std::nullopt;
    }
    if (entry.address != address || entry.size != size || entry.written != written ||
        entry.texel != texel) {
        // The recorder binds something else than planned: it obtains the rest itself.
        ++plan_stats.mismatched;
        plan->next = plan->entries.size();
        return std::nullopt;
    }
    ++plan->next;
    if (liverpool->LabelWrittenOver(plan->labels_done, plan->labels_pushed, address, size)) {
        // PERF-082: a label was written into it after the stage read it.
        ++plan_stats.redone;
        return std::nullopt;
    }
    // MT-003: only an image registered over the range since the plan (any image change used to
    // redo every planned range under an image: ~15% of the plans in races).
    static const bool precise_redo = Common::PerfFeatureEnabled(125);
    if (texture_cache.RegistryGeneration() != plan->generation &&
        (precise_redo ? texture_cache.ImageRegisteredSince(plan->generation, address, size)
                      : texture_cache.AnyImageInRegion(address, size))) {
        // An image over the range appeared since: obtained again, with its export.
        ++plan_stats.redone;
        return std::nullopt;
    }
    if (entry.overlapped && buffer_cache.ExportImagesOver(address, size)) {
        // PERF-079: an image over it had to be exported first; obtained again after it.
        ++plan_stats.redone;
        return std::nullopt;
    }
    // R5: with several stage workers, another may have found the range dirty for a later job.
    static const bool settle = VideoCore::BufferCache::StageWorkerCount() > 1;
    if (settle) {
        buffer_cache.SettleAheadRange(address, size);
    }
    ++plan_stats.used;
    return std::pair<const VideoCore::Buffer*, u64>{entry.buffer, entry.offset};
}

Rasterizer::ShadowInfo& Rasterizer::ShadowOf(const Shader::Info* canonical,
                                              const Shader::Info* seed) {
    // PERF-065: most draws use the infos of the draws before them; the map is looked up once.
    auto& cached = shadow_lookup[(reinterpret_cast<uintptr_t>(canonical) >> 6) %
                                 shadow_lookup.size()];
    if (cached.first != canonical) {
        cached = {canonical, &shadow_infos[canonical]};
    }
    auto& shadow = *cached.second;
    // A cached info's identity never changes; a different one means the address was reused.
    if (!shadow.info || shadow.identity != canonical->identity) {
        // FIX-077: the command thread's copy, when it made one for this info.
        if (seed && seed->identity == canonical->identity) {
            shadow.info = std::make_unique<Shader::Info>(*seed);
            shadow.identity = seed->identity;
            return shadow;
        }
        // The command thread rewrites the cached info's per-draw fields under the lock.
        std::scoped_lock lk{pipeline_cache.LookupMutex()};
        shadow.info = std::make_unique<Shader::Info>(*canonical);
        shadow.identity = canonical->identity;
    }
    return shadow;
}

bool Rasterizer::InstallDrawStages(const Pipeline* pipeline,
                                   std::array<SelectedStage, Shader::MaxStageTypes>& stages,
                                   bool validate) {
    const auto& canonical = pipeline->CanonicalStages();
    std::array<const Shader::Info*, Shader::MaxStageTypes> draw_stages{};
    for (u32 stage = 0; stage < Shader::MaxStageTypes; ++stage) {
        auto& selected = stages[stage];
        if (!canonical[stage] || !selected.present) {
            if (canonical[stage] || selected.present) {
                return false;
            }
            continue;
        }
        auto& shadow = ShadowOf(canonical[stage], selected.seed.get());
        auto& info = *shadow.info;
        std::copy_n(selected.user_data.begin(), selected.num_user_data, shadow.user_data.begin());
        info.user_data = std::span<const u32>{shadow.user_data.data(), selected.num_user_data};
        info.pgm_base = selected.pgm_base;
        // PERF-061: the flattened user data the command thread read is used as is; the
        // recorder no longer walks the shader resource tables again to check it (in millions of
        // checks it never differed). -DisablePerf 69 reads them again and compares.
        static const bool trust_selection = Common::PerfFeatureEnabled(69);
        if (validate && trust_selection) {
            // PERF-065: moved, not copied; the selection is recycled after the draw and keeps
            // the shadow's old buffer (its capacity).
            std::swap(info.flattened_ud_buf, selected.flattened);
            info.flattened_ud_src.assign(info.flattened_ud_buf.size(), 0);
        } else {
            // The sharps are read again from the memory as the recorder sees it now.
            info.RefreshFlatBuf();
            if (validate && info.flattened_ud_buf != selected.flattened) {
                return false;
            }
        }
        draw_stages[stage] = &info;
    }
    pipeline->SetDrawStages(draw_stages);
    return true;
}

const GraphicsPipeline* Rasterizer::AcquireGraphicsPipeline(const DrawIndirectParams& params,
                                                            bool selected_allowed) {
    if (!SelectsPipelinesAhead()) {
        std::scoped_lock lk{pipeline_cache.LookupMutex()};
        return pipeline_cache.GetGraphicsPipeline(params);
    }
    auto& stats = select_stats;
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.since >= std::chrono::seconds{2}) {
        if (stats.ahead + stats.compute_ahead + stats.mismatched + stats.recorder != 0) {
            LOG_WARNING(Render_Vulkan,
                        "PERF-047 pipelines in {:.1f} s: {} selected on the command thread "
                        "(PERF-055: {} of them compute), {} selected again here (memory "
                        "changed), {} selected here (quad lists, indirect draws); {} checked "
                        "against a selection here, {} differed",
                        std::chrono::duration<double>(now - stats.since).count(),
                        stats.ahead + stats.compute_ahead, stats.compute_ahead, stats.mismatched,
                        stats.recorder, stats.verified, stats.verify_failed);
        }
        if (plan_stats.used + plan_stats.redone + plan_stats.mismatched != 0) {
            LOG_WARNING(Render_Vulkan,
                        "PERF-067 buffer stage in 2.0 s: {} shader buffers obtained ahead and "
                        "used, {} obtained again here (an image appeared over them), {} plans "
                        "left for a different binding; plans stopped: watched shader {}, sharp "
                        "{}, written {}, buffer {} (texel read {}, image {}, stream ring {}, arena {}, "
                        "residency {}, staging {})",
                        plan_stats.used, plan_stats.redone, plan_stats.mismatched,
                        plan_refusals[0].exchange(0), plan_refusals[1].exchange(0),
                        plan_refusals[3].exchange(0), plan_refusals[2].exchange(0), buffer_cache.ahead_refusals[0].exchange(0),
                        buffer_cache.ahead_refusals[1].exchange(0),
                        buffer_cache.ahead_refusals[2].exchange(0),
                        buffer_cache.ahead_refusals[3].exchange(0),
                        buffer_cache.ahead_refusals[4].exchange(0),
                        buffer_cache.ahead_refusals[5].exchange(0));
            plan_stats = {};
        }
        stats = {};
        stats.since = now;
    }
    // PERF-068: a quad list's selection holds both of its pipelines.
    const bool quad_selection = selected_pipeline && selected_pipeline->is_quad_list;
    if ((selected_allowed || quad_selection) && selected_pipeline) {
        const bool tess = quad_selection && params.tessellate_quads;
        const auto* pipeline =
            tess ? selected_pipeline->quad_tess_pipeline : selected_pipeline->pipeline;
        if (!pipeline) {
            ++stats.ahead;
            return nullptr;
        }
        if (InstallDrawStages(pipeline,
                              tess ? selected_pipeline->quad_tess_stages : selected_pipeline->stages,
                              true)) {
            ++stats.ahead;
            // Every 64th selection is checked against one made here, from the recorder's
            // registers and memory; a difference is reported and this one used.
            static const bool verify = !Common::LeanRun();
            // PERF-063: not for captured work; the game may have rewritten the memory.
            if (!verify || read_capture || ++select_verify_count % 64 != 0) {
                return pipeline;
            }
            ++stats.verified;
            std::array<SelectedStage, Shader::MaxStageTypes> own{};
            const GraphicsPipeline* own_pipeline;
            {
                // FIX-077: skipped while the command thread holds the lock, as for compute.
                std::unique_lock lk{pipeline_cache.LookupMutex(), std::try_to_lock};
                if (!lk.owns_lock()) {
                    return pipeline;
                }
                own_pipeline = pipeline_cache.GetGraphicsPipeline(params);
                if (own_pipeline) {
                    CopySelectedStages(own);
                }
            }
            if (own_pipeline == pipeline) {
                return pipeline;
            }
            ++stats.verify_failed;
            static u32 reports = 0;
            if (reports++ < 20) {
                LOG_ERROR(Render_Vulkan,
                          "PERF-047: pipeline selected on the command thread ({:#x}) differs from "
                          "the one selected here ({:#x})",
                          std::hash<GraphicsPipelineKey>{}(pipeline->GetGraphicsKey()),
                          own_pipeline ? std::hash<GraphicsPipelineKey>{}(
                                             own_pipeline->GetGraphicsKey())
                                       : 0);
            }
            if (own_pipeline) {
                InstallDrawStages(own_pipeline, own, false);
            }
            return own_pipeline;
        }
        // The user data now leads to other sharps than the selection read (memory written in
        // between): select here, from what the recorder sees.
        ++stats.mismatched;
    } else {
        ++stats.recorder;
    }
    std::array<SelectedStage, Shader::MaxStageTypes> stages{};
    const GraphicsPipeline* pipeline;
    {
        std::scoped_lock lk{pipeline_cache.LookupMutex()};
        pipeline = pipeline_cache.GetGraphicsPipeline(params);
        if (!pipeline) {
            return nullptr;
        }
        CopySelectedStages(stages);
    }
    InstallDrawStages(pipeline, stages, false);
    return pipeline;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->DrawRegs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = ctx->cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->DrawCbExtent(cb);
        std::construct_at(&desc, col_buf, hint);
        image_id = ctx->bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->DrawDbExtent();
        auto& [image_id, desc] = ctx->db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = ctx->bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        ctx->db_desc.first = {};
    }
    // DIAG-116 (SHADGT_DIAG_PASSES=1): draws per render pass (color target 0 and depth target,
    // with their sizes), every 4 s: where a frame's draws go.
    static const bool diag_passes = [] {
        const char* env = std::getenv("SHADGT_DIAG_PASSES");
        return env && (env[0] == '1' || env[0] == '2');
    }();
    if (diag_passes) {
        struct PassStats {
            u64 draws{};
            u64 first_seen{};
            u32 cb_w{}, cb_h{}, db_w{}, db_h{};
            u32 cbs{};
        };
        static std::unordered_map<u64, PassStats> passes;
        static u64 sequence = 0;
        static auto window = std::chrono::steady_clock::now();
        static u64 frames_at_start = DebugState.GetFrameNum();
        VAddr cb_addr = 0, db_addr = 0;
        u32 cb_w = 0, cb_h = 0, db_w = 0, db_h = 0, cbs = 0;
        for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
            if (ctx->cb_descs[cb].image_id) {
                const auto& info = texture_cache.GetImage(ctx->cb_descs[cb].image_id).info;
                if (cbs++ == 0) {
                    cb_addr = info.guest_address;
                    cb_w = info.size.width;
                    cb_h = info.size.height;
                }
            }
        }
        if (ctx->db_desc.first) {
            const auto& info = texture_cache.GetImage(ctx->db_desc.first).info;
            db_addr = info.guest_address;
            db_w = info.size.width;
            db_h = info.size.height;
        }
        // Once, 230 s in: the color targets of the other passes are written to user\log
        // (pass_<w>x<h>_<format>.bin) when the main view starts drawing again.
        static std::unordered_map<u64, VideoCore::ImageId> to_dump;
        static bool dumped = false;
        static const auto dump_start = std::chrono::steady_clock::now();
        static const bool dump_passes = std::getenv("SHADGT_DIAG_PASSES")[0] == '2';
        static bool cleared = !dump_passes;
        if (!cleared && std::chrono::steady_clock::now() - dump_start > std::chrono::seconds{229}) {
            cleared = true; // only targets of the last second before the dump
            to_dump.clear();
        }
        if (dump_passes && !dumped && cleared && cbs != 0 && !(cb_w == 1920 && cb_h == 1080)) {
            to_dump[(u64(cb_w) << 32) | cb_h] = ctx->cb_descs[0].image_id;
        }
        if (dump_passes && !dumped && cb_w == 1920 && cb_h == 1080 && !to_dump.empty() &&
            std::chrono::steady_clock::now() - dump_start > std::chrono::seconds{230}) {
            dumped = true;
            const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
            for (const auto& [size_key, id] : to_dump) {
                if (!id) {
                    continue;
                }
                const auto& info = texture_cache.GetImage(id).info;
                LOG_WARNING(Render_Vulkan, "DIAG-116: pass target {}x{} -> image {}x{} {} {} bits depth {}",
                            size_key >> 32, size_key & 0xffffffff, info.size.width, info.size.height,
                            vk::to_string(info.pixel_format), info.num_bits, info.props.is_depth);
                texture_cache.DumpImage(
                    id, dir / fmt::format("pass_{}x{}_{}_{}bit.bin", info.size.width,
                                          info.size.height, vk::to_string(info.pixel_format),
                                          info.num_bits));
            }
            LOG_WARNING(Render_Vulkan, "DIAG-116: dumped {} pass targets", to_dump.size());
        }
        auto& stats = passes[cb_addr * 31 + db_addr];
        if (stats.draws++ == 0) {
            stats.first_seen = sequence++;
            stats = {1, stats.first_seen, cb_w, cb_h, db_w, db_h, cbs};
        }
        if (const auto now = std::chrono::steady_clock::now(); now - window >= std::chrono::seconds{4}) {
            const u64 frames = std::max<u64>(1, DebugState.GetFrameNum() - frames_at_start);
            std::vector<std::pair<u64, PassStats>> list(passes.begin(), passes.end());
            std::ranges::sort(list, std::greater{}, [](const auto& e) { return e.second.draws; });
            u64 total = 0;
            for (const auto& [k, p] : list) {
                total += p.draws;
            }
            std::string text;
            for (size_t i = 0; i < std::min<size_t>(list.size(), 20); ++i) {
                const auto& p = list[i].second;
                text += fmt::format("\n  {:5.0f} draws/frame ({:4.1f}%): {} color {}x{}, depth {}x{}",
                                    double(p.draws) / frames, 100.0 * p.draws / total, p.cbs, p.cb_w,
                                    p.cb_h, p.db_w, p.db_h);
            }
            LOG_WARNING(Render_Vulkan, "DIAG-116: {:.0f} draws/frame over {} frames in {} passes:{}",
                        double(total) / frames, frames, list.size(), text);
            passes.clear();
            window = now;
            frames_at_start = DebugState.GetFrameNum();
        }
    }
}

// PERF-028: quad lists are drawn as triangle lists through generated indices instead of the
// tessellation helper shaders. On GT Sport's Nurburgring the grass is drawn as quad lists, and
// through the helpers some patches got corners matching no vertex shader output (RenderDoc:
// 66 of 714 patches of one draw, with correct vertex data and sequential indices), stretching
// grass and terrain across the screen.
// Quad (v0, v1, v2, v3) becomes two triangles that keep its winding and its provoking vertex:
// v0 with the first-vertex convention, v3 with the last-vertex one (the OpenGL rule for
// independent quads). Which diagonal the PS4 itself splits along is not documented.
static constexpr std::array<u32, 6> QuadCornersFirstVertex{0, 1, 2, 0, 2, 3};
static constexpr std::array<u32, 6> QuadCornersLastVertex{0, 1, 3, 1, 2, 3};
static constexpr u64 MaxQuadListIndexBytes = 16ULL << 20;

template <typename T>
static u32 ExpandQuadListIndices(const T* in, u32 count, const std::optional<T> restart,
                                 const std::array<u32, 6>& corners, T* out) {
    u32 written = 0;
    if (!restart) {
        for (u32 quad = 0; quad + 4 <= count; quad += 4) {
            for (const u32 corner : corners) {
                out[written++] = in[quad + corner];
            }
        }
        return written;
    }
    // A restart index ends the current primitive; an unfinished quad is dropped.
    std::array<T, 4> quad{};
    u32 filled = 0;
    for (u32 i = 0; i < count; ++i) {
        if (in[i] == *restart) {
            filled = 0;
            continue;
        }
        quad[filled++] = in[i];
        if (filled == 4) {
            for (const u32 corner : corners) {
                out[written++] = quad[corner];
            }
            filled = 0;
        }
    }
    return written;
}

static std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
                                          const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->DrawRegs().color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->DrawCbExtent(0));
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

// DIAG-044: a history of the images each draw and dispatch read and wrote, to find what a picture
// was made from (GT Sport's car thumbnail comes out as a black silhouette).
namespace {
enum class DiagRole : u8 { Sampled, Target, Storage, Depth };

struct DiagImageRef {
    u64 uid;
    u32 slot;
    VAddr address;
    u32 width;
    u32 height;
    vk::Format format;
    u64 version;
    VideoCore::ImageFlagBits flags;
    DiagRole role;
};

struct DiagDrawRecord {
    u64 seq;
    u64 vs;
    u64 fs;
    bool compute;
    u32 count;
    boost::container::small_vector<DiagImageRef, 6> images;
    struct BufferRef {
        VAddr address;
        u32 size;
        bool written;
    };
    boost::container::small_vector<BufferRef, 4> buffers;
    std::string empty;
};

std::mutex g_diag_history_mutex;
std::deque<DiagDrawRecord> g_diag_history;
u64 g_diag_seq = 0;
constexpr size_t DiagHistorySize = 150000;

const char* RoleName(DiagRole role) {
    switch (role) {
    case DiagRole::Sampled:
        return "reads";
    case DiagRole::Target:
        return "renders to";
    case DiagRole::Storage:
        return "writes storage";
    case DiagRole::Depth:
        return "depth";
    }
    return "?";
}
} // namespace

// PERF-039: the per-draw history behind DIAG-041/043 (the thumbnail investigation) and the
// empty-binding notes it carries cost every draw a record, image lookups and a lock. It is kept
// only with SHADGT_DIAG_HISTORY=1.
static bool DiagHistoryEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_DIAG_HISTORY");
        return env && env[0] == '1';
    }();
    return enabled;
}

void Rasterizer::RecordDiagHistory(const Pipeline* pipeline, bool compute, u32 count) {
    if (!DiagHistoryEnabled()) {
        return;
    }
    DiagDrawRecord record{};
    record.compute = compute;
    record.count = count;
    for (const auto* info : pipeline->GetStages()) {
        if (!info) {
            continue;
        }
        if (info->sw_stage == Shader::SwStage::Fragment) {
            record.fs = info->pgm_hash;
        } else if (record.vs == 0) {
            record.vs = info->pgm_hash;
        }
    }
    const auto add = [&](VideoCore::ImageId id, DiagRole role) {
        if (!id) {
            return;
        }
        const auto& image = texture_cache.GetImage(id);
        for (const auto& existing : record.images) {
            if (existing.uid == image.image_uid && existing.role == role) {
                return;
            }
        }
        record.images.push_back({image.image_uid, id.index, image.info.guest_address,
                                 image.info.size.width, image.info.size.height,
                                 image.info.pixel_format, image.contents_version, image.flags,
                                 role});
    };
    if (!compute) {
        const auto& key = static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey();
        for (u32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
            add(ctx->cb_descs[cb].image_id, DiagRole::Target);
        }
        add(ctx->db_desc.first, DiagRole::Depth);
    }
    for (const auto id : ctx->bound_images) {
        const auto& image = texture_cache.GetImage(id);
        if (image.binding.is_target) {
            continue;
        }
        const bool storage =
            image.binding.force_general ||
            std::ranges::find(ctx->diag_storage_images, id) != ctx->diag_storage_images.end();
        add(id, storage ? DiagRole::Storage : DiagRole::Sampled);
    }
    for (const auto& bound : ctx->bound_buffers) {
        if (bound.guest_address != 0) {
            record.buffers.push_back({bound.guest_address, bound.size, bound.is_written});
        }
    }
    record.empty = ctx->diag_empty_bindings;
    std::scoped_lock lk{g_diag_history_mutex};
    record.seq = ++g_diag_seq;
    g_diag_history.push_back(std::move(record));
    if (g_diag_history.size() > DiagHistorySize) {
        g_diag_history.pop_front();
    }
}

void Rasterizer::TraceImageDependencies(VideoCore::ImageId source_id) {
    std::scoped_lock lk{g_diag_history_mutex};
    const auto& source = texture_cache.GetImage(source_id);
    struct Pending {
        u64 uid;
        size_t before;
        u32 depth;
    };
    std::vector<Pending> queue{{source.image_uid, g_diag_history.size(), 0}};
    std::unordered_set<u64> visited{source.image_uid};
    std::vector<size_t> producers;
    std::unordered_set<size_t> producer_set;
    std::unordered_map<u64, DiagImageRef> images;
    std::unordered_map<u64, u32> image_depth{{source.image_uid, 0}};
    constexpr size_t MaxProducersPerImage = 400;
    constexpr size_t MaxProducers = 12000;
    for (size_t q = 0; q < queue.size() && producers.size() < MaxProducers; ++q) {
        const auto [uid, before, depth] = queue[q];
        size_t found = 0;
        for (size_t i = before; i-- > 0 && found < (depth == 0 ? size_t{3000} : MaxProducersPerImage);) {
            const auto& record = g_diag_history[i];
            const bool writes = std::ranges::any_of(record.images, [&](const DiagImageRef& ref) {
                return ref.uid == uid && ref.role != DiagRole::Sampled;
            });
            if (!writes) {
                continue;
            }
            ++found;
            if (!producer_set.insert(i).second) {
                continue;
            }
            producers.push_back(i);
            for (const auto& ref : record.images) {
                images.try_emplace(ref.uid, ref);
                if (ref.role == DiagRole::Sampled && depth < 12 && visited.insert(ref.uid).second) {
                    image_depth[ref.uid] = depth + 1;
                    queue.push_back({ref.uid, i, depth + 1});
                }
            }
        }
    }
    std::ranges::sort(producers);
    LOG_WARNING(Render_Vulkan,
                "DIAG-044: [{}] image {} at {:#x} ({}x{} {}) was made by {} draws and dispatches "
                "from {} images; history holds {} records",
                diag_trace_tag, source_id.index, source.info.guest_address, source.info.size.width,
                source.info.size.height, vk::to_string(source.info.pixel_format), producers.size(),
                visited.size(), g_diag_history.size());
    for (const size_t i : producers) {
        const auto& record = g_diag_history[i];
        std::string text;
        for (const auto& ref : record.images) {
            text += fmt::format(" | {} uid {} {:#x} {}x{} {} ver {} flags {:#x}",
                                RoleName(ref.role), ref.uid, ref.address, ref.width, ref.height,
                                vk::to_string(ref.format), ref.version, u32(ref.flags));
        }
        for (const auto& buffer : record.buffers) {
            text += fmt::format(" | buf {:#x}+{:#x}{}", buffer.address, buffer.size,
                                buffer.written ? " written" : "");
        }
        LOG_WARNING(Render_Vulkan, "DIAG-044: [{}] #{} {} vs {:#x} fs {:#x} count {}{}{}",
                    diag_trace_tag, record.seq, record.compute ? "dispatch" : "draw", record.vs,
                    record.fs, record.count, text, record.empty);
    }
    // How each image got its contents (uploads from guest memory, draws, copies).
    for (const u64 uid : visited) {
        std::string text;
        for (const auto& mod : VideoCore::RecentImageModifications(uid, 24)) {
            std::string_view file = mod.file ? mod.file : "?";
            if (const auto slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
                file.remove_prefix(slash + 1);
            }
            text += fmt::format(" | ver {} by {}:{}", mod.version, file, mod.line);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-044: [{}] uid {} depth {} modifications:{}",
                    diag_trace_tag, uid, image_depth[uid], text);
    }
    // The current pictures of every image still alive, nearest to the thumbnail first.
    std::vector<std::pair<u32, u64>> order;
    for (const u64 uid : visited) {
        order.emplace_back(image_depth[uid], uid);
    }
    std::ranges::sort(order);
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
    u32 dumped = 0;
    for (const auto& [depth, uid] : order) {
        const auto it = images.find(uid);
        const VideoCore::ImageId id =
            uid == source.image_uid
                ? source_id
                : (it != images.end() ? VideoCore::ImageId{it->second.slot} : VideoCore::ImageId{});
        if (!texture_cache.IsImageAlive(id, uid) || dumped >= 200) {
            continue;
        }
        const auto& image = texture_cache.GetImage(id);
        if (u64(image.info.size.width) * image.info.size.height > 4096ULL * 4096) {
            continue;
        }
        if (texture_cache.DumpImage(
                id, dir / fmt::format("dep{}_d{:02}_uid{}_{:#x}_{}x{}_{}.bin", diag_trace_tag,
                                      depth, uid, image.info.guest_address, image.info.size.width,
                                      image.info.size.height,
                                      vk::to_string(image.info.pixel_format)))) {
            ++dumped;
        }
    }
    LOG_WARNING(Render_Vulkan, "DIAG-044: dumped {} of those images to {}", dumped, dir.string());
}

extern std::atomic<u64> g_prepared_frames; // DIAG-137, vk_presenter.cpp

namespace {
// DIAG-080 (SHADGT_LOG_DRAW_SET=1): every second, the vertex/pixel shader pairs drawn, with how
// many of their draws were recorded and how many were dropped (no pipeline yet, or resources not
// bound), written to user/log/draw_set.txt. For finding effects that never reach the screen.
struct DrawSetEntry {
    u32 drawn{};
    u32 no_pipeline{};
    u32 no_resources{};
    u32 indirect{};
    u32 prim{}, instances{}, indices{}, width{}, height{}, format{}, blend{}, targets{};
};
struct DrawShape {
    u32 prim{}, instances{}, indices{}, width{}, height{}, format{}, blend{}, targets{};
};
thread_local DrawShape t_draw_shape{};
std::mutex g_draw_set_mutex;
std::map<std::pair<u64, u64>, DrawSetEntry> g_draw_set;
std::chrono::steady_clock::time_point g_draw_set_start{};
FILE* g_draw_set_file{};

bool DrawSetPerFrame() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_LOG_DRAW_SET");
        return env && env[0] == '2';
    }();
    return enabled;
}

bool DrawSetEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_LOG_DRAW_SET");
        return env && (env[0] == '1' || env[0] == '2');
    }();
    return enabled;
}

void NoteDrawKey(std::pair<u64, u64> key, int outcome);

void NoteDrawSet(const std::array<const Shader::Info*, MaxShaderStages>& infos, int outcome) {
    const auto hash = [&](Shader::SwStage stage) {
        const auto* info = infos[static_cast<u32>(stage)];
        return info ? info->pgm_hash : 0;
    };
    NoteDrawKey(std::make_pair(hash(Shader::SwStage::Vertex), hash(Shader::SwStage::Fragment)),
                outcome);
}

// Dispatches are listed with 0xc0c0c0c0 as the vertex shader and the compute shader second.
void NoteDispatch(const ComputePipeline* pipeline, int outcome) {
    const u64 cs = pipeline ? pipeline->GetStage(Shader::SwStage::Compute).pgm_hash : 0;
    NoteDrawKey(std::make_pair(u64{0xc0c0c0c0}, cs), outcome);
}

void NoteDrawKey(std::pair<u64, u64> key, int outcome) {
    std::scoped_lock lock{g_draw_set_mutex};
    const auto now = std::chrono::steady_clock::now();
    if (g_draw_set_start == std::chrono::steady_clock::time_point{}) {
        g_draw_set_start = now;
        g_draw_set_file = std::fopen("user/log/draw_set.txt", "w");
    }
    static auto window = now;
    // DIAG-137 (SHADGT_LOG_DRAW_SET=2): one line per game frame instead: its number and the
    // pairs drawn in it.
    if (DrawSetPerFrame()) {
        static u64 frame = 0;
        const u64 current = g_prepared_frames.load(std::memory_order_relaxed);
        if (current != frame && g_draw_set_file) {
            std::fprintf(g_draw_set_file, "F %llu", static_cast<unsigned long long>(frame));
            for (const auto& [pair, e] : g_draw_set) {
                std::fprintf(g_draw_set_file, " %08llx:%08llx:%u",
                             static_cast<unsigned long long>(pair.first & 0xffffffff),
                             static_cast<unsigned long long>(pair.second & 0xffffffff), e.drawn);
            }
            std::fputc('\n', g_draw_set_file);
            g_draw_set.clear();
            frame = current;
        }
        auto& e = g_draw_set[key];
        if (outcome == 0) ++e.drawn;
        return;
    }
    if (now - window >= std::chrono::seconds{1} && g_draw_set_file) {
        const double t = std::chrono::duration<double>(now - g_draw_set_start).count();
        for (const auto& [pair, e] : g_draw_set) {
            std::fprintf(g_draw_set_file, "%.0f %016llx %016llx %u %u %u %u\n", t,
                         static_cast<unsigned long long>(pair.first),
                         static_cast<unsigned long long>(pair.second), e.drawn, e.no_pipeline,
                         e.no_resources, e.indirect);
            std::fprintf(g_draw_set_file, "  shape %u %u %u %ux%u fmt %u blend %u targets %u\n",
                         e.prim, e.instances, e.indices, e.width, e.height, e.format, e.blend,
                         e.targets);
        }
        std::fflush(g_draw_set_file);
        g_draw_set.clear();
        window = now;
    }
    auto& e = g_draw_set[key];
    e.prim = t_draw_shape.prim;
    e.instances = std::max(e.instances, t_draw_shape.instances);
    e.indices = std::max(e.indices, t_draw_shape.indices);
    e.width = t_draw_shape.width;
    e.height = t_draw_shape.height;
    e.format = t_draw_shape.format;
    e.blend = t_draw_shape.blend;
    e.targets = t_draw_shape.targets;
    if (outcome == 0) ++e.drawn;
    else if (outcome == 1) ++e.no_pipeline;
    else if (outcome == 3) ++e.indirect;
    else ++e.no_resources;
}
} // namespace

static void NoteMenuBarDraw(u64 vs, u64 ps); // FIX-127
static thread_local bool t_frosted_rt = false; // DIAG-139: the draw renders the bar's blur

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    ++Common::GetWorkCounters().draws;
#ifdef RYTY_GPU
    if (native) {
        native->Draw(is_indexed, index_offset); // NAT-100
        return;
    }
#endif
    if (frame_dump_write_old) {
        FrameDumpWriteOld(); // NAT-116
    }
    // EXP-119 (SHADGT_CEIL_SKIPDRAW=1 after SHADGT_CEIL_AFTER seconds, measurement only, nothing
    // is drawn): the frame rate with draws costing nothing on the recorder.
    static const bool ceil_skip = [] {
        const char* env = std::getenv("SHADGT_CEIL_SKIPDRAW");
        return env && env[0] == '1';
    }();
    if (ceil_skip) {
        static const auto after = [] {
            const char* env = std::getenv("SHADGT_CEIL_AFTER");
            return std::chrono::steady_clock::now() + std::chrono::seconds(env ? std::atoi(env) : 0);
        }();
        if (std::chrono::steady_clock::now() >= after) {
            return;
        }
    }
    using Common::Phase;
    using Common::PhaseTimer;
    PhaseTimer total_timer{Phase::DrawTotal};

    {
        PhaseTimer t{Phase::DrawSetup};
        SubmitChunkIfNeeded();
        scheduler.PopPendingOperations();
    }
    // MT-002 test (SHADGT_MT_RING_TEST=1): the draw's commands go through a second ring and are
    // put back in order as a chunk, as a worker's will be.
    static const bool ring_test = [] {
        const char* env = std::getenv("SHADGT_MT_RING_TEST");
        return env && env[0] == '1';
    }();
    struct ChunkScope {
        Scheduler* scheduler{};
        CommandRing* worker{};
        u64 start{};
        ~ChunkScope() {
            if (worker) {
                Scheduler::SetThreadRing(nullptr);
                scheduler->RecordChunk(worker, start, worker->Recorded());
            }
        }
    } chunk_scope;
    if (ring_test && scheduler.HasCommandThread()) {
        static std::unique_ptr<CommandRing> test_ring = scheduler.NewWorkerRing();
        chunk_scope = {&scheduler, test_ring.get(), test_ring->Recorded()};
        Scheduler::SetThreadRing(test_ring.get());
    }

    const auto& regs = liverpool->DrawRegs();
    if (DrawSetEnabled()) {
        auto& shape = t_draw_shape;
        shape = {};
        shape.prim = u32(regs.primitive_type);
        shape.instances = regs.num_instances.NumInstances();
        shape.indices = regs.num_indices;
        for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
            if (regs.color_buffers[i]) {
                if (shape.targets == 0) {
                    shape.width = regs.color_buffers[i].Pitch();
                    shape.height = regs.color_buffers[i].Height();
                    shape.format = u32(regs.color_buffers[i].GetDataFmt());
                    shape.blend = regs.blend_control[i].enable;
                }
                ++shape.targets;
            }
        }
    }
    // EXP-128 (SHADGT_PASS_SKIP=<w>x<h>:<n>,...): draws into a color target of that size are
    // recorded only every n-th frame; the target keeps its last contents in between (a pass the
    // game redraws every frame, such as a reflection or mirror view, updated at a lower rate).
    static const auto pass_skips = [] {
        std::vector<std::array<u32, 3>> rules;
        if (const char* env = std::getenv("SHADGT_PASS_SKIP")) {
            std::string text = env;
            size_t at = 0;
            while (at < text.size()) {
                u32 w = 0, h = 0, n = 0;
                if (std::sscanf(text.c_str() + at, "%ux%u:%u", &w, &h, &n) == 3 && n > 1) {
                    rules.push_back({w, h, n});
                }
                const size_t comma = text.find(',', at);
                if (comma == std::string::npos) {
                    break;
                }
                at = comma + 1;
            }
        }
        return rules;
    }();
    if (!pass_skips.empty() && regs.color_buffers[0]) {
        const auto& cb = regs.color_buffers[0];
        for (const auto& [w, h, n] : pass_skips) {
            if (cb.Pitch() == w && cb.Height() == h && DebugState.GetFrameNum() % n != 0) {
                return;
            }
        }
    }
    const GraphicsPipeline* pipeline{};
    bool quad_triangles = false;
    u32 num_quad_indices = 0;
    {
        PhaseTimer t{Phase::Pipeline};
        if (!FilterDraw()) {
            return;
        }
        const bool quad_list = regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        quad_triangles = quad_list && Common::PerfFeatureEnabled(28) &&
                         CanDrawQuadListAsTriangles(is_indexed, index_offset);
        pipeline = AcquireGraphicsPipeline(
            {
                .vertex_sgpr_offset = 0,
                .instance_sgpr_offset = 0,
                .tessellate_quads = quad_list && !quad_triangles,
            },
            !quad_list);
        if (!pipeline) {
            if (DrawSetEnabled()) NoteDrawSet(pipeline_cache.SelectedInfos(), 1);
            return;
        }
    }
    // DIAG-138 (SHADGT_SKIP_PS=<hex,hex,...>): draws with these pixel shaders are skipped, to
    // find which draws make an effect.
    static const std::vector<u64> skip_ps = [] {
        std::vector<u64> list;
        if (const char* env = std::getenv("SHADGT_SKIP_PS")) {
            std::string text{env};
            size_t position = 0;
            while (position < text.size()) {
                const size_t end = std::min(text.find(',', position), text.size());
                list.push_back(std::strtoull(text.substr(position, end - position).c_str(), nullptr, 16));
                position = end + 1;
            }
        }
        return list;
    }();
    {
        const auto& infos = pipeline_cache.SelectedInfos();
        const auto* vs = infos[static_cast<u32>(Shader::SwStage::Vertex)];
        const auto* ps = infos[static_cast<u32>(Shader::SwStage::Fragment)];
        if (ps) {
            NoteMenuBarDraw(vs ? vs->pgm_hash : 0, ps->pgm_hash); // FIX-127
        }
    }
    if (!skip_ps.empty()) {
        const auto* ps = pipeline_cache.SelectedInfos()[static_cast<u32>(Shader::SwStage::Fragment)];
        if (ps && std::ranges::find(skip_ps, ps->pgm_hash & 0xffffffff) != skip_ps.end()) {
            return;
        }
    }

    {
        PhaseTimer t{Phase::RenderTargets};
        PrepareRenderState(pipeline);
    }
    VideoCore::BufferCache::t_ceil_draw = true; // EXP-127
    struct CeilDrawEnd {
        ~CeilDrawEnd() {
            VideoCore::BufferCache::t_ceil_draw = false;
        }
    } ceil_draw_end;
    {
        PhaseTimer t{Phase::VertexIndex};
        BindVertexBuffers(pipeline);
        if (quad_triangles) {
            num_quad_indices = BindQuadListIndices(is_indexed, index_offset);
        } else if (is_indexed) {
            BindIndexBuffer(index_offset);
        }
    }
    if (!BindResources(pipeline)) {
        if (DrawSetEnabled()) NoteDrawSet(pipeline_cache.SelectedInfos(), 2);
        return;
    }
    // DIAG-113 (SHADGT_DIAG_REUSE=1): how many draws bind exactly what a draw of the last
    // ~33 ms bound (pipeline, image views, non-stream buffers; stream offsets ignored), i.e. what
    // a draw-state cache could reuse.
    static const bool diag_reuse = [] {
        const char* env = std::getenv("SHADGT_DIAG_REUSE");
        return env && env[0] == '1';
    }();
    if (diag_reuse) {
        static std::unordered_set<u64> current, previous;
        static auto window = std::chrono::steady_clock::now();
        static auto report = window;
        static u64 draws = 0, hits = 0, stream_bindings = 0;
        const auto now = std::chrono::steady_clock::now();
        if (now - window >= std::chrono::milliseconds(33)) {
            previous.swap(current);
            current.clear();
            window = now;
        }
        u64 h = XXH3_64bits(&pipeline, sizeof(pipeline));
        for (const auto& info : ctx->image_infos) {
            h = XXH3_64bits_withSeed(&info.imageView, sizeof(info.imageView), h);
        }
        const vk::Buffer stream = buffer_cache.GetStreamBuffer().Handle();
        for (const auto& info : ctx->buffer_infos) {
            if (info.buffer == stream) {
                ++stream_bindings;
                continue;
            }
            h = XXH3_64bits_withSeed(&info, sizeof(info), h);
        }
        ++draws;
        hits += previous.contains(h) || current.contains(h);
        current.insert(h);
        if (now - report >= std::chrono::seconds(2)) {
            LOG_WARNING(Render_Vulkan,
                        "DIAG-113: {} draws, {} bind what a draw of the last ~33 ms bound "
                        "({:.0f}%), {:.1f} stream bindings per draw",
                        draws, hits, draws ? 100.0 * hits / draws : 0.0,
                        draws ? double(stream_bindings) / draws : 0.0);
            draws = hits = stream_bindings = 0;
            report = now;
        }
    }
    if (DrawSetEnabled()) {
        const auto& stages = pipeline->GetStages();
        const auto* vs = stages[u32(Shader::SwStage::Vertex)];
        const auto* fs = stages[u32(Shader::SwStage::Fragment)];
        NoteDrawKey(std::make_pair(vs ? vs->pgm_hash : 0, fs ? fs->pgm_hash : 0), 0);
    }
    RenderState state;
    {
        PhaseTimer t{Phase::BeginRendering};
        state = BeginRendering(pipeline);
        FinalizeTextureLayouts(&state);
        if (frame_dump_collecting) {
            FrameDumpNoteOld(); // NAT-116
        }
        if (FrameDumpActive()) {
            FrameDumpPass(regs, state, pipeline); // NAT-117
        }

        if (ctx->needs_barrier) {
            runtime.FlushBarriers();
        }
    }

    {
        PhaseTimer t{Phase::Descriptors};
        pipeline->BindResources(ctx->set_writes, ctx->push_data);
    }
    {
        PhaseTimer t{Phase::DynamicState};
        UpdateDynamicState(pipeline, is_indexed, quad_triangles);
        scheduler.BeginRendering(state);
    }
    PhaseTimer record_timer{Phase::Record};

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (quad_triangles) {
        // Non-indexed quads index from 0 with the first vertex as the vertex offset, which
        // gives the vertex shader the same vertex indices as the original draw.
        if (num_quad_indices != 0) {
            cmdbuf.drawIndexed(num_quad_indices, regs.num_instances.NumInstances(), 0,
                               s32(vertex_offset), instance_offset);
        }
    } else if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    DebugState.IncDrawCall();

    // DIAG-041: draws into GT Sport's car thumbnail (448x126 linear targets) and into the watched
    // images of its chain: the shaders, every image and buffer they touch, and empty bindings.
    // DIAG-043: kept in memory and written out (with image dumps) when the thumbnail is drawn.
    bool dump_thumbnail_chain = false;
    RecordDiagHistory(pipeline, false, regs.num_indices);
    // DIAG-104 (with SHADGT_DIAG_HISTORY=1): while user\log\trace-frame exists, the next whole
    // frame drawn into a display buffer is traced: every draw it came from, how each picture got
    // its contents, and the pictures dumped to user\log (dep frame_*.bin). For GT Sport's pink
    // effect rectangles.
    if (DiagHistoryEnabled()) {
        // Also on its own: every 10th frame, 4 points of the finished frame are read, and a frame
        // with the pink rectangles (magenta where they always appear) is traced, at most once
        // per 15 s and 6 times per run. Traces are numbered (frame1, frame2, ...).
        static auto last_check = std::chrono::steady_clock::time_point{};
        static auto last_trace = std::chrono::steady_clock::time_point{};
        static bool asked = false;
        static u32 traces = 0;
        static u32 frames = 0;
        static VideoCore::ImageId pending{};
        const auto now = std::chrono::steady_clock::now();
        const auto trigger = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "trace-frame";
        if (!asked && now - last_check > std::chrono::milliseconds{500}) {
            last_check = now;
            std::error_code ec;
            asked = std::filesystem::exists(trigger, ec);
        }
        const auto rt = ctx->cb_descs[0].image_id;
        if (rt) {
            const auto& info = texture_cache.GetImage(rt).info;
            const bool display = info.pixel_format == vk::Format::eA2B10G10R10UnormPack32 &&
                                 info.size.width >= 1280;
            if (display && pending && pending != rt) {
                bool pink = false;
                if (!asked && traces < 6 && ++frames % 10 == 0 &&
                    now - last_trace > std::chrono::seconds{15}) {
                    const auto& pinfo = texture_cache.GetImage(pending).info;
                    const u32 w = pinfo.size.width;
                    const u32 h = pinfo.size.height;
                    const std::array<std::pair<u32, u32>, 4> points{{{w / 10, h * 93 / 100},
                                                                     {w / 2, h * 93 / 100},
                                                                     {w * 9 / 10, h * 93 / 100},
                                                                     {w * 9 / 10, h * 70 / 100}}};
                    std::array<u32, 4> texels{};
                    u32 magenta = 0;
                    if (texture_cache.ReadTexels32(pending, points, texels)) {
                        for (const u32 t : texels) {
                            const u32 c0 = t & 1023;
                            const u32 c1 = (t >> 10) & 1023;
                            const u32 c2 = (t >> 20) & 1023;
                            magenta += c0 > 800 && c2 > 800 && c1 < 300;
                        }
                    }
                    pink = magenta >= 2;
                }
                if (asked || pink) {
                    ++traces;
                    last_trace = now;
                    diag_trace_tag = fmt::format("frame{}", traces);
                    LOG_WARNING(Render_Vulkan, "DIAG-104: tracing frame ({})",
                                pink ? "pink seen" : "asked");
                    TraceImageDependencies(pending);
                    std::error_code ec;
                    std::filesystem::remove(trigger, ec);
                    asked = false;
                    pending = {};
                } else {
                    pending = rt;
                }
            } else if (display) {
                pending = rt;
            }
        }
    }
    // DIAG-045: GT Sport accumulates the thumbnail's car frames into a 1600x900 target; trace the
    // frame being added while the images it was made from are still alive.
    {
        const auto rt = ctx->cb_descs[0].image_id;
        const auto& stages = pipeline->GetStages();
        const auto* fs = stages[u32(Shader::SwStage::Fragment)];
        if (rt && fs && fs->pgm_hash == 0x5589ec47) {
            const auto& info = texture_cache.GetImage(rt).info;
            static u32 accumulation_traces = 0;
            if (info.size.width == 1600 && info.size.height == 900 && accumulation_traces < 2) {
                for (const auto image_id : ctx->bound_images) {
                    if (!texture_cache.GetImage(image_id).binding.is_target) {
                        ++accumulation_traces;
                        diag_trace_source = image_id;
                        diag_trace_tag = fmt::format("acc{}", accumulation_traces);
                        break;
                    }
                }
            }
        }
    }
    {
        bool thumbnail = false;
        bool watched = false;
        for (u32 cb = 0; cb < std::bit_width(pipeline->GetGraphicsKey().mrt_mask); ++cb) {
            const auto image_id = ctx->cb_descs[cb].image_id;
            if (image_id) {
                const auto& info = texture_cache.GetImage(image_id).info;
                thumbnail |=
                    info.size.width == 448 && info.size.height == 126 && !info.props.is_tiled;
                watched |= VideoCore::IsWatchedImageAddress(info.guest_address, info.guest_size);
            }
        }
        if (thumbnail || watched) {
            static u64 records = 0;
            std::string images;
            for (const auto image_id : ctx->bound_images) {
                const auto& image = texture_cache.GetImage(image_id);
                images += fmt::format(
                    " | img {} {:#x} {}x{} {} {} gpu={} ver={}", image_id.index,
                    image.info.guest_address, image.info.size.width, image.info.size.height,
                    vk::to_string(image.info.pixel_format), AmdGpu::NameOf(image.info.tile_mode),
                    True(image.flags & VideoCore::ImageFlagBits::GpuModified),
                    image.contents_version);
            }
            for (u32 cb = 0; cb < std::bit_width(pipeline->GetGraphicsKey().mrt_mask); ++cb) {
                if (const auto image_id = ctx->cb_descs[cb].image_id) {
                    const auto& image = texture_cache.GetImage(image_id);
                    images += fmt::format(
                        " | rt{} {} {:#x} {}x{} {} mask {:#x}", cb, image_id.index,
                        image.info.guest_address, image.info.size.width, image.info.size.height,
                        vk::to_string(image.info.pixel_format), regs.color_target_mask.GetMask(cb));
                }
            }
            for (const auto& bound : ctx->bound_buffers) {
                if (bound.guest_address != 0) {
                    images += fmt::format(" | buf {:#x}+{:#x}{}", bound.guest_address, bound.size,
                                          bound.is_written ? " w" : "");
                }
            }
            images += ctx->diag_empty_bindings;
            images += fmt::format(" | {} {}", is_indexed ? "indexed" : "draw", regs.num_indices);
            const auto hash = [&](Shader::SwStage stage) -> u64 {
                const auto* info = pipeline->GetStages()[u32(stage)];
                return info ? info->pgm_hash : 0;
            };
            VideoCore::PushDiagRecord(fmt::format(
                "DIAG-041: {} draw {} vs {:#x} fs {:#x}{}", thumbnail ? "thumbnail" : "watched",
                ++records, hash(Shader::SwStage::Vertex), hash(Shader::SwStage::Fragment), images));
            if (thumbnail) {
                VideoCore::FlushDiagRecords(400);
                // DIAG-043: once per thumbnail source picture, the pictures of the whole chain.
                static u64 dumped_version = ~0ULL;
                for (const auto image_id : ctx->bound_images) {
                    const auto& image = texture_cache.GetImage(image_id);
                    if (!image.binding.is_target && image.contents_version != dumped_version) {
                        dumped_version = image.contents_version;
                        dump_thumbnail_chain = true;
                        diag_trace_source = image_id;
                        break;
                    }
                }
            }
        }
    }

    ResetBindings(false);

    if (diag_trace_source && !dump_thumbnail_chain) {
        TraceImageDependencies(diag_trace_source);
        diag_trace_source = {};
    }
    if (dump_thumbnail_chain) {
        diag_trace_tag = "thumb";
        static u32 dumps = 0;
        if (dumps < 3) {
            ++dumps;
            const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
            std::vector<VideoCore::ImageId> seen;
            for (const VAddr base : VideoCore::WatchedImageWindows()) {
                texture_cache.ForEachImageInRegion(base, VideoCore::WatchedImageWindowSize,
                                                   [&](VideoCore::ImageId id, VideoCore::Image&) {
                                                       if (std::ranges::find(seen, id) ==
                                                           seen.end()) {
                                                           seen.push_back(id);
                                                       }
                                                   });
            }
            u32 index = 0;
            for (const auto id : seen) {
                const auto& image = texture_cache.GetImage(id);
                const auto path = dir / fmt::format("chain{}_{:02}_{:#x}_{}x{}_{}_ver{}.bin", dumps,
                                                    index++, image.info.guest_address,
                                                    image.info.size.width, image.info.size.height,
                                                    vk::to_string(image.info.pixel_format),
                                                    image.contents_version);
                texture_cache.DumpImage(id, path);
            }
            LOG_WARNING(Render_Vulkan, "DIAG-043: dumped {} images of the thumbnail chain to {}",
                        index, dir.string());
        }
        if (diag_trace_source) {
            TraceImageDependencies(diag_trace_source);
            diag_trace_source = {};
        }
    }
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    RENDERER_TRACE;
    ++Common::GetWorkCounters().draws;
#ifdef RYTY_GPU
    if (native) {
        // NAT-100: GT Sport has no indirect draws (static analysis); not drawn in native mode.
        static std::atomic<u32> logged{};
        if (logged++ < 8) {
            LOG_WARNING(Render_Vulkan, "NAT-102: indirect draw skipped in native mode");
        }
        return;
    }
#endif

    SubmitChunkIfNeeded();
    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
        // PERF-028: the vertex count is in GPU memory, so quads cannot be expanded here.
        .tessellate_quads = true,
    };
    const GraphicsPipeline* pipeline = AcquireGraphicsPipeline(params, false);
    if (!pipeline) {
        if (DrawSetEnabled()) NoteDrawSet(pipeline_cache.SelectedInfos(), 1);
        return;
    }
    if (DrawSetEnabled()) NoteDrawSet(pipeline_cache.SelectedInfos(), 3);

    PrepareRenderState(pipeline);
    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer();
    }

    const auto [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);
    ctx->needs_barrier |= runtime.IsBufferAccessed(buffer, base, stride * max_count);
    ctx->bound_buffers.emplace_back(buffer, base, stride * max_count, false); // FIX-016

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        ctx->needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
        ctx->bound_buffers.emplace_back(count_buffer, count_offset, 4, false); // FIX-016
    }

    if (!BindResources(pipeline)) {
        return;
    }
    auto state = BeginRendering(pipeline);
    FinalizeTextureLayouts(&state);

    if (ctx->needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(ctx->set_writes, ctx->push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_offset, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_offset,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    }
    RecordDiagHistory(pipeline, false, max_count);

    ResetBindings(false);
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;
    ++Common::GetWorkCounters().dispatches;
#ifdef RYTY_GPU
    if (native) {
        native->DispatchDirect(); // NAT-100
        return;
    }
#endif
    Common::PhaseTimer total_timer{Common::Phase::DispatchTotal};

    SubmitChunkIfNeeded();
    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->DrawCsRegs();
    // PERF-055: selected on the command thread when it decoded the dispatch.
    const ComputePipeline* pipeline = AcquireComputePipeline();
    if (!pipeline) {
        if (DrawSetEnabled()) NoteDispatch(nullptr, 1);
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->DrawRegs(), cs_program, *this)) {
        if (DrawSetEnabled()) NoteDispatch(pipeline, 3);
        return;
    }

    if (!BindResources(pipeline)) {
        if (DrawSetEnabled()) NoteDispatch(pipeline, 2);
        return;
    }
    if (DrawSetEnabled()) NoteDispatch(pipeline, 0);
    if (FrameDumpActive()) {
        FrameDumpDispatch(cs, cs_program, false); // NAT-118
    }

    if (ctx->needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(ctx->set_writes, ctx->push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    DebugState.IncDispatch();
    RecordDiagHistory(pipeline, true, cs_program.dim_x * cs_program.dim_y * cs_program.dim_z);

    ResetBindings(true);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;
    ++Common::GetWorkCounters().dispatches;
#ifdef RYTY_GPU
    if (native) {
        native->DispatchIndirect(address, offset, size); // NAT-100
        return;
    }
#endif

    SubmitChunkIfNeeded();
    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->DrawCsRegs();
    // PERF-055: looked up here (no selection for indirect dispatches), with fresh stage copies.
    const ComputePipeline* pipeline = AcquireComputePipeline();
    if (!pipeline) {
        return;
    }
    if (FrameDumpActive()) {
        FrameDumpDispatch(pipeline->GetStage(Shader::SwStage::Compute), cs_program,
                          true); // NAT-118
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    ctx->needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    ctx->bound_buffers.emplace_back(buffer, base, size, false); // FIX-016

    if (!BindResources(pipeline)) {
        return;
    }

    if (ctx->needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(ctx->set_writes, ctx->push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    DebugState.IncDispatch();
    RecordDiagHistory(pipeline, true, 0);

    ResetBindings(true);
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

// DIAG-011: constants the traced tonemap draw used, re-checked after later submissions.
struct TonemapConstantCheck {
    VAddr address{};
    std::array<u32, 76> used{};
    u32 checks_left{};
    u64 id{};
};
static std::vector<TonemapConstantCheck> g_tonemap_checks;

static void RecheckTonemapConstants() {
    for (auto& check : g_tonemap_checks) {
        if (check.checks_left == 0) {
            continue;
        }
        --check.checks_left;
        const auto* now = reinterpret_cast<const u32*>(check.address);
        u32 changed{};
        for (u32 i = 0; i < check.used.size(); ++i) {
            changed += now[i] != check.used[i];
        }
        LOG_WARNING(Render_Vulkan,
                    "Tonemap constants recheck {} at {:#x} (pass {}): {} dwords changed; "
                    "used [4..7]={:08x} {:08x} {:08x} {:08x} [47]={:08x}, now [4..7]={:08x} "
                    "{:08x} {:08x} {:08x} [47]={:08x}",
                    check.id, check.address, 2 - check.checks_left, changed, check.used[4],
                    check.used[5], check.used[6], check.used[7], check.used[47], now[4], now[5],
                    now[6], now[7], now[47]);
    }
    std::erase_if(g_tonemap_checks, [](const auto& check) { return check.checks_left == 0; });
}

void Rasterizer::WhenGpuDone(Common::UniqueFunction<void>&& func) {
    scheduler.DeferPriorityOperation(std::move(func));
}

void Rasterizer::OnSubmit() {
    RecheckTonemapConstants();
    texture_cache.ReleaseFinishedReadbacks();
    buffer_cache.ReleaseFinishedAsyncReadbacks();
    // GDS readbacks not taken by a deferred fence complete once this submission executes.
    if (auto async_readbacks = buffer_cache.TakePendingAsyncReadbacks(); !async_readbacks.empty()) {
        scheduler.DeferPriorityOperation([this, async_readbacks = std::move(async_readbacks)] {
            buffer_cache.CompleteAsyncReadbacks(async_readbacks);
        });
    }
    buffer_cache.TickFrame();
    ProcessDownloadsTimed(DrainSource::Submit);
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

void Rasterizer::OnFence(DrainSource source) {
    ProcessDownloadsTimed(source);
}

void Rasterizer::FinishForGds() {
    const auto start = std::chrono::steady_clock::now();
    scheduler.Finish();
    RecordDrain(DrainSource::GdsStore, start);
}

void Rasterizer::ProcessDownloadsTimed(DrainSource source) {
    if (!texture_cache.HasPendingReadbacks()) {
        texture_cache.ProcessDownloadImages();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    texture_cache.ProcessDownloadImages();
    RecordDrain(source, start);
}

void Rasterizer::RecordDrain(DrainSource source, std::chrono::steady_clock::time_point start) {
    // DIAG-016: synchronous GPU drains per source, reported every 2 seconds.
    const auto now = std::chrono::steady_clock::now();
    auto& stats = drain_stats;
    if (stats.window_start == std::chrono::steady_clock::time_point{}) {
        stats.window_start = now;
    }
    const u32 index = static_cast<u32>(source);
    ++stats.count[index];
    stats.total_us[index] +=
        std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
    if (now - stats.window_start < std::chrono::seconds{2}) {
        return;
    }
    static constexpr std::array names = {"submit",    "gfx_eos",         "gfx_eop",  "gfx_write",
                                         "asc_write", "asc_release_mem", "gds_store"};
    std::string summary;
    for (u32 i = 0; i < names.size(); ++i) {
        if (stats.count[i] != 0) {
            summary += fmt::format(" {}={}/{:.1f}ms", names[i], stats.count[i],
                                   stats.total_us[i] / 1000.0);
        }
    }
    LOG_WARNING(Render_Vulkan, "GPU drains in {:.1f} s (count/total):{}",
                std::chrono::duration<double>(now - stats.window_start).count(), summary);
    stats = {};
    stats.window_start = now;
}

extern std::atomic<u64> g_defer_readbacks; // DIAG-068
extern std::atomic<u64> g_defer_address;
extern std::atomic<u64> g_defer_order;
extern std::atomic<u64> g_defer_unflushed; // D5

// DIAG-129 (SHADGT_DIAG_DIRECT_HAZARD=1): ranges recently bound as written, checked against the
// ranges the GPU reads in place. An overlap means an in-place read can see bytes a shader wrote
// into a cached copy that guest memory does not have yet.
static bool DirectHazardDiag() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_DIAG_DIRECT_HAZARD");
        return env && env[0] == '1';
    }();
    return enabled;
}
struct WrittenRange {
    VAddr address;
    u64 size;
    u64 hash;
};
static std::mutex g_written_mutex;
static std::array<WrittenRange, 4096> g_written_ranges{};
static u32 g_written_next = 0;
static void NoteWrittenRange(VAddr address, u64 size, u64 hash) {
    std::scoped_lock lk{g_written_mutex};
    g_written_ranges[g_written_next++ % g_written_ranges.size()] = {address, size, hash};
}
static void CheckInPlaceHazard(const char* what, VAddr address, u64 size, u64 reader) {
    std::scoped_lock lk{g_written_mutex};
    static std::unordered_set<u64> seen;
    for (const auto& w : g_written_ranges) {
        if (w.size == 0 || w.address >= address + size || address >= w.address + w.size) {
            continue;
        }
        const u64 key = reader * 31 + w.hash;
        if (seen.size() < 400 && seen.insert(key).second) {
            LOG_WARNING(Render_Vulkan,
                        "DIAG-129: {} read in place {:#x}+{:#x} by {:#x} overlaps written "
                        "{:#x}+{:#x} by {:#x}",
                        what, address, size, reader, w.address, w.size, w.hash);
        }
        return;
    }
}

// FIX-123: ranges a shader wrote recently are not read in place. The written bytes live in the
// buffer cache's copy until they are read back, and the GPU-written mark of a range can be gone
// before guest memory has them: GT Sport's compute shader 0xa82d2fe writes 64-byte records that
// dozens of later shaders read, and read in place they saw stale bytes (car screens: the car
// view blown out to white, the dashboard panel white or magenta, the white frame after picking a
// car). Writes are remembered for 250 ms by 1 KB block (reads up to 64 KB) and by 1 MB block
// (larger reads), in direct-mapped tables. -DisablePerf 145 reads them in place as before.
namespace {
constexpr u32 FineBits = 10;
constexpr u32 CoarseBits = 20;
constexpr u64 RecentWriteMs = 250;
std::array<std::atomic<u64>, 1u << 20> g_fine_writes{};
std::array<std::atomic<u64>, 1u << 14> g_coarse_writes{};
u32 NowMs() {
    return u32(std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}
bool RecentWritesEnabled() {
    static const bool enabled = Common::PerfFeatureEnabled(145);
    return enabled;
}
template <size_t N>
void MarkBlocks(std::array<std::atomic<u64>, N>& table, u32 bits, VAddr address, u64 size,
                u32 now) {
    const u64 first = address >> bits;
    const u64 last = (address + size - 1) >> bits;
    for (u64 block = first; block <= last; ++block) {
        table[block % N].store((block << 32) | now, std::memory_order_relaxed);
    }
}
template <size_t N>
bool AnyRecent(const std::array<std::atomic<u64>, N>& table, u32 bits, VAddr address, u64 size,
               u32 now) {
    const u64 first = address >> bits;
    const u64 last = (address + size - 1) >> bits;
    for (u64 block = first; block <= last; ++block) {
        const u64 entry = table[block % N].load(std::memory_order_relaxed);
        if ((entry >> 32) == block && u32(now - u32(entry)) < RecentWriteMs) {
            return true;
        }
    }
    return false;
}
} // namespace

bool RytyGtsInRace(); // FIX-110, ryty_game_patch.cpp

// FIX-124: outside races buffers are not read in place. GT Sport's car screens still showed a
// blown-out car or a white dashboard panel every few seconds with FIX-122/123 (no cause found
// yet; it needs small and large ranges read in place together). The menus run at 60 fps either
// way; races keep the direct reads. -DisablePerf 146 reads in place everywhere.
// FIX-127: GT Sport's menu top bar (ps 0x90b5dcdd, its frosted glass: drawn in every menu, the race
// preview and the pause menu, never while driving) also turns in-place reads off, and with them
// the fence waits (FIX-125): on the race preview, a race already, the bar's frosted glass came out
// dark every other frame on Lance's PC (exclusive fullscreen at 240 Hz) even with FIX-126.
// -DisablePerf 149 leaves the race preview to FIX-126 alone.
static std::atomic<u32> g_menu_bar_ms{0};
static void NoteMenuBarDraw(u64 vs, u64 ps) {
    if ((ps & 0xffffffff) == 0x90b5dcdd) {
        g_menu_bar_ms.store(NowMs() | 1, std::memory_order_relaxed);
    }
}
static bool MenuBarShown() {
    static const bool enabled = Common::PerfFeatureEnabled(149);
    const u32 last = g_menu_bar_ms.load(std::memory_order_relaxed);
    return enabled && last != 0 && u32(NowMs() - last) < 500;
}

static bool DirectReadsHere() {
    static const bool menus_off = Common::PerfFeatureEnabled(146);
    return !menus_off || (RytyGtsInRace() && !MenuBarShown());
}

// FIX-125: graphics fences wait for the GPU only while buffers are read in place (and for 0.5 s
// after the last such read). The wait is what makes in-place reads safe, but it changes the
// game's frame pacing: GT Sport's frosted-glass bars (top bar, car list) lost their blur about
// every third frame in the menus, where FIX-124 reads nothing in place anyway.
// -DisablePerf 147 holds them always.
static std::atomic<u32> g_last_direct_read_ms{0};
static void NoteDirectRead() {
    g_last_direct_read_ms.store(NowMs() | 1, std::memory_order_relaxed);
    // DIAG-141 (SHADGT_DIAG_FROSTED=1): in-place reads while the menu bar is on screen.
    static const bool diag = [] {
        const char* env = std::getenv("SHADGT_DIAG_FROSTED");
        return env && env[0] == '1';
    }();
    if (diag && MenuBarShown()) {
        static std::atomic<u32> n{};
        if (const u32 c = ++n; c <= 20 || c % 1000 == 0) {
            LOG_WARNING(Render_Vulkan, "DIAG-141: in-place read {} while the menu bar shows", c);
        }
    }
}
static bool DirectReadsRecently() {
    static const bool enabled = Common::PerfFeatureEnabled(147);
    if (!enabled) {
        return true;
    }
    const u32 last = g_last_direct_read_ms.load(std::memory_order_relaxed);
    return last != 0 && u32(NowMs() - last) < 500;
}

static void NoteShaderWrite(VAddr address, u64 size) {
    if (!RecentWritesEnabled() || size == 0) {
        return;
    }
    const u32 now = NowMs();
    if (size <= (u64{16} << 20)) {
        MarkBlocks(g_fine_writes, FineBits, address, size, now);
    }
    MarkBlocks(g_coarse_writes, CoarseBits, address, size, now);
}

static bool WrittenRecently(VAddr address, u64 size) {
    if (!RecentWritesEnabled() || size == 0) {
        return false;
    }
    const u32 now = NowMs();
    if (size <= (u64{64} << 10)) {
        return AnyRecent(g_fine_writes, FineBits, address, size, now);
    }
    return AnyRecent(g_coarse_writes, CoarseBits, address, size, now);
}

std::optional<std::pair<const VideoCore::Buffer*, u64>> Rasterizer::DirectReadBuffer(VAddr address,
                                                                                    u64 size) {
    // D6: a range only the CPU writes, read by the GPU in place (vertex and index buffers).
    auto* direct = buffer_cache.GetDirectMemory();
    if (!direct || size == 0 || direct->IsVolatile(address, size) ||
        !direct->EnsureImported(address, size) || buffer_cache.IsRegionGpuModified(address, size) ||
        WrittenRecently(address, size) || !DirectReadsHere() || // FIX-123, FIX-124
        buffer_cache.DirectReadHitsImage(address, size)) { // FIX-122
        return std::nullopt;
    }
    if (DirectHazardDiag()) {
        CheckInPlaceHazard("vertex/index", address, size, 0);
    }
    auto found = direct->FindBuffer(address, size);
    if (found) {
        NoteDirectRead(); // FIX-125
    }
    return found;
}

void Rasterizer::FlushDeferredFences() {
    if (deferred_unflushed.exchange(false, std::memory_order_acq_rel)) {
        scheduler.Flush();
    }
}

bool Rasterizer::DeferFenceSignal(VAddr address, Common::UniqueFunction<void>&& signal,
                                  bool compute_queue, std::span<const u8> value,
                                  bool flush) {
    // Hardware writes a fence when the work before it has finished. With readbacks pending,
    // signal after the GPU completes instead of draining the GPU now. A later write to an address
    // that already has a deferred write is deferred too, so that address keeps its order;
    // fences to other addresses are signaled immediately as before.
    // Pending GDS readbacks ride along with fences deferred for image readbacks; they do not
    // cause extra deferrals (each deferral adds a submit and a guest wait).
    // PERF-010: GPU-written pages the CPU faulted on before are read back with this fence.
    // They ride along like GDS readbacks: deferring fences for them made races slower
    // (14-18 FPS vs 20-35), as did deferring compute-queue fences for image readbacks
    // (PERF-009 v1). Compute-queue fences are only deferred to keep an address's order.
    buffer_cache.RecordHotPageReadbacks();
    // FIX-032: GT Sport's car thumbnail is drawn in passes (a silhouette, then the car) and
    // read by the CPU after a compute-queue fence. Signaled before the later passes were read
    // back, the CPU encoded the silhouette and the game stopped with "BREAK!
    // thumbnail_functions.ad:473". Compute-queue fences wait for large image readbacks too
    // (they are rare, unlike the small ones PERF-009 v1 slowed races with).
    // -DisablePerf 40 signals them right away.
    static const bool wait_large = Common::PerfFeatureEnabled(40);
    // FIX-070: small images read back every frame (GT Sport's 8x8 to 1x1 brightness chain for
    // auto exposure) no longer hold the fence. Each one deferred a fence, and every later fence
    // waited behind it (FIX-049), so the game waited for full GPU completion every frame: races
    // ran at about 30 FPS with the recorder 75-88% busy. They are read back when the GPU gets
    // there without holding anything; the game sees the value a frame later at most. Large ones
    // (car thumbnails, FIX-032) still hold the fence. -DisablePerf 93 holds fences for all.
    static const bool small_in_background = Common::PerfFeatureEnabled(93);
    if (small_in_background && !compute_queue && texture_cache.HasPendingReadbacks() &&
        !texture_cache.HasLargePendingReadbacks()) {
        texture_cache.ReleaseFinishedReadbacks();
        if (auto small = texture_cache.RecordPendingReadbacks(); !small.empty()) {
            ++background_readbacks;
            scheduler.DeferPriorityOperation([this, small = std::move(small)]() mutable {
                texture_cache.CompleteReadbacks(small);
            });
        }
    }
    const bool readbacks_pending = compute_queue
                                       ? wait_large && texture_cache.HasLargePendingReadbacks()
                                       : texture_cache.HasPendingReadbacks();
    bool address_pending;
    {
        std::scoped_lock lk{deferred_fences_mutex};
        address_pending = deferred_fence_addresses.contains(address);
    }
    // FIX-049: fences reach memory in order, as on hardware. A fence signaled now while an
    // earlier one is still deferred would let the game see the later fence first, reuse memory
    // the earlier fence's label lives in, and then have that label written over the new data:
    // in the Nurburgring benchmark a deferred label landed on a command buffer the game had
    // refilled, which stopped the emulator with "Invalid PM4 type 0" (also in runs of October 8
    // and 9). The deferred operations run in GPU order, so deferring this one keeps the order.
    // -DisablePerf 68 signals other addresses at once as before.
    static const bool keep_order = Common::PerfFeatureEnabled(68);
    const bool earlier_deferred = keep_order && deferred_fences.load() != 0;
    // D3: with direct memory the GPU reads guest memory in place, so the game may only reuse
    // it once the GPU has finished: graphics fences wait for the GPU.
    // EXP-129 (SHADGT_DIRECT_NOHOLD=1, measurement only, may render wrong): no hold.
    // =<seconds>: only after that many seconds (the race, past menus and loading).
    static const u64 no_hold_after = [] {
        const char* env = std::getenv("SHADGT_DIRECT_NOHOLD");
        return env && *env ? std::strtoull(env, nullptr, 10) : ~u64{0};
    }();
    static const auto hold_start = std::chrono::steady_clock::now();
    const bool no_hold =
        no_hold_after != ~u64{0} &&
        std::chrono::steady_clock::now() - hold_start > std::chrono::seconds(no_hold_after);
    // DIAG-130 (SHADGT_DIRECT_HOLD_COMPUTE=1): compute-queue fences wait for the GPU too.
    static const bool hold_compute = [] {
        const char* env = std::getenv("SHADGT_DIRECT_HOLD_COMPUTE");
        return env && env[0] == '1';
    }();
    // FIX-126: only the labels written between draws (EOS, WRITE_DATA) wait for the GPU; the
    // end-of-pipe fences (EOP, which pass flush) are signaled as before. Held EOPs made GT Sport's
    // frosted-glass top bar lose its blur on about one frame in five on the pre-race screen
    // (a race, so FIX-125 still holds there); races render the same without them.
    // -DisablePerf 148 holds both. DIAG-136 (SHADGT_DIRECT_HOLD_KIND=eop|labels|all) picks.
    static const int hold_kind = [] {
        const char* env = std::getenv("SHADGT_DIRECT_HOLD_KIND");
        if (env && *env) {
            return env[0] == 'e' ? 1 : env[0] == 'l' ? 2 : 0;
        }
        return Common::PerfFeatureEnabled(148) ? 2 : 0;
    }();
    const bool kind_held = hold_kind == 0 || (hold_kind == 1) == flush;
    // FIX-127: no waits while the menu bar is on screen, even when a few in-place reads slipped
    // through (planned ahead before the bar was drawn): they kept the waits on all along.
    bool direct_hold = !no_hold && (!compute_queue || hold_compute) && kind_held &&
                             buffer_cache.GetDirectMemory() != nullptr &&
                             DirectReadsRecently() && !MenuBarShown(); // FIX-125, FIX-127
#ifdef RYTY_GPU
    // NAT-100: native mode reads all guest memory in place: graphics fences wait for the GPU.
    direct_hold = direct_hold || (native && !compute_queue);
#endif
    if (direct_hold && MenuBarShown()) {
        // DIAG-141: fences held while the menu bar is on screen.
        static const bool diag = [] {
            const char* env = std::getenv("SHADGT_DIAG_FROSTED");
            return env && env[0] == '1';
        }();
        static std::atomic<u32> n{};
        if (diag) {
            if (const u32 c = ++n; c <= 20 || c % 1000 == 0) {
                LOG_WARNING(Render_Vulkan, "DIAG-141: fence {} held while the menu bar shows", c);
            }
        }
    }
    if (!readbacks_pending && !address_pending && !earlier_deferred && !direct_hold) {
        return false;
    }
    // DIAG-068: why fences wait for the GPU.
    ++(readbacks_pending ? g_defer_readbacks : address_pending ? g_defer_address : g_defer_order);
    if (readbacks_pending) {
        last_readback_fence_ns.store(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count(),
            std::memory_order_relaxed);
    }
    texture_cache.ReleaseFinishedReadbacks();
    buffer_cache.ReleaseFinishedAsyncReadbacks();
    auto readbacks = texture_cache.RecordPendingReadbacks();
    auto async_readbacks = buffer_cache.TakePendingAsyncReadbacks();
    const bool brings_data = !readbacks.empty() || !async_readbacks.empty();
    if (brings_data) {
        ++readback_fences;
    }
    {
        std::scoped_lock lk{deferred_fences_mutex};
        ++deferred_fence_addresses[address];
        if (value.empty()) {
            pending_fence_bytes.erase(address);
        } else {
            pending_fence_bytes[address].assign(value.begin(), value.end());
        }
    }
    ++deferred_fences;
    const auto deferred_at = std::chrono::steady_clock::now();
    scheduler.DeferPriorityOperation(
        [this, address, deferred_at, brings_data, readbacks = std::move(readbacks),
         async_readbacks = std::move(async_readbacks), signal = std::move(signal)]() mutable {
            texture_cache.CompleteReadbacks(readbacks);
            buffer_cache.CompleteAsyncReadbacks(async_readbacks);
            signal();
            RecordDeferredFenceLatency(deferred_at);
            {
                std::scoped_lock lk{deferred_fences_mutex};
                if (--deferred_fence_addresses[address] == 0) {
                    deferred_fence_addresses.erase(address);
                    pending_fence_bytes.erase(address);
                }
            }
            --deferred_fences;
            if (brings_data) {
                --readback_fences;
            }
        });
    // Submit so the deferred tick can complete; the GPU thread does not wait.
    // D5: in direct mode most fences are held only to keep order (labels written by WRITE_DATA
    // between draws). Those ride the next submit, the next EOP's or the one the recorder makes
    // when it runs out of work (a game waiting for one stops sending work).
    // DIAG-134 (SHADGT_DIRECT_HOLD_FLUSH=1): held fences submit at once instead of riding the
    // next submit (D5).
    static const bool hold_flush = [] {
        const char* env = std::getenv("SHADGT_DIRECT_HOLD_FLUSH");
        return env && env[0] == '1';
    }();
    if (flush || hold_flush || readbacks_pending || address_pending || !direct_hold) {
        scheduler.Flush();
        deferred_unflushed.store(false, std::memory_order_relaxed);
    } else {
        deferred_unflushed.store(true, std::memory_order_release);
        ++g_defer_unflushed;
    }
    return true;
}

void Rasterizer::FrameDumpNoteOld() {
    const auto note = [&](VideoCore::ImageId id) {
        if (id && std::ranges::find(frame_dump_old, id, &std::pair<VideoCore::ImageId, u64>::first) ==
                      frame_dump_old.end()) {
            frame_dump_old.emplace_back(id, texture_cache.GetImage(id).image_uid);
        }
    };
    for (const auto& binding : ctx->cb_descs) {
        note(binding.image_id);
    }
    note(ctx->db_desc.first);
}

void Rasterizer::FrameDumpFlip() {
    static const double dump_at = [] {
        const char* env = std::getenv("RYGT_FRAME_DUMP_AT");
        return env && *env ? std::atof(env) : -1.0;
    }();
    static const auto start = std::chrono::steady_clock::now();
    if (dump_at < 0 || frame_dump_done) {
        return;
    }
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // NAT-128 (RYGT_FRAME_DUMP_FRAMES=n): n frames are collected (events of each, the targets and
    // textures as the last one left them), for work the game does every few frames only.
    static const u32 dump_frames = [] {
        const char* env = std::getenv("RYGT_FRAME_DUMP_FRAMES");
        return env && *env ? std::max(std::atoi(env), 1) : 1;
    }();
    static u32 frames_seen = 0;
    if (!frame_dump_collecting) {
        if (now >= dump_at) {
            frame_dump_collecting = true; // the next frame's targets are noted
            frame_dump_old.clear();
            frames_seen = 0;
        }
        return;
    }
    if (++frames_seen < dump_frames) {
        FrameDumpEvent(fmt::format("---- flip {} ----", frames_seen));
        return;
    }
    frame_dump_collecting = false;
    frame_dump_done = true;
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) /
                     fmt::format("dump-{}-{}", native ? "native" : "old", u32(dump_at));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    FrameDumpWriteEvents(dir);
#ifdef RYTY_GPU
    if (native) {
        native->DumpFrame(dir);
        return;
    }
#endif
    // Copies recorded at the flip crashed the driver; the next draw writes them.
    frame_dump_dir = dir;
    frame_dump_write_old = true;
}

void Rasterizer::FrameDumpPass(const AmdGpu::Regs& regs, const RenderState& state,
                               const GraphicsPipeline* pipeline) {
    std::string text = "pass";
    for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
        const auto& attachment = state.color_attachments[cb];
        if (!attachment.image_view) {
            continue;
        }
        const auto& col_buf = regs.color_buffers[cb];
        const auto extent = liverpool->DrawCbExtent(cb);
        text += fmt::format(" c{}={:#x} {}x{} fmt {}/{}{}", cb, col_buf.Address(), extent.width,
                            extent.height, u32(col_buf.GetDataFmt()),
                            u32(col_buf.GetNumberFmt()), attachment.is_clear ? " CLEAR" : "");
    }
    const auto& db = state.depth_stencil_attachment;
    if (db.image_view) {
        text += fmt::format(" z={:#x} stencil={:#x} htile={:#x}{}{}",
                            regs.depth_buffer.DepthAddress(), regs.depth_buffer.StencilAddress(),
                            regs.depth_htile_data_base.GetAddress(),
                            db.depth_clear ? fmt::format(" ZCLEAR {}", regs.depth_clear)
                                           : std::string{},
                            db.stencil_clear ? " SCLEAR" : "");
    }
    text += fmt::format(" | {}x{}", state.width, state.height);
    std::scoped_lock lock{frame_dump_mutex};
    if (TraceRange().first != TraceRange().second && text != trace_last_pass) {
        trace_last_pass = text;
        TraceText(text); // NAT-130
    }
    if (!frame_dump_collecting) {
        return;
    }
    if (!frame_dump_events.empty() && frame_dump_events.back().draws != 0 &&
        frame_dump_events.back().key == text) {
        ++frame_dump_events.back().draws;
        return;
    }
    // NAT-118: how the pass's first draw tests and writes depth, and its programs.
    const auto& dc = regs.depth_control;
    u64 vs = 0, ps = 0;
    if (pipeline) {
        const auto& stages = pipeline->GetStages();
        const auto* v = stages[u32(Shader::SwStage::Vertex)];
        const auto* f = stages[u32(Shader::SwStage::Fragment)];
        vs = v ? v->pgm_hash : 0;
        ps = f ? f->pgm_hash : 0;
    }
    // NAT-125: the textures the pass's first draw samples (T# address and size, all stages).
    std::string sampled;
    if (pipeline) {
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            for (const auto& image : stage->images) {
                if (image.is_written) {
                    continue;
                }
                const auto tsharp = image.GetSharp(*stage);
                sampled += fmt::format(" {:#x}:{}x{}m{}l{}", tsharp.Address(), tsharp.width + 1,
                                       tsharp.height + 1, tsharp.last_level - tsharp.base_level + 1,
                                       tsharp.last_array - tsharp.base_array + 1);
            }
        }
    }
    std::string detail = fmt::format(
        "first draw: prim {} n {} z test {} write {} func {} stencil {} dbctl clear {}/{} "
        "vs {:#x} ps {:#x} tex{}",
        u32(regs.primitive_type), regs.num_indices, u32(dc.depth_enable),
        u32(dc.depth_write_enable), u32(dc.depth_func),
        u32(dc.stencil_enable), u32(regs.depth_render_control.depth_clear_enable),
        u32(regs.depth_render_control.stencil_clear_enable), vs, ps, sampled);
    frame_dump_events.push_back({std::move(text), std::move(detail), 1});
}

void Rasterizer::FrameDumpEvent(std::string text) {
    std::scoped_lock lock{frame_dump_mutex};
    if (TraceRange().first != TraceRange().second) {
        TraceText(text); // NAT-130
    }
    if (frame_dump_collecting) {
        frame_dump_events.push_back({std::move(text), {}, 0});
    }
}

std::pair<VAddr, VAddr> Rasterizer::TraceRange() {
    static const std::pair<VAddr, VAddr> range = [] {
        const char* env = std::getenv("RYGT_TRACE_RANGE");
        if (!env || !*env) {
            return std::pair<VAddr, VAddr>{};
        }
        char* end = nullptr;
        const VAddr lo = std::strtoull(env, &end, 16);
        const VAddr hi = end && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : lo + 1;
        return std::pair<VAddr, VAddr>{lo, std::max(hi, lo)};
    }();
    return range;
}

void Rasterizer::TraceText(const std::string& text) {
    // Any address in the text inside the range.
    for (size_t at = text.find("0x"); at != std::string::npos; at = text.find("0x", at + 2)) {
        const VAddr address = std::strtoull(text.c_str() + at + 2, nullptr, 16);
        if (Traced(address, 1)) {
            LOG_WARNING(Render_Vulkan, "NAT-130: {} (frame {})", text, DebugState.GetFrameNum());
            return;
        }
    }
}

void Rasterizer::FrameDumpDispatch(const Shader::Info& cs, const AmdGpu::ComputeProgram& program,
                                   bool indirect) {
    std::string text = indirect ? fmt::format("dispatch cs {:#x} indirect", cs.pgm_hash)
                                : fmt::format("dispatch cs {:#x} {}x{}x{}", cs.pgm_hash,
                                              program.dim_x, program.dim_y, program.dim_z);
    for (const auto& desc : cs.buffers) {
        if (!desc.IsSpecial() && desc.is_written) {
            const auto sharp = desc.GetSharp(cs);
            text += fmt::format(" writes {:#x}+{:#x}", sharp.base_address, sharp.GetSize());
        }
    }
    for (const auto& desc : cs.images) {
        if (desc.is_written) {
            const auto sharp = desc.GetSharp(cs);
            text += fmt::format(" stores {:#x} {}x{}", sharp.Address(), sharp.width + 1,
                                sharp.height + 1);
        }
    }
    // NAT-130: what it samples (after the outputs, so a trace of a range names the writer).
    for (const auto& desc : cs.images) {
        if (!desc.is_written) {
            const auto sharp = desc.GetSharp(cs);
            text += fmt::format(" | reads {}x{}m{}l{} at {:x}", sharp.width + 1, sharp.height + 1,
                                sharp.last_level - sharp.base_level + 1,
                                sharp.last_array - sharp.base_array + 1, sharp.Address());
        }
    }
    FrameDumpEvent(std::move(text));
}

void Rasterizer::FrameDumpWriteEvents(const std::filesystem::path& dir) {
    std::scoped_lock lock{frame_dump_mutex};
    if (std::FILE* file = std::fopen((dir / "events.txt").string().c_str(), "w")) {
        for (const auto& [text, detail, draws] : frame_dump_events) {
            if (draws != 0) {
                std::fprintf(file, "%s | %u draws | %s\n", text.c_str(), draws, detail.c_str());
            } else {
                std::fprintf(file, "%s\n", text.c_str());
            }
        }
        std::fclose(file);
    }
    frame_dump_events.clear();
}

void Rasterizer::FrameDumpWriteOld() {
    frame_dump_write_old = false;
    u32 index = 0, written = 0;
    for (const auto& [id, uid] : frame_dump_old) {
        const u32 order = index++;
        if (!texture_cache.IsImageAlive(id, uid)) {
            continue;
        }
        const auto& info = texture_cache.GetImage(id).info;
        if (info.num_samples > 1) {
            LOG_WARNING(Render_Vulkan, "NAT-116: {:#x} {}x{} has {} samples, not dumped",
                        info.guest_address, info.size.width, info.size.height, info.num_samples);
            continue;
        }
        if (texture_cache.DumpImage(
                id, frame_dump_dir / fmt::format("{:03}_{:x}_{}x{}_{}{}.bin", order,
                                                 info.guest_address, info.size.width,
                                                 info.size.height, u32(info.pixel_format),
                                                 info.props.is_depth ? "_depth" : ""))) {
            ++written;
        }
    }
    LOG_WARNING(Render_Vulkan, "NAT-116: {} of {} render targets of the frame dumped to {}",
                written, frame_dump_old.size(), frame_dump_dir.string());
    frame_dump_old.clear();
    // NAT-123: the GPU-written textures the frame sampled, and the image each one read.
    std::string list;
    u32 textures = 0;
    for (const auto& [key, id, uid] : frame_dump_textures) {
        if (!texture_cache.IsImageAlive(id, uid)) {
            continue;
        }
        const auto& info = texture_cache.GetImage(id).info;
        list += fmt::format("{} -> image {:#x} {}x{} mips {} layers {} {}{}\n", key,
                            info.guest_address, info.size.width, info.size.height,
                            info.resources.levels, info.resources.layers,
                            vk::to_string(info.pixel_format),
                            info.num_samples > 1 ? " (multisampled, not dumped)" : "");
        if (info.num_samples == 1 &&
            u64(info.size.width) * info.size.height <= 4096ull * 4096 &&
            texture_cache.DumpImage(id, frame_dump_dir / fmt::format("tex_{}{}.bin", key,
                                                                     info.props.is_depth
                                                                         ? "_depth"
                                                                         : ""))) {
            ++textures;
        }
    }
    if (std::FILE* file = std::fopen((frame_dump_dir / "textures.txt").string().c_str(), "w")) {
        std::fwrite(list.data(), 1, list.size(), file);
        std::fclose(file);
    }
    LOG_WARNING(Render_Vulkan, "NAT-123: {} GPU-written textures of the frame dumped", textures);
    frame_dump_textures.clear();
}

std::string Rasterizer::FrameDumpTextureKey(const VideoCore::ImageInfo& info) {
    return fmt::format("{:x}_{}x{}_m{}_l{}_f{}", info.guest_address, info.size.width,
                       info.size.height, info.resources.levels, info.resources.layers,
                       u32(info.pixel_format));
}

#ifdef RYTY_GPU
bool Rasterizer::NativeInlineWrite(VAddr address, std::span<const u8> data) {
    return native && native->InlineWrite(address, data);
}
#endif

void Rasterizer::InlineDeferredWrite(VAddr address, std::span<const u8> data) {
#ifdef RYTY_GPU
    if (native) {
        return; // NAT-100: native mode writes WRITE_DATA on the GPU in order (NativeInlineWrite)
    }
#endif
    // FIX-019: hardware performs WRITE_DATA in command order, so work after it reads the new
    // value. A deferred write reaches guest memory only after the GPU completes, through the
    // backing (no write fault, so nothing uploads it), and the GPU copy kept the old value for
    // every later draw and dispatch: counters and tables reset that way stayed stale.
    // -DisablePerf 30 restores the old behavior.
    static const bool enabled = Common::PerfFeatureEnabled(30);
    constexpr u64 TrackedAddressLimit = 1ULL << VideoCore::MemoryTracker::MAX_CPU_PAGE_BITS;
    if (!enabled || data.empty() || address % 4 != 0 || data.size() % 4 != 0 ||
        data.size() > 65536 || address + data.size() > TrackedAddressLimit ||
        !IsMapped(address, data.size())) {
        return;
    }
    static std::atomic<u64> count{};
    if (const u64 n = ++count; n <= 40 || n % 1000 == 0) {
        u32 first = 0;
        std::memcpy(&first, data.data(), sizeof(first));
        LOG_WARNING(Render_Vulkan,
                    "FIX-019: deferred WRITE_DATA {} at {:#x}+{:#x} (first dword {:#x}) written to "
                    "the GPU copy in order; GPU-modified={}",
                    n, address, data.size(), first,
                    buffer_cache.IsRegionGpuModified(address, data.size()));
    }
    buffer_cache.InlineGuestWrite(address, data);
}

std::optional<u32> Rasterizer::PendingFenceDword(VAddr address) {
    // PERF-014: GPU work that waits on a fence the command thread has already processed runs
    // after the fenced work in the one Vulkan queue, so a GPU-side wait that the fence's value
    // satisfies needs no CPU-side wait for GPU completion. This is the order the emulator
    // already relies on when fences are not deferred.
    std::scoped_lock lk{deferred_fences_mutex};
    for (const auto& [fence_address, bytes] : pending_fence_bytes) {
        if (address >= fence_address && address + sizeof(u32) <= fence_address + bytes.size()) {
            u32 value;
            std::memcpy(&value, bytes.data() + (address - fence_address), sizeof(u32));
            return value;
        }
    }
    return std::nullopt;
}

std::atomic<u64> g_defer_readbacks{};
std::atomic<u64> g_defer_address{};
std::atomic<u64> g_defer_order{};
std::atomic<u64> g_defer_unflushed{};

void Rasterizer::RecordDeferredFenceLatency(std::chrono::steady_clock::time_point deferred_at) {
    // DIAG-015: how long the guest waits on a deferred fence, reported every 2 seconds.
    const auto now = std::chrono::steady_clock::now();
    const u64 latency_us =
        std::chrono::duration_cast<std::chrono::microseconds>(now - deferred_at).count();
    std::scoped_lock lk{deferred_fences_mutex};
    auto& stats = deferred_fence_stats;
    if (stats.count == 0 && stats.window_start == std::chrono::steady_clock::time_point{}) {
        stats.window_start = now;
        stats.window_frame = u64(DebugState.GetFrameNum());
    }
    ++stats.count;
    stats.total_us += latency_us;
    stats.max_us = std::max(stats.max_us, latency_us);
    if (now - stats.window_start >= std::chrono::seconds{2}) {
        const u64 frames = u64(DebugState.GetFrameNum()) - stats.window_frame;
        LOG_WARNING(
            Render_Vulkan,
            "Deferred fences: {} in {} frames, latency avg {:.2f} ms max {:.2f} ms; DIAG-068: "
            "deferred for image readbacks {}, same address {}, behind an earlier one {}; FIX-070: "
            "{} small readbacks without a fence; D5: {} without a submit of their own",
            stats.count, frames, stats.total_us / 1000.0 / stats.count, stats.max_us / 1000.0,
            g_defer_readbacks.exchange(0), g_defer_address.exchange(0), g_defer_order.exchange(0),
            background_readbacks.exchange(0), g_defer_unflushed.exchange(0));
        stats = {};
        stats.window_start = now;
        stats.window_frame = u64(DebugState.GetFrameNum());
    }
}

void Rasterizer::SubmitChunkIfNeeded() {
    // PERF-003: with readback on, the guest waits for GPU completion at fences. Submitting in
    // chunks lets the GPU run the frame while it is still being recorded.
    // PERF-007: chunks only help while a fence will wait for GPU completion, that is while
    // readbacks or deferred fences are pending. Otherwise each extra submit is pure overhead
    // (about 14% of the GPU thread in GT Sport), so batch normally.
    static constexpr u32 DrawsPerSubmit = 128;
    // PERF-059 (experiment, SHADGT_CHUNK_DRAWS=N): submit every N draws whatever is pending. GT
    // Sport waits for each submission's GPU completion before it submits the next (DIAG-051), so
    // GPU work that starts only after the whole submission is recorded adds to every frame.
    static const u32 always_chunk = [] {
        const char* env = std::getenv("SHADGT_CHUNK_DRAWS");
        return env && *env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    if (always_chunk != 0) {
        if (++draws_since_submit >= always_chunk) {
            draws_since_submit = 0;
            scheduler.Flush();
        }
        return;
    }
    if (!texture_cache.ReadbackLinearImages()) {
        return;
    }
    if (++draws_since_submit < DrawsPerSubmit) {
        return;
    }
    // PERF-016: a CPU access that drains the GPU waits for everything recorded so far. While
    // such drains keep happening (GT Sport: once per race frame), submitting in chunks lets the
    // GPU run ahead so less is left to wait for.
    static const bool chunk_for_drains = Common::PerfFeatureEnabled(16);
    if (deferred_fences.load() == 0 && !texture_cache.HasPendingReadbacks() &&
        !(chunk_for_drains && buffer_cache.DrainedRecently())) {
        return;
    }
    draws_since_submit = 0;
    scheduler.Flush();
}

void Rasterizer::FlushForDeferredFences() {
    // A GPU-side wait may depend on a deferred fence (directly or through the guest CPU). Submit
    // any recorded work so the deferred operations can complete instead of deadlocking.
    if (deferred_fences.load() != 0) {
        scheduler.Flush();
    }
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    {
        const int special = IsComputeImageCopy(pipeline)    ? 1
                            : IsComputeMetaClear(pipeline)  ? 2
                            : IsComputeImageClear(pipeline) ? 3
                                                            : 0;
        if (special != 0) {
            // DIAG-122 (SHADGT_DIAG_SPECIAL=1): which dispatches the renderer replaces.
            static const bool diag = std::getenv("SHADGT_DIAG_SPECIAL") != nullptr;
            static std::atomic<u32> logged{};
            if (diag && logged++ < 400) {
                const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
                std::string bufs;
                for (const auto& desc : info.buffers) {
                    const auto sharp = desc.GetSharp(info);
                    bufs += fmt::format(" {:#x}+{:#x}{}", sharp.base_address, sharp.GetSize(),
                                        desc.is_written ? "w" : "");
                }
                LOG_WARNING(Render_Vulkan, "DIAG-122: cs {:#x} {}{}", info.pgm_hash,
                            special == 1 ? "image copy" : special == 2 ? "meta clear" : "image clear",
                            bufs);
            }
            return false;
        }
    }

#ifdef RYTY_GPU
    if (t_draw_packet != nullptr && Native::ShadowEnabled()) {
        // NAT-032: the sharps this draw reads here, against what its job read on its thread.
        Native::ShadowCheckSharps(t_draw_packet, SharpsHash(pipeline->GetStages()));
    }
#endif
    ctx->set_write_index = 0;
    ctx->set_writes.clear();
    ctx->buffer_infos.clear();
    ctx->image_infos.clear();
    ctx->bound_textures.clear();
    ctx->image_binding_base = 0; // PERF-121

    if (buffer_plan && !buffer_plan->uploads.empty()) {
        // PERF-067: the uploads the buffer stage prepared for this draw's buffers.
        buffer_cache.RecordAheadUploads(buffer_plan->uploads);
    }

    bool uses_dma = false;

    // Preserve the compiled stage binding numbers while resolving every buffer alias before
    // texture descriptors and attachment layouts are finalized.
    Shader::Backend::Bindings binding{};
    std::array<StageResourceBindings, Shader::MaxStageTypes> stage_bindings{};
    const auto stages = pipeline->GetStages();
    ctx->push_data = MakeUserData(liverpool->DrawRegs());
    for (u32 i = 0; i < stages.size(); ++i) {
        const auto* stage = stages[i];
        if (!stage) {
            continue;
        }
        ctx->set_writes.resize(ctx->set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage_bindings[i] = PlanStageResourceBindings(*stage, binding);
        auto buffer_binding = stage_bindings[i].buffers;
        Common::PhaseTimer t{Common::Phase::Buffers};
        BindBuffers(*stage, buffer_binding, ctx->push_data);
        binding = stage_bindings[i].next;
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
    }

    for (u32 i = 0; i < stages.size(); ++i) {
        if (const auto* stage = stages[i]) {
            auto texture_binding = stage_bindings[i].textures;
            Common::PhaseTimer t{Common::Phase::Textures};
            BindTextures(*stage, texture_binding);
        }
    }
    const u32 num_color_targets =
        pipeline->IsCompute()
            ? 0u
            : std::bit_width(
                  static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey().mrt_mask);
    {
        Common::PhaseTimer t{Common::Phase::TextureRebind};
        RebindTextures(pipeline->IsCompute(), num_color_targets);
    }
    if (pipeline->IsCompute()) {
        FinalizeTextureLayouts();
    }

    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    Common::PhaseTimer vtx_timer{Common::Phase::VtxBind}; // DIAG-119
    const auto& regs = liverpool->DrawRegs();
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    // DIAG-081 (SHADGT_TRACE_GS_INPUTS=N): the vertex inputs of the first N geometry-shader
    // draws after 150 s: each V# and the first floats at its address.
    if (static const int trace_gs = [] {
            const char* env = std::getenv("SHADGT_TRACE_GS_INPUTS");
            return env ? std::atoi(env) : 0;
        }();
        trace_gs > 0 && pipeline->GetStages()[u32(Shader::SwStage::Geometry)] != nullptr) {
        static const auto start = std::chrono::steady_clock::now();
        static std::atomic<int> traced{};
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds{150} &&
            traced < trace_gs) {
            ++traced;
            std::string text;
            for (const auto& b : guest_buffers) {
                std::array<float, 8> f{};
                Common::CopyGuest(f.data(), reinterpret_cast<const void*>(b.base_address),
                                  sizeof(f));
                text += fmt::format(" [{:#x} size {:#x} stride {} fmt {}/{}: {} {} {} {} {} {} {} {}]",
                                    u64(b.base_address), b.GetSize(), b.GetStride(),
                                    u32(b.GetDataFmt()), u32(b.GetNumberFmt()), f[0], f[1],
                                    f[2], f[3], f[4], f[5], f[6], f[7]);
            }
            LOG_WARNING(Render_Vulkan, "DIAG-081: gs draw es {:#x} gs {:#x} prim {} indices {} "
                        "inst {}: {} buffers{}",
                        pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                        pipeline->GetStage(Shader::SwStage::Geometry).pgm_hash,
                        u32(regs.primitive_type), regs.num_indices,
                        regs.num_instances.NumInstances(), guest_buffers.size(), text);
        }
    }

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(bindings, attributes);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

#ifdef RYTY_GPU
    // FIX-053: a V# whose start is not mapped is garbage (the game rewrote its descriptors
    // before this draw was recorded). Binding it made the buffer cache try to hold hundreds of
    // megabytes and the GPU ran out of memory; such a binding is left empty instead.
    for (size_t index = 0; index < guest_buffers.size(); ++index) {
        auto& buffer = guest_buffers[index];
        if (buffer.base_address != 0 && buffer.GetSize() > 0 &&
            memory->MappedAreaEnd(buffer.base_address, buffer.base_address + 1) == 0) {
            static std::atomic<u32> logged{};
            if (const u32 n = logged++; n < 10 || n % 500 == 0) {
                const auto& draw_regs = liverpool->DrawRegs();
                LOG_WARNING(Render_Vulkan,
                            "FIX-053: vertex buffer at unmapped {:#x}+{:#x} not bound ({} times; "
                            "draw of {} indices x {} instances, vs {:#x})",
                            u64{buffer.base_address}, buffer.GetSize(), n + 1,
                            draw_regs.num_indices, draw_regs.num_instances.num_instances,
                            pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash);
            }
            // DIAG-071: where the V# was read from, what is there now, and the renderer's own
            // writes to it (garbage V#s came back in race summaries with a faster recorder).
            static std::atomic<u32> traced{};
            const auto& attribs = pipeline->GetFetchShader().attributes;
            const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
            if (index < attribs.size() && attribs[index].sgpr_base != Shader::IR::NumScalarRegs &&
                traced++ < 30) {
                const auto& attrib = attribs[index];
                u64 table = 0;
                std::memcpy(&table, &vs.user_data[attrib.sgpr_base], sizeof(table));
                table &= 0xFFFFFFFFFFFFULL;
                const VAddr source = table + attrib.dword_offset * 4;
                std::array<u32, 4> now{};
                Common::CopyGuest(now.data(), reinterpret_cast<const void*>(source), sizeof(now));
                std::array<u32, 4> read{};
                std::memcpy(read.data(), &buffer, sizeof(read));
                LOG_WARNING(Render_Vulkan,
                            "DIAG-071: garbage V# {:08x} {:08x} {:08x} {:08x} read from {:#x} "
                            "(table {:#x} + {:#x}); now {:08x} {:08x} {:08x} {:08x}; renderer "
                            "writes near it:{}; range: {}; readbacks over it:{}; {}",
                            read[0], read[1], read[2], read[3], source, table,
                            attrib.dword_offset * 4, now[0], now[1], now[2], now[3],
                            memory->DescribeBackingWrites(source, 4096),
                            buffer_cache.DescribeRange(source, 16),
                            buffer_cache.DescribeRecentReadbacks(source, 16),
                            buffer_cache.DescribeReadbackPage(source));
            }
            buffer.base_address = 0;
        }
    }
#endif
    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            // Bound each descriptor before merging, so open-ended ranges cannot bridge unmapped
            // guest memory and make another vertex buffer appear resident in the first buffer.
            const u64 size = memory->ClampRangeSize(buffer.base_address, buffer.GetSize());
            ranges.emplace_back(buffer.base_address, buffer.base_address + size);
        }
    }

    // Merge connecting ranges together
    VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    const bool quad_list = regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        if (quad_list) {
            buffer_cache.RefreshReadPages(range.base_address, size,
                                          pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                                          true);
        }
        // D6: vertices read in place from imported guest memory.
        // Off by default: vertex fetch over PCIe made the GPU the limit (90% busy, 27 fps).
        // SHADGT_DIRECT_VB=<max KB>: only ranges up to that size (small, often rewritten ones).
        static const u64 direct_vertex_limit = [] {
            const char* env = std::getenv("SHADGT_DIRECT_VB");
            return env && *env ? std::strtoull(env, nullptr, 10) * 1024 : u64{0};
        }();
        if (const auto direct = size <= direct_vertex_limit
                                    ? DirectReadBuffer(range.base_address, size)
                                    : std::nullopt) {
            std::tie(range.buffer, range.offset) = *direct;
        } else
        // MT-004: obtained ahead by the buffer stage when it planned this draw's vertex buffers.
        if (const auto planned =
                buffer_plan && buffer_plan->vertices
                    ? TakePlannedBuffer(range.base_address, static_cast<u32>(size), false, false)
                    : std::nullopt) {
            std::tie(range.buffer, range.offset) = *planned;
        } else {
            Common::PhaseTimer obtain_timer{Common::Phase::BufObtain}; // DIAG-119
            std::tie(range.buffer, range.offset) =
                buffer_cache.ObtainBuffer(range.base_address, size, false);
        }
        ctx->needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
        // FIX-016: record the vertex fetch as a read of the range, so a later upload into it
        // (GT Sport rewrites grass vertex buffers between draws) waits for this draw. Without
        // it the copy can overwrite the vertices while this draw still reads them, mixing
        // old and new data (the suspected cause of stretched NÃ¼rburgring grass and terrain).
        ctx->bound_buffers.emplace_back(range.buffer, range.offset, size, false);
    }

    // Bind vertex buffers
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        u64 host_size{};
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            const u64 offset =
                host_buffer_info->offset + buffer.base_address - host_buffer_info->base_address;
            host_offsets.push_back(offset);
            const u64 mapped_size = memory->ClampRangeSize(buffer.base_address, buffer.GetSize());
            host_size = std::min(mapped_size, host_buffer_info->buffer->SizeBytes() - offset);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(host_size);
        host_strides.push_back(buffer.GetStride());
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void Rasterizer::BindIndexBuffer(u32 index_offset) {
    Common::PhaseTimer idx_timer{Common::Phase::IdxBind}; // DIAG-119
    const auto& regs = liverpool->DrawRegs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    // MT-005: obtained ahead by the buffer stage when it planned this draw's index buffer.
    const auto planned = buffer_plan && buffer_plan->index
                             ? TakePlannedBuffer(index_address, index_buffer_size, false, false)
                             : std::nullopt;
    static const bool direct_indices = [] { // D6: SHADGT_DIRECT_IB=0 turns it off
        const char* env = std::getenv("SHADGT_DIRECT_IB");
        return !env || env[0] != '0';
    }();
    const auto direct = planned || !direct_indices
                            ? std::nullopt
                            : DirectReadBuffer(index_address, index_buffer_size);
    const auto [buffer, offset] =
        planned  ? *planned
        : direct ? *direct
                 : buffer_cache.ObtainBuffer(index_address, index_buffer_size, false);
    ctx->needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
    ctx->bound_buffers.emplace_back(buffer, offset, index_buffer_size, false); // FIX-016
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindIndexBuffer(buffer->Handle(), offset, index_type);
}

bool Rasterizer::CanDrawQuadListAsTriangles(bool is_indexed, u32 index_offset) {
    const auto& regs = liverpool->DrawRegs();
    if (u64(regs.num_indices) / 4 * 6 * sizeof(u32) > MaxQuadListIndexBytes) {
        static const bool logged = [&] {
            LOG_WARNING(Render_Vulkan, "Quad list of {} vertices drawn through tessellation",
                        regs.num_indices);
            return true;
        }();
        return false;
    }
    if (!is_indexed) {
        return true;
    }
    // The indices are read from guest memory here, so they must not be waiting in a GPU write.
    const u32 index_size = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2 : 4;
    const VAddr address = regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
    if (buffer_cache.IsRegionGpuModified(address, u64(regs.num_indices) * index_size)) {
        static const bool logged = [&] {
            LOG_WARNING(Render_Vulkan,
                        "Quad list with GPU-written indices drawn through tessellation");
            return true;
        }();
        return false;
    }
    return true;
}

u32 Rasterizer::BindQuadListIndices(bool is_indexed, u32 index_offset) {
    const auto& regs = liverpool->DrawRegs();
    const auto& corners = regs.polygon_control.provoking_vtx_last == AmdGpu::ProvokingVtxLast::Last
                              ? QuadCornersLastVertex
                              : QuadCornersFirstVertex;
    const u32 count = regs.num_indices;
    const u32 max_indices = count / 4 * 6;
    if (max_indices == 0) {
        return 0;
    }
    auto& stream = buffer_cache.GetStreamBuffer();
    const auto cmdbuf = scheduler.CommandBuffer();
    if (!is_indexed) {
        const auto [data, offset] = stream.Map(u64(max_indices) * sizeof(u32), sizeof(u32));
        auto* out = reinterpret_cast<u32*>(data);
        for (u32 quad = 0; quad + 4 <= count; quad += 4) {
            for (const u32 corner : corners) {
                *out++ = quad + corner;
            }
        }
        stream.Commit();
        cmdbuf.bindIndexBuffer(stream.Handle(), offset, vk::IndexType::eUint32);
        return max_indices;
    }
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr address = regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
    const bool restart = (regs.enable_primitive_restart & 1) != 0;
    const auto [data, offset] = stream.Map(u64(max_indices) * index_size, sizeof(u32));
    u32 written;
    if (is_index16) {
        written = ExpandQuadListIndices<u16>(
            reinterpret_cast<const u16*>(address), count,
            restart ? std::optional<u16>{u16(regs.primitive_restart_index)} : std::nullopt, corners,
            reinterpret_cast<u16*>(data));
    } else {
        written = ExpandQuadListIndices<u32>(
            reinterpret_cast<const u32*>(address), count,
            restart ? std::optional<u32>{regs.primitive_restart_index} : std::nullopt, corners,
            reinterpret_cast<u32*>(data));
    }
    stream.Commit();
    cmdbuf.bindIndexBuffer(stream.Handle(), offset,
                           is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);
    return written;
}

void Rasterizer::ResetBindings(bool is_compute) {
    for (auto& image_id : ctx->bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, is_written, guest_address] : ctx->bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        const auto write_flag =
            is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone;
        runtime.AccessBuffer(buffer, offset, size, dst_stage,
                             vk::AccessFlagBits2::eShaderRead | write_flag);
        if (is_written && guest_address != 0) {
            // A texture or attachment refresh during binding may have consumed the earlier
            // invalidation. The scheduled shader write makes these aliases stale again.
            texture_cache.InvalidateMemoryFromGPU(guest_address, size);
        }
    }
    {
        // DIAG-057: the resources this draw uses, for the batch measurement.
        boost::container::small_vector<std::tuple<vk::Buffer, u64, u64, bool>, 32> buffers;
        for (const auto& bound : ctx->bound_buffers) {
            buffers.emplace_back(bound.buffer->Handle(), bound.offset, bound.size,
                                 bound.is_written);
        }
        boost::container::small_vector<vk::Image, 32> images;
        for (const auto id : ctx->bound_images) {
            images.push_back(texture_cache.GetImage(id).GetImage());
        }
        // Render targets are left out: the color slots past the draw's targets keep ids of
        // images that may be gone.
        runtime.DiagBatchDraw(buffers, images);
        runtime.NoteSessionDraw(buffers); // PERF-066
    }
    ctx->bound_images.clear();
    ctx->bound_buffers.clear();
    ctx->diag_empty_bindings.clear();
    ctx->diag_storage_images.clear();
    ctx->needs_barrier = false;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written &&
                (t_special_dry_run ? texture_cache.IsMeta(address).has_value()
                                   : texture_cache.ClearMeta(address))) {
                // FIX-103: GT Sport reuses memory between frames' passes, so a CMASK address of
                // one pass can be a color picture's memory in another. When the written range
                // holds a picture, the dispatch is the game clearing that picture: run it.
                // Skipping it left a 16 KB block of an effect picture uncleared, and its stale
                // invalid numbers became the pink and black rectangles of dust, smoke and rain.
                const auto sharp = desc.GetSharp(info);
                if (buffer_cache.HasGpuImageAlias(address, std::max<u64>(sharp.GetSize(), 4))) {
                    if (t_special_dry_run) {
                        return false;
                    }
                    if (FrameDumpActive()) {
                        FrameDumpEvent(fmt::format("dispatch: metadata clear {:#x} (holds a "
                                                   "picture, run)", address));
                    }
                    if (DiagHistoryEnabled()) {
                        static std::atomic<u32> logged{};
                        if (logged++ < 3000) {
                            LOG_WARNING(Render_Vulkan,
                                        "DIAG-105: compute clear {:#x}+{:#x} kept (holds a "
                                        "picture)", address, sharp.GetSize());
                        }
                    }
                    return false;
                }
                if (DiagHistoryEnabled()) {
                    static std::atomic<u32> logged{};
                    if (logged++ < 3000) {
                        LOG_WARNING(Render_Vulkan, "DIAG-105: compute meta clear {:#x}", address);
                    }
                }
                if (FrameDumpActive()) {
                    FrameDumpEvent(fmt::format("dispatch: metadata clear {:#x}", address));
                }
                // FIX-112: the metadata is marked cleared, and the dispatch still runs. GT Sport
                // fills a 20 KB block (0x10057f8000) with this shader every frame, and that
                // memory was a CMASK of another pass; skipped, the block kept old values and the
                // top bar's frosted-glass blur came out black (always, or now and then).
                // -DisablePerf 135 skips the dispatch as before.
                static const bool run_clears = Common::PerfFeatureEnabled(135);
                if (run_clears) {
                    return false;
                }
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->DrawCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    if (t_special_dry_run) {
        return true; // FIX-111
    }
    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->DrawCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    if (t_special_dry_run) {
        return true; // FIX-111
    }
    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

// DIAG-017: bindings on the page that precedes every device loss in GT Sport.
static bool ShouldTraceCrashPage(VAddr address) {
    static constexpr VAddr TracedBegin = 0x3f80000000;
    static constexpr VAddr TracedEnd = TracedBegin + 0x10000;
    if (address < TracedBegin || address >= TracedEnd) {
        return false;
    }
    static std::atomic<u64> count{};
    const u64 n = ++count;
    return n <= 64 || n % 1000 == 0;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const u64 alignment = instance.StorageMinAlignment();
    const bool watched = IsWatchedShader(stage.pgm_hash);
    static std::atomic<u64> watched_binds{};
    const u64 watched_bind = watched ? ++watched_binds : 0;
    const bool log_watched = watched && (watched_bind <= 12 || watched_bind % 300 == 0);
    // DIAG-032: once, well into the session, dump the GPU copy of every buffer of a grass vertex
    // dispatch (cs 0xab6a2d10), then the vertex ranges of the grass draws that follow.
    static std::atomic<u32> grass_dispatches{};
    static const u32 dump_at = [] {
        const char* env = std::getenv("SHADGT_DUMP_GRASS_AT");
        return env && *env ? u32(std::stoul(env)) : 0u;
    }();
    if (stage.pgm_hash == 0xab6a2d10 && ++grass_dispatches == dump_at) {
        const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "grass-dump";
        std::filesystem::create_directories(dir);
        std::ofstream index{dir / "index.txt"};
        u32 i = 0;
        for (const auto& desc : stage.buffers) {
            const u32 index_number = i++;
            if (desc.IsSpecial()) {
                continue;
            }
            const auto vsharp = desc.GetSharp(stage);
            const u64 size = vsharp.base_address
                                 ? memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize())
                                 : 0;
            index << fmt::format("buffer {} {} {:#x}+{:#x} stride {} records {:#x}\n", index_number,
                                 desc.is_written ? "write" : "read", u64(vsharp.base_address), size,
                                 vsharp.GetStride(), vsharp.num_records);
            if (size != 0 && size <= 64_MB) {
                buffer_cache.DumpRange(vsharp.base_address, size,
                                       dir / fmt::format("dispatch_buffer{}", index_number));
            }
        }
        buffer_cache.grass_dump_dir = dir;
        buffer_cache.grass_vertex_dumps_left = 5;
        LOG_WARNING(Render_Vulkan, "DIAG-032: dumped the buffers of grass dispatch {} to {}",
                    dump_at, dir.string());
    }
    u32 buffer_index = 0;
    // D3: guest addresses of the buffers this stage reads through the direct-memory page table
    // (base low, base high, size, valid per buffer), written into its table at the end.
    auto* direct = stage.uses_direct ? buffer_cache.GetDirectMemory() : nullptr;
    static thread_local std::vector<std::array<u32, 4>> direct_entries;
    if (direct) {
        direct_entries.assign(stage.buffers.size(), {});
    }
    for (const auto& desc : stage.buffers) {
        ++buffer_index;
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                ctx->buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                ctx->needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                ctx->bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                RefreshGpuWrittenConstants(stage, vk_buffer, offset);
                ctx->buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->DrawRegs().clipper_control.user_clip_plane_enable == 0) {
                    ctx->buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->DrawRegs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    ctx->buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                ctx->buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::DirectTable) {
                // DIAG-133 (SHADGT_DIAG_SLOW_US=<us>): busy-waits that long per direct table, to
                // tell timing races from wrong reads.
                static const s64 slow_us = [] {
                    const char* env = std::getenv("SHADGT_DIAG_SLOW_US");
                    return env && *env ? std::atoll(env) : s64{0};
                }();
                if (slow_us > 0) {
                    const auto until =
                        std::chrono::steady_clock::now() + std::chrono::microseconds(slow_us);
                    while (std::chrono::steady_clock::now() < until) {
                    }
                }
                // D3: page table L1, zero page, then the entries.
                std::vector<u32> table(4 + stage.buffers.size() * 4, 0u);
                if (direct) {
                    const u64 l1 = direct->L1Address();
                    const u64 zero = direct->ZeroPageAddress();
                    table[0] = u32(l1);
                    table[1] = u32(l1 >> 32);
                    table[2] = u32(zero);
                    table[3] = u32(zero >> 32);
                    for (size_t i = 0; i < stage.buffers.size(); ++i) {
                        std::memcpy(&table[4 + i * 4], direct_entries[i].data(), 16);
                    }
                }
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 table_size = static_cast<u32>(table.size() * sizeof(u32));
                const u64 offset = vk_buffer.Copy(table.data(), table_size, alignment);
                ctx->buffer_infos.emplace_back(vk_buffer.Handle(), offset, table_size);
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                ctx->buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = liverpool->DrawCsRegs();
                const u64 lds_size = u64(cs_program.SharedMemSize()) * cs_program.NumWorkgroups();
                // GCN LDS is undefined at workgroup launch, so only reserve a GPU-only region.
                const auto offset = lds_buffer.Reserve(lds_size, alignment);
                ASSERT_MSG(offset, "Emulated shared memory size {:#x} exceeds the stream buffer",
                           lds_size);
                ctx->buffer_infos.emplace_back(lds_buffer.Handle(), *offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            // FIX-031: a V# that ends past the GPU's 40-bit address space is not a buffer. GT
            // Sport's dealership shader 0xae32f77f has V#s made of float constants (base and
            // records 0x3f800000, terabytes long); clamped to guest memory they became 1 GB
            // buffers over module code, uploading GBs until the GPU ran out of memory (and the
            // tracking froze libc). Bind no buffer, so reads give 0 and writes are dropped.
            // -DisablePerf 39 binds the clamped range.
            static const bool null_impossible = Common::PerfFeatureEnabled(39);
            constexpr u64 GpuAddressLimit = 1ULL << 40;
            // FIX-036: also a V# of 1 GB or more whose base is outside guest memory (the same
            // shader, base 0x92 at the purchase dialog); ClampRangeSize asserted on it.
            const bool impossible =
                null_impossible && vsharp.num_records != UINT32_MAX &&
                (vsharp.GetSize() >= GpuAddressLimit ||
                 u64(vsharp.base_address) + vsharp.GetSize() > GpuAddressLimit ||
                 (vsharp.GetSize() >= 1_GB && vsharp.base_address != 0 &&
                  !memory->IsValidMapping(vsharp.base_address)));
            if (impossible) {
                static std::atomic<u32> logged{};
                if (const u32 n = ++logged; n <= 20 || n % 1000 == 0) {
                    // DIAG-039: where each V# dword came from (user data index, its value,
                    // and the guest address it was loaded from).
                    std::string source;
                    const auto& fetch = desc.sharp_fetch;
                    const bool single =
                        fetch.summary == std::remove_cvref_t<decltype(fetch)>::Summary::SingleLoad;
                    for (u32 i = 0; i < 4; ++i) {
                        if (!single && !((fetch.load_mask >> i) & 1)) {
                            source += fmt::format(" imm={:08x}", fetch.immediates[i]);
                            continue;
                        }
                        const u32 off = single ? u32(fetch.offsets[0]) + i : u32(fetch.offsets[i]);
                        const u64 src =
                            off < stage.flattened_ud_src.size() ? stage.flattened_ud_src[off] : 0;
                        source += fmt::format(
                            " [{}]={:08x}{}", off,
                            off < stage.flattened_ud_buf.size() ? stage.flattened_ud_buf[off] : 0,
                            src ? fmt::format("@{:#x}", src) : std::string(" ud"));
                    }
                    LOG_WARNING(Render,
                                "FIX-031: buffer {} (binding {}) for stage {:#x} bound empty: "
                                "base={:#x}, stride={}, records={:#x} ends past the GPU address "
                                "space or starts outside guest memory; V# from{}",
                                n, ctx->buffer_infos.size(), stage.pgm_hash, u64(vsharp.base_address),
                                vsharp.GetStride(), vsharp.num_records, source);
                }
            }
            // FIX-046: a V# outside the guest address space would be tracked like the T# at
            // 0xa000.
            const bool unmapped = !impossible && vsharp.base_address != 0 &&
                                  vsharp.GetSize() != 0 && !IsMappedStart(vsharp.base_address);
            if (unmapped) {
                static std::atomic<u32> rejected{};
                if (const u32 n = ++rejected; n <= 20 || n % 1000 == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "FIX-046: buffer {} of stage {:#x} bound empty: {:#x}+{:#x} is "
                                "outside the guest address space",
                                n, stage.pgm_hash, u64(vsharp.base_address), vsharp.GetSize());
                }
            }
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0 || impossible || unmapped) {
                // DIAG-041: empty buffer bindings of the draw.
                if (DiagHistoryEnabled()) {
                    ctx->diag_empty_bindings += fmt::format(
                        " | EMPTY buf {} of {:#x}: base={:#x} stride={} records={:#x}",
                        ctx->buffer_infos.size(), stage.pgm_hash, u64(vsharp.base_address),
                        vsharp.GetStride(), vsharp.num_records);
                }
                ctx->buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                // D3: a read-only buffer the GPU reads in place: no obtain, a null binding.
                // Ranges with GPU-written bytes only the arena holds take the usual path.
                static const bool direct_reads = [] { // D3 bisect: SHADGT_DIRECT_READS=0
                    const char* env = std::getenv("SHADGT_DIRECT_READS");
                    return !env || env[0] != '0';
                }();
                const bool read_in_place = [&] {
                    // DIAG-128 bisect: SHADGT_DIRECT_READS_STAGES=<hex mask of SwStage bits>,
                    // SHADGT_DIRECT_READS_MAX=<bytes> (larger ranges take the usual path),
                    // SHADGT_DIRECT_READS_MIN=<bytes> (smaller ranges take the usual path).
                    static const u32 stage_mask = [] {
                        const char* env = std::getenv("SHADGT_DIRECT_READS_STAGES");
                        return env && *env ? u32(std::strtoul(env, nullptr, 16)) : ~0u;
                    }();
                    static const u64 max_size = [] {
                        const char* env = std::getenv("SHADGT_DIRECT_READS_MAX");
                        return env && *env ? std::strtoull(env, nullptr, 10) : ~u64{0};
                    }();
                    static const u64 min_size = [] {
                        const char* env = std::getenv("SHADGT_DIRECT_READS_MIN");
                        return env && *env ? std::strtoull(env, nullptr, 10) : u64{0};
                    }();
                    if (!direct || !direct_reads || !desc.IsDirectEligible() ||
                        size >= (u64{1} << 32) || size > max_size || size < min_size ||
                        !((stage_mask >> u32(stage.sw_stage)) & 1)) {
                        return false;
                    }
                    Common::PhaseTimer direct_timer{Common::Phase::BufDirect}; // DIAG-119
                    return !direct->IsVolatile(vsharp.base_address, size) &&
                           direct->EnsureImported(vsharp.base_address, size) &&
                           !buffer_cache.IsRegionGpuModified(vsharp.base_address, size) &&
                           !WrittenRecently(vsharp.base_address, size) && // FIX-123
                           DirectReadsHere() && // FIX-124
                           !buffer_cache.DirectReadHitsImage(vsharp.base_address, size); // FIX-122
                }();
                if (read_in_place) {
                    NoteDirectRead(); // FIX-125
                    if (DirectHazardDiag()) {
                        CheckInPlaceHazard("buffer", vsharp.base_address, size, stage.pgm_hash);
                    }
                    // DIAG-131 (SHADGT_DIAG_DIRECT_CHANGE=1): bytes read in place that the CPU
                    // changed between recording (what a copy would have used) and the GPU's
                    // completion.
                    static const bool change_diag = [] {
                        const char* env = std::getenv("SHADGT_DIAG_DIRECT_CHANGE");
                        return env && env[0] == '1';
                    }();
                    if (change_diag && size <= 64_KB) {
                        const VAddr address = vsharp.base_address;
                        const u64 before = Core::HashGuestBytes(address, size);
                        const u64 reader = stage.pgm_hash;
                        scheduler.DeferPriorityOperation([address, size, before, reader] {
                            static std::mutex mutex;
                            static std::unordered_map<u64, u64> counts;
                            static u64 total = 0;
                            if (Core::HashGuestBytes(address, size) == before) {
                                return;
                            }
                            std::scoped_lock lk{mutex};
                            const u64 n = ++counts[reader];
                            ++total;
                            if (n <= 3 || n == 100 || n == 1000 || n == 10000) {
                                LOG_WARNING(Render_Vulkan,
                                            "DIAG-131: {:#x}+{:#x} read in place by {:#x} changed "
                                            "before the GPU finished ({} times for it, {} total)",
                                            address, size, reader, n, total);
                            }
                        });
                    }
                    direct_entries[buffer_index - 1] = {u32(vsharp.base_address),
                                                        u32(u64(vsharp.base_address) >> 32),
                                                        u32(size), 1u};
                    ctx->buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                // Max-record descriptors intentionally expose an open-ended guest range.
                // Clamp it to mapped memory before creating a finite Vulkan binding.
                if (size != vsharp.GetSize() && vsharp.num_records != UINT32_MAX) {
                    LOG_ERROR(Render,
                              "Clamped size from {} to {} for stage {:#x}: base={:#x}, stride={}, "
                              "records={:#x}",
                              vsharp.GetSize(), size, stage.pgm_hash, u64(vsharp.base_address),
                              vsharp.GetStride(), vsharp.num_records);
                }
                // DIAG-006: trace GT Sport's 304-byte fragment constant buffers (tonemap).
                if (stage.sw_stage == Shader::SwStage::Fragment && size == 304 &&
                    !desc.is_written) {
                    static std::unordered_set<u64> seen_shaders;
                    static u64 traced = 0;
                    const bool first =
                        seen_shaders.size() < 64 && seen_shaders.insert(stage.pgm_hash).second;
                    if (first ||
                        (stage.pgm_hash == 0xb33ec4df && (++traced <= 20 || traced % 300 == 0))) {
                        const auto* words = reinterpret_cast<const u32*>(vsharp.base_address);
                        // DIAG-008: where each V# dword came from (user data or guest memory).
                        std::string vsharp_source;
                        const auto& fetch = desc.sharp_fetch;
                        for (u32 i = 0; i < 4; ++i) {
                            const bool single =
                                fetch.summary ==
                                std::remove_cvref_t<decltype(fetch)>::Summary::SingleLoad;
                            if (!single && !((fetch.load_mask >> i) & 1)) {
                                vsharp_source += fmt::format(" imm={:08x}", fetch.immediates[i]);
                                continue;
                            }
                            const u32 off =
                                single ? u32(fetch.offsets[0]) + i : u32(fetch.offsets[i]);
                            const u64 src = off < stage.flattened_ud_src.size()
                                                ? stage.flattened_ud_src[off]
                                                : 0;
                            const u32 now = src ? *reinterpret_cast<const u32*>(src) : 0;
                            vsharp_source += fmt::format(
                                " [{}]={:08x}{}", off,
                                off < stage.flattened_ud_buf.size() ? stage.flattened_ud_buf[off]
                                                                    : 0,
                                src ? fmt::format("@{:#x}/now {:08x}", src, now)
                                    : std::string(" ud"));
                        }
                        const auto dump = liverpool->FindRecentConstDump(vsharp.base_address, size);
                        const auto describe_cmd = [&](VAddr address, u64 bytes) {
                            const auto cmd = liverpool->FindRecentCmdBuffer(address, bytes);
                            return cmd ? fmt::format("{}{:#x}+{:#x}@{} ago",
                                                     cmd->ce_count ? "ib " : "dcb ", cmd->address,
                                                     cmd->size,
                                                     liverpool->CmdBufferSequence() - cmd->sequence)
                                       : std::string("none");
                        };
                        const u64 table_address =
                            !stage.flattened_ud_src.empty() &&
                                    fetch.offsets[0] < stage.flattened_ud_src.size()
                                ? stage.flattened_ud_src[fetch.offsets[0]]
                                : 0;
                        if (g_tonemap_checks.size() < 32) {
                            static u64 check_id = 0;
                            auto& check = g_tonemap_checks.emplace_back();
                            check.address = vsharp.base_address;
                            std::memcpy(check.used.data(), words, sizeof(check.used));
                            check.checks_left = 2;
                            check.id = ++check_id;
                        }
                        LOG_WARNING(Render_Vulkan, "Constants in cmdbuf: buffer {} table {}",
                                    describe_cmd(vsharp.base_address, size),
                                    table_address ? describe_cmd(table_address, 16)
                                                  : std::string("ud"));
                        const auto [ce, de] = liverpool->CeDeCounters();
                        LOG_WARNING(
                            Render_Vulkan,
                            "Constants V# source{} pending_submits={} ce={} de={} dump={}",
                            vsharp_source, liverpool->PendingSubmits(), ce, de,
                            dump ? fmt::format("{:#x}+{:#x} {} dumps ago at ce={} de={}",
                                               dump->address, dump->size,
                                               liverpool->ConstDumpSequence() - dump->sequence,
                                               dump->ce_count, dump->de_count)
                                 : std::string("none"));
                        LOG_WARNING(Render_Vulkan,
                                    "Constants {}_{:#x} at {:#x} frame {} cpu_modified={} "
                                    "gpu_modified={} [4..7]={:08x} {:08x} {:08x} {:08x} "
                                    "[47]={:08x}",
                                    stage.hw_stage, stage.pgm_hash, u64(vsharp.base_address),
                                    DebugState.GetFrameNum(),
                                    buffer_cache.IsRegionCpuModified(vsharp.base_address, size),
                                    buffer_cache.IsRegionGpuModified(vsharp.base_address, size),
                                    words[4], words[5], words[6], words[7], words[47]);
                    }
                }
                if (ShouldTraceCrashPage(vsharp.base_address)) {
                    LOG_WARNING(Render_Vulkan,
                                "Crash-page V# shader={}_{:#x} base={:#x} size={:#x} stride={} "
                                "records={:#x} written={} formatted={}",
                                stage.hw_stage, stage.pgm_hash, u64(vsharp.base_address), size,
                                vsharp.GetStride(), vsharp.num_records, desc.is_written,
                                desc.is_formatted);
                }
                if (log_watched) {
                    LOG_WARNING(Render_Vulkan,
                                "DIAG-030: shader {:#x} use {} buffer {} {} {:#x}+{:#x} stride {} "
                                "records {:#x}: {}",
                                stage.pgm_hash, watched_bind, buffer_index - 1,
                                desc.is_written ? "write" : "read", u64(vsharp.base_address), size,
                                vsharp.GetStride(), vsharp.num_records,
                                buffer_cache.DescribeRange(vsharp.base_address, size));
                }
                if (watched && !desc.is_written && size <= 32_MB) {
                    buffer_cache.RefreshReadPages(vsharp.base_address, size, stage.pgm_hash, false);
                }
                VideoCore::g_gpu_write_kind = "shader";
                VideoCore::g_gpu_write_tag = stage.pgm_hash;
                const auto [buffer, offset] = [&] {
                    Common::PhaseTimer obtain_timer{Common::Phase::BufObtain};
                    if (const auto planned = TakePlannedBuffer(
                            vsharp.base_address, u32(size), desc.is_written, desc.is_formatted)) {
                        return *planned; // PERF-067
                    }
                    return buffer_cache.ObtainBuffer(vsharp.base_address, size, desc.is_written,
                                                     desc.is_formatted);
                }();
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                ctx->push_data.AddOffset(binding.buffer, adjust);
                ctx->buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                ctx->bound_buffers.emplace_back(buffer, offset, size, desc.is_written,
                                           vsharp.base_address);
                if (desc.is_written) {
                    NoteShaderWrite(vsharp.base_address, size); // FIX-123
                    if (DirectHazardDiag()) {
                        NoteWrittenRange(vsharp.base_address, size, stage.pgm_hash); // DIAG-129
                    }
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                }
                ctx->needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
                } // D3
            }
        }

        auto& set_write = ctx->set_writes[ctx->set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &ctx->buffer_infos.back();
        ++binding.buffer;
    }
}

void Rasterizer::RefreshGpuWrittenConstants(const Shader::Info& stage,
                                            const VideoCore::Buffer& flat_buffer, u64 flat_offset) {
    // The SRT walker copied these dwords from guest memory while recording. A GPU write to that
    // memory is not visible there (and may not have executed yet), so replace the stale values
    // with a copy that runs after earlier GPU work, as hardware scalar loads would.
    constexpr u64 TrackedAddressLimit = 1ULL << VideoCore::MemoryTracker::MAX_CPU_PAGE_BITS;
    const auto& sources = stage.flattened_ud_src;
    u32 refreshed{};
    for (u32 begin = Shader::NUM_USER_DATA_REGS; begin < sources.size();) {
        const VAddr address = sources[begin];
        u32 end = begin + 1;
        while (end < sources.size() && address != 0 &&
               sources[end] == address + u64(end - begin) * sizeof(u32)) {
            ++end;
        }
        const u64 size = u64(end - begin) * sizeof(u32);
        if (address != 0 && address + size <= TrackedAddressLimit &&
            buffer_cache.IsRegionGpuModified(address, size)) {
            const auto [buffer, buffer_offset] = buffer_cache.ObtainBuffer(address, size, false);
            const vk::BufferCopy copy = {
                .srcOffset = buffer_offset,
                .dstOffset = flat_offset + u64(begin) * sizeof(u32),
                .size = size,
            };
            runtime.CopyBuffer(buffer, &flat_buffer, std::span{&copy, 1});
            refreshed += end - begin;
        }
        begin = end;
    }
    if (refreshed == 0) {
        return;
    }
    // The copies are tracked writes to the flat buffer; order them before this draw's reads.
    ctx->needs_barrier = true;
    // Each refresh ends the current render pass, so report how often it happens.
    if (const u64 count = ++gpu_constant_refresh_count;
        count >= 1000 && count == next_gpu_constant_refresh_report) {
        next_gpu_constant_refresh_report *= 10;
        LOG_WARNING(Render_Vulkan, "GPU-written flattened constants refreshed for {} bindings",
                    count);
    }
    static constexpr size_t MaxLoggedShaders = 64;
    if (logged_gpu_constant_shaders.size() < MaxLoggedShaders &&
        logged_gpu_constant_shaders.insert(stage.pgm_hash).second) {
        LOG_WARNING(Render_Vulkan,
                    "Refreshing {} GPU-written flattened constants on the GPU for shader {}_{:#x}",
                    refreshed, stage.hw_stage, stage.pgm_hash);
    }
}

void Rasterizer::LogInvalidTextureContext(const Shader::Info& stage, u32 sharp_offset) {
    // This compute shader supplies several live UI textures. Capture its bad descriptor
    // inputs in normal-speed runs too; the Khronos layer makes GT Sport too slow to inspect.
    const bool trace_gt_sport_ui = stage.pgm_hash == 0xaa3822a3;
    if ((!EmulatorSettings.IsVkValidationEnabled() && !trace_gt_sport_ui) ||
        invalid_texture_context_count >= (trace_gt_sport_ui ? 4 : 16)) {
        return;
    }
    ++invalid_texture_context_count;
    const auto words =
        std::span{stage.flattened_ud_buf.data(),
                  std::min<size_t>(stage.flattened_ud_buf.size(), trace_gt_sport_ui ? 96 : 256)};
    LOG_WARNING(Render_Vulkan, "Invalid T# context shader={}_{:#x} flatbuf[0:{}]={:08x}",
                stage.hw_stage, stage.pgm_hash, words.size(), fmt::join(words, " "));
    const auto sharp_begin = sharp_offset > 16 ? sharp_offset - 16 : 0;
    const auto sharp_end = std::min<size_t>(stage.flattened_ud_buf.size(), sharp_offset + 8);
    if (sharp_begin < sharp_end) {
        const auto sharp_words =
            std::span{stage.flattened_ud_buf.data() + sharp_begin, sharp_end - sharp_begin};
        LOG_WARNING(Render_Vulkan, "Invalid T# table flatbuf[{}:{}]={:08x}", sharp_begin, sharp_end,
                    fmt::join(sharp_words, " "));
    }
    for (u32 index = 0; index < stage.buffers.size(); ++index) {
        const auto& resource = stage.buffers[index];
        if (resource.IsSpecial()) {
            continue;
        }
        const auto buffer = resource.GetSharp(stage);
        const u64 size = buffer.GetSize();
        const u64 checked_size = std::min<u64>(size, 4_KB);
        const bool gpu_modified =
            buffer.base_address != 0 && checked_size != 0 &&
            buffer_cache.IsRegionGpuModified(buffer.base_address, checked_size);
        LOG_WARNING(Render_Vulkan,
                    "Invalid T# buffer context shader={}_{:#x} buffer={} address={:#x} "
                    "size={:#x} checked_size={:#x} gpu_modified={} written={}",
                    stage.hw_stage, stage.pgm_hash, index, u64(buffer.base_address), size,
                    checked_size, gpu_modified, resource.is_written);
    }
}

// DIAG-031: where each dword of a rejected T# came from: its flat buffer slot, the guest address
// the SRT walker read it from, that address's value now, and whether the GPU has written it.
template <typename Fetch>
static void LogRejectedSharpSource(const Shader::Info& stage, const Fetch& fetch, u32 image,
                                   VideoCore::BufferCache& buffer_cache) {
    static std::atomic<u32> logged{};
    if (++logged > 30) {
        return;
    }
    std::string dwords;
    for (u32 i = 0; i < fetch.offsets.size(); ++i) {
        const bool single = fetch.summary == Fetch::Summary::SingleLoad;
        if (!single && !((fetch.load_mask >> i) & 1)) {
            dwords += fmt::format(" imm={:08x}", fetch.immediates[i]);
            continue;
        }
        const u32 off = single ? u32(fetch.offsets[0]) + i : u32(fetch.offsets[i]);
        const u64 src = off < stage.flattened_ud_src.size() ? stage.flattened_ud_src[off] : 0;
        dwords += fmt::format(" [{}]={:08x}{}", off,
                              off < stage.flattened_ud_buf.size() ? stage.flattened_ud_buf[off] : 0,
                              src ? fmt::format("@{:#x} now {:08x} gpu={}", src,
                                                *reinterpret_cast<const u32*>(src),
                                                buffer_cache.IsRegionGpuModified(src, 4))
                                  : std::string(" (user data)"));
    }
    LOG_WARNING(Render_Vulkan,
                "DIAG-031: rejected T# {} of {}_{:#x}: fetch summary {} mask {:#x}:{}", image,
                stage.hw_stage, stage.pgm_hash, u32(fetch.summary), fetch.load_mask, dwords);
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const u32 first_image_idx = ctx->image_infos.size();
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    // PERF-121: stages fill image_bindings one after the other, so bound_textures can point at
    // their descriptions instead of copying them.
    u32 num_images = ctx->image_binding_base;
    const u32 first_binding = num_images;
    static const bool decode_once = Common::PerfFeatureEnabled(54);
    if (decode_once && ctx->decoded_tsharps.empty()) {
        ctx->decoded_tsharps.resize(4096);
    }
    if (const u64 epoch = tsharp_epoch.load(std::memory_order_acquire); epoch != ctx->decoded_epoch) {
        // FIX-046: memory was unmapped; T#s are checked against the mappings again.
        ctx->decoded_epoch = epoch;
        for (auto& entry : ctx->decoded_tsharps) {
            entry.valid = false;
        }
    }
    std::optional<Common::PhaseTimer> pass1_timer{std::in_place, Common::Phase::TexPass1};
    u64 stage_tex_hash = 0xcbf29ce484222325ULL ^ u64(stage.pgm_hash); // DIAG-120
    // PERF-123: the stage reads the same T#s as in its previous draw and no image was registered
    // or unregistered since: the bindings found then are bound again without decoding or lookups.
    static const bool stage_memo_enabled = [] {
        const char* env = std::getenv("SHADGT_STAGE_IMAGE_MEMO");
        return Common::PerfFeatureEnabled(123) && (!env || env[0] != '0');
    }();
    auto& stage_memo = ctx->stage_image_memos[u32(stage.hw_stage) & 7];
    u64 memo_hash = 0;
    const u64 memo_generation = texture_cache.RegistryGeneration();
    const u64 memo_epoch = ctx->decoded_epoch;
    bool memo_hit = false;
    if (stage_memo_enabled && decode_once && !stage.images.empty()) {
        memo_hash = 0xcbf29ce484222325ULL;
        for (const auto& image_desc : stage.images) {
            const auto tsharp = image_desc.GetSharp(stage);
            std::array<u32, 8> words;
            std::memcpy(words.data(), &tsharp, sizeof(words));
            const u8 flags = u8(image_desc.is_depth) | u8(image_desc.is_written) << 1 |
                             u8(image_desc.is_array) << 2;
            memo_hash = (memo_hash ^ XXH3_64bits(words.data(), sizeof(words)) ^ flags) *
                        0x100000001b3ULL;
        }
        memo_hit = stage_memo.valid && stage_memo.stage == &stage &&
                   stage_memo.hash == memo_hash && stage_memo.generation == memo_generation &&
                   stage_memo.epoch == memo_epoch &&
                   num_images + stage_memo.bindings.size() <= ctx->image_bindings.size();
    }
    static std::atomic<u64> memo_hits{}, memo_misses{};
    if (memo_hit) {
        if (const u64 n = ++memo_hits; n % 200000 == 0) {
            LOG_WARNING(Render_Vulkan, "PERF-123: {} stage image sets reused, {} looked up", n,
                        memo_misses.load());
        }
        for (const auto& cached : stage_memo.bindings) {
            auto& entry = ctx->image_bindings[num_images++];
            entry.image_id = cached.image_id;
            std::construct_at(&entry.desc, cached.desc);
            if (cached.image_id) {
                auto& image = texture_cache.GetImage(cached.image_id);
                if (image.binding.is_bound) {
                    image.binding.force_general |=
                        cached.desc.type == VideoCore::TextureCache::BindingType::Storage;
                }
                image.binding.is_bound = 1u;
                texture_cache.NoteImageUsed(cached.image_id);
            }
        }
        image_descriptor_array_sizes.assign(stage_memo.array_sizes.begin(),
                                            stage_memo.array_sizes.end());
    } else if (memo_hash != 0) {
        ++memo_misses;
    }
    for (const auto& image_desc :
         memo_hit ? std::span<const Shader::ImageResource>{}
                  : std::span<const Shader::ImageResource>{stage.images.data(),
                                                           stage.images.size()}) {
        const auto tsharp = image_desc.GetSharp(stage);
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        // PERF-042: a T# decoded before skips the checks and the decode below.
        DecodedTsharp* decoded = nullptr;
        bool decoded_hit = false;
        if (decode_once) {
            std::array<u32, 8> words;
            std::memcpy(words.data(), &tsharp, sizeof(words));
            const u8 flags = u8(image_desc.is_depth) | u8(image_desc.is_written) << 1 |
                             u8(image_desc.is_array) << 2;
            u64 hash = XXH3_64bits(words.data(), sizeof(words));
            hash ^= u64(flags) * 0x9E3779B97F4A7C15ull;
            stage_tex_hash = (stage_tex_hash ^ hash) * 0x100000001b3ULL; // DIAG-120
            decoded = &ctx->decoded_tsharps[hash & (ctx->decoded_tsharps.size() - 1)];
            decoded_hit = decoded->valid && decoded->flags == flags && decoded->sharp == words;
            if (!decoded_hit) {
                decoded->valid = false;
                decoded->sharp = words;
                decoded->flags = flags;
                decoded->find_memo = {};
            }
        }
        const auto bind_null_image = [&] {
            const u32 array_size =
                ForEachImageDescriptorBinding(stage, image_desc, [&](bool is_written) {
                    auto& [image_id, desc] = ctx->image_bindings[num_images++];
                    image_id = {};
                    desc = {};
                    // Preserve every compiled mip slot and its sampled/storage type when
                    // rejecting the guest descriptor, including reused scratch entries.
                    desc.type = is_written ? VideoCore::TextureCache::BindingType::Storage
                                           : VideoCore::TextureCache::BindingType::Texture;
                });
            image_descriptor_array_sizes.push_back(array_size);
        };
        if (!decoded_hit &&
            (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid)) {
            bind_null_image();
            continue;
        }

        if (!decoded_hit && !memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}, shader={}_{:#x}, sharp_offset={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt), stage.hw_stage, stage.pgm_hash,
                        image_desc.sharp_fetch.offsets[0]);
            LogInvalidTextureContext(stage, image_desc.sharp_fetch.offsets[0]);
            LogRejectedSharpSource(stage, image_desc.sharp_fetch, num_images, buffer_cache);
            bind_null_image();
            continue;
        }

        if (!decoded_hit &&
            LiverpoolToVK::TrySurfaceFormat(data_fmt, num_fmt) == vk::Format::eUndefined) {
            LOG_WARNING(Render_Vulkan,
                        "Binding null image for unsupported T# format data={} number={} "
                        "address={:#x} shader={}_{:#x}",
                        static_cast<u32>(data_fmt), static_cast<u32>(num_fmt), tsharp.Address(),
                        stage.hw_stage, stage.pgm_hash);
            bind_null_image();
            continue;
        }

        const auto geometry_error =
            decoded_hit ? VideoCore::ImageDescriptorGeometryError::None
                        : VideoCore::CheckImageDescriptorGeometry(tsharp, instance.GetImageLimits());
        if (geometry_error != VideoCore::ImageDescriptorGeometryError::None) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# geometry={} address={:#x} type={} "
                        "extent={}x{}x{} pitch={} layers={} levels={} base_level={} samples={} "
                        "data_format={} num_format={} shader={}_{:#x} sharp_offset={}",
                        magic_enum::enum_name(geometry_error), tsharp.Address(),
                        AmdGpu::NameOf(tsharp.GetType()), tsharp.width + 1, tsharp.height + 1,
                        tsharp.depth + 1, tsharp.Pitch(), tsharp.NumLayers(), tsharp.NumLevels(),
                        tsharp.base_level, tsharp.NumSamples(), static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt), stage.hw_stage, stage.pgm_hash,
                        image_desc.sharp_fetch.offsets[0]);
            LogInvalidTextureContext(stage, image_desc.sharp_fetch.offsets[0]);
            bind_null_image();
            continue;
        }

        if (ShouldTraceCrashPage(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan,
                        "Crash-page T# shader={}_{:#x} address={:#x} type={} extent={}x{}x{} "
                        "pitch={} layers={} levels={} tiling={} data_format={} num_format={} "
                        "written={} sharp_offset={}",
                        stage.hw_stage, stage.pgm_hash, tsharp.Address(),
                        AmdGpu::NameOf(tsharp.GetType()), tsharp.width + 1, tsharp.height + 1,
                        tsharp.depth + 1, tsharp.Pitch(), tsharp.NumLayers(), tsharp.NumLevels(),
                        u32(tsharp.tiling_index), static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt), image_desc.is_written,
                        image_desc.sharp_fetch.offsets[0]);
        }
        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);

        if (!decoded_hit) {
            // FIX-046: the image must start inside the guest address space. A garbage T# of
            // GT Sport's grass compute shader 0xaa3822a3 with base 0xa000 passed the checks
            // above (they only bound the address to 40 bits), and tracking its 4 MB stopped the
            // emulator (Protect: "addr 0xa000 out of bounds", October 9 22:03). Like the
            // rejections above, the binding becomes a null image. Checked when a T# is first
            // decoded.
            const VideoCore::TextureCache::ImageDesc checked{tsharp, image_desc};
            const auto& info = checked.info;
            // FIX-081: GT Sport's grass compute shader 0xaa3822a3 reads the T# at flattened
            // offset 109 from data that is mostly floats when the texture is not used (addresses
            // like 0x3f80000000 and 0xbf80000000 are the bits of 1.0f and -1.0f). Creating and
            // detiling those images reset the GPU ("device lost") in races, past every geometry
            // check. From that slot only an image the game already made (a render target, a
            // texture another draw uploaded) is used; anything else becomes a null image.
            if (stage.pgm_hash == 0xaa3822a3 && image_desc.sharp_fetch.offsets[0] == 109 &&
                !texture_cache.AnyImageInRegion(info.guest_address, 1)) {
                static std::atomic<u32> skipped{};
                if (const u32 n = ++skipped; n <= 10 || n % 5000 == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "FIX-081: grass T# {} at {:#x} is not an existing image; bound "
                                "as a null image",
                                n, info.guest_address);
                }
                bind_null_image();
                continue;
            }
            if (!IsMappedStart(info.guest_address)) {
                static std::atomic<u32> rejected{};
                if (const u32 n = ++rejected; n <= 20 || n % 1000 == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "FIX-046: rejecting T# {} at {:#x}+{:#x} outside the guest address "
                                "space, shader={}_{:#x}, sharp_offset={}",
                                n, info.guest_address, info.guest_size, stage.hw_stage,
                                stage.pgm_hash, image_desc.sharp_fetch.offsets[0]);
                }
                bind_null_image();
                continue;
            }
            if (decoded) {
                std::construct_at(&decoded->desc, checked);
                decoded->valid = true;
            }
        }
        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = ctx->image_bindings[num_images++];
            if (decoded) {
                std::construct_at(&desc, decoded->desc);
            } else {
                std::construct_at(&desc, tsharp, image_desc);
            }

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            {
                Common::PhaseTimer find_timer{Common::Phase::TexFindImage};
                // PERF-114: a decoded T# with one plain binding looks its image up through a memo.
                if (decoded && num_bindings == 1 &&
                    mip_fallback_mode == Shader::MipStorageFallbackMode::None) {
                    image_id = texture_cache.FindImageMemo(desc, decoded->find_memo);
                } else {
                    image_id = texture_cache.FindImage(desc);
                }
            }
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
            // DIAG-139 (SHADGT_DIAG_FROSTED=1): the pictures GT Sport's frosted glass
            // (ps 0x90b5dcdd) reads, a few times a second.
            static const bool diag_frosted = [] {
                const char* env = std::getenv("SHADGT_DIAG_FROSTED");
                return env && env[0] == '1';
            }();
            if (diag_frosted && ((stage.pgm_hash & 0xffffffff) == 0x90b5dcdd || t_frosted_rt ||
                                 (image->info.guest_address == 0x1009bc0000 && RytyGtsInRace()))) {
                static auto last = std::chrono::steady_clock::now();
                static u32 in_window = 0;
                const auto now = std::chrono::steady_clock::now();
                if (now - last > std::chrono::milliseconds(250)) {
                    last = now;
                    in_window = 0;
                }
                if (RytyGtsInRace() || in_window++ < 4) {
                    LOG_WARNING(Render_Vulkan,
                                "DIAG-139: frame {} ps {:#x} reads image {} at {:#x} {}x{} fmt {} "
                                "tile {} flags {:#x} contents {} race {}{}",
                                g_prepared_frames.load(std::memory_order_relaxed),
                                stage.pgm_hash & 0xffffffff, image_id.index,
                                image->info.guest_address, image->info.size.width,
                                image->info.size.height, u32(image->info.pixel_format),
                                u32(image->info.tile_mode), u32(image->flags),
                                image->contents_version, RytyGtsInRace(),
                                t_frosted_rt ? " (into the bar's blur)" : "");
                }
            }
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    pass1_timer.reset();
    if (stage_memo_enabled && memo_hash != 0 && !memo_hit) {
        // PERF-123: kept when no image was (un)registered while looking them up.
        stage_memo.valid = texture_cache.RegistryGeneration() == memo_generation;
        stage_memo.stage = &stage;
        stage_memo.hash = memo_hash;
        stage_memo.generation = memo_generation;
        stage_memo.epoch = memo_epoch;
        stage_memo.bindings.assign(ctx->image_bindings.begin() + first_binding,
                                   ctx->image_bindings.begin() + num_images);
        stage_memo.array_sizes.assign(image_descriptor_array_sizes.begin(),
                                      image_descriptor_array_sizes.end());
    }
    {
        // DIAG-120: how often a stage binds exactly the T#s it bound in the previous draw.
        static std::array<u64, 8> last_hash{};
        static u64 same = 0, total = 0;
        static auto since = std::chrono::steady_clock::now();
        auto& last = last_hash[u32(stage.hw_stage) & 7];
        same += last == stage_tex_hash;
        ++total;
        last = stage_tex_hash;
        if (const auto now = std::chrono::steady_clock::now();
            now - since >= std::chrono::seconds{2}) {
            LOG_WARNING(Render_Vulkan, "DIAG-120: texture sets in 2 s: {} of {} same as the "
                        "stage's previous draw", same, total);
            same = total = 0;
            since = now;
        }
    }
    // Second pass to re-bind images that were updated after binding
    for (u32 i = first_binding; i < num_images; ++i) {
        auto& [image_id, desc] = ctx->image_bindings[i];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            if (DiagHistoryEnabled()) {
                ctx->diag_empty_bindings += fmt::format(" | EMPTY img {} of {:#x}", i, stage.pgm_hash);
            }
            ctx->image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                image_id = texture_cache.FindImage(desc);
            }

            ctx->bound_images.emplace_back(image_id);
            if (is_storage) {
                ctx->diag_storage_images.push_back(image_id);
            }

            auto& image = texture_cache.GetImage(image_id);
            if (frame_dump_collecting && !is_storage &&
                True(image.flags & VideoCore::ImageFlagBits::GpuModified)) {
                auto key = FrameDumpTextureKey(desc.info); // NAT-123
                if (std::ranges::find(frame_dump_textures, key,
                                      [](const auto& t) { return std::get<0>(t); }) ==
                    frame_dump_textures.end()) {
                    frame_dump_textures.emplace_back(std::move(key), image_id, image.image_uid);
                }
            }
            std::optional<Common::PhaseTimer> part_timer{std::in_place, Common::Phase::TexView};
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            part_timer.emplace(Common::Phase::TexTransit);
            const auto binding = image.binding;

            if (binding.is_target && image.info.props.is_depth) {
                const bool depth_write = ctx->db_desc.second.view_info.is_storage;
                const bool stencil_write = image.info.props.has_stencil &&
                                           liverpool->DrawRegs().depth_control.stencil_enable &&
                                           !depth_write;
                const auto layout =
                    DepthAttachmentLayout(image.info.props.has_stencil, depth_write, stencil_write,
                                          true, instance.IsAttachmentFeedbackLoopLayoutSupported());
                ctx->needs_barrier |=
                    runtime.Transit(&image, layout, vk::PipelineStageFlagBits2::eAllGraphics,
                                    vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
            } else if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                // FIX-130: a draw that samples the color target it renders into (a feedback loop)
                // must see what earlier draws wrote to it. Consecutive draws in one rendering pass
                // have no dependency, so their reads raced the earlier writes. GT Sport blurs each
                // frosted-glass panel in two passes into one picture, the second reading what the
                // first wrote; on the race preview the top bar's blur came out dark now and then
                // (on Lance's PC most frames). -DisablePerf 152 leaves it as before.
                static const bool feedback_barrier = Common::PerfFeatureEnabled(152);
                if (feedback_barrier && image.binding.is_target) {
                    scheduler.EndRendering();
                    runtime.FlushBarriers();
                    const vk::MemoryBarrier2 written{
                        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader |
                                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                         vk::AccessFlagBits2::eColorAttachmentRead |
                                         vk::AccessFlagBits2::eColorAttachmentWrite,
                    };
                    scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
                        .memoryBarrierCount = 1,
                        .pMemoryBarriers = &written,
                    });
                }
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    ctx->needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    ctx->needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    ctx->needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    ctx->needs_barrier |= runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            ctx->image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            ctx->bound_textures.emplace_back(ctx->image_infos.size() - 1, image_id, &desc, image.GetImage());
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = first_binding;
    ctx->image_binding_base = num_images;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = ctx->image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = ctx->set_writes[ctx->set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &ctx->image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    Common::PhaseTimer samplers_timer{Common::Phase::TexSamplers};
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                liverpool->DrawRegs().ta_bc_base.Address() == 0)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid S# max_aniso={}, filter_mode={}, mip_filter={}, "
                        "border_color_type={}, border_color_base={:#x}, shader={}_{:#x}, "
                        "sharp_offset={}, min_lod={}, max_lod={}",
                        static_cast<u32>(ssharp.max_aniso.Value()),
                        static_cast<u32>(ssharp.filter_mode.Value()),
                        static_cast<u32>(ssharp.mip_filter.Value()),
                        static_cast<u32>(ssharp.border_color_type.Value()),
                        liverpool->DrawRegs().ta_bc_base.Address(), stage.hw_stage, stage.pgm_hash,
                        sampler.sharp_fetch.offsets[0], ssharp.MinLod(), ssharp.MaxLod());
            ssharp = AmdGpu::Sampler{};
        }
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, liverpool->DrawRegs().ta_bc_base, sampler.is_depth);
        ctx->image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = ctx->set_writes[ctx->set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &ctx->image_infos.back();
    }
}

void Rasterizer::RebindTextures(bool is_compute, u32 num_color_targets) {
    const auto refresh_targets = [&](bool gpu_dirty_only) {
        if (is_compute) {
            return;
        }
        for (u32 cb = 0; cb < num_color_targets; ++cb) {
            auto& [image_id, desc] = ctx->cb_descs[cb];
            if (!image_id) {
                continue;
            }
            if (texture_cache.GetImage(image_id).binding.needs_rebind) {
                image_id = ctx->bound_images.emplace_back(texture_cache.FindImage(desc));
            }
            auto& image = texture_cache.GetImage(image_id);
            image.binding.is_target = 1u;
            if (!gpu_dirty_only || True(image.flags & VideoCore::ImageFlagBits::GpuDirty)) {
                texture_cache.UpdateImage(image_id);
            }
        }
        auto& [image_id, desc] = ctx->db_desc;
        if (image_id) {
            if (texture_cache.GetImage(image_id).binding.needs_rebind) {
                image_id = ctx->bound_images.emplace_back(texture_cache.FindImage(desc));
            }
            auto& image = texture_cache.GetImage(image_id);
            image.binding.is_target = 1u;
            if (!gpu_dirty_only || True(image.flags & VideoCore::ImageFlagBits::GpuDirty)) {
                texture_cache.UpdateImage(image_id);
            }
        }
    };
    // Target refreshes can export sampled/storage aliases too. Finish them before the final
    // texture refresh sweep, rather than allowing BeginRendering to invalidate a bound image.
    refresh_targets(false);
    // A later stage may expand an image that an earlier stage already bound. Preserve each
    // descriptor's image and range until all image discovery and refresh operations finish.
    for (auto& bound : ctx->bound_textures) {
        const auto old_binding = texture_cache.GetImage(bound.image_id).binding;
        if (old_binding.needs_rebind) {
            bound.image_id = texture_cache.FindImage(*bound.desc);
            auto* image = &texture_cache.GetImage(bound.image_id);
            if (const auto depth_id = texture_cache.GetAssociatedDepth(*image)) {
                bound.image_id = depth_id;
                image = &texture_cache.GetImage(depth_id);
            }
            image->binding.is_target |= old_binding.is_target;
            image->binding.force_general |= old_binding.force_general;
            ctx->bound_images.emplace_back(bound.image_id);
        }
        auto& image = texture_cache.GetImage(bound.image_id);
        image.binding.is_bound = 1u;
        const auto& view = texture_cache.FindTexture(bound.image_id, *bound.desc);
        ctx->image_infos[bound.descriptor_index].imageView = *view.image_view;
        bound.backing_image = image.GetImage();
    }
    // A CPU-dirty alias refreshed later in the sweep can export an earlier bound image and
    // leave it GpuDirty. Consume that arena authority before a pending storage-image write.
    // GpuDirty aliases themselves block competing exports, so this cleanup cannot export
    // another overlapping authoritative image and restart the cycle.
    refresh_targets(true);
    for (auto& bound : ctx->bound_textures) {
        auto& image = texture_cache.GetImage(bound.image_id);
        if (False(image.flags & VideoCore::ImageFlagBits::GpuDirty)) {
            continue;
        }
        const auto& view = texture_cache.FindTexture(bound.image_id, *bound.desc);
        ctx->image_infos[bound.descriptor_index].imageView = *view.image_view;
        bound.backing_image = image.GetImage();
    }
}

void Rasterizer::FinalizeTextureLayouts(RenderState* render_state) {
    const auto& regs = liverpool->DrawRegs();
    for (u32 i = 0; i < ctx->bound_textures.size(); ++i) {
        const auto& bound = ctx->bound_textures[i];
        if (std::ranges::any_of(std::span{ctx->bound_textures.data(), i}, [&](const auto& previous) {
                return previous.image_id == bound.image_id &&
                       previous.backing_image == bound.backing_image;
            })) {
            continue;
        }
        auto& image = texture_cache.GetImage(bound.image_id);
        const auto backing = std::ranges::find_if(image.backing_images, [&](const auto& backing) {
            return backing.image.image == bound.backing_image;
        });
        ASSERT(backing != image.backing_images.end());

        ImageBindingRequirements requirements{
            .is_depth = image.info.props.is_depth,
            .has_stencil = image.info.props.has_stencil,
            .feedback_layout_supported = instance.IsAttachmentFeedbackLoopLayoutSupported(),
            .is_compute = render_state == nullptr,
        };
        for (const auto& other : ctx->bound_textures) {
            if (other.image_id == bound.image_id && other.backing_image == bound.backing_image) {
                requirements.is_storage |=
                    other.desc->type == VideoCore::TextureCache::BindingType::Storage;
            }
        }
        // The image may have separate single- and multi-sample backings. Attachment layout
        // requirements apply only to the physical backing used by the attachment.
        if (render_state && image.GetImage() == bound.backing_image) {
            for (u32 cb = 0; cb < render_state->num_color_attachments; ++cb) {
                requirements.color_attachment |= ctx->cb_descs[cb].image_id == bound.image_id;
            }
            requirements.depth_attachment = ctx->db_desc.first == bound.image_id;
            requirements.depth_write = ctx->db_desc.second.view_info.is_storage ||
                                       render_state->depth_stencil_attachment.depth_clear;
            requirements.stencil_write =
                requirements.has_stencil &&
                (render_state->depth_stencil_attachment.stencil_clear ||
                 (regs.depth_control.stencil_enable && !requirements.depth_write));
        }
        const auto state = FinalImageBindingState(requirements);
        auto* active_backing = image.backing;
        image.backing = &*backing;
        // Every descriptor of this backing must agree over its entire view. A final full-image
        // transition also repairs partial sampled/storage overlaps and attachment promotions.
        ctx->needs_barrier |= runtime.Transit(&image, state.layout, state.stages, state.access);
        image.backing = active_backing;

        for (const auto& other : ctx->bound_textures) {
            if (other.image_id == bound.image_id && other.backing_image == bound.backing_image) {
                ctx->image_infos[other.descriptor_index].imageLayout = state.layout;
            }
        }
        if (requirements.color_attachment) {
            for (u32 cb = 0; cb < render_state->num_color_attachments; ++cb) {
                if (ctx->cb_descs[cb].image_id == bound.image_id) {
                    render_state->color_attachments[cb].image_layout = state.layout;
                }
            }
        }
        if (requirements.depth_attachment) {
            render_state->depth_stencil_attachment.image_layout = state.layout;
        }
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    ctx->attachment_feedback_loop = {};
    const auto& regs = liverpool->DrawRegs();
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = ctx->cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = ctx->bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        {
            // DIAG-139: render targets in GT Sport's frosted-glass scratch memory.
            static const bool diag_frosted = [] {
                const char* env = std::getenv("SHADGT_DIAG_FROSTED");
                return env && env[0] == '1';
            }();
            if (cb == 0) {
                t_frosted_rt = diag_frosted && image->info.guest_address == 0x1009bc0000 &&
                               image->info.size.width == 960 && image->info.size.height == 200;
            }
            if (diag_frosted && image->info.guest_address == 0x1009bc0000 &&
                image->info.size.width == 960) {
                static auto last = std::chrono::steady_clock::now();
                static u32 in_window = 0;
                const auto now = std::chrono::steady_clock::now();
                if (now - last > std::chrono::milliseconds(250)) {
                    last = now;
                    in_window = 0;
                }
                if (RytyGtsInRace() || in_window++ < 4) {
                    LOG_WARNING(Render_Vulkan,
                                "DIAG-139: frame {} render target image {} at {:#x} {}x{} fmt {} "
                                "contents {} race {}",
                                g_prepared_frames.load(std::memory_order_relaxed), image_id.index, image->info.guest_address, image->info.size.width,
                                image->info.size.height, u32(image->info.pixel_format),
                                image->contents_version, RytyGtsInRace());
                }
            }
        }
        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
        if (DiagHistoryEnabled() && image->info.size.width >= 128 &&
            (image->info.pixel_format == vk::Format::eR16G16B16A16Sfloat ||
             image->info.pixel_format == vk::Format::eB10G11R11UfloatPack32 ||
             image->info.pixel_format == vk::Format::eR16Sfloat)) {
            // DIAG-105: the first draws into each float render target: whether its CMASK said
            // cleared, for GT Sport's pink effect rectangles.
            static std::mutex mutex;
            static std::unordered_map<u64, u32> seen;
            std::scoped_lock lock{mutex};
            if (seen.size() < 20000 && seen[image->image_uid]++ < 2) {
                LOG_WARNING(Render_Vulkan,
                            "DIAG-105: uid {} {:#x} {}x{} {} cmask {:#x} fast clear {} "
                            "clear {} cleared {} ver {}",
                            image->image_uid, image->info.guest_address, image->info.size.width,
                            image->info.size.height, vk::to_string(image->info.pixel_format),
                            col_buf.CmaskAddress(), bool(col_buf.info.fast_clear),
                            is_clear, seen[image->image_uid], image->contents_version);
            }
        }

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            ctx->needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            ctx->attachment_feedback_loop |= vk::ImageAspectFlagBits::eColor;
        } else {
            ctx->needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = ctx->db_desc.first; image_id) {
        auto& desc = ctx->db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const auto clear_aspects = DepthStencilClearAspects(
            regs.depth_buffer, regs.depth_control, regs.depth_render_control,
            texture_cache.IsMetaCleared(htile_address, slice));
        const bool is_depth_clear = bool(clear_aspects & vk::ImageAspectFlagBits::eDepth);
        const bool is_stencil_clear = bool(clear_aspects & vk::ImageAspectFlagBits::eStencil);
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil &&
            (is_stencil_clear || (regs.depth_control.stencil_enable && !desc.view_info.is_storage));
        const auto new_layout = DepthAttachmentLayout(
            has_stencil, desc.view_info.is_storage || is_depth_clear, stencil_write,
            image.binding.is_bound, instance.IsAttachmentFeedbackLoopLayoutSupported());
        if (image.binding.is_bound &&
            (desc.view_info.is_storage || is_depth_clear || stencil_write)) {
            ctx->attachment_feedback_loop |= image.aspect_mask;
        }
        ctx->needs_barrier |=
            runtime.Transit(&image, new_layout,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits2::eLateFragmentTests |
                                (image.binding.is_bound ? vk::PipelineStageFlagBits2::eAllGraphics
                                                        : vk::PipelineStageFlagBits2::eNone),
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                                (image.binding.is_bound ? vk::AccessFlagBits2::eShaderRead
                                                        : vk::AccessFlagBits2::eNone),
                            desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->DrawCbExtent(0);
    const auto& mrt1_hint = liverpool->DrawCbExtent(1);
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->DrawRegs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->DrawRegs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));
    if (FrameDumpActive()) {
        FrameDumpEvent(fmt::format("resolve {:#x} {}x{} x{} -> {:#x} {}x{} x{}", // NAT-122
                                   mrt0_image.info.guest_address, mrt0_image.info.size.width,
                                   mrt0_image.info.size.height, mrt0_image.info.num_samples,
                                   mrt1_image.info.guest_address, mrt1_image.info.size.width,
                                   mrt1_image.info.size.height, mrt1_image.info.num_samples));
    }

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->DrawRegs().color_buffers[0].Address(),
                                 liverpool->DrawRegs().color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->DrawRegs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->DrawDbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->DrawDbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->DrawRegs().depth_view.slice_start;
    sub_range.extent.layers = liverpool->DrawRegs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    if (num_bytes == 0) {
        return;
    }
#ifdef RYTY_GPU
    if (native) {
        native->FillBuffer(address, num_bytes, value, is_gds); // NAT-100
        return;
    }
#endif
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        const bool meta = texture_cache.ClearMeta(address);
        if (FrameDumpActive()) {
            FrameDumpEvent(fmt::format("fill {:#x}+{:#x} = {:#x}{}", address, num_bytes, value,
                                       meta ? " (metadata clear)" : ""));
        }
        if (DiagHistoryEnabled()) {
            // DIAG-105: fills, and whether they cleared a known CMASK/HTILE.
            static std::atomic<u32> logged{};
            if (logged++ < 3000) {
                LOG_WARNING(Render_Vulkan, "DIAG-105: fill {:#x}+{:#x} value {:#x} meta {}",
                            address, num_bytes, value, meta);
            }
        }
        const bool cpu_path = !buffer_cache.IsRegionGpuModified(address, num_bytes) &&
                              !buffer_cache.HasGpuImageAlias(address, num_bytes);
        if (cpu_path) {
            u32* buffer = std::bit_cast<u32*>(address);
            Core::MemoryManager::NoteEmulatorWrite(address, num_bytes, &value);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            VideoCore::BumpUploadEpoch();
            return;
        }
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        VideoCore::g_gpu_write_kind = "FillBuffer";
        VideoCore::g_gpu_write_tag = value;
        return buffer_cache.ObtainBuffer(address, num_bytes, true);
    }();
    if (!is_gds) {
        texture_cache.InvalidateMemoryFromGPU(address, num_bytes);
    }
    runtime.FillBuffer(buffer, offset, num_bytes, value);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (num_bytes == 0) {
        return;
    }
#ifdef RYTY_GPU
    if (native) {
        native->CopyBuffer(dst, src, num_bytes, dst_gds, src_gds); // NAT-100
        return;
    }
#endif
    if (FrameDumpActive()) {
        FrameDumpEvent(fmt::format("copy {:#x}{} <- {:#x}{} ({:#x} bytes)", dst,
                                   dst_gds ? " (GDS)" : "", src, src_gds ? " (GDS)" : "",
                                   num_bytes)); // NAT-117
    }
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes) &&
        !buffer_cache.HasGpuImageAlias(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !buffer_cache.HasGpuImageAlias(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            VideoCore::BumpUploadEpoch();
            return;
        }
    }
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes, false, true);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        VideoCore::g_gpu_write_kind = src_gds ? "CopyBuffer from GDS" : "CopyBuffer from";
        VideoCore::g_gpu_write_tag = src;
        return buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
    }();
    if (!dst_gds) {
        texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    }
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
    if (src_gds && !dst_gds) {
        // PERF-009: GT Sport copies GDS counters to memory that its CPU then touches every frame.
        buffer_cache.RecordGdsReadback(dst, src, num_bytes);
    }
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
#ifdef RYTY_GPU
    if (native) {
        return native->ReadGds(gds_offset); // NAT-100
    }
#endif
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks, u64 exact_write_size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
#ifdef RYTY_GPU
    if (native) {
        native->OnCpuWrite(addr, size, exact_write_size); // NAT-105, NAT-109
        return true;
    }
#endif
    buffer_cache.InvalidateMemory(addr, size, assume_locks, exact_write_size);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

void Rasterizer::WaitForEarlyFences(VAddr page) {
    liverpool->WaitForEarlyFences(page);
}

void Rasterizer::WaitForGuard(VAddr page) {
    liverpool->WaitForGuard(page);
}

bool Rasterizer::CanWriteThrough(VAddr addr, u64 size) {
#ifdef RYTY_GPU
    if (native) {
        return false; // NAT-105: writes to watched pages always reach OnCpuWrite
    }
#endif
    const VAddr page = Common::AlignDown(addr, VAddr{4096});
    if (texture_cache.AnyImageInRegion(page, 4096)) {
        return false;
    }
    return buffer_cache.CanWriteThrough(addr, size);
}

void Rasterizer::FinishWriteThrough(VAddr page, std::span<const u8> before, VAddr store,
                                    u64 store_size) {
    buffer_cache.FinishWriteThrough(page, before, store, store_size);
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size, false, assume_locks);
    return true;
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
#ifdef RYTY_GPU
    // Ryty reports mappings only once libSceVideoOut and the rasterizer exist, so memory the game
    // mapped earlier is missing here, yet the page manager tracks it. A CPU write fault there was
    // declined, Ryty then unprotected the whole 16 KB view, and the caches kept stale copies
    // (exploded geometry, wrong glyphs). Committed guest memory counts as mapped.
    return boost::icl::contains(mapped_ranges, range) || memory->IsValidMapping(addr, size);
#else
    return boost::icl::contains(mapped_ranges, range);
#endif
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

bool Rasterizer::IsMappedStart(VAddr address) {
    // FIX-046: only an address outside the guest address space is refused; that is what stops
    // page tracking (Protect: "out of bounds"). A first version also required the start to be in
    // a mapped area (FIX-027's test) and refused GT Sport's 105 MB GPU resource at 0x2900000000
    // (likely its shadow map; shadows broke in Lance's 22:18 run).
#ifdef RYTY_GPU
    // Ryty's IsValidMapping asks Windows whether the page is committed, which refused that same
    // resource: it is never CPU-mapped. Anything in Ryty's guest arena counts as guest address
    // space, as in shadGT; reads of its uncommitted parts give zeros.
    constexpr VAddr GuestArenaBase = 0x200000000;
    constexpr VAddr GuestArenaEnd = 0xFC00000000;
    if (address >= GuestArenaBase && address < GuestArenaEnd) {
        return true;
    }
#endif
    return memory->IsValidMapping(address);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    tsharp_epoch.fetch_add(1, std::memory_order_release); // FIX-046
#ifdef RYTY_GPU
    if (native) {
        native->OnUnmap(addr, size); // NAT-100
    }
#endif
    if (auto* direct = buffer_cache.GetDirectMemory()) {
        direct->OnUnmap(addr, size); // D1
    }
    buffer_cache.InvalidateMemory(addr, size);
    // FIX-026: the texture cache has no cache-wide lock since upstream #5219; freeing images here
    // on a game thread raced with the GPU thread freeing the same image ("Trying to unregister an
    // already unregistered image" when starting a race). Free them on the GPU thread and wait,
    // as CPU fault flushes already do. -DisablePerf 35 frees them here.
    static const bool unmap_on_gpu_thread = Common::PerfFeatureEnabled(35);
    if (unmap_on_gpu_thread && !IsGpuThread()) {
        liverpool->SendCommand<true>([&] { texture_cache.UnmapMemory(addr, size); });
    } else {
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed,
                                    const bool quad_triangles) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed, quad_triangles);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = DynState(); // NAT-101
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->DrawRegs();

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.primitive_type == AmdGpu::PrimitiveType::QuadList &&
            regs.depth_buffer.DepthValid() &&
            regs.depth_buffer.z_info.format == AmdGpu::DepthBuffer::ZFormat::Z32Float &&
            !regs.depth_shader_control.z_export_enable && !regs.polygon_control.NeedsBias()) {
            const auto far_depth =
                ReadOnlyFarDepthMax(viewport.minDepth, viewport.maxDepth, regs.depth_control);
            if (far_depth != viewport.maxDepth) {
                static const bool logged = [&] {
                    LOG_INFO(
                        Render_Vulkan,
                        "Applying read-only far-depth precision workaround: viewport=[{}, {}], "
                        "guest clamp=[{}, {}], scale={}, offset={}",
                        viewport.minDepth, viewport.maxDepth, vp_d.zmin, vp_d.zmax, zscale,
                        zoffset);
                    return true;
                }();
                viewport.maxDepth = far_depth;
            }
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = DynState(); // NAT-101
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->DrawRegs();
    auto& dynamic_state = DynState(); // NAT-101

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        const auto& sc = regs.stencil_control;
        const auto depth_compare = !depth_test_enabled ? AmdGpu::CompareFunc::Always
                                   : regs.depth_control.depth_bounds_enable
                                       ? AmdGpu::CompareFunc::Less
                                       : regs.depth_control.depth_func;
        const auto front_ref = ResolveStencilReference(
            sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front,
            regs.depth_control.stencil_ref_func, front, depth_compare);
        const auto back_ref =
            regs.depth_control.backface_enable
                ? ResolveStencilReference(sc.stencil_fail_back, sc.stencil_zpass_back,
                                          sc.stencil_zfail_back, regs.depth_control.stencil_bf_func,
                                          back, depth_compare)
                : front_ref;
        if (!front_ref.exact || !back_ref.exact) {
            LOG_WARNING(Render_Vulkan,
                        "Stencil operations require conflicting references: front test={} op={} "
                        "selected={}, back test={} op={} selected={}",
                        front.stencil_test_val, front.stencil_op_val, front_ref.value,
                        back.stencil_test_val, back.stencil_op_val, back_ref.value);
        }
        dynamic_state.SetStencilReferences(front_ref.value, back_ref.value);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed, const bool quad_triangles) const {
    const auto& regs = liverpool->DrawRegs();
    auto& dynamic_state = DynState(); // NAT-101

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    // PERF-028: generated quad indices have their restarts already applied.
    const auto prim_restart =
        !quad_triangles && (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->DrawRegs();
    auto& dynamic_state = DynState(); // NAT-101
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->DrawRegs();
    auto& dynamic_state = DynState(); // NAT-101
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(ctx->attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

bool Rasterizer::IsGpuThread() const {
    return liverpool->IsGpuThread(std::this_thread::get_id());
}

bool Rasterizer::CommandThreadFaultIsUrgent() {
    return liverpool->CommandThreadFaultIsUrgent();
}

void Rasterizer::OnGpuThreadFault() {
    liverpool->OnGpuThreadFault();
}

#ifdef __linux__
bool Rasterizer::IsGpuThreadTid(u32 tid) const {
    return liverpool->IsGpuThreadTid(tid);
}
#endif

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan
