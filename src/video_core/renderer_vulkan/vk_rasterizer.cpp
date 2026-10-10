// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    scheduler.StartPerfMonitor();

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });
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

void Rasterizer::CopySelectedStages(std::array<SelectedStage, Shader::MaxStageTypes>& out) {
    const auto& infos = pipeline_cache.SelectedInfos();
    for (u32 stage = 0; stage < Shader::MaxStageTypes; ++stage) {
        auto& selected = out[stage];
        const Shader::Info* info = stage < infos.size() ? infos[stage] : nullptr;
        selected.present = info != nullptr;
        if (!info) {
            continue;
        }
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
    if (!info) {
        return;
    }
    selected.pgm_base = info->pgm_base;
    selected.num_user_data = static_cast<u32>(info->user_data.size());
    ASSERT(selected.num_user_data <= selected.user_data.size());
    std::ranges::copy(info->user_data, selected.user_data.begin());
    selected.flattened = info->flattened_ud_buf;
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
                std::scoped_lock lk{pipeline_cache.LookupMutex()};
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

bool Rasterizer::PrepareBuffersAhead(const SelectedPipeline& selected, BufferPlan& plan) {
    const Pipeline* pipeline = selected.is_compute
                                   ? static_cast<const Pipeline*>(selected.compute)
                                   : static_cast<const Pipeline*>(selected.pipeline);
    if (!pipeline || IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return true; // Nothing to obtain, or BindResources is not reached.
    }
    plan.generation = texture_cache.RegistryGeneration();
    struct FlatView {
        const std::vector<u32>& flattened_ud_buf;
        VAddr pgm_base;
    };
    static const bool null_impossible = Common::PerfFeatureEnabled(39);
    constexpr u64 GpuAddressLimit = 1ULL << 40;
    const auto& canonical = pipeline->CanonicalStages();
    for (u32 stage = 0; stage < Shader::MaxStageTypes; ++stage) {
        const auto& sel = selected.stages[stage];
        const Shader::Info* info = canonical[stage];
        if (!info || !sel.present) {
            continue;
        }
        if (IsWatchedShader(info->pgm_hash)) {
            ++plan_refusals[0];
            return false; // Its read pages are checked again before each use (FIX-017).
        }
        const FlatView view{sel.flattened, sel.pgm_base};
        for (const auto& desc : info->buffers) {
            if (desc.IsSpecial()) {
                continue;
            }
            if (!desc.sharp_fetch.FitsIn(sel.flattened.size())) {
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
            const auto result = buffer_cache.ObtainBufferAhead(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, plan.uploads);
            if (!result) {
                ++plan_refusals[2];
                return false;
            }
            plan.entries.push_back({vsharp.base_address, size, desc.is_written, desc.is_formatted,
                                    result->first, result->second});
        }
    }
    return true;
}

std::optional<std::pair<const VideoCore::Buffer*, u64>> Rasterizer::TakePlannedBuffer(
    VAddr address, u32 size, bool written, bool texel) {
    auto* plan = buffer_plan;
    if (!plan || plan->next >= plan->entries.size()) {
        return std::nullopt;
    }
    const auto& entry = plan->entries[plan->next];
    if (entry.address != address || entry.size != size || entry.written != written ||
        entry.texel != texel) {
        // The recorder binds something else than planned: it obtains the rest itself.
        ++plan_stats.mismatched;
        plan->next = plan->entries.size();
        return std::nullopt;
    }
    ++plan->next;
    if (texture_cache.RegistryGeneration() != plan->generation &&
        texture_cache.AnyImageInRegion(address, size)) {
        // An image over the range appeared since: obtained again, with its export.
        ++plan_stats.redone;
        return std::nullopt;
    }
    ++plan_stats.used;
    return std::pair<const VideoCore::Buffer*, u64>{entry.buffer, entry.offset};
}

Rasterizer::ShadowInfo& Rasterizer::ShadowOf(const Shader::Info* canonical) {
    // PERF-065: most draws use the infos of the draws before them; the map is looked up once.
    auto& cached = shadow_lookup[(reinterpret_cast<uintptr_t>(canonical) >> 6) %
                                 shadow_lookup.size()];
    if (cached.first != canonical) {
        cached = {canonical, &shadow_infos[canonical]};
    }
    auto& shadow = *cached.second;
    // A cached info's identity never changes; a different one means the address was reused.
    if (!shadow.info || shadow.identity != canonical->identity) {
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
        auto& shadow = ShadowOf(canonical[stage]);
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
                        "{}, buffer {} (texel read {}, image {}, stream ring {}, arena {}, "
                        "residency {}, staging {})",
                        plan_stats.used, plan_stats.redone, plan_stats.mismatched,
                        plan_refusals[0].exchange(0), plan_refusals[1].exchange(0),
                        plan_refusals[2].exchange(0), buffer_cache.ahead_refusals[0].exchange(0),
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
                std::scoped_lock lk{pipeline_cache.LookupMutex()};
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
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->DrawCbExtent(cb);
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->DrawDbExtent();
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
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
            add(cb_descs[cb].image_id, DiagRole::Target);
        }
        add(db_desc.first, DiagRole::Depth);
    }
    for (const auto id : bound_images) {
        const auto& image = texture_cache.GetImage(id);
        if (image.binding.is_target) {
            continue;
        }
        const bool storage =
            image.binding.force_general ||
            std::ranges::find(diag_storage_images, id) != diag_storage_images.end();
        add(id, storage ? DiagRole::Storage : DiagRole::Sampled);
    }
    for (const auto& bound : bound_buffers) {
        if (bound.guest_address != 0) {
            record.buffers.push_back({bound.guest_address, bound.size, bound.is_written});
        }
    }
    record.empty = diag_empty_bindings;
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
    constexpr size_t MaxProducers = 6000;
    for (size_t q = 0; q < queue.size() && producers.size() < MaxProducers; ++q) {
        const auto [uid, before, depth] = queue[q];
        size_t found = 0;
        for (size_t i = before; i-- > 0 && found < MaxProducersPerImage;) {
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

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    ++Common::GetWorkCounters().draws;
    using Common::Phase;
    using Common::PhaseTimer;
    PhaseTimer total_timer{Phase::DrawTotal};

    {
        PhaseTimer t{Phase::DrawSetup};
        SubmitChunkIfNeeded();
        scheduler.PopPendingOperations();
    }

    const auto& regs = liverpool->DrawRegs();
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
            return;
        }
    }

    {
        PhaseTimer t{Phase::RenderTargets};
        PrepareRenderState(pipeline);
    }
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
        return;
    }
    RenderState state;
    {
        PhaseTimer t{Phase::BeginRendering};
        state = BeginRendering(pipeline);
        FinalizeTextureLayouts(&state);

        if (needs_barrier) {
            runtime.FlushBarriers();
        }
    }

    {
        PhaseTimer t{Phase::Descriptors};
        pipeline->BindResources(set_writes, push_data);
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
    // DIAG-045: GT Sport accumulates the thumbnail's car frames into a 1600x900 target; trace the
    // frame being added while the images it was made from are still alive.
    {
        const auto rt = cb_descs[0].image_id;
        const auto& stages = pipeline->GetStages();
        const auto* fs = stages[u32(Shader::SwStage::Fragment)];
        if (rt && fs && fs->pgm_hash == 0x5589ec47) {
            const auto& info = texture_cache.GetImage(rt).info;
            static u32 accumulation_traces = 0;
            if (info.size.width == 1600 && info.size.height == 900 && accumulation_traces < 2) {
                for (const auto image_id : bound_images) {
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
            const auto image_id = cb_descs[cb].image_id;
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
            for (const auto image_id : bound_images) {
                const auto& image = texture_cache.GetImage(image_id);
                images += fmt::format(
                    " | img {} {:#x} {}x{} {} {} gpu={} ver={}", image_id.index,
                    image.info.guest_address, image.info.size.width, image.info.size.height,
                    vk::to_string(image.info.pixel_format), AmdGpu::NameOf(image.info.tile_mode),
                    True(image.flags & VideoCore::ImageFlagBits::GpuModified),
                    image.contents_version);
            }
            for (u32 cb = 0; cb < std::bit_width(pipeline->GetGraphicsKey().mrt_mask); ++cb) {
                if (const auto image_id = cb_descs[cb].image_id) {
                    const auto& image = texture_cache.GetImage(image_id);
                    images += fmt::format(
                        " | rt{} {} {:#x} {}x{} {} mask {:#x}", cb, image_id.index,
                        image.info.guest_address, image.info.size.width, image.info.size.height,
                        vk::to_string(image.info.pixel_format), regs.color_target_mask.GetMask(cb));
                }
            }
            for (const auto& bound : bound_buffers) {
                if (bound.guest_address != 0) {
                    images += fmt::format(" | buf {:#x}+{:#x}{}", bound.guest_address, bound.size,
                                          bound.is_written ? " w" : "");
                }
            }
            images += diag_empty_bindings;
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
                for (const auto image_id : bound_images) {
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
        return;
    }

    PrepareRenderState(pipeline);
    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer();
    }

    const auto [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, stride * max_count);
    bound_buffers.emplace_back(buffer, base, stride * max_count, false); // FIX-016

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
        bound_buffers.emplace_back(count_buffer, count_offset, 4, false); // FIX-016
    }

    if (!BindResources(pipeline)) {
        return;
    }
    auto state = BeginRendering(pipeline);
    FinalizeTextureLayouts(&state);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
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
    Common::PhaseTimer total_timer{Common::Phase::DispatchTotal};

    SubmitChunkIfNeeded();
    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->DrawCsRegs();
    // PERF-055: selected on the command thread when it decoded the dispatch.
    const ComputePipeline* pipeline = AcquireComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->DrawRegs(), cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

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

    SubmitChunkIfNeeded();
    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->DrawCsRegs();
    // PERF-055: looked up here (no selection for indirect dispatches), with fresh stage copies.
    const ComputePipeline* pipeline = AcquireComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, false); // FIX-016

    if (!BindResources(pipeline)) {
        return;
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, push_data);

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

bool Rasterizer::DeferFenceSignal(VAddr address, Common::UniqueFunction<void>&& signal,
                                  bool compute_queue, std::span<const u8> value) {
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
    if (!readbacks_pending && !address_pending && !earlier_deferred) {
        return false;
    }
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
    scheduler.Flush();
    return true;
}

void Rasterizer::InlineDeferredWrite(VAddr address, std::span<const u8> data) {
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
            Render_Vulkan, "Deferred fences: {} in {} frames, latency avg {:.2f} ms max {:.2f} ms",
            stats.count, frames, stats.total_us / 1000.0 / stats.count, stats.max_us / 1000.0);
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
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();
    bound_textures.clear();

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
    push_data = MakeUserData(liverpool->DrawRegs());
    for (u32 i = 0; i < stages.size(); ++i) {
        const auto* stage = stages[i];
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage_bindings[i] = PlanStageResourceBindings(*stage, binding);
        auto buffer_binding = stage_bindings[i].buffers;
        Common::PhaseTimer t{Common::Phase::Buffers};
        BindBuffers(*stage, buffer_binding, push_data);
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
    const auto& regs = liverpool->DrawRegs();
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

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
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainBuffer(range.base_address, size, false);
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
        // FIX-016: record the vertex fetch as a read of the range, so a later upload into it
        // (GT Sport rewrites grass vertex buffers between draws) waits for this draw. Without
        // it the copy can overwrite the vertices while this draw still reads them, mixing
        // old and new data (the suspected cause of stretched NÃ¼rburgring grass and terrain).
        bound_buffers.emplace_back(range.buffer, range.offset, size, false);
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
    const auto& regs = liverpool->DrawRegs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(index_address, index_buffer_size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
    bound_buffers.emplace_back(buffer, offset, index_buffer_size, false); // FIX-016
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
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, is_written, guest_address] : bound_buffers) {
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
        for (const auto& bound : bound_buffers) {
            buffers.emplace_back(bound.buffer->Handle(), bound.offset, bound.size,
                                 bound.is_written);
        }
        boost::container::small_vector<vk::Image, 32> images;
        for (const auto id : bound_images) {
            images.push_back(texture_cache.GetImage(id).GetImage());
        }
        // Render targets are left out: the color slots past the draw's targets keep ids of
        // images that may be gone.
        runtime.DiagBatchDraw(buffers, images);
        runtime.NoteSessionDraw(buffers); // PERF-066
    }
    bound_images.clear();
    bound_buffers.clear();
    diag_empty_bindings.clear();
    diag_storage_images.clear();
    needs_barrier = false;
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
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
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
    for (const auto& desc : stage.buffers) {
        ++buffer_index;
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                RefreshGpuWrittenConstants(stage, vk_buffer, offset);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->DrawRegs().clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
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
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = liverpool->DrawCsRegs();
                const u64 lds_size = u64(cs_program.SharedMemSize()) * cs_program.NumWorkgroups();
                // GCN LDS is undefined at workgroup launch, so only reserve a GPU-only region.
                const auto offset = lds_buffer.Reserve(lds_size, alignment);
                ASSERT_MSG(offset, "Emulated shared memory size {:#x} exceeds the stream buffer",
                           lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), *offset, lds_size);
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
                                n, buffer_infos.size(), stage.pgm_hash, u64(vsharp.base_address),
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
                    diag_empty_bindings += fmt::format(
                        " | EMPTY buf {} of {:#x}: base={:#x} stride={} records={:#x}",
                        buffer_infos.size(), stage.pgm_hash, u64(vsharp.base_address),
                        vsharp.GetStride(), vsharp.num_records);
                }
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
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
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                bound_buffers.emplace_back(buffer, offset, size, desc.is_written,
                                           vsharp.base_address);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                }
                needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
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
    needs_barrier = true;
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
    const u32 first_image_idx = image_infos.size();
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    u32 num_images{};
    static const bool decode_once = Common::PerfFeatureEnabled(54);
    if (decode_once && decoded_tsharps.empty()) {
        decoded_tsharps.resize(4096);
    }
    if (const u64 epoch = tsharp_epoch.load(std::memory_order_acquire); epoch != decoded_epoch) {
        // FIX-046: memory was unmapped; T#s are checked against the mappings again.
        decoded_epoch = epoch;
        for (auto& entry : decoded_tsharps) {
            entry.valid = false;
        }
    }
    for (const auto& image_desc : stage.images) {
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
            decoded = &decoded_tsharps[hash & (decoded_tsharps.size() - 1)];
            decoded_hit = decoded->valid && decoded->flags == flags && decoded->sharp == words;
            if (!decoded_hit) {
                decoded->valid = false;
                decoded->sharp = words;
                decoded->flags = flags;
            }
        }
        const auto bind_null_image = [&] {
            const u32 array_size =
                ForEachImageDescriptorBinding(stage, image_desc, [&](bool is_written) {
                    auto& [image_id, desc] = image_bindings[num_images++];
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
            auto& [image_id, desc] = image_bindings[num_images++];
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
                image_id = texture_cache.FindImage(desc);
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
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (u32 i = 0; i < num_images; ++i) {
        auto& [image_id, desc] = image_bindings[i];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            if (DiagHistoryEnabled()) {
                diag_empty_bindings += fmt::format(" | EMPTY img {} of {:#x}", i, stage.pgm_hash);
            }
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);
            if (is_storage) {
                diag_storage_images.push_back(image_id);
            }

            auto& image = texture_cache.GetImage(image_id);
            std::optional<Common::PhaseTimer> part_timer{std::in_place, Common::Phase::TexView};
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            part_timer.emplace(Common::Phase::TexTransit);
            const auto binding = image.binding;

            if (binding.is_target && image.info.props.is_depth) {
                const bool depth_write = db_desc.second.view_info.is_storage;
                const bool stencil_write = image.info.props.has_stencil &&
                                           liverpool->DrawRegs().depth_control.stencil_enable &&
                                           !depth_write;
                const auto layout =
                    DepthAttachmentLayout(image.info.props.has_stencil, depth_write, stencil_write,
                                          true, instance.IsAttachmentFeedbackLoopLayoutSupported());
                needs_barrier |=
                    runtime.Transit(&image, layout, vk::PipelineStageFlagBits2::eAllGraphics,
                                    vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
            } else if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    needs_barrier |= runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            bound_textures.emplace_back(image_infos.size() - 1, image_id, desc, image.GetImage());
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

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
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

void Rasterizer::RebindTextures(bool is_compute, u32 num_color_targets) {
    const auto refresh_targets = [&](bool gpu_dirty_only) {
        if (is_compute) {
            return;
        }
        for (u32 cb = 0; cb < num_color_targets; ++cb) {
            auto& [image_id, desc] = cb_descs[cb];
            if (!image_id) {
                continue;
            }
            if (texture_cache.GetImage(image_id).binding.needs_rebind) {
                image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            }
            auto& image = texture_cache.GetImage(image_id);
            image.binding.is_target = 1u;
            if (!gpu_dirty_only || True(image.flags & VideoCore::ImageFlagBits::GpuDirty)) {
                texture_cache.UpdateImage(image_id);
            }
        }
        auto& [image_id, desc] = db_desc;
        if (image_id) {
            if (texture_cache.GetImage(image_id).binding.needs_rebind) {
                image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
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
    for (auto& bound : bound_textures) {
        const auto old_binding = texture_cache.GetImage(bound.image_id).binding;
        if (old_binding.needs_rebind) {
            bound.image_id = texture_cache.FindImage(bound.desc);
            auto* image = &texture_cache.GetImage(bound.image_id);
            if (const auto depth_id = texture_cache.GetAssociatedDepth(*image)) {
                bound.image_id = depth_id;
                image = &texture_cache.GetImage(depth_id);
            }
            image->binding.is_target |= old_binding.is_target;
            image->binding.force_general |= old_binding.force_general;
            bound_images.emplace_back(bound.image_id);
        }
        auto& image = texture_cache.GetImage(bound.image_id);
        image.binding.is_bound = 1u;
        const auto& view = texture_cache.FindTexture(bound.image_id, bound.desc);
        image_infos[bound.descriptor_index].imageView = *view.image_view;
        bound.backing_image = image.GetImage();
    }
    // A CPU-dirty alias refreshed later in the sweep can export an earlier bound image and
    // leave it GpuDirty. Consume that arena authority before a pending storage-image write.
    // GpuDirty aliases themselves block competing exports, so this cleanup cannot export
    // another overlapping authoritative image and restart the cycle.
    refresh_targets(true);
    for (auto& bound : bound_textures) {
        auto& image = texture_cache.GetImage(bound.image_id);
        if (False(image.flags & VideoCore::ImageFlagBits::GpuDirty)) {
            continue;
        }
        const auto& view = texture_cache.FindTexture(bound.image_id, bound.desc);
        image_infos[bound.descriptor_index].imageView = *view.image_view;
        bound.backing_image = image.GetImage();
    }
}

void Rasterizer::FinalizeTextureLayouts(RenderState* render_state) {
    const auto& regs = liverpool->DrawRegs();
    for (u32 i = 0; i < bound_textures.size(); ++i) {
        const auto& bound = bound_textures[i];
        if (std::ranges::any_of(std::span{bound_textures.data(), i}, [&](const auto& previous) {
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
        for (const auto& other : bound_textures) {
            if (other.image_id == bound.image_id && other.backing_image == bound.backing_image) {
                requirements.is_storage |=
                    other.desc.type == VideoCore::TextureCache::BindingType::Storage;
            }
        }
        // The image may have separate single- and multi-sample backings. Attachment layout
        // requirements apply only to the physical backing used by the attachment.
        if (render_state && image.GetImage() == bound.backing_image) {
            for (u32 cb = 0; cb < render_state->num_color_attachments; ++cb) {
                requirements.color_attachment |= cb_descs[cb].image_id == bound.image_id;
            }
            requirements.depth_attachment = db_desc.first == bound.image_id;
            requirements.depth_write = db_desc.second.view_info.is_storage ||
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
        needs_barrier |= runtime.Transit(&image, state.layout, state.stages, state.access);
        image.backing = active_backing;

        for (const auto& other : bound_textures) {
            if (other.image_id == bound.image_id && other.backing_image == bound.backing_image) {
                image_infos[other.descriptor_index].imageLayout = state.layout;
            }
        }
        if (requirements.color_attachment) {
            for (u32 cb = 0; cb < render_state->num_color_attachments; ++cb) {
                if (cb_descs[cb].image_id == bound.image_id) {
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
    attachment_feedback_loop = {};
    const auto& regs = liverpool->DrawRegs();
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop |= vk::ImageAspectFlagBits::eColor;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
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

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
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
            attachment_feedback_loop |= image.aspect_mask;
        }
        needs_barrier |=
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
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
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
    buffer_cache.InvalidateMemory(addr, size, assume_locks, exact_write_size);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

void Rasterizer::WaitForEarlyFences(VAddr page) {
    liverpool->WaitForEarlyFences(page);
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
    return boost::icl::contains(mapped_ranges, range);
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
    return memory->IsValidMapping(address);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    tsharp_epoch.fetch_add(1, std::memory_order_release); // FIX-046
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

    auto& dynamic_state = scheduler.GetDynamicState();
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

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->DrawRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

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
    auto& dynamic_state = scheduler.GetDynamicState();

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
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->DrawRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
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
