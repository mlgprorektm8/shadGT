// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <filesystem>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_shader_program.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
struct Liverpool;
union Regs;
} // namespace AmdGpu

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

/// DIAG-030: shaders whose inputs are logged (GT Sport's grass compute shaders by default, or
/// the comma-separated hashes in SHADGT_WATCH_SHADERS).
bool IsWatchedShader(u64 hash);

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    /// PERF-028: draw a quad list through the tessellation helpers instead of as triangles
    /// (indirect draws, or indices the command thread cannot read).
    bool tessellate_quads{};
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void Sync();

    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {});

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

    /// The translated program of a guest shader, if it was compiled (diagnostic bundles).
    const Program* FindProgram(u64 hash) const {
        const auto it = program_cache.find(hash);
        return it != program_cache.end() ? it->second.get() : nullptr;
    }

private:
    bool RefreshGraphicsKey();
    /// Registers the pipeline keys are built from: the current ones, or during read-ahead the
    /// registers at an upcoming draw (PERF-019).
    const AmdGpu::Regs& Regs() const;
    bool HasSupportedColorTargets() const;
    struct PipelineBuild;
    class PipelineBuildWorkers;
    /// PERF-019: builds the pipeline for the current key on a worker thread.
    std::shared_ptr<PipelineBuild> StartPipelineBuild(bool urgent);
    /// A build of the current key, infos and modules.
    std::shared_ptr<PipelineBuild> MakePipelineBuild();
    void QueueBuild(const std::shared_ptr<PipelineBuild>& build, bool urgent);
    /// Builds the pipeline unless a worker already started; returns when it is built.
    void FinishBuild(PipelineBuild& build);
    /// PERF-019: starts builds for the new pipelines of the draws after the current one.
    void ReadAhead(const DrawIndirectParams params, bool restore = true);
    /// PERF-020: while new pipelines keep appearing, reads each command buffer ahead as soon as
    /// it starts, so its first new pipelines are built before their draws.
    void ReadAheadAtBufferStart();
    /// Selects swizzled-alpha blend emulation for a lane-dependent blend, or reports it once.
    void RefreshSwizzledBlend(u32 cb, Shader::PsColorBuffer& color_buffer,
                              const AmdGpu::BlendControl& bc);
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpFailedShader(std::span<const u32> code, u64 hash, Shader::HwStage stage);
    void LogRepeatedPermutation(const Program& program, size_t perm_idx, Shader::HwStage hw_stage,
                                u64 pgm_hash);
    std::unordered_set<u64> failed_shaders;
    std::vector<u32> diag_last_spv;
    /// FIX-043 diagnostic: hash of the SPIR-V that CompileModule produced last.
    u64 last_spv_hash{};
    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);
    bool IsTessEmulatedDraw() const;
    /// PERF-028: the current draw is a quad list drawn as a triangle list.
    bool QuadListAsTriangles() const;
    void RecoverAttributeFlags(Shader::Info& info, const Shader::StageSpecialization& spec,
                               const Shader::ShaderParams& params,
                               const Shader::RuntimeInfo& runtime_info, size_t perm_idx);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    /// PERF-013: per program, one past the highest permutation index in the stored cache, so a
    /// permutation created at runtime never reuses (and overwrites) a stored index.
    tsl::robin_map<size_t, size_t> stored_perm_end;
    /// FIX-018: programs translated again with the else-scope fix, and the first permutation
    /// index made with it; stored permutations below it are not preloaded.
    struct RetranslatedProgram {
        size_t boundary;
        u32 fixes; ///< 1: FIX-018 else scope, 2/4: FIX-020 wave64 lane reads (v1/v2)
    };
    std::unordered_map<u64, RetranslatedProgram> else_scope_fixed;
    std::unordered_set<u64> else_scope_checked;
    std::vector<std::unique_ptr<Program>> retired_programs;
    std::filesystem::path else_scope_fixed_path;
    void LoadElseScopeFixed();
    void SaveElseScopeFixed() const;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    // The driver's VkPipelineCache, kept on disk so cached pipelines are not recompiled by the
    // driver on every launch.
    void CreateDriverCache();
    void SaveDriverCache(bool wait);
    void MaybeSaveDriverCache();
    std::filesystem::path driver_cache_path;
    u32 pipelines_at_driver_save{};
    std::chrono::steady_clock::time_point driver_cache_saved_at{};
    std::jthread driver_cache_writer;

    std::unordered_set<u64> logged_swizzled_blends;

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    // PERF-019: pipelines being built ahead of their draws.
    const AmdGpu::Regs* regs_override{};
    u32 draws_since_scan{};
    std::chrono::steady_clock::time_point last_pipeline_miss{};
    /// PERF-021: register states already evaluated by the read-ahead, so reading the same
    /// commands again (every command buffer start re-reads what follows) costs only a hash.
    std::unordered_set<u64> read_ahead_states;
    struct ReadAheadStats {
        u32 scans{};
        u32 draws{};
        u32 evaluated{};
        u32 started{};
        double ms{};
        std::chrono::steady_clock::time_point since{};
    } read_ahead_stats;
    u32 last_scan_draws{};
    tsl::robin_map<GraphicsPipelineKey, std::shared_ptr<PipelineBuild>> pending_builds;
    // Last member, so the workers stop before anything they use is destroyed.
    std::unique_ptr<PipelineBuildWorkers> build_workers;
};

} // namespace Vulkan
