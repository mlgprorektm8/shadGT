// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/spin_mutex.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
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

    /// PERF-047: with pipelines selected on the command thread, both GPU threads look pipelines
    /// up; every GetGraphicsPipeline/GetComputePipeline call holds this lock, and so does any
    /// read of the infos below or copy of a cached info.
    Common::SpinMutex& LookupMutex() {
        return lookup_mutex;
    }
    /// The infos the last GetGraphicsPipeline selected, with that draw's user data.
    const std::array<const Shader::Info*, MaxShaderStages>& SelectedInfos() const {
        return infos;
    }

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
    /// PERF-034: with the draw pipe, starts builds for the upcoming draw states the command
    /// thread queued. During a draw (`restore`), its key is computed again afterwards.
    void BuildQueuedStates(const DrawIndirectParams params, bool restore);
    /// PERF-038: claims a build nobody started, so it is never built; false if one started.
    static bool CancelBuild(PipelineBuild& build);
    /// PERF-038: queues a stored pipeline's build behind all other builds.
    void QueueBackgroundBuild(const std::shared_ptr<PipelineBuild>& build);
    /// PERF-038: on this thread, between draws: takes stored pipelines the reader thread
    /// prepared and queues their background builds.
    void InstallPrewarmed();
    struct PrewarmRecord;
    bool InstallPrewarmRecord(PrewarmRecord& record);
    void PrewarmRead(std::stop_token stop);
    void ReportPrewarm(std::chrono::steady_clock::time_point now);
    /// PERF-034: waits while a worker builds a pipeline, starting builds for the states the
    /// command thread queues meanwhile. Returns at once if no worker took the build, which
    /// FinishBuild then does on this thread.
    void WaitForBuild(PipelineBuild& build, const DrawIndirectParams params);
    /// Selects swizzled-alpha blend emulation for a lane-dependent blend, or reports it once.
    void RefreshSwizzledBlend(u32 cb, Shader::PsColorBuffer& color_buffer,
                              const AmdGpu::BlendControl& bc);
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpFailedShader(std::span<const u32> code, u64 hash, Shader::HwStage stage);
    void LogRepeatedPermutation(const Program& program, size_t perm_idx, Shader::HwStage hw_stage,
                                u64 pgm_hash);
    std::unordered_set<u64> failed_shaders;
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
    /// PERF-032: stored translations are loaded when their program is first used (no startup
    /// precompile), and new ones are stored.
    bool use_stored_shaders{};
    void ListStoredPermutations(u64 pgm_hash, Program& program);
    /// PERF-038: a program's stored permutations as read from the store (any thread), and their
    /// insertion into the program (this thread).
    struct StoredListing {
        u64 pgm_hash{};
        struct Entry {
            size_t perm_idx;
            std::vector<u8> meta;
            bool has_spv;
        };
        std::vector<Entry> entries;
    };
    static StoredListing ReadStoredListing(u64 pgm_hash);
    void ApplyStoredListing(StoredListing&& listing, Program& program);
    vk::ShaderModule LoadStoredModule(u64 pgm_hash, size_t perm_idx);
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
    std::atomic<bool> driver_cache_writing{}; // PERF-036
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
    // PERF-034: the registers of the queued draw state being evaluated.
    std::unique_ptr<AmdGpu::Regs> queued_state_regs;
    struct QueuedStateStats {
        u32 evaluated{};
        u32 started{};
        u32 waits{};
        double wait_ms{};
        double ms{};
        std::chrono::steady_clock::time_point since{};
    } queued_state_stats;
    tsl::robin_map<GraphicsPipelineKey, std::shared_ptr<PipelineBuild>> pending_builds;
    // PERF-038: stored pipelines built in the background (opt-in, SHADGT_PREWARM=1).
    bool prewarm_enabled{};
    std::mutex prewarm_mutex;
    std::condition_variable_any prewarm_cv;
    std::deque<std::unique_ptr<PrewarmRecord>> prewarm_records;
    std::atomic<size_t> prewarm_queued{};
    std::atomic<u64> prewarm_built{};
    std::atomic<u64> prewarm_read{};
    std::atomic<u64> prewarm_total{};
    std::atomic<bool> prewarm_reader_done{};
    struct PrewarmStats {
        u64 installed{};
        u64 skipped{};
        u64 hits{};
        u64 cancelled{};
        u64 promoted{};
        u64 built_before{};
        double ms{};
        std::chrono::steady_clock::time_point since{};
    } prewarm_stats;
    std::jthread prewarm_reader;
    Common::SpinMutex lookup_mutex; // PERF-047, PERF-060
    // Last member, so the workers stop before anything they use is destroyed.
    std::unique_ptr<PipelineBuildWorkers> build_workers;
};

} // namespace Vulkan
