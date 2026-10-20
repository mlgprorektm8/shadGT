// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <boost/container/small_vector.hpp>

namespace Shader {
struct Info;
struct PushData;
} // namespace Shader

namespace Vulkan {

static constexpr auto AllGraphicsStageBits =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
    vk::ShaderStageFlagBits::eFragment;

class Instance;
class Scheduler;
class DescriptorHeap;

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
             bool is_compute = false);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        // PERF-017: the fully optimized pipeline once a worker has linked it.
        if (const VkPipeline optimized = optimized_pipeline.load(std::memory_order_acquire)) {
            return optimized;
        }
        return *pipeline;
    }

    vk::PipelineLayout GetLayout() const noexcept {
        return *pipeline_layout;
    }

    auto GetStages() const {
        static_assert(static_cast<u32>(Shader::SwStage::Compute) == Shader::MaxStageTypes - 1);
        if (is_compute) {
            return std::span{stages.cend() - 1, stages.cend()};
        } else {
            return std::span{stages.cbegin(), stages.cend() - 1};
        }
    }

    const Shader::Info& GetStage(Shader::SwStage stage) const noexcept {
        return *stages[u32(stage)];
    }

    /// PERF-047: the pipeline cache's infos the pipeline was made from (stages may point at the
    /// GPU recorder's copies of them while a draw is recorded).
    const std::array<const Shader::Info*, Shader::MaxStageTypes>& CanonicalStages() const {
        return canonical_stages;
    }
    /// PERF-047: the infos the draw being recorded binds through (the recorder thread only).
    void SetDrawStages(const std::array<const Shader::Info*, Shader::MaxStageTypes>& infos) const {
        stages = infos;
    }

    bool IsCompute() const {
        return is_compute;
    }

    /// NAT-100 / SHD-002: how the pipeline takes its descriptors (native mode binds them
    /// itself): one set per stage (Shader::Backend::StageSet), one of them pushed.
    static constexpr u32 NoPushSet = ~0u;
    u32 PushSet() const {
        return push_set;
    }
    vk::DescriptorSetLayout SetLayout(u32 set) const {
        return set < set_layouts.size() ? set_layouts[set] : vk::DescriptorSetLayout{};
    }

    using DescriptorWrites = std::vector<vk::WriteDescriptorSet>;
    /// SHD-002: the descriptor set of each write.
    using WriteSets = std::vector<u8>;
    void BindResources(DescriptorWrites& set_writes, const WriteSets& write_sets,
                       const Shader::PushData& push_data) const;

protected:
    [[nodiscard]] std::string GetDebugString() const;

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    std::atomic<VkPipeline> optimized_pipeline{};
    vk::UniquePipelineLayout pipeline_layout;
    /// SHD-002: the layouts of the stage sets (set_layouts also holds the empty layout of sets
    /// between them).
    std::array<vk::UniqueDescriptorSetLayout, Shader::Backend::MaxDescriptorSets> desc_layouts;
    std::array<vk::DescriptorSetLayout, Shader::Backend::MaxDescriptorSets> set_layouts{};
    mutable std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    std::array<const Shader::Info*, Shader::MaxStageTypes> canonical_stages{};
    u32 push_set = NoPushSet;
    bool is_compute;
};

} // namespace Vulkan
