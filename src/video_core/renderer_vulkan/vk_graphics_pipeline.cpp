// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "common/thread.h"
#include "shader_recompiler/backend/spirv/emit_spirv_discard_frag.h"
#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

using Shader::Backend::SPIRV::AuxShaderType;

static constexpr std::array LogicalStageToStageBit = {
    vk::ShaderStageFlagBits::eFragment,
    vk::ShaderStageFlagBits::eTessellationControl,
    vk::ShaderStageFlagBits::eTessellationEvaluation,
    vk::ShaderStageFlagBits::eVertex,
    vk::ShaderStageFlagBits::eGeometry,
    vk::ShaderStageFlagBits::eCompute,
};

// PERF-021: time of the last pipeline a draw needed that did not exist yet.
static std::atomic<s64> last_pipeline_miss_ms{};

static s64 SteadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void GraphicsPipeline::NotePipelineMiss() {
    last_pipeline_miss_ms.store(SteadyMs(), std::memory_order_relaxed);
}

// PERF-017: workers that link fully optimized pipelines from stage libraries, so the command
// thread only waits for the fast link.
class PipelineLinkWorkers {
public:
    static PipelineLinkWorkers& Instance() {
        static PipelineLinkWorkers workers;
        return workers;
    }

    void Push(std::function<void()>&& job) {
        {
            std::scoped_lock lk{mutex};
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
    }

private:
    PipelineLinkWorkers() {
        const u32 count = std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u);
        for (u32 i = 0; i < count; ++i) {
            threads.emplace_back([this](std::stop_token stop) {
                Common::SetCurrentThreadName("shadPS4:PipelineLink");
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
            });
        }
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::function<void()>> jobs;
    std::vector<std::jthread> threads;
};

// PERF-018: shader stage libraries (pre-rasterization and fragment shader) of runtime
// pipelines, shared by every pipeline whose stages, layout and stage state are the same. Many new
// pipelines differ from an earlier one only in blending, targets or vertex input, so they reuse
// both compiled shader libraries and only link. Libraries live until exit.
class StageLibraryCache {
public:
    static StageLibraryCache& Instance() {
        static StageLibraryCache cache;
        return cache;
    }

    vk::Pipeline Find(u64 key) {
        std::scoped_lock lk{mutex};
        const auto it = libraries.find(key);
        return it != libraries.end() ? it->second : vk::Pipeline{};
    }

    void Insert(u64 key, vk::Pipeline library) {
        std::scoped_lock lk{mutex};
        libraries.emplace(key, library);
    }

private:
    std::mutex mutex;
    std::unordered_map<u64, vk::Pipeline> libraries;
};

class LibraryKey {
public:
    explicit LibraryKey(u64 kind) {
        Add(kind);
    }
    template <typename T>
    void Add(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* bytes = reinterpret_cast<const u8*>(&value);
        data.insert(data.end(), bytes, bytes + sizeof(T));
    }
    u64 Hash() const {
        return XXH3_64bits(data.data(), data.size());
    }

private:
    std::vector<u8> data;
};

GraphicsPipeline::GraphicsPipeline(
    const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
    const Shader::Profile& profile, const GraphicsPipelineKey& key_,
    vk::PipelineCache pipeline_cache, std::span<const Shader::Info*, MaxShaderStages> infos,
    std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
    const Shader::Gcn::FetchShaderData* fetch_shader_, std::span<const vk::ShaderModule> modules,
    SerializationSupport& sdata, bool preloading)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache}, key{key_} {
    if (fetch_shader_) {
        fetch_shader = *fetch_shader_;
    }

    const vk::Device device = instance.GetDevice();
    std::ranges::copy(infos, stages.begin());
    BuildDescSetLayout(preloading);
    const auto debug_str = GetDebugString();

    const vk::PushConstantRange push_constants = {
        .stageFlags = AllGraphicsStageBits,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create graphics pipeline layout: {}", vk::to_string(layout_result));
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "Graphics PipelineLayout {}", debug_str);

    if (!preloading) {
        VertexInputs<AmdGpu::Buffer> guest_buffers;
        if (!instance.IsVertexInputDynamicState()) {
            const auto& vs_info = runtime_infos[u32(Shader::SwStage::Vertex)].sw.vs;
            GetVertexInputs(sdata.vertex_attributes, sdata.vertex_bindings, sdata.divisors,
                            guest_buffers, vs_info.step_rate_0, vs_info.step_rate_1);
        }
    }

    const vk::PipelineVertexInputDivisorStateCreateInfo divisor_state = {
        .vertexBindingDivisorCount = static_cast<u32>(sdata.divisors.size()),
        .pVertexBindingDivisors = sdata.divisors.data(),
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .pNext = sdata.divisors.empty() ? nullptr : &divisor_state,
        .vertexBindingDescriptionCount = static_cast<u32>(sdata.vertex_bindings.size()),
        .pVertexBindingDescriptions = sdata.vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(sdata.vertex_attributes.size()),
        .pVertexAttributeDescriptions = sdata.vertex_attributes.data(),
    };

    const auto topology = LiverpoolToVK::PrimitiveType(key.prim_type);
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = topology,
    };

    const bool is_rect_list = key.prim_type == AmdGpu::PrimitiveType::RectList;
    const bool is_quad_list = key.prim_type == AmdGpu::PrimitiveType::QuadList;
    const vk::PipelineTessellationStateCreateInfo tessellation_state = {
        .patchControlPoints = is_rect_list ? 3U : (is_quad_list ? 4U : key.patch_control_points),
    };

    vk::StructureChain raster_chain = {
        vk::PipelineRasterizationStateCreateInfo{
            .depthClampEnable = key.depth_clamp_enable &&
                                (!key.depth_clip_enable || instance.IsDepthClipEnableSupported()),
            .rasterizerDiscardEnable = false,
            .polygonMode = LiverpoolToVK::PolygonMode(key.polygon_mode),
            .lineWidth = 1.0f,
        },
        vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT{
            .provokingVertexMode = key.provoking_vtx_last == AmdGpu::ProvokingVtxLast::First
                                       ? vk::ProvokingVertexModeEXT::eFirstVertex
                                       : vk::ProvokingVertexModeEXT::eLastVertex,
        },
        vk::PipelineRasterizationDepthClipStateCreateInfoEXT{
            .depthClipEnable = key.depth_clip_enable,
        },
    };

    if (!instance.IsProvokingVertexSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT>();
    }
    if (!instance.IsDepthClipEnableSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationDepthClipStateCreateInfoEXT>();
    }

    if (!preloading) {
        const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
        sdata.multisampling = {
            .rasterizationSamples = LiverpoolToVK::NumSamples(
                key.num_samples, instance.GetColorSampleCounts() & instance.GetDepthSampleCounts()),
            .sampleShadingEnable =
                fs_info.addr_flags.persp_sample_ena || fs_info.addr_flags.linear_sample_ena,
        };
    }

    const vk::PipelineViewportDepthClipControlCreateInfoEXT clip_control = {
        .negativeOneToOne = key.clip_space == AmdGpu::ClipSpace::MinusWToW,
    };

    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .pNext = instance.IsDepthClipControlSupported() ? &clip_control : nullptr,
    };

    boost::container::static_vector<vk::DynamicState, 32> dynamic_states = {
        vk::DynamicState::eViewportWithCount,  vk::DynamicState::eScissorWithCount,
        vk::DynamicState::eBlendConstants,     vk::DynamicState::eDepthTestEnable,
        vk::DynamicState::eDepthWriteEnable,   vk::DynamicState::eDepthCompareOp,
        vk::DynamicState::eDepthBiasEnable,    vk::DynamicState::eDepthBias,
        vk::DynamicState::eStencilTestEnable,  vk::DynamicState::eStencilReference,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilOp,          vk::DynamicState::eCullMode,
        vk::DynamicState::eFrontFace,          vk::DynamicState::eRasterizerDiscardEnable,
        vk::DynamicState::eLineWidth,          vk::DynamicState::ePrimitiveRestartEnable,
    };

    if (instance.IsDepthBoundsSupported()) {
        dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
        dynamic_states.push_back(vk::DynamicState::eDepthBounds);
    }
    if (instance.IsDynamicColorWriteMaskSupported()) {
        dynamic_states.push_back(vk::DynamicState::eColorWriteMaskEXT);
    }
    if (instance.IsVertexInputDynamicState()) {
        dynamic_states.push_back(vk::DynamicState::eVertexInputEXT);
    } else if (!sdata.vertex_bindings.empty()) {
        dynamic_states.push_back(vk::DynamicState::eVertexInputBindingStride);
    }

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages>
        shader_stages;
    auto stage = u32(Shader::SwStage::Vertex);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::Geometry);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eGeometry,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::TessellationControl);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        const auto type = is_quad_list ? AuxShaderType::QuadListTCS : AuxShaderType::RectListTCS;
        if (!preloading) {
            const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
            sdata.tcs = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
                type, fs_info, *infos[u32(Shader::SwStage::Vertex)],
                infos[u32(Shader::SwStage::Fragment)]);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = CompileSPV(sdata.tcs, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::TessellationEval);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        if (!preloading) {
            const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
            sdata.tes = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
                AuxShaderType::PassthroughTES, fs_info, *infos[u32(Shader::SwStage::Vertex)],
                infos[u32(Shader::SwStage::Fragment)]);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = CompileSPV(sdata.tes, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::Fragment);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs.clip_distance_emulation) {
        if (!preloading) {
            const auto& vs = runtime_infos[static_cast<u32>(Shader::SwStage::Vertex)].hw.vs;

            sdata.fragment = Shader::Backend::SPIRV::EmitDiscardFragmentShader(vs.outputs);
        }
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = CompileSPV(sdata.fragment, instance.GetDevice()),
            .pName = "main",
        });
    }

    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    for (auto& stage : shader_stages) {
        stage.pNext = instance.IsSubgroupSize64Supported() ? &subgroup_size_ci : nullptr;
    }

    const auto depth_format =
        instance.GetSupportedFormat(LiverpoolToVK::DepthFormat(key.z_format, key.stencil_format),
                                    vk::FormatFeatureFlagBits2::eDepthStencilAttachment);
    std::array<vk::Format, Shader::IR::NumRenderTargets> color_formats;
    for (s32 i = 0; i < key.num_color_attachments; ++i) {
        const auto& col_buf = key.color_buffers[i];
        const auto format = LiverpoolToVK::SurfaceFormat(col_buf.data_format, col_buf.num_format);
        const auto color_format =
            instance.GetSupportedFormat(format, vk::FormatFeatureFlagBits2::eColorAttachment);
        if (!instance.IsFormatSupported(color_format,
                                        vk::FormatFeatureFlagBits2::eColorAttachment)) {
            LOG_WARNING(Render_Vulkan,
                        "color buffer format {} does not support COLOR_ATTACHMENT_BIT",
                        vk::to_string(color_format));
        }
        color_formats[i] = color_format;
    }

    std::array<vk::SampleCountFlagBits, AmdGpu::NUM_COLOR_BUFFERS> color_samples;
    std::ranges::transform(key.color_samples, color_samples.begin(), [&instance](u8 num_samples) {
        return num_samples ? LiverpoolToVK::NumSamples(num_samples, instance.GetColorSampleCounts())
                           : vk::SampleCountFlagBits::e1;
    });
    const vk::AttachmentSampleCountInfoAMD mixed_samples = {
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentSamples = color_samples.data(),
        .depthStencilAttachmentSamples =
            LiverpoolToVK::NumSamples(key.depth_samples, instance.GetDepthSampleCounts()),
    };

    const vk::PipelineRenderingCreateInfo pipeline_rendering_ci = {
        .pNext = instance.IsMixedDepthSamplesSupported() ? &mixed_samples : nullptr,
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat = key.z_format != AmdGpu::DepthBuffer::ZFormat::Invalid
                                     ? depth_format
                                     : vk::Format::eUndefined,
        .stencilAttachmentFormat = key.stencil_format != AmdGpu::DepthBuffer::StencilFormat::Invalid
                                       ? depth_format
                                       : vk::Format::eUndefined,
    };

    std::array<vk::PipelineColorBlendAttachmentState, AmdGpu::NUM_COLOR_BUFFERS> attachments;
    for (u32 i = 0; i < key.num_color_attachments; i++) {
        const auto& control = key.blend_controls[i];

        const auto src_color = LiverpoolToVK::BlendFactor(control.color_src_factor);
        const auto dst_color = LiverpoolToVK::BlendFactor(control.color_dst_factor);
        const auto color_blend = LiverpoolToVK::BlendOp(control.color_func);

        const auto src_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_src_factor)
                                   : src_color;
        const auto dst_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_dst_factor)
                                   : dst_color;
        const auto alpha_blend =
            control.separate_alpha_blend ? LiverpoolToVK::BlendOp(control.alpha_func) : color_blend;

        // Vulkan ignores blend factors for min/max, but a factor that zeroes one operand
        // makes the operation collapse to a plain selection: min(s, 0) is 0 and max(s, 0)
        // is s for normalized alpha. Rewrite those to the equivalent add so the result is
        // exact instead of leaving the other operand to survive.
        auto eff_src_alpha = src_alpha;
        auto eff_dst_alpha = dst_alpha;
        auto eff_alpha_blend = alpha_blend;
        if (alpha_blend == vk::BlendOp::eMin || alpha_blend == vk::BlendOp::eMax) {
            const bool takes_max = alpha_blend == vk::BlendOp::eMax;
            if (src_alpha == vk::BlendFactor::eOne && dst_alpha == vk::BlendFactor::eZero) {
                eff_alpha_blend = vk::BlendOp::eAdd;
                eff_src_alpha = takes_max ? vk::BlendFactor::eOne : vk::BlendFactor::eZero;
                eff_dst_alpha = vk::BlendFactor::eZero;
            } else if (src_alpha == vk::BlendFactor::eZero && dst_alpha == vk::BlendFactor::eOne) {
                eff_alpha_blend = vk::BlendOp::eAdd;
                eff_src_alpha = vk::BlendFactor::eZero;
                eff_dst_alpha = takes_max ? vk::BlendFactor::eOne : vk::BlendFactor::eZero;
            }
        }

        const auto color_scaled_min_max =
            (color_blend == vk::BlendOp::eMin || color_blend == vk::BlendOp::eMax) &&
            (src_color != vk::BlendFactor::eOne || dst_color != vk::BlendFactor::eOne) &&
            !key.color_buffers[i].blend_self_scale;
        const auto alpha_scaled_min_max =
            (eff_alpha_blend == vk::BlendOp::eMin || eff_alpha_blend == vk::BlendOp::eMax) &&
            (eff_src_alpha != vk::BlendFactor::eOne || eff_dst_alpha != vk::BlendFactor::eOne);
        if (color_scaled_min_max || alpha_scaled_min_max) {
            LOG_WARNING(
                Render_Vulkan,
                "Unimplemented use of min/max blend op with blend factor not equal to one.");
        }

        attachments[i] = vk::PipelineColorBlendAttachmentState{
            .blendEnable = control.enable,
            .srcColorBlendFactor = src_color,
            .dstColorBlendFactor = dst_color,
            .colorBlendOp = color_blend,
            .srcAlphaBlendFactor = eff_src_alpha,
            .dstAlphaBlendFactor = eff_dst_alpha,
            .alphaBlendOp = eff_alpha_blend,
            .colorWriteMask =
                instance.IsDynamicColorWriteMaskSupported()
                    ? vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                    : key.write_masks[i],
        };

        // The shader squares its color output for this attachment (see PsColorBuffer), so the
        // factors must not scale the operands again.
        if (key.color_buffers[i].blend_self_scale) {
            LOG_WARNING(
                Render_Vulkan,
                "Emulating scaled min/max blend with squared shader output on attachment {}", i);
            attachments[i].srcColorBlendFactor = vk::BlendFactor::eOne;
            attachments[i].dstColorBlendFactor = vk::BlendFactor::eOne;
        }

        // On GCN GPU there is an additional mask which allows to control color components exported
        // from a pixel shader. A situation possible, when the game may mask out the alpha channel,
        // while it is still need to be used in blending ops. For such cases, HW will default alpha
        // to 1 and perform the blending, while shader normally outputs 0 in the last component.
        // Unfortunatelly, Vulkan doesn't provide any control on blend inputs, so below we detecting
        // such cases and override alpha value in order to emulate HW behaviour.
        const auto has_alpha_masked_out =
            (key.cb_shader_mask.GetMask(i) & AmdGpu::ColorBufferMask::ComponentA) == 0;
        const auto has_src_alpha_in_src_blend = src_color == vk::BlendFactor::eSrcAlpha ||
                                                src_color == vk::BlendFactor::eOneMinusSrcAlpha;
        const auto has_src_alpha_in_dst_blend = dst_color == vk::BlendFactor::eSrcAlpha ||
                                                dst_color == vk::BlendFactor::eOneMinusSrcAlpha;
        if (has_alpha_masked_out && has_src_alpha_in_src_blend) {
            attachments[i].srcColorBlendFactor = src_color == vk::BlendFactor::eSrcAlpha
                                                     ? vk::BlendFactor::eOne
                                                     : vk::BlendFactor::eZero; // 1-A
        }
        if (has_alpha_masked_out && has_src_alpha_in_dst_blend) {
            attachments[i].dstColorBlendFactor = dst_color == vk::BlendFactor::eSrcAlpha
                                                     ? vk::BlendFactor::eOne
                                                     : vk::BlendFactor::eZero; // 1-A
        }

        if (key.color_buffers[i].blend_swizzled_alpha) {
            LiverpoolToVK::SetSwizzledAlphaBlend(attachments[i]);
            static const bool logged = [] {
                LOG_INFO(Render_Vulkan, "Using dual-source swizzled source-alpha blend emulation");
                return true;
            }();
        } else if (key.color_buffers[i].blend_swizzled_factors) {
            // Both functions match for this emulation.
            LiverpoolToVK::SetSwizzledFactorBlend(attachments[i], control.color_func);
            static const bool logged = [] {
                LOG_INFO(Render_Vulkan, "Using dual-source swizzled factor blend emulation");
                return true;
            }();
        }
    }

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable =
            instance.IsLogicOpSupported() && key.logic_op != AmdGpu::ColorControl::LogicOp::Copy,
        .logicOp = LiverpoolToVK::LogicOp(key.logic_op),
        .attachmentCount = key.num_color_attachments,
        .pAttachments = attachments.data(),
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    // Required by spec unless VK_EXT_extended_dynamic_state3 is supported.
    // In practice, we use dynamic state for all of it.
    constexpr vk::PipelineDepthStencilStateCreateInfo depth_stencil_info = {};

    const vk::GraphicsPipelineCreateInfo pipeline_info = {
        .pNext = &pipeline_rendering_ci,
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = !instance.IsVertexInputDynamicState() ? &vertex_input_info : nullptr,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = &tessellation_state,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_chain.get(),
        .pMultisampleState = &sdata.multisampling,
        .pDepthStencilState =
            !instance.IsExtendedDynamicState3Supported() ? &depth_stencil_info : nullptr,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = *pipeline_layout,
    };

    const auto create_start = std::chrono::steady_clock::now();
    if (!preloading && instance.IsGraphicsPipelineLibrarySupported()) {
        // PERF-017: a pipeline first needed during play is built from four separately compiled
        // libraries and linked without cross-stage optimization, which the driver does in a
        // fraction of a full compile. A worker then links the same libraries with link-time
        // optimization, and draws switch to that pipeline once it exists. Both pipelines come
        // from the same create info, so they render the same.
        library_link = std::make_unique<LibraryLink>();
        boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages>
            pre_raster_stages;
        boost::container::static_vector<vk::PipelineShaderStageCreateInfo, 1> fragment_stages;
        for (const auto& stage_info : shader_stages) {
            if (stage_info.stage == vk::ShaderStageFlagBits::eFragment) {
                fragment_stages.push_back(stage_info);
            } else {
                pre_raster_stages.push_back(stage_info);
            }
        }
        const std::array<vk::GraphicsPipelineLibraryFlagsEXT, 4> parts = {
            vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface,
            vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders,
            vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader,
            vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface,
        };
        std::array<vk::GraphicsPipelineLibraryCreateInfoEXT, 4> library_infos{};
        std::array<vk::GraphicsPipelineCreateInfo, 4> part_infos{};
        for (size_t i = 0; i < parts.size(); ++i) {
            library_infos[i] = vk::GraphicsPipelineLibraryCreateInfoEXT{
                .pNext = &pipeline_rendering_ci,
                .flags = parts[i],
            };
            auto& part_info = part_infos[i];
            part_info = pipeline_info;
            part_info.pNext = &library_infos[i];
            part_info.flags = vk::PipelineCreateFlagBits::eLibraryKHR |
                              vk::PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT;
            part_info.stageCount = 0;
            part_info.pStages = nullptr;
            if (i == 1) {
                part_info.stageCount = static_cast<u32>(pre_raster_stages.size());
                part_info.pStages = pre_raster_stages.data();
            } else if (i == 2) {
                part_info.stageCount = static_cast<u32>(fragment_stages.size());
                part_info.pStages = fragment_stages.data();
            }
        }
        const auto create_library = [&](size_t i) {
            auto [library_result, library] =
                device.createGraphicsPipeline(pipeline_cache, part_infos[i]);
            ASSERT_MSG(library_result == vk::Result::eSuccess,
                       "Failed to create graphics pipeline library: {}",
                       vk::to_string(library_result));
            return library;
        };

        // PERF-018: the shader libraries are keyed by everything their create info reads: the
        // layout's bindings, the stage modules, and the state of their pipeline subset.
        LibraryKey pre_raster_key{1};
        pre_raster_key.Add(desc_layout_hash);
        pre_raster_key.Add(uses_push_descriptors);
        for (const auto& stage_info : pre_raster_stages) {
            pre_raster_key.Add(stage_info.stage);
            pre_raster_key.Add(static_cast<VkShaderModule>(stage_info.module));
        }
        pre_raster_key.Add(tessellation_state.patchControlPoints);
        pre_raster_key.Add(
            raster_chain.get<vk::PipelineRasterizationStateCreateInfo>().depthClampEnable);
        pre_raster_key.Add(key.polygon_mode);
        pre_raster_key.Add(key.provoking_vtx_last);
        pre_raster_key.Add(key.depth_clip_enable);
        pre_raster_key.Add(key.clip_space);
        for (const auto state : dynamic_states) {
            pre_raster_key.Add(state);
        }

        LibraryKey fragment_key{2};
        fragment_key.Add(desc_layout_hash);
        fragment_key.Add(uses_push_descriptors);
        for (const auto& stage_info : fragment_stages) {
            fragment_key.Add(static_cast<VkShaderModule>(stage_info.module));
        }
        fragment_key.Add(sdata.multisampling.rasterizationSamples);
        fragment_key.Add(sdata.multisampling.sampleShadingEnable);
        fragment_key.Add(sdata.multisampling.minSampleShading);
        fragment_key.Add(pipeline_rendering_ci.colorAttachmentCount);
        for (u32 i = 0; i < pipeline_rendering_ci.colorAttachmentCount; ++i) {
            fragment_key.Add(color_formats[i]);
            fragment_key.Add(color_samples[i]);
        }
        fragment_key.Add(pipeline_rendering_ci.depthAttachmentFormat);
        fragment_key.Add(pipeline_rendering_ci.stencilAttachmentFormat);
        fragment_key.Add(mixed_samples.depthStencilAttachmentSamples);
        for (const auto state : dynamic_states) {
            fragment_key.Add(state);
        }

        auto& library_cache = StageLibraryCache::Instance();
        const u64 pre_raster_hash = pre_raster_key.Hash();
        const u64 fragment_hash = fragment_key.Hash();
        // -DisablePerf 18 compiles every library for every pipeline, one after another.
        const bool share_libraries = Common::PerfFeatureEnabled(18);
        vk::Pipeline pre_raster_library =
            share_libraries ? library_cache.Find(pre_raster_hash) : vk::Pipeline{};
        vk::Pipeline fragment_library =
            share_libraries ? library_cache.Find(fragment_hash) : vk::Pipeline{};
        const bool pre_raster_reused = bool(pre_raster_library);
        const bool fragment_reused = bool(fragment_library);
        // PERF-018: the two shader libraries are compiled at the same time, the fragment one
        // on another thread.
        std::future<vk::Pipeline> fragment_future;
        if (!fragment_library && share_libraries) {
            fragment_future = std::async(std::launch::async, create_library, size_t{2});
        }
        std::array<vk::Pipeline, 4> library_handles{};
        library_handles[0] = create_library(0);
        if (!pre_raster_library) {
            pre_raster_library = create_library(1);
            library_cache.Insert(pre_raster_hash, pre_raster_library);
        }
        library_handles[3] = create_library(3);
        if (fragment_future.valid()) {
            fragment_library = fragment_future.get();
        } else if (!fragment_library) {
            fragment_library = create_library(2);
        }
        if (!fragment_reused) {
            library_cache.Insert(fragment_hash, fragment_library);
        }
        library_handles[1] = pre_raster_library;
        library_handles[2] = fragment_library;
        library_link->owned_libraries[0] = vk::UniquePipeline{library_handles[0], device};
        library_link->owned_libraries[1] = vk::UniquePipeline{library_handles[3], device};
        const auto libraries_done = std::chrono::steady_clock::now();

        const vk::PipelineLibraryCreateInfoKHR link_libraries = {
            .libraryCount = static_cast<u32>(library_handles.size()),
            .pLibraries = library_handles.data(),
        };
        const vk::GraphicsPipelineCreateInfo link_info = {
            .pNext = &link_libraries,
            .layout = *pipeline_layout,
        };
        auto [link_result, linked] = device.createGraphicsPipelineUnique(pipeline_cache, link_info);
        ASSERT_MSG(link_result == vk::Result::eSuccess, "Failed to link graphics pipeline: {}",
                   vk::to_string(link_result));
        pipeline = std::move(linked);
        SetObjectName(device, *pipeline, "Graphics Pipeline {}", debug_str);
        LOG_WARNING(Render_Vulkan,
                    "Graphics pipeline {}: libraries {:.1f} ms (pre-raster {}, fragment {}), fast "
                    "link {:.1f} ms",
                    debug_str,
                    std::chrono::duration<double, std::milli>(libraries_done - create_start)
                        .count(),
                    pre_raster_reused ? "reused" : "compiled",
                    fragment_reused ? "reused" : "compiled",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              libraries_done)
                        .count());

        auto promise = std::make_shared<std::promise<void>>();
        library_link->optimized = promise->get_future();
        const auto layout = *pipeline_layout;
        PipelineLinkWorkers::Instance().Push(
            [this, device, layout, library_handles, promise, cache = pipeline_cache] {
                // PERF-021: the fast-linked pipeline renders the same, so the optimized one can
                // wait until new pipelines stop appearing (the worker only sleeps meanwhile).
                if (Common::PerfFeatureEnabled(21)) {
                    while (SteadyMs() - last_pipeline_miss_ms.load(std::memory_order_relaxed) <
                           3000) {
                        std::this_thread::sleep_for(std::chrono::milliseconds{100});
                    }
                }
                const vk::PipelineLibraryCreateInfoKHR libraries_info = {
                    .libraryCount = static_cast<u32>(library_handles.size()),
                    .pLibraries = library_handles.data(),
                };
                const vk::GraphicsPipelineCreateInfo optimized_info = {
                    .pNext = &libraries_info,
                    .flags = vk::PipelineCreateFlagBits::eLinkTimeOptimizationEXT,
                    .layout = layout,
                };
                auto [result, optimized] = device.createGraphicsPipeline(cache, optimized_info);
                if (result == vk::Result::eSuccess) {
                    optimized_pipeline.store(optimized, std::memory_order_release);
                }
                promise->set_value();
            });
        return;
    }

    auto [pipeline_result, pipe] =
        device.createGraphicsPipelineUnique(pipeline_cache, pipeline_info);
    // PERF-DIAG-013: driver compile time of graphics pipelines created at runtime.
    if (!preloading) {
        LOG_WARNING(Render_Vulkan, "Graphics pipeline {}: driver create {:.1f} ms", debug_str,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              create_start)
                        .count());
    }
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create graphics pipeline: {}",
               vk::to_string(pipeline_result));
    pipeline = std::move(pipe);
    SetObjectName(device, *pipeline, "Graphics Pipeline {}", debug_str);
}

GraphicsPipeline::~GraphicsPipeline() {
    if (library_link && library_link->optimized.valid()) {
        library_link->optimized.wait();
    }
    if (const VkPipeline optimized = optimized_pipeline.exchange(VK_NULL_HANDLE)) {
        instance.GetDevice().destroyPipeline(optimized);
    }
}

template <typename Attribute, typename Binding>
void GraphicsPipeline::GetVertexInputs(
    VertexInputs<Attribute>& attributes, VertexInputs<Binding>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const {
    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
    if (fetch_shader.Empty() || fetch_shader.attributes.empty()) {
        return;
    }
    const auto& vs_info = GetStage(Shader::SwStage::Vertex);
    for (u32 semantic = 0; semantic < fetch_shader.attributes.size(); ++semantic) {
        const auto& attrib = fetch_shader.attributes[semantic];
        const auto step_rate = attrib.GetStepRate();
        const auto buffer = attrib.GetSharp(vs_info);
        attributes.push_back(Attribute{
            .location = semantic,
            .binding = semantic,
            .format = LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt()),
            .offset = 0,
        });
        bindings.push_back(Binding{
            .binding = semantic,
            .stride = buffer.GetStride(),
            .inputRate = step_rate == InstanceIdType::None ? vk::VertexInputRate::eVertex
                                                           : vk::VertexInputRate::eInstance,
        });
        const u32 divisor = step_rate == InstanceIdType::OverStepRate0
                                ? step_rate_0
                                : (step_rate == InstanceIdType::OverStepRate1 ? step_rate_1 : 1);
        if constexpr (std::is_same_v<Binding, vk::VertexInputBindingDescription2EXT>) {
            bindings.back().divisor = divisor;
        } else if (step_rate != InstanceIdType::None) {
            divisors.push_back(vk::VertexInputBindingDivisorDescriptionEXT{
                .binding = semantic,
                .divisor = divisor,
            });
        }
        guest_buffers.emplace_back(buffer);
    }
}

// Declare templated GetVertexInputs for necessary types.
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription>& attributes,
    VertexInputs<vk::VertexInputBindingDescription>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const;
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription2EXT>& attributes,
    VertexInputs<vk::VertexInputBindingDescription2EXT>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const;

void GraphicsPipeline::BuildDescSetLayout(bool preloading) {
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    u32 binding{};

    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        const auto stage_bit = LogicalStageToStageBit[u32(stage->sw_stage)];
        for (const auto& buffer : stage->buffers) {
            const auto sharp =
                preloading ? AmdGpu::Buffer{}
                           : buffer.GetSharp(*stage); // See for the comment in compute PL creation
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
        for (const auto& image : stage->images) {
            const u32 num_bindings = image.NumBindings(*stage);
            bindings.push_back({
                .binding = binding,
                .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                                   : vk::DescriptorType::eSampledImage,
                .descriptorCount = num_bindings,
                .stageFlags = stage_bit,
            });
            binding += num_bindings;
        }
        for (const auto& sampler : stage->samplers) {
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eSampler,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
    }
    uses_push_descriptors = binding < instance.MaxPushDescriptors();
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    desc_layout_hash = XXH3_64bits(bindings.data(), bindings.size() * sizeof(bindings[0]));
    auto [layout_result, layout] =
        instance.GetDevice().createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create graphics descriptor set layout: {}", vk::to_string(layout_result));
    desc_layout = std::move(layout);
}

} // namespace Vulkan
