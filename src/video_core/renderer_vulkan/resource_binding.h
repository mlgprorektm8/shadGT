// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/depth_attachment.h"

namespace Vulkan {

struct StageResourceBindings {
    Shader::Backend::Bindings buffers;
    Shader::Backend::Bindings textures;
    Shader::Backend::Bindings next;
};

inline StageResourceBindings PlanStageResourceBindings(const Shader::Info& stage,
                                                       Shader::Backend::Bindings start) {
    StageResourceBindings result{start, start, start};
    result.textures.buffer += stage.buffers.size();
    result.textures.unified += stage.buffers.size();
    stage.AddBindings(result.next);
    return result;
}

template <typename Func>
u32 ForEachImageDescriptorBinding(const Shader::Info& stage,
                                  const Shader::ImageResource& resource, Func&& bind) {
    const u32 count = resource.NumBindings(stage);
    for (u32 i = 0; i < count; ++i) {
        bind(resource.is_written);
    }
    return count;
}

struct ImageBindingRequirements {
    bool is_depth{};
    bool is_storage{};
    bool color_attachment{};
    bool depth_attachment{};
    bool has_stencil{};
    bool depth_write{};
    bool stencil_write{};
    bool feedback_layout_supported{};
    bool is_compute{};
};

struct ImageBindingState {
    vk::ImageLayout layout;
    vk::PipelineStageFlags2 stages;
    vk::AccessFlags2 access;
};

inline ImageBindingState FinalImageBindingState(const ImageBindingRequirements& requirements) {
    ImageBindingState result{
        .layout = requirements.is_depth ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                        : vk::ImageLayout::eShaderReadOnlyOptimal,
        .stages = requirements.is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics,
        .access = vk::AccessFlagBits2::eShaderRead,
    };
    if (requirements.color_attachment) {
        result.layout = requirements.feedback_layout_supported
                            ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                            : vk::ImageLayout::eGeneral;
        result.stages |= vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        result.access |=
            vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
    } else if (requirements.depth_attachment) {
        result.layout = DepthAttachmentLayout(requirements.has_stencil, requirements.depth_write,
                                              requirements.stencil_write, true,
                                              requirements.feedback_layout_supported);
        result.stages |= vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                         vk::PipelineStageFlagBits2::eLateFragmentTests;
        // Scheduler uses STORE for both aspects. Its late-fragment store writes must remain
        // tracked even when guest depth/stencil writes are disabled and the layout is read-only.
        result.access |= vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    }
    if (requirements.is_storage) {
        result.layout = vk::ImageLayout::eGeneral;
        result.access |= vk::AccessFlagBits2::eShaderWrite;
    }
    return result;
}

} // namespace Vulkan
