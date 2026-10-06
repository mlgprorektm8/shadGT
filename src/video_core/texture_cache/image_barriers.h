// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <vector>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {

struct ImageBarrierState {
    vk::PipelineStageFlags2 pl_stage = vk::PipelineStageFlagBits2::eAllCommands;
    vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eNone;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;

    bool operator==(const ImageBarrierState&) const = default;
};

/// Layout and access tracking of one physical image.
struct ImageBarrierTracking {
    /// Most recently requested state. Descriptors and attachments read its layout.
    ImageBarrierState state;
    /// Per-subresource states (level-major) after a transition of part of the image.
    std::vector<ImageBarrierState> subresource_states;
    /// When set, the image is partially tracked but every subresource has this state, without
    /// storing one entry per subresource. A partial transition that changes nothing records it
    /// instead of allocating and filling subresource_states.
    std::optional<ImageBarrierState> uniform_subresource_state;
};

/// Appends the barriers that move the image (or the given subresource range) to the destination
/// state and updates the tracking. A barrier on a subresource range is equivalent to the same
/// barrier on each subresource in it, so adjacent layers with identical old state share one.
template <typename Barriers>
void ComputeImageBarriers(Barriers& barriers, ImageBarrierTracking& tracking,
                          SubresourceExtent resources, vk::Image image,
                          vk::ImageAspectFlags aspect_mask, vk::ImageLayout dst_layout,
                          vk::AccessFlags2 dst_mask, vk::PipelineStageFlags2 dst_stage,
                          std::optional<SubresourceRange> subres_range) {
    auto& last_state = tracking.state;
    auto& subresource_states = tracking.subresource_states;
    auto& uniform_state = tracking.uniform_subresource_state;

    constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                 vk::AccessFlagBits2::eShaderWrite |
                                 vk::AccessFlagBits2::eMemoryWrite;
    const auto needs_barrier = [&](const ImageBarrierState& state) {
        return state.layout != dst_layout || state.access_mask != dst_mask ||
               static_cast<bool>(state.access_mask & write_flags);
    };
    const auto add_barrier = [&](const ImageBarrierState& state, u32 base_level, u32 levels,
                                 u32 base_layer, u32 layers) {
        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = state.pl_stage,
            .srcAccessMask = state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = state.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = base_level,
                .levelCount = levels,
                .baseArrayLayer = base_layer,
                .layerCount = layers,
            },
        });
    };
    const auto finish = [&] {
        last_state.layout = dst_layout;
        last_state.access_mask = dst_mask;
        last_state.pl_stage = dst_stage;
    };

    const bool needs_partial_transition =
        subres_range &&
        (subres_range->base != SubresourceBase{} || subres_range->extent != resources);
    const bool partially_transited = !subresource_states.empty() || uniform_state.has_value();

    if (!needs_partial_transition && !partially_transited) {
        // Full resource transition.
        if (!needs_barrier(last_state)) {
            return;
        }
        add_barrier(last_state, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
        finish();
        return;
    }

    if (subresource_states.empty()) {
        // Every subresource currently has the same state.
        const ImageBarrierState common = uniform_state.value_or(last_state);
        if (!needs_barrier(common)) {
            // No subresource changes. A partial transition leaves each subresource at the
            // common state (with its original stage); a full transition ends partial tracking.
            if (needs_partial_transition) {
                uniform_state = common;
            } else {
                uniform_state.reset();
            }
            finish();
            return;
        }
        if (!needs_partial_transition) {
            add_barrier(common, 0, resources.levels, 0, resources.layers);
            uniform_state.reset();
            finish();
            return;
        }
        subresource_states.assign(size_t(resources.levels) * resources.layers, common);
        uniform_state.reset();
    }

    // In case of partial transition, we need to change the specified subresources only.
    // Otherwise all subresources need to be set to the same state so we can use a full
    // resource transition for the next time.
    const u32 start_mip = needs_partial_transition ? subres_range->base.level : u16{0};
    const u32 end_mip = needs_partial_transition
                            ? subres_range->base.level + subres_range->extent.levels
                            : resources.levels;
    const u32 start_layer = needs_partial_transition ? subres_range->base.layer : u16{0};
    const u32 end_layer = needs_partial_transition
                              ? subres_range->base.layer + subres_range->extent.layers
                              : resources.layers;
    const auto state_index = [&](u32 mip, u32 layer) {
        const auto index = size_t(mip) * resources.layers + layer;
        ASSERT(index < subresource_states.size());
        return index;
    };
    for (u32 mip = start_mip; mip < end_mip; ++mip) {
        for (u32 layer = start_layer; layer < end_layer;) {
            const ImageBarrierState old_state = subresource_states[state_index(mip, layer)];
            if (!needs_barrier(old_state)) {
                ++layer;
                continue;
            }
            u32 run_end = layer + 1;
            while (run_end < end_layer &&
                   subresource_states[state_index(mip, run_end)] == old_state) {
                ++run_end;
            }
            add_barrier(old_state, mip, 1, layer, run_end - layer);
            for (; layer < run_end; ++layer) {
                auto& state = subresource_states[state_index(mip, layer)];
                state.layout = dst_layout;
                state.access_mask = dst_mask;
                state.pl_stage = dst_stage;
            }
        }
    }

    if (!needs_partial_transition) {
        subresource_states.clear();
    }
    finish();
}

} // namespace VideoCore
