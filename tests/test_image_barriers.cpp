// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <random>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "common/logging/log.h"
#include "video_core/texture_cache/image_barriers.h"

namespace {

using VideoCore::ImageBarrierState;
using VideoCore::ImageBarrierTracking;
using VideoCore::SubresourceExtent;
using VideoCore::SubresourceRange;
using Barriers = std::vector<vk::ImageMemoryBarrier2>;

const vk::Image TestImage = reinterpret_cast<VkImage>(0x1234);

struct ReferenceTracking {
    ImageBarrierState state;
    std::vector<ImageBarrierState> subresource_states;
};

// The barrier algorithm used before the lazy uniform state, kept as the equivalence reference.
void ReferenceBarriers(Barriers& barriers, ReferenceTracking& tracking,
                       SubresourceExtent resources, vk::ImageLayout dst_layout,
                       vk::AccessFlags2 dst_mask, vk::PipelineStageFlags2 dst_stage,
                       std::optional<SubresourceRange> subres_range) {
    auto& last_state = tracking.state;
    auto& subresource_states = tracking.subresource_states;
    const bool needs_partial_transition =
        subres_range && (subres_range->base != VideoCore::SubresourceBase{} ||
                         subres_range->extent != resources);
    const bool partially_transited = !subresource_states.empty();
    constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                 vk::AccessFlagBits2::eShaderWrite |
                                 vk::AccessFlagBits2::eMemoryWrite;
    if (needs_partial_transition || partially_transited) {
        if (!partially_transited) {
            subresource_states.resize(resources.levels * resources.layers);
            std::fill(subresource_states.begin(), subresource_states.end(), last_state);
        }
        const u32 start_mip = needs_partial_transition ? subres_range->base.level : u16{0};
        const u32 end_mip = needs_partial_transition
                                ? subres_range->base.level + subres_range->extent.levels
                                : resources.levels;
        const u32 start_layer = needs_partial_transition ? subres_range->base.layer : u16{0};
        const u32 end_layer = needs_partial_transition
                                  ? subres_range->base.layer + subres_range->extent.layers
                                  : resources.layers;
        for (u32 mip = start_mip; mip < end_mip; ++mip) {
            for (u32 layer = start_layer; layer < end_layer; ++layer) {
                auto& state = subresource_states[mip * resources.layers + layer];
                const bool is_write = static_cast<bool>(state.access_mask & write_flags);
                if (state.layout != dst_layout || state.access_mask != dst_mask || is_write) {
                    barriers.emplace_back(vk::ImageMemoryBarrier2{
                        .srcStageMask = state.pl_stage,
                        .srcAccessMask = state.access_mask,
                        .dstStageMask = dst_stage,
                        .dstAccessMask = dst_mask,
                        .oldLayout = state.layout,
                        .newLayout = dst_layout,
                        .image = TestImage,
                        .subresourceRange{vk::ImageAspectFlagBits::eColor, mip, 1, layer, 1},
                    });
                    state.layout = dst_layout;
                    state.access_mask = dst_mask;
                    state.pl_stage = dst_stage;
                }
            }
        }
        if (!needs_partial_transition) {
            subresource_states.clear();
        }
    } else {
        const bool is_write = static_cast<bool>(last_state.access_mask & write_flags);
        if (last_state.layout == dst_layout && last_state.access_mask == dst_mask && !is_write) {
            return;
        }
        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = last_state.pl_stage,
            .srcAccessMask = last_state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = last_state.layout,
            .newLayout = dst_layout,
            .image = TestImage,
            .subresourceRange{vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0,
                              VK_REMAINING_ARRAY_LAYERS},
        });
    }
    last_state.layout = dst_layout;
    last_state.access_mask = dst_mask;
    last_state.pl_stage = dst_stage;
}

using SubresourceBarrier = std::tuple<u32, u32, u64, u64, u64, u64, u32, u32>;

// Expands every barrier to the individual subresources it covers.
std::vector<SubresourceBarrier> Expand(const Barriers& barriers, SubresourceExtent resources) {
    std::vector<SubresourceBarrier> result;
    for (const auto& barrier : barriers) {
        const auto& range = barrier.subresourceRange;
        const u32 levels = range.levelCount == VK_REMAINING_MIP_LEVELS
                               ? resources.levels - range.baseMipLevel
                               : range.levelCount;
        const u32 layers = range.layerCount == VK_REMAINING_ARRAY_LAYERS
                               ? resources.layers - range.baseArrayLayer
                               : range.layerCount;
        EXPECT_EQ(barrier.image, TestImage);
        for (u32 mip = range.baseMipLevel; mip < range.baseMipLevel + levels; ++mip) {
            for (u32 layer = range.baseArrayLayer; layer < range.baseArrayLayer + layers;
                 ++layer) {
                result.emplace_back(mip, layer, u64(barrier.srcStageMask),
                                    u64(barrier.srcAccessMask), u64(barrier.dstStageMask),
                                    u64(barrier.dstAccessMask), u32(barrier.oldLayout),
                                    u32(barrier.newLayout));
            }
        }
    }
    std::ranges::sort(result);
    return result;
}

ImageBarrierState LogicalState(const ReferenceTracking& tracking, SubresourceExtent resources,
                               u32 mip, u32 layer) {
    return tracking.subresource_states.empty()
               ? tracking.state
               : tracking.subresource_states[mip * resources.layers + layer];
}

ImageBarrierState LogicalState(const ImageBarrierTracking& tracking, SubresourceExtent resources,
                               u32 mip, u32 layer) {
    if (!tracking.subresource_states.empty()) {
        return tracking.subresource_states[mip * resources.layers + layer];
    }
    return tracking.uniform_subresource_state.value_or(tracking.state);
}

struct Transition {
    vk::ImageLayout layout;
    vk::AccessFlags2 access;
    vk::PipelineStageFlags2 stage;
};

// States the renderer uses: sampled, storage, attachment, feedback, and transfer.
const std::array Transitions{
    Transition{vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
               vk::PipelineStageFlagBits2::eAllCommands},
    Transition{vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
               vk::PipelineStageFlagBits2::eAllGraphics},
    Transition{vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
               vk::PipelineStageFlagBits2::eComputeShader},
    Transition{vk::ImageLayout::eGeneral,
               vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
               vk::PipelineStageFlagBits2::eAllCommands},
    Transition{vk::ImageLayout::eColorAttachmentOptimal,
               vk::AccessFlagBits2::eColorAttachmentWrite |
                   vk::AccessFlagBits2::eColorAttachmentRead,
               vk::PipelineStageFlagBits2::eColorAttachmentOutput},
    Transition{vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
               vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eColorAttachmentRead |
                   vk::AccessFlagBits2::eColorAttachmentWrite,
               vk::PipelineStageFlagBits2::eAllGraphics |
                   vk::PipelineStageFlagBits2::eColorAttachmentOutput},
    Transition{vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
               vk::PipelineStageFlagBits2::eCopy},
    Transition{vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
               vk::PipelineStageFlagBits2::eCopy},
};

} // namespace

std::array<Common::Log::Level, Common::Log::NUM_LOG_CLASSES> Common::Log::g_class_levels{};

void Common::Log::VLog(Class, Level, const char* file, int line, const char*, fmt::string_view,
                       fmt::format_args) {
    ADD_FAILURE() << "Unexpected image-barrier diagnostic at " << file << ':' << line;
}

void assert_fail_impl() {
    std::abort();
}

[[noreturn]] void unreachable_impl() {
    std::abort();
}

TEST(ImageBarriers, MatchesReferenceForRandomTransitionSequences) {
    std::mt19937 random{20261005};
    const std::array extents{SubresourceExtent{1, 1}, SubresourceExtent{1, 6},
                             SubresourceExtent{5, 1}, SubresourceExtent{3, 4},
                             SubresourceExtent{4, 9}};
    for (const auto resources : extents) {
        for (u32 sequence = 0; sequence < 300; ++sequence) {
            ReferenceTracking reference;
            ImageBarrierTracking tracking;
            for (u32 step = 0; step < 40; ++step) {
                const auto& transition = Transitions[random() % Transitions.size()];
                std::optional<SubresourceRange> range;
                switch (random() % 4) {
                case 0:
                    break; // Full transition.
                case 1:
                    range = SubresourceRange{{}, resources}; // Explicit full range.
                    break;
                default: {
                    const u16 base_level = u16(random() % resources.levels);
                    const u16 base_layer = u16(random() % resources.layers);
                    const u16 levels = u16(1 + random() % (resources.levels - base_level));
                    const u16 layers = u16(1 + random() % (resources.layers - base_layer));
                    range = SubresourceRange{{base_level, base_layer}, {levels, layers}};
                    break;
                }
                }
                Barriers expected;
                Barriers actual;
                ReferenceBarriers(expected, reference, resources, transition.layout,
                                  transition.access, transition.stage, range);
                VideoCore::ComputeImageBarriers(actual, tracking, resources, TestImage,
                                                vk::ImageAspectFlagBits::eColor,
                                                transition.layout, transition.access,
                                                transition.stage, range);
                ASSERT_EQ(Expand(actual, resources), Expand(expected, resources))
                    << "extent=" << resources.levels << 'x' << resources.layers
                    << " sequence=" << sequence << " step=" << step;
                ASSERT_LE(actual.size(), expected.size());
                ASSERT_EQ(tracking.state, reference.state);
                for (u32 mip = 0; mip < resources.levels; ++mip) {
                    for (u32 layer = 0; layer < resources.layers; ++layer) {
                        ASSERT_EQ(LogicalState(tracking, resources, mip, layer),
                                  LogicalState(reference, resources, mip, layer))
                            << "sequence=" << sequence << " step=" << step << " mip=" << mip
                            << " layer=" << layer;
                    }
                }
            }
        }
    }
}

TEST(ImageBarriers, UnchangedPartialTransitionsKeepPerSubresourceStateUnallocated) {
    const SubresourceExtent resources{10, 512};
    ImageBarrierTracking tracking;
    Barriers barriers;
    const auto sampled = vk::ImageLayout::eShaderReadOnlyOptimal;
    const auto read = vk::AccessFlagBits2::eShaderRead;
    VideoCore::ComputeImageBarriers(barriers, tracking, resources, TestImage,
                                    vk::ImageAspectFlagBits::eColor, sampled, read,
                                    vk::PipelineStageFlagBits2::eAllCommands, std::nullopt);
    ASSERT_EQ(barriers.size(), 1U);
    barriers.clear();
    // The per-draw pattern: a view-range transition, then a full final-layout transition.
    for (u32 draw = 0; draw < 100; ++draw) {
        VideoCore::ComputeImageBarriers(
            barriers, tracking, resources, TestImage, vk::ImageAspectFlagBits::eColor, sampled,
            read, vk::PipelineStageFlagBits2::eAllCommands, SubresourceRange{{0, 0}, {9, 512}});
        EXPECT_TRUE(tracking.subresource_states.empty());
        VideoCore::ComputeImageBarriers(barriers, tracking, resources, TestImage,
                                        vk::ImageAspectFlagBits::eColor, sampled, read,
                                        vk::PipelineStageFlagBits2::eAllGraphics, std::nullopt);
        EXPECT_FALSE(tracking.uniform_subresource_state.has_value());
    }
    EXPECT_TRUE(barriers.empty());
}

TEST(ImageBarriers, AdjacentLayersShareOneBarrierPerLevel) {
    const SubresourceExtent resources{3, 512};
    ImageBarrierTracking tracking;
    Barriers barriers;
    VideoCore::ComputeImageBarriers(barriers, tracking, resources, TestImage,
                                    vk::ImageAspectFlagBits::eColor,
                                    vk::ImageLayout::eShaderReadOnlyOptimal,
                                    vk::AccessFlagBits2::eShaderRead,
                                    vk::PipelineStageFlagBits2::eAllCommands,
                                    SubresourceRange{{0, 0}, {2, 512}});
    // One barrier per transitioned level instead of one per level and layer.
    EXPECT_EQ(barriers.size(), 2U);
    barriers.clear();
    VideoCore::ComputeImageBarriers(barriers, tracking, resources, TestImage,
                                    vk::ImageAspectFlagBits::eColor,
                                    vk::ImageLayout::eShaderReadOnlyOptimal,
                                    vk::AccessFlagBits2::eShaderRead,
                                    vk::PipelineStageFlagBits2::eAllCommands, std::nullopt);
    EXPECT_EQ(barriers.size(), 1U);
    EXPECT_EQ(barriers[0].subresourceRange.baseMipLevel, 2U);
    EXPECT_EQ(barriers[0].subresourceRange.layerCount, 512U);
    EXPECT_TRUE(tracking.subresource_states.empty());
}
