// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "video_core/amdgpu/regs_depth.h"

namespace Vulkan {

struct StencilReference {
    u32 value;
    bool exact;
};

// Vulkan shares one reference between the stencil comparison and all replacement operations.
// GCN has separate test/op references and an operation that writes all ones.
constexpr StencilReference ResolveStencilReference(
    AmdGpu::StencilFunc fail, AmdGpu::StencilFunc pass, AmdGpu::StencilFunc depth_fail,
    AmdGpu::CompareFunc compare, const AmdGpu::StencilRefMask& ref,
    AmdGpu::CompareFunc depth_compare = AmdGpu::CompareFunc::Less) {
    const std::array ops{
        compare == AmdGpu::CompareFunc::Always ? AmdGpu::StencilFunc::Keep : fail,
        compare == AmdGpu::CompareFunc::Never || depth_compare == AmdGpu::CompareFunc::Never
            ? AmdGpu::StencilFunc::Keep
            : pass,
        compare == AmdGpu::CompareFunc::Never || depth_compare == AmdGpu::CompareFunc::Always
            ? AmdGpu::StencilFunc::Keep
            : depth_fail};
    u32 value = ref.stencil_test_val;
    for (const auto op : ops) {
        if (op == AmdGpu::StencilFunc::ReplaceOp) {
            value = ref.stencil_op_val;
        }
    }
    for (const auto op : ops) {
        if (op == AmdGpu::StencilFunc::Ones) {
            value = 0xff;
        }
    }
    bool exact = compare == AmdGpu::CompareFunc::Always || compare == AmdGpu::CompareFunc::Never ||
                 ((value ^ ref.stencil_test_val) & ref.stencil_mask) == 0;
    for (const auto op : ops) {
        if (op == AmdGpu::StencilFunc::ReplaceTest) {
            exact &= ((value ^ ref.stencil_test_val) & ref.stencil_write_mask) == 0;
        } else if (op == AmdGpu::StencilFunc::ReplaceOp) {
            exact &= ((value ^ ref.stencil_op_val) & ref.stencil_write_mask) == 0;
        }
    }
    return {value, exact};
}

} // namespace Vulkan
