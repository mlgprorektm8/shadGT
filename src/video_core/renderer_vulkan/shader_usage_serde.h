// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/serdes.h"
#include "shader_recompiler/info.h"

namespace Vulkan {

/// FIX-043's record of which bound-resource properties a translation read, stored after the
/// shader's Info from meta version 16. Without it a stored permutation, whose key leaves those
/// properties out, would never match the key rebuilt from the loaded shader.
inline void WriteResourceUsage(Serialization::Writer& out, const Shader::Info& info) {
    out.Write(u8{info.resource_usage_known});
    out.Write(u64{info.buffer_stride_used.to_ullong()});
    out.Write(u64{info.image_srgb_used.to_ullong()});
}

inline void ReadResourceUsage(Serialization::Reader& in, Shader::Info& info) {
    u8 known{};
    u64 buffers{};
    u64 images{};
    in.Read(known);
    in.Read(buffers);
    in.Read(images);
    info.resource_usage_known = known != 0;
    info.buffer_stride_used = std::bitset<Shader::NUM_BUFFERS>{buffers};
    info.image_srgb_used = std::bitset<Shader::NUM_IMAGES>{images};
}

} // namespace Vulkan
