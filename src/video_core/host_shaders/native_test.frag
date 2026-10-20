// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-012: the native renderer backend's self test samples a small texture, tinted by a push
// constant (ryty_gpu/native/rhi/rhi_selftest.cpp).

#version 450

layout(set = 0, binding = 0) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;
layout(push_constant) uniform PushConstants {
    vec4 tint;
} pc;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main() {
    color = texture(sampler2D(tex, samp), uv) * pc.tint;
}
