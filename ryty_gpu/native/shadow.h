// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-003: shadow mode for the native renderer (RYGT_NATIVE_SHADOW=1, implies the NAT-002 hooks).
// When a GT Sport render job ends, the thread that recorded it reads the job's own commands and
// works out the GPU state at each of its draws, as the native renderer will. The command processor
// still renders everything as before; at each draw it compares its own state with what the job
// worked out (matched by the draw packet's address) and NAT-003 lines report the differences.
// That shows whether jobs can be turned into GPU work on their own threads, each starting from a
// known base state, and which registers they inherit from earlier work if not.

#pragma once

#include "common/types.h"

namespace AmdGpu {
union Regs;
}

namespace Native {

bool ShadowEnabled();

/// A render job began on this thread; `dcb` is its Gnm DrawCommandBuffer (fresh command chunk).
void ShadowJobBegin(const u8* dcb);
/// The job current on this thread is ending: call before PGL links its last chunk (its commands
/// end at the command buffer's write pointer), and ShadowJobDecode after.
void ShadowJobEnd();
void ShadowJobDecode();

/// Command processor: the registers at the start of a graphics submission (the base state jobs
/// start from), and at each draw packet.
void ShadowSubmitBase(const AmdGpu::Regs& regs);
void ShadowCheckDraw(const void* packet, const AmdGpu::Regs& regs);

} // namespace Native
