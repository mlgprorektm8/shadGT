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

/// Whether render jobs are walked on their threads (shadow mode, or OPT-010 native prepare mode:
/// RYGT_NATIVE_PREPARE=1).
bool ShadowEnabled();
/// Whether the command processor compares its state with the jobs' (RYGT_NATIVE_SHADOW=1).
bool ShadowComparing();

/// A render job began on this thread; `dcb` is its Gnm DrawCommandBuffer (fresh command chunk).
void ShadowJobBegin(const u8* dcb);
/// NAT-006: the PGL context current on this thread forked a child job (call after the fork):
/// the parent's commands stopped at the link slot `slot` (its write pointer before the fork).
/// `run_begin` and `run_limit` are the parent's command buffer begin and end pointers before
/// the fork; `caller` is the eboot address the fork was called from (for diagnostics).
void ShadowFork(const u8* parent_context, const u32* slot, const u32* run_begin,
                const u32* run_limit, const u8* child_context, u64 caller);
/// The job current on this thread is ending: call before PGL links its last chunk (its commands
/// end at the command buffer's write pointer), and ShadowJobDecode after.
void ShadowJobEnd(const u8* current_context);
void ShadowJobDecode();

/// Command processor: the registers at the start of a graphics submission (the base state jobs
/// start from), and at each draw packet.
void ShadowSubmitBase(const AmdGpu::Regs& regs);
void ShadowCheckDraw(const void* packet, const AmdGpu::Regs& regs);
/// NAT-030: right after ShadowCheckDraw, the pipeline the command processor selected for that
/// draw (nullptr if none; `cp_selected` false if it did not select one ahead).
void ShadowCheckPipeline(const void* cp_pipeline, bool cp_selected);
/// NAT-032: on the recorder, the sharps hash of the draw at `packet` as it binds it.
void ShadowCheckSharps(const void* packet, u64 hash);

} // namespace Native
