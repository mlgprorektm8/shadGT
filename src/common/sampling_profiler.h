// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "common/types.h"

namespace Common::SamplingProfiler {

/// DIAG-048: with SHADGT_PROFILE=1, threads registered here are sampled about 1,000 times per
/// second (suspended briefly, their stack unwound a few frames). Every 10 s the log lists, per
/// thread, the code addresses with the most samples: where the thread was (self) and every
/// function on its stack (inclusive), as module + offset, to be resolved offline with the
/// build's PDB. Windows only; does nothing elsewhere or without the variable.
void RegisterCurrentThread(const char* name);

/// DIAG-048: a game thread, running on the guest stack [stack_low, stack_high). Its samples show
/// where it waits: the emulated library function on its stack (guest code itself has no unwind
/// data, so the walk stops at the first guest frame and guest addresses are logged raw).
/// Null bounds: the thread runs on its host stack (the game's main thread).
void RegisterGuestThread(const char* name, const void* stack_low, const void* stack_high);

/// A code address as "module+offset" (Windows), or hex.
std::string DescribeCode(u64 address);

/// The calling thread's return addresses (up to 24), each as " module+offset" (Windows only).
std::string DescribeStack();

} // namespace Common::SamplingProfiler
