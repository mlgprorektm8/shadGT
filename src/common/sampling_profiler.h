// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Common::SamplingProfiler {

/// DIAG-048: with SHADGT_PROFILE=1, threads registered here are sampled about 1,000 times per
/// second (suspended briefly, their stack unwound a few frames). Every 10 s the log lists, per
/// thread, the code addresses with the most samples: where the thread was (self) and every
/// function on its stack (inclusive), as module + offset, to be resolved offline with the
/// build's PDB. Windows only; does nothing elsewhere or without the variable.
void RegisterCurrentThread(const char* name);

} // namespace Common::SamplingProfiler
