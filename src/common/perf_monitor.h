// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

namespace Common {

/// PERF-DIAG-007: CPU time used by each thread of this process since the previous call,
/// as "name pct%" entries for the busiest threads (percent of one core), plus the process total.
/// Returns an empty string where unsupported.
std::string SampleThreadCpuUsage(size_t max_threads);

} // namespace Common
