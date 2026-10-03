// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace AmdGpu {

// WAIT_ON_DE_COUNTER_DIFF waits until CE's lead over DE is less than the limit.
constexpr bool ShouldWaitOnDeCounter(u32 ce_count, u32 de_count, u32 limit) {
    return u32(ce_count - de_count) >= limit;
}

} // namespace AmdGpu
