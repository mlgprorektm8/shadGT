// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Native {

/// NAT-002: hooks on GT Sport's (CUSA03220 v1.69) draw, dispatch and submit functions, when
/// RYGT_NATIVE_HOOKS=1. Called from DllMain, before the game's code runs.
void InstallGtsHooks(u8* exe_base);

} // namespace Native
