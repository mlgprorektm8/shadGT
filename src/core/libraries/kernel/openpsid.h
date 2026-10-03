// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/types.h"

namespace Libraries::Kernel {

using OpenPsId = std::array<u8, 16>;
s32 PS4_SYSV_ABI sceKernelGetOpenPsId(OpenPsId* id);

} // namespace Libraries::Kernel
