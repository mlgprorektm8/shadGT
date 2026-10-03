// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::Kernel {

struct OrbisKernelUuid {
    u32 timeLow;
    u16 timeMid;
    u16 timeHiAndVersion;
    u8 clockSeqHiAndReserved;
    u8 clockSeqLow;
    u8 node[6];
};
static_assert(sizeof(OrbisKernelUuid) == 0x10);
s32 PS4_SYSV_ABI sceKernelUuidCreate(OrbisKernelUuid* uuid);

} // namespace Libraries::Kernel
