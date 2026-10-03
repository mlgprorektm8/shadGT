// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::DeviceService {

constexpr s32 ErrorInvalidArgument = static_cast<s32>(0x809b0080);
constexpr s32 ErrorAlreadyInitialized = static_cast<s32>(0x809b000b);

struct MemoryParam {
    void* buffer;
    u32 size;
    u32 reserved;
};
static_assert(sizeof(MemoryParam) == 16);

struct InitParam {
    u32 size;
    u32 reserved;
    const MemoryParam* memory_param;
};
static_assert(sizeof(InitParam) == 16);

s32 PS4_SYSV_ABI sceDeviceServiceInitialize(u32 flags, const InitParam* param);
s32 PS4_SYSV_ABI sceDeviceServiceTerminate();
s32 PS4_SYSV_ABI sceDeviceServiceGetEventState(s32 clear);
s32 PS4_SYSV_ABI sceDeviceServiceQueryDeviceInfo_(u32 type, const void* condition, s32 generation,
                                                  void* devices, s32 capacity, s32* count,
                                                  s32* total, u64 device_size);
void RegisterLib(Core::Loader::SymbolsResolver* sym);

} // namespace Libraries::DeviceService
