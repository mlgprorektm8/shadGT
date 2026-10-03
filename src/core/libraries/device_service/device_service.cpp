// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include "core/libraries/device_service/device_service.h"
#include "core/libraries/libs.h"

namespace Libraries::DeviceService {

static std::atomic<bool> initialized{false};

s32 PS4_SYSV_ABI sceDeviceServiceInitialize(u32 flags, const InitParam* param) {
    // These flags and parameter fields are validated by the firmware's DeviceService export.
    if (flags != 0 && (flags & 0x33) == 0) {
        return ErrorInvalidArgument;
    }
    if (initialized.load()) {
        return ErrorAlreadyInitialized;
    }
    if (param && (param->size != sizeof(InitParam) || param->reserved != 0 ||
                  (param->memory_param && param->memory_param->reserved != 0))) {
        return ErrorInvalidArgument;
    }
    bool expected = false;
    if (!initialized.compare_exchange_strong(expected, true)) {
        return ErrorAlreadyInitialized;
    }
    return 0;
}

s32 PS4_SYSV_ABI sceDeviceServiceTerminate() {
    initialized.store(false);
    return 0;
}

s32 PS4_SYSV_ABI sceDeviceServiceGetEventState(s32 clear) {
    if (!initialized.load()) {
        return ErrorInvalidArgument;
    }
    // The host has no Mbus devices: no attach/detach events are pending. SDL pads use libScePad.
    return 0;
}

s32 PS4_SYSV_ABI sceDeviceServiceQueryDeviceInfo_(u32 type, const void* condition, s32 generation,
                                                  void* devices, s32 capacity, s32* count,
                                                  s32* total, u64 device_size) {
    if (!initialized.load() || condition || capacity < 0 || (capacity != 0 && !devices)) {
        return ErrorInvalidArgument;
    }
    switch (type) {
    case 0x1001:
    case 0x2001:
    case 0x3001:
    case 0x4001:
    case 0x4002:
    case 0x5001:
    case 0x6001:
    case 0x7001:
        break;
    default:
        return ErrorInvalidArgument;
    }
    // Empty enumeration must initialize the output counts; a generic success stub did not.
    // No record is returned, so the caller's record storage is deliberately left untouched.
    if (count) {
        *count = 0;
    }
    if (total) {
        *total = 0;
    }
    return 0;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("84fDxStrG44", "libSceDeviceService", 1, "libSceMbus", sceDeviceServiceInitialize);
    LIB_FUNCTION("Uq8uW74rVpU", "libSceDeviceService", 1, "libSceMbus", sceDeviceServiceTerminate);
    LIB_FUNCTION("9ddRUOV8Q5A", "libSceDeviceService", 1, "libSceMbus",
                 sceDeviceServiceGetEventState);
    LIB_FUNCTION("UNMEa+5lrUA", "libSceDeviceService", 1, "libSceMbus",
                 sceDeviceServiceQueryDeviceInfo_);
}

} // namespace Libraries::DeviceService
