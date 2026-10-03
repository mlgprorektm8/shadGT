// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <fstream>
#include <mutex>
#include "common/path_util.h"
#include "core/libraries/kernel/openpsid.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/uuid.h"

namespace Libraries::Kernel {

s32 PS4_SYSV_ABI sceKernelGetOpenPsId(OpenPsId* id) {
    // The firmware export returns positive POSIX errors, rather than SCE kernel errors.
    if (!id) {
        return POSIX_EINVAL;
    }
    static std::mutex mutex;
    const std::scoped_lock lock{mutex};
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "openpsid.bin";
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        return POSIX_EIO;
    }
    OpenPsId value{};
    if (exists) {
        std::ifstream input(path, std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(value.data()), value.size()) ||
            input.peek() != std::char_traits<char>::eof()) {
            return POSIX_EIO;
        }
    } else {
        OrbisKernelUuid uuid{};
        const s32 result = sceKernelUuidCreate(&uuid);
        if (result != ORBIS_OK) {
            return result - ORBIS_KERNEL_ERROR_UNKNOWN;
        }
        std::memcpy(value.data(), &uuid, value.size());
        // Persist a virtual console identity shared by games in this user profile.
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(value.data()), value.size());
        output.close();
        if (!output) {
            return POSIX_EIO;
        }
    }
    *id = value;
    return ORBIS_OK;
}

} // namespace Libraries::Kernel
