// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include "common/path_util.h"
#include "core/libraries/kernel/openpsid.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/uuid.h"

static std::filesystem::path profile;
static u32 generated_ids;
static s32 generation_result;

const std::filesystem::path& Common::FS::GetUserPath(PathType) {
    return profile;
}

s32 PS4_SYSV_ABI Libraries::Kernel::sceKernelUuidCreate(OrbisKernelUuid* uuid) {
    ++generated_ids;
    *uuid = {0x11223344, 0x5566, 0x7788, 0x99, 0xaa, {1, 2, 3, 4, 5, 6}};
    return generation_result;
}

class OpenPsIdTest : public testing::Test {
    void SetUp() override {
        profile = std::filesystem::current_path() /
                  ("openpsid-test-" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(profile);
        generated_ids = 0;
        generation_result = ORBIS_OK;
    }
    void TearDown() override {
        std::filesystem::remove(profile / "openpsid.bin");
        std::filesystem::remove(profile);
    }
};

TEST_F(OpenPsIdTest, PersistsAndReturnsCompleteIdentityAcrossCalls) {
    Libraries::Kernel::OpenPsId first{};
    ASSERT_EQ(Libraries::Kernel::sceKernelGetOpenPsId(&first), 0);
    EXPECT_EQ(generated_ids, 1);
    EXPECT_EQ(std::filesystem::file_size(profile / "openpsid.bin"), 16);
    EXPECT_NE(first, Libraries::Kernel::OpenPsId{});
    Libraries::Kernel::OpenPsId second{};
    ASSERT_EQ(Libraries::Kernel::sceKernelGetOpenPsId(&second), 0);
    EXPECT_EQ(first, second);
    EXPECT_EQ(generated_ids, 1);
}

TEST_F(OpenPsIdTest, NullOutputReturnsPositivePosixErrorWithoutCreatingIdentity) {
    EXPECT_EQ(Libraries::Kernel::sceKernelGetOpenPsId(nullptr), POSIX_EINVAL);
    EXPECT_FALSE(std::filesystem::exists(profile / "openpsid.bin"));
    EXPECT_EQ(generated_ids, 0);
}

TEST_F(OpenPsIdTest, CorruptIdentityReturnsErrorWithoutChangingCallerBuffer) {
    std::ofstream(profile / "openpsid.bin", std::ios::binary) << "bad";
    Libraries::Kernel::OpenPsId output;
    output.fill(0xa5);
    const auto original = output;
    EXPECT_EQ(Libraries::Kernel::sceKernelGetOpenPsId(&output), POSIX_EIO);
    EXPECT_EQ(output, original);
    EXPECT_EQ(generated_ids, 0);
}

TEST_F(OpenPsIdTest, UuidFailureReturnsPosixErrorWithoutChangingCallerBuffer) {
    generation_result = ORBIS_KERNEL_ERROR_EFAULT;
    Libraries::Kernel::OpenPsId output;
    output.fill(0xa5);
    const auto original = output;
    EXPECT_EQ(Libraries::Kernel::sceKernelGetOpenPsId(&output), POSIX_EFAULT);
    EXPECT_EQ(output, original);
    EXPECT_FALSE(std::filesystem::exists(profile / "openpsid.bin"));
}
