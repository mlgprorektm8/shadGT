// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>
#include "core/libraries/device_service/device_service.h"
#include "core/loader/symbols_resolver.h"

// Registration is not exercised by these API behavior tests.
void LinkSymbolImpl(Core::Loader::SymbolsResolver*, const char*, const char*, u16, const char*, u64,
                    Core::Loader::SymbolType) {}

using namespace Libraries::DeviceService;

class DeviceServiceTest : public testing::Test {
    void SetUp() override {
        sceDeviceServiceTerminate();
    }
    void TearDown() override {
        sceDeviceServiceTerminate();
    }
};

TEST_F(DeviceServiceTest, InitializationValidatesParametersAndLifecycle) {
    EXPECT_EQ(sceDeviceServiceInitialize(4, nullptr), ErrorInvalidArgument);
    const InitParam invalid_size{8, 0, nullptr};
    EXPECT_EQ(sceDeviceServiceInitialize(3, &invalid_size), ErrorInvalidArgument);
    const InitParam invalid_reserved{16, 1, nullptr};
    EXPECT_EQ(sceDeviceServiceInitialize(3, &invalid_reserved), ErrorInvalidArgument);
    const MemoryParam invalid_memory{nullptr, 0, 1};
    const InitParam invalid_memory_param{16, 0, &invalid_memory};
    EXPECT_EQ(sceDeviceServiceInitialize(3, &invalid_memory_param), ErrorInvalidArgument);
    const InitParam valid{16, 0, nullptr};
    EXPECT_EQ(sceDeviceServiceInitialize(3, &valid), 0);
    EXPECT_EQ(sceDeviceServiceInitialize(3, nullptr), ErrorAlreadyInitialized);
    EXPECT_EQ(sceDeviceServiceInitialize(3, &invalid_size), ErrorAlreadyInitialized);
    EXPECT_EQ(sceDeviceServiceTerminate(), 0);
    EXPECT_EQ(sceDeviceServiceInitialize(0, nullptr), 0);
}

TEST_F(DeviceServiceTest, EmptyPeripheralEnumerationInitializesCountsWithoutOverwritingRecords) {
    ASSERT_EQ(sceDeviceServiceInitialize(3, nullptr), 0);
    std::array<u8, 0x70> records;
    records.fill(0xa5);
    s32 count = -1;
    s32 total = -1;
    EXPECT_EQ(sceDeviceServiceQueryDeviceInfo_(0x7001, nullptr, 0, records.data(), 1, &count,
                                               &total, records.size()),
              0);
    EXPECT_EQ(count, 0);
    EXPECT_EQ(total, 0);
    for (auto byte : records) {
        EXPECT_EQ(byte, 0xa5);
    }
    EXPECT_EQ(sceDeviceServiceGetEventState(0), 0);
    EXPECT_EQ(sceDeviceServiceGetEventState(1), 0);
}

TEST_F(DeviceServiceTest, RejectsInvalidQueriesWithoutChangingOutputs) {
    s32 count = 123;
    s32 total = 456;
    EXPECT_EQ(sceDeviceServiceQueryDeviceInfo_(0x7001, nullptr, 0, nullptr, 0, &count, &total, 0),
              ErrorInvalidArgument);
    ASSERT_EQ(sceDeviceServiceInitialize(3, nullptr), 0);
    EXPECT_EQ(sceDeviceServiceQueryDeviceInfo_(0x7001, nullptr, 0, nullptr, 1, &count, &total, 0),
              ErrorInvalidArgument);
    EXPECT_EQ(sceDeviceServiceQueryDeviceInfo_(0xffff, nullptr, 0, nullptr, 0, &count, &total, 0),
              ErrorInvalidArgument);
    EXPECT_EQ(sceDeviceServiceQueryDeviceInfo_(0, nullptr, 0, nullptr, 0, &count, &total, 0),
              ErrorInvalidArgument);
    EXPECT_EQ(count, 123);
    EXPECT_EQ(total, 456);
}

TEST_F(DeviceServiceTest, AllowsOptionalOutputCountsForEmptyQuery) {
    ASSERT_EQ(sceDeviceServiceInitialize(3, nullptr), 0);
    EXPECT_EQ(sceDeviceServiceQueryDeviceInfo_(0x7001, nullptr, 0, nullptr, 0, nullptr, nullptr, 0),
              0);
}
