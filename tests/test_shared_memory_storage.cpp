// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>
#include "shader_recompiler/ir/passes/shared_memory_to_storage.h"

using Shader::BufferResource;
using Shader::BufferResourceList;
using Shader::BufferType;
using Shader::IR::Opcode;
using Shader::Optimization::HasSharedMemoryStorageBuffer;
using Shader::Optimization::SharedMemoryAccessIndexShift;

TEST(SharedMemoryStorage, UsesTypedElementIndicesForConvertedByteAddresses) {
    EXPECT_EQ(SharedMemoryAccessIndexShift(Opcode::LoadSharedU16), 1);
    EXPECT_EQ(SharedMemoryAccessIndexShift(Opcode::WriteSharedU16), 1);
    EXPECT_EQ(SharedMemoryAccessIndexShift(Opcode::LoadSharedU32), 2);
    EXPECT_EQ(SharedMemoryAccessIndexShift(Opcode::WriteSharedU32), 2);
    EXPECT_EQ(SharedMemoryAccessIndexShift(Opcode::SharedAtomicIAdd32), 2);
    EXPECT_EQ(SharedMemoryAccessIndexShift(Opcode::LoadSharedU64), 3);
    EXPECT_EQ(SharedMemoryAccessIndexShift(Opcode::WriteSharedU64), 3);
    EXPECT_EQ(SharedMemoryAccessIndexShift(Opcode::SharedAtomicCmpSwap64), 3);

    constexpr u32 byte_address = 0x120;
    EXPECT_EQ(byte_address >> SharedMemoryAccessIndexShift(Opcode::LoadSharedU16), 0x90);
    EXPECT_EQ(byte_address >> SharedMemoryAccessIndexShift(Opcode::LoadSharedU32), 0x48);
    EXPECT_EQ(byte_address >> SharedMemoryAccessIndexShift(Opcode::LoadSharedU64), 0x24);
}

TEST(SharedMemoryStorage, StorageBufferBarriersAreEnabledOnlyWhenLdsWasLowered) {
    BufferResourceList buffers;
    EXPECT_FALSE(HasSharedMemoryStorageBuffer(buffers));

    buffers.emplace_back(BufferResource{.buffer_type = BufferType::Guest});
    EXPECT_FALSE(HasSharedMemoryStorageBuffer(buffers));

    buffers.emplace_back(BufferResource{.buffer_type = BufferType::SharedMemory});
    EXPECT_TRUE(HasSharedMemoryStorageBuffer(buffers));
}
