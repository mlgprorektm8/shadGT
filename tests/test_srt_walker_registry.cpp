// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "shader_recompiler/ir/passes/srt.h"

// PERF-038: the background pipeline reader decoded shader metadata on its own thread, which
// registers SRT walker code, while the GPU thread did the same; the shared code buffer and its
// map were corrupted (Lance's 20:12 run: a reader spinning at 98% of a core, then a crash).
TEST(SrtWalkerRegistry, ConcurrentRegistrationKeepsEveryCopy) {
    constexpr u32 Threads = 4;
    constexpr u32 PerThread = 3000;
    std::atomic<u32> mismatches{};
    std::vector<std::thread> threads;
    for (u32 t = 0; t < Threads; ++t) {
        threads.emplace_back([&, t] {
            for (u32 i = 0; i < PerThread; ++i) {
                // ret, then the thread and index as padding nobody executes.
                std::array<u8, 12> code{0xC3};
                const u64 tag = (u64(t) << 32) | i;
                std::memcpy(code.data() + 4, &tag, sizeof(tag));
                const u64 key = tag + 1;
                const auto* func = reinterpret_cast<const u8*>(
                    Shader::RegisterWalkerCode(code.data(), code.size(), key));
                if (std::memcmp(func, code.data(), code.size()) != 0) {
                    ++mismatches;
                }
                // The same key and code again comes back as the copy registered first.
                const auto* again = reinterpret_cast<const u8*>(
                    Shader::RegisterWalkerCode(code.data(), code.size(), key));
                if (again != func) {
                    ++mismatches;
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(mismatches.load(), 0u);
}
