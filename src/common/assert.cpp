// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstring>
#include <cstdio>
#include <thread>
#include "common/logging/log.h"
#include "common/arch.h"
#include "common/assert.h"
#include "common/recoverable.h"
#include "core/signals.h"
#include "emulator.h"
#if defined(RYTY_GPU) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(ARCH_X86_64)
#define Crash() __asm__ __volatile__("int $3")
#elif defined(ARCH_ARM64)
#define Crash() __asm__ __volatile__("brk 0")
#else
#error "Missing Crash() implementation for target CPU architecture."
#endif

namespace Common {
int& RecoverableDepth() {
    static thread_local int depth = 0;
    return depth;
}
} // namespace Common

void assert_fail_impl() {
    if (Common::RecoverableDepth() > 0) {
        // The failed check is already logged; the RecoverableScope's owner handles it.
        throw Common::RecoverableFailure("assertion failed (see the log line above)");
    }
    Core::Signals::Instance()->RemoveHandlers();
    // The log is written by an asynchronous sink: give it a moment, so the failed check reaches
    // shad_log.txt and not only the console (two October 9 crashes left no trace in the file).
    Common::Log::Flush();
#if defined(RYTY_GPU) && defined(_WIN32)
    // Under Ryty no debugger or crash dump follows; log the stack as module+offset instead.
    {
        void* frames[48];
        const USHORT count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
        for (USHORT i = 0; i < count; ++i) {
            HMODULE module = nullptr;
            char name[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   static_cast<LPCSTR>(frames[i]), &module)) {
                GetModuleFileNameA(module, name, sizeof(name));
            }
            const char* base = std::strrchr(name, '\\');
            std::fprintf(stderr, "  #%u %s+0x%llx\n", static_cast<unsigned>(i),
                         base ? base + 1 : name,
                         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(frames[i]) -
                             reinterpret_cast<uintptr_t>(module)));
        }
        std::fflush(stderr);
        Common::Log::Flush();
    }
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    Common::Log::Flush();
    Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    Crash();
}

[[noreturn]] void unreachable_impl() {
    assert_fail_impl();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
    LOG_CRITICAL(Debug, "Assertion Failed!\n{}", msg);
    assert_fail_impl();
}
