// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32
#include <windows.h>
#endif
#include <fmt/format.h>
#include "common/arch.h"
#include "common/assert.h"
#include "common/recoverable.h"
#include "core/signals.h"
#include "emulator.h"
#include "video_core/diag_bundle.h"

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

// The host call stack of a failed check, as module+offset frames (tools/mcp/symbolize.py
// resolves them with the PDB).
static void LogHostStack() {
#ifdef _WIN32
    void* frames[32];
    const USHORT count = RtlCaptureStackBackTrace(2, 32, frames, nullptr);
    std::string text;
    for (USHORT i = 0; i < count; ++i) {
        const auto address = reinterpret_cast<u64>(frames[i]);
        const auto host = Core::DescribeHostAddress(address);
        text += fmt::format(" | {}", host.empty() ? fmt::format("{:#x}", address) : host);
    }
    LOG_CRITICAL(Debug, "Host stack:{}", text);
#endif
}

void assert_fail_impl() {
    LogHostStack();
    if (Common::RecoverableDepth() > 0) {
        // The failed check is already logged; the RecoverableScope's owner handles it.
        throw Common::RecoverableFailure("assertion failed (see the log line above)");
    }
    VideoCore::DiagBundle::OnCrash("assertion");
    Core::Signals::Instance()->RemoveHandlers();
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
