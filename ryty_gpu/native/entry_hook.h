// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-001: hooks on the entry of functions in the converted game (GTSport.exe). The function's
// first instructions are moved to a trampoline and replaced by a jump, so every call reaches the
// hook, whichever call site or tail jump it comes from. The moved bytes must be whole,
// position-independent instructions that nothing branches into; the caller passes them and they
// are checked before anything is written.

#pragma once

#include <atomic>
#include <span>

#include "common/types.h"

namespace Native {

/// GTSport.exe RVA of an address in the game's eboot.bin (the converter moves the code by this).
constexpr u64 kEbootToExe = 0x10000;

/// Executable memory within +-2 GB of the game's code, so rel32 jumps reach it.
class NearCode {
public:
    explicit NearCode(u8* exe_base) : exe_base{exe_base} {}
    /// 64-byte aligned, zeroed, executable and writable; nullptr when nothing near is free.
    u8* Alloc(size_t size);

private:
    u8* exe_base;
    u8* block = nullptr;
    size_t used = 0;
};

/// Makes the function at eboot address `eboot_va` jump to `hook`. `prologue` is the function's
/// first instructions (at least 5 bytes) as the game has them. Returns a pointer that runs the
/// original function (the moved instructions, then a jump back), or nullptr when the bytes
/// differ or no memory was found. Must run before the game's code does (DllMain).
void* HookEntry(NearCode& code, u8* exe_base, u64 eboot_va, std::span<const u8> prologue,
                void* hook, const char* name);

/// Like HookEntry, but the hook is a machine-code stub that only increments the returned
/// counter and continues into the original function: no registers or arguments are touched.
std::atomic<u64>* CountEntry(NearCode& code, u8* exe_base, u64 eboot_va,
                             std::span<const u8> prologue, const char* name);

} // namespace Native
