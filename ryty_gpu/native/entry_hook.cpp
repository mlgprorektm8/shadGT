// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include <cstring>
#include <new>
#include <windows.h>

#include "ryty_gpu/native/entry_hook.h"

namespace Native {

namespace {

constexpr size_t kBlockSize = 0x10000;

// jmp qword ptr [rip+0] followed by the absolute target: 14 bytes, reaches anywhere.
u8* EmitAbsoluteJump(u8* at, const void* target) {
    static constexpr u8 kJump[6] = {0xff, 0x25, 0, 0, 0, 0};
    std::memcpy(at, kJump, sizeof(kJump));
    std::memcpy(at + 6, &target, 8);
    return at + 14;
}

// The moved instructions, then a jump back behind them. A prologue that is a single rel32 jump (a
// thunk) becomes an absolute jump to its target instead.
u8* EmitMovedPrologue(u8* at, u8* function, std::span<const u8> prologue) {
    if (prologue.size() == 5 && prologue[0] == 0xe9) {
        s32 rel = 0;
        std::memcpy(&rel, prologue.data() + 1, 4);
        return EmitAbsoluteJump(at, function + 5 + rel);
    }
    std::memcpy(at, prologue.data(), prologue.size());
    return EmitAbsoluteJump(at + prologue.size(), function + prologue.size());
}

u8* CheckedFunction(u8* exe_base, u64 eboot_va, std::span<const u8> prologue, const char* name) {
    u8* function = exe_base + kEbootToExe + eboot_va;
    MEMORY_BASIC_INFORMATION info{};
    if (prologue.size() < 5 || !VirtualQuery(function, &info, sizeof(info)) ||
        info.State != MEM_COMMIT) {
        std::fprintf(stderr, "[native] %s: %#llx is not mapped\n", name,
                     static_cast<unsigned long long>(eboot_va));
        return nullptr;
    }
    if (std::memcmp(function, prologue.data(), prologue.size()) != 0) {
        std::fprintf(stderr, "[native] %s: code at %#llx differs, not hooked\n", name,
                     static_cast<unsigned long long>(eboot_va));
        return nullptr;
    }
    return function;
}

// Replaces the prologue with a rel32 jump to `target` (the rest of the moved bytes become nops).
bool PatchEntry(u8* function, size_t length, const u8* target, const char* name) {
    const s64 distance = target - (function + 5);
    if (distance != static_cast<s32>(distance)) {
        std::fprintf(stderr, "[native] %s: stub out of rel32 range\n", name);
        return false;
    }
    u8 bytes[16];
    bytes[0] = 0xe9;
    const s32 rel = static_cast<s32>(distance);
    std::memcpy(bytes + 1, &rel, 4);
    std::memset(bytes + 5, 0x90, length - 5);
    DWORD old_protect = 0;
    if (!VirtualProtect(function, length, PAGE_EXECUTE_READWRITE, &old_protect)) {
        return false;
    }
    std::memcpy(function, bytes, length);
    DWORD ignored = 0;
    VirtualProtect(function, length, old_protect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), function, length);
    return true;
}

} // namespace

u8* NearCode::Alloc(size_t size) {
    size = (size + 63) & ~size_t{63};
    if (block == nullptr || used + size > kBlockSize) {
        block = nullptr;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const u64 granularity = system.dwAllocationGranularity;
        const u64 origin = reinterpret_cast<u64>(exe_base);
        for (u64 distance = granularity; distance < (1ULL << 30) && !block;
             distance += granularity) {
            for (const s64 sign : {-1, 1}) {
                const u64 address = (origin + sign * static_cast<s64>(distance)) & ~(granularity - 1);
                if (void* memory = VirtualAlloc(reinterpret_cast<void*>(address), kBlockSize,
                                                MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE)) {
                    block = static_cast<u8*>(memory);
                    break;
                }
            }
        }
        used = 0;
        if (block == nullptr) {
            return nullptr;
        }
    }
    u8* memory = block + used;
    used += size;
    return memory;
}

void* HookEntry(NearCode& code, u8* exe_base, u64 eboot_va, std::span<const u8> prologue,
                void* hook, const char* name) {
    u8* function = CheckedFunction(exe_base, eboot_va, prologue, name);
    if (function == nullptr || prologue.size() > 16) {
        return nullptr;
    }
    u8* memory = code.Alloc(64);
    if (memory == nullptr) {
        std::fprintf(stderr, "[native] %s: no memory near the game's code\n", name);
        return nullptr;
    }
    u8* trampoline = memory;
    u8* jumper = EmitMovedPrologue(trampoline, function, prologue);
    EmitAbsoluteJump(jumper, hook);
    FlushInstructionCache(GetCurrentProcess(), memory, 64);
    if (!PatchEntry(function, prologue.size(), jumper, name)) {
        return nullptr;
    }
    return trampoline;
}

std::atomic<u64>* CountEntry(NearCode& code, u8* exe_base, u64 eboot_va,
                             std::span<const u8> prologue, const char* name) {
    u8* function = CheckedFunction(exe_base, eboot_va, prologue, name);
    if (function == nullptr || prologue.size() > 16) {
        return nullptr;
    }
    // The counter gets its own cache line; the stub follows it.
    u8* memory = code.Alloc(128);
    if (memory == nullptr) {
        std::fprintf(stderr, "[native] %s: no memory near the game's code\n", name);
        return nullptr;
    }
    auto* counter = new (memory) std::atomic<u64>{0};
    u8* stub = memory + 64;
    // lock inc qword ptr [rip + counter]
    static constexpr u8 kLockInc[4] = {0xf0, 0x48, 0xff, 0x05};
    std::memcpy(stub, kLockInc, sizeof(kLockInc));
    const s32 rel = static_cast<s32>(memory - (stub + 8));
    std::memcpy(stub + 4, &rel, 4);
    EmitMovedPrologue(stub + 8, function, prologue);
    FlushInstructionCache(GetCurrentProcess(), memory, 128);
    if (!PatchEntry(function, prologue.size(), stub, name)) {
        return nullptr;
    }
    return counter;
}

} // namespace Native
