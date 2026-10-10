// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// DIAG-056: NVIDIA Nsight Systems timeline ranges and marks. Built in only with the CMake option
// SHADGT_NVTX_INCLUDE (the nvtx3 headers of an Nsight Systems install); otherwise every call is
// empty. With the headers built in but no Nsight capture running, each call returns at once.

#include <string>
#include "common/types.h"

#ifdef SHADGT_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace Common::Nvtx {

#ifdef SHADGT_NVTX
constexpr bool Enabled = true;

inline void Mark(const char* name) {
    nvtxMarkA(name);
}
inline void Mark(const std::string& name) {
    nvtxMarkA(name.c_str());
}
/// A range that may end on another thread.
inline u64 Start(const char* name) {
    return nvtxRangeStartA(name);
}
inline u64 Start(const std::string& name) {
    return nvtxRangeStartA(name.c_str());
}
inline void End(u64 id) {
    nvtxRangeEnd(id);
}
inline void Push(const char* name) {
    nvtxRangePushA(name);
}
inline void Pop() {
    nvtxRangePop();
}
#else
constexpr bool Enabled = false;

inline void Mark(const char*) {}
inline void Mark(const std::string&) {}
inline u64 Start(const char*) {
    return 0;
}
inline u64 Start(const std::string&) {
    return 0;
}
inline void End(u64) {}
inline void Push(const char*) {}
inline void Pop() {}
#endif

/// A range on the current thread for the scope.
class Scope {
public:
    explicit Scope(const char* name) {
        Push(name);
    }
    ~Scope() {
        Pop();
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

} // namespace Common::Nvtx
