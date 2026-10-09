// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <stdexcept>

namespace Common {

/// Thrown instead of stopping the emulator when an ASSERT or UNREACHABLE fails inside a
/// RecoverableScope on the same thread. The failed check is logged before it is thrown.
class RecoverableFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Number of RecoverableScopes active on this thread (see assert.cpp).
int& RecoverableDepth();

/// While alive, failed assertions on this thread throw RecoverableFailure. Use it only around
/// work whose failure can be contained, such as translating one shader: the caller catches the
/// failure, reports it, and continues without the result.
class RecoverableScope {
public:
    RecoverableScope() {
        ++RecoverableDepth();
    }
    ~RecoverableScope() {
        --RecoverableDepth();
    }
    RecoverableScope(const RecoverableScope&) = delete;
    RecoverableScope& operator=(const RecoverableScope&) = delete;
};

} // namespace Common
