// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan {

/// What a fence (EOP/EOS label or interrupt, WRITE_DATA, RELEASE_MEM) found when it was processed.
struct FenceReadbackState {
    bool compute_queue;
    /// GPU-written images queued for readback to guest memory, of any size / 64 KB or more.
    bool queued_any;
    bool queued_large;
    /// Earlier deferred fences whose readbacks have not reached guest memory yet.
    bool in_flight;
    /// A deferred write to the same address is still pending (keeps that address's order).
    bool address_pending;
};

/// Whether a fence must be signaled only after readbacks complete instead of right away.
///
/// Hardware writes an EOP/EOS/RELEASE_MEM label (and raises its interrupt) after all earlier
/// work on the queue has finished and its writes are visible in memory. Readbacks of GPU-written
/// images to guest memory stand in for those writes here, so a fence must not become visible
/// before every readback issued before it has landed: the queued ones (taken with this fence)
/// and the in-flight ones (taken by an earlier deferred fence; deferred operations run in order,
/// so deferring behind them is enough).
///
/// order_all (FIX-041, perf id 41): the hardware rule above, for both queues and any size.
/// Without it, the previous rules apply: graphics fences wait for queued readbacks; compute
/// fences only for queued ones of 64 KB or more (FIX-032, perf id 40: wait_large); neither waits
/// for in-flight readbacks.
constexpr bool ShouldDeferFence(const FenceReadbackState& s, bool order_all, bool wait_large) {
    if (s.address_pending) {
        return true;
    }
    if (order_all) {
        return s.queued_any || s.in_flight;
    }
    return s.compute_queue ? wait_large && s.queued_large : s.queued_any;
}

} // namespace Vulkan
