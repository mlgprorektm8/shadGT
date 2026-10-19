// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-004: reads one GT Sport render job's commands on the thread that recorded it, keeping the
// GPU state the command processor would have at each packet. PGL links a job's command chunks
// with chained INDIRECT_BUFFER packets (written into a 16-byte slot at the end of the previous
// chunk when a chunk closes); the walk follows them and ends where the job's commands end.

#pragma once

#include <array>

#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/pm4_opcodes.h"

namespace AmdGpu {
union Regs;
}

namespace Native {

struct JobRange {
    const u32* begin = nullptr;       ///< first chunk (the job's command buffer at its start)
    const u32* first_limit = nullptr; ///< end of the first chunk's command space
    const u32* end = nullptr;         ///< where the job's commands end
};

/// The render target sizes hinted by the NOP after CB/DB setups (as the command processor
/// keeps them: the real size, smaller than pitch x height).
struct JobExtents {
    std::array<AmdGpu::CbDbExtent, 8> cb{};
    AmdGpu::CbDbExtent db{};
};

class JobVisitor {
public:
    virtual ~JobVisitor() = default;
    /// A draw packet, with the state at the draw (the packet's own fields applied).
    virtual void OnDraw(const u32* packet, AmdGpu::PM4ItOpcode opcode, const AmdGpu::Regs& regs,
                        const JobExtents& extents) = 0;
    /// A packet that is neither state nor a draw: cache flushes, events, waits, memory writes,
    /// dispatches. Native recording ends a segment of draws there.
    virtual void OnOther(const u32* packet, AmdGpu::PM4ItOpcode opcode) {}
};

struct WalkStats {
    u64 packets = 0;
    u64 draws = 0;
    u64 chain_links = 0;
    u64 call_links = 0;
    bool bad_packet = false;
    bool overrun = false;
};

/// Walks the job's commands; `regs` holds the state at the start and the state at the end after.
void WalkJob(const JobRange& job, AmdGpu::Regs& regs, JobExtents& extents, JobVisitor& visitor,
             WalkStats& stats);

/// Whether CPU code may read the range (guards the walk against bad links).
bool Readable(const void* address, size_t size);

} // namespace Native
