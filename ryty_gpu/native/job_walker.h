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
    // NAT-005: where a failed walk stopped, and the chunks it went through.
    const u32* stop = nullptr;
    u32 stop_header = 0;
    const char* reason = nullptr;
    std::array<std::pair<const u32*, const u32*>, 16> chunks{}; // (begin, limit)
    u32 num_chunks = 0;
    u32 forks = 0;        // NAT-006: children skipped
    u32 clear_states = 0; // CLEAR_STATE packets (registers back to their defaults)
};

/// A walk through one job's commands that stops and goes on (NAT-006). A job hands work to a child
/// job by forking: PGL ends the job's command run with a 16-byte link slot that will lead into the
/// child's commands, and the job's own commands go on after it (or in a fresh chunk), linked after
/// the child's last chunk. The GPU runs the child in between, but the child is recorded on another
/// thread and may not be finished, so the walk skips it and goes on with the job's own commands.
class JobCursor {
public:
    /// Starts at the job's first chunk; false (with `stats` saying why) if it can't be read.
    bool Start(const u32* begin, const u32* first_limit, WalkStats& stats);
    /// Reads the job's commands up to `until` (the write pointer at a fork or at the job's end).
    bool WalkTo(const u32* until, AmdGpu::Regs& regs, JobExtents& extents, JobVisitor& visitor,
                WalkStats& stats);
    /// The job forked at the point the walk reached; its commands go on at `next`, in command
    /// space ending at `next_limit` (the command buffer's end pointer after the fork).
    void SkipFork(const u32* next, const u32* next_limit, WalkStats& stats);
    bool Ok() const {
        return ok;
    }
    const u32* Position() const {
        return p;
    }
    const u32* Limit() const {
        return limit;
    }

private:
    bool Fail(bool overrun, const char* reason, WalkStats& stats);

    const u32* p = nullptr;
    const u32* limit = nullptr; // the walk stays inside the chunk it is in
    bool ok = false;
};

/// Walks the job's commands; `regs` holds the state at the start and the state at the end after.
void WalkJob(const JobRange& job, AmdGpu::Regs& regs, JobExtents& extents, JobVisitor& visitor,
             WalkStats& stats);

/// Whether CPU code may read the range (guards the walk against bad links).
bool Readable(const void* address, size_t size);
/// NAT-106: how many bytes from the start of the range the CPU can read.
u64 ReadablePrefix(const void* address, u64 size);

} // namespace Native
