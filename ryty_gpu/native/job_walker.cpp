// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <span>
#include <windows.h>

#include "ryty_gpu/native/job_walker.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/regs.h"

namespace Native {

using AmdGpu::PM4ItOpcode;
using AmdGpu::Regs;

namespace {

// Context registers whose writes carry a size hint in the NOP that follows them (the command
// processor's ContextRegs).
constexpr u32 kDbZInfo = 0xA010;
constexpr std::array<u32, 8> kCbColorBase{0xA318, 0xA327, 0xA336, 0xA345,
                                          0xA354, 0xA363, 0xA372, 0xA381};
constexpr std::array<u32, 8> kCbColorCmask{0xA31F, 0xA32E, 0xA33D, 0xA34C,
                                           0xA35B, 0xA36A, 0xA379, 0xA388};
constexpr u32 kNopHeader = 0xc0001000;

// The command processor's handling of those hints (liverpool.cpp, SetContextReg).
void NoteExtentHints(u32 reg_addr, u32 count_field, const u32* payload, JobExtents& extents) {
    for (u32 i = 0; i < 8; ++i) {
        if (reg_addr == kCbColorBase[i]) {
            const u32 nop_offset = count_field;
            extents.cb[i].raw = (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) &&
                                        payload[nop_offset] == kNopHeader
                                    ? payload[nop_offset + 1]
                                    : 0;
            return;
        }
        if (reg_addr == kCbColorCmask[i]) {
            if (count_field == 0x04 && payload[4] == kNopHeader) {
                extents.cb[i].raw = payload[5];
            }
            return;
        }
    }
    if (reg_addr == kDbZInfo) {
        extents.db.raw = count_field == 8 && payload[20] == kNopHeader ? payload[21] : 0;
    }
}

} // namespace

namespace {
// NAT-147: 64 KiB granules of the low 1 TiB known to be readable (committed, not no-access),
// so the recorder's checks don't call VirtualQuery per draw (with many pages protected one by
// one for write tracking, VirtualQuery walked them one region at a time: a quarter of the
// recorder's time). Only "readable" is remembered; unmapping forgets it (ForgetReadable).
constexpr u32 GranuleBits = 16;
constexpr u64 Granules = (u64{1} << 40) >> GranuleBits;
std::atomic<u64> readable_bits[Granules / 64];

bool GranuleKnown(u64 granule) {
    return granule < Granules &&
           (readable_bits[granule / 64].load(std::memory_order_relaxed) >> (granule % 64)) & 1;
}

bool QueryReadable(const u8* p, const u8* end) {
    while (p < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(p, &info, sizeof(info)) || info.State != MEM_COMMIT ||
            (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
            return false;
        }
        p = static_cast<const u8*>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}

// Whether the whole granule is readable (checked once, then remembered).
bool GranuleReadable(u64 granule) {
    if (GranuleKnown(granule)) {
        return true;
    }
    if (granule >= Granules) {
        return false;
    }
    const auto* begin = reinterpret_cast<const u8*>(granule << GranuleBits);
    if (!QueryReadable(begin, begin + (u64{1} << GranuleBits))) {
        return false;
    }
    readable_bits[granule / 64].fetch_or(u64{1} << (granule % 64), std::memory_order_relaxed);
    return true;
}
} // namespace

void ForgetReadable(u64 address, u64 size) {
    if (size == 0) {
        return;
    }
    for (u64 granule = address >> GranuleBits;
         granule <= (address + size - 1) >> GranuleBits && granule < Granules; ++granule) {
        readable_bits[granule / 64].fetch_and(~(u64{1} << (granule % 64)),
                                               std::memory_order_relaxed);
    }
}

u64 ReadablePrefix(const void* address, u64 size) {
    {
        // NAT-147: whole readable granules first.
        const u64 begin = reinterpret_cast<u64>(address);
        u64 granule = begin >> GranuleBits;
        const u64 last = (begin + std::max<u64>(size, 1) - 1) >> GranuleBits;
        while (granule <= last && GranuleReadable(granule)) {
            ++granule;
        }
        if (granule > last) {
            return size;
        }
    }
    const u8* const start = static_cast<const u8*>(address);
    const u8* p = start;
    const u8* const end = start + size;
    while (p < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(p, &info, sizeof(info)) || info.State != MEM_COMMIT ||
            (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
            break;
        }
        p = static_cast<const u8*>(info.BaseAddress) + info.RegionSize;
    }
    return static_cast<u64>(std::min(p, end) - start);
}

bool Readable(const void* address, size_t size) {
    {
        // NAT-147: every granule of the range known readable (or found readable now).
        const u64 begin = reinterpret_cast<u64>(address);
        const u64 last = (begin + std::max<u64>(size, 1) - 1) >> GranuleBits;
        bool all = true;
        for (u64 granule = begin >> GranuleBits; granule <= last; ++granule) {
            if (!GranuleReadable(granule)) {
                all = false;
                break;
            }
        }
        if (all) {
            return true;
        }
    }
    // The range may span several regions (separate mappings, or pages protected differently).
    const u8* p = static_cast<const u8*>(address);
    const u8* const end = p + size;
    while (p < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(p, &info, sizeof(info)) || info.State != MEM_COMMIT ||
            (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
            return false;
        }
        p = static_cast<const u8*>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}

namespace {

// CLEAR_STATE: the command processor resets every register to its default
// (AmdGpu::Regs::SetDefaults); the walk resets the graphics ones it keeps.
void ClearState(Regs& regs) {
    static const std::unique_ptr<Regs> defaults = [] {
        auto r = std::make_unique<Regs>();
        r->SetDefaults();
        return r;
    }();
    constexpr std::array<std::pair<u32, u32>, 3> kRanges = {{
        {Regs::ShRegWordOffset, 0x200},       // graphics stages (compute from 0x2E00 is apart)
        {Regs::ContextRegWordOffset, 0x400},  // context
        {Regs::UconfigRegWordOffset, 0x1000}, // primitive type, index type, instances...
    }};
    for (const auto& [first, count] : kRanges) {
        std::memcpy(&regs.reg_array[first], &defaults->reg_array[first], count * sizeof(u32));
    }
}

void NoteChunk(WalkStats& stats, const u32* begin, const u32* limit) {
    if (stats.num_chunks < stats.chunks.size()) {
        stats.chunks[stats.num_chunks] = {begin, limit};
    }
    ++stats.num_chunks;
}

} // namespace

bool JobCursor::Fail(bool overrun, const char* reason, WalkStats& stats) {
    (overrun ? stats.overrun : stats.bad_packet) = true;
    stats.stop = p;
    stats.reason = reason;
    stats.stop_header = p != nullptr && p < limit ? *p : 0;
    ok = false;
    return false;
}

bool JobCursor::Start(const u32* begin, const u32* first_limit, WalkStats& stats) {
    p = begin;
    // 16 bytes beyond the command space hold the link to the next chunk.
    limit = first_limit + 4;
    ok = true;
    if (begin == nullptr || limit <= begin || !Readable(begin, (limit - begin) * sizeof(u32))) {
        return Fail(false, "first chunk not readable", stats);
    }
    NoteChunk(stats, begin, limit);
    return true;
}

void JobCursor::SkipFork(const u32* next, const u32* next_limit, WalkStats& stats) {
    if (!ok) {
        return;
    }
    ++stats.forks;
    // Usually the job goes on right after the link slot, in the chunk the walk is in.
    const bool in_chunk = next == p + 4 && next_limit + 4 <= limit;
    p = next;
    limit = next_limit + 4;
    if (in_chunk) {
        return;
    }
    if (next == nullptr || limit <= next || !Readable(next, (limit - next) * sizeof(u32))) {
        Fail(false, "chunk after a fork not readable", stats);
        return;
    }
    NoteChunk(stats, next, limit);
}

bool JobCursor::WalkTo(const u32* until, Regs& regs, JobExtents& extents, JobVisitor& visitor,
                       WalkStats& stats) {
    constexpr u32 kComputeFirst = 0x200;
    constexpr u32 kComputeLast = kComputeFirst + sizeof(AmdGpu::ComputeProgram) / sizeof(u32);
    if (!ok) {
        return false;
    }
    while (p != until) {
        if (p >= limit) {
            return Fail(true, "walked past the chunk without reaching the end", stats);
        }
        if (++stats.packets > (1u << 22)) {
            return Fail(true, "too many packets", stats);
        }
        const u32 header = *p;
        const u32 type = header >> 30;
        if (type == 2) {
            ++p;
            continue;
        }
        if (type == 0) {
            const u32 first = header & 0xffff;
            const u32 count = ((header >> 16) & 0x3fff) + 1;
            if (p + 1 + count > limit) {
                return Fail(false, "type-0 packet crosses the chunk end", stats);
            }
            if (first + count <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[first], p + 1, count * sizeof(u32));
            }
            p += count + 1;
            continue;
        }
        if (type != 3) {
            return Fail(false, "type-1 packet", stats);
        }
        const u32 count = ((header >> 16) & 0x3fff) + 1; // dwords after the header
        if (p + 1 + count > limit) {
            // packets never cross a chunk's end
            return Fail(false, "packet crosses the chunk end", stats);
        }
        const auto opcode = static_cast<PM4ItOpcode>((header >> 8) & 0xff);
        const u32* body = p + 1;
        const auto set_regs = [&](u32 base) {
            const u32 offset = body[0] & 0xffff;
            if (count > 1 && base + offset + count - 1 <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[base + offset], body + 1, (count - 1) * sizeof(u32));
            }
            return base + offset;
        };
        switch (opcode) {
        case PM4ItOpcode::Nop:
            break;
        case PM4ItOpcode::SetContextReg: {
            const u32 reg_addr = set_regs(Regs::ContextRegWordOffset);
            NoteExtentHints(reg_addr, (header >> 16) & 0x3fff, body + 1, extents);
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const u32 offset = body[0] & 0xffff;
            if (offset < kComputeFirst || offset > kComputeLast) {
                set_regs(Regs::ShRegWordOffset);
            }
            break;
        }
        case PM4ItOpcode::SetUconfigReg:
            set_regs(Regs::UconfigRegWordOffset);
            break;
        case PM4ItOpcode::SetConfigReg:
            set_regs(Regs::ConfigRegWordOffset);
            break;
        case PM4ItOpcode::IndexType:
            regs.index_buffer_type.raw = body[0];
            break;
        case PM4ItOpcode::NumInstances:
            regs.num_instances.num_instances = body[0];
            break;
        case PM4ItOpcode::IndexBase:
            regs.index_base_address.base_addr_lo = body[0];
            regs.index_base_address.base_addr_hi = body[1];
            break;
        case PM4ItOpcode::IndexBufferSize:
            regs.num_indices = body[0];
            break;
        case PM4ItOpcode::ContextControl:
            break;
        case PM4ItOpcode::ClearState:
            ClearState(regs);
            ++stats.clear_states;
            break;
        case PM4ItOpcode::DrawIndex2:
            regs.max_index_size = body[0];
            regs.index_base_address.base_addr_lo = body[1];
            regs.index_base_address.base_addr_hi = body[2];
            regs.num_indices = body[3];
            regs.draw_initiator = body[4];
            ++stats.draws;
            visitor.OnDraw(p, opcode, regs, extents);
            break;
        case PM4ItOpcode::DrawIndexOffset2:
            regs.max_index_size = body[0];
            regs.num_indices = body[2];
            regs.draw_initiator = body[3];
            ++stats.draws;
            visitor.OnDraw(p, opcode, regs, extents);
            break;
        case PM4ItOpcode::DrawIndexAuto:
            regs.num_indices = body[0];
            regs.draw_initiator = body[1];
            ++stats.draws;
            visitor.OnDraw(p, opcode, regs, extents);
            break;
        case PM4ItOpcode::IndirectBuffer: {
            const auto* ib = reinterpret_cast<const AmdGpu::PM4CmdIndirectBuffer*>(p);
            if (ib->chain) {
                // PGL links its command chunks: the walk continues in the next one.
                const u32* next = ib->Address<const u32>();
                if (!Readable(next, ib->ib_size * sizeof(u32))) {
                    return Fail(false, "chained chunk not readable", stats);
                }
                ++stats.chain_links;
                p = next;
                limit = next + ib->ib_size;
                NoteChunk(stats, next, limit);
                continue;
            }
            ++stats.call_links; // not used by GT Sport's jobs; its commands are not followed
            visitor.OnOther(p, opcode);
            break;
        }
        default:
            visitor.OnOther(p, opcode);
            break;
        }
        p += count + 1;
    }
    return true;
}

void WalkJob(const JobRange& job, Regs& regs, JobExtents& extents, JobVisitor& visitor,
             WalkStats& stats) {
    JobCursor cursor;
    if (cursor.Start(job.begin, job.first_limit, stats)) {
        cursor.WalkTo(job.end, regs, extents, visitor, stats);
    }
}

} // namespace Native
