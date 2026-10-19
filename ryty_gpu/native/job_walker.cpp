// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
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

bool Readable(const void* address, size_t size) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State != MEM_COMMIT ||
        (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
        return false;
    }
    return static_cast<const u8*>(address) + size <=
           static_cast<const u8*>(info.BaseAddress) + info.RegionSize;
}

void WalkJob(const JobRange& job, Regs& regs, JobExtents& extents, JobVisitor& visitor,
             WalkStats& stats) {
    constexpr u32 kComputeFirst = 0x200;
    constexpr u32 kComputeLast = kComputeFirst + sizeof(AmdGpu::ComputeProgram) / sizeof(u32);
    const u32* p = job.begin;
    // The walk stays inside the chunk it is in (16 bytes beyond the command space hold the link).
    const u32* limit = job.first_limit + 4;
    if (job.begin == nullptr || limit <= job.begin ||
        !Readable(job.begin, (limit - job.begin) * sizeof(u32))) {
        stats.bad_packet = true;
        return;
    }
    while (p != job.end) {
        if (p >= limit || ++stats.packets > (1u << 22)) {
            stats.overrun = true;
            return;
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
                stats.bad_packet = true;
                return;
            }
            if (first + count <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[first], p + 1, count * sizeof(u32));
            }
            p += count + 1;
            continue;
        }
        if (type != 3) {
            stats.bad_packet = true;
            return;
        }
        const u32 count = ((header >> 16) & 0x3fff) + 1; // dwords after the header
        if (p + 1 + count > limit) {
            stats.bad_packet = true; // packets never cross a chunk's end
            return;
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
                    stats.bad_packet = true;
                    return;
                }
                ++stats.chain_links;
                p = next;
                limit = next + ib->ib_size;
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
}

} // namespace Native
