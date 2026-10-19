// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <windows.h>
#include <fmt/format.h>
#include <xxhash.h>

#include "common/logging/log.h"
#include "ryty_gpu/native/shadow.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/regs.h"

namespace Native {
namespace {

using AmdGpu::PM4ItOpcode;
using AmdGpu::Regs;

const bool g_enabled = [] {
    const char* env = std::getenv("RYGT_NATIVE_SHADOW");
    return env && env[0] == '1';
}();

// The state compared at each draw (dword indices into Regs::reg_array): the graphics stages' SH
// registers (programs, user data; compute registers from 0x2E00 are kept apart by the command
// processor), all context registers, and the draw registers outside those (uconfig).
constexpr u32 kShFirst = Regs::ShRegWordOffset;
constexpr u32 kShCount = 0x200;
constexpr u32 kCtxFirst = Regs::ContextRegWordOffset;
constexpr u32 kCtxCount = 0x400;
#define NAT_FIELD(field) static_cast<u32>(offsetof(Regs, field) / sizeof(u32))
const std::array<u32, 3> kDrawFields = {NAT_FIELD(primitive_type), NAT_FIELD(num_indices),
                                        NAT_FIELD(num_instances)};
#undef NAT_FIELD

constexpr int kShGroups = kShCount / 64;
constexpr int kCtxGroups = kCtxCount / 64;
constexpr int kGroups = kShGroups + kCtxGroups + 1;
constexpr u32 kFullDwords = kShCount + kCtxCount + 3;

std::string GroupName(int group) {
    if (group < kShGroups) {
        return fmt::format("sh {:#x}", kShFirst + group * 64);
    }
    if (group < kShGroups + kCtxGroups) {
        return fmt::format("ctx {:#x}", kCtxFirst + (group - kShGroups) * 64);
    }
    return "draw";
}

u32 FullIndexToRegister(u32 index) {
    if (index < kShCount) {
        return kShFirst + index;
    }
    if (index < kShCount + kCtxCount) {
        return kCtxFirst + index - kShCount;
    }
    return kDrawFields[index - kShCount - kCtxCount];
}

using Hashes = std::array<u64, kGroups>;
using Full = std::array<u32, kFullDwords>;

void HashRegs(const u32* regs, Hashes& out) {
    for (int g = 0; g < kShGroups; ++g) {
        out[g] = XXH3_64bits(regs + kShFirst + g * 64, 64 * sizeof(u32));
    }
    for (int g = 0; g < kCtxGroups; ++g) {
        out[kShGroups + g] = XXH3_64bits(regs + kCtxFirst + g * 64, 64 * sizeof(u32));
    }
    u32 fields[kDrawFields.size()];
    for (size_t i = 0; i < kDrawFields.size(); ++i) {
        fields[i] = regs[kDrawFields[i]];
    }
    out[kGroups - 1] = XXH3_64bits(fields, sizeof(fields));
}

void CaptureFull(const u32* regs, Full& out) {
    for (u32 i = 0; i < kFullDwords; ++i) {
        out[i] = regs[FullIndexToRegister(i)];
    }
}

// What a job worked out at one of its draws, keyed by the draw packet's address.
struct Entry {
    Hashes hashes;
    std::unique_ptr<Full> full; // one draw in 32 keeps every value, to name the registers that differ
    u32 epoch;
};
struct Shard {
    std::mutex mutex;
    std::unordered_map<u64, Entry> map;
};
std::array<Shard, 64> g_shards;
std::atomic<u32> g_epoch{0}; // graphics submissions seen by the command processor

Shard& ShardOf(u64 key) {
    return g_shards[(key * 0x9E3779B97F4A7C15ULL) >> 58];
}

// The registers at the start of the latest graphics submission: the state a job starts from.
std::shared_mutex g_base_mutex;
std::unique_ptr<Regs> g_base;

struct Stats {
    std::atomic<u64> jobs_begun, jobs_decoded, jobs_unmatched_end, jobs_without_base;
    std::atomic<u64> draws, packets, chain_links, call_links, bad_packets, overruns;
    std::atomic<u64> decode_ns;
    std::atomic<u64> checked, identical, different, not_decoded;
    std::atomic<u64> group_different[kGroups];
};
Stats g_stats{};

std::mutex g_register_mutex;
struct RegisterDiff {
    u64 count = 0;
    u32 cp_value = 0;
    u32 job_value = 0;
};
std::unordered_map<u32, RegisterDiff> g_register_diffs; // sampled draws only

struct Job {
    const u8* dcb;
    const u32* begin;
    const u32* first_limit; // end of the first chunk's command space
    const u32* end;
};
thread_local std::vector<Job> t_jobs;
thread_local Job t_ended{};
thread_local bool t_has_ended = false;
thread_local std::unique_ptr<Regs> t_regs;
thread_local u32 t_sample = 0;

const u32* DcbBegin(const u8* dcb) {
    return *reinterpret_cast<const u32* const*>(dcb); // Gnm CommandBuffer::m_beginptr
}
const u32* DcbEnd(const u8* dcb) {
    return *reinterpret_cast<const u32* const*>(dcb + 0x8); // m_endptr
}
const u32* DcbCmdPtr(const u8* dcb) {
    return *reinterpret_cast<const u32* const*>(dcb + 0x10); // m_cmdptr
}

bool Readable(const void* address, size_t size) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State != MEM_COMMIT ||
        (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
        return false;
    }
    return static_cast<const u8*>(address) + size <=
           static_cast<const u8*>(info.BaseAddress) + info.RegionSize;
}

void Snapshot(const u32* packet, const Regs& regs, u32 epoch) {
    Entry entry;
    HashRegs(regs.reg_array.data(), entry.hashes);
    if ((t_sample++ & 31) == 0) {
        entry.full = std::make_unique<Full>();
        CaptureFull(regs.reg_array.data(), *entry.full);
    }
    entry.epoch = epoch;
    const u64 key = reinterpret_cast<u64>(packet);
    Shard& shard = ShardOf(key);
    std::scoped_lock lock{shard.mutex};
    if (shard.map.size() > 20000) {
        // Draws the command processor never checked (not submitted, or not reached yet after a
        // long stall): forget the ones from more than a few submissions ago.
        std::erase_if(shard.map, [epoch](const auto& item) { return epoch - item.second.epoch > 8; });
    }
    shard.map.insert_or_assign(key, std::move(entry));
}

// Reads one job's commands like the command processor does, keeping only the GPU state, and
// records a snapshot at each draw. Chained chunks are followed; the walk ends where the job's
// commands end.
void Decode(const Job& job) {
    const auto start = std::chrono::steady_clock::now();
    if (!t_regs) {
        t_regs = std::make_unique<Regs>();
    }
    Regs& regs = *t_regs;
    {
        std::shared_lock lock{g_base_mutex};
        if (!g_base) {
            ++g_stats.jobs_without_base;
            return;
        }
        std::memcpy(&regs.reg_array[kShFirst], &g_base->reg_array[kShFirst], kShCount * sizeof(u32));
        std::memcpy(&regs.reg_array[kCtxFirst], &g_base->reg_array[kCtxFirst],
                    kCtxCount * sizeof(u32));
        for (const u32 field : kDrawFields) {
            regs.reg_array[field] = g_base->reg_array[field];
        }
    }
    const u32 epoch = g_epoch.load(std::memory_order_relaxed);
    constexpr u32 kComputeFirst = 0x200;
    constexpr u32 kComputeLast = kComputeFirst + sizeof(AmdGpu::ComputeProgram) / sizeof(u32);
    const u32* p = job.begin;
    // The walk stays inside the chunk it is in (16 bytes beyond the command space hold the link).
    const u32* limit = job.first_limit + 4;
    if (job.begin == nullptr || limit <= job.begin ||
        !Readable(job.begin, (limit - job.begin) * sizeof(u32))) {
        ++g_stats.bad_packets;
        return;
    }
    u64 packets = 0, draws = 0;
    while (p != job.end) {
        if (p >= limit || ++packets > (1u << 22)) {
            ++g_stats.overruns;
            break;
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
                ++g_stats.bad_packets;
                break;
            }
            if (first + count <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[first], p + 1, count * sizeof(u32));
            }
            p += count + 1;
            continue;
        }
        if (type != 3) {
            ++g_stats.bad_packets;
            break;
        }
        const u32 count = ((header >> 16) & 0x3fff) + 1; // dwords after the header
        if (p + 1 + count > limit) {
            ++g_stats.bad_packets; // packets never cross a chunk's end
            break;
        }
        const auto opcode = static_cast<PM4ItOpcode>((header >> 8) & 0xff);
        const u32* body = p + 1;
        const auto set_regs = [&](u32 base) {
            const u32 offset = body[0] & 0xffff;
            if (count > 1 && base + offset + count - 1 <= Regs::NumRegs) {
                std::memcpy(&regs.reg_array[base + offset], body + 1, (count - 1) * sizeof(u32));
            }
        };
        switch (opcode) {
        case PM4ItOpcode::SetContextReg:
            set_regs(Regs::ContextRegWordOffset);
            break;
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
        case PM4ItOpcode::DrawIndex2:
            regs.max_index_size = body[0];
            regs.index_base_address.base_addr_lo = body[1];
            regs.index_base_address.base_addr_hi = body[2];
            regs.num_indices = body[3];
            regs.draw_initiator = body[4];
            Snapshot(p, regs, epoch);
            ++draws;
            break;
        case PM4ItOpcode::DrawIndexOffset2:
            regs.max_index_size = body[0];
            regs.num_indices = body[2];
            regs.draw_initiator = body[3];
            Snapshot(p, regs, epoch);
            ++draws;
            break;
        case PM4ItOpcode::DrawIndexAuto:
            regs.num_indices = body[0];
            regs.draw_initiator = body[1];
            Snapshot(p, regs, epoch);
            ++draws;
            break;
        case PM4ItOpcode::IndirectBuffer: {
            const auto* ib = reinterpret_cast<const AmdGpu::PM4CmdIndirectBuffer*>(p);
            if (ib->chain) {
                // PGL links its command chunks: the walk continues in the next one.
                const u32* next = ib->Address<const u32>();
                if (!Readable(next, ib->ib_size * sizeof(u32))) {
                    ++g_stats.bad_packets;
                    p = job.end;
                    continue;
                }
                ++g_stats.chain_links;
                p = next;
                limit = next + ib->ib_size;
                continue;
            }
            ++g_stats.call_links; // not used by GT Sport's jobs; its commands are not followed
            break;
        }
        default:
            break;
        }
        p += count + 1;
    }
    g_stats.packets += packets;
    g_stats.draws += draws;
    ++g_stats.jobs_decoded;
    g_stats.decode_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
}

void ReporterLoop() {
    using namespace std::chrono_literals;
    Stats last{};
    const auto take = [](const std::atomic<u64>& value, std::atomic<u64>& previous) {
        const u64 now = value.load(std::memory_order_relaxed);
        const u64 delta = now - previous.load(std::memory_order_relaxed);
        previous.store(now, std::memory_order_relaxed);
        return delta;
    };
    while (true) {
        std::this_thread::sleep_for(10s);
        const u64 jobs = take(g_stats.jobs_decoded, last.jobs_decoded);
        const u64 draws = take(g_stats.draws, last.draws);
        const u64 packets = take(g_stats.packets, last.packets);
        const u64 ns = take(g_stats.decode_ns, last.decode_ns);
        LOG_INFO(Lib_GnmDriver,
                 "NAT-003 jobs (10 s): {} begun, {} decoded on their threads ({} draws, {} "
                 "packets, {} chunk links, {:.1f} us per job, {:.0f} ns per draw); problems: {} bad "
                 "packets, {} overruns, {} calls not followed, {} without base state, {} unmatched "
                 "ends",
                 take(g_stats.jobs_begun, last.jobs_begun), jobs, draws, packets,
                 take(g_stats.chain_links, last.chain_links), jobs ? ns / 1000.0 / jobs : 0.0,
                 draws ? double(ns) / draws : 0.0, take(g_stats.bad_packets, last.bad_packets),
                 take(g_stats.overruns, last.overruns), take(g_stats.call_links, last.call_links),
                 take(g_stats.jobs_without_base, last.jobs_without_base),
                 take(g_stats.jobs_unmatched_end, last.jobs_unmatched_end));
        const u64 checked = take(g_stats.checked, last.checked);
        const u64 identical = take(g_stats.identical, last.identical);
        std::string groups;
        for (int g = 0; g < kGroups; ++g) {
            if (const u64 n = take(g_stats.group_different[g], last.group_different[g])) {
                groups += fmt::format(" [{}: {}]", GroupName(g), n);
            }
        }
        LOG_INFO(Lib_GnmDriver,
                 "NAT-003 compare (10 s): {} draws checked by the command processor, {} identical "
                 "({:.2f}%), {} different, {} not decoded by a job; differing register groups:{}",
                 checked, identical, checked ? 100.0 * identical / checked : 0.0,
                 take(g_stats.different, last.different),
                 take(g_stats.not_decoded, last.not_decoded), groups.empty() ? " none" : groups);
        std::vector<std::pair<u32, RegisterDiff>> diffs;
        {
            std::scoped_lock lock{g_register_mutex};
            diffs.assign(g_register_diffs.begin(), g_register_diffs.end());
        }
        if (!diffs.empty()) {
            std::ranges::sort(diffs, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
            std::string line;
            for (size_t i = 0; i < std::min<size_t>(diffs.size(), 24); ++i) {
                line += fmt::format(" [{:#x}: {} (cp {:#x}, job {:#x})]", diffs[i].first,
                                    diffs[i].second.count, diffs[i].second.cp_value,
                                    diffs[i].second.job_value);
            }
            LOG_INFO(Lib_GnmDriver, "NAT-003 registers that differ (sampled draws, since start, {} registers):{}",
                     diffs.size(), line);
        }
    }
}

std::atomic<bool> g_reporter_started{false};

} // namespace

bool ShadowEnabled() {
    return g_enabled;
}

void ShadowJobBegin(const u8* dcb) {
    ++g_stats.jobs_begun;
    t_jobs.push_back({dcb, DcbBegin(dcb), DcbEnd(dcb), nullptr});
}

void ShadowJobEnd() {
    t_has_ended = false;
    if (t_jobs.empty()) {
        ++g_stats.jobs_unmatched_end;
        return;
    }
    t_ended = t_jobs.back();
    t_jobs.pop_back();
    t_ended.end = DcbCmdPtr(t_ended.dcb);
    t_has_ended = true;
}

void ShadowJobDecode() {
    if (t_has_ended) {
        t_has_ended = false;
        Decode(t_ended);
    }
}

void ShadowSubmitBase(const Regs& regs) {
    ++g_epoch;
    std::unique_lock lock{g_base_mutex};
    if (!g_base) {
        g_base = std::make_unique<Regs>();
    }
    std::memcpy(&g_base->reg_array[kShFirst], &regs.reg_array[kShFirst], kShCount * sizeof(u32));
    std::memcpy(&g_base->reg_array[kCtxFirst], &regs.reg_array[kCtxFirst], kCtxCount * sizeof(u32));
    for (const u32 field : kDrawFields) {
        g_base->reg_array[field] = regs.reg_array[field];
    }
}

void ShadowCheckDraw(const void* packet, const Regs& regs) {
    if (!g_reporter_started.exchange(true)) {
        std::thread{ReporterLoop}.detach();
    }
    ++g_stats.checked;
    const u64 key = reinterpret_cast<u64>(packet);
    Entry entry;
    {
        Shard& shard = ShardOf(key);
        std::scoped_lock lock{shard.mutex};
        const auto it = shard.map.find(key);
        if (it == shard.map.end()) {
            ++g_stats.not_decoded;
            return;
        }
        entry = std::move(it->second);
        shard.map.erase(it);
    }
    Hashes cp;
    HashRegs(regs.reg_array.data(), cp);
    bool same = true;
    for (int g = 0; g < kGroups; ++g) {
        if (cp[g] != entry.hashes[g]) {
            same = false;
            ++g_stats.group_different[g];
        }
    }
    if (same) {
        ++g_stats.identical;
        return;
    }
    ++g_stats.different;
    if (entry.full) {
        Full now;
        CaptureFull(regs.reg_array.data(), now);
        std::scoped_lock lock{g_register_mutex};
        for (u32 i = 0; i < kFullDwords; ++i) {
            if (now[i] != (*entry.full)[i]) {
                auto& diff = g_register_diffs[FullIndexToRegister(i)];
                ++diff.count;
                diff.cp_value = now[i];
                diff.job_value = (*entry.full)[i];
            }
        }
    }
}

} // namespace Native
