// Stand-ins for the parts of shadGT's emulator core the renderer depends on. In the Ryty build the
// guest's memory, kernel and event queues belong to Ryty's libkernel and libc, so these forward to
// them or to Windows directly.

#include "ryty_gpu/ryty_diag.h"
#include <windows.h>
#include <intrin.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <string>
#include <unordered_map>

#include <boost/icl/interval_set.hpp>
#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "common/read_capture.h"
#include "core/address_space.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/kernel/time.h"
#include "core/guest_hash.h"
#include "core/memory.h"
#include "imgui/friends_layer.h"
#include "imgui/invitation_prompt_layer.h"
#include "imgui/shadnet_notifications_layer.h"

namespace {

void* RytyExport(const char* module, const char* name) {
    HMODULE handle = GetModuleHandleA(module);
    void* address = handle ? reinterpret_cast<void*>(GetProcAddress(handle, name)) : nullptr;
    if (address == nullptr) {
        LOG_CRITICAL(Lib_GnmDriver, "Ryty export {} not found in {}", name, module);
    }
    return address;
}

template <typename T>
T RytyFunction(const char* module, const char* name) {
    return reinterpret_cast<T>(RytyExport(module, name));
}

// libc.prx: GuestArenaMapAlias / GuestArenaUnmapAlias (Microsoft ABI, exported by NID).
using MapAliasFn = void* (*)(std::uintptr_t, std::size_t);
using UnmapAliasFn = void (*)(void*);

bool IsCommitted(const MEMORY_BASIC_INFORMATION& info) {
    return info.State == MEM_COMMIT;
}

bool IsReadable(DWORD protect) {
    if ((protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0) {
        return false;
    }
    return (protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                       PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
}

bool IsWritable(DWORD protect) {
    if ((protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0) {
        return false;
    }
    return (protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE)) != 0;
}

bool IsExecutable(DWORD protect) {
    return (protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                       PAGE_EXECUTE_WRITECOPY)) != 0;
}

// PERF-080: alias views of single 16 KiB pages are kept for reuse. The renderer writes fence
// labels and readbacks into guest memory thousands of times a second, and mapping and unmapping
// a view for each write cost about 2% of the recorder in races. A mapping change drops the views
// over it.
struct AliasCache {
    std::mutex mutex;
    std::unordered_map<VAddr, void*> views;
};

AliasCache& GetAliasCache() {
    static AliasCache cache;
    return cache;
}

void DropAliases(VAddr address, u64 size) {
    static const auto unmap_alias = RytyFunction<UnmapAliasFn>("libc.prx", "674-w6lwWXI");
    if (unmap_alias == nullptr) {
        return;
    }
    auto& cache = GetAliasCache();
    std::scoped_lock lk{cache.mutex};
    if (cache.views.empty()) {
        return;
    }
    constexpr VAddr AliasPage = 0x4000;
    const VAddr first = address & ~(AliasPage - 1);
    const VAddr end = address + std::max<u64>(size, 1);
    if ((end - first) / AliasPage > cache.views.size()) {
        for (auto it = cache.views.begin(); it != cache.views.end();) {
            if (it->first >= first && it->first < end) {
                if (it->second) {
                    unmap_alias(it->second);
                }
                it = cache.views.erase(it);
            } else {
                ++it;
            }
        }
        return;
    }
    for (VAddr page = first; page < end; page += AliasPage) {
        if (const auto it = cache.views.find(page); it != cache.views.end()) {
            if (it->second) {
                unmap_alias(it->second);
            }
            cache.views.erase(it);
        }
    }
}

// Copies through an alias view of the same pages, so protections set for write tracking do not
// fault and the copy does not count as a guest write.
// PERF-112: a write that fits in one 16 KiB page goes through a cached alias view of the page,
// without the VirtualQuery TryWriteBacking made for each write (fence and label writes: ~3% of the
// recorder in a 20-car race). Pages without an alias (not Ryty shared memory) are remembered as
// such (a null view) and return false, so the caller takes the usual path.
bool WriteThroughCachedAlias(VAddr address, const void* source, u64 size) {
    static const auto map_alias = RytyFunction<MapAliasFn>("libc.prx", "TXQU8CpoEtI");
    static const auto unmap_alias = RytyFunction<UnmapAliasFn>("libc.prx", "674-w6lwWXI");
    static const bool cache_views = [] {
        const char* env = std::getenv("SHADGT_ALIAS_CACHE");
        return !(env && env[0] == '0');
    }();
    if (!cache_views || map_alias == nullptr || unmap_alias == nullptr) {
        return false;
    }
    constexpr VAddr AliasPage = 0x4000;
    const VAddr first = address & ~(AliasPage - 1);
    if (((address + size + AliasPage - 1) & ~(AliasPage - 1)) - first != AliasPage) {
        return false;
    }
    auto& cache = GetAliasCache();
    std::scoped_lock lk{cache.mutex};
    auto it = cache.views.find(first);
    if (it == cache.views.end()) {
        if (cache.views.size() >= 4096) {
            for (const auto& [page, view] : cache.views) {
                if (view) {
                    unmap_alias(view);
                }
            }
            cache.views.clear();
        }
        it = cache.views.emplace(first, map_alias(first, AliasPage)).first;
    }
    if (it->second == nullptr) {
        return false;
    }
    std::memcpy(static_cast<u8*>(it->second) + (address - first), source, size);
    return true;
}

bool CopyThroughAlias(VAddr address, void* dest, const void* source, u64 size) {
    static const auto map_alias = RytyFunction<MapAliasFn>("libc.prx", "TXQU8CpoEtI");
    static const auto unmap_alias = RytyFunction<UnmapAliasFn>("libc.prx", "674-w6lwWXI");
    if (map_alias == nullptr || unmap_alias == nullptr) {
        return false;
    }
    // Ryty aliases whole shared pages (16 KiB) only; alias the pages around the range.
    constexpr VAddr AliasPage = 0x4000;
    const VAddr first = address & ~(AliasPage - 1);
    const VAddr last = (address + size + AliasPage - 1) & ~(AliasPage - 1);
    // On unless SHADGT_ALIAS_CACHE=0. The two race crashes that made it opt-in were occlusion
    // results written to garbage addresses (FIX-072), not the cache.
    static const bool cache_views = [] {
        const char* env = std::getenv("SHADGT_ALIAS_CACHE");
        return !(env && env[0] == '0');
    }();
    if (cache_views && dest == nullptr && last - first == AliasPage) {
        // Writes only: reads come in bulk for uploads and would fill the cache.
        auto& cache = GetAliasCache();
        std::scoped_lock lk{cache.mutex};
        auto it = cache.views.find(first);
        if (it == cache.views.end()) {
            if (cache.views.size() >= 4096) {
                for (const auto& [page, view] : cache.views) {
                    if (view) {
                        unmap_alias(view);
                    }
                }
                cache.views.clear();
            }
            if (void* view = map_alias(first, AliasPage)) {
                it = cache.views.emplace(first, view).first;
            }
        }
        if (it != cache.views.end() && it->second != nullptr) {
            std::memcpy(static_cast<u8*>(it->second) + (address - first), source, size);
            return true;
        }
    }
    void* alias = map_alias(first, last - first);
    if (alias == nullptr) {
        // Not Ryty shared memory (the title's own image, for example): lift the protection page
        // tracking set for the copy and put it back, rather than drop GPU results or read zeros.
        static std::atomic<u32> logged{};
        if (logged++ < 20) {
            LOG_WARNING(Render_Vulkan, "No alias for {:#x}+{:#x}; copying {} under a lifted protection",
                        address, size, dest != nullptr ? "from it" : "into it");
        }
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) ||
            !IsCommitted(info)) {
            return false;
        }
        const bool exec = IsExecutable(info.Protect);
        const DWORD lifted = dest != nullptr ? (exec ? PAGE_EXECUTE_READ : PAGE_READONLY)
                                             : (exec ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
        DWORD old_protect;
        if (!VirtualProtect(reinterpret_cast<void*>(address), size, lifted, &old_protect)) {
            return false;
        }
        if (dest != nullptr) {
            std::memcpy(dest, reinterpret_cast<const void*>(address), size);
        } else {
            std::memcpy(reinterpret_cast<void*>(address), source, size);
        }
        DWORD ignored;
        VirtualProtect(reinterpret_cast<void*>(address), size, old_protect, &ignored);
        return true;
    }
    auto* target = static_cast<u8*>(alias) + (address - first);
    if (dest != nullptr) {
        std::memcpy(dest, target, size);
    } else {
        std::memcpy(target, source, size);
    }
    unmap_alias(alias);
    return true;
}

} // namespace

namespace Core::RytyMemory {

// Every 16 KB shared view is its own region, so walking a mapped area takes thousands of
// queries, and draws ask about buffers all over the same areas (each draw's constants sit at a
// new place in a ring) and about addresses that are not mapped at all. Intervals found by walks,
// mapped or not, are kept until Ryty reports a map, unmap or protection change touching them; a
// change only cuts a mapped interval short, since the memory below it is unaffected. Only
// libSceVideoOut receives Ryty's reports; the other libraries built from this file never get one
// and so never cache.
// Ranges that page tracking (AddressSpace::Protect) has made unreadable. Copies from guest memory
// outside them need no query when the area was readable when it was walked.
struct NoReadSet {
    SRWLOCK lock = SRWLOCK_INIT;
    boost::icl::interval_set<VAddr> ranges;
};

NoReadSet& GetNoReadSet() {
    static NoReadSet set;
    return set;
}

void NoteReadable(VAddr address, u64 size, bool readable) {
    auto& set = GetNoReadSet();
    const auto range = boost::icl::interval<VAddr>::right_open(address, address + size);
    AcquireSRWLockExclusive(&set.lock);
    if (readable) {
        set.ranges.subtract(range);
    } else {
        set.ranges.add(range);
    }
    ReleaseSRWLockExclusive(&set.lock);
}

// Whether page tracking made any of [address, +size) unreadable (covers: all of it).
bool NoReadIntersects(VAddr address, u64 size, bool whole = false) {
    auto& set = GetNoReadSet();
    const auto range = boost::icl::interval<VAddr>::right_open(address, address + size);
    AcquireSRWLockShared(&set.lock);
    const bool result = whole ? boost::icl::contains(set.ranges, range)
                              : boost::icl::intersects(set.ranges, range);
    ReleaseSRWLockShared(&set.lock);
    return result;
}

struct AreaCache {
    struct Interval {
        VAddr end;
        bool mapped;
        bool complete; // mapped: end is where the area ends, not just how far a walk got
        bool exec;     // mapped: some of it is executable (guest code)
        bool views;    // mapped: made of separate 16 KB views (Ryty's shared guest memory)
        bool readable; // mapped: readable apart from the pages page tracking made unreadable
    };
    SRWLOCK lock = SRWLOCK_INIT;
    std::map<VAddr, Interval> intervals; // by start
    u64 generation = 0;
    std::atomic<bool> enabled{false};
    /// Bumped with every reported change; a thread's last answer is reused while it is unchanged.
    std::atomic<u64> version{0};

    // The interval containing address, or end().
    std::map<VAddr, Interval>::iterator Find(VAddr address) {
        auto it = intervals.upper_bound(address);
        if (it == intervals.begin()) {
            return intervals.end();
        }
        --it;
        return address < it->second.end ? it : intervals.end();
    }
};

AreaCache& GetAreaCache() {
    static AreaCache cache;
    return cache;
}

struct AreaAnswer {
    VAddr end;     // 0 when the address is not mapped
    bool complete; // false when the walk stopped at its cap before reaching limit
    bool exec = false;  // some of the walked memory is executable
    bool views = true;  // every walked region is a separate view of at most 16 KB
    bool readable = true; // every walked region is readable, or unreadable only for page tracking
};

// Whether a region lies in an allocation of at most 16 KB (one of Ryty's shared views); page
// protections split a view into several regions, so the allocation, not the region, decides.
bool InSmallView(const MEMORY_BASIC_INFORMATION& info) {
    const VAddr allocation = reinterpret_cast<VAddr>(info.AllocationBase);
    return reinterpret_cast<VAddr>(info.BaseAddress) + info.RegionSize - allocation <= 0x4000;
}

struct AreaStats {
    std::atomic<u64> lookups, hits, unmapped, queries, changes, stores_skipped;
    std::atomic<u64> last_report;
};
AreaStats g_area_stats{};

void ReportAreaStats() {
    const u64 now = GetTickCount64();
    u64 last = g_area_stats.last_report.load(std::memory_order_relaxed);
    if (now - last < 2000 || !g_area_stats.last_report.compare_exchange_strong(last, now)) {
        return;
    }
    LOG_WARNING(Render_Vulkan,
                "Mapping lookups in 2 s: {} lookups, {} cache hits, {} unmapped, {} region queries, "
                "{} map changes, {} answers not kept",
                g_area_stats.lookups.exchange(0), g_area_stats.hits.exchange(0),
                g_area_stats.unmapped.exchange(0), g_area_stats.queries.exchange(0),
                g_area_stats.changes.exchange(0), g_area_stats.stores_skipped.exchange(0));
}

// Walks committed regions from end (the end of a committed region) towards limit, jumping over
// intervals already known, so walks from different places in one area share their work.
AreaAnswer ContinueWalk(VAddr end, VAddr limit, bool caching, AreaAnswer flags) {
    auto& cache = GetAreaCache();
    bool exec = flags.exec;
    bool views = flags.views;
    bool readable = flags.readable;
    for (int regions = 0; regions < 4096; ++regions) {
        if (end >= limit) {
            return {end, false, exec, views, readable};
        }
        if (caching) {
            AcquireSRWLockShared(&cache.lock);
            const auto it = cache.Find(end);
            const AreaCache::Interval known =
                it != cache.intervals.end() ? it->second : AreaCache::Interval{};
            ReleaseSRWLockShared(&cache.lock);
            if (known.end != 0) {
                if (!known.mapped) {
                    return {end, true, exec, views, readable};
                }
                end = known.end;
                exec |= known.exec;
                views &= known.views;
                readable &= known.readable;
                if (known.complete) {
                    return {end, true, exec, views, readable};
                }
                continue;
            }
        }
        MEMORY_BASIC_INFORMATION info;
        g_area_stats.queries.fetch_add(1, std::memory_order_relaxed);
        if (!VirtualQuery(reinterpret_cast<void*>(end), &info, sizeof(info)) ||
            !IsCommitted(info) || reinterpret_cast<VAddr>(info.BaseAddress) != end) {
            return {end, true, exec, views, readable};
        }
        exec |= IsExecutable(info.Protect);
        views &= InSmallView(info);
        readable &= IsReadable(info.Protect) || NoReadIntersects(end, info.RegionSize, true);
        end += info.RegionSize;
    }
    return {end, false, exec, views, readable}; // the cap: how far the area goes is still unknown
}

// Records [start, interval.end) as mapped (merged with the mapped intervals it covers or touches
// from below) or as not mapped, unless a change was reported since generation was read.
void StoreInterval(VAddr start, AreaCache::Interval interval, u64 generation) {
    auto& cache = GetAreaCache();
    AcquireSRWLockExclusive(&cache.lock);
    if (cache.generation != generation) {
        g_area_stats.stores_skipped.fetch_add(1, std::memory_order_relaxed);
        ReleaseSRWLockExclusive(&cache.lock);
        return;
    }
    if (cache.intervals.size() > 16384) {
        cache.intervals.clear(); // garbage addresses can leave many small unmapped intervals
    }
    auto it = cache.intervals.upper_bound(start);
    if (it != cache.intervals.begin() && std::prev(it)->second.end >= start) {
        --it;
    }
    while (it != cache.intervals.end() && it->first < interval.end) {
        if (interval.mapped && it->second.mapped) {
            start = std::min(start, it->first);
            interval.exec |= it->second.exec;
            interval.views &= it->second.views;
            interval.readable &= it->second.readable;
            if (it->second.end > interval.end) {
                interval.end = it->second.end; // the walk stopped at its limit inside it
                interval.complete = it->second.complete;
            }
        } else if (it->second.end <= start) {
            ++it; // ends where this one starts: adjacent, not overlapping
            continue;
        }
        it = cache.intervals.erase(it);
    }
    cache.intervals[start] = interval;
    ReleaseSRWLockExclusive(&cache.lock);
}

static AreaAnswer LookupAreaWalk(VAddr virtual_addr, VAddr limit);

// FIX-055: guest memory is Ryty's guest arena and the game's own image (its code and static data,
// below 2^40). Everything else is the emulator's: Ryty's own heaps start right where the arena
// ends. A garbage descriptor on GT Sport's race summary pointed vertex buffers at 0xfcc2437bc7;
// counted as mapped, page tracking write-protected Ryty's heap, a heap operation faulted inside
// the fault handler and the process aborted (with 50 GB committed on the way).
struct GuestRange {
    VAddr start;
    VAddr end;
};

static std::optional<GuestRange> GuestRangeOf(VAddr address) {
    constexpr GuestRange Arena{0x200000000, 0xFC00000000};
    if (address >= Arena.start && address < Arena.end) {
        return Arena;
    }
    static const GuestRange image = [] {
        const auto base = reinterpret_cast<VAddr>(GetModuleHandleW(nullptr));
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
        return GuestRange{base, base + nt->OptionalHeader.SizeOfImage};
    }();
    if (address >= image.start && address < image.end) {
        return image;
    }
    return std::nullopt;
}

AreaAnswer LookupArea(VAddr virtual_addr, VAddr limit) {
    const auto range = GuestRangeOf(virtual_addr);
    if (!range) {
        return {0, true};
    }
    AreaAnswer answer = LookupAreaWalk(virtual_addr, std::min(limit, range->end));
    if (answer.end > range->end) {
        answer.end = range->end;
        answer.complete = true;
    }
    return answer;
}

static AreaAnswer LookupAreaWalk(VAddr virtual_addr, VAddr limit) {
    auto& cache = GetAreaCache();
    const bool caching = cache.enabled.load(std::memory_order_acquire);
    // Most lookups land in the interval the thread looked at last: that answer is used again
    // without the lock while no change was reported since (draws bind runs of nearby buffers).
    struct Memo {
        VAddr start;
        AreaCache::Interval interval;
        u64 version;
    };
    thread_local Memo memo{};
    const u64 version = cache.version.load(std::memory_order_acquire);
    if (caching && memo.interval.end != 0 && memo.version == version &&
        memo.start <= virtual_addr && virtual_addr < memo.interval.end) {
        const auto& known = memo.interval;
        if (!known.mapped) {
            return {0, true};
        }
        if (known.complete || known.end >= limit) {
            return {known.end, known.complete, known.exec, known.views, known.readable};
        }
    }
    g_area_stats.lookups.fetch_add(1, std::memory_order_relaxed);
    ReportAreaStats();
    VAddr start = 0;
    AreaCache::Interval found{};
    u64 generation = 0;
    if (caching) {
        AcquireSRWLockShared(&cache.lock);
        const auto it = cache.Find(virtual_addr);
        if (it != cache.intervals.end()) {
            start = it->first;
            found = it->second;
        }
        generation = cache.generation;
        ReleaseSRWLockShared(&cache.lock);
        if (found.end != 0) {
            memo = {start, found, version};
        }
        if (found.end != 0 && !found.mapped) {
            g_area_stats.hits.fetch_add(1, std::memory_order_relaxed);
            return {0, true};
        }
        if (found.end != 0 && (found.complete || found.end >= limit)) {
            g_area_stats.hits.fetch_add(1, std::memory_order_relaxed);
            return {found.end, found.complete, found.exec, found.views, found.readable};
        }
    }
    AreaAnswer answer;
    if (found.end != 0) {
        answer = ContinueWalk(found.end, limit, caching,
                              {0, false, found.exec, found.views, found.readable});
    } else {
        MEMORY_BASIC_INFORMATION info;
        g_area_stats.queries.fetch_add(1, std::memory_order_relaxed);
        if (!VirtualQuery(reinterpret_cast<void*>(virtual_addr), &info, sizeof(info))) {
            return {0, true};
        }
        start = reinterpret_cast<VAddr>(info.BaseAddress);
        if (!IsCommitted(info)) {
            g_area_stats.unmapped.fetch_add(1, std::memory_order_relaxed);
            if (caching) {
                const AreaCache::Interval unmapped{start + info.RegionSize, false, true, false,
                                                   false, false};
                StoreInterval(start, unmapped, generation);
                memo = {start, unmapped, version};
            }
            return {0, true};
        }
        const bool readable = IsReadable(info.Protect) ||
                              NoReadIntersects(start, info.RegionSize, true);
        answer = ContinueWalk(start + info.RegionSize, limit, caching,
                              {0, false, IsExecutable(info.Protect), InSmallView(info), readable});
    }
    // "complete" in the cache means the area's end was seen; reaching the limit or the cap is not.
    const bool area_end_seen = answer.complete;
    if (caching) {
        const AreaCache::Interval walked{answer.end,  true,         area_end_seen,
                                         answer.exec, answer.views, answer.readable};
        StoreInterval(start, walked, generation);
        memo = {start, walked, version};
    }
    return {answer.end, area_end_seen || answer.end >= limit, answer.exec, answer.views,
            answer.readable};
}

void NoteMappingChange(VAddr address, u64 size) {
    DropAliases(address, size); // PERF-080
    auto& cache = GetAreaCache();
    const VAddr change_end = address + std::max<u64>(size, 1);
    g_area_stats.changes.fetch_add(1, std::memory_order_relaxed);
    AcquireSRWLockExclusive(&cache.lock);
    ++cache.generation;
    cache.version.fetch_add(1, std::memory_order_acq_rel);
    auto it = cache.intervals.upper_bound(address);
    if (it != cache.intervals.begin()) {
        --it;
    }
    while (it != cache.intervals.end() && it->first < change_end) {
        auto& interval = it->second;
        // A mapped interval and the address right after it (a mapping placed there extends the
        // area); an unmapped interval anywhere the change touches.
        const bool touched = interval.mapped ? address <= interval.end : address < interval.end;
        if (!touched) {
            ++it;
            continue;
        }
        if (interval.mapped && address > it->first) {
            interval.end = std::min(interval.end, address);
            interval.complete = false;
            ++it;
        } else {
            it = cache.intervals.erase(it);
        }
    }
    cache.enabled.store(true, std::memory_order_release);
    ReleaseSRWLockExclusive(&cache.lock);
}

VAddr MappedAreaEnd(VAddr virtual_addr, VAddr limit) {
    return LookupArea(virtual_addr, limit).end;
}

bool IsValidMapping(VAddr virtual_addr, u64 size) {
    const VAddr end = virtual_addr + std::max<u64>(size, 1);
    const AreaAnswer answer = LookupArea(virtual_addr, end);
    if (answer.end >= end) {
        return true;
    }
    if (answer.complete) {
        return false;
    }
    // Longer than the walk's cap: check the rest region by region.
    VAddr address = answer.end;
    while (address < end) {
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) ||
            !IsCommitted(info)) {
            return false;
        }
        address = reinterpret_cast<VAddr>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}

// Whether [virtual_addr, +size) is mapped, and if so whether its area is made of 16 KB views.
bool IsMappedRange(VAddr virtual_addr, u64 size, bool& views) {
    const VAddr end = virtual_addr + std::max<u64>(size, 1);
    const AreaAnswer answer = LookupArea(virtual_addr, end);
    views = answer.views;
    return answer.end >= end;
}

// Whether [virtual_addr, +size) can be read with a plain copy: mapped, readable for the guest and
// not made unreadable by page tracking.
bool IsReadableRange(VAddr virtual_addr, u64 size) {
    const VAddr end = virtual_addr + std::max<u64>(size, 1);
    const AreaAnswer answer = LookupArea(virtual_addr, end);
    return answer.end >= end && answer.readable && !NoReadIntersects(virtual_addr, size);
}

// Guarded reads: the renderer reads guest memory directly when the caches say it is readable,
// but the game can release it at the same moment. A fault inside a guarded read on memory page
// tracking did not protect is passed back here (RytyGpuHandleWriteFault returns 2, Ryty lets the
// structured handler below take it), and the caller falls back to the safe copy.
thread_local int g_guarded_reads = 0;

bool InGuardedRead() {
    return g_guarded_reads != 0;
}

std::atomic<OwnWriteObserver> g_own_write_observer{nullptr};

void SetOwnWriteObserver(OwnWriteObserver observer) {
    g_own_write_observer.store(observer, std::memory_order_release);
}

void NoteOwnWrite(VAddr address, u64 size) {
    if (const auto observer = g_own_write_observer.load(std::memory_order_acquire)) {
        observer(address, size);
    }
}

bool IsWritableTarget(VAddr virtual_addr, u64 size) {
    if (virtual_addr == 0 || size == 0) {
        return false;
    }
    if (IsGuestReadable(virtual_addr, size)) {
        return true; // Guest memory: tracking protections fault and are handled.
    }
    // Elsewhere (the game's own heap blocks): committed and writable for the whole range.
    for (VAddr cursor = virtual_addr; cursor < virtual_addr + size;) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) == 0 ||
            info.State != MEM_COMMIT || !IsWritable(info.Protect)) {
            return false;
        }
        cursor = reinterpret_cast<VAddr>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}

bool IsGuestReadable(VAddr virtual_addr, u64 size) {
    const VAddr end = virtual_addr + std::max<u64>(size, 1);
    const AreaAnswer answer = LookupArea(virtual_addr, end);
    return answer.end >= end && answer.readable;
}

bool TrackingMadeUnreadable(VAddr address) {
    // A page the game released or protected itself is not page tracking's, even when tracking
    // protected it earlier: handling that fault as tracking's read released memory (a fault
    // download's snapshot of 0x2900000000 after a race).
    const VAddr page = address & ~VAddr{0xFFF};
    return NoReadIntersects(page, 0x1000) && IsGuestReadable(page, 0x1000);
}

// Called through volatile pointers: the compiler treats memcpy and the extern "C" XXH3 as calls
// that cannot raise, and a call it believes cannot raise is not covered by the __try scope (a
// fault in it went unhandled).
using CopyFn = void* (*)(void*, const void*, size_t);
using HashFn = XXH64_hash_t (*)(const void*, size_t);
static CopyFn volatile g_guarded_copy = &std::memcpy;
static HashFn volatile g_guarded_hash = &XXH3_64bits;

static bool GuardedCopyRaw(void* dest, const void* source, size_t size) {
    __try {
        g_guarded_copy(dest, source, size);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return true;
}

static bool GuardedHashRaw(const void* source, size_t size, u64* hash) {
    __try {
        *hash = g_guarded_hash(source, size);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return true;
}

bool GuardedCopy(void* dest, const void* source, size_t size) {
    ++g_guarded_reads;
    const bool ok = GuardedCopyRaw(dest, source, size);
    --g_guarded_reads;
    return ok;
}

u64 HashGuestMemory(VAddr address, u64 size) {
    // Pages page tracking made unreadable are read as in shadGT (the fault is handled); only
    // memory the game itself released or protected is read through the safe copy.
    const VAddr end = address + std::max<u64>(size, 1);
    if (const AreaAnswer answer = LookupArea(address, end); answer.end >= end && answer.readable) {
        u64 hash;
        ++g_guarded_reads;
        const bool ok = GuardedHashRaw(reinterpret_cast<const void*>(address), size, &hash);
        --g_guarded_reads;
        if (ok) {
            return hash;
        }
    }
    // Released pool memory and pages the game or page tracking made unreadable: through the
    // copy that reads them without faulting (zeros where nothing is mapped).
    thread_local std::vector<u8> bytes;
    bytes.resize(size);
    Core::Memory::Instance()->CopySparseMemory(address, bytes.data(), size);
    return XXH3_64bits(bytes.data(), size);
}

bool IsPlainData(VAddr virtual_addr, u64 size) {
    const VAddr end = virtual_addr + std::max<u64>(size, 1);
    const AreaAnswer answer = LookupArea(virtual_addr, end);
    return answer.end >= end && !answer.exec;
}

std::vector<ExecRange> ExecRanges(VAddr virtual_addr, u64 size) {
    std::vector<ExecRange> ranges;
    // Page tracking asks this for every protection change; mapped data with no code in its area
    // needs no queries.
    if (IsPlainData(virtual_addr, size)) {
        ranges.push_back({virtual_addr, size, false});
        return ranges;
    }
    const VAddr end = virtual_addr + size;
    VAddr address = virtual_addr;
    while (address < end) {
        MEMORY_BASIC_INFORMATION info;
        VAddr next = end;
        bool executable = false;
        if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info))) {
            next = std::min<VAddr>(end, reinterpret_cast<VAddr>(info.BaseAddress) + info.RegionSize);
            executable = IsCommitted(info) && IsExecutable(info.Protect);
        }
        if (!ranges.empty() && ranges.back().executable == executable) {
            ranges.back().size += next - address;
        } else {
            ranges.push_back({address, next - address, executable});
        }
        address = next;
    }
    return ranges;
}

} // namespace Core::RytyMemory

namespace Core {

struct AddressSpace::Impl {};

AddressSpace::AddressSpace() {
    // Ryty's libc owns the guest address space; nothing is reserved here.
}

AddressSpace::~AddressSpace() = default;

void AddressSpace::Protect(VAddr virtual_addr, u64 size, MemoryPermission perms) {
    const bool read = True(perms & MemoryPermission::Read);
    const bool write = True(perms & MemoryPermission::Write);
    const bool execute = True(perms & MemoryPermission::Execute);
    DWORD protect = PAGE_NOACCESS;
    if (execute) {
        protect = write ? PAGE_EXECUTE_READWRITE : (read ? PAGE_EXECUTE_READ : PAGE_EXECUTE);
    } else if (write) {
        protect = PAGE_READWRITE;
    } else if (read) {
        protect = PAGE_READONLY;
    }
    // FIX-055: never the emulator's own memory (page tracking of a garbage range).
    const auto guest = RytyMemory::GuestRangeOf(virtual_addr);
    if (!guest || virtual_addr + size > guest->end) {
        static std::atomic<u32> refused{};
        if (const u32 n = ++refused; n <= 10 || n % 1000 == 0) {
            LOG_WARNING(Render_Vulkan, "FIX-055: refused to protect {:#x}+{:#x} outside guest "
                        "memory ({} times)", virtual_addr, size, n);
        }
        return;
    }
    // Ryty maps guest memory as many separate views; VirtualProtect cannot cross from one to the
    // next, so protect each committed region of the range on its own. When the range is known to
    // be mapped, try it whole and then in 16 KB pieces (Ryty's view size) before asking Windows
    // where each region ends.
    RytyMemory::NoteReadable(virtual_addr, size, read || execute);
    VAddr address = virtual_addr;
    const VAddr end = virtual_addr + size;
    DWORD old_protect;
    bool views = false;
    if (RytyMemory::IsMappedRange(virtual_addr, size, views)) {
        const bool one_view = (virtual_addr >> 14) == ((end - 1) >> 14);
        if ((!views || one_view) &&
            VirtualProtect(reinterpret_cast<void*>(virtual_addr), size, protect, &old_protect)) {
            return;
        }
        constexpr VAddr ViewSize = 0x4000;
        while (address < end) {
            const VAddr piece_end = std::min<VAddr>(end, (address + ViewSize) & ~(ViewSize - 1));
            if (!VirtualProtect(reinterpret_cast<void*>(address), piece_end - address, protect,
                                &old_protect)) {
                break; // not a 16 KB view: walk the regions from here
            }
            address = piece_end;
        }
    }
    while (address < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) != sizeof(info)) {
            break;
        }
        const VAddr region_end = std::min<VAddr>(
            end, reinterpret_cast<VAddr>(info.BaseAddress) + info.RegionSize);
        if (IsCommitted(info)) {
            if (!VirtualProtect(reinterpret_cast<void*>(address), region_end - address, protect,
                                &old_protect)) {
                LOG_ERROR(Kernel_Vmm, "VirtualProtect {:#x}+{:#x} failed: {}", address,
                          region_end - address, GetLastError());
            }
        }
        address = region_end;
    }
}

boost::icl::interval_set<VAddr> AddressSpace::GetUsableRegions() {
    return {};
}

MemoryManager::MemoryManager() = default;

MemoryManager::~MemoryManager() = default;

u64 MemoryManager::ClampRangeSize(VAddr virtual_addr, u64 size) {
    static constexpr u64 MinSizeToClamp = 1ull << 30;
    if (size < MinSizeToClamp) {
        return size;
    }
    const VAddr end = RytyMemory::MappedAreaEnd(virtual_addr, virtual_addr + size);
    return end == 0 ? size : std::min<u64>(size, end - virtual_addr);
}

void MemoryManager::CopySparseMemory(VAddr virtual_addr, u8* dest, u64 size) {
    if (RytyMemory::IsReadableRange(virtual_addr, size) &&
        RytyMemory::GuardedCopy(dest, reinterpret_cast<const void*>(virtual_addr), size)) {
        return;
    }
    while (size) {
        MEMORY_BASIC_INFORMATION info;
        u64 chunk = size;
        if (::VirtualQuery(reinterpret_cast<void*>(virtual_addr), &info, sizeof(info))) {
            const VAddr region_end = reinterpret_cast<VAddr>(info.BaseAddress) + info.RegionSize;
            chunk = std::min<u64>(size, region_end - virtual_addr);
            if (!IsCommitted(info)) {
                std::memset(dest, 0, chunk);
            } else if (IsReadable(info.Protect)) {
                std::memcpy(dest, reinterpret_cast<const void*>(virtual_addr), chunk);
            } else if (!CopyThroughAlias(virtual_addr, dest, nullptr, chunk)) {
                std::memset(dest, 0, chunk);
            }
        } else {
            std::memset(dest, 0, chunk);
        }
        size -= chunk;
        virtual_addr += chunk;
        dest += chunk;
    }
}

void MemoryManager::CopySparseMemory(VAddr virtual_addr, u8* dest, u64 size,
                                     const Common::ReadCapture* capture) {
    if (!capture) {
        CopySparseMemory(virtual_addr, dest, size);
        return;
    }
    capture->Copy(virtual_addr, dest, size,
                  [this](VAddr address, u8* to, u64 n) { CopySparseMemory(address, to, n); });
}

namespace {
// DIAG-071: the renderer's recent writes into guest memory, for DescribeBackingWrites.
struct BackingWrite {
    std::atomic<u64> seq{};
    VAddr address{};
    u64 size{};
    s64 ns{};
    const char* file{};
    u32 line{};
};
constexpr size_t NumBackingWrites = 16384;
std::array<BackingWrite, NumBackingWrites> g_backing_writes{};
std::atomic<u64> g_backing_write_seq{};

s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void RecordBackingWrite(VAddr address, u64 size, const std::source_location& loc) {
    const u64 seq = g_backing_write_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    auto& write = g_backing_writes[seq % NumBackingWrites];
    write.seq.store(0, std::memory_order_relaxed);
    write.address = address;
    write.size = size;
    write.ns = NowNs();
    write.file = loc.file_name();
    write.line = loc.line();
    write.seq.store(seq, std::memory_order_release);
}
} // namespace

std::string MemoryManager::DescribeBackingWrites(VAddr address, u64 margin) {
    const s64 now = NowNs();
    const VAddr start = address > margin ? address - margin : 0;
    const VAddr end = address + margin;
    std::string out;
    u32 found = 0;
    const u64 newest = g_backing_write_seq.load(std::memory_order_acquire);
    for (u64 seq = newest; seq > 0 && newest - seq < NumBackingWrites && found < 8; --seq) {
        const auto& write = g_backing_writes[seq % NumBackingWrites];
        if (write.seq.load(std::memory_order_acquire) != seq) {
            continue;
        }
        if (write.address < end && start < write.address + write.size) {
            std::string_view file = write.file ? write.file : "?";
            if (const auto slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
                file.remove_prefix(slash + 1);
            }
            out += fmt::format(" [{:#x}+{:#x} {:.1f} ms ago {}:{}]", write.address, write.size,
                               (now - write.ns) / 1e6, file, write.line);
            ++found;
        }
    }
    return out.empty() ? std::string{" none"} : out;
}

bool MemoryManager::TryWriteBacking(void* address, const void* data, u64 size,
                                    std::source_location loc) {
    const auto virtual_addr = reinterpret_cast<VAddr>(address);
    if (!RytyMemory::IsValidMapping(virtual_addr, size)) {
        return false;
    }
    RecordBackingWrite(virtual_addr, size, loc); // DIAG-071
    static const bool cached_alias_first = Common::PerfFeatureEnabled(112);
    if (cached_alias_first && WriteThroughCachedAlias(virtual_addr, data, size)) {
        RytyMemory::NoteOwnWrite(virtual_addr, size); // FIX-073
        return true;
    }
    MEMORY_BASIC_INFORMATION info;
    if (::VirtualQuery(address, &info, sizeof(info)) && IsWritable(info.Protect) &&
        reinterpret_cast<VAddr>(info.BaseAddress) + info.RegionSize >= virtual_addr + size) {
        std::memcpy(address, data, size);
        RytyMemory::NoteOwnWrite(virtual_addr, size); // FIX-073
        return true;
    }
    const bool written = CopyThroughAlias(virtual_addr, nullptr, data, size);
    if (written) {
        RytyMemory::NoteOwnWrite(virtual_addr, size); // FIX-073
    }
    return written;
}

void MemoryManager::NoteEmulatorWrite(VAddr address, u64 size, const void*,
                                      std::source_location loc) {
    RecordBackingWrite(address, size, loc); // DIAG-071
}

} // namespace Core

namespace Libraries::Kernel {

// PS4 Pro (Neo) mode follows the General.neo_mode setting, as in shadGT itself, so the memory
// layout, tiling and Gnm init packets match what the game sees from its own sceKernelIsNeoMode.
s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    return EmulatorSettings.IsNeo() ? 1 : 0;
}

// libkernel.prx exports these by NID with the System V ABI.
u64 PS4_SYSV_ABI sceKernelReadTsc() {
    using Fn = u64(PS4_SYSV_ABI*)();
    static const auto read_tsc = RytyFunction<Fn>("libkernel.prx", "-2IRUCO--PM");
    return read_tsc ? read_tsc() : __rdtsc();
}

u64 PS4_SYSV_ABI sceKernelGetProcessTime() {
    using Fn = u64(PS4_SYSV_ABI*)();
    static const auto process_time = RytyFunction<Fn>("libkernel.prx", "4J2sUJmuHZQ");
    return process_time ? process_time() : 0;
}

s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* ver) {
    // Ryty does not export this; GT Sport's process parameters (PT_SCE_PROCPARAM) give SDK 5.05.
    if (ver == nullptr) {
        return 0x80020016; // ORBIS_KERNEL_ERROR_EINVAL
    }
    *ver = 0x05050001;
    return 0;
}

u64 PS4_SYSV_ABI sceKernelGetTscFrequency() {
    using Fn = u64(PS4_SYSV_ABI*)();
    static const auto frequency = RytyFunction<Fn>("libkernel.prx", "1j3S3n-tTW4");
    return frequency ? frequency() : 0;
}

// Event queues belong to Ryty's libkernel; these proxies forward through its C interface.
namespace {

using AddEventFn = int (*)(s64, std::uintptr_t, s16, u16, u32, std::intptr_t, void*, int, int);
using TriggerEventFn = int (*)(s64, std::uintptr_t, s16, void*);
using DeleteEventFn = int (*)(s64, std::uintptr_t, s16);
using ExistsFn = int (*)(s64);

constexpr int TriggerModeData = 0;
constexpr int TriggerModePs4Display = 1;
constexpr int TriggerModeUser = 2;

std::mutex g_equeue_mutex;
std::unordered_map<OrbisKernelEqueue, std::unique_ptr<EqueueInternal>> g_equeues;

} // namespace

EqueueInternal* GetEqueue(OrbisKernelEqueue eq) {
    static const auto exists = RytyFunction<ExistsFn>("libkernel.prx", "RytyEqueueExists_nid_no_patch");
    if (exists == nullptr || !exists(eq)) {
        return nullptr;
    }
    std::scoped_lock lock{g_equeue_mutex};
    auto& proxy = g_equeues[eq];
    if (!proxy) {
        proxy = std::make_unique<EqueueInternal>(eq, "ryty");
    }
    return proxy.get();
}

bool EqueueInternal::AddEvent(EqueueEvent& event) {
    static const auto add = RytyFunction<AddEventFn>("libkernel.prx", "RytyEqueueAddEvent_nid_no_patch");
    if (add == nullptr) {
        return false;
    }
    int mode = TriggerModeData;
    if (event.event.filter == OrbisKernelEvent::Filter::VideoOut) {
        mode = TriggerModePs4Display;
    } else if (event.event.filter == OrbisKernelEvent::Filter::User) {
        mode = TriggerModeUser;
    }
    return add(m_handle, event.event.ident, event.event.filter, event.event.flags,
               event.event.fflags, static_cast<std::intptr_t>(event.event.data), event.event.udata,
               mode, event.IsTriggered() ? 1 : 0) == 0;
}

bool EqueueInternal::TriggerEvent(u64 ident, s16 filter, void* trigger_data) {
    RyDiag::Mark(RyDiag::Zone::GpuEventSent,
                 (u64(u16(filter)) << 48) | (ident & 0xFFFFFFFFFFFFull));
    static const auto trigger =
        RytyFunction<TriggerEventFn>("libkernel.prx", "RytyEqueueTriggerEvent_nid_no_patch");
    return trigger != nullptr && trigger(m_handle, ident, filter, trigger_data) == 0;
}

bool EqueueInternal::RemoveEvent(u64 id, s16 filter) {
    static const auto remove =
        RytyFunction<DeleteEventFn>("libkernel.prx", "RytyEqueueDeleteEvent_nid_no_patch");
    return remove != nullptr && remove(m_handle, id, filter) == 0;
}

} // namespace Libraries::Kernel

// shadGT's online overlays are not part of the Ryty build.
namespace ImGui::Friends {
void Register() {}
} // namespace ImGui::Friends
namespace ImGui::InvitationPrompt {
void Register() {}
} // namespace ImGui::InvitationPrompt
namespace ImGui::ShadNetNotify {
void Register() {}
} // namespace ImGui::ShadNetNotify

// shadGT's HLE libraries start their worker threads (Kernel::Thread) through the guest pthread API.
// Those workers never run guest code, so in the Ryty build they are plain host threads. The NP
// bundle hands the game's own threads and mutexes to these functions, so it forwards them to Ryty's
// libkernel instead (ryty_np_shims.cpp).
#ifndef RYTY_GUEST_PTHREADS
namespace Libraries::Kernel {

namespace {
struct HostThread {
    std::thread thread;
    void* result = nullptr;
};
} // namespace

int PS4_SYSV_ABI posix_pthread_attr_init(PthreadAttrT* attr) {
    if (attr) {
        *attr = nullptr;
    }
    return 0;
}

int PS4_SYSV_ABI posix_pthread_attr_destroy(PthreadAttrT* attr) {
    if (attr) {
        *attr = nullptr;
    }
    return 0;
}

int PS4_SYSV_ABI posix_pthread_create(PthreadT* thread, const PthreadAttrT*,
                                      PthreadEntryFunc start_routine, void* arg) {
    auto* host = new HostThread;
    host->thread = std::thread([host, start_routine, arg] { host->result = start_routine(arg); });
    *thread = reinterpret_cast<PthreadT>(host);
    return 0;
}

int PS4_SYSV_ABI posix_pthread_join(PthreadT pthread, void** thread_return) {
    auto* host = reinterpret_cast<HostThread*>(pthread);
    if (host->thread.joinable()) {
        host->thread.join();
    }
    if (thread_return) {
        *thread_return = host->result;
    }
    delete host;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_detach(PthreadT pthread) {
    auto* host = reinterpret_cast<HostThread*>(pthread);
    host->thread.detach();
    return 0;
}

} // namespace Libraries::Kernel
#endif // RYTY_GUEST_PTHREADS
