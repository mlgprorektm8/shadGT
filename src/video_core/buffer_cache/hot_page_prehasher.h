// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

#include <tsl/robin_map.h>
#include <xxhash.h>

#include "common/types.h"

namespace VideoCore {

/// PERF-049: hashes the PERF-033 hot pages of a new upload epoch on background threads, so the
/// recorder finds most hashes ready instead of hashing them itself (about 8% of its time in
/// races). A hash is used only in the epoch it was taken in: the epoch had already started when
/// hashing began, as it has when the recorder hashes, and within one epoch the memory submitted
/// work reads is stable (PERF-012). A page with no ready hash is hashed by the caller as before.
class HotPagePrehasher {
public:
    /// Fills the list with the pages worth hashing (those with an upload record).
    using PageSource = std::function<void(std::vector<VAddr>&)>;
    /// Runs the function while no guest memory can be unmapped; false skips the page.
    using MappingGuard = std::function<void(const std::function<void()>&)>;
    using PageValid = std::function<bool(VAddr)>;

    struct Batch {
        u32 epoch{};
        std::vector<VAddr> pages;
        tsl::robin_map<VAddr, u32> index;
        std::unique_ptr<std::atomic<u64>[]> hashes;
        std::unique_ptr<std::atomic<bool>[]> ready;
        std::atomic<size_t> next{};

        /// The hash taken in `epoch_`, if one is ready.
        bool Lookup(VAddr page, u32 epoch_, u64& hash) const {
            if (epoch_ != epoch) {
                return false;
            }
            const auto it = index.find(page);
            if (it == index.end() || !ready[it->second].load(std::memory_order_acquire)) {
                return false;
            }
            hash = hashes[it->second].load(std::memory_order_relaxed);
            return true;
        }
    };

    HotPagePrehasher(u32 num_threads, size_t page_size_, const std::atomic<u32>& epoch_counter_,
                     PageSource source_, MappingGuard guard_, PageValid valid_)
        : page_size{page_size_}, epoch_counter{epoch_counter_}, source{std::move(source_)},
          guard{std::move(guard_)}, valid{std::move(valid_)} {
        for (u32 i = 0; i < num_threads; ++i) {
            threads.emplace_back([this](std::stop_token stop) { Work(stop); });
        }
    }

    ~HotPagePrehasher() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        {
            std::scoped_lock lk{mutex};
        }
        cv.notify_all();
        threads.clear();
    }

    HotPagePrehasher(const HotPagePrehasher&) = delete;
    HotPagePrehasher& operator=(const HotPagePrehasher&) = delete;

    /// A new upload epoch began; any thread.
    void OnEpoch(u32 epoch) {
        {
            std::scoped_lock lk{mutex};
            wanted_epoch = epoch;
        }
        cv.notify_all();
    }

    /// The current batch (may belong to an older epoch; Lookup checks).
    std::shared_ptr<const Batch> Current() const {
        std::scoped_lock lk{mutex};
        return batch;
    }

    /// Blocks until the batch of `epoch` is fully hashed or abandoned (tests).
    void WaitIdle(u32 epoch) {
        std::unique_lock lk{mutex};
        idle_cv.wait(lk, [&] { return finished_epoch >= epoch; });
    }

private:
    void Work(std::stop_token stop) {
        while (true) {
            std::shared_ptr<Batch> local;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, stop, [&] {
                    return wanted_epoch != 0 && (!batch || batch->epoch != wanted_epoch ||
                                                 batch->next.load() < batch->pages.size());
                });
                if (stop.stop_requested()) {
                    return;
                }
                if (!batch || batch->epoch != wanted_epoch) {
                    batch = Build(wanted_epoch);
                }
                local = batch;
            }
            HashBatch(*local);
            {
                std::scoped_lock lk{mutex};
                if (local->next.load() >= local->pages.size()) {
                    finished_epoch = std::max(finished_epoch, local->epoch);
                }
            }
            idle_cv.notify_all();
        }
    }

    std::shared_ptr<Batch> Build(u32 epoch) {
        auto next_batch = std::make_shared<Batch>();
        next_batch->epoch = epoch;
        source(next_batch->pages);
        const size_t count = next_batch->pages.size();
        next_batch->index.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            next_batch->index.emplace(next_batch->pages[i], static_cast<u32>(i));
        }
        next_batch->hashes = std::make_unique<std::atomic<u64>[]>(count);
        next_batch->ready = std::make_unique<std::atomic<bool>[]>(count);
        return next_batch;
    }

    void HashBatch(Batch& work) {
        constexpr size_t Chunk = 16;
        const size_t count = work.pages.size();
        while (true) {
            const size_t first = work.next.fetch_add(Chunk, std::memory_order_relaxed);
            if (first >= count) {
                return;
            }
            // A later epoch makes this batch useless: the rest is skipped.
            if (epoch_counter.load(std::memory_order_acquire) != work.epoch) {
                work.next.store(count, std::memory_order_relaxed);
                return;
            }
            const size_t last = std::min(first + Chunk, count);
            guard([&] {
                for (size_t i = first; i < last; ++i) {
                    const VAddr page = work.pages[i];
                    if (!valid(page)) {
                        continue;
                    }
                    work.hashes[i].store(XXH3_64bits(reinterpret_cast<const void*>(page), page_size),
                                         std::memory_order_relaxed);
                    work.ready[i].store(true, std::memory_order_release);
                }
            });
        }
    }

    const size_t page_size;
    const std::atomic<u32>& epoch_counter;
    PageSource source;
    MappingGuard guard;
    PageValid valid;
    mutable std::mutex mutex;
    std::condition_variable_any cv;
    std::condition_variable_any idle_cv;
    u32 wanted_epoch{};
    u32 finished_epoch{};
    std::shared_ptr<Batch> batch;
    std::vector<std::jthread> threads;
};

} // namespace VideoCore
