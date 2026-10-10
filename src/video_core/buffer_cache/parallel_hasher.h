// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

#include <xxhash.h>

#include "common/types.h"

#if defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace VideoCore {

/// PERF-046: hashes many pages with XXH3 on the calling thread and a few helpers. GT Sport's
/// races check about 7 GB of unchanged PERF-012 hot pages per 2 s (PERF-033), once per upload
/// epoch, on the recorder thread. The caller starts hashing at once; helpers that wake in time
/// take pages too. Results are the same as hashing on one thread. One caller at a time.
class ParallelPageHasher {
public:
    explicit ParallelPageHasher(u32 num_helpers) {
        for (u32 i = 0; i < num_helpers; ++i) {
            threads.emplace_back([this](std::stop_token stop) { Work(stop); });
        }
    }

    ~ParallelPageHasher() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        {
            std::scoped_lock lk{mutex};
        }
        cv.notify_all();
        threads.clear();
    }

    ParallelPageHasher(const ParallelPageHasher&) = delete;
    ParallelPageHasher& operator=(const ParallelPageHasher&) = delete;

    /// Writes XXH3_64bits(pages[i], page_size) to hashes[i] for every page.
    void Hash(std::span<const u8* const> pages_, size_t page_size_, std::span<u64> hashes_) {
        {
            // Helpers join a batch only under the lock, so once none is active none can still
            // be working on the previous batch when this one is handed over.
            std::unique_lock lk{mutex};
            while (active != 0) {
                lk.unlock();
                Pause();
                lk.lock();
            }
            pages = pages_;
            hashes = hashes_;
            page_size = page_size_;
            next.store(0, std::memory_order_relaxed);
            done.store(0, std::memory_order_relaxed);
            ++generation;
        }
        cv.notify_all();
        HashPages();
        while (done.load(std::memory_order_acquire) != pages_.size()) {
            Pause();
        }
    }

private:
    static void Pause() {
#if defined(_MSC_VER) || defined(__x86_64__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }

    void Work(std::stop_token stop) {
        u64 seen = 0;
        while (true) {
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, stop, [&] { return generation != seen; });
                if (stop.stop_requested()) {
                    return;
                }
                seen = generation;
                ++active;
            }
            HashPages();
            std::scoped_lock lk{mutex};
            --active;
        }
    }

    void HashPages() {
        const size_t count = pages.size();
        for (size_t i = next.fetch_add(1, std::memory_order_relaxed); i < count;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
            hashes[i] = XXH3_64bits(pages[i], page_size);
            done.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    std::span<const u8* const> pages;
    std::span<u64> hashes;
    size_t page_size{};
    std::atomic<size_t> next{};
    std::atomic<size_t> done{};
    u32 active{}; ///< Helpers in a batch; changed under the mutex.
    u64 generation{};
    std::mutex mutex;
    std::condition_variable_any cv;
    std::vector<std::jthread> threads;
};

} // namespace VideoCore
