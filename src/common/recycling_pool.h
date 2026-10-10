// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <new>
#include <vector>

#if defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Common {

/// PERF-056: blocks allocated on one thread and freed on another (the draw pipe's jobs, made on
/// the command thread and run on the recorder) are kept in per-size free lists instead of going
/// back to the system heap, whose cross-thread frees are slow. Thread-safe; sizes above the
/// largest class use the heap directly.
class RecyclingPool {
public:
    static constexpr std::array<size_t, 6> ClassSizes{64, 128, 256, 512, 1024, 2048};
    static constexpr size_t MaxKept = 4096; // per class

    static void* Allocate(size_t size) {
        const size_t index = ClassOf(size);
        if (index == ClassSizes.size()) {
            return ::operator new(size);
        }
        auto& list = Instance().lists[index];
        {
            Guard guard{list.lock};
            if (!list.blocks.empty()) {
                void* block = list.blocks.back();
                list.blocks.pop_back();
                return block;
            }
        }
        return ::operator new(ClassSizes[index]);
    }

    static void Free(void* block, size_t size) {
        if (!block) {
            return;
        }
        const size_t index = ClassOf(size);
        if (index == ClassSizes.size()) {
            ::operator delete(block);
            return;
        }
        auto& list = Instance().lists[index];
        {
            Guard guard{list.lock};
            if (list.blocks.size() < MaxKept) {
                list.blocks.push_back(block);
                return;
            }
        }
        ::operator delete(block);
    }

    static constexpr size_t ClassOf(size_t size) {
        for (size_t i = 0; i < ClassSizes.size(); ++i) {
            if (size <= ClassSizes[i]) {
                return i;
            }
        }
        return ClassSizes.size();
    }

private:
    struct SpinLock {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
    };
    struct Guard {
        explicit Guard(SpinLock& lock_) : lock{lock_} {
            while (lock.flag.test_and_set(std::memory_order_acquire)) {
                while (lock.flag.test(std::memory_order_relaxed)) {
#if defined(_MSC_VER) || defined(__x86_64__)
                    _mm_pause();
#endif
                }
            }
        }
        ~Guard() {
            lock.flag.clear(std::memory_order_release);
        }
        SpinLock& lock;
    };
    struct List {
        SpinLock lock;
        std::vector<void*> blocks;
        List() {
            blocks.reserve(MaxKept);
        }
    };

    static RecyclingPool& Instance() {
        // Never destroyed: blocks may be freed during static destruction.
        static RecyclingPool* pool = new RecyclingPool;
        return *pool;
    }

    std::array<List, ClassSizes.size()> lists;
};

} // namespace Common

namespace Common {

/// PERF-056: objects (vectors with their capacity) handed from one thread to another and back.
template <typename T, size_t MaxKept = 256>
class Recycler {
public:
    T Take() {
        Lock();
        if (items.empty()) {
            Unlock();
            return T{};
        }
        T item = std::move(items.back());
        items.pop_back();
        Unlock();
        return item;
    }

    void Give(T&& item) {
        Lock();
        if (items.size() < MaxKept) {
            items.push_back(std::move(item));
        }
        Unlock();
    }

private:
    void Lock() {
        while (flag.test_and_set(std::memory_order_acquire)) {
            while (flag.test(std::memory_order_relaxed)) {
#if defined(_MSC_VER) || defined(__x86_64__)
                _mm_pause();
#endif
            }
        }
    }
    void Unlock() {
        flag.clear(std::memory_order_release);
    }

    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    std::vector<T> items;
};

} // namespace Common
