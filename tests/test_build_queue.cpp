// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/build_queue.h"

using Vulkan::BuildQueue;
using namespace std::chrono_literals;

namespace {
std::string PopName(BuildQueue& queue, std::stop_token stop, bool* background = nullptr) {
    std::string name;
    BuildQueue::Job job;
    bool is_background = false;
    if (!queue.Pop(job, is_background, stop)) {
        return "stopped";
    }
    job();
    if (background) {
        *background = is_background;
    }
    return name;
}
} // namespace

TEST(BuildQueue, UrgentThenNormalThenBackground) {
    BuildQueue queue{4};
    std::vector<std::string> order;
    queue.Push([&] { order.push_back("background"); }, BuildQueue::Kind::Background);
    queue.Push([&] { order.push_back("normal1"); }, BuildQueue::Kind::Normal);
    queue.Push([&] { order.push_back("normal2"); }, BuildQueue::Kind::Normal);
    queue.Push([&] { order.push_back("urgent"); }, BuildQueue::Kind::Urgent);
    std::stop_source stop;
    for (int i = 0; i < 4; ++i) {
        bool background = false;
        PopName(queue, stop.get_token(), &background);
        if (background) {
            queue.FinishedBackground();
        }
    }
    EXPECT_EQ(order, (std::vector<std::string>{"urgent", "normal1", "normal2", "background"}));
}

TEST(BuildQueue, BackgroundJobsAreCapped) {
    BuildQueue queue{1};
    for (int i = 0; i < 3; ++i) {
        queue.Push([] {}, BuildQueue::Kind::Background);
    }
    std::stop_source stop;
    BuildQueue::Job job;
    bool background = false;
    ASSERT_TRUE(queue.Pop(job, background, stop.get_token()));
    EXPECT_TRUE(background);
    // A second worker waits while the first background job runs, but takes normal work.
    std::atomic<bool> got{};
    std::thread second{[&] {
        BuildQueue::Job next;
        bool next_background = false;
        if (queue.Pop(next, next_background, stop.get_token())) {
            EXPECT_TRUE(next_background);
            got = true;
            queue.FinishedBackground();
        }
    }};
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(got.load()) << "only one background job runs at a time";
    queue.FinishedBackground();
    second.join();
    EXPECT_TRUE(got.load());
    EXPECT_EQ(queue.BackgroundQueued(), 1u);
}

TEST(BuildQueue, PauseHoldsBackgroundButNotNormalJobs) {
    BuildQueue queue{4};
    queue.Push([] {}, BuildQueue::Kind::Background);
    const auto start = BuildQueue::Clock::now();
    queue.PauseBackground(start + 100ms);
    queue.Push([] {}, BuildQueue::Kind::Normal);
    std::stop_source stop;
    BuildQueue::Job job;
    bool background = true;
    ASSERT_TRUE(queue.Pop(job, background, stop.get_token()));
    EXPECT_FALSE(background);
    EXPECT_LT(BuildQueue::Clock::now() - start, 50ms);
    ASSERT_TRUE(queue.Pop(job, background, stop.get_token()));
    EXPECT_TRUE(background);
    EXPECT_GE(BuildQueue::Clock::now() - start, 100ms) << "background waited out the pause";
    queue.FinishedBackground();
}

TEST(BuildQueue, StopWakesAWaitingWorker) {
    BuildQueue queue{2};
    std::stop_source stop;
    std::atomic<bool> returned{};
    std::thread worker{[&] {
        BuildQueue::Job job;
        bool background = false;
        EXPECT_FALSE(queue.Pop(job, background, stop.get_token()));
        returned = true;
    }};
    std::this_thread::sleep_for(20ms);
    stop.request_stop();
    queue.NotifyAll();
    worker.join();
    EXPECT_TRUE(returned.load());
}
