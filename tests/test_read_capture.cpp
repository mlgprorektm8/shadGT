// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <xxhash.h>

#include "common/read_capture.h"

namespace {

using Common::ReadCapture;
constexpr u64 Page = ReadCapture::PageSize;

std::array<u8, Page> Filled(u8 value) {
    std::array<u8, Page> bytes;
    bytes.fill(value);
    return bytes;
}

TEST(ReadCapture, FindsCopiesWithTheirHash) {
    ReadCapture capture{64};
    const auto a = Filled(1);
    const auto b = Filled(2);
    ASSERT_TRUE(capture.Add(0x10000, a.data()));
    ASSERT_TRUE(capture.Add(0x20000, b.data()));
    u64 hash = 0;
    const u8* copy = capture.Find(0x20000, &hash);
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy[0], 2);
    EXPECT_EQ(copy[Page - 1], 2);
    EXPECT_EQ(hash, XXH3_64bits(b.data(), Page));
    EXPECT_EQ(capture.Find(0x30000), nullptr);
    EXPECT_EQ(capture.NumPages(), 2u);
}

TEST(ReadCapture, CopyIsIndependentOfLaterSourceWrites) {
    ReadCapture capture{8};
    auto source = Filled(7);
    ASSERT_TRUE(capture.Add(0x5000, source.data()));
    source.fill(9); // the game rewrites the page after the fence
    EXPECT_EQ(capture.Find(0x5000)[100], 7);
}

TEST(ReadCapture, WatchedPagesAreListedButHaveNoCopy) {
    ReadCapture capture{8};
    ASSERT_TRUE(capture.AddWatched(0x7000));
    EXPECT_TRUE(capture.Lists(0x7000));
    EXPECT_FALSE(capture.Has(0x7000));
    EXPECT_FALSE(capture.Lists(0x8000));
}

TEST(ReadCapture, CopyMixesCapturedAndLivePages) {
    ReadCapture capture{8};
    const auto captured = Filled(0xAA);
    ASSERT_TRUE(capture.Add(0x1000, captured.data()));
    std::vector<u8> out(Page * 2);
    // From the middle of the captured page into the next (live) one.
    capture.Copy(0x1800, out.data(), Page, [](VAddr address, u8* to, u64 n) {
        EXPECT_EQ(address, 0x2000u);
        std::fill_n(to, n, u8{0x55});
    });
    EXPECT_EQ(out[0], 0xAA);
    EXPECT_EQ(out[Page / 2 - 1], 0xAA);
    EXPECT_EQ(out[Page / 2], 0x55);
    EXPECT_EQ(out[Page - 1], 0x55);
}

TEST(ReadCapture, FullCaptureRefusesMorePages) {
    ReadCapture capture{2};
    const auto bytes = Filled(3);
    EXPECT_TRUE(capture.Add(0x1000, bytes.data()));
    EXPECT_TRUE(capture.Add(0x2000, bytes.data()));
    EXPECT_FALSE(capture.Add(0x3000, bytes.data()));
    EXPECT_TRUE(capture.Full());
    EXPECT_EQ(capture.Find(0x3000), nullptr);
}

TEST(ReadCapture, ReaderSeesCompletePagesWhileWriterAdds) {
    constexpr u32 Count = 2000;
    ReadCapture capture{Count};
    std::atomic<bool> done{false};
    std::atomic<u32> bad{0};
    std::thread reader([&] {
        while (!done.load()) {
            for (u32 i = 1; i <= Count; ++i) {
                u64 hash;
                if (const u8* copy = capture.Find(VAddr(i) * Page, &hash)) {
                    if (copy[0] != u8(i) || copy[Page - 1] != u8(i) ||
                        hash != XXH3_64bits(copy, Page)) {
                        ++bad;
                    }
                }
            }
        }
    });
    for (u32 i = 1; i <= Count; ++i) {
        const auto bytes = Filled(u8(i));
        ASSERT_TRUE(capture.Add(VAddr(i) * Page, bytes.data()));
    }
    done = true;
    reader.join();
    EXPECT_EQ(bad.load(), 0u);
    EXPECT_EQ(capture.NumPages(), Count);
}

TEST(ReadCapture, IdsDiffer) {
    ReadCapture a{1};
    ReadCapture b{1};
    EXPECT_NE(a.Id(), b.Id());
}

} // namespace
