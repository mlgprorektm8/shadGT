// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include "video_core/buffer_cache/uploaded_pages.h"

using VideoCore::UploadedPageContents;

TEST(UploadedPages, SkipsAPageOnlyWhileItsBytesAndHotStayAreTheSame) {
    UploadedPageContents pages;
    constexpr VAddr page = 0x10000;
    EXPECT_FALSE(pages.Unchanged(page, 1, 0xabc)); // never uploaded
    pages.Record(page, 1, 0xabc, 0xabc);
    EXPECT_TRUE(pages.Unchanged(page, 1, 0xabc));
    // The CPU changed the bytes.
    EXPECT_FALSE(pages.Unchanged(page, 1, 0xdef));
    // The page left the hot set (a GPU write) and became hot again: the GPU copy may hold GPU
    // data even if the guest bytes look the same.
    EXPECT_FALSE(pages.Unchanged(page, 2, 0xabc));
}

TEST(UploadedPages, KeepsNoRecordForBytesThatChangedDuringTheUpload) {
    UploadedPageContents pages;
    constexpr VAddr page = 0x20000;
    pages.Record(page, 3, 0x111, 0x111);
    pages.Record(page, 3, 0x222, 0x333); // written while copied: what the GPU got is unknown
    EXPECT_FALSE(pages.Unchanged(page, 3, 0x111));
    EXPECT_FALSE(pages.Unchanged(page, 3, 0x222));
    EXPECT_FALSE(pages.Unchanged(page, 3, 0x333));
    EXPECT_EQ(pages.Size(), 0u);
}

TEST(UploadedPages, ForgetsAPartlyUploadedPage) {
    UploadedPageContents pages;
    constexpr VAddr page = 0x30000;
    pages.Record(page, 1, 0x5, 0x5);
    pages.Forget(page);
    EXPECT_FALSE(pages.Unchanged(page, 1, 0x5));
}
