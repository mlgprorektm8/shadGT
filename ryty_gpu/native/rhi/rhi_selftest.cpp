// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NAT-012: self test of the native renderer's backend, outside the game. A test harness loads the
// renderer library and calls RytyNativeRhiSelfTest; it creates its own Vulkan device, uploads a
// texture, draws it with a pipeline built through the backend interface (also from four threads
// recording their own command lists at once) and reads the picture back.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <fmt/format.h>

#include "ryty_gpu/native/rhi/rhi.h"
#include "ryty_gpu/native/rhi/rhi_vulkan.h"
#include "video_core/host_shaders/fs_tri_vert.h"
#include "video_core/host_shaders/native_test_comp.h"
#include "video_core/host_shaders/native_test_frag.h"

namespace {

using namespace Native::Rhi;

constexpr u32 kVkFormatR8G8B8A8Unorm = 37;
constexpr u32 kSize = 64;

struct Rgba {
    u8 r, g, b, a;
    bool operator==(const Rgba&) const = default;
};

bool RunSelfTest(std::string& log) {
    std::string error;
    auto device = CreateStandaloneVulkanDevice(&error);
    if (!device) {
        log += "no device: " + error + "\n";
        return false;
    }
    const DeviceCaps& caps = device->Caps();
    log += fmt::format("device: host import {} (alignment {}), push constants {} bytes, push "
                       "descriptors {}\n",
                       caps.import_host_memory, caps.host_import_alignment,
                       caps.max_push_constant_bytes, caps.max_push_bindings);

    // A 4x4 texture in four 2x2 colours: red, green / blue, white.
    const std::array<Rgba, 4> quadrant{{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255},
                                        {255, 255, 255, 255}}};
    auto upload = device->CreateBuffer({.size = 4 * 4 * 4,
                                        .usage = BufferUsageTransferSrc,
                                        .memory = MemoryKind::Upload});
    auto readback = device->CreateBuffer({.size = kSize * kSize * 4,
                                          .usage = BufferUsageTransferDst,
                                          .memory = MemoryKind::Readback});
    if (!upload || !readback || !upload->Mapped() || !readback->Mapped()) {
        log += "buffers failed\n";
        return false;
    }
    for (u32 y = 0; y < 4; ++y) {
        for (u32 x = 0; x < 4; ++x) {
            const Rgba c = quadrant[(y / 2) * 2 + x / 2];
            std::memcpy(upload->Mapped() + (y * 4 + x) * 4, &c, 4);
        }
    }
    auto texture = device->CreateImage({.format = FormatFromVk(kVkFormatR8G8B8A8Unorm),
                                        .width = 4,
                                        .height = 4,
                                        .usage = ImageUsageSampled | ImageUsageTransferDst});
    auto target = device->CreateImage({.format = FormatFromVk(kVkFormatR8G8B8A8Unorm),
                                       .width = kSize,
                                       .height = kSize,
                                       .usage = ImageUsageColorTarget | ImageUsageTransferSrc});
    if (!texture || !target) {
        log += "images failed\n";
        return false;
    }
    auto texture_view = device->CreateImageView(texture.get(), {});
    auto target_view = device->CreateImageView(target.get(), {});
    auto sampler = device->CreateSampler({.min_filter = Filter::Nearest,
                                          .mag_filter = Filter::Nearest,
                                          .mip_filter = Filter::Nearest,
                                          .address = {AddressMode::ClampToEdge,
                                                      AddressMode::ClampToEdge,
                                                      AddressMode::ClampToEdge}});
    const std::array<BindingSlot, 2> slots{{{0, BindingType::SampledImage, StagePixel},
                                            {1, BindingType::Sampler, StagePixel}}};
    auto layout = device->CreatePipelineLayout(
        {.bindings = slots, .push_constant_bytes = 16, .push_constant_stages = StagePixel});
    if (!texture_view || !target_view || !sampler || !layout) {
        log += "views, sampler or layout failed\n";
        return false;
    }
    const std::array<ShaderCode, 2> shaders{{
        {ShaderStage::Vertex, std::span<const u32>{FS_TRI_VERT}},
        {ShaderStage::Pixel, std::span<const u32>{NATIVE_TEST_FRAG}},
    }};
    GraphicsPipelineDesc pipeline_desc{.layout = layout.get(), .shaders = shaders, .color_count = 1};
    pipeline_desc.color_formats[0] = FormatFromVk(kVkFormatR8G8B8A8Unorm);
    auto pipeline = device->CreateGraphicsPipeline(pipeline_desc);
    if (!pipeline) {
        log += "pipeline failed\n";
        return false;
    }
    auto fence = device->CreateFence(0);

    const std::array<Binding, 2> bindings{{
        {.slot = 0, .type = BindingType::SampledImage, .view = texture_view.get()},
        {.slot = 1, .type = BindingType::Sampler, .sampler = sampler.get()},
    }};
    const auto draw = [&](CommandList* list, Rect scissor, std::array<float, 4> tint, bool clear) {
        RenderingDesc rendering{.width = kSize, .height = kSize, .color_count = 1};
        rendering.colors[0] = {.view = target_view.get(),
                               .load = clear ? LoadOp::Clear : LoadOp::Load,
                               .clear = {0, 0, 0, 0}};
        list->BeginRendering(rendering);
        const Viewport viewport{0, 0, float(kSize), float(kSize), 0, 1};
        list->SetViewports({&viewport, 1});
        list->SetScissors({&scissor, 1});
        list->BindPipeline(pipeline.get());
        list->PushBindings(bindings);
        list->PushConstants(0, {reinterpret_cast<const u8*>(tint.data()), sizeof(tint)});
        list->Draw(3, 1, 0, 0);
        list->EndRendering();
    };
    const BufferImageCopy whole_texture{.extent = {4, 4, 1}};
    const BufferImageCopy whole_target{.extent = {kSize, kSize, 1}};
    const auto read = [&](u32 x, u32 y) {
        Rgba c;
        std::memcpy(&c, readback->Mapped() + (y * kSize + x) * 4, 4);
        return c;
    };
    bool ok = true;
    const auto expect = [&](const char* what, u32 x, u32 y, Rgba want) {
        const Rgba got = read(x, y);
        if (!(got == want)) {
            ok = false;
            log += fmt::format("{}: pixel {},{} is {},{},{},{}, expected {},{},{},{}\n", what, x, y,
                               got.r, got.g, got.b, got.a, want.r, want.g, want.b, want.a);
        }
    };

    // 1. One list: upload, draw the whole target, read back.
    {
        CommandList* list = device->AcquireCommandList();
        list->Begin();
        list->CopyBufferToImage(upload.get(), texture.get(), {&whole_texture, 1});
        list->Barrier(ScopeTransfer, ScopeDraw);
        draw(list, {0, 0, kSize, kSize}, {1, 1, 1, 1}, true);
        list->Barrier(ScopeDraw, ScopeTransfer);
        list->CopyImageToBuffer(target.get(), readback.get(), {&whole_target, 1});
        list->Barrier(ScopeTransfer, ScopeHost);
        list->End();
        CommandList* lists[] = {list};
        device->GraphicsQueue().Submit(lists, {}, {fence.get(), 1});
        if (!fence->Wait(1, 5'000'000'000ULL)) {
            log += "single list: GPU did not finish\n";
            return false;
        }
        expect("single list", 8, 8, quadrant[0]);
        expect("single list", 56, 8, quadrant[1]);
        expect("single list", 8, 56, quadrant[2]);
        expect("single list", 56, 56, quadrant[3]);
        log += fmt::format("single list: {}\n", ok ? "picture correct" : "WRONG");
    }

    // 2. Four threads record a list each (one quadrant, its own tint) at the same time; one
    //    submission runs them in order, then a fifth list reads back.
    {
        const std::array<std::array<float, 4>, 4> tints{{{1, 1, 1, 1}, {0, 1, 1, 1}, {1, 0, 1, 1},
                                                         {1, 1, 0, 1}}};
        std::array<CommandList*, 5> lists{};
        std::vector<std::thread> threads;
        for (u32 i = 0; i < 4; ++i) {
            threads.emplace_back([&, i] {
                CommandList* list = device->AcquireCommandList();
                list->Begin();
                const Rect scissor{s32(i % 2 * kSize / 2), s32(i / 2 * kSize / 2), kSize / 2, kSize / 2};
                for (u32 repeat = 0; repeat < 250; ++repeat) {
                    draw(list, scissor, tints[i], false);
                }
                list->End();
                lists[i] = list;
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        CommandList* tail = device->AcquireCommandList();
        tail->Begin();
        tail->Barrier(ScopeDraw, ScopeTransfer);
        tail->CopyImageToBuffer(target.get(), readback.get(), {&whole_target, 1});
        tail->Barrier(ScopeTransfer, ScopeHost);
        tail->End();
        lists[4] = tail;
        const auto start = std::chrono::steady_clock::now();
        device->GraphicsQueue().Submit(lists, {}, {fence.get(), 2});
        if (!fence->Wait(2, 5'000'000'000ULL)) {
            log += "four threads: GPU did not finish\n";
            return false;
        }
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto tinted = [&](u32 i) {
            const Rgba c = quadrant[i];
            return Rgba{u8(c.r * tints[i][0]), u8(c.g * tints[i][1]), u8(c.b * tints[i][2]), c.a};
        };
        const bool before = ok;
        expect("four threads", 8, 8, tinted(0));
        expect("four threads", 56, 8, tinted(1));
        expect("four threads", 8, 56, tinted(2));
        expect("four threads", 56, 56, tinted(3));
        log += fmt::format("four threads, 1,000 draws: {} ({:.2f} ms on the GPU)\n",
                           ok == before ? "picture correct" : "WRONG", ms);
    }

    // 3. Recording cost: 4,000 draws (a race frame) into one list on one thread.
    {
        CommandList* list = device->AcquireCommandList();
        const auto start = std::chrono::steady_clock::now();
        list->Begin();
        RenderingDesc rendering{.width = kSize, .height = kSize, .color_count = 1};
        rendering.colors[0] = {.view = target_view.get()};
        list->BeginRendering(rendering);
        const Viewport viewport{0, 0, float(kSize), float(kSize), 0, 1};
        const Rect scissor{0, 0, kSize, kSize};
        const std::array<float, 4> tint{1, 1, 1, 1};
        for (u32 i = 0; i < 4000; ++i) {
            list->SetViewports({&viewport, 1});
            list->SetScissors({&scissor, 1});
            list->BindPipeline(pipeline.get());
            list->PushBindings(bindings);
            list->PushConstants(0, {reinterpret_cast<const u8*>(tint.data()), sizeof(tint)});
            list->Draw(3, 1, 0, 0);
        }
        list->EndRendering();
        list->End();
        const auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        CommandList* lists[] = {list};
        device->GraphicsQueue().Submit(lists, {}, {fence.get(), 3});
        fence->Wait(3, 5'000'000'000ULL);
        log += fmt::format("recording 4,000 draws (pipeline, bindings, constants, draw each): {:.0f} us "
                           "({:.2f} us per draw)\n",
                           us, us / 4000);
    }
    // 4. Secondary lists (each draws one half through a pipeline with dynamic raster, depth and
    //    stencil state) run from a primary list, as the executor will run render jobs.
    if (caps.secondary_lists) {
        GraphicsPipelineDesc dynamic_desc = pipeline_desc;
        dynamic_desc.dynamic_raster_depth_stencil = true;
        auto dynamic_pipeline = device->CreateGraphicsPipeline(dynamic_desc);
        if (!dynamic_pipeline) {
            log += "dynamic-state pipeline failed\n";
            return false;
        }
        const std::array<std::array<float, 4>, 2> tints{{{0, 0, 1, 1}, {1, 0, 0, 1}}};
        std::array<CommandList*, 2> secondaries{};
        for (u32 i = 0; i < 2; ++i) {
            CommandList* list = device->AcquireCommandList(CommandListLevel::Secondary);
            list->Begin();
            RenderingDesc rendering{.width = kSize, .height = kSize, .color_count = 1};
            rendering.colors[0] = {.view = target_view.get()};
            list->BeginRendering(rendering);
            const Viewport viewport{0, 0, float(kSize), float(kSize), 0, 1};
            const Rect scissor{0, s32(i * kSize / 2), kSize, kSize / 2};
            list->SetViewports({&viewport, 1});
            list->SetScissors({&scissor, 1});
            list->BindPipeline(dynamic_pipeline.get());
            list->SetRasterState({});
            list->SetDepthStencilState({});
            list->PushBindings(bindings);
            list->PushConstants(0, {reinterpret_cast<const u8*>(tints[i].data()), sizeof(tints[i])});
            list->Draw(3, 1, 0, 0);
            list->EndRendering();
            list->End();
            secondaries[i] = list;
        }
        CommandList* primary = device->AcquireCommandList();
        primary->Begin();
        primary->ExecuteSecondary(secondaries);
        primary->Barrier(ScopeDraw, ScopeTransfer);
        primary->CopyImageToBuffer(target.get(), readback.get(), {&whole_target, 1});
        primary->Barrier(ScopeTransfer, ScopeHost);
        primary->End();
        CommandList* lists[] = {primary};
        device->GraphicsQueue().Submit(lists, {}, {fence.get(), 4});
        if (!fence->Wait(4, 5'000'000'000ULL)) {
            log += "secondary lists: GPU did not finish\n";
            return false;
        }
        for (CommandList* secondary : secondaries) {
            device->MarkExecuted(secondary, {fence.get(), 4});
        }
        const bool before = ok;
        const auto tinted = [&](u32 q, u32 t) {
            const Rgba c = quadrant[q];
            return Rgba{u8(c.r * tints[t][0]), u8(c.g * tints[t][1]), u8(c.b * tints[t][2]), c.a};
        };
        expect("secondary lists", 8, 8, tinted(0, 0));
        expect("secondary lists", 56, 8, tinted(1, 0));
        expect("secondary lists", 8, 56, tinted(2, 1));
        expect("secondary lists", 56, 56, tinted(3, 1));
        log += fmt::format("secondary lists with dynamic state: {}\n",
                           ok == before ? "picture correct" : "WRONG");
    }

    // 5. A layout with more bindings than can be pushed takes a descriptor set per draw.
    {
        std::vector<BindingSlot> many(caps.max_push_bindings + 8);
        for (u32 i = 0; i < many.size(); ++i) {
            many[i] = {i, i == 1 ? BindingType::Sampler : BindingType::SampledImage, StagePixel};
        }
        auto big_layout = device->CreatePipelineLayout(
            {.bindings = many, .push_constant_bytes = 16, .push_constant_stages = StagePixel});
        GraphicsPipelineDesc big_desc = pipeline_desc;
        big_desc.layout = big_layout.get();
        auto big_pipeline = big_layout ? device->CreateGraphicsPipeline(big_desc) : nullptr;
        if (!big_pipeline) {
            log += "pipeline with many bindings failed\n";
            return false;
        }
        CommandList* list = device->AcquireCommandList();
        list->Begin();
        RenderingDesc rendering{.width = kSize, .height = kSize, .color_count = 1};
        rendering.colors[0] = {.view = target_view.get(), .load = LoadOp::Clear, .clear = {0, 0, 0, 0}};
        list->BeginRendering(rendering);
        const Viewport viewport{0, 0, float(kSize), float(kSize), 0, 1};
        const Rect scissor{0, 0, kSize, kSize};
        const std::array<float, 4> tint{1, 1, 1, 1};
        list->SetViewports({&viewport, 1});
        list->SetScissors({&scissor, 1});
        list->BindPipeline(big_pipeline.get());
        list->PushBindings(bindings);
        list->PushConstants(0, {reinterpret_cast<const u8*>(tint.data()), sizeof(tint)});
        list->Draw(3, 1, 0, 0);
        list->EndRendering();
        list->Barrier(ScopeDraw, ScopeTransfer);
        list->CopyImageToBuffer(target.get(), readback.get(), {&whole_target, 1});
        list->Barrier(ScopeTransfer, ScopeHost);
        list->End();
        CommandList* lists[] = {list};
        device->GraphicsQueue().Submit(lists, {}, {fence.get(), 5});
        if (!fence->Wait(5, 5'000'000'000ULL)) {
            log += "many bindings: GPU did not finish\n";
            return false;
        }
        const bool before = ok;
        expect("many bindings", 8, 8, quadrant[0]);
        expect("many bindings", 56, 56, quadrant[3]);
        log += fmt::format("{} bindings through a descriptor set: {}\n", many.size(),
                           ok == before ? "picture correct" : "WRONG");
    }

    // 6. Compute with a specialization constant writing a storage buffer.
    {
        constexpr u32 kCount = 256;
        auto output = device->CreateBuffer({.size = kCount * 4,
                                            .usage = BufferUsageStorage | BufferUsageTransferSrc,
                                            .memory = MemoryKind::Readback});
        const std::array<BindingSlot, 1> compute_slots{{{0, BindingType::StorageBuffer, StageCompute}}};
        auto compute_layout = device->CreatePipelineLayout({.bindings = compute_slots});
        const std::array<SpecConstant, 1> constants{{{0, 1000}}};
        auto compute = compute_layout && output
                           ? device->CreateComputePipeline(
                                 {.layout = compute_layout.get(),
                                  .shader = {ShaderStage::Compute,
                                             std::span<const u32>{NATIVE_TEST_COMP}, "main",
                                             constants}})
                           : nullptr;
        if (!compute) {
            log += "compute pipeline failed\n";
            return false;
        }
        CommandList* list = device->AcquireCommandList();
        list->Begin();
        list->BindPipeline(compute.get());
        const Binding out_binding{.slot = 0, .type = BindingType::StorageBuffer, .buffer = output.get(),
                                  .size = kCount * 4};
        list->PushBindings({&out_binding, 1});
        list->Dispatch(kCount / 64, 1, 1);
        list->Barrier(ScopeCompute, ScopeHost);
        list->End();
        CommandList* lists[] = {list};
        device->GraphicsQueue().Submit(lists, {}, {fence.get(), 6});
        if (!fence->Wait(6, 5'000'000'000ULL)) {
            log += "compute: GPU did not finish\n";
            return false;
        }
        u32 wrong = 0;
        for (u32 i = 0; i < kCount; ++i) {
            u32 value;
            std::memcpy(&value, output->Mapped() + i * 4, 4);
            wrong += value != 1000 + i;
        }
        ok = ok && wrong == 0;
        log += fmt::format("compute with a specialization constant: {}\n",
                           wrong == 0 ? "values correct" : fmt::format("{} WRONG values", wrong));
    }

    device->Collect();
    return ok;
}

} // namespace

extern "C" __declspec(dllexport) int RytyNativeRhiSelfTest(char* report, int report_size) {
    std::string log;
    const bool ok = RunSelfTest(log);
    log += ok ? "NAT-012 self test passed\n" : "NAT-012 self test FAILED\n";
    if (report && report_size > 0) {
        const size_t n = std::min<size_t>(log.size(), static_cast<size_t>(report_size) - 1);
        std::memcpy(report, log.data(), n);
        report[n] = '\0';
    }
    return ok ? 0 : 1;
}
