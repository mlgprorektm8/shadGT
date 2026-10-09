// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "common/types.h"

/// One-pass diagnostic bundles. When a trigger fires, the rasterizer records every draw and
/// dispatch of the next frames, snapshots every image they touch at each frame boundary (raw
/// texels, tagged with the draw sequence number they follow), and writes the shaders they used
/// (guest code, listing, SPIR-V), the guest bytes of the buffers they read, the PM4 command
/// buffers submitted meanwhile, and the draw history, into user/log/bundles/<stamp>-<reason>/.
/// tools/mcp/analyze_bundle.py turns a bundle into one report of every anomaly.
namespace VideoCore::DiagBundle {

enum class Trigger : u32 {
    Now,    ///< at the next draw or dispatch
    Target, ///< at a draw rendering to a color target of width x height
    Shader, ///< at a draw or dispatch using the shader (any stage)
};

struct Request {
    std::string reason;
    Trigger trigger{Trigger::Now};
    u32 width{};
    u32 height{};
    u64 shader{};
    /// Frames recorded after the trigger; image snapshots are taken at each frame boundary, so
    /// with 2 the second frame's draws have their images both before and after them.
    u32 frames{2};
    /// Target/shader triggers: matching episodes to let pass before capturing (an episode ends
    /// after 120 frames without a match), e.g. 1 to capture the second car thumbnail.
    u32 skip{};
    /// Frames to wait after the trigger before capturing (e.g. the end of a thumbnail render).
    u32 delay{};
    /// Episodes to capture in a row: after each bundle the trigger is armed again for the next
    /// episode, e.g. 2 to capture the first (good) and second (bad) car thumbnail in one run.
    u32 repeat{1};
};

/// SHADGT_DIAG=1: keep each shader's guest code and SPIR-V so bundles can include them.
bool Enabled();

/// Parses "now", "target=WxH" or "shader=0xHASH".
std::optional<Request> ParseTrigger(std::string_view trigger, std::string reason, u32 frames);

/// Arms a bundle request (replacing an armed one that has not fired).
void Arm(Request request);
/// The armed request, if any (cheap when none is armed).
std::optional<Request> Armed();
/// Clears the armed request once the rasterizer has started on it.
void Disarm();

/// A new, empty bundle folder under user/log/bundles.
std::filesystem::path NewBundleDir(std::string_view reason);

/// PM4 submissions are copied into dir while a capture is active.
void BeginPm4Capture(const std::filesystem::path& dir);
void EndPm4Capture();
void CapturePm4(std::string_view queue, std::span<const u32> first, std::span<const u32> second);

/// Called on a crash (signal handler or failed assertion): the registered writer saves what it
/// can (the draw history) into a crash bundle. Best effort; never blocks on locks.
void SetCrashWriter(std::function<void(const std::filesystem::path&)> writer);
void OnCrash(std::string_view what);

/// Called by the perf monitor every 2 s: when no frame has been presented for 6 s (and the game
/// is not paused), logs once per hang where every guest thread is (HANG: ...).
void OnMonitorTick();

/// Called for each long frame (DIAG-033): arms a "stall" bundle once when it exceeds
/// SHADGT_BUNDLE_STALL_MS (diagnostic runs only).
void OnStall(double frame_ms);

} // namespace VideoCore::DiagBundle
