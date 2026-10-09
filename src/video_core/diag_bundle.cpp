// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>

#include <fmt/format.h>

#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/debug_state.h"
#include "video_core/diag_bundle.h"

namespace VideoCore::DiagBundle {

namespace {
std::mutex armed_mutex;
std::vector<Request> armed;
std::atomic<bool> has_armed{false};

std::mutex pm4_mutex;
std::filesystem::path pm4_dir;
std::ofstream pm4_index;
u32 pm4_count = 0;
std::atomic<bool> pm4_active{false};

std::mutex crash_mutex;
std::function<void(const std::filesystem::path&)> crash_writer;

std::string Sanitize(std::string_view text) {
    std::string out;
    for (const char c : text) {
        out += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_';
    }
    return out.substr(0, 48);
}
} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_DIAG");
        return env && *env && *env != '0';
    }();
    return enabled;
}

std::optional<Request> ParseTrigger(std::string_view trigger, std::string reason, u32 frames) {
    Request request{.reason = std::move(reason), .frames = std::max(frames, 1u)};
    if (trigger.empty() || trigger == "now") {
        request.trigger = Trigger::Now;
        return request;
    }
    if (trigger.starts_with("target=")) {
        const auto value = trigger.substr(7);
        const auto x = value.find('x');
        if (x == std::string_view::npos) {
            return std::nullopt;
        }
        request.trigger = Trigger::Target;
        request.width =
            static_cast<u32>(std::strtoul(std::string(value.substr(0, x)).c_str(), nullptr, 10));
        request.height =
            static_cast<u32>(std::strtoul(std::string(value.substr(x + 1)).c_str(), nullptr, 10));
        return request.width && request.height ? std::optional{request} : std::nullopt;
    }
    if (trigger.starts_with("shader=")) {
        request.trigger = Trigger::Shader;
        request.shader = std::strtoull(std::string(trigger.substr(7)).c_str(), nullptr, 16);
        return request.shader ? std::optional{request} : std::nullopt;
    }
    return std::nullopt;
}

void Arm(Request request) {
    std::scoped_lock lk{armed_mutex};
    LOG_WARNING(Render_Vulkan,
                "DIAG-BUNDLE: armed '{}' (trigger {}, {} frames, skip {}, delay {}, repeat {})",
                request.reason, static_cast<u32>(request.trigger), request.frames, request.skip,
                request.delay, request.repeat);
    // Several triggers can wait at once (e.g. the car thumbnail and a dealership shader); arming
    // the same reason again replaces it.
    std::erase_if(armed, [&](const Request& r) { return r.reason == request.reason; });
    armed.push_back(std::move(request));
    has_armed = true;
}

static u32 EnvU32(const char* name, u32 fallback) {
    const char* value = std::getenv(name);
    return value && *value ? static_cast<u32>(std::atoi(value)) : fallback;
}

// One trigger with options: "target=800x450,skip=0,repeat=2,delay=30,frames=3". Options not
// given come from the defaults.
static void ArmItem(std::string_view item, u32 frames, u32 skip, u32 delay, u32 repeat) {
    std::string_view trigger = item.substr(0, item.find(','));
    for (size_t at = item.find(','); at != std::string_view::npos;) {
        const size_t end = item.find(',', at + 1);
        const auto option = item.substr(at + 1, end == std::string_view::npos ? end : end - at - 1);
        const auto eq = option.find('=');
        const auto key = option.substr(0, eq);
        const u32 value = eq == std::string_view::npos
                              ? 0
                              : static_cast<u32>(std::atoi(std::string(option.substr(eq + 1)).c_str()));
        if (key == "frames") {
            frames = value;
        } else if (key == "skip") {
            skip = value;
        } else if (key == "delay") {
            delay = value;
        } else if (key == "repeat") {
            repeat = value;
        }
        at = end;
    }
    auto request = ParseTrigger(trigger, fmt::format("env-{}", trigger), frames);
    if (!request) {
        LOG_ERROR(Render_Vulkan, "DIAG-BUNDLE: trigger '{}' not understood", item);
        return;
    }
    request->skip = skip;
    request->delay = delay;
    request->repeat = std::max(repeat, 1u);
    Arm(std::move(*request));
}

// Bundles armed at startup without the IPC/MCP controller, for runs started by hand.
// SHADGT_BUNDLE_TRIGGER: ';'-separated triggers ("target=WxH" or "shader=0xHASH", each with
// optional ",skip=,delay=,repeat=,frames=" options); SHADGT_BUNDLE_SKIP/DELAY/REPEAT/FRAMES are
// the defaults. SHADGT_BUNDLE_AUTO adds standing triggers when SHADGT_DIAG is on (default: the
// first use of the dealership shader that reads garbage descriptors); set it to 0 to disable.
static void ArmFromEnvironment() {
    const u32 frames = EnvU32("SHADGT_BUNDLE_FRAMES", 2);
    const u32 skip = EnvU32("SHADGT_BUNDLE_SKIP", 0);
    const u32 delay = EnvU32("SHADGT_BUNDLE_DELAY", 0);
    const u32 repeat = EnvU32("SHADGT_BUNDLE_REPEAT", 1);
    std::string triggers;
    if (const char* env = std::getenv("SHADGT_BUNDLE_TRIGGER"); env && *env) {
        triggers = env;
    }
    if (Enabled()) {
        const char* automatic = std::getenv("SHADGT_BUNDLE_AUTO");
        const std::string standing =
            automatic ? automatic : "shader=0xae32f77f,frames=2,skip=0,delay=0,repeat=1";
        if (!standing.empty() && standing != "0") {
            triggers += (triggers.empty() ? "" : ";") + standing;
        }
    }
    std::string_view rest = triggers;
    while (!rest.empty()) {
        const auto end = rest.find(';');
        if (const auto item = rest.substr(0, end); !item.empty()) {
            ArmItem(item, frames, skip, delay, repeat);
        }
        rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 1);
    }
}

std::vector<Request> Armed() {
    static std::once_flag from_environment;
    std::call_once(from_environment, ArmFromEnvironment);
    if (!has_armed.load(std::memory_order_relaxed)) {
        return {};
    }
    std::scoped_lock lk{armed_mutex};
    return armed;
}

void Disarm(std::string_view reason) {
    std::scoped_lock lk{armed_mutex};
    std::erase_if(armed, [&](const Request& r) { return r.reason == reason; });
    has_armed = !armed.empty();
}

std::filesystem::path NewBundleDir(std::string_view reason) {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &time);
#else
    localtime_r(&time, &local);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
    const auto base = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "bundles";
    auto dir = base / fmt::format("{}-{}", stamp, Sanitize(reason));
    for (u32 n = 2; std::filesystem::exists(dir); ++n) {
        dir = base / fmt::format("{}-{}-{}", stamp, Sanitize(reason), n);
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

void BeginPm4Capture(const std::filesystem::path& dir) {
    std::scoped_lock lk{pm4_mutex};
    pm4_dir = dir;
    std::error_code ec;
    std::filesystem::create_directories(pm4_dir, ec);
    pm4_index.open(pm4_dir / "index.jsonl", std::ios::trunc);
    pm4_count = 0;
    pm4_active = true;
}

void EndPm4Capture() {
    std::scoped_lock lk{pm4_mutex};
    pm4_active = false;
    pm4_index.close();
}

void CapturePm4(std::string_view queue, std::span<const u32> first, std::span<const u32> second) {
    if (!pm4_active.load(std::memory_order_relaxed)) {
        return;
    }
    std::scoped_lock lk{pm4_mutex};
    if (!pm4_active || pm4_count >= 65536) {
        return;
    }
    const u32 n = ++pm4_count;
    const auto write = [&](std::string_view part, std::span<const u32> data) {
        if (data.empty()) {
            return std::string{};
        }
        auto name = fmt::format("{:04}_{}_{}.bin", n, queue, part);
        Common::FS::IOFile{pm4_dir / name, Common::FS::FileAccessMode::Create}.WriteSpan(data);
        return name;
    };
    const auto first_name = write(queue == "gfx" ? "dcb" : "acb", first);
    const auto second_name = write("ccb", second);
    pm4_index << fmt::format(
                     R"({{"n": {}, "queue": "{}", "frame": {}, "first": "{}", "first_dwords": {}, )"
                     R"("second": "{}", "second_dwords": {}}})",
                     n, queue, DebugState.GetFrameNum(), first_name, first.size(), second_name,
                     second.size())
              << '\n';
}

void SetCrashWriter(std::function<void(const std::filesystem::path&)> writer) {
    std::scoped_lock lk{crash_mutex};
    crash_writer = std::move(writer);
}

void OnCrash(std::string_view what) {
    static std::atomic<bool> written{false};
    if (written.exchange(true)) {
        return;
    }
    std::unique_lock lk{crash_mutex, std::try_to_lock};
    if (!lk.owns_lock() || !crash_writer) {
        return;
    }
    try {
        const auto dir = NewBundleDir(fmt::format("crash-{}", what));
        crash_writer(dir);
        LOG_CRITICAL(Render_Vulkan, "DIAG-BUNDLE: crash bundle written to {}", dir.string());
    } catch (...) {
    }
}

void OnMonitorTick() {
    static u32 last_frame = ~0u;
    static u32 still_ticks = 0;
    static bool reported = false;
    const u32 frame = DebugState.GetFrameNum();
    if (frame != last_frame || DebugState.IsGuestThreadsPaused()) {
        last_frame = frame;
        still_ticks = 0;
        reported = false;
        return;
    }
    if (++still_ticks < 3 || reported || frame == 0) {
        return;
    }
    reported = true;
    LOG_ERROR(Render_Vulkan, "HANG: no frame presented for {} s (frame {}); guest threads:{}",
              still_ticks * 2, frame, DebugState.DescribeGuestThreads());
}

void OnStall(double frame_ms) {
    static const double threshold = [] {
        const char* env = std::getenv("SHADGT_BUNDLE_STALL_MS");
        return env && *env ? std::atof(env) : 0.0;
    }();
    static std::atomic<bool> armed_once{false};
    if (threshold <= 0.0 || frame_ms < threshold || armed_once.exchange(true)) {
        return;
    }
    Arm({.reason = fmt::format("stall-{:.0f}ms", frame_ms), .trigger = Trigger::Now, .frames = 2});
}

} // namespace VideoCore::DiagBundle
