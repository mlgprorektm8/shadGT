// Entry for the renderer library (libSceVideoOut.prx, which also exports libSceGnmDriver): opens
// the window on its own thread, then registers shadGT's Gnm driver and video output.

#include <windows.h>
#include <tlhelp32.h>
#include <condition_variable>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/guest_clock.h"
#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "common/singleton.h"
#include "core/emulator_settings.h"
#include "core/user_settings.h"
#include "core/libraries/gnmdriver/gnmdriver.h"
#include "core/libraries/ime/ime.h"
#include "core/libraries/ime/ime_dialog.h"
#include "core/libraries/videoout/video_out.h"
#include "core/memory.h"
#include "core/signals.h"
#include "core/loader/symbols_resolver.h"
#include "input/controller.h"
#include "ryty_gpu/ryty_framegen.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "ryty_gpu/ryty_input_bridge.h"
#include "ryty_gpu/ryty_library.h"
#include "ryty_gpu/native/gts_hooks.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "core/debug_state.h"
#include "imgui.h"
#ifdef ENABLE_DISCORD_RPC
#include "common/discord_rpc_handler.h"
#endif
#include <chrono>
#include <cmath>

namespace Libraries::SystemService {
void NoteGameDrew(); // ryty_stubs.cpp: the splash screen ends at the game's first drawn frame
}
namespace Vulkan {
void RytyWatchWrites(std::uint64_t address, std::uint32_t ms, std::string label); // ryty_write_watch.cpp
}

Frontend::WindowSDL* g_window = nullptr;

namespace {

std::mutex g_window_mutex;
std::condition_variable g_window_cv;
bool g_window_ready = false;
bool g_renderer_ready = false;

void RunWindowThread() {
    auto* controllers = Common::Singleton<Input::GameControllers>::Instance();
    static Frontend::WindowSDL window(EmulatorSettings.GetWindowWidth(),
                                      EmulatorSettings.GetWindowHeight(), controllers,
                                      "ryGT - Gran Turismo Sport");
    window.SetIcon({}); // shadGT's Gran Turismo Sport icon
    {
        std::scoped_lock lock{g_window_mutex};
        g_window = &window;
        g_window_ready = true;
    }
    g_window_cv.notify_all();
    // Events reach ImGui, which the presenter initializes, so pump only once it exists.
    {
        std::unique_lock lock{g_window_mutex};
        g_window_cv.wait(lock, [] { return g_renderer_ready; });
    }
    window.InitTimers();
    while (window.IsOpen()) {
        window.WaitEvent();
    }
    // The window was closed: end the game without DLL teardown (see Emulator::Shutdown).
    // Stop trigger effects before the deliberate TerminateProcess skips SDL cleanup.
    Input::TriggerFeedback::stopping = true;
    const auto released = Input::TriggerFeedback::Vibration(0);
    for (int slot = 0; slot < 4; ++slot) {
        (*controllers)[slot]->SetTriggerFeedback(released, released, 0, 0);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Flush queued HID releases.
    TerminateProcess(GetCurrentProcess(), 0);
}

// Signals the named event Ryty's save data library waits on before GT Sport's first save access
// once the game has drawn steadily for half a second (the boot screen is up), so loading cannot
// finish before the boot screen is listening for it. Before the boot screen nothing draws: the
// save access is held before the game decides to create a save, so its "Saving" indicator, which
// would otherwise draw first on a fresh save, has not appeared yet.
void StartFirstDrawWatcher() {
    char name[64];
    std::snprintf(name, sizeof(name), "Local\\RytyFirstDraw-%lu", GetCurrentProcessId());
    HANDLE event = CreateEventA(nullptr, TRUE, FALSE, name);
    if (event == nullptr) {
        return;
    }
    std::thread([event] {
        auto& draws = Common::GetWorkCounters().draws;
        while (draws.load(std::memory_order_relaxed) == 0) {
            Sleep(2);
        }
        Libraries::SystemService::NoteGameDrew(); // ends the splash screen
        u64 last = draws.load(std::memory_order_relaxed);
        for (int drawing = 0; drawing < 5;) {
            Sleep(100);
            const u64 now = draws.load(std::memory_order_relaxed);
            drawing = now != last ? drawing + 1 : 0;
            last = now;
        }
        LOG_INFO(Lib_GnmDriver, "Ryty: the boot screen is drawing; releasing the first save access");
        SetEvent(event);
    }).detach();
}

// The FPS counter (shadGT's Show FPS Counter setting): the game's frames per second, in small
// type in the window's top left corner, over the game and the splash screen.
// PERF-098: the precompile of stored shaders and pipelines, as a translucent blue bar with
// shadowed white text near the bottom of the splash screen (and of the game, if it is still
// going when the game starts drawing).
void DrawPrecompileProgress() {
    using Clock = std::chrono::steady_clock;
    const auto progress = Vulkan::GetPrecompileProgress();
    static Clock::time_point finished{};
    static bool shown = false;
    if (progress.total == 0) {
        return;
    }
    float alpha = 1.0f;
    if (progress.active) {
        shown = true;
        finished = {};
    } else {
        if (!shown) {
            return; // nothing was left to build: no bar at all
        }
        if (finished == Clock::time_point{}) {
            finished = Clock::now();
        }
        // "Done" stays a moment, then fades out.
        const float since = std::chrono::duration<float>(Clock::now() - finished).count();
        alpha = std::clamp(1.0f - (since - 1.5f) / 0.5f, 0.0f, 1.0f);
        if (alpha <= 0.0f) {
            return;
        }
    }
    const auto a = [alpha](float value) { return static_cast<int>(value * alpha); };
    const float fraction =
        progress.total ? float(progress.done) / float(progress.total) : 0.0f;
    char text[96];
    if (progress.active) {
        std::snprintf(text, sizeof(text), "Precompiling shaders   %llu / %llu   %.0f%%",
                      static_cast<unsigned long long>(progress.done),
                      static_cast<unsigned long long>(progress.total), fraction * 100.0f);
    } else {
        std::snprintf(text, sizeof(text), "Shaders ready");
    }
    auto* draw = ImGui::GetForegroundDrawList();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float width = std::clamp(viewport->Size.x * 0.36f, 240.0f, 720.0f);
    const float bar_height = 6.0f;
    const ImVec2 bar_min{viewport->Pos.x + (viewport->Size.x - width) * 0.5f,
                         viewport->Pos.y + viewport->Size.y * 0.90f};
    const ImVec2 bar_max{bar_min.x + width, bar_min.y + bar_height};
    draw->AddRectFilled(bar_min, bar_max, IM_COL32(25, 80, 190, a(70)), bar_height * 0.5f);
    if (fraction > 0.0f) {
        draw->AddRectFilled(bar_min, {bar_min.x + width * fraction, bar_max.y},
                            IM_COL32(70, 150, 255, a(175)), bar_height * 0.5f);
    }
    const ImVec2 size = ImGui::CalcTextSize(text);
    const ImVec2 at{bar_min.x + (width - size.x) * 0.5f, bar_min.y - size.y - 6.0f};
    for (const ImVec2 offset : {ImVec2{-1, 0}, ImVec2{1, 0}, ImVec2{0, -1}, ImVec2{0, 1},
                                ImVec2{1, 1}, ImVec2{2, 2}}) {
        draw->AddText({at.x + offset.x, at.y + offset.y}, IM_COL32(0, 0, 0, a(150)), text);
    }
    draw->AddText(at, IM_COL32(255, 255, 255, a(235)), text);
}

// PERF-100: while a new shader compiles and the game is held, the same translucent style: a
// notice with a moving bar over the last frame. Not for compiles shorter than 0.15 s.
void DrawCompileNotice() {
    using Clock = std::chrono::steady_clock;
    static Clock::time_point paused_since{};
    static Clock::time_point last_paused{};
    const auto now = Clock::now();
    if (Common::GuestClock::IsPaused() || Vulkan::CompileSlowMotion()) {
        if (paused_since == Clock::time_point{} || now - last_paused > std::chrono::milliseconds{300}) {
            paused_since = now;
        }
        last_paused = now;
    }
    if (paused_since == Clock::time_point{} || now - paused_since < std::chrono::milliseconds{150} ||
        now - last_paused > std::chrono::milliseconds{300}) {
        return;
    }
    const char* text = "Compiling new shaders...";
    auto* draw = ImGui::GetForegroundDrawList();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImVec2 size = ImGui::CalcTextSize(text);
    // Bottom left: the text, with a small bar under it.
    const float margin = 18.0f;
    const float width = std::max(size.x, 160.0f);
    const float bar_height = 4.0f;
    const ImVec2 bar_min{viewport->Pos.x + margin,
                         viewport->Pos.y + viewport->Size.y - margin - bar_height};
    const ImVec2 bar_max{bar_min.x + width, bar_min.y + bar_height};
    draw->AddRectFilled(bar_min, bar_max, IM_COL32(25, 80, 190, 70), bar_height * 0.5f);
    // A segment sweeping across: the time a compile takes is not known.
    const float t = std::fmod(std::chrono::duration<float>(now.time_since_epoch()).count(), 1.2f) / 1.2f;
    const float segment = width * 0.3f;
    const float x0 = bar_min.x + (width + segment) * t - segment;
    draw->AddRectFilled({std::max(x0, bar_min.x), bar_min.y},
                        {std::min(x0 + segment, bar_max.x), bar_max.y},
                        IM_COL32(70, 150, 255, 175), bar_height * 0.5f);
    const ImVec2 at{bar_min.x, bar_min.y - size.y - 5.0f};
    for (const ImVec2 offset : {ImVec2{-1, 0}, ImVec2{1, 0}, ImVec2{0, -1}, ImVec2{0, 1},
                                ImVec2{1, 1}, ImVec2{2, 2}}) {
        draw->AddText({at.x + offset.x, at.y + offset.y}, IM_COL32(0, 0, 0, 150), text);
    }
    draw->AddText(at, IM_COL32(255, 255, 255, 235), text);
}

void DrawFpsCounter() {
    Ryty::DrawControllerOverlay();
    DrawPrecompileProgress();
    if (!Vulkan::GetPrecompileProgress().active) {
        DrawCompileNotice();
    }
    if (!EmulatorSettings.IsShowFpsCounter()) {
        return;
    }
    using Clock = std::chrono::steady_clock;
    static Clock::time_point window_start = Clock::now();
    static u32 window_frames = DebugState.GetFrameNum();
    static u64 window_shown = Vulkan::RytyFrameGen::ShownFrames();
    static double fps = 0.0;
    static double shown_fps = 0.0;
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - window_start).count();
    if (seconds >= 0.5) {
        const u32 frames = DebugState.GetFrameNum();
        const u64 shown = Vulkan::RytyFrameGen::ShownFrames();
        fps = (frames - window_frames) / seconds;
        // With frame generation: the frames on screen too (game and generated).
        shown_fps = shown > window_shown ? (shown - window_shown) / seconds : 0.0;
        window_frames = frames;
        window_shown = shown;
        window_start = now;
    }
    char text[48];
    if (shown_fps > 0.0) {
        std::snprintf(text, sizeof(text), "%.0f FPS (%.0f shown)", fps, shown_fps);
    } else {
        std::snprintf(text, sizeof(text), "%.0f FPS", fps);
    }
    auto* draw = ImGui::GetForegroundDrawList();
    const ImVec2 origin = ImGui::GetMainViewport()->Pos;
    const ImVec2 size = ImGui::CalcTextSize(text);
    const ImVec2 at{origin.x + 8.0f, origin.y + 6.0f};
    draw->AddRectFilled({at.x - 4.0f, at.y - 2.0f}, {at.x + size.x + 4.0f, at.y + size.y + 2.0f},
                        IM_COL32(0, 0, 0, 140), 3.0f);
    draw->AddText(at, IM_COL32(235, 235, 235, 255), text);
}

// DIAG-047 (SHADGT_HEAP_REPORT=1): every 20 s, the largest heap's busy blocks by size, with the
// first bytes of one block of each size. The walk holds the heap's lock, so nothing it does may
// allocate: the table is allocated before.
void StartHeapReport() {
    const char* env = std::getenv("SHADGT_HEAP_REPORT");
    if (env == nullptr) {
        return;
    }
    // "1": every 20 s; a larger number: every that many seconds.
    const int seconds = std::atoi(env);
    const DWORD interval_ms = DWORD(seconds > 1 ? seconds : 20) * 1000;
    std::thread([interval_ms] {
        struct Slot {
            u64 size;
            u64 count;
            u8 sample[128];
        };
        constexpr size_t TableSize = 1 << 16;
        std::vector<Slot> table(TableSize);
        std::vector<HANDLE> heaps(256);
        for (;;) {
            Sleep(interval_ms);
            const DWORD number = GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data());
            HANDLE largest = nullptr;
            u64 largest_bytes = 0;
            for (DWORD h = 0; h < std::min<DWORD>(number, heaps.size()); ++h) {
                std::fill(table.begin(), table.end(), Slot{});
                u64 bytes = 0, blocks = 0, dropped = 0;
                if (!HeapLock(heaps[h])) {
                    continue;
                }
                PROCESS_HEAP_ENTRY entry{};
                while (HeapWalk(heaps[h], &entry)) {
                    if (!(entry.wFlags & PROCESS_HEAP_ENTRY_BUSY)) {
                        continue;
                    }
                    bytes += entry.cbData;
                    ++blocks;
                    size_t i = (entry.cbData * 0x9E3779B97F4A7C15ull) >> 48;
                    size_t probes = 0;
                    while (table[i].count != 0 && table[i].size != entry.cbData &&
                           probes++ < 64) {
                        i = (i + 1) % TableSize;
                    }
                    if (table[i].count == 0) {
                        table[i].size = entry.cbData;
                        std::memcpy(table[i].sample, entry.lpData,
                                    std::min<size_t>(entry.cbData, sizeof(table[i].sample)));
                    } else if (table[i].size != entry.cbData) {
                        ++dropped;
                        continue;
                    }
                    ++table[i].count;
                }
                HeapUnlock(heaps[h]);
                if (bytes < (64ull << 20)) {
                    continue;
                }
                std::vector<Slot> top;
                for (const auto& slot : table) {
                    if (slot.count != 0) {
                        top.push_back(slot);
                    }
                }
                std::ranges::sort(top, [](const Slot& a, const Slot& b) {
                    return a.size * a.count > b.size * b.count;
                });
                std::string lines;
                for (size_t t = 0; t < std::min<size_t>(top.size(), 10); ++t) {
                    lines += fmt::format("\n  {} x {} bytes ({} MB):", top[t].count, top[t].size,
                                         top[t].size * top[t].count >> 20);
                    for (size_t b = 0; b < std::min<size_t>(top[t].size, 128); ++b) {
                        lines += fmt::format("{}{:02x}", b % 8 ? "" : " ", top[t].sample[b]);
                    }
                    // Qwords pointing into a module (vtables, code), as module+offset.
                    for (size_t q = 0; q + 8 <= std::min<size_t>(top[t].size, 128); q += 8) {
                        u64 value;
                        std::memcpy(&value, top[t].sample + q, 8);
                        HMODULE module = nullptr;
                        char name[MAX_PATH];
                        if (value > 0x10000 &&
                            GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                               reinterpret_cast<LPCSTR>(value), &module) &&
                            GetModuleFileNameA(module, name, sizeof(name))) {
                            const char* base = std::strrchr(name, '\\');
                            lines += fmt::format(" [+{}: {}+{:#x}]", q, base ? base + 1 : name,
                                                 value - reinterpret_cast<u64>(module));
                        }
                    }
                }
                LOG_WARNING(Lib_GnmDriver,
                            "DIAG-047: heap {} ({} of {}): {} MB in {} busy blocks{}{}",
                            fmt::ptr(heaps[h]), h, number, bytes >> 20, blocks,
                            dropped ? fmt::format(" ({} not counted by size)", dropped) : "",
                            lines);
            }
        }
    }).detach();
}

} // namespace

void Ryty::RytyLibraryInitialize() {
    InitializeCommon("shadgt_ryty.log");
    LOG_INFO(Lib_GnmDriver, "Ryty: starting shadGT renderer");
    // shadGT's main() loads the player list (user/users.json, created with defaults when
    // missing); a gamepad is only assigned to a player that exists ("User 1 not found").
    UserSettings.Load();
    InstallControllerPorts(); // before the window thread opens the controllers
    if (std::getenv("RYTY_SELFTEST_GUARD") != nullptr) {
        // A guarded read of an inaccessible page must come back false, not end the process.
        void* page = VirtualAlloc(nullptr, 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
        u8 byte = 0;
        const bool copied = Core::RytyMemory::GuardedCopy(&byte, page, 1);
        LOG_WARNING(Lib_GnmDriver, "Ryty: guarded read self-test: {}",
                    copied ? "read an inaccessible page?" : "fault caught");
        VirtualFree(page, 0, MEM_RELEASE);
    }

    std::thread(RunWindowThread).detach();
    {
        std::unique_lock lock{g_window_mutex};
        g_window_cv.wait(lock, [] { return g_window_ready; });
    }

    Core::Loader::SymbolsResolver resolver;
    Libraries::GnmDriver::RegisterLib(&resolver);
    Libraries::VideoOut::RegisterLib(&resolver);
    // ryGT: shadGT's on-screen keyboard (inline keyboard and keyboard dialog).
    Libraries::Ime::RegisterLib(&resolver);
    Libraries::ImeDialog::RegisterLib(&resolver);
    BindExports();
    StartInputBridge();
    StartFirstDrawWatcher();
    StartHeapReport();
    // DIAG-078: SHADGT_LOG_EXCEPTIONS=1 counts every exception raised in GTSport.exe code (first
    // chance, any handler) by code and exe offset, and logs the counts every 5 seconds.
    if (const char* env = std::getenv("SHADGT_LOG_EXCEPTIONS"); env && env[0] == '1') {
        static PVECTORED_EXCEPTION_HANDLER handler = [](EXCEPTION_POINTERS* pointers) -> LONG {
            static std::mutex mutex;
            static std::unordered_map<u64, u64> counts;
            static auto last = std::chrono::steady_clock::now();
            static const u64 base = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
            const u64 rip = pointers->ContextRecord->Rip;
            const u64 code = pointers->ExceptionRecord->ExceptionCode;
            // SHADGT_LOG_EXCEPTIONS_AT=<from>-<to> (exe offsets): log each exception there.
            static const std::pair<u64, u64> range = [] {
                const char* at = std::getenv("SHADGT_LOG_EXCEPTIONS_AT");
                if (!at || !*at) {
                    return std::pair<u64, u64>{0, 0};
                }
                const std::string text{at};
                const auto dash = text.find('-');
                return std::pair<u64, u64>{std::stoull(text.substr(0, dash), nullptr, 16),
                                           std::stoull(text.substr(dash + 1), nullptr, 16)};
            }();
            if (rip - base >= range.first && rip - base < range.second) {
                std::fprintf(stderr, "[exc] DIAG-078: exception %llx at exe+%llx thread %lu rsp %llx\n",
                             static_cast<unsigned long long>(code),
                             static_cast<unsigned long long>(rip - base), GetCurrentThreadId(),
                             static_cast<unsigned long long>(pointers->ContextRecord->Rsp));
                std::fflush(stderr);
            }
            if (rip < base || rip >= base + 0x4000000) {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            std::scoped_lock lock{mutex};
            ++counts[(code << 32) | (rip - base)];
            const auto now = std::chrono::steady_clock::now();
            if (now - last >= std::chrono::seconds(5)) {
                last = now;
                std::vector<std::pair<u64, u64>> top(counts.begin(), counts.end());
                std::ranges::sort(top,
                                  [](const auto& a, const auto& b) { return a.second > b.second; });
                std::string line;
                for (std::size_t i = 0; i < std::min<std::size_t>(top.size(), 2000); ++i) {
                    line += fmt::format(" {:x}@{:#x}={}", top[i].first >> 32,
                                        top[i].first & 0xffffffff, top[i].second);
                }
                std::fprintf(stderr, "[exc] DIAG-078: exceptions in GTSport.exe (%zu sites):%s\n",
                             top.size(), line.c_str());
                std::fflush(stderr);
            }
            return EXCEPTION_CONTINUE_SEARCH;
        };
        // Registered again every few seconds so it stays ahead of handlers added later.
        std::thread([] {
            PVOID registered = nullptr;
            for (;;) {
                PVOID next = AddVectoredExceptionHandler(1, handler);
                if (registered) {
                    RemoveVectoredExceptionHandler(registered);
                }
                registered = next;
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }).detach();
    }
    // DIAG-075: SHADGT_WATCH_WRITE=<hex address>:<seconds>:<ms>[;...] logs the writes to a guest
    // address (thread, code address, value) for ms milliseconds, seconds after the renderer
    // starts; several watches run one after another.
    if (const char* watch = std::getenv("SHADGT_WATCH_WRITE"); watch != nullptr && *watch != 0) {
        std::thread([list = std::string(watch)] {
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t pos = 0; pos < list.size();) {
                const std::size_t end = std::min(list.find(';', pos), list.size());
                const std::string item = list.substr(pos, end - pos);
                pos = end + 1;
                const auto c1 = item.find(':');
                const auto c2 = item.find(':', c1 + 1);
                if (c1 == std::string::npos || c2 == std::string::npos) {
                    continue;
                }
                // "x<offset>": an instruction breakpoint at that offset in GTSport.exe.
                const bool execute = item[0] == 'x';
                u64 address = std::stoull(item.substr(execute ? 1 : 0, c1 - (execute ? 1 : 0)),
                                          nullptr, 16);
                if (execute) {
                    address += reinterpret_cast<u64>(GetModuleHandleA(nullptr));
                }
                const double at = std::stod(item.substr(c1 + 1, c2 - c1 - 1));
                const u32 ms = static_cast<u32>(std::stoul(item.substr(c2 + 1)));
                std::this_thread::sleep_until(start + std::chrono::milliseconds(
                                                          static_cast<long long>(at * 1000)));
                Vulkan::RytyWatchWrites(address, ms,
                                        fmt::format("{} {:#x}", execute ? "exec" : "watch", address));
                std::this_thread::sleep_for(std::chrono::milliseconds(ms + 200));
            }
        }).detach();
    }
    Ryty::SetOverlayDraw(&DrawFpsCounter);
#ifdef ENABLE_DISCORD_RPC
    if (EmulatorSettings.IsDiscordRPCEnabled()) {
        auto* rpc = Common::Singleton<DiscordRPCHandler::RPC>::Instance();
        rpc->init();
        rpc->setStatusPlaying("Gran Turismo Sport", "CUSA03220");
    }
#endif
    {
        std::scoped_lock lock{g_window_mutex};
        g_renderer_ready = true;
    }
    g_window_cv.notify_all();
}

// Ryty's libkernel maps and unmaps guest memory, so it reports those changes here; shadGT's own
// memory manager would otherwise do this. Without them the renderer keeps cached GPU copies and
// page tracking for memory the game has freed and reused, and its watcher counts go wrong.
extern "C" __declspec(dllexport) void RytyGpuUnmapMemory(u64 address, u64 size) {
    Core::RytyMemory::NoteMappingChange(address, size);
    if (auto* rasterizer = Core::Memory::Instance()->GetRasterizer()) {
        rasterizer->UnmapMemory(address, size);
    }
}

// Ryty calls this after any change to guest mappings or protections (after an unmap completes,
// for example), so lookups cached in between are not kept.
extern "C" __declspec(dllexport) void RytyGpuMappingChanged(u64 address, u64 size) {
    Core::RytyMemory::NoteMappingChange(address, size);
}

// Ryty calls this after the game changed the protection of guest memory (sceKernelMprotect),
// which replaced the protections the renderer's page tracking had set there.
extern "C" __declspec(dllexport) void RytyGpuProtectionChanged(u64 address, u64 size) {
    Core::RytyMemory::NoteMappingChange(address, size);
    if (auto* rasterizer = Core::Memory::Instance()->GetRasterizer()) {
        rasterizer->GetPageManager().ReapplyProtection(address, size);
    }
}

extern "C" __declspec(dllexport) void RytyGpuMapMemory(u64 address, u64 size) {
    Core::RytyMemory::NoteMappingChange(address, size);
    if (auto* rasterizer = Core::Memory::Instance()->GetRasterizer()) {
        rasterizer->MapMemory(address, size);
        rasterizer->RegisterMemory(address, size);
    }
}

void RytyProtectGuestRedZones(); // ryty_red_zone.cpp
void RytyApplyGamePatches();     // ryty_game_patch.cpp (GTS-001)

// Ryty loads this library before the game's code starts, so the game code can still be patched
// safely here (FIX-074).
extern "C" BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        // ryGT: overlay layers (OBS, Steam) hook Vulkan and have hung the game on exit. The
        // Vulkan loader reads this when the renderer creates its instance, later.
        if (GetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", nullptr, 0) == 0) {
            SetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", L"~implicit~");
        }
        RytyProtectGuestRedZones();
        RytyApplyGamePatches();
        // NAT-002: hook points for the native renderer (RYGT_NATIVE_HOOKS=1).
        Native::InstallGtsHooks(reinterpret_cast<u8*>(GetModuleHandleA(nullptr)));
    }
    return TRUE;
}

// The renderer write-protects guest pages it caches on the GPU. Ryty's exception handler offers
// each guest write fault here first; true means the renderer owned the page, dropped its cached
// copy and unprotected it, so the write can be retried.
extern "C" __declspec(dllexport) int RytyGpuHandleWriteFault(void* exception_pointers) {
    auto* pointers = static_cast<EXCEPTION_POINTERS*>(exception_pointers);
    if (pointers == nullptr || pointers->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
        pointers->ExceptionRecord->NumberParameters < 2) {
        return 0;
    }
    const auto address = reinterpret_cast<void*>(pointers->ExceptionRecord->ExceptionInformation[1]);
    if (Core::Memory::Instance()->GetRasterizer() == nullptr) {
        return Core::RytyMemory::InGuardedRead() ? 2 : 0;
    }
    // 2: a guarded read hit memory the game released or protected; its structured handler
    // takes it and the read falls back to the safe copy (Ryty then does not treat it as fatal).
    if (Core::RytyMemory::InGuardedRead() &&
        !Core::RytyMemory::TrackingMadeUnreadable(reinterpret_cast<VAddr>(address))) {
        return 2;
    }
    if (Core::Signals::Instance()->DispatchAccessViolation(pointers, address)) {
        // DIAG-077: SHADGT_LOG_FAULTS=1 counts the write faults the renderer handled by the
        // code address in GTSport.exe that wrote, and logs the most frequent every 5 seconds.
        static const bool log_faults = [] {
            const char* env = std::getenv("SHADGT_LOG_FAULTS");
            return env && env[0] == '1';
        }();
        if (log_faults) {
            static std::mutex mutex;
            static std::unordered_map<u64, u64> counts;
            static auto last = std::chrono::steady_clock::now();
            const u64 base = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
            const u64 rip = pointers->ContextRecord->Rip;
            std::scoped_lock lock{mutex};
            ++counts[rip - base];
            const auto now = std::chrono::steady_clock::now();
            if (now - last >= std::chrono::seconds(5)) {
                last = now;
                std::vector<std::pair<u64, u64>> top(counts.begin(), counts.end());
                std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second > b.second; });
                std::string line;
                for (std::size_t i = 0; i < std::min<std::size_t>(top.size(), 25); ++i) {
                    line += fmt::format(" {:#x}={}", top[i].first, top[i].second);
                }
                LOG_WARNING(Lib_GnmDriver, "DIAG-077: write faults by exe offset ({} sites):{}",
                            top.size(), line);
            }
        }
        return 1;
    }
    return Core::RytyMemory::InGuardedRead() ? 2 : 0;
}

// FIX-044 for Ryty: the guest's TSC, which stands still while the GPU command thread compiles
// shaders or pipelines (common/guest_clock.h). Ryty's libkernel derives the game's monotonic clocks
// from it, so a compile stall does not look like hundreds of milliseconds of game time (GT Sport's
// car thumbnail shot then breaks: "file_size > 10 * 1024" when a new car is bought).
extern "C" __declspec(dllexport) u64 RytyGpuGuestTsc() {
    return Common::GuestClock::ReadTsc();
}

// Ryty calls this before the kernel writes guest memory directly (a file read into a guest
// buffer), which would fail on a write-protected page instead of faulting.
extern "C" __declspec(dllexport) void RytyGpuInvalidateMemory(u64 address, u64 size) {
    if (auto* rasterizer = Core::Memory::Instance()->GetRasterizer()) {
        rasterizer->InvalidateMemory(address, size);
    }
}
