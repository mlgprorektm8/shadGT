// Emulator-core pieces the renderer and window reference but that have no role when Ryty runs the
// game: the emulator object, guest signal dispatch, CPU patching, game archives and the overlays.

#include <cstdio>
#include "core/libraries/kernel/kernel.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <vector>

#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/native_clock.h"
#include "core/emulator_settings.h"
#include "core/cpu_patches.h"
#include "core/devtools/layer.h"
#include "core/file_format/npbind.h"
#include "core/file_sys/ifile.h"
#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/keyboard/keyboard.h"
#include "core/libraries/mouse/sdl_mouse.h"
#include "core/libraries/system/systemservice.h"
#include "core/libraries/system/userservice.h"
#include "core/memory.h"
#include "emulator.h"
#include "imgui/big_picture/settings_dialog_layer.h"
#include "imgui/friends_layer.h"
#include "imgui/invitation_prompt_layer.h"
#include "imgui/notifications_layer.h"
#include "imgui/shadnet_notifications_layer.h"
#include "ryty_gpu/ryty_library.h"

namespace Ryty {
static void (*g_overlay_draw)() = nullptr;

void SetOverlayDraw(void (*draw)()) {
    g_overlay_draw = draw;
}
} // namespace Ryty

namespace Core {

// Constructed once at renderer startup. As ElfInfo's friend, it fills in what shadGT's loader
// would have read from the game: GT Sport 1.69 (CUSA03220), SDK 5.05.
Emulator::Emulator() : memory{}, controllers{}, linker{} {
    auto& game_info = Common::ElfInfo::Instance();
    game_info.initialized = true;
    game_info.game_serial = "CUSA03220";
    game_info.title = "Gran Turismo Sport";
    game_info.app_ver = "01.69";
    game_info.firmware_ver = 0x05050000;
    game_info.raw_firmware_ver = 0x05050000;
    game_info.sdk_ver = 0x05050001;
    game_info.game_folder = std::filesystem::current_path() / "app0";

    const auto sce_sys = game_info.game_folder / "sce_sys";

    // The splash screen the console shows while the game starts (Show Splash setting).
    if (std::ifstream splash{sce_sys / "pic1.png", std::ios::binary}) {
        game_info.splash_data.assign(std::istreambuf_iterator<char>{splash},
                                     std::istreambuf_iterator<char>{});
    }

    // NpTrophy registers a context only for a service label listed here (trophyNN.trp to the
    // NNth communication ID in npbind.dat), as shadGT's loader fills it.
    NPBindFile npbind;
    if (npbind.Load((sce_sys / "npbind.dat").string())) {
        const auto np_comm_ids = npbind.GetNpCommIds();
        game_info.SetNpCommIds(np_comm_ids);
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(sce_sys / "trophy", error)) {
            auto name = entry.path().filename().string();
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (!name.starts_with("trophy") || !name.ends_with(".trp")) {
                continue;
            }
            const auto index = std::atoi(name.substr(6, name.size() - 10).c_str());
            if (index >= 0 && index < static_cast<int>(np_comm_ids.size())) {
                game_info.trophy_index_map[index] = np_comm_ids[index];
            }
        }
    }
}
Emulator::~Emulator() = default;

void Emulator::Shutdown() {
    // Skip DLL_PROCESS_DETACH: the GPU driver's teardown can deadlock when the exit starts here.
    TerminateProcess(GetCurrentProcess(), 0);
}

std::string MemoryManager::DescribeAddress(VAddr addr) {
    return fmt::format("{:#x} (mapped by Ryty)", addr);
}

#ifndef RYTY_HAS_CPU_PATCHES // the renderer library builds core/cpu_patches.cpp (FIX-074)
namespace WindowsGuestRedZoneProtection {
bool IsStaticPatchingEnabled() noexcept {
    return false;
}
} // namespace WindowsGuestRedZoneProtection
#endif

namespace FileSys {
bool IsZArchiveFile(const std::filesystem::path&) {
    return false;
}
std::optional<std::vector<u8>> ReadGameFile(const std::filesystem::path&, std::string_view) {
    return std::nullopt;
}
} // namespace FileSys

namespace Devtools {
void Layer::SetupSettings() {}
void Layer::Draw() {
    if (Ryty::g_overlay_draw) {
        Ryty::g_overlay_draw();
    }
}
bool Layer::ShouldKeepDrawing() {
    return false;
}
void Layer::DrawNullGpuNotice() {}
} // namespace Devtools

} // namespace Core

namespace Overlay {
void ToggleSimpleFps() {}
void ToggleQuitWindow() {}
void ShowVolume() {}
} // namespace Overlay

namespace Libraries::Kernel {

// Guest signals are handled by Ryty; the renderer's fault handler never forwards to the guest.
thread_local Pthread* g_curthread = nullptr;

Ucontext::Ucontext(PCONTEXT context) : host_context(context) {}

bool Pthread::DispatchSignal(s32, Siginfo*, Ucontext*) {
    return false;
}

u64 PS4_SYSV_ABI sceKernelGetEventData(const OrbisKernelEvent* ev) {
    return ev->data;
}

namespace Dev {
u64& GetInitialPtc() {
    static u64 initial_ptc = 0;
    return initial_ptc;
}
Common::NativeClock* GetClock() {
    static Common::NativeClock clock;
    return &clock;
}
} // namespace Dev

#ifndef RYTY_GUEST_PTHREADS
// The NP guest-pthread bridge provides its own firmware query.
// ryGT: for the on-screen keyboard dialog (its panel size depends on the firmware version).
s32 PS4_SYSV_ABI sceKernelGetSystemSwVersion(SwVersionStruct* ret) {
    if (ret != nullptr) {
        constexpr u32 firmware = 0x09000000;
        ret->hex_representation = firmware;
        std::snprintf(ret->text_representation, 28, "%2x.%03x.%03x", firmware >> 0x18,
                      firmware >> 0xc & 0xfff, firmware & 0xfff);
    }
    return 0;
}

#endif
} // namespace Libraries::Kernel

namespace Libraries::SystemService {
// The game's splash screen (sce_sys/pic1.png), with the Show Splash setting. GT Sport asks to hide
// it right at start (sceSystemServiceHideSplashScreen, which Ryty's libSceSystemService forwards
// here), long before it draws anything, so the splash stays until the game's first drawn frame
// too, as the console shows it through the loading.
static std::atomic<bool> g_splash_hidden{false};
static std::atomic<bool> g_game_drew{false};

bool IsSplashVisible() {
    // PERF-102: also while the shader precompile holds the game.
    return Vulkan::PrecompileHoldsGame() ||
           (EmulatorSettings.IsShowSplash() && !(g_splash_hidden.load(std::memory_order_relaxed) &&
                                                 g_game_drew.load(std::memory_order_relaxed)));
}

void NoteGameDrew() {
    g_game_drew.store(true, std::memory_order_relaxed);
}
} // namespace Libraries::SystemService

extern "C" __declspec(dllexport) void RytyGpuHideSplashScreen() {
    if (!Libraries::SystemService::g_splash_hidden.exchange(true)) {
        LOG_INFO(Lib_SystemService, "The game hid its splash screen");
    }
}

// Keyboard and mouse input reach the game through Ryty's own libraries.
namespace Libraries::Keyboard {
bool PushSDLEvent(SDL_Event&) {
    return false;
}
} // namespace Libraries::Keyboard
namespace Libraries::Mouse {
bool PushSDLEvent(SDL_Event const&) {
    return false;
}
} // namespace Libraries::Mouse

namespace ImGui::Friends {
void Unregister() {}
void Toggle() {}
} // namespace ImGui::Friends
namespace ImGui::InvitationPrompt {
void Unregister() {}
} // namespace ImGui::InvitationPrompt
namespace ImGui::ShadNetNotify {
void Unregister() {}
} // namespace ImGui::ShadNetNotify

namespace ImGuiEmuSettings {
void OpenInGameSettingsDialog() {}
} // namespace ImGuiEmuSettings

namespace shadNotifications {
void QueueNotification(std::string, float, position, stockIcons) {}
void QueueNotification(std::string, float, position, std::vector<u8>, bool) {}
} // namespace shadNotifications

// User login events go to the game through Ryty's libSceUserService.
namespace Libraries::UserService {
void AddUserServiceEvent(OrbisUserServiceEvent) {}
} // namespace Libraries::UserService
