//  SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#include "ipc.h"

#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

#include <fmt/format.h>

#include <SDL3/SDL.h>

#include "common/elf_info.h"
#include "common/memory_patcher.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "common/types.h"
#include "core/debug_state.h"
#include "core/debugger.h"
#include "core/emulator_settings.h"
#include "core/emulator_state.h"
#include "core/libraries/audio/audioout.h"
#include "input/controller.h"
#include "input/input_handler.h"
#include "sdl_window.h"
#include "src/core/libraries/usbd/usbd.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;

/**
 * Protocol summary:
 * - IPC is enabled by setting the SHADPS4_ENABLE_IPC environment variable to "true"
 * - Input will be stdin & output stderr
 * - Strings are sent as UTF8
 * - Each communication line is terminated by a newline character ('\n')
 * - Each command parameter will be separated by a newline character ('\n'),
 *   variadic commands will start sending the number of parameters after the cmd word.
 *   Any ('\n') in the parameter must be escaped by a backslash ('\\n')
 * - Numbers can be sent with any base. Must prefix the number with '0x' for hex,
 *   '0b' for binary, or '0' for octal. Decimal numbers
 *   will be sent without any prefix.
 * - Output will be started by (';')
 * - The IPC server(this) will send a block started by
 *   #IPC_ENABLED
 *   and ended by
 *   #IPC_END
 *   In between, it will send the current capabilities and commands before the emulator start
 * - The IPC client(e.g., launcher) will send RUN then START to continue the execution
 **/

/**
 * Command list:
 * - CAPABILITIES:
 *   - ENABLE_MEMORY_PATCH: enables PATCH_MEMORY command
 *   - ENABLE_EMU_CONTROL: enables PAUSE, RESUME, STOP, TOGGLE_FULLSCREEN commands
 *   - ENABLE_TEST_AUTOMATION: enables PAD, SCREENSHOT, STATUS commands
 * - INPUT CMD:
 *   - RUN: start the emulator execution
 *   - START: start the game execution
 *   - PATCH_MEMORY(
 *       modName: str, offset: str, value: str,
 *       target: str, size: str, isOffset: number, littleEndian: number,
 *       patchMask: number, maskOffset: number
 *     ): add a memory patch, check @ref MemoryPatcher::PatchMemory for details
 *   - PAUSE: pause the game execution
 *   - RESUME: resume the game execution
 *   - STOP: stop and quit the emulator
 *   - TOGGLE_FULLSCREEN: enable / disable fullscreen
 *   - PAD(
 *       buttons: number, lx: number, ly: number, rx: number, ry: number,
 *       l2: number, r2: number, hold_frames: number
 *     ): inject player 1's pad state for hold_frames presented frames (0 clears it).
 *       buttons is an OrbisPadButtonDataOffset mask (Cross = 0x4000); sticks are 0-255 with
 *       128 centred, triggers 0-255. It is merged with real input and needs no window focus.
 *       Replies ;PAD_OK <until_frame>
 *   - SCREENSHOT(path: str): save the next presented game frame (before host scaling) as a
 *       PNG at path. Replies ;SCREENSHOT_QUEUED <path>; the file appears once the frame is
 *       presented and written.
 *   - STATUS: replies one line
 *       ;STATUS frames=<n> fps=<x> paused=<0|1> serial=<id> app_ver=<v> title=<rest of line>
 *       fps is measured over 500 ms while the command runs.
 * - OUTPUT CMD:
 *   - RESTART(argn: number, argv: ...string): Request restart of the emulator, must call STOP
 **/

void IPC::Init() {
    const char* enabledEnv = std::getenv("SHADPS4_ENABLE_IPC");
    enabled = enabledEnv && strcmp(enabledEnv, "true") == 0;
    if (!enabled) {
        return;
    }

    EmulatorState::GetInstance()->SetAutoPatchesLoadEnabled(false);
    RedirectToPipes();

    input_thread = std::jthread([this] {
        Common::SetCurrentThreadName("IPC Read thread");
        this->InputLoop();
    });

    std::cerr << ";#IPC_ENABLED\n";
    std::cerr << ";ENABLE_MEMORY_PATCH\n";
    std::cerr << ";ENABLE_EMU_CONTROL\n";
    std::cerr << ";ENABLE_TEST_AUTOMATION\n";
    std::cerr << ";#IPC_END\n";
    std::cerr.flush();

    const auto ok = run_semaphore.try_acquire_for(std::chrono::seconds(5));
    if (!ok) {
        std::cerr << "IPC: Failed to acquire run semaphore, closing process.\n";
        exit(1);
    }
}

void IPC::RedirectToPipes() {
#ifdef _WIN32
    // DIAG-056: under a profiler launcher (NVIDIA Nsight Systems) the emulator's stdin/stderr
    // are not the client's, so the client names two pipes it created: commands come in on
    // SHADPS4_IPC_PIPE_IN, replies go out on SHADPS4_IPC_PIPE_OUT (stderr is redirected there).
    const char* in_name = std::getenv("SHADPS4_IPC_PIPE_IN");
    const char* out_name = std::getenv("SHADPS4_IPC_PIPE_OUT");
    if (!in_name || !out_name || !*in_name || !*out_name) {
        return;
    }
    const auto open_pipe = [](const char* name, DWORD access, int flags) {
        const HANDLE handle =
            CreateFileA(name, access, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            std::cerr << "IPC: cannot open pipe " << name << "\n";
            exit(1);
        }
        return _open_osfhandle(reinterpret_cast<intptr_t>(handle), flags);
    };
    const int in_fd = open_pipe(in_name, GENERIC_READ, _O_RDONLY | _O_BINARY);
    const int out_fd = open_pipe(out_name, GENERIC_WRITE, _O_WRONLY | _O_BINARY);
    std::cerr.flush();
    _dup2(in_fd, 0);
    _dup2(out_fd, 2);
    std::cin.clear();
#endif
}

void IPC::SendRestart(const std::vector<std::string>& args) {
    std::cerr << ";RESTART\n";
    std::cerr << ";" << args.size() << "\n";
    for (const auto& arg : args) {
        std::cerr << ";" << arg << "\n";
    }
    std::cerr.flush();
}

void IPC::InputLoop() {
    auto next_str = [&] -> const std::string& {
        static std::string line_buffer;
        do {
            if (!std::getline(std::cin, line_buffer, '\n')) {
                // The client closed stdin: stop reading instead of spinning on EOF.
                while (true) {
                    std::this_thread::sleep_for(std::chrono::hours(1));
                }
            }
        } while (!line_buffer.empty() && line_buffer.back() == '\\');
        return line_buffer;
    };
    auto next_u64 = [&] -> u64 {
        auto& str = next_str();
        return std::stoull(str, nullptr, 0);
    };

    while (true) {
        auto& cmd = next_str();
        if (cmd.empty()) {
            continue;
        }
        if (cmd == "RUN") {
            run_semaphore.release();
        } else if (cmd == "START") {
            start_semaphore.release();
        } else if (cmd == "PATCH_MEMORY") {
            const MemoryPatcher::patchInfo entry = {
                .gameSerial = "*",
                .modNameStr = next_str(),
                .offsetStr = next_str(),
                .valueStr = next_str(),
                .targetStr = next_str(),
                .sizeStr = next_str(),
                .isOffset = next_u64() != 0,
                .littleEndian = next_u64() != 0,
                .patchMask = static_cast<MemoryPatcher::PatchMask>(next_u64()),
                .maskOffset = static_cast<int>(next_u64()),
            };
            MemoryPatcher::AddPatchToQueue(entry);
        } else if (cmd == "PAUSE") {
            DebugState.PauseGuestThreads();
        } else if (cmd == "RESUME") {
            DebugState.ResumeGuestThreads();
        } else if (cmd == "STOP") {
            SDL_Event event;
            SDL_memset(&event, 0, sizeof(event));
            event.type = SDL_EVENT_QUIT;
            SDL_PushEvent(&event);
        } else if (cmd == "TOGGLE_FULLSCREEN") {
            SDL_Event event;
            SDL_memset(&event, 0, sizeof(event));
            event.type = SDL_EVENT_TOGGLE_FULLSCREEN;
            SDL_PushEvent(&event);
        } else if (cmd == "PAD") {
            const u32 buttons = static_cast<u32>(next_u64());
            std::array<s32, 6> axes{};
            for (auto& axis : axes) {
                axis = static_cast<s32>(next_u64());
            }
            const u32 hold_frames = static_cast<u32>(next_u64());
            auto& controllers = *Common::Singleton<Input::GameControllers>::Instance();
            controllers[0]->Inject(buttons, axes, hold_frames);
            std::cerr << ";PAD_OK " << DebugState.GetFrameNum() + hold_frames << std::endl;
        } else if (cmd == "SCREENSHOT") {
            const std::string path = next_str();
            VideoCore::RequestScreenshotToPath(
                std::filesystem::path(std::u8string(path.begin(), path.end())));
            std::cerr << ";SCREENSHOT_QUEUED " << path << std::endl;
        } else if (cmd == "STATUS") {
            using namespace std::chrono;
            const u32 frames_before = DebugState.GetFrameNum();
            const auto time_before = steady_clock::now();
            std::this_thread::sleep_for(milliseconds(500));
            const u32 frames = DebugState.GetFrameNum();
            const double seconds = duration<double>(steady_clock::now() - time_before).count();
            const auto& elf = Common::ElfInfo::Instance();
            std::cerr << fmt::format(";STATUS frames={} fps={:.1f} paused={} serial={} "
                                     "app_ver={} title={}",
                                     frames, (frames - frames_before) / seconds,
                                     DebugState.IsGuestThreadsPaused() ? 1 : 0, elf.GameSerial(),
                                     elf.AppVer(), elf.Title())
                      << std::endl;
        } else if (cmd == "ADJUST_VOLUME") {
            int value = static_cast<int>(next_u64());
            bool is_game_specific = next_u64() != 0;
            EmulatorSettings.SetVolumeSlider(value, is_game_specific);
            Libraries::AudioOut::AdjustVol();
        } else if (cmd == "SET_FSR") {
            bool use_fsr = next_u64() != 0;
            if (presenter) {
                presenter->GetFsrSettingsRef().enable = use_fsr;
            }
        } else if (cmd == "SET_RCAS") {
            bool use_rcas = next_u64() != 0;
            if (presenter) {
                presenter->GetFsrSettingsRef().use_rcas = use_rcas;
            }
        } else if (cmd == "SET_RCAS_ATTENUATION") {
            int value = static_cast<int>(next_u64());
            if (presenter) {
                presenter->GetFsrSettingsRef().rcas_attenuation =
                    static_cast<float>(value / 1000.0f);
            }
        } else if (cmd == "USB_LOAD_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                std::string file_name = next_str();
                const u8 pad = next_u64();
                const u8 slot = next_u64();
                ref->LoadFigure(file_name, pad, slot);
            }
        } else if (cmd == "USB_REMOVE_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                const u8 pad = next_u64();
                const u8 slot = next_u64();
                bool full_remove = next_u64() != 0;
                ref->RemoveFigure(pad, slot, full_remove);
            }
        } else if (cmd == "USB_MOVE_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                const u8 new_pad = next_u64();
                const u8 new_index = next_u64();
                const u8 old_pad = next_u64();
                const u8 old_index = next_u64();
                ref->MoveFigure(new_pad, new_index, old_pad, old_index);
            }
        } else if (cmd == "USB_TEMP_REMOVE_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                const u8 index = next_u64();
                ref->TempRemoveFigure(index);
            }
        } else if (cmd == "USB_CANCEL_REMOVE_FIGURE") {
            const auto ref = Libraries::Usbd::usb_backend->GetImplRef();
            if (ref) {
                const u8 index = next_u64();
                ref->CancelRemoveFigure(index);
            }
        } else if (cmd == "RELOAD_INPUTS") {
            std::string config = next_str();
            Input::ParseInputConfig(config);
        } else {
            std::cerr << ";UNKNOWN CMD: " << cmd << std::endl;
        }
    }
}
