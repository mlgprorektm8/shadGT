// Controller input for the Ryty build: shadGT's window and input handler fill its GameController
// states (keyboard bindings and SDL gamepads), and this thread forwards them to Ryty's libScePad,
// which is what the game reads. The game's rumble and light bar requests travel back the same way.

#include <windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <nlohmann/json.hpp>

#include "common/logging/log.h"
#include "common/singleton.h"
#include "input/controller.h"
#include "video_core/renderdoc.h"
#include "ryty_gpu/ryty_input_bridge.h"

namespace {

// Mirrors Ryty's PadInputState (core/libs/prx/libScePad/include/PadState.hpp).
struct RytyPadTouchPoint {
    bool active = false;
    std::uint16_t x = 0;
    std::uint16_t y = 0;
    std::uint8_t id = 0;
};

struct RytyPadInputState {
    std::uint32_t buttons = 0;
    std::array<std::uint8_t, 4> sticks{128, 128, 128, 128};
    std::uint8_t analogButtonsL2 = 0;
    std::uint8_t analogButtonsR2 = 0;
    bool touchLeft = false;
    bool touchRight = false;
    bool hasMotion = false;
    std::array<float, 3> accel{0.0f, 9.80665f, 0.0f};
    std::array<float, 3> gyro{0.0f, 0.0f, 0.0f};
    std::array<RytyPadTouchPoint, 2> touch{};
    std::uint8_t deviceKind = 0;
};

// Mirrors Ryty's PadOutputState: what the game asked of the controller (rumble, light bar).
struct RytyPadTriggerRequest {
    bool valid = false;
    std::uint8_t effect[11]{};
    std::uint8_t fallback = 0;
    std::uint32_t mode = 0;
};

struct RytyPadOutputState {
    std::uint32_t sequence = 0;
    std::uint8_t vibrationLarge = 0;
    std::uint8_t vibrationSmall = 0;
    bool lightBarValid = false;
    std::uint8_t lightBar[3]{};
    RytyPadTriggerRequest trigger[2];
    bool triggerTouched = false;
    bool motionEnabled = true;
    int vibrationMode = 0;
};

// Test hook: RYTY_TEST_SCRIPT="50:shot;52:cross;65:shot" presses buttons and saves game-only
// screenshots at the given seconds after the renderer starts, without touching the host keyboard.
// "r2on"/"r2off" hold the accelerator (R2, fully pressed) until released.
std::atomic<std::uint32_t> g_scripted_buttons{0};
std::atomic<bool> g_scripted_r2{false};
// "steerN" holds the left stick's X axis at N (0 full left, 128 centre, 255 full right) until
// "steeroff".
std::atomic<int> g_scripted_lx{-1};

std::uint32_t ButtonFromName(std::string_view name) {
    using B = Libraries::Pad::OrbisPadButtonDataOffset;
    static const std::pair<std::string_view, B> names[] = {
        {"cross", B::Cross}, {"circle", B::Circle}, {"square", B::Square}, {"triangle", B::Triangle},
        {"options", B::Options}, {"up", B::Up}, {"down", B::Down}, {"left", B::Left}, {"right", B::Right},
        {"l1", B::L1}, {"r1", B::R1}, {"touchpad", B::TouchPad}};
    for (const auto& [key, button] : names) {
        if (key == name) return static_cast<std::uint32_t>(button);
    }
    return 0;
}

void RunTestScript(std::string script) {
    const auto start = std::chrono::steady_clock::now();
    std::size_t position = 0;
    while (position < script.size()) {
        const auto end = std::min(script.find(';', position), script.size());
        const std::string step = script.substr(position, end - position);
        position = end + 1;
        const auto colon = step.find(':');
        if (colon == std::string::npos) continue;
        const double at = std::stod(step.substr(0, colon));
        const std::string action = step.substr(colon + 1);
        std::this_thread::sleep_until(start + std::chrono::milliseconds(static_cast<long long>(at * 1000)));
        if (action == "r2on" || action == "r2off") {
            g_scripted_r2 = action == "r2on";
            LOG_INFO(Input, "Ryty test script: {} at {} s", action, at);
        } else if (action.starts_with("steer")) {
            g_scripted_lx = action == "steeroff" ? -1 : std::stoi(action.substr(5));
            LOG_INFO(Input, "Ryty test script: {} at {} s", action, at);
        } else if (action == "shot") {
            VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::GameOnly);
            LOG_INFO(Input, "Ryty test script: screenshot at {} s", at);
        } else if (action == "shotwindow") {
            // What the window shows (the splash screen, overlays), not just the game's image.
            VideoCore::RequestScreenshot(VideoCore::ScreenshotRequest::WithOverlays);
            LOG_INFO(Input, "Ryty test script: window screenshot at {} s", at);
        } else if (action.starts_with("burst") || action.starts_with("wburst")) {
            // "burstN": N screenshot requests; one is taken per frame, so N consecutive frames.
            // "wburstN" takes them of the window (splash screen and overlays included).
            const bool window = action[0] == 'w';
            const std::string count_text = action.substr(window ? 6 : 5);
            const int count = count_text.empty() ? 30 : std::stoi(count_text);
            for (int i = 0; i < count; ++i) {
                VideoCore::RequestScreenshot(window ? VideoCore::ScreenshotRequest::WithOverlays
                                                    : VideoCore::ScreenshotRequest::GameOnly);
            }
            LOG_INFO(Input, "Ryty test script: {} screenshots from {} s", count, at);
        } else if (const auto button = ButtonFromName(action); button != 0) {
            g_scripted_buttons |= button;
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            g_scripted_buttons &= ~button;
            LOG_INFO(Input, "Ryty test script: pressed {} at {} s", action, at);
        }
    }
}

// libScePad.prx: PadPublishInput (Microsoft ABI, exported by NID).
using PublishFn = void (*)(const RytyPadInputState&);
// libScePad.prx: PadFetchOutput (Microsoft ABI, exported by NID); true when the output changed.
using FetchOutputFn = bool (*)(std::uint32_t*, RytyPadOutputState*);
// Per player (slot 0-3) versions: PadPublishInputSlot, PadFetchOutputSlot, and PadSetConnectedSlots
// (bit N: player N+1 has a controller). libSceUserService.prx: UserServiceSetPlayers logs players
// 2-4 in and out with the same mask.
using PublishSlotFn = void (*)(int, const RytyPadInputState&);
using FetchOutputSlotFn = bool (*)(int, std::uint32_t*, RytyPadOutputState*);
using SetMaskFn = void (*)(std::uint32_t);

constexpr int MaxPlayers = 4;

// The GT Sport config next to GTSport.exe (user/custom_configs/CUSA03220.json) as text.
std::string ReadGameConfig() {
    char path[MAX_PATH];
    const DWORD length = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) {
        return {};
    }
    std::string config{path, length};
    config = config.substr(0, config.find_last_of("/\\") + 1) + "user/custom_configs/CUSA03220.json";
    std::FILE* file = std::fopen(config.c_str(), "rb");
    if (!file) {
        return {};
    }
    std::string text;
    char chunk[4096];
    while (const size_t n = std::fread(chunk, 1, sizeof(chunk), file)) {
        text.append(chunk, n);
    }
    std::fclose(file);
    return text;
}

// Input.keyboard_player in the GT Sport config (the launcher's Controls window): the player the
// keyboard controls (its bindings name that player); that player is logged in without a gamepad.
int KeyboardPlayer() {
    const std::string text = ReadGameConfig();
    const size_t key = text.find("\"keyboard_player\"");
    if (key == std::string::npos) {
        return 1;
    }
    const size_t value = text.find_first_not_of(" \t\r\n:", key + 17);
    if (value == std::string::npos || text[value] < '1' || text[value] > '4') {
        return 1;
    }
    return text[value] - '0';
}

std::uint8_t ClampAxis(s32 value) {
    return static_cast<std::uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

unsigned TriggerStrength() {
    // The launcher writes per-game Input settings. Read at most once per second,
    // so the slider can be adjusted while driving without restarting the game.
    static auto next = std::chrono::steady_clock::time_point{};
    static unsigned percent = 100;
    const auto now = std::chrono::steady_clock::now();
    if (now < next) return percent;
    next = now + std::chrono::seconds(1);
    const auto config = nlohmann::json::parse(ReadGameConfig(), nullptr, false);
    if (config.is_discarded() || !config.is_object()) return percent;
    const auto input = config.find("Input");
    if (input == config.end() || !input->is_object()) return percent = 100;
    const auto value = input->find("trigger_strength");
    if (value == input->end() || !value->is_number_integer()) return percent = 100;
    // Clamp using a wide type before conversion, including hand-edited JSON.
    if (value->is_number_unsigned()) return percent = static_cast<unsigned>(std::min(value->get<std::uint64_t>(), std::uint64_t{200}));
    return percent = static_cast<unsigned>(std::clamp(value->get<std::int64_t>(), std::int64_t{0}, std::int64_t{200}));
}

void ForwardInput() {
    HMODULE pad = GetModuleHandleA("libScePad.prx");
    const auto publish = pad ? reinterpret_cast<PublishFn>(GetProcAddress(pad, "MCBKDgnIVB8")) : nullptr;
    if (publish == nullptr) {
        LOG_ERROR(Input, "Ryty: libScePad.prx has no PadPublishInput; controller input is disabled");
        return;
    }
    const auto fetch_output =
        reinterpret_cast<FetchOutputFn>(GetProcAddress(pad, "FDjxaWJZf9Y"));
    const auto publish_slot = reinterpret_cast<PublishSlotFn>(GetProcAddress(pad, "bzBgHlPTQcw"));
    const auto fetch_output_slot =
        reinterpret_cast<FetchOutputSlotFn>(GetProcAddress(pad, "oEbP2t18XtY"));
    const auto set_pad_mask = reinterpret_cast<SetMaskFn>(GetProcAddress(pad, "WeP2kdNeVQs"));
    HMODULE user_service = GetModuleHandleA("libSceUserService.prx");
    const auto set_players =
        user_service ? reinterpret_cast<SetMaskFn>(GetProcAddress(user_service, "pupOOOgZWjs"))
                     : nullptr;
    // Players 2-4 need the per-player exports; without them only player 1 is forwarded.
    const int players = publish_slot && fetch_output_slot && set_pad_mask ? MaxPlayers : 1;
    LOG_INFO(Input, "Ryty: forwarding {} player(s){}", players,
             set_players ? "" : " (no player logins: libSceUserService.prx is older)");
    auto* controllers = Common::Singleton<Input::GameControllers>::Instance();
    std::array<RytyPadInputState, MaxPlayers> last{};
    std::array<bool, MaxPlayers> first{true, true, true, true};
    std::array<std::uint32_t, MaxPlayers> output_seen{};
    std::array<RytyPadOutputState, MaxPlayers> last_output{};
    std::uint32_t last_mask = ~0u;
    const int keyboard_player = players > 1 ? KeyboardPlayer() : 1;
    if (keyboard_player > 1) {
        LOG_INFO(Input, "Ryty: the keyboard controls player {}", keyboard_player);
    }
    while (true) {
        // Player 1 always plays (keyboard and first gamepad); players 2-4 while shadGT has a
        // gamepad in their slot (the user folder's users.json lists who players 2-4 are).
        std::uint32_t mask = 1u | (1u << (keyboard_player - 1));
        for (int slot = 1; slot < players; ++slot) {
            if ((*controllers)[slot]->m_sdl_gamepad != nullptr) {
                mask |= 1u << slot;
            }
        }
        if (mask != last_mask) {
            if (set_pad_mask) set_pad_mask(mask);
            if (set_players) set_players(mask);
            if (last_mask != ~0u) {
                LOG_INFO(Input, "Ryty: players with a controller: {:#x}", mask);
            }
            last_mask = mask;
        }
        for (int slot = 0; slot < players; ++slot) {
            if (slot > 0 && (mask >> slot & 1u) == 0) {
                continue;
            }
            auto& controller = *(*controllers)[slot];
            // Rumble and light bar requests go to that player's controller, as shadGT's pad
            // library would send them (SetVibration takes the small motor first).
            RytyPadOutputState output;
            const bool changed = players > 1
                                     ? fetch_output_slot(slot, &output_seen[slot], &output)
                                     : fetch_output && fetch_output(&output_seen[slot], &output);
            if (changed) {
                auto& previous = last_output[slot];
                if (output.vibrationLarge != previous.vibrationLarge ||
                    output.vibrationSmall != previous.vibrationSmall) {
                    const bool sent =
                        controller.SetVibration(output.vibrationSmall, output.vibrationLarge);
                    static int logged = 0;
                    if (logged++ < 10) {
                        LOG_INFO(Input, "Ryty: player {} rumble large {} small {} -> controller ({})",
                                 slot + 1, output.vibrationLarge, output.vibrationSmall,
                                 controller.m_sdl_gamepad ? (sent ? "sent" : "refused")
                                                          : "no gamepad");
                    }
                }
                if (output.lightBarValid &&
                    std::memcmp(output.lightBar, previous.lightBar, sizeof(output.lightBar)) != 0) {
                    controller.SetLightBarRGB(output.lightBar[0], output.lightBar[1],
                                              output.lightBar[2]);
                }
                previous = output;
            }
            // Telemetry describes the locally driven car (player 1 only). Explicit
            // game trigger commands take priority, including an explicit release.
            const unsigned strength = TriggerStrength();
            const auto feedback = slot == 0 ? Input::TriggerFeedback::CurrentOutput(strength)
                                           : Input::TriggerFeedback::Output{};
            u8 left_rumble = feedback.left_rumble, right_rumble = feedback.right_rumble;
            auto left = feedback.left, right = feedback.right;
            const auto& saved = last_output[slot];
            if (saved.triggerTouched) {
                std::copy_n(saved.trigger[0].effect, 11, left.begin());
                std::copy_n(saved.trigger[1].effect, 11, right.begin());
                // An untouched side is a release rather than an all-zero vendor command.
                if (!saved.trigger[0].valid) left = Input::TriggerFeedback::Vibration(0);
                if (!saved.trigger[1].valid) right = Input::TriggerFeedback::Vibration(0);
                left = Input::TriggerFeedback::ScaleEffect(left, strength);
                right = Input::TriggerFeedback::ScaleEffect(right, strength);
                left_rumble = Input::TriggerFeedback::Scale(saved.trigger[0].fallback, strength);
                right_rumble = Input::TriggerFeedback::Scale(saved.trigger[1].fallback, strength);
            }
            const char* triggers = std::getenv("RYGT_ADAPTIVE_TRIGGERS");
            if ((triggers && triggers[0] == '0') || Input::TriggerFeedback::stopping) {
                left = right = Input::TriggerFeedback::Vibration(0);
                left_rumble = right_rumble = 0;
            }
            controller.SetTriggerFeedback(left, right, left_rumble, right_rumble);
            const auto state = controller.ReadState();
            RytyPadInputState input{};
            input.buttons = static_cast<std::uint32_t>(state.buttonsState);
            input.sticks = {ClampAxis(state.axes[0]), ClampAxis(state.axes[1]),
                            ClampAxis(state.axes[2]), ClampAxis(state.axes[3])};
            input.analogButtonsL2 = ClampAxis(state.axes[4]);
            input.analogButtonsR2 = ClampAxis(state.axes[5]);
            if (slot == 0) {
                input.buttons |= g_scripted_buttons.load();
                if (const int lx = g_scripted_lx.load(); lx >= 0) {
                    input.sticks[0] = static_cast<std::uint8_t>(lx);
                }
                if (g_scripted_r2) {
                    input.buttons |=
                        static_cast<std::uint32_t>(Libraries::Pad::OrbisPadButtonDataOffset::R2);
                    input.analogButtonsR2 = 255;
                }
            }
            auto& previous = last[slot];
            if (first[slot] || input.buttons != previous.buttons ||
                input.sticks != previous.sticks ||
                input.analogButtonsL2 != previous.analogButtonsL2 ||
                input.analogButtonsR2 != previous.analogButtonsR2) {
                if (players > 1) {
                    publish_slot(slot, input);
                } else {
                    publish(input);
                }
                previous = input;
                first[slot] = false;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
}

// Input.controller_ports (the launcher's Controller ports window): "vvvv:pppp#n" per player,
// comma separated, empty for any: the n-th connected controller with USB vendor id vvvv and
// product id pppp (hex) plays as that player.
struct PortChoice {
    int vendor = -1;
    int product = -1;
    int nth = 0;
};
std::array<PortChoice, MaxPlayers> g_ports{};

int PreferredSlot(u16 vendor, u16 product, int nth) {
    for (int slot = 0; slot < MaxPlayers; ++slot) {
        const auto& port = g_ports[slot];
        if (port.vendor == vendor && port.product == product && port.nth == nth) {
            return slot;
        }
    }
    return -1;
}

} // namespace

namespace Ryty {

void InstallControllerPorts() {
    const std::string text = ReadGameConfig();
    const size_t key = text.find("\"controller_ports\"");
    if (key == std::string::npos) {
        return;
    }
    const size_t open = text.find('"', text.find(':', key) + 1);
    const size_t close = open == std::string::npos ? open : text.find('"', open + 1);
    if (close == std::string::npos) {
        return;
    }
    const std::string list = text.substr(open + 1, close - open - 1);
    size_t position = 0;
    bool any = false;
    for (int slot = 0; slot < MaxPlayers && position <= list.size(); ++slot) {
        const size_t end = std::min(list.find(',', position), list.size());
        const std::string entry = list.substr(position, end - position);
        position = end + 1;
        unsigned vendor = 0, product = 0;
        int nth = 0;
        if (std::sscanf(entry.c_str(), "%x:%x#%d", &vendor, &product, &nth) >= 2) {
            g_ports[slot] = {static_cast<int>(vendor), static_cast<int>(product), nth};
            LOG_INFO(Input, "Ryty: player {} is controller {:04x}:{:04x} #{}", slot + 1, vendor,
                     product, nth);
            any = true;
        }
    }
    if (any) {
        Input::g_preferred_controller_slot = PreferredSlot;
    }
}

void StartInputBridge() {
    std::thread(ForwardInput).detach();
    if (const char* script = std::getenv("RYTY_TEST_SCRIPT"); script != nullptr && *script != 0) {
        std::thread(RunTestScript, std::string(script)).detach();
    }
}

} // namespace Ryty
