// The in-game Controllers window (ryGT): which controller plays as player 1 and player 2 can be
// changed at any time while playing, with no restart. F3 or holding Share/View + Options/Menu on
// a controller opens and closes it; the mouse, the keyboard or a controller works it. While it is
// open the game gets released controls, and the choices are saved for the launcher and the next
// start (Input.controller_ports). It looks like the launcher's Controller ports window (same
// colours, layout and live controller preview) and scales with the game's resolution.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#include <imgui.h>
#include <SDL3/SDL_gamepad.h>

#include "common/singleton.h"
#include "imgui/renderer/imgui_core.h"
#include "input/controller.h"
#include "ryty_gpu/ryty_input_bridge.h"

namespace Ryty {
namespace {

// The launcher's dark theme (loader/RyGTLoader.cs, class Dark).
ImU32 Rgb(int r, int g, int b, float alpha = 1.0f) {
    return IM_COL32(r, g, b, static_cast<int>(std::clamp(alpha, 0.0f, 1.0f) * 255.0f));
}
ImVec4 Rgbf(int r, int g, int b, float alpha = 1.0f) {
    return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, alpha);
}
const ImVec4 kBack = Rgbf(30, 30, 30, 0.97f);
const ImVec4 kField = Rgbf(45, 45, 48);
const ImVec4 kButton = Rgbf(58, 58, 61);
const ImVec4 kButtonHover = Rgbf(75, 75, 80);
const ImVec4 kAccent = Rgbf(178, 24, 36);
const ImVec4 kAccentHover = Rgbf(206, 40, 52);
const ImVec4 kHeading = Rgbf(240, 96, 96);
const ImVec4 kBorder = Rgbf(70, 70, 74);
const ImVec4 kText = Rgbf(230, 230, 230);
const ImVec4 kDimText = Rgbf(150, 150, 150);

// Which controller the preview draws (as the launcher's preview: PortsForm.cs, PadKind).
enum class PadKind { DualShock4, DualSense, Xbox, Generic };

// What the preview shows of a controller, by position: cross is the bottom face button (Xbox A),
// circle the right one (B), square the left one (X), triangle the top one (Y); share is View on
// Xbox, options Menu, ps the PS or Xbox button. Sticks -1..1, triggers 0..1.
struct PadView {
    PadKind kind = PadKind::DualShock4;
    bool connected = false;
    ImU32 light = IM_COL32(40, 110, 230, 255);
    bool cross = false, circle = false, square = false, triangle = false;
    bool l1 = false, r1 = false, l3 = false, r3 = false;
    bool share = false, options = false, ps = false, touchpad = false;
    bool up = false, down = false, left = false, right = false;
    float lx = 0, ly = 0, rx = 0, ry = 0, l2 = 0, r2 = 0;
};

PadView ReadPad(Input::GameController* controller) {
    PadView v;
    SDL_Gamepad* pad = controller ? controller->m_sdl_gamepad : nullptr;
    if (!pad) {
        return v;
    }
    switch (SDL_GetGamepadType(pad)) {
    case SDL_GAMEPAD_TYPE_PS3:
    case SDL_GAMEPAD_TYPE_PS4:
        v.kind = PadKind::DualShock4;
        break;
    case SDL_GAMEPAD_TYPE_PS5:
        v.kind = PadKind::DualSense;
        break;
    case SDL_GAMEPAD_TYPE_XBOX360:
    case SDL_GAMEPAD_TYPE_XBOXONE:
        v.kind = PadKind::Xbox;
        break;
    default:
        v.kind = PadKind::Generic;
        break;
    }
    v.connected = SDL_GamepadConnected(pad);
    if (!v.connected) {
        return v;
    }
    const auto colour = controller->GetLightBarRGB();
    if (colour.r || colour.g || colour.b) {
        v.light = IM_COL32(colour.r, colour.g, colour.b, 255);
    }
    const auto button = [pad](SDL_GamepadButton b) { return SDL_GetGamepadButton(pad, b); };
    const auto axis = [pad](SDL_GamepadAxis a) {
        return std::clamp(SDL_GetGamepadAxis(pad, a) / 32767.0f, -1.0f, 1.0f);
    };
    v.cross = button(SDL_GAMEPAD_BUTTON_SOUTH);
    v.circle = button(SDL_GAMEPAD_BUTTON_EAST);
    v.square = button(SDL_GAMEPAD_BUTTON_WEST);
    v.triangle = button(SDL_GAMEPAD_BUTTON_NORTH);
    v.l1 = button(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
    v.r1 = button(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
    v.l3 = button(SDL_GAMEPAD_BUTTON_LEFT_STICK);
    v.r3 = button(SDL_GAMEPAD_BUTTON_RIGHT_STICK);
    v.share = button(SDL_GAMEPAD_BUTTON_BACK);
    v.options = button(SDL_GAMEPAD_BUTTON_START);
    v.ps = button(SDL_GAMEPAD_BUTTON_GUIDE);
    v.touchpad = button(SDL_GAMEPAD_BUTTON_TOUCHPAD);
    v.up = button(SDL_GAMEPAD_BUTTON_DPAD_UP);
    v.down = button(SDL_GAMEPAD_BUTTON_DPAD_DOWN);
    v.left = button(SDL_GAMEPAD_BUTTON_DPAD_LEFT);
    v.right = button(SDL_GAMEPAD_BUTTON_DPAD_RIGHT);
    v.lx = axis(SDL_GAMEPAD_AXIS_LEFTX);
    v.ly = axis(SDL_GAMEPAD_AXIS_LEFTY);
    v.rx = axis(SDL_GAMEPAD_AXIS_RIGHTX);
    v.ry = axis(SDL_GAMEPAD_AXIS_RIGHTY);
    v.l2 = std::max(0.0f, axis(SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
    v.r2 = std::max(0.0f, axis(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
    return v;
}

// A body outline on the 400 x 260 canvas, built like the launcher's GraphicsPath: elliptical arcs
// (GDI+ AddArc: bounding box, start and sweep in degrees), lines and cubic Béziers.
struct Outline {
    std::vector<ImVec2> points;
    void Add(float x, float y) {
        if (points.empty() || std::abs(points.back().x - x) > 0.01f ||
            std::abs(points.back().y - y) > 0.01f) {
            points.push_back({x, y});
        }
    }
    void Arc(float x, float y, float w, float h, float start, float sweep, int n = 16) {
        constexpr float deg = 3.14159265f / 180.0f;
        for (int i = 0; i <= n; ++i) {
            const float a = (start + sweep * i / n) * deg;
            Add(x + w * 0.5f + w * 0.5f * std::cos(a), y + h * 0.5f + h * 0.5f * std::sin(a));
        }
    }
    void Line(float x1, float y1, float x2, float y2) {
        Add(x1, y1);
        Add(x2, y2);
    }
    void Bezier(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4,
                int n = 20) {
        Add(x1, y1);
        for (int i = 1; i <= n; ++i) {
            const float t = float(i) / n, u = 1.0f - t;
            Add(u * u * u * x1 + 3 * u * u * t * x2 + 3 * u * t * t * x3 + t * t * t * x4,
                u * u * u * y1 + 3 * u * u * t * y2 + 3 * u * t * t * y3 + t * t * t * y4);
        }
    }
};

// The chosen controller drawn with its live state, as the launcher draws it: a 400 x 260 canvas
// scaled into `size` at `origin`.
class Painter {
public:
    Painter(ImDrawList* dl_, ImVec2 origin, ImVec2 size, const PadView& v_) : dl{dl_}, v{v_} {
        scale = std::min(size.x / 400.0f, size.y / 260.0f);
        base = {origin.x + (size.x - 400.0f * scale) * 0.5f, origin.y};
        alpha = v.connected ? 1.0f : 0.45f;
        edge = Rgb(95, 95, 104, alpha);
        off = Rgb(34, 34, 38, alpha);
        on = Rgb(206, 40, 52, alpha);
        thin = std::max(1.0f, scale);
    }

    void Paint(const std::string& caption) {
        switch (v.kind) {
        case PadKind::DualSense:
            DualSense();
            break;
        case PadKind::Xbox:
            Xbox(true);
            break;
        case PadKind::Generic:
            Xbox(false);
            break;
        default:
            DualShock4();
            break;
        }
        const ImVec2 text = ImGui::CalcTextSize(caption.c_str());
        dl->AddText({base.x + (400.0f * scale - text.x) * 0.5f, P(0, 238).y},
                    ImGui::GetColorU32(kText), caption.c_str());
    }

private:
    ImVec2 P(float x, float y) const {
        return {base.x + x * scale, base.y + y * scale};
    }
    ImU32 Fade(int r, int g, int b) const {
        return Rgb(r, g, b, alpha);
    }
    ImU32 FadeU32(ImU32 c) const {
        const ImU32 a = static_cast<ImU32>(((c >> IM_COL32_A_SHIFT) & 0xFF) * alpha);
        return (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT);
    }

    void Body(const Outline& outline, ImU32 colour, ImU32 seam = 0) {
        std::vector<ImVec2> pts;
        pts.reserve(outline.points.size());
        for (const auto& p : outline.points) {
            pts.push_back(P(p.x, p.y));
        }
        dl->AddConcavePolyFilled(pts.data(), static_cast<int>(pts.size()), colour);
        dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), seam ? seam : edge,
                        ImDrawFlags_Closed, 2.0f * scale);
    }
    void Bar(float x, float y, float w, float h, float amount) {
        dl->AddRectFilled(P(x, y), P(x + w, y + h), off);
        if (amount > 0.0f) {
            dl->AddRectFilled(P(x, y), P(x + w * std::clamp(amount, 0.0f, 1.0f), y + h), on);
        }
        dl->AddRect(P(x, y), P(x + w, y + h), edge, 0.0f, 0, thin);
    }
    void Rounded(float x, float y, float w, float h, float r, bool pressed, ImU32 fill = 0,
                 ImU32 rim = 0) {
        dl->AddRectFilled(P(x, y), P(x + w, y + h), pressed ? on : (fill ? fill : off), r * scale);
        dl->AddRect(P(x, y), P(x + w, y + h), rim ? rim : edge, r * scale, 0, thin);
    }
    void Pill(float x, float y, float w, float h, bool pressed) {
        Rounded(x, y, w, h, std::min(w, h) * 0.5f, pressed);
    }
    // `fill` replaces the colour of the whole button (pressed or not); `idle` only its colour when
    // not pressed.
    void Round(float cx, float cy, float r, bool pressed, ImU32 fill = 0, ImU32 idle = 0,
               ImU32 rim = 0) {
        dl->AddCircleFilled(P(cx, cy), r * scale,
                            fill ? fill : (pressed ? on : (idle ? idle : off)), 32);
        dl->AddCircle(P(cx, cy), r * scale, rim ? rim : edge, 32, thin);
    }
    void DPad(float cx, float cy, float length, float width, ImU32 fill = 0, ImU32 rim = 0) {
        const float h = width * 0.5f;
        Rounded(cx - h, cy - h - length, width, length, 3, v.up, fill, rim);
        Rounded(cx - h, cy + h, width, length, 3, v.down, fill, rim);
        Rounded(cx - h - length, cy - h, length, width, 3, v.left, fill, rim);
        Rounded(cx + h, cy - h, length, width, 3, v.right, fill, rim);
    }
    void Face(float x, float y, bool pressed, ImU32 symbol, int kind, ImU32 idle = 0,
              ImU32 rim = 0) {
        Round(x, y, 12, pressed, 0, idle, rim);
        const ImU32 c = pressed ? Fade(255, 255, 255) : symbol;
        const float w = 2.0f * scale;
        switch (kind) {
        case 0:
            dl->AddTriangle(P(x, y - 6), P(x + 6, y + 5), P(x - 6, y + 5), c, w);
            break;
        case 1:
            dl->AddLine(P(x - 5, y - 5), P(x + 5, y + 5), c, w);
            dl->AddLine(P(x + 5, y - 5), P(x - 5, y + 5), c, w);
            break;
        case 2:
            dl->AddRect(P(x - 5, y - 5), P(x + 5, y + 5), c, 0.0f, 0, w);
            break;
        default:
            dl->AddCircle(P(x, y), 6 * scale, c, 24, w);
            break;
        }
    }
    void Letter(float x, float y, bool pressed, const char* letter, ImU32 colour, bool colours) {
        Round(x, y, 12, pressed, pressed ? (colours ? colour : on) : off);
        const float size = 13.0f * scale;
        ImFont* font = ImGui::GetFont();
        const ImVec2 extent = font->CalcTextSizeA(size, FLT_MAX, 0.0f, letter);
        const ImVec2 at = P(x, y);
        dl->AddText(font, size, {at.x - extent.x * 0.5f, at.y - extent.y * 0.5f},
                    pressed ? Fade(255, 255, 255) : (colours ? colour : Fade(200, 200, 206)),
                    letter);
    }
    void Stick(float x, float y, float sx, float sy, bool pressed) {
        dl->AddCircleFilled(P(x, y), 24 * scale, off, 32);
        dl->AddCircle(P(x, y), 24 * scale, edge, 32, thin);
        Round(x + sx * 12, y + sy * 12, 14, pressed);
    }

    void DualShock4() {
        Bar(70, 6, 70, 16, v.l2);
        Bar(260, 6, 70, 16, v.r2);
        Pill(66, 26, 78, 12, v.l1);
        Pill(256, 26, 78, 12, v.r1);
        Outline o;
        o.Arc(40, 40, 120, 70, 180, 90);
        o.Line(100, 40, 300, 40);
        o.Arc(240, 40, 120, 70, 270, 90);
        o.Bezier(360, 75, 385, 140, 395, 215, 350, 225);
        o.Bezier(350, 225, 315, 232, 300, 180, 270, 160);
        o.Line(270, 160, 130, 160);
        o.Bezier(130, 160, 100, 180, 85, 232, 50, 225);
        o.Bezier(50, 225, 5, 215, 15, 140, 40, 75);
        Body(o, Fade(58, 58, 64));
        dl->AddRectFilled(P(150, 40), P(250, 43), FadeU32(v.light));
        Rounded(140, 44, 120, 58, 8, v.touchpad);
        Pill(118, 52, 12, 22, v.share);
        Pill(270, 52, 12, 22, v.options);
        Round(200, 132, 10, v.ps);
        DPad(89, 95, 22, 18);
        Face(312, 66, v.triangle, Fade(64, 196, 160), 0);
        Face(312, 120, v.cross, Fade(120, 150, 230), 1);
        Face(285, 93, v.square, Fade(220, 130, 190), 2);
        Face(339, 93, v.circle, Fade(235, 90, 90), 3);
        Stick(140, 150, v.lx, v.ly, v.l3);
        Stick(260, 150, v.rx, v.ry, v.r3);
    }

    void DualSense() {
        // Black triggers and bumpers.
        Bar(68, 4, 74, 16, v.l2);
        Bar(258, 4, 74, 16, v.r2);
        Pill(62, 24, 82, 12, v.l1);
        Pill(256, 24, 82, 12, v.r1);
        // The white shell with its rounder grips.
        Outline shell;
        shell.Arc(38, 38, 124, 80, 180, 90);
        shell.Line(100, 38, 300, 38);
        shell.Arc(238, 38, 124, 80, 270, 90);
        shell.Bezier(362, 78, 392, 140, 400, 222, 352, 230);
        shell.Bezier(352, 230, 318, 236, 300, 200, 274, 184);
        shell.Line(274, 184, 126, 184);
        shell.Bezier(126, 184, 100, 200, 82, 236, 48, 230);
        shell.Bezier(48, 230, 0, 222, 8, 140, 38, 78);
        Body(shell, Fade(242, 242, 244), Fade(168, 168, 176));
        // The white touchpad with the light strips beside it.
        dl->AddRectFilled(P(134, 42), P(266, 106), v.touchpad ? on : Fade(248, 248, 250), 10 * scale);
        dl->AddRect(P(134, 42), P(266, 106), Fade(190, 190, 198), 10 * scale, 0, thin);
        const ImU32 light = FadeU32(v.light);
        dl->AddLine(P(128, 50), P(128, 98), light, 3.0f * scale);
        dl->AddLine(P(272, 50), P(272, 98), light, 3.0f * scale);
        // Create, Options, PS and mute (black).
        Pill(110, 48, 10, 22, v.share);
        Pill(280, 48, 10, 22, v.options);
        Round(200, 130, 10, v.ps);
        Pill(191, 146, 18, 7, false);
        // The d-pad and face buttons are clear light grey, with grey symbols.
        const ImU32 clear = Fade(218, 218, 224), clear_edge = Fade(150, 150, 160);
        DPad(86, 97, 22, 18, clear, clear_edge);
        const ImU32 symbol = Fade(118, 118, 128);
        Face(316, 70, v.triangle, symbol, 0, clear, clear_edge);
        Face(316, 124, v.cross, symbol, 1, clear, clear_edge);
        Face(290, 97, v.square, symbol, 2, clear, clear_edge);
        Face(342, 97, v.circle, symbol, 3, clear, clear_edge);
        Stick(148, 156, v.lx, v.ly, v.l3);
        Stick(252, 156, v.rx, v.ry, v.r3);
    }

    void Xbox(bool colours) {
        Bar(72, 8, 70, 16, v.l2);
        Bar(258, 8, 70, 16, v.r2);
        Pill(60, 30, 86, 13, v.l1);
        Pill(254, 30, 86, 13, v.r1);
        Outline o;
        o.Arc(40, 45, 140, 80, 180, 90);
        o.Line(110, 45, 290, 45);
        o.Arc(220, 45, 140, 80, 270, 90);
        o.Bezier(360, 85, 392, 150, 392, 225, 345, 230);
        o.Bezier(345, 230, 310, 234, 292, 196, 262, 176);
        o.Line(262, 176, 138, 176);
        o.Bezier(138, 176, 108, 196, 90, 234, 55, 230);
        o.Bezier(55, 230, 8, 225, 8, 150, 40, 85);
        Body(o, Fade(58, 58, 64));
        Round(200, 70, 14, v.ps);
        const ImU32 mark = v.ps ? Fade(255, 255, 255) : Fade(200, 200, 206);
        dl->AddLine(P(194, 64), P(206, 76), mark, 2.0f * scale);
        dl->AddLine(P(206, 64), P(194, 76), mark, 2.0f * scale);
        Rounded(160, 96, 18, 12, 5, v.share);
        Rounded(222, 96, 18, 12, 5, v.options);
        Rounded(193, 112, 14, 9, 4, false);
        Stick(116, 96, v.lx, v.ly, v.l3);
        DPad(152, 144, 20, 16);
        Stick(248, 148, v.rx, v.ry, v.r3);
        Letter(300, 120, v.cross, "A", Fade(96, 184, 72), colours);
        Letter(324, 96, v.circle, "B", Fade(225, 64, 64), colours);
        Letter(276, 96, v.square, "X", Fade(64, 132, 228), colours);
        Letter(300, 72, v.triangle, "Y", Fade(236, 196, 48), colours);
    }

    ImDrawList* dl;
    const PadView& v;
    ImVec2 base{};
    float scale = 1.0f, alpha = 1.0f, thin = 1.0f;
    ImU32 edge = 0, off = 0, on = 0;
};

void DrawPreview(ImDrawList* dl, ImVec2 origin, ImVec2 size, const PadView& v,
                 const std::string& caption) {
    Painter{dl, origin, size, v}.Paint(caption);
}

} // namespace

void DrawControllerOverlay() {
    static bool was_open = false;
    static bool focus = false;
    static int shown_player = 0;
    const bool open = ControllerOverlayOpen();
    if (open != was_open) {
        was_open = open;
        ImGui::GetIO().MouseDrawCursor = open;
        if (open) {
            // Controller and keyboard events keep reaching the bindings (no stuck buttons when it
            // closes); the window reads the controllers itself for navigation and the preview.
            ImGui::Core::AcquireGamepadInputCapture();
            focus = true;
        } else {
            ImGui::Core::ReleaseGamepadInputCapture();
        }
    }
    if (!open) {
        return;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    // Sized like the launcher's window on a 1080p screen, and in proportion above that.
    const float s = std::clamp(viewport->Size.y / 1000.0f, 1.0f, 2.4f);
    const float text_px = 17.0f * s;
    const float global = ImGui::GetFontSize() / std::max(1.0f, ImGui::GetStyle().FontSizeBase);
    ImGui::PushFont(nullptr, text_px / std::max(0.01f, global));

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {18.0f * s, 14.0f * s});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8.0f * s, 5.0f * s});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8.0f * s, 6.0f * s});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f * s);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f * s);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 4.0f * s);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 1.0f);
    constexpr int kStyleVars = 9;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, kBack);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, Rgbf(40, 40, 42, 0.99f));
    ImGui::PushStyleColor(ImGuiCol_Border, kBorder);
    ImGui::PushStyleColor(ImGuiCol_Text, kText);
    ImGui::PushStyleColor(ImGuiCol_TextDisabled, kDimText);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, kField);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, kButton);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, kButtonHover);
    ImGui::PushStyleColor(ImGuiCol_Button, kButton);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kButtonHover);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, kBorder);
    ImGui::PushStyleColor(ImGuiCol_Header, Rgbf(178, 24, 36, 0.55f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, Rgbf(206, 40, 52, 0.65f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, kAccent);
    ImGui::PushStyleColor(ImGuiCol_Separator, kBorder);
    ImGui::PushStyleColor(ImGuiCol_NavCursor, kAccentHover);
    constexpr int kStyleColors = 16;

    const float width = std::min(900.0f * s, viewport->Size.x * 0.94f);
    ImGui::SetNextWindowPos({viewport->Pos.x + viewport->Size.x * 0.5f,
                             viewport->Pos.y + viewport->Size.y * 0.5f},
                            ImGuiCond_Always, {0.5f, 0.5f});
    ImGui::SetNextWindowSize({width, 0.0f}, ImGuiCond_Always);
    if (focus) {
        ImGui::SetNextWindowFocus();
        focus = false;
    }
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoScrollbar;
    if (ImGui::Begin("Controllers##ryGT", nullptr, flags)) {
        // Heading in the launcher's style (red capitals).
        ImGui::PushFont(nullptr, text_px * 1.15f / std::max(0.01f, global));
        ImGui::TextColored(kHeading, "CONTROLLERS");
        ImGui::PopFont();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(
            "Pick which controller plays as each player (Split Screen is for 2 players). It "
            "applies at once, and the game gets no input while this window is open. \"Next "
            "connected\" takes the controllers nobody picked. Press buttons to see them in the "
            "preview; pick a player to preview their controller.");
        ImGui::PopTextWrapPos();
        ImGui::Spacing();

        const auto controllers = ListControllers();
        const auto choices = ControllerPortChoices();
        auto* pads = Common::Singleton<Input::GameControllers>::Instance();
        if (ImGui::BeginTable("##layout", 2, ImGuiTableFlags_None)) {
            ImGui::TableSetupColumn("players", ImGuiTableColumnFlags_WidthFixed, 400.0f * s);
            ImGui::TableSetupColumn("preview", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            for (int player = 0; player < 2; ++player) {
                ImGui::PushID(player);
                ImGui::PushFont(nullptr, text_px * 1.05f / std::max(0.01f, global));
                ImGui::TextUnformatted(player == 0 ? "Player 1" : "Player 2");
                ImGui::PopFont();
                const std::string& choice = choices[player];
                std::string current = "Next connected (automatic)";
                if (!choice.empty()) {
                    current = "Not connected: " + choice;
                    for (const auto& c : controllers) {
                        if (c.key == choice) {
                            current = c.name;
                        }
                    }
                }
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::BeginCombo("##port", current.c_str())) {
                    shown_player = player;
                    if (ImGui::Selectable("Next connected (automatic)", choice.empty())) {
                        SetControllerPortChoice(player, "");
                    }
                    for (const auto& c : controllers) {
                        std::string label = c.name;
                        if (c.port >= 0 && c.port != player) {
                            label += c.port == 0 ? "   (player 1 now)" : "   (player 2 now)";
                        }
                        if (ImGui::Selectable(label.c_str(), c.key == choice)) {
                            SetControllerPortChoice(player, c.key);
                        }
                    }
                    ImGui::EndCombo();
                }
                if (ImGui::IsItemHovered() || ImGui::IsItemFocused()) {
                    shown_player = player;
                }
                std::string plays = player == 0 ? "Plays with: the keyboard (no controller)"
                                                : "No controller: this player stays out";
                for (const auto& c : controllers) {
                    if (c.port == player) {
                        plays = "Plays with: " + c.name;
                    }
                }
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextDisabled("%s", plays.c_str());
                ImGui::PopTextWrapPos();
                ImGui::Dummy({0.0f, 10.0f * s});
                ImGui::PopID();
            }
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextDisabled("The keyboard plays as player %d (change it in RyGT.exe, Button "
                                "bindings).",
                                KeyboardPlayerSetting());
            if (controllers.size() > 2) {
                ImGui::TextDisabled("%d controllers are connected; 2 can play.",
                                    static_cast<int>(controllers.size()));
            }
            ImGui::PopTextWrapPos();

            ImGui::TableSetColumnIndex(1);
            Input::GameController* pad = (*pads)[shown_player];
            std::string name = "no controller";
            for (const auto& c : controllers) {
                if (c.port == shown_player) {
                    name = c.name;
                }
            }
            const ImVec2 at = ImGui::GetCursorScreenPos();
            const float avail = ImGui::GetContentRegionAvail().x;
            const ImVec2 size{avail, std::min(avail * 0.65f, 300.0f * s)};
            DrawPreview(ImGui::GetWindowDrawList(), {at.x + 12.0f * s, at.y},
                        {size.x - 12.0f * s, size.y}, ReadPad(pad),
                        std::string(shown_player == 0 ? "Player 1: " : "Player 2: ") + name);
            ImGui::Dummy(size);
            ImGui::EndTable();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kAccentHover);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, kAccent);
        if (ImGui::Button("Done", {130.0f * s, 0.0f})) {
            ToggleControllerOverlay();
        }
        ImGui::PopStyleColor(3);
        ImGui::SameLine();
        if (ImGui::Button("All automatic", {170.0f * s, 0.0f})) {
            SetControllerPortChoice(0, "");
            SetControllerPortChoice(1, "");
        }
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("F3, or hold Share + Options (View + Menu), opens and closes this.");
    }
    ImGui::End();
    ImGui::PopStyleColor(kStyleColors);
    ImGui::PopStyleVar(kStyleVars);
    ImGui::PopFont();
}

} // namespace Ryty
