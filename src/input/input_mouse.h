// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "SDL3/SDL.h"
#include "common/types.h"

namespace Input {

enum MouseMode {
    Off = 0,
    Joystick,
    Gyro,
    Touchpad,
};

bool ToggleMouseModeTo(MouseMode m);
void SetMouseToJoystick(int joystick);
void SetMouseParams(float mouse_deadzone_offset, float mouse_speed, float mouse_speed_offset);
extern Uint32 mouse_wheel_hold_ms;
void SetMouseGyroRollMode(bool mode);

void EmulateJoystick(GameController* controller, u32 interval);
void EmulateGyro(GameController* controller, u32 interval);

void ApplyMouseInputBlockers();

// Polls the mouse for changes
Uint32 MousePolling(void* param, Uint32 id, Uint32 interval);

} // namespace Input
