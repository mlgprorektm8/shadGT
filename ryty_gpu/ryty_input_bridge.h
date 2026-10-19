#pragma once

#include <array>
#include <string>
#include <vector>

namespace Ryty {

/// Forwards shadGT's controller state (keyboard bindings and gamepads) to Ryty's libScePad.
void StartInputBridge();

/// Applies the launcher's controller port choices; call before the window opens controllers.
void InstallControllerPorts();

/// The in-game Controllers window (F3, or Share/View + Options/Menu held on a controller): whether
/// it is open (the game gets released controls meanwhile), and opening or closing it.
bool ControllerOverlayOpen();
void ToggleControllerOverlay();
/// Draws the window when it is open; called from the overlay draw every frame.
void DrawControllerOverlay();

/// A connected controller: its port key ("vvvv:pppp#n", as the launcher writes it), its name, and
/// the player port it is in (-1: none).
struct ConnectedController {
    std::string key;
    std::string name;
    int port = -1;
};
std::vector<ConnectedController> ListControllers();
/// The port key chosen for players 1 and 2 ("" = the next connected controller).
std::array<std::string, 2> ControllerPortChoices();
/// Chooses a player's controller ("" = automatic): applied at once and saved to the config.
void SetControllerPortChoice(int player, const std::string& key);
/// The player the keyboard controls (Input.keyboard_player).
int KeyboardPlayerSetting();

} // namespace Ryty
