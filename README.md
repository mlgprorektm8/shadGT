<!--
SPDX-FileCopyrightText: 2026 shadPS4 Emulator Project
SPDX-License-Identifier: GPL-2.0-or-later
-->

<h1 align="center">
  <br>
  <a href="https://shadPS4.net/"><img src="https://github.com/shadGT-emu/shadGT/blob/main/.github/shadGT.png" width="220"></a>
  <br>
  <b>shadGT</b>
  <br>
</h1>

<h1 align="center">
 <a href="[https://discord.gg/bFJxfftGW6](https://discord.gg/De94trHtj5)">
        <img src="https://img.shields.io/discord/1080089157554155590?color=5865F2&label=shadGT%20Discord&logo=Discord&logoColor=white" width="275">
 <a href="https://github.com/shadGT-emu/shadGT/releases/latest">
        <img src="https://img.shields.io/github/downloads/shadGT-emu/shadGT/total.svg" width="140">
 <a href="https://shadGT.net/">
        <img src="https://img.shields.io/badge/shadGT-website-8A2BE2" width="150">
 <a href="https://x.com/shadGT">
        <img src="https://img.shields.io/badge/-Join%20us-black?logo=X&logoColor=white" width="100">
 <a href="https://github.com/shadGT-emu/shadGT/stargazers">
        <img src="https://img.shields.io/github/stars/shadGT-emu/shadGT" width="120">
</h1>


# General information

**shadGT** is an early **PlayStation 4** emulator for **Windows**, written in C++.

If you encounter problems or have doubts, do not hesitate to look at the [**Quickstart**](https://github.com/shadGT-emu/shadGT/wiki/I.-Quick-start-%5BUsers%5D).\
To verify that a game works, you can look at [**shadGT Game Compatibility**](https://github.com/shadGT-compatibility/shadGT-game-compatibility).\
To discuss shadGT development, suggest ideas or to ask for help, join our [**Discord server**](https://discord.gg/bFJxfftGW6).\
To get the latest news, go to our [**X (Twitter)**](https://x.com/shadGT) or our [**website**](https://shadGT.net/).\
You can donate to the project via our [**Kofi page**](https://ko-fi.com/shadGT).

# Status

> [!IMPORTANT]
> shadGT is early in development, don't expect a flawless experience.

Currently, the emulator can successfully run games like [**Bloodborne**](https://www.youtube.com/watch?v=5sZgWyVflFM), [**Dark Souls Remastered**](https://www.youtube.com/watch?v=-3PA-Xwszts), [**Red Dead Redemption**](https://www.youtube.com/watch?v=Al7yz_5nLag), and many other games.

# Why

This project began for fun. Given our limited free time, it may take some time before shadGT can run more complex games, but we're committed to making small, regular updates.

# Building

## Docker

For building shadGT in a containerized environment using Docker and VSCode, check the instructions here:  
[**Docker Build Instructions**](https://github.com/shadGT-emu/shadGT/blob/main/documents/building-docker.md)

## Windows

Check the build instructions for [**Windows**](https://github.com/shadGT-emu/shadGT/blob/main/documents/building-windows.md).

# Usage examples

> [!IMPORTANT]
> For a user-friendly GUI, download the [**QtLauncher**](https://github.com/shadGT-emu/shadGT-qtlauncher/releases).

To get the list of all available commands and also a more detailed description of what each command does, please refer to the `--help` flag's output.

Below is a list of commonly used command patterns:
```sh
shadGT CUSA00001 # Searches for a game folder called CUSA00001 in the list of game install folders, and boots it.
shadGT --fullscreen true --config-clean CUSA00001    # the game argument is always the last one,
shadGT -g CUSA00001 --fullscreen true --config-clean # ...unless manually specified otherwise.
shadGT /path/to/game.elf # Boots a PS4 ELF file directly. Useful if you want to boot an executable that is not named eboot.bin.
shadGT CUSA00001 -- -flag1 -flag2 # Passes '-flag1' and '-flag2' to the game executable in argv.
```

# Debugging and reporting issues

For more information on how to test, debug and report issues with the emulator or games, read the [**Debugging documentation**](https://github.com/shadGT-emu/shadGT/blob/main/documents/Debugging/Debugging.md).

# Keyboard and Mouse Mappings

> [!NOTE]
> Some keyboards may also require you to hold the Fn key to use the F\* keys. Mac users should use the Command key instead of Control, and need to use Command+F11 for full screen to avoid conflicting with system key bindings.

| Button | Function |
|-------------|-------------|
F10 | FPS Counter
Ctrl+F10 | Video Debug Info
F11 | Fullscreen
F12 | Trigger RenderDoc Capture (or game-only screenshot if RenderDoc is unavailable)
Alt+F12 | Capture screenshot including HUD/dialog overlays

> [!NOTE]
> Xbox and DualShock controllers work out of the box.

| Controller button | Keyboard equivalent |
|-------------|-------------|
LEFT AXIS UP | W |
LEFT AXIS DOWN | S |
LEFT AXIS LEFT | A |
LEFT AXIS RIGHT | D |
RIGHT AXIS UP | I |
RIGHT AXIS DOWN | K |
RIGHT AXIS LEFT | J |
RIGHT AXIS RIGHT | L |
TRIANGLE | Numpad 8 or C |
CIRCLE | Numpad 6 or B |
CROSS | Numpad 2 or N |
SQUARE | Numpad 4 or V |
PAD UP | UP |
PAD DOWN | DOWN |
PAD LEFT | LEFT |
PAD RIGHT | RIGHT |
OPTIONS | RETURN |
BACK BUTTON / TOUCH PAD | SPACE |
L1 | Q |
R1 | U |
L2 | E |
R2 | O |
L3 | X |
R3 | M |

Keyboard and mouse inputs can be customized in the settings menu by clicking the Controller button, and further details and help on controls are  also found there. Custom bindings are saved per-game. Inputs support up to three keys per binding, mouse buttons, mouse movement mapped to joystick input, and more.


# Firmware files

shadGT can load some PlayStation 4 firmware files.
The following firmware modules are supported and must be placed in shadGT's `sys_modules` folder.

<div align="center">

| Modules                        | Modules                        | Modules                        | Modules                        |
|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
| libSceAt9Enc.sprx              | libSceAudiodec.sprx            | libSceAudiodecCpu.sprx         | libSceAudiodecCpuDdp.sprx      |
| libSceAudiodecCpuDtsHdLbr.sprx | libSceAudiodecCpuHevag.sprx    | libSceAudiodecCpuM4aac.sprx    | libSceAvPlayer.sprx            |
| libSceAvPlayerStreaming.sprx   | libSceBeisobmf.sprx            | libSceBemp2sys.sprx            | libSceCesCs.sprx               |
| libSceFont.sprx                | libSceFontFt.sprx              | libSceFreeTypeOl.sprx          | libSceFreeTypeOptOl.sprx       |
| libSceFreeTypeOt.sprx          | libSceJpegDec.sprx             | libSceJpegEnc.sprx             | libSceJson.sprx                |
| libSceJson2.sprx               | libSceLibcInternal.sprx        | libSceNgs2.sprx                | libScePngEnc.sprx              |
| libScePsmKitSystem.sprx        | libSceRtc.sprx                 | libSceRudp.sprx                | libSceSystemGesture.sprx       |
| libSceUlt.sprx                 | libSceWkFontConfig.sprx        | libSceXml.sprx                 | libSceDepth.sprx               |
| libScePadTracker.sprx          | libSceMoveTracker.sprx         |
</div>

> [!Caution]
> The above modules are required to run the games properly and must be dumped from your legally owned PlayStation 4 console.



# Main credits

- [**georgemoralis**](https://github.com/georgemoralis)
- [**psucien**](https://github.com/psucien)
- [**viniciuslrangel**](https://github.com/viniciuslrangel)
- [**roamic**](https://github.com/roamic)
- [**squidbus**](https://github.com/squidbus)
- [**frodo**](https://github.com/baggins183)
- [**Stephen Miller**](https://github.com/StevenMiller123)
- [**kalaposfos13**](https://github.com/kalaposfos13)

<a href="https://github.com/shadPS4-emu/shadPS4/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=shadPS4-emu/shadPS4&max=24">
</a>

# Contributing

If you want to contribute, please read the [**CONTRIBUTING.md**](https://github.com/shadGT-emu/shadGT/blob/main/CONTRIBUTING.md) file.\
Open a PR and we'll check it :)


# Special Thanks

A few noteworthy teams/projects who've helped us along the way are:

- [**Panda3DS**](https://github.com/wheremyfoodat/Panda3DS): A multiplatform 3DS emulator from our co-author wheremyfoodat. They have been incredibly helpful in understanding and solving problems that came up from natively executing the x64 code of PS4 binaries

- [**fpPS4**](https://github.com/red-prig/fpPS4): The fpPS4 team has assisted massively with understanding some of the more complex parts of the PS4 operating system and libraries, by helping with reverse engineering work and research.

- **yuzu**: Our shader compiler has been designed with yuzu's Hades compiler as a blueprint. This allowed us to focus on the challenges of emulating a modern AMD GPU while having a high-quality optimizing shader compiler implementation as a base.

- [**felix86**](https://github.com/OFFTKP/felix86): A new x86-64 → RISC-V Linux userspace emulator

- [**emudev.org**](https://emudev.org/): A network of people interested in the documentation, emulation, simulation and re-implementation of hardware near extinction . Belongs to my friend skmp and me (shadow) also a member of it

# License

- [**GPL-2.0 license**](https://github.com/shadPS4-emu/shadPS4/blob/main/LICENSE)
