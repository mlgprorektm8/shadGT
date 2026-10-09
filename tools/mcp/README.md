# shadGT MCP server

Lets Claude Code launch Gran Turismo Sport under shadGT, look at it, press buttons and read
the logs, so a test can run without someone at the keyboard. It is a stdio MCP server
(`shadgt_mcp.py`, official `mcp` Python SDK, 1.x or 2.x) registered in the repo's `.mcp.json`.

Requirements: Python 3.10+, `pip install mcp pillow`, a built `shadGT.exe` and the GT Sport
profile in `Build/gt-sport-fixed` (the same prerequisites as
`scripts/Run-GTSportPerformance.ps1`).

## How it runs the game

`launch` starts `shadGT.exe "<eboot>" --show-fps` from the profile folder exactly like
`Run-GTSportPerformance.ps1`: same diagnostics-off config overrides (restored when the game
exits), `VK_LOADER_LAYERS_DISABLE=~implicit~`, `SHADGT_DISABLE_PERF`, and on exit the log is
copied to `runs/<stamp>-<commit>.log` with its `Analyze-GTSportRun.ps1` summary.

It also sets `SHADPS4_ENABLE_IPC=true` and drives the emulator over its IPC protocol
(`src/core/ipc/ipc.cpp`: commands on stdin, `;`-prefixed replies on stderr). It reads the
`#IPC_ENABLED ... #IPC_END` block, then sends `RUN` and `START`.

IPC switches off automatic patch loading from `user/patches`. The GT Sport 1.69 boot fix
does not depend on it: `MemoryPatcher::OnGameLoaded` applies the built-in patch
(`ApplyBuiltInPatches`) unconditionally for CUSA03220/CUSA02168. The server does not resend
it with `PATCH_MEMORY`. Any other patch you rely on from `user/patches` is not loaded in an
MCP run.

Paths: the build defaults to `Build/x64-Clang-Release` in this checkout (falling back to the
main checkout's), the profile to `Build/gt-sport-fixed` (same fallback). Override with
`launch(build_dir=...)` or the `SHADGT_BUILD_DIR` / `SHADGT_PROFILE_DIR` / `SHADGT_GAME_PATH`
environment variables.

## Tools

| Tool | What it does |
| --- | --- |
| `launch(build_dir?, extra_args?, env?, disable_perf?, perf_overrides=true)` | Start the game with IPC, return pid + capabilities |
| `stop(timeout_s=20)` | IPC `STOP` (clean quit), kill after the timeout; returns exit code, archived run log, perf summary |
| `pause()` / `resume()` | IPC `PAUSE` / `RESUME` (guest threads) |
| `status()` | Running/pid/uptime; with automation also frames, fps, paused, serial, version, title |
| `read_log(pattern?, tail=100, which="game")` | Regex-filtered tail of `user/log/shad_log.txt` (`which="launcher"` → `shadgt.log`) |
| `wait_for_log(pattern, timeout_s=60)` | Wait for a new log line matching a regex |
| `list_dumps(limit=10)` | Recent `runs/`, `crash-dumps/`, screenshots, RenderDoc captures, and DIAG-033 stall lines |
| `screenshot(max_width=1280)` | Image of the current frame (full-size PNG kept in `user/screenshots/mcp/`) |
| `press(button, hold_ms=100, hold_frames?)` | Press buttons on player 1, e.g. `cross`, `l1+cross`, `lstick_left` |
| `input_sequence(steps)` | `[{"press": "cross"}, {"wait_ms": 500}, {"press": "up", "hold_frames": 2}, {"screenshot": true}]` |
| `capture_frame()` | Presses the capture hotkey (F12): RenderDoc capture if loaded, else a game-only screenshot; returns the new files |

The game log runs at the `*:Warning` filter in MCP runs (as in performance runs), so
`wait_for_log` only sees warnings and errors unless you launch with `perf_overrides=false`.

## IPC automation (Phase 2) and the fallback (Phase 1)

Builds from the `mcp-server` branch advertise `ENABLE_TEST_AUTOMATION` and accept:

- `PAD <buttons> <lx> <ly> <rx> <ry> <l2> <r2> <hold_frames>` (one value per line, as
  every IPC parameter): injects player 1's pad state for `hold_frames` presented frames
  (`0` clears it). It's merged with real input in `Input::GameController`: buttons are OR'd,
  a stick away from centre (128) or a higher trigger value overrides. It needs no window focus.
  Replies `;PAD_OK <until_frame>`.
- `SCREENSHOT <path>`: the presenter saves the next game frame (before FSR/post-processing,
  no overlays) as PNG at `path`. Replies `;SCREENSHOT_QUEUED <path>`.
- `STATUS`: replies `;STATUS frames=<n> fps=<x> paused=<0|1> serial=<id> app_ver=<v> title=<…>`
  (fps is measured over 500 ms).

With an older build (no `ENABLE_TEST_AUTOMATION`), the server falls back. `screenshot`
captures the shadGT window (PrintWindow, then a screen grab if that comes back black).
`press` sends keyboard scancodes with SendInput, mapped through the profile's
`user/input_config` (`global.ini` + `default.ini`; non-keypad keys preferred, so Cross is `n`).
The keyboard fallback and `capture_frame` bring the shadGT window to the front.

`press` without `hold_frames` holds for `hold_ms` of wall time, then clears. Use `hold_frames`
when the game must see the press for an exact number of frames.

## Smoke test without the Claude Code CLI

`python tools/mcp/smoke_test.py` drives the server over MCP stdio, like Claude Code would. It
launches the game, waits 45 s, takes a screenshot, presses Cross, waits 3 s, takes another
screenshot, calls `status`, then stops the game. The results go to
`<profile>/runs/mcp-smoke-<stamp>.log` and the screenshots go next to it as `-1.png` / `-2.png`.
`--fake` runs it against the stub instead of the game.

## Tests

`python tools/mcp/tests/test_shadgt_mcp.py` runs the server over real MCP stdio against
`tests/fake_shadgt.py`, a stub that speaks the IPC protocol. The game isn't launched.
