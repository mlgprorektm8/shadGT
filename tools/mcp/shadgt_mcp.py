"""shadGT MCP server: lets Claude Code launch and drive Gran Turismo Sport under shadGT.

Runs over stdio (official `mcp` SDK, FastMCP). The emulator is started with the same
executable, arguments, environment and config overrides as scripts/Run-GTSportPerformance.ps1,
plus SHADPS4_ENABLE_IPC=true so it can be controlled over its stdin/stderr IPC protocol
(src/core/ipc/ipc.cpp).

When the emulator advertises ENABLE_TEST_AUTOMATION, pad input, screenshots and status go
through the IPC commands PAD / SCREENSHOT / STATUS (no window focus needed). Older builds
fall back to Windows window capture and SendInput keyboard presses mapped through the
profile's input config.
"""

from __future__ import annotations

import collections
import ctypes
import ctypes.wintypes as wt
import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:  # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import Image
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP, Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_bundle as analyze_bundle_module  # noqa: E402
import inventory  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Paths and defaults (mirroring scripts/Run-GTSportPerformance.ps1)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAME_PATH = "E:/Console Games/PS4 Games/CUSA03220/eboot.bin"


def _main_checkout() -> Path:
    """The main checkout (worktrees share the profile and builds that live under its Build/)."""
    try:
        common = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--path-format=absolute",
             "--git-common-dir"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return Path(common).parent
    except Exception:
        return REPO_ROOT


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_build_dir() -> Path:
    env = os.environ.get("SHADGT_BUILD_DIR")
    if env:
        return Path(env)
    return _first_existing(REPO_ROOT / "Build/x64-Clang-Release",
                           _main_checkout() / "Build/x64-Clang-Release")


def _default_profile_dir() -> Path:
    env = os.environ.get("SHADGT_PROFILE_DIR")
    if env:
        return Path(env)
    return _first_existing(REPO_ROOT / "Build/gt-sport-fixed",
                           _main_checkout() / "Build/gt-sport-fixed")


GAME_PATH = os.environ.get("SHADGT_GAME_PATH", DEFAULT_GAME_PATH)

# Diagnostics-off overrides, identical to Run-GTSportPerformance.ps1 (log filter at Warning).
PERF_OVERRIDES: dict[str, dict[str, Any]] = {
    "Debug": {"debug_dump": False, "shader_collect": False},
    "GPU": {"dump_shaders": False},
    "Log": {"filter": "*:Warning", "sync": False},
    "Vulkan": {
        "pipeline_cache_enabled": True,
        "renderdoc_enabled": False,
        "vkvalidation_enabled": False,
        "vkvalidation_core_enabled": False,
        "vkvalidation_sync_enabled": False,
        "vkvalidation_gpu_enabled": False,
        "vkcrash_diagnostic_enabled": False,
        "vkguest_markers": False,
        "vkhost_markers": False,
    },
}

# OrbisPadButtonDataOffset (src/core/libraries/pad/pad.h).
PAD_BUTTONS = {
    "l3": 0x2, "r3": 0x4, "options": 0x8,
    "up": 0x10, "right": 0x20, "down": 0x40, "left": 0x80,
    "l2": 0x100, "r2": 0x200, "l1": 0x400, "r1": 0x800,
    "triangle": 0x1000, "circle": 0x2000, "cross": 0x4000, "square": 0x8000,
    "touchpad": 0x100000,
}
BUTTON_ALIASES = {
    "x": "cross", "o": "circle", "start": "options", "touchpad_center": "touchpad",
    "pad_up": "up", "pad_down": "down", "pad_left": "left", "pad_right": "right",
    "dpad_up": "up", "dpad_down": "down", "dpad_left": "left", "dpad_right": "right",
}
# Stick directions accepted by press()/input_sequence: (axis index, value).
STICK_DIRECTIONS = {
    "lstick_left": (0, 0), "lstick_right": (0, 255), "lstick_up": (1, 0), "lstick_down": (1, 255),
    "rstick_left": (2, 0), "rstick_right": (2, 255), "rstick_up": (3, 0), "rstick_down": (3, 255),
}
# input_config output names for the Phase 1 keyboard fallback.
CONFIG_OUTPUT_FOR_BUTTON = {
    "cross": "cross", "circle": "circle", "square": "square", "triangle": "triangle",
    "l1": "l1", "r1": "r1", "l2": "l2", "r2": "r2", "l3": "l3", "r3": "r3",
    "options": "options", "touchpad": "touchpad_center",
    "up": "pad_up", "down": "pad_down", "left": "pad_left", "right": "pad_right",
    "lstick_left": "axis_left_x_minus", "lstick_right": "axis_left_x_plus",
    "lstick_up": "axis_left_y_minus", "lstick_down": "axis_left_y_plus",
    "rstick_left": "axis_right_x_minus", "rstick_right": "axis_right_x_plus",
    "rstick_up": "axis_right_y_minus", "rstick_down": "axis_right_y_plus",
}


def normalize_button(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = BUTTON_ALIASES.get(key, key)
    if key not in PAD_BUTTONS and key not in STICK_DIRECTIONS:
        valid = sorted(set(PAD_BUTTONS) | set(STICK_DIRECTIONS))
        raise ValueError(f"Unknown button {name!r}; valid: {', '.join(valid)}")
    return key


# ---------------------------------------------------------------------------------------------
# IPC client


class IpcError(RuntimeError):
    pass


class EmulatorProcess:
    """A shadGT process started with SHADPS4_ENABLE_IPC=true, driven over stdin/stderr.

    Lines the emulator writes to stderr that start with ';' are IPC output; the rest is kept
    as plain stderr text. Requests are serialized; each command's reply is matched by prefix.
    """

    def __init__(self, command: list[str], cwd: Path, env: dict[str, str],
                 stdout_path: Path | None = None, stderr_path: Path | None = None):
        self.command = command
        self.started_at = time.time()
        self._stdout_file = open(stdout_path, "wb") if stdout_path else subprocess.DEVNULL
        self._stderr_log = open(stderr_path, "a", encoding="utf-8") if stderr_path else None
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            command, cwd=str(cwd), env=env, stdin=subprocess.PIPE,
            stdout=self._stdout_file, stderr=subprocess.PIPE, creationflags=flags,
        )
        self.capabilities: set[str] = set()
        self.handshake_done = False
        self._ipc_lines: collections.deque[str] = collections.deque()
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=200)
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()

    @property
    def pid(self) -> int:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if self._stderr_log:
                self._stderr_log.write(line + "\n")
                self._stderr_log.flush()
            with self._cond:
                if line.startswith(";"):
                    self._ipc_lines.append(line[1:])
                else:
                    self._stderr_tail.append(line)
                self._cond.notify_all()
        with self._cond:
            self._cond.notify_all()

    def _wait_line(self, predicate, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for i, line in enumerate(self._ipc_lines):
                    if predicate(line):
                        del self._ipc_lines[i]
                        return line
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise IpcError("Timed out waiting for an IPC reply")
                if not self.alive() and not self._reader.is_alive():
                    raise IpcError(f"Emulator exited (code {self.proc.returncode})")
                self._cond.wait(min(remaining, 0.25))

    def send(self, *lines: str) -> None:
        if not self.alive():
            raise IpcError(f"Emulator is not running (exit code {self.proc.returncode})")
        payload = "".join(str(line).replace("\n", "\\n") + "\n" for line in lines)
        with self._send_lock:
            assert self.proc.stdin is not None
            self.proc.stdin.write(payload.encode("utf-8"))
            self.proc.stdin.flush()

    def request(self, lines: list[str], reply_prefix: str, timeout: float = 10.0) -> str:
        with self._request_lock:
            self.send(*lines)
            return self._wait_line(lambda l: l.startswith(reply_prefix), timeout)

    def handshake(self, timeout: float = 30.0) -> set[str]:
        """Reads the #IPC_ENABLED ... #IPC_END block, then sends RUN and START.

        The emulator exits if RUN does not arrive within 5 s of #IPC_END.
        """
        self._wait_line(lambda l: l == "#IPC_ENABLED", timeout)
        caps: set[str] = set()
        while True:
            line = self._wait_line(lambda l: True, 5.0)
            if line == "#IPC_END":
                break
            caps.add(line)
        self.capabilities = caps
        self.send("RUN")
        self.send("START")
        self.handshake_done = True
        return caps

    def has(self, capability: str) -> bool:
        return capability in self.capabilities

    def stderr_tail(self, n: int = 20) -> list[str]:
        with self._cond:
            return list(self._stderr_tail)[-n:]

    def pending_ipc_lines(self) -> list[str]:
        """Unsolicited IPC output (e.g. RESTART, UNKNOWN CMD), consumed."""
        with self._request_lock, self._cond:
            lines = list(self._ipc_lines)
            self._ipc_lines.clear()
            return lines

    # Commands ------------------------------------------------------------------------------

    def pad(self, buttons: int, axes: list[int], hold_frames: int) -> int:
        reply = self.request(["PAD", hex(buttons), *[str(a) for a in axes], str(hold_frames)],
                             "PAD_OK")
        return int(reply.split()[1])

    def screenshot(self, path: Path) -> None:
        self.request(["SCREENSHOT", str(path)], "SCREENSHOT_QUEUED")

    def status(self) -> dict[str, Any]:
        reply = self.request(["STATUS"], "STATUS ", timeout=10.0)
        return parse_status(reply)

    def close(self) -> None:
        for f in (self._stdout_file, self._stderr_log):
            try:
                if f not in (None, subprocess.DEVNULL):
                    f.close()
            except Exception:
                pass


def parse_status(reply: str) -> dict[str, Any]:
    """Parses 'STATUS frames=.. fps=.. paused=.. serial=.. app_ver=.. title=<rest>'."""
    body = reply[len("STATUS "):]
    result: dict[str, Any] = {}
    title_at = body.find("title=")
    if title_at >= 0:
        result["title"] = body[title_at + len("title="):]
        body = body[:title_at]
    for token in body.split():
        key, _, value = token.partition("=")
        if key in ("frames", "paused"):
            result[key] = int(value)
        elif key == "fps":
            result[key] = float(value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------------------------
# Windows helpers (Phase 1 fallbacks)

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
else:  # pragma: no cover - the emulator builds on Windows only
    user32 = gdi32 = None

WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def find_window(pid: int) -> int | None:
    """Largest visible top-level window owned by pid."""
    found: list[tuple[int, int]] = []

    def callback(hwnd, _):
        owner = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            rect = wt.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            found.append((rect.right * rect.bottom, hwnd))
        return True

    user32.EnumWindows(WNDENUMPROC(callback), 0)
    return max(found)[1] if found else None


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG),
                ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


def capture_window(hwnd: int):
    """Client-area capture with PrintWindow(PW_RENDERFULLCONTENT); falls back to a screen grab
    of the window rectangle when PrintWindow returns a black frame (some Vulkan swapchains)."""
    from PIL import Image as PILImage, ImageGrab

    rect = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    width, height = rect.right, rect.bottom
    if width <= 0 or height <= 0:
        raise RuntimeError("Window has no client area (minimized?)")
    hdc = user32.GetDC(hwnd)
    mem = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, width, height)
    gdi32.SelectObject(mem, bmp)
    PW_CLIENTONLY, PW_RENDERFULLCONTENT = 0x1, 0x2
    user32.PrintWindow(hwnd, mem, PW_CLIENTONLY | PW_RENDERFULLCONTENT)
    header = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32, 0, 0, 0,
                              0, 0, 0)
    buf = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(mem, bmp, 0, height, buf, ctypes.byref(header), 0)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mem)
    user32.ReleaseDC(hwnd, hdc)
    image = PILImage.frombuffer("RGB", (width, height), buf, "raw", "BGRX", 0, 1)
    if image.getbbox() is None:  # all black
        point = wt.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(point))
        image = ImageGrab.grab(bbox=(point.x, point.y, point.x + width, point.y + height),
                               all_screens=True)
    return image


# SendInput keyboard (scancodes, which SDL reads).
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x1, 0x2, 0x8


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("padding", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


# SDL key names used in input_config -> (scancode, extended).
SCANCODES: dict[str, tuple[int, bool]] = {
    **{c: (code, False) for c, code in zip("qwertyuiop", range(0x10, 0x1A))},
    **{c: (code, False) for c, code in zip("asdfghjkl", range(0x1E, 0x27))},
    **{c: (code, False) for c, code in zip("zxcvbnm", range(0x2C, 0x33))},
    **{str(d): (0x02 + (d - 1) if d else 0x0B, False) for d in range(10)},
    **{f"f{n}": (0x3B + n - 1, False) for n in range(1, 11)},
    "f11": (0x57, False), "f12": (0x58, False),
    "enter": (0x1C, False), "space": (0x39, False), "escape": (0x01, False),
    "tab": (0x0F, False), "backspace": (0x0E, False),
    "lshift": (0x2A, False), "rshift": (0x36, False), "lctrl": (0x1D, False),
    "rctrl": (0x1D, True), "lalt": (0x38, False), "ralt": (0x38, True),
    "up": (0x48, True), "down": (0x50, True), "left": (0x4B, True), "right": (0x4D, True),
    "insert": (0x52, True), "delete": (0x53, True), "home": (0x47, True), "end": (0x4F, True),
    "pgup": (0x49, True), "pgdown": (0x51, True),
    "kp0": (0x52, False), "kp1": (0x4F, False), "kp2": (0x50, False), "kp3": (0x51, False),
    "kp4": (0x4B, False), "kp5": (0x4C, False), "kp6": (0x4D, False), "kp7": (0x47, False),
    "kp8": (0x48, False), "kp9": (0x49, False), "kpplus": (0x4E, False),
    "kpminus": (0x4A, False), "kpenter": (0x1C, True),
}


def send_keys(keys: list[str], down: bool) -> None:
    inputs = (INPUT * len(keys))()
    for i, key in enumerate(keys):
        scan, extended = SCANCODES[key]
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
        if not down:
            flags |= KEYEVENTF_KEYUP
        inputs[i].type = INPUT_KEYBOARD
        inputs[i].u.ki = KEYBDINPUT(0, scan, flags, 0, None)
    sent = user32.SendInput(len(keys), inputs, ctypes.sizeof(INPUT))
    if sent != len(keys):
        raise RuntimeError(f"SendInput failed (error {ctypes.get_last_error()})")


def focus_window(hwnd: int) -> None:
    # Alt tap lets a background process take the foreground (SetForegroundWindow rule).
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    send_keys(["lalt"], True)
    send_keys(["lalt"], False)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.1)


def parse_input_config(profile_dir: Path) -> dict[str, list[list[str]]]:
    """output name -> keyboard key combos, from global.ini plus default.ini (unified config)
    or the game's own file, as Input::ParseInputConfig reads them."""
    cfg_dir = profile_dir / "user/input_config"
    files = [cfg_dir / "global.ini"]
    unified = True
    try:
        config = json.loads((profile_dir / "user/config.json").read_text(encoding="utf-8-sig"))
        unified = bool(config.get("General", {}).get("use_unified_input_config", True))
    except Exception:
        pass
    game_file = cfg_dir / "CUSA03220.ini"
    files.append(cfg_dir / "default.ini" if unified or not game_file.exists() else game_file)
    bindings: dict[str, list[list[str]]] = {}
    for file in files:
        if not file.exists():
            continue
        for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = "".join(line.split()).split("#", 1)[0]
            if "=" not in line:
                continue
            output, inputs = line.split("=", 1)
            output = output.split(":", 1)[0]
            keys = [k.split(":", 1)[0] for k in inputs.split(",")]
            if all(k in SCANCODES for k in keys):
                bindings.setdefault(output, []).append(keys)
    return bindings


def keys_for(output: str, bindings: dict[str, list[list[str]]]) -> list[str]:
    combos = bindings.get(output)
    if not combos:
        raise RuntimeError(f"No keyboard binding for {output!r} in the input config")
    # Prefer non-keypad keys: keypad scancodes depend on Num Lock.
    combos = sorted(combos, key=lambda c: any(k.startswith("kp") for k in c))
    return combos[0]


def png_complete(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if size < 64:
            return False
        with open(path, "rb") as f:
            f.seek(-12, os.SEEK_END)
            return f.read(12)[4:8] == b"IEND"
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------
# Session state


class Session:
    def __init__(self) -> None:
        self.emu: EmulatorProcess | None = None
        self.profile_dir = _default_profile_dir()
        self.build_dir = _default_build_dir()
        self.stamp = ""
        self.config_backup: Path | None = None
        self.lock = threading.RLock()
        self.watcher: threading.Thread | None = None
        self.last_exit: dict[str, Any] | None = None

    @property
    def log_path(self) -> Path:
        return self.profile_dir / "user/log/shad_log.txt"

    def require(self) -> EmulatorProcess:
        if not self.emu or not self.emu.alive():
            raise RuntimeError("The emulator is not running; call launch first.")
        return self.emu

    def automation(self) -> EmulatorProcess | None:
        emu = self.require()
        return emu if emu.has("ENABLE_TEST_AUTOMATION") else None

    # Config overrides (restored when the emulator exits) ----------------------------------

    def apply_overrides(self, overrides: dict[str, dict[str, Any]]) -> None:
        config_path = self.profile_dir / "user/config.json"
        backup = config_path.with_name("config.json.mcp-backup")
        if backup.exists():
            # A previous server died mid-run: restore its original first.
            shutil.copy2(backup, config_path)
        shutil.copy2(config_path, backup)
        self.config_backup = backup
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        self._previous = {s: {k: config.get(s, {}).get(k) for k in keys}
                          for s, keys in overrides.items()}
        for section, keys in overrides.items():
            config.setdefault(section, {}).update(keys)
        config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")

    def restore_overrides(self) -> None:
        """Puts back only the overridden keys, keeping settings the player changed in-game."""
        if not self.config_backup:
            return
        config_path = self.profile_dir / "user/config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            for section, keys in self._previous.items():
                for key, value in keys.items():
                    if value is None:
                        config.get(section, {}).pop(key, None)
                    else:
                        config.setdefault(section, {})[key] = value
            config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")
            self.config_backup.unlink(missing_ok=True)
        except Exception:
            shutil.copy2(self.config_backup, config_path)
            self.config_backup.unlink(missing_ok=True)
        self.config_backup = None

    def on_exit(self, emu: EmulatorProcess) -> None:
        emu.proc.wait()
        with self.lock:
            self.restore_overrides()
            run_log = None
            summary = None
            if self.log_path.exists() and self.stamp and not os.environ.get(
                    "SHADGT_MCP_TEST_COMMAND"):
                runs = self.profile_dir / "runs"
                runs.mkdir(exist_ok=True)
                commit = git_commit(self.build_dir)
                run_log = runs / f"{self.stamp}-{commit}.log"
                shutil.copy2(self.log_path, run_log)
                analyzer = REPO_ROOT / "scripts/Analyze-GTSportRun.ps1"
                if analyzer.exists():
                    try:
                        summary = subprocess.run(
                            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(analyzer), "-LogPath", str(run_log)],
                            capture_output=True, text=True, timeout=120,
                        ).stdout
                        Path(f"{run_log}.summary.txt").write_text(summary, encoding="utf-8")
                    except Exception as e:
                        summary = f"Analyze-GTSportRun.ps1 failed: {e}"
            inventory_md = None
            inventory_summary = None
            log_for_inventory = run_log or (self.log_path if self.log_path.exists() else None)
            if log_for_inventory:
                try:
                    out = (run_log.with_name(run_log.stem + ".inventory") if run_log else
                           self.profile_dir / "runs" / f"{self.stamp}.inventory")
                    out.parent.mkdir(exist_ok=True)
                    inventory_summary = inventory.build([log_for_inventory], out)["summary"]
                    inventory_md = f"{out}.md"
                except Exception as e:
                    inventory_summary = f"inventory failed: {e}"
            self.last_exit = {
                "exit_code": emu.proc.returncode,
                "run_log": str(run_log) if run_log else None,
                "summary": summary,
                "inventory": inventory_md,
                "inventory_summary": inventory_summary,
                "stderr_tail": emu.stderr_tail(20),
            }
            emu.close()


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


SESSION = Session()
mcp = FastMCP("shadgt")


def _is_shadgt_running() -> bool:
    if sys.platform != "win32":
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq shadGT.exe", "/NH"],
                         capture_output=True, text=True).stdout
    return "shadGT.exe" in out


# ---------------------------------------------------------------------------------------------
# Tools: process control


@mcp.tool()
def launch(build_dir: str | None = None, extra_args: list[str] | None = None,
           env: dict[str, str] | None = None, disable_perf: str = "",
           perf_overrides: bool = True, game_path: str | None = None,
           log_filter: str | None = None, validation: bool | None = None,
           diag: bool = False) -> dict:
    """Start GT Sport under shadGT with IPC control, like scripts/Run-GTSportPerformance.ps1.

    build_dir: folder containing shadGT.exe (default Build/x64-Clang-Release).
    extra_args: extra emulator arguments, appended after `<eboot> --show-fps`.
    env: extra environment variables for the emulator.
    disable_perf: PERF ids to switch off for an A/B run (SHADGT_DISABLE_PERF), e.g. "14,15".
    perf_overrides: apply the script's diagnostics-off config overrides (restored on exit).
    log_filter: log filter for this run instead of "*:Warning" (e.g. "*:Info").
    validation: enable the Vulkan validation layers (core + sync) from Build/tools/VulkanSDK;
      their messages go to the log and into the run's inventory. Much slower. Defaults to diag.
    diag: diagnostic run (SHADGT_DIAG=1): shaders keep their guest code and SPIR-V so the
      bundle tool can include them, and validation is on unless validation=false.
    Returns pid, IPC capabilities and paths. The built-in GT Sport 1.69 boot patch is applied
    by the emulator itself, so IPC disabling automatic patch loading does not affect it.
    """
    with SESSION.lock:
        if SESSION.emu and SESSION.emu.alive():
            raise RuntimeError(f"Emulator already running (pid {SESSION.emu.pid}); stop it first.")
        if _is_shadgt_running():
            raise RuntimeError("Close the current emulator before starting another run.")
        if build_dir:
            SESSION.build_dir = Path(build_dir)
        test_command = os.environ.get("SHADGT_MCP_TEST_COMMAND")
        game = game_path or GAME_PATH
        if test_command:
            command = json.loads(test_command)
        else:
            executable = SESSION.build_dir / "shadGT.exe"
            for file in (executable, SESSION.profile_dir / "user/config.json", Path(game)):
                if not file.exists():
                    raise FileNotFoundError(f"Missing launch prerequisite: {file}")
            command = [str(executable), game, "--show-fps"]
        command += list(extra_args or [])

        SESSION.stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        SESSION.last_exit = None
        if validation is None:
            validation = diag
        overrides = {section: dict(keys) for section, keys in PERF_OVERRIDES.items()}
        if log_filter:
            overrides["Log"]["filter"] = log_filter
        if validation:
            overrides["Vulkan"].update({"vkvalidation_enabled": True,
                                        "vkvalidation_core_enabled": True,
                                        "vkvalidation_sync_enabled": True})
        if perf_overrides and not test_command:
            SESSION.apply_overrides(overrides)
            if SESSION.log_path.exists():
                shutil.copy2(SESSION.log_path, SESSION.profile_dir /
                             f"log-before-performance-{SESSION.stamp}.txt")

        environment = dict(os.environ)
        for name in ("VK_LOADER_LAYERS_ENABLE", "VK_LAYER_PATH", "CDL_OUTPUT_PATH"):
            environment.pop(name, None)
        environment["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
        environment["SHADGT_DISABLE_PERF"] = disable_perf
        environment["SHADPS4_ENABLE_IPC"] = "true"
        if validation:
            sdk = _main_checkout() / "Build/tools/VulkanSDK/Bin"
            environment["VK_LAYER_PATH"] = str(sdk)
        if diag:
            environment["SHADGT_DIAG"] = "1"
        environment.update(env or {})

        try:
            emu = EmulatorProcess(
                command, SESSION.profile_dir, environment,
                stdout_path=SESSION.profile_dir / f"stdout-mcp-{SESSION.stamp}.txt",
                stderr_path=SESSION.profile_dir / f"stderr-mcp-{SESSION.stamp}.txt",
            )
            caps = emu.handshake()
        except Exception:
            SESSION.restore_overrides()
            raise
        SESSION.emu = emu
        SESSION.watcher = threading.Thread(target=SESSION.on_exit, args=(emu,), daemon=True)
        SESSION.watcher.start()
        return {
            "pid": emu.pid,
            "command": command,
            "capabilities": sorted(caps),
            "input_mode": "ipc" if "ENABLE_TEST_AUTOMATION" in caps else "keyboard (SendInput)",
            "screenshot_mode": "ipc" if "ENABLE_TEST_AUTOMATION" in caps else "window capture",
            "profile_dir": str(SESSION.profile_dir),
            "log": str(SESSION.log_path),
        }


@mcp.tool()
def stop(timeout_s: float = 20.0) -> dict:
    """Stop the emulator: IPC STOP (clean quit), then kill it after timeout_s.
    Returns the exit code, the archived run log under runs/ and its performance summary."""
    emu = SESSION.emu
    if not emu:
        raise RuntimeError("No emulator was launched by this server.")
    killed = False
    if emu.alive():
        try:
            emu.send("STOP")
        except Exception:
            pass
        try:
            emu.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            emu.proc.kill()
            emu.proc.wait()
            killed = True
    if SESSION.watcher:
        SESSION.watcher.join(timeout=180)
    return {"killed": killed, **(SESSION.last_exit or {"exit_code": emu.proc.returncode})}


@mcp.tool()
def pause() -> str:
    """Pause guest threads (IPC PAUSE)."""
    SESSION.require().send("PAUSE")
    return "paused"


@mcp.tool()
def resume() -> str:
    """Resume guest threads (IPC RESUME)."""
    SESSION.require().send("RESUME")
    return "resumed"


@mcp.tool()
def status() -> dict:
    """Process state; with IPC automation also frame count, fps (measured over 0.5 s), paused
    state and the game's serial/version/title."""
    emu = SESSION.emu
    if not emu:
        return {"running": False, "last_exit": SESSION.last_exit}
    result: dict[str, Any] = {
        "running": emu.alive(),
        "pid": emu.pid,
        "uptime_s": round(time.time() - emu.started_at, 1),
        "capabilities": sorted(emu.capabilities),
    }
    if not emu.alive():
        result["exit_code"] = emu.proc.returncode
        result["last_exit"] = SESSION.last_exit
        return result
    if emu.has("ENABLE_TEST_AUTOMATION"):
        result.update(emu.status())
    if sys.platform == "win32":
        hwnd = find_window(emu.pid)
        if hwnd:
            result["window_title"] = window_title(hwnd)
    unsolicited = emu.pending_ipc_lines()
    if unsolicited:
        result["ipc_messages"] = unsolicited
    return result


# ---------------------------------------------------------------------------------------------
# Tools: logs and dumps


def _log_file(which: str) -> Path:
    names = {"game": "shad_log.txt", "launcher": "shadgt.log"}
    return SESSION.profile_dir / "user/log" / names.get(which, which)


@mcp.tool()
def read_log(pattern: str | None = None, tail: int = 100, which: str = "game") -> dict:
    """Read the emulator log. which: "game" (user/log/shad_log.txt, the per-run log) or
    "launcher" (user/log/shadgt.log). pattern: regex filter (case-insensitive). Returns the
    last `tail` matching lines."""
    path = _log_file(which)
    if not path.exists():
        return {"path": str(path), "lines": [], "note": "log does not exist yet"}
    regex = re.compile(pattern, re.IGNORECASE) if pattern else None
    lines: collections.deque[str] = collections.deque(maxlen=max(1, tail))
    total = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if regex is None or regex.search(line):
                total += 1
                lines.append(line.rstrip("\n"))
    return {"path": str(path), "matches": total, "lines": list(lines)}


@mcp.tool()
def wait_for_log(pattern: str, timeout_s: float = 60.0, which: str = "game",
                 from_start: bool = False) -> dict:
    """Wait until a line matching regex `pattern` appears in the log. Only lines written after
    this call are considered unless from_start is true. Returns the line, or found=false on
    timeout (or when the emulator exits)."""
    path = _log_file(which)
    regex = re.compile(pattern, re.IGNORECASE)
    offset = 0 if from_start or not path.exists() else path.stat().st_size
    deadline = time.monotonic() + timeout_s
    pending = ""
    while True:
        if path.exists():
            size = path.stat().st_size
            if size < offset:  # log recreated by a new run
                offset = 0
            if size > offset:
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read(size - offset)
                offset = size
                pending += chunk.decode("utf-8", errors="replace")
                *complete, pending = pending.split("\n")
                for line in complete:
                    if regex.search(line):
                        return {"found": True, "line": line.rstrip("\r"),
                                "waited_s": round(timeout_s - (deadline - time.monotonic()), 1)}
        emu = SESSION.emu
        if time.monotonic() >= deadline:
            return {"found": False, "reason": "timeout"}
        if emu and not emu.alive() and time.monotonic() - emu.started_at > 1:
            # Give the final flush one more pass.
            if getattr(wait_for_log, "_exited_once", None) == emu.pid:
                return {"found": False, "reason": f"emulator exited ({emu.proc.returncode})"}
            wait_for_log._exited_once = emu.pid  # type: ignore[attr-defined]
        time.sleep(0.25)


def _recent_files(folder: Path, limit: int, pattern: str = "*") -> list[dict]:
    if not folder.exists():
        return []
    entries = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"path": str(p), "modified": dt.datetime.fromtimestamp(p.stat().st_mtime)
             .isoformat(timespec="seconds"), "size": p.stat().st_size if p.is_file() else None}
            for p in entries[:limit]]


@mcp.tool()
def list_dumps(limit: int = 10) -> dict:
    """List recent diagnostic output: archived run logs and summaries (runs/), Vulkan
    crash-diagnostic reports (crash-dumps/), screenshots and RenderDoc captures, plus the
    stall recorder's latest DIAG-033 lines from the game log (stalls are logged, not filed)."""
    profile = SESSION.profile_dir
    stalls = read_log(r"DIAG-033 stall", tail=limit)
    return {
        "runs": _recent_files(profile / "runs", limit),
        "crash_dumps": _recent_files(profile / "crash-dumps", limit),
        "screenshots": _recent_files(profile / "user/screenshots", limit, "**/*.png"),
        "captures": _recent_files(profile / "user/captures", limit),
        "stall_lines": stalls.get("lines", []),
        "stall_count": stalls.get("matches", 0),
    }


# ---------------------------------------------------------------------------------------------
# Tools: screen and input


def _to_image(pil_image, max_width: int) -> Image:
    if max_width and pil_image.width > max_width:
        height = round(pil_image.height * max_width / pil_image.width)
        pil_image = pil_image.resize((max_width, height))
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


def _grab_png(path: Path) -> None:
    """Saves the current game frame (IPC) or the window (fallback) as a PNG at path."""
    emu = SESSION.require()
    path.parent.mkdir(parents=True, exist_ok=True)
    if emu.has("ENABLE_TEST_AUTOMATION"):
        if path.exists():
            path.unlink()
        emu.screenshot(path.resolve())
        deadline = time.monotonic() + 15
        while not png_complete(path):
            if time.monotonic() > deadline:
                raise TimeoutError("No frame was presented within 15 s (game paused or hung?)")
            if not emu.alive():
                raise RuntimeError("Emulator exited before the screenshot was written")
            time.sleep(0.05)
    else:
        hwnd = find_window(emu.pid)
        if not hwnd:
            raise RuntimeError("shadGT window not found")
        capture_window(hwnd).save(path)


@mcp.tool()
def screenshot(max_width: int = 1280, save_path: str | None = None) -> Image:
    """Screenshot of the game. With IPC automation: the presented game frame (before host
    scaling, no overlays) saved by the emulator, no focus needed. Otherwise a capture of the
    shadGT window. The full-size PNG is kept at save_path (default user/screenshots/mcp/)."""
    from PIL import Image as PILImage

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = Path(save_path) if save_path else (
        SESSION.profile_dir / "user/screenshots/mcp" / f"{stamp}.png")
    _grab_png(path)
    image = PILImage.open(path)
    image.load()
    return _to_image(image, max_width)


def _press_ipc(emu: EmulatorProcess, keys: list[str], hold_ms: int,
               hold_frames: int | None) -> dict:
    buttons = 0
    axes = [128, 128, 128, 128, 0, 0]
    for key in keys:
        if key in PAD_BUTTONS:
            buttons |= PAD_BUTTONS[key]
            if key == "l2":
                axes[4] = 255
            if key == "r2":
                axes[5] = 255
        else:
            index, value = STICK_DIRECTIONS[key]
            axes[index] = value
    if hold_frames:
        until = emu.pad(buttons, axes, hold_frames)
        # Wait for the release frame so sequences stay ordered.
        deadline = time.monotonic() + max(5.0, hold_frames / 10)
        while emu.status()["frames"] < until and time.monotonic() < deadline:
            time.sleep(0.05)
        return {"mode": "ipc", "buttons": hex(buttons), "until_frame": until}
    # Time-based: hold "forever", then clear after hold_ms.
    emu.pad(buttons, axes, 1_000_000)
    time.sleep(max(hold_ms, 1) / 1000)
    emu.pad(0, [128, 128, 128, 128, 0, 0], 0)
    return {"mode": "ipc", "buttons": hex(buttons), "hold_ms": hold_ms}


def _press_keyboard(emu: EmulatorProcess, keys: list[str], hold_ms: int) -> dict:
    hwnd = find_window(emu.pid)
    if not hwnd:
        raise RuntimeError("shadGT window not found")
    bindings = parse_input_config(SESSION.profile_dir)
    scan_keys: list[str] = []
    for key in keys:
        for k in keys_for(CONFIG_OUTPUT_FOR_BUTTON[key], bindings):
            if k not in scan_keys:
                scan_keys.append(k)
    if user32.GetForegroundWindow() != hwnd:
        focus_window(hwnd)
    send_keys(scan_keys, True)
    time.sleep(max(hold_ms, 16) / 1000)
    send_keys(scan_keys, False)
    return {"mode": "keyboard", "keys": scan_keys, "hold_ms": hold_ms}


def _press(buttons: str | list[str], hold_ms: int, hold_frames: int | None) -> dict:
    names = [buttons] if isinstance(buttons, str) else list(buttons)
    keys = [normalize_button(n) for part in names for n in part.split("+")]
    emu = SESSION.require()
    if emu.has("ENABLE_TEST_AUTOMATION"):
        return _press_ipc(emu, keys, hold_ms, hold_frames)
    return _press_keyboard(emu, keys, hold_ms)


@mcp.tool()
def press(button: str, hold_ms: int = 100, hold_frames: int | None = None) -> dict:
    """Press and release pad buttons on player 1. button: cross, circle, square, triangle,
    l1, r1, l2, r2, l3, r3, options, touchpad, up/down/left/right (d-pad),
    lstick_left/right/up/down, rstick_left/right/up/down; combine with '+' ("l1+cross").
    hold_ms: how long to hold. hold_frames: hold for that many presented frames instead
    (IPC only). With IPC this needs no window focus; the keyboard fallback focuses the window."""
    return _press(button, hold_ms, hold_frames)


@mcp.tool()
def input_sequence(steps: list[dict]) -> list[dict]:
    """Run a list of inputs in order. Each step is one of
    {"press": "cross", "hold_ms": 100} / {"press": "l1+cross", "hold_frames": 2},
    {"wait_ms": 500}, {"screenshot": true} (the result notes the saved path).
    Returns one result per step."""
    results: list[dict] = []
    for step in steps:
        if "press" in step:
            results.append(_press(step["press"], int(step.get("hold_ms", 100)),
                                  step.get("hold_frames")))
            gap = int(step.get("then_wait_ms", 0))
            if gap:
                time.sleep(gap / 1000)
        elif "wait_ms" in step:
            time.sleep(int(step["wait_ms"]) / 1000)
            results.append({"waited_ms": int(step["wait_ms"])})
        elif step.get("screenshot"):
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            path = SESSION.profile_dir / "user/screenshots/mcp" / f"seq-{stamp}.png"
            screenshot(save_path=str(path))
            results.append({"screenshot": str(path)})
        else:
            raise ValueError(f"Unknown step {step!r}")
    return results


@mcp.tool()
def capture_frame(timeout_s: float = 60.0) -> dict:
    """Press the capture-frame hotkey (F12 by default, from global.ini): a RenderDoc capture
    when RenderDoc is loaded, otherwise a game-only screenshot. Focuses the shadGT window.
    Returns the files that appeared in user/captures and user/screenshots."""
    emu = SESSION.require()
    profile = SESSION.profile_dir
    folders = [profile / "user/captures", profile / "user/screenshots"]

    def snapshot() -> dict[Path, float]:
        files = {}
        for folder in folders:
            if folder.exists():
                for p in folder.rglob("*"):
                    if p.is_file():
                        files[p] = p.stat().st_mtime
        return files

    before = snapshot()
    hwnd = find_window(emu.pid)
    if not hwnd:
        raise RuntimeError("shadGT window not found")
    keys = keys_for("hotkey_capture_frame", parse_input_config(profile))
    focus_window(hwnd)
    send_keys(keys, True)
    time.sleep(0.1)
    send_keys(keys, False)
    deadline = time.monotonic() + timeout_s
    new: list[str] = []
    while time.monotonic() < deadline:
        time.sleep(1.0)
        after = snapshot()
        new = sorted(str(p) for p, m in after.items() if before.get(p) != m)
        if new:
            time.sleep(2.0)  # let a RenderDoc capture finish writing
            after = snapshot()
            new = sorted(str(p) for p, m in after.items() if before.get(p) != m)
            break
    return {"keys": keys, "new_files": new,
            "note": None if new else "nothing new appeared before the timeout"}


# ---------------------------------------------------------------------------------------------
# Tools: error inventory


@mcp.tool()
def error_inventory(logs: list[str] | None = None, warnings: bool = False) -> dict:
    """Every distinct crash, GPU/validation error, unimplemented function hit, stub, error
    (and with warnings=true, warning) in the given logs (default: the latest run's log), with
    counts and first occurrence. Writes <log>.inventory.md/.json; returns the summary and
    the top entries of each category."""
    if logs:
        paths = [Path(l) for l in logs]
    else:
        last = (SESSION.last_exit or {}).get("run_log")
        if last:
            paths = [Path(last)]
        else:
            candidates = sorted((SESSION.profile_dir / "runs").glob("2*.log"),
                                key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError("No run logs found")
            paths = [candidates[-1]]
    out = paths[0].with_name(paths[0].stem + ".inventory")
    data = inventory.build(paths, out, warnings)
    top: dict[str, list] = {}
    for group in data["groups"]:
        bucket = top.setdefault(group["category"], [])
        if len(bucket) < 15:
            bucket.append(f"{group['count']}x {group['location']} {group['function']}: "
                          f"{group['example'][:160]}")
    return {"report": f"{out}.md", "summary": data["summary"], "top": top}


# ---------------------------------------------------------------------------------------------
# Tools: diagnostic bundles


BUNDLE_WRITTEN = re.compile(r"DIAG-BUNDLE: written to (.+?) \(draws")


def _bundles_dir() -> Path:
    return SESSION.profile_dir / "user/log/bundles"


@mcp.tool()
def bundle(reason: str = "manual", trigger: str = "now", frames: int = 2,
           timeout_s: float = 300.0, focus: list[str] | None = None, wait: bool = True) -> dict:
    """Capture a one-pass diagnostic bundle and analyze it.

    trigger: "now"; "target=WxH" (the first draw rendering to a WxH color target);
      "shader=0xHASH" (the first draw/dispatch using that shader); or "log:<regex>" (when a new
      log line matches). frames: frames recorded after the trigger (2 = the second frame's
      draws have every image they touch both before and after them).
    focus: images to trace in the report (WxH, uid=N or 0xADDRESS).
    The emulator writes draws, image snapshots, shaders (with SPIR-V in diag runs), buffers and
    PM4 into user/log/bundles/<stamp>-<reason>/; the log is copied in, then the analyzer
    writes report/report.md. wait=false only arms it (analyze later with analyze_bundle).
    Launch with diag=true for shader binaries and validation messages."""
    emu = SESSION.require()
    if not emu.has("ENABLE_DIAG_BUNDLE"):
        raise RuntimeError("This build has no diagnostic bundles (ENABLE_DIAG_BUNDLE).")
    log_offset = SESSION.log_path.stat().st_size if SESSION.log_path.exists() else 0
    if trigger.startswith("log:"):
        found = wait_for_log(trigger[4:], timeout_s=timeout_s)
        if not found.get("found"):
            return {"captured": False, "reason": f"log pattern not seen: {found.get('reason')}"}
        trigger = "now"
    reply = emu.request(["DIAG_BUNDLE", reason, str(frames), trigger], "DIAG_BUNDLE_")
    if not reply.startswith("DIAG_BUNDLE_ARMED"):
        raise RuntimeError(reply)
    if not wait:
        return {"armed": True, "trigger": trigger}
    deadline = time.monotonic() + timeout_s
    folder = None
    while time.monotonic() < deadline and folder is None:
        if not emu.alive():
            raise RuntimeError("The emulator exited before the bundle was written")
        if SESSION.log_path.exists():
            with open(SESSION.log_path, "rb") as f:
                f.seek(log_offset)
                text = f.read().decode("utf-8", errors="replace")
            m = BUNDLE_WRITTEN.search(text)
            if m:
                folder = Path(m.group(1))
        time.sleep(0.5)
    if folder is None:
        return {"captured": False, "reason": f"no bundle within {timeout_s} s (trigger not hit?)"}
    return analyze_bundle(str(folder), focus)


@mcp.tool()
def analyze_bundle(path: str | None = None, focus: list[str] | None = None) -> dict:
    """Analyze a diagnostic bundle (default: the newest) into report/report.md: NaN/Inf
    images, reads of empty or never-written images, empty bindings, untranslated shaders,
    validation errors, stubs hit, black targets, and the producer chain of each focus image
    (WxH, uid=N or 0xADDRESS) with PNG previews. Returns the counts and the top findings."""
    if path:
        folder = Path(path)
    else:
        candidates = sorted((p for p in _bundles_dir().glob("*") if
                             (p / "manifest.json").exists()), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No bundles found")
        folder = candidates[-1]
    if SESSION.log_path.exists() and not (folder / "log.txt").exists():
        shutil.copy2(SESSION.log_path, folder / "log.txt")
    report = analyze_bundle_module.analyze(folder, focus or [])
    top = {k: v[:8] for k, v in report["findings"].items()}
    return {"bundle": str(folder), "report": str(folder / "report/report.md"),
            "counts": report["counts"], "top": top,
            "chains": [{"uid": c["uid"], "lines": c["lines"][:60]} for c in report["chains"]]}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
