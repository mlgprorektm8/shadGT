"""Stand-in for shadGT.exe that speaks the IPC protocol of src/core/ipc/ipc.cpp.

Used by test_shadgt_mcp.py so the MCP server can be tested without the game. It prints the
#IPC_ENABLED block to stderr, exits if RUN does not arrive within 5 s (like IPC::Init),
presents "frames" at 60 Hz, and answers PAD / SCREENSHOT / STATUS like the real build.
Every command it receives is appended to $FAKE_SHADGT_RECORD as a JSON line.

FAKE_SHADGT_NO_AUTOMATION=1 leaves out ENABLE_TEST_AUTOMATION (an older build).
"""

import json
import os
import sys
import threading
import time

frames = 0
paused = False
# The "screen" changes with every button press (FAKE_SHADGT_SCREEN_OFFSET shifts it, so a
# replay sees different screens than were recorded).
screen = int(os.environ.get("FAKE_SHADGT_SCREEN_OFFSET", "0"))
pending_screenshots: list[str] = []
lock = threading.Lock()
quit_event = threading.Event()
run_event = threading.Event()
record_path = os.environ.get("FAKE_SHADGT_RECORD")


def out(line: str) -> None:
    sys.stderr.write(";" + line + "\n")
    sys.stderr.flush()


def record(entry) -> None:
    if record_path:
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def presenter() -> None:
    global frames
    while not quit_event.is_set():
        time.sleep(1 / 60)
        with lock:
            if paused:
                continue
            frames += 1
            shots = list(pending_screenshots)
            pending_screenshots.clear()
        for path in shots:
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (64, 36), (40 * (screen % 6), 80, 160))
            ImageDraw.Draw(image).rectangle([4 * (screen % 12), 4, 4 * (screen % 12) + 10, 20],
                                            fill=(255, 255, 255))
            image.save(path)


def reader() -> None:
    global paused, screen
    lines = iter(sys.stdin.readline, "")

    def arg() -> str:
        return next(lines).rstrip("\n")

    for raw in lines:
        cmd = raw.rstrip("\n")
        if not cmd:
            continue
        if cmd == "RUN":
            record(["RUN"])
            run_event.set()
        elif cmd == "START":
            record(["START"])
        elif cmd == "PAD" and automation:
            values = [arg() for _ in range(8)]
            record(["PAD", *values])
            with lock:
                until = frames + int(values[7], 0)
                if int(values[0], 0) != 0:
                    screen += 1
            out(f"PAD_OK {until}")
        elif cmd == "SCREENSHOT" and automation:
            path = arg()
            record(["SCREENSHOT", path])
            with lock:
                pending_screenshots.append(path)
            out(f"SCREENSHOT_QUEUED {path}")
        elif cmd == "STATUS" and automation:
            record(["STATUS"])
            with lock:
                before = frames
            time.sleep(0.5)
            with lock:
                now = frames
            out(f"STATUS frames={now} fps={(now - before) / 0.5:.1f} paused={int(paused)} "
                f"serial=CUSA03220 app_ver=01.69 title=Gran Turismo SPORT")
        elif cmd == "PAUSE":
            record(["PAUSE"])
            with lock:
                paused = True
        elif cmd == "RESUME":
            record(["RESUME"])
            with lock:
                paused = False
        elif cmd == "STOP":
            record(["STOP"])
            quit_event.set()
            return
        else:
            out(f"UNKNOWN CMD: {cmd}")


automation = os.environ.get("FAKE_SHADGT_NO_AUTOMATION") != "1"
if os.environ.get("SHADPS4_ENABLE_IPC") != "true":
    sys.exit("fake_shadgt: SHADPS4_ENABLE_IPC is not set")
print("fake shadGT starting", file=sys.stderr, flush=True)
threading.Thread(target=reader, daemon=True).start()
out("#IPC_ENABLED")
out("ENABLE_MEMORY_PATCH")
out("ENABLE_EMU_CONTROL")
if automation:
    out("ENABLE_TEST_AUTOMATION")
out("#IPC_END")
if not run_event.wait(5):
    print("IPC: Failed to acquire run semaphore, closing process.", file=sys.stderr, flush=True)
    sys.exit(1)
threading.Thread(target=presenter, daemon=True).start()
quit_event.wait()
sys.exit(0)
