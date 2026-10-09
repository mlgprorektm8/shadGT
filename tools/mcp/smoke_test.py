"""First smoke test of the shadGT MCP server, without the Claude Code CLI.

Starts tools/mcp/shadgt_mcp.py over MCP stdio (as Claude Code would) and runs:
launch, wait 45 s, screenshot, press cross, wait 3 s, screenshot, status, stop.
Every tool result is printed and written to <profile>/runs/mcp-smoke-<stamp>.log; the two
screenshots are saved next to it as mcp-smoke-<stamp>-1.png / -2.png.

    python tools/mcp/smoke_test.py            # the real game (default build and profile)
    python tools/mcp/smoke_test.py --fake     # against tests/fake_shadgt.py, no game
"""

import argparse
import asyncio
import base64
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shadgt_mcp  # noqa: E402


def attr(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


async def run(profile: Path, env: dict[str, str], wait_s: float, log_path: Path,
              shot_base: Path) -> bool:
    log = open(log_path, "w", encoding="utf-8")

    def out(text: str) -> None:
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    params = StdioServerParameters(command=sys.executable, args=[str(HERE / "shadgt_mcp.py")],
                                   env=env)
    ok = True
    out(f"shadGT MCP smoke test {dt.datetime.now().isoformat(timespec='seconds')}")
    out(f"profile: {profile}")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(name: str, args: dict, label: str | None = None):
                nonlocal ok
                out(f"\n=== {label or name} {json.dumps(args) if args else ''}")
                try:
                    result = await asyncio.wait_for(session.call_tool(name, args), 300)
                except Exception as e:
                    ok = False
                    out(f"FAILED: {type(e).__name__}: {e}")
                    return None
                if attr(result, "is_error", "isError"):
                    ok = False
                    out("ERROR: " + " ".join(getattr(c, "text", "") for c in result.content))
                    return result
                structured = attr(result, "structured_content", "structuredContent")
                if structured is not None:
                    out(json.dumps(structured.get("result", structured), indent=2))
                else:
                    for content in result.content:
                        if content.type == "image":
                            out(f"[image {content.mimeType if hasattr(content, 'mimeType') else ''}"
                                f" {len(base64.b64decode(content.data))} bytes]")
                        else:
                            out(getattr(content, "text", str(content)))
                return result

            async def shot(n: int):
                path = shot_base.with_name(f"{shot_base.name}-{n}.png")
                await call("screenshot", {"save_path": str(path), "max_width": 640},
                           f"screenshot {n}")
                if path.exists():
                    from PIL import Image
                    size = Image.open(path).size
                    out(f"saved {path} ({size[0]}x{size[1]}, {path.stat().st_size} bytes)")
                else:
                    out(f"screenshot {n} was not saved")

            launched = await call("launch", {})
            if launched is None or attr(launched, "is_error", "isError"):
                out("\nLaunch failed; stopping here.")
                log.close()
                return False
            await call("input_sequence", {"steps": [{"wait_ms": int(wait_s * 1000)}]},
                       f"wait {wait_s:g} s")
            await shot(1)
            await call("press", {"button": "cross"})
            await call("input_sequence", {"steps": [{"wait_ms": 3000}]}, "wait 3 s")
            await shot(2)
            await call("status", {})
            await call("stop", {})
    out(f"\nRESULT: {'OK' if ok else 'FAILED (see above)'}")
    out(f"log: {log_path}")
    log.close()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="use the stub emulator, no game")
    parser.add_argument("--wait", type=float, default=None,
                        help="seconds to wait after launch (default 45, 2 with --fake)")
    args = parser.parse_args()

    env = dict(os.environ)
    tmp = None
    if args.fake:
        tmp = tempfile.TemporaryDirectory()
        profile = Path(tmp.name)
        (profile / "user/log").mkdir(parents=True)
        (profile / "user/config.json").write_text("{}", encoding="utf-8")
        env.update({
            "SHADGT_PROFILE_DIR": str(profile),
            "SHADGT_MCP_TEST_COMMAND": json.dumps([sys.executable,
                                                   str(HERE / "tests/fake_shadgt.py")]),
        })
    else:
        profile = Path(env.get("SHADGT_PROFILE_DIR") or shadgt_mcp._default_profile_dir())
    wait_s = args.wait if args.wait is not None else (2 if args.fake else 45)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = profile / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    log_path = runs / f"mcp-smoke-{stamp}.log"
    ok = asyncio.run(run(profile, env, wait_s, log_path, runs / f"mcp-smoke-{stamp}"))
    if tmp:
        tmp.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
