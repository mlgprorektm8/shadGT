"""Drive the shadGT MCP tools from a shell, without an MCP client.

    python tools/mcp/shadgt_ctl.py serve                 # keep running (holds the emulator)
    python tools/mcp/shadgt_ctl.py launch
    python tools/mcp/shadgt_ctl.py press button=cross
    python tools/mcp/shadgt_ctl.py screenshot            # prints the saved PNG path
    python tools/mcp/shadgt_ctl.py checkpoint_save name=dealership "description=before buying"
    python tools/mcp/shadgt_ctl.py input_sequence 'steps=[{"press": "down"}, {"wait_ms": 500}]'
    python tools/mcp/shadgt_ctl.py stop

Arguments are key=value; values are parsed as JSON when they parse (numbers, lists, true),
otherwise kept as text. The server listens on 127.0.0.1 only (port 8765 or SHADGT_CTL_PORT).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("SHADGT_CTL_PORT", "8765"))


def serve() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import shadgt_mcp as server

    tools = {name: getattr(server, name) for name in (
        "launch", "stop", "pause", "resume", "status", "read_log", "wait_for_log", "list_dumps",
        "press", "input_sequence", "capture_frame", "checkpoint_save", "checkpoint_list",
        "checkpoint_delete", "checkpoint_edit", "checkpoint_load", "error_inventory", "tour")}
    lock = threading.Lock()

    def screenshot(save_path: str | None = None, **_):
        import datetime as dt
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        path = Path(save_path) if save_path else (
            server.SESSION.profile_dir / "user/screenshots/mcp" / f"{stamp}.png")
        server._grab_png(path)
        return {"path": str(path)}

    tools["screenshot"] = screenshot

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name, args = request["tool"], request.get("args", {})
            try:
                if name not in tools:
                    raise KeyError(f"unknown tool {name}; tools: {', '.join(sorted(tools))}")
                # One tool at a time, like a person at the controller.
                with lock:
                    result = {"ok": True, "result": tools[name](**args)}
            except Exception as e:
                result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc(limit=4)}
            body = json.dumps(result, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"shadgt_ctl serving on 127.0.0.1:{PORT}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        if server.SESSION.emu and server.SESSION.emu.alive():
            server.stop()


def parse_args(items: list[str]) -> dict:
    args = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            args[key] = json.loads(value)
        except json.JSONDecodeError:
            args[key] = value
    return args


def call(tool: str, args: dict, timeout: float = 1800) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/", data=json.dumps({"tool": tool, "args": args}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if sys.argv[1] == "serve":
        serve()
        return 0
    result = call(sys.argv[1], parse_args(sys.argv[2:]))
    print(json.dumps(result.get("result") if result["ok"] else result, indent=2, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
