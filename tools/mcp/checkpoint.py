"""Jump to a saved GT Sport test point.

    python tools\\mcp\\checkpoint.py list
    python tools\\mcp\\checkpoint.py load dealership     # restore, launch, replay; then play on
    python tools\\mcp\\checkpoint.py delete dealership

`load` restores the checkpoint's save data, launches the game and replays the recorded inputs,
checking each screen. When it arrives, the game keeps running for you; this window waits until
the game is closed (then the run log and its error inventory are archived under runs/).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shadgt_mcp as server  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "load", "delete"):
        print(__doc__)
        return 2
    command = sys.argv[1]
    if command == "list":
        for cp in server.checkpoint_list():
            print(f"{cp['name']:24} {cp['steps']:3} steps  {cp['description']}")
        return 0
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    name = sys.argv[2]
    if command == "delete":
        print(server.checkpoint_delete(name))
        return 0
    result = server.checkpoint_load(name)
    print(json.dumps(result, indent=2))
    if not result.get("arrived"):
        server.stop()
        return 1
    print("Arrived. The game keeps running; close it when you are done.")
    server.SESSION.emu.proc.wait()
    if server.SESSION.watcher:
        server.SESSION.watcher.join(timeout=180)
    exit_info = server.SESSION.last_exit or {}
    print(f"Run log: {exit_info.get('run_log')}\nInventory: {exit_info.get('inventory')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
