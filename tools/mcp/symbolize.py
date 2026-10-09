"""Resolve host frames ("shadGT.exe+0x1234") in shadGT logs and reports to function:file:line.

    python tools/mcp/symbolize.py <log or report> [--exe Build/x64-Clang-Release/shadGT.exe]

Crash, assertion and hang reports log host code as module+offset. With the PDB next to the
executable (Release builds write one), llvm-symbolizer resolves the shadGT.exe frames; frames
in other modules (drivers, Vulkan layers) are left as they are. Prints the file with each
resolved frame followed by [function file:line].
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

FRAME_RE = re.compile(r"\b(shadGT\.exe)\+(0x[0-9a-fA-F]+)")


def find_symbolizer() -> str | None:
    found = shutil.which("llvm-symbolizer")
    if found:
        return found
    default = Path("C:/Program Files/LLVM/bin/llvm-symbolizer.exe")
    return str(default) if default.exists() else None


def resolve(exe: Path, offsets: list[int]) -> dict[int, str]:
    symbolizer = find_symbolizer()
    if not symbolizer or not offsets:
        return {}
    # --relative-address: the inputs are offsets from the image base (module+offset).
    addresses = "\n".join(hex(o) for o in offsets) + "\n"
    out = subprocess.run([symbolizer, f"--obj={exe}", "--functions=linkage", "--demangle",
                          "--relative-address"],
                         input=addresses, capture_output=True, text=True, timeout=120).stdout
    results: dict[int, str] = {}
    blocks = out.strip().split("\n\n")
    for offset, block in zip(offsets, blocks):
        lines = block.strip().splitlines()
        # Inlined frames come first (innermost); prefer the first one in shadGT's own source.
        pairs = [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]
        pairs = [p for p in pairs if p[0] != "??"]
        if not pairs:
            continue
        own = [p for p in pairs if re.search(r"[\\/]src[\\/]", p[1])]
        function, location = (own or pairs)[0]
        location = re.sub(r"^.*[\\/](src[\\/])", r"\1", location)
        results[offset] = f"{function} {location}"
    return results


def symbolize_text(text: str, exe: Path) -> str:
    offsets = sorted({int(m.group(2), 16) for m in FRAME_RE.finditer(text)})
    names = resolve(exe, offsets)
    if not names:
        return text
    return FRAME_RE.sub(lambda m: f"{m.group(0)} [{names.get(int(m.group(2), 16), '?')}]",
                        text)


def default_exe() -> Path:
    return Path(__file__).resolve().parents[2] / "Build/x64-Clang-Release/shadGT.exe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--exe", type=Path, default=default_exe())
    args = parser.parse_args()
    if not args.exe.with_suffix(".pdb").exists():
        print(f"warning: no PDB next to {args.exe}; frames stay unresolved", file=sys.stderr)
    text = args.file.read_text(encoding="utf-8", errors="replace")
    lines = [l for l in text.splitlines() if FRAME_RE.search(l)]
    sys.stdout.write(symbolize_text("\n".join(lines), args.exe) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
