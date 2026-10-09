"""Error inventory: every distinct problem in one or more shadGT logs, counted and classified.

Reads shad_log.txt-format lines ("[Class] <Level> (thread) file:line func: message"), groups
messages that differ only in numbers/addresses/hashes, and classifies each group:

  crash        unhandled exceptions, access violations, asserts, device loss, fatal exits
  gpu          Vulkan validation messages and other GPU-side errors
  unimplemented  guest calls into functions shadGT does not implement ("Stub: X called",
               "Not Resolved", UNIMPLEMENTED asserts)
  stub         HLE functions that are stubbed or dummies ("(STUBBED)", "(DUMMY)")
  error        any other <Error>/<Critical> line
  warning      any other <Warning> line (only listed with --warnings)

Writes <out>.md (ranked, readable) and <out>.json (machine readable).

    python tools/mcp/inventory.py <log> [<log> ...] [--out PATH] [--warnings]
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

LINE_RE = re.compile(
    r"^\[(?P<cls>[^\]]+)\] <(?P<level>\w+)> \((?P<thread>[^)]*)\) (?P<loc>\S+?:\d+) "
    r"(?P<func>[^:]+): (?P<msg>.*)$")

# Replaced, in order, to turn a message into a grouping key.
NORMALIZERS = [
    (re.compile(r"(?<![\w])/(app0|temp0|download0|data|sys|hostapp|av_contents|savedata\d*)"
                r"/[^\s,;)]*"), "<guest-path>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<hex>"),
    (re.compile(r"(?<![A-Za-z_])-?\d+(\.\d+)?"), "<n>"),
    (re.compile(r"[A-Za-z]:[\\/][^\s,;)]+"), "<path>"),
    (re.compile(r"\s+"), " "),
]

CATEGORY_ORDER = ["crash", "gpu", "unimplemented", "stub", "error", "warning"]

CRASH_PATTERNS = re.compile(
    r"Unhandled Exception|access violation|Assertion Failed|Unreachable code|"
    r"Unimplemented code|DeviceLost|device lost|ErrorDeviceLost|Fatal|"
    r"DIAG-034|Exception code")
GPU_PATTERNS = re.compile(r"Validation|VUID-|vkvalidation|SYNC-HAZARD|Vulkan error",
                          re.IGNORECASE)
UNIMPLEMENTED_PATTERNS = re.compile(r"^Stub: |Not Resolved|Unimplemented code",
                                    re.IGNORECASE)
STUB_PATTERNS = re.compile(r"\((STUBBED|DUMMY)\)")


def normalize(text: str) -> str:
    for regex, repl in NORMALIZERS:
        text = regex.sub(repl, text)
    return text.strip()


@dataclass
class Group:
    key: str
    category: str
    level: str
    cls: str
    loc: str
    func: str
    count: int = 0
    first_log: str = ""
    first_line: int = 0
    example: str = ""
    examples: list[str] = field(default_factory=list)
    threads: set[str] = field(default_factory=set)

    def to_json(self) -> dict:
        return {
            "category": self.category, "level": self.level, "class": self.cls,
            "location": self.loc, "function": self.func, "count": self.count,
            "first": f"{self.first_log}:{self.first_line}", "example": self.example,
            "examples": self.examples,
            "threads": sorted(self.threads)[:8],
        }


def classify(level: str, cls: str, func: str, msg: str) -> str | None:
    if msg.startswith("Linker: Stub resolved"):
        return "warning"  # resolved at load; "Stub: X called" is logged if the game calls it
    full = f"{func}: {msg}"
    if level == "Critical" or CRASH_PATTERNS.search(full):
        if GPU_PATTERNS.search(full) and level != "Critical":
            return "gpu"
        return "crash"
    if GPU_PATTERNS.search(full) or cls.startswith("Render.Vulkan") and "Validation" in msg:
        return "gpu"
    if UNIMPLEMENTED_PATTERNS.search(msg):
        return "unimplemented"
    if STUB_PATTERNS.search(msg):
        return "stub"
    if level == "Error":
        return "error"
    if level == "Warning":
        return "warning"
    return None


def scan(paths: list[Path]) -> dict[str, Group]:
    groups: dict[str, Group] = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for number, line in enumerate(f, 1):
                m = LINE_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                level, cls, func, msg = m["level"], m["cls"], m["func"], m["msg"]
                category = classify(level, cls, func, msg)
                if category is None:
                    continue
                thread = re.sub(r"@@.*$", "", m["thread"])
                key = f"{category}|{m['loc']}|{normalize(msg)}"
                group = groups.get(key)
                if group is None:
                    group = groups[key] = Group(key, category, level, cls, m["loc"], func.strip(),
                                                first_log=path.name, first_line=number,
                                                example=msg[:400])
                group.count += 1
                if len(group.examples) < 6 and msg[:200] not in group.examples:
                    group.examples.append(msg[:200])
                group.threads.add(thread)
    return groups


def ranked(groups: dict[str, Group], warnings: bool) -> list[Group]:
    items = [g for g in groups.values() if warnings or g.category != "warning"]
    return sorted(items, key=lambda g: (CATEGORY_ORDER.index(g.category), -g.count))


def write(groups: dict[str, Group], out: Path, logs: list[Path], warnings: bool) -> dict:
    items = ranked(groups, warnings)
    summary = {c: sum(1 for g in items if g.category == c) for c in CATEGORY_ORDER}
    data = {"logs": [str(p) for p in logs], "summary": summary,
            "groups": [g.to_json() for g in items]}
    out.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    lines = [f"# Error inventory", "", "Logs: " + ", ".join(p.name for p in logs), "",
             "| Category | Distinct |", "| --- | --- |"]
    lines += [f"| {c} | {n} |" for c, n in summary.items() if n or c != "warning"]
    for category in CATEGORY_ORDER:
        members = [g for g in items if g.category == category]
        if not members:
            continue
        lines += ["", f"## {category} ({len(members)})", "",
                  "| # | Count | Where | Message (first occurrence) |", "| --- | --- | --- | --- |"]
        for i, g in enumerate(members, 1):
            message = g.example
            if len(g.examples) > 1:
                others = "; ".join(e[:90] for e in g.examples[1:4])
                message += f" (+{len(g.examples) - 1} variants: {others})"
            message = message.replace("|", "\\|")
            lines.append(f"| {i} | {g.count} | `{g.loc}` {g.func} | {message} "
                         f"({g.first_log}:{g.first_line}) |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def build(logs: list[Path], out: Path, warnings: bool = False) -> dict:
    return write(scan(logs), out, logs, warnings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="output path without extension (default: next to the first log)")
    parser.add_argument("--warnings", action="store_true", help="also list plain warnings")
    args = parser.parse_args()
    out = args.out or args.logs[0].with_name(args.logs[0].stem + ".inventory")
    data = build(args.logs, out, args.warnings)
    print(json.dumps(data["summary"]))
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
