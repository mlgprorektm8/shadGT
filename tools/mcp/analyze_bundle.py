"""Offline analyzer for shadGT diagnostic bundles (video_core/diag_bundle.h).

    python tools/mcp/analyze_bundle.py <bundle dir> [--focus WxH|uid=N|0xADDRESS ...]

Reads the bundle's draw history, image snapshots, shaders and log and writes report.md and
report.json into <bundle>/report/, listing every anomaly found in one pass:

  - images with NaN or Inf texels
  - draws and dispatches that read an image that was empty (all zero) or never written
  - draws and dispatches with empty buffer or image bindings
  - shaders that could not be translated, and dropped exports (log)
  - Vulkan validation messages (log, when the run had validation on)
  - stubbed or unimplemented functions the game called (log)
  - render targets that ended all black or constant

For each image named with --focus (and each image with NaN/Inf), the report walks back
through the draws that made it: every producer, every image it read, and their statistics,
down to the first pass whose inputs look sound but whose output does not. PNG previews of the
images in those chains are written to <bundle>/report/previews/.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inventory  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Texel decoding (vk::to_string names)

COMPONENT_RE = re.compile(r"([RGBAXDS])(\d+)")
NUMERIC = ("Unorm", "Snorm", "Uint", "Sint", "Sfloat", "Srgb", "Ufloat", "Uscaled", "Sscaled")


def _half_or_float(raw: np.ndarray, bits: int) -> np.ndarray:
    if bits == 16:
        return raw.view(np.float16).astype(np.float32)
    if bits == 32:
        return raw.view(np.float32)
    raise ValueError


def _ufloat(bits: np.ndarray, mantissa: int) -> np.ndarray:
    """Unsigned small floats (5-bit exponent) of B10G11R11 / E5B9G9R9 style."""
    exponent = (bits >> mantissa) & 0x1F
    frac = (bits & ((1 << mantissa) - 1)).astype(np.float32)
    value = np.where(exponent == 0, frac / (1 << mantissa) * 2.0 ** -14,
                     (1 + frac / (1 << mantissa)) * np.exp2(exponent.astype(np.float32) - 15))
    value = np.where(exponent == 31, np.where(frac == 0, np.inf, np.nan), value)
    return value.astype(np.float32)


def decode(data: bytes, width: int, height: int, fmt: str) -> tuple[np.ndarray, bool] | None:
    """Returns (H x W x C float32 array, is_float) or None when the format is unknown."""
    count = width * height
    try:
        if fmt == "B10G11R11UfloatPack32":
            v = np.frombuffer(data, np.uint32, count)
            r, g, b = _ufloat(v & 0x7FF, 6), _ufloat((v >> 11) & 0x7FF, 6), _ufloat(v >> 22, 5)
            return np.stack([r, g, b], -1).reshape(height, width, 3), True
        if fmt == "E5B9G9R9UfloatPack32":
            v = np.frombuffer(data, np.uint32, count)
            e = ((v >> 27) & 0x1F).astype(np.float32)
            scale = np.exp2(e - 15 - 9)
            rgb = [((v >> s) & 0x1FF).astype(np.float32) * scale for s in (0, 9, 18)]
            return np.stack(rgb, -1).reshape(height, width, 3), True
        if fmt in ("A2B10G10R10UnormPack32", "A2R10G10B10UnormPack32"):
            v = np.frombuffer(data, np.uint32, count)
            c = [((v >> s) & 0x3FF) / 1023.0 for s in (0, 10, 20)] + [(v >> 30) / 3.0]
            if fmt.startswith("A2R"):
                c = [c[2], c[1], c[0], c[3]]
            return np.stack(c, -1).astype(np.float32).reshape(height, width, 4), False
        if fmt == "R5G6B5UnormPack16":
            v = np.frombuffer(data, np.uint16, count)
            c = [((v >> 11) & 31) / 31.0, ((v >> 5) & 63) / 63.0, (v & 31) / 31.0]
            return np.stack(c, -1).astype(np.float32).reshape(height, width, 3), False
        if fmt in ("D24UnormS8Uint", "X8D24UnormPack32"):
            v = np.frombuffer(data, np.uint32, count)
            return ((v & 0xFFFFFF) / float(0xFFFFFF)).astype(np.float32).reshape(
                height, width, 1), False
        if fmt in ("D32SfloatS8Uint",):
            return np.frombuffer(data, np.float32, count).reshape(height, width, 1), True
        if fmt in ("D16UnormS8Uint",):
            return (np.frombuffer(data, np.uint16, count) / 65535.0).astype(np.float32).reshape(
                height, width, 1), False
        numeric = next((n for n in NUMERIC if fmt.endswith(n)), None)
        if numeric is None:
            return None
        comps = COMPONENT_RE.findall(fmt[: -len(numeric)])
        if not comps:
            return None
        bits = {int(b) for _, b in comps}
        if len(bits) != 1:
            return None
        bits = bits.pop()
        n = len(comps)
        dtype = {8: np.uint8, 16: np.uint16, 32: np.uint32}[bits]
        raw = np.frombuffer(data, dtype, count * n).reshape(height, width, n)
        if numeric in ("Sfloat", "Ufloat"):
            return _half_or_float(raw, bits), True
        if numeric in ("Unorm", "Srgb"):
            out = raw.astype(np.float32) / float((1 << bits) - 1)
        elif numeric == "Snorm":
            signed = raw.view({8: np.int8, 16: np.int16, 32: np.int32}[bits])
            out = np.maximum(signed.astype(np.float32) / float((1 << (bits - 1)) - 1), -1.0)
        elif numeric in ("Sint", "Sscaled"):
            out = raw.view({8: np.int8, 16: np.int16, 32: np.int32}[bits]).astype(np.float32)
        else:
            out = raw.astype(np.float32)
        # BGRA in memory -> RGBA
        if fmt.startswith("B8G8R8") and n >= 3:
            out = out[..., [2, 1, 0] + ([3] if n == 4 else [])]
        return out, False
    except (ValueError, KeyError):
        return None


@dataclass
class Stats:
    nan: int = 0
    inf: int = 0
    zero_fraction: float = 0.0
    black: bool = False
    constant: bool = False
    mean: list[float] = field(default_factory=list)
    minimum: list[float] = field(default_factory=list)
    maximum: list[float] = field(default_factory=list)
    alpha_zero: bool = False
    decoded: bool = True

    def summary(self) -> str:
        if not self.decoded:
            return "format not decoded"
        parts = []
        if self.nan:
            parts.append(f"{self.nan} NaN")
        if self.inf:
            parts.append(f"{self.inf} Inf")
        if self.zero_fraction >= 0.999:
            parts.append("all zero")
        elif self.black:
            parts.append("black")
        elif self.constant:
            parts.append("constant")
        means = ", ".join(f"{m:.3g}" for m in self.mean)
        parts.append(f"mean ({means}) max {max(self.maximum) if self.maximum else 0:.3g}")
        return "; ".join(parts)

    @property
    def bad(self) -> bool:
        return self.decoded and (self.nan > 0 or self.inf > 0)

    @property
    def empty(self) -> bool:
        return self.decoded and self.zero_fraction >= 0.999


def image_stats(array: np.ndarray) -> Stats:
    finite = np.isfinite(array)
    nan = int(np.isnan(array).sum())
    inf = int(np.isinf(array).sum())
    clean = np.where(finite, array, 0.0)
    texel_zero = np.all(clean == 0, axis=-1)
    channels = clean.shape[-1]
    color = clean[..., : min(3, channels)]
    flat = clean.reshape(-1, channels)
    return Stats(
        nan=nan, inf=inf, zero_fraction=float(texel_zero.mean()),
        black=bool(np.abs(color).max() < 1.0 / 255.0),
        constant=bool(np.all(flat == flat[0])) if flat.size else True,
        mean=[float(x) for x in clean.reshape(-1, channels).mean(0)],
        minimum=[float(x) for x in flat.min(0)] if flat.size else [],
        maximum=[float(x) for x in flat.max(0)] if flat.size else [],
        alpha_zero=channels == 4 and bool(np.all(clean[..., 3] == 0)),
    )


def save_preview(array: np.ndarray, is_float: bool, path: Path) -> None:
    from PIL import Image

    data = np.where(np.isfinite(array), array, 0.0)
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, -1)
    data = data[..., :3]
    if data.shape[-1] == 2:
        data = np.concatenate([data, np.zeros_like(data[..., :1])], -1)
    if is_float:
        data = data / (1.0 + data)  # Reinhard, so HDR targets stay visible
    peak = data.max()
    if 0 < peak < 0.05:
        data = data / peak  # very dark: stretch, the report states it
    image = Image.fromarray((np.clip(data, 0, 1) * 255).astype(np.uint8))
    nan_mask = ~np.all(np.isfinite(array), axis=-1)
    if nan_mask.any():
        pixels = np.array(image)
        pixels[nan_mask] = [255, 0, 255]  # NaN/Inf in magenta
        image = Image.fromarray(pixels)
    image.thumbnail((640, 640))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


# ---------------------------------------------------------------------------------------------
# Bundle model


@dataclass
class Snapshot:
    snapshot: int
    after_seq: int
    uid: int
    entry: dict
    stats: Stats | None = None
    array: np.ndarray | None = None
    is_float: bool = False


class Bundle:
    def __init__(self, folder: Path):
        self.dir = folder
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        self.records = []
        with open(folder / "draws.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        self.by_seq = {r["seq"]: r for r in self.records}
        # Per image, the sequence numbers of the draws and dispatches that wrote it.
        self.writer_seqs: dict[int, list[int]] = defaultdict(list)
        for record in self.records:
            for ref in record["images"]:
                if ref["role"] != "read":
                    self.writer_seqs[ref["uid"]].append(record["seq"])
        self.snapshots: dict[int, list[Snapshot]] = defaultdict(list)
        self.image_info: dict[int, dict] = {}
        images = folder / "images.jsonl"
        if images.exists():
            with open(images, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    snap = Snapshot(entry["snapshot"], entry["after_seq"], entry["uid"], entry)
                    self.snapshots[entry["uid"]].append(snap)
                    if "width" in entry:
                        self.image_info[entry["uid"]] = entry
        for record in self.records:
            for ref in record["images"]:
                self.image_info.setdefault(ref["uid"], {
                    "uid": ref["uid"], "address": ref["address"], "width": ref["width"],
                    "height": ref["height"], "format": ref["format"]})
        self.start = self.manifest.get("start_seq", 0)
        self.end = self.manifest.get("end_seq", 0)
        self.log = folder / "log.txt"

    def window(self):
        return [r for r in self.records if self.start <= r["seq"] <= self.end]

    def load(self, snap: Snapshot) -> Stats | None:
        if snap.stats is not None or "file" not in snap.entry:
            return snap.stats
        path = self.dir / snap.entry["file"]
        info = snap.entry
        decoded = decode(path.read_bytes(), info["width"], info["height"], info["format"])
        if decoded is None:
            snap.stats = Stats(decoded=False)
        else:
            snap.array, snap.is_float = decoded
            snap.stats = image_stats(snap.array)
        return snap.stats

    def snapshot_before(self, uid: int, seq: int) -> Snapshot | None:
        """The latest snapshot of uid taken before draw seq, if no recorded draw wrote the
        image between that snapshot and seq (so it shows what seq read)."""
        candidates = [s for s in self.snapshots.get(uid, []) if s.after_seq < seq and
                      "file" in s.entry]
        if not candidates:
            return None
        snap = max(candidates, key=lambda s: s.after_seq)
        return None if self.written_between(uid, snap.after_seq, seq) else snap

    def snapshot_after(self, uid: int, seq: int) -> Snapshot | None:
        """The first snapshot of uid after draw seq with no later writer in between."""
        candidates = sorted((s for s in self.snapshots.get(uid, []) if s.after_seq >= seq and
                             "file" in s.entry), key=lambda s: s.after_seq)
        for snap in candidates:
            return None if self.written_between(uid, seq, snap.after_seq + 1) else snap
        return None

    def written_between(self, uid: int, low: int, high: int) -> bool:
        """Whether a recorded draw or dispatch with low < seq < high wrote uid."""
        seqs = self.writer_seqs.get(uid, [])
        i = bisect.bisect_right(seqs, low)
        return i < len(seqs) and seqs[i] < high

    def writers(self, uid: int, before: int | None = None):
        seqs = self.writer_seqs.get(uid, [])
        if before is not None:
            seqs = seqs[:bisect.bisect_left(seqs, before)]
        return [self.by_seq[s] for s in seqs]

    def describe(self, uid: int) -> str:
        info = self.image_info.get(uid, {})
        return (f"uid {uid} {info.get('address', '?')} {info.get('width', '?')}x"
                f"{info.get('height', '?')} {info.get('format', '?')}")


def shader_text(record: dict) -> str:
    return " ".join(f"{k} {v}" for k, v in record["shaders"].items())


# ---------------------------------------------------------------------------------------------
# Analysis


def match_focus(bundle: Bundle, focus: str) -> list[int]:
    if focus.startswith("uid="):
        return [int(focus[4:])]
    if focus.startswith("0x"):
        return [uid for uid, i in bundle.image_info.items() if i.get("address") == focus.lower()]
    m = re.fullmatch(r"(\d+)x(\d+)", focus)
    if m:
        w, h = int(m[1]), int(m[2])
        return [uid for uid, i in bundle.image_info.items()
                if i.get("width") == w and i.get("height") == h]
    raise ValueError(f"bad focus {focus!r}")


def trace(bundle: Bundle, uid: int, seq: int, depth: int, seen: set, lines: list[str],
          previews: list[tuple[Snapshot, str]], max_depth: int = 6) -> None:
    """Producers of image uid before draw seq, their inputs and statistics, recursively."""
    indent = "  " * depth
    writers = bundle.writers(uid, seq)[-3:]
    if not writers:
        lines.append(f"{indent}- no recorded draw or dispatch wrote {bundle.describe(uid)} "
                     f"before #{seq} (uploaded, copied or cleared, or before the history)")
        return
    for record in writers:
        key = (uid, record["seq"])
        if key in seen:
            continue
        seen.add(key)
        after = bundle.snapshot_after(uid, record["seq"])
        out_stats = bundle.load(after).summary() if after else "no snapshot right after it"
        if after:
            previews.append((after, f"uid{uid}_after{record['seq']}"))
        lines.append(f"{indent}- #{record['seq']} {record['type']} ({shader_text(record)}) "
                     f"writes {bundle.describe(uid)} -> {out_stats}")
        if record.get("empty_bindings"):
            lines.append(f"{indent}  EMPTY BINDINGS:{record['empty_bindings']}")
        inputs = [i for i in record["images"] if i["role"] in ("read", "storage", "depth") and
                  i["uid"] != uid]
        for ref in inputs:
            before = bundle.snapshot_before(ref["uid"], record["seq"])
            stats = bundle.load(before) if before else None
            note = stats.summary() if stats else "no snapshot of what it read"
            flag = " <-- EMPTY" if stats and stats.empty else (
                " <-- NaN/Inf" if stats and stats.bad else "")
            lines.append(f"{indent}  reads ({ref['role']}) {bundle.describe(ref['uid'])}: "
                         f"{note}{flag}")
            if before:
                previews.append((before, f"uid{ref['uid']}_before{record['seq']}"))
            if depth < max_depth and (stats is None or stats.empty or stats.bad or depth < 2):
                trace(bundle, ref["uid"], record["seq"], depth + 1, seen, lines, previews,
                      max_depth)


def analyze(folder: Path, focus: list[str], warnings: bool = False) -> dict:
    bundle = Bundle(folder)
    report_dir = folder / "report"
    report_dir.mkdir(exist_ok=True)
    window = bundle.window()
    findings: dict[str, list] = defaultdict(list)
    previews: list[tuple[Snapshot, str]] = []

    # 1. Image statistics for every snapshot.
    for uid, snaps in bundle.snapshots.items():
        for snap in snaps:
            stats = bundle.load(snap)
            if stats and stats.bad:
                findings["nan_inf"].append(
                    f"{bundle.describe(uid)} snapshot {snap.snapshot} (after #{snap.after_seq}): "
                    f"{stats.summary()}")
                previews.append((snap, f"uid{uid}_s{snap.snapshot}_naninf"))

    # 2. Reads of empty or never-written images, and empty bindings, in the capture window.
    reported = set()
    for record in window:
        for ref in record["images"]:
            if ref["role"] != "read" and ref["role"] != "storage":
                continue
            uid = ref["uid"]
            before = bundle.snapshot_before(uid, record["seq"])
            stats = bundle.load(before) if before else None
            written = bool(bundle.writers(uid, record["seq"]))
            key = (uid, shader_text(record))
            if key in reported:
                continue
            if stats and stats.empty and ref["role"] == "read":
                reported.add(key)
                findings["reads_empty"].append(
                    f"#{record['seq']} {record['type']} ({shader_text(record)}) reads "
                    f"{bundle.describe(uid)}, which is all zero (snapshot {before.snapshot}"
                    f"{'' if written else ', never written by a recorded draw or dispatch'})")
            elif not written and before is None and ref["role"] == "read":
                mods = bundle.image_info.get(uid, {}).get("modifications", [])
                if not mods:
                    reported.add(key)
                    findings["reads_unwritten"].append(
                        f"#{record['seq']} {record['type']} ({shader_text(record)}) reads "
                        f"{bundle.describe(uid)}: no draw, dispatch or modification of it was "
                        f"recorded")
        if record.get("empty_bindings"):
            key = ("empty", shader_text(record), record["empty_bindings"])
            if key not in reported:
                reported.add(key)
                findings["empty_bindings"].append(
                    f"#{record['seq']} {record['type']} ({shader_text(record)}):"
                    f"{record['empty_bindings']}")

    # 3. Targets that ended black or constant.
    final_snapshot = bundle.manifest.get("snapshots", 0)
    for uid, snaps in bundle.snapshots.items():
        last = [s for s in snaps if s.snapshot == final_snapshot and "file" in s.entry]
        written_in_window = any(i["uid"] == uid and i["role"] in ("target", "storage")
                                for r in window for i in r["images"])
        if last and written_in_window:
            stats = bundle.load(last[0])
            if stats and stats.decoded and (stats.black or stats.constant) and not stats.bad:
                findings["black_targets"].append(
                    f"{bundle.describe(uid)}: {stats.summary()}")

    # 4. Focus chains.
    chains = []
    focus_uids = []
    for f in focus:
        focus_uids += match_focus(bundle, f)
    for uid in dict.fromkeys(focus_uids):
        lines: list[str] = []
        end = bundle.end + 1
        snaps = [s for s in bundle.snapshots.get(uid, []) if "file" in s.entry]
        if snaps:
            last = max(snaps, key=lambda s: s.after_seq)
            stats = bundle.load(last)
            lines.append(f"{bundle.describe(uid)} at the end: "
                         f"{stats.summary() if stats else 'not decoded'}")
            previews.append((last, f"focus_uid{uid}"))
        trace(bundle, uid, end, 0, set(), lines, previews)
        chains.append({"uid": uid, "lines": lines})

    # 5. Log: shader failures, dropped exports, validation, stubs and unimplemented calls.
    log_summary = None
    if bundle.log.exists():
        data = inventory.build([bundle.log], report_dir / "log-inventory", warnings)
        log_summary = data["summary"]
        for group in data["groups"]:
            category = {"shader": "shader_failures", "gpu": "validation", "crash": "crashes",
                        "unimplemented": "unimplemented", "stub": "stubs"}.get(group["category"])
            if category:
                findings[category].append(f"{group['count']}x {group['location']} "
                                          f"{group['function']}: {group['example'][:300]}")
        recompiler = re.compile(r"Render_Recompiler.*<(Warning|Error)>")
        seen_lines = set()
        with open(bundle.log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if recompiler.search(line) and "SrtWalker" not in line:
                    key = inventory.normalize(line.split(": ", 1)[-1])
                    if key not in seen_lines and len(seen_lines) < 100:
                        seen_lines.add(key)
                        findings["recompiler_warnings"].append(line.strip()[:300])

    # Previews.
    written_previews = []
    for snap, name in previews[:300]:
        if snap.array is None:
            bundle.load(snap)
        if snap.array is not None:
            path = report_dir / "previews" / f"{name}.png"
            if not path.exists():
                save_preview(snap.array, snap.is_float, path)
            written_previews.append(str(path.relative_to(folder)))

    order = ["crashes", "shader_failures", "nan_inf", "validation", "reads_empty",
             "reads_unwritten", "empty_bindings", "unimplemented", "recompiler_warnings",
             "black_targets", "stubs"]
    titles = {
        "crashes": "Crashes", "shader_failures": "Shaders not translated / exports dropped",
        "nan_inf": "Images with NaN or Inf", "validation": "Vulkan validation",
        "reads_empty": "Reads of empty images", "reads_unwritten": "Reads of never-written images",
        "empty_bindings": "Draws with empty bindings",
        "unimplemented": "Unimplemented functions called",
        "recompiler_warnings": "Shader recompiler warnings",
        "black_targets": "Targets that ended black or constant", "stubs": "Stubs called",
    }
    m = bundle.manifest
    lines = [f"# Bundle report: {folder.name}", "",
             f"Reason: {m.get('reason')}; draws {bundle.start}..{bundle.end} "
             f"({len(window)} in the capture, {len(bundle.records)} in the history); "
             f"{m.get('snapshots', 0)} snapshots, {m.get('images_dumped', 0)} images dumped, "
             f"{m.get('shaders', 0)} shaders.", "",
             "| Finding | Count |", "| --- | --- |"]
    lines += [f"| {titles[k]} | {len(findings[k])} |" for k in order if findings.get(k)]
    for key in order:
        items = findings.get(key)
        if not items:
            continue
        lines += ["", f"## {titles[key]} ({len(items)})", ""]
        lines += [f"- {item}" for item in items[:200]]
        if len(items) > 200:
            lines.append(f"- ... {len(items) - 200} more")
    for chain in chains:
        lines += ["", f"## Chain of {bundle.describe(chain['uid'])}", ""] + chain["lines"]
    if written_previews:
        lines += ["", "## Previews", "",
                  "PNG previews (float targets tone-mapped, very dark images stretched, NaN/Inf "
                  "in magenta) are in report/previews/."]
    report = {"bundle": str(folder), "manifest": m,
              "counts": {k: len(v) for k, v in findings.items()},
              "findings": findings, "chains": chains, "log_summary": log_summary,
              "previews": written_previews}
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--focus", action="append", default=[],
                        help="image to trace: WxH, uid=N or 0xADDRESS (repeatable)")
    parser.add_argument("--warnings", action="store_true")
    args = parser.parse_args()
    report = analyze(args.bundle, args.focus, args.warnings)
    print(json.dumps(report["counts"]))
    print(f"report: {args.bundle / 'report' / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
