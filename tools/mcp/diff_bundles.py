"""Compare a good and a bad diagnostic bundle of the same scene, draw by draw.

    python tools/mcp/diff_bundles.py <good bundle> <bad bundle> [--target 800x450:R8G8B8A8Unorm]

Finds, in each bundle, the last draw that writes the target (the image the game reads back, e.g.
the car thumbnail) and every draw/dispatch in the capture that feeds it. The draws of the last
captured frame are aligned between the bundles by their shaders and image shapes (addresses
differ between runs), and along the chain each aligned pair is compared:

  - draws present in only one bundle,
  - vertex/index count and empty bindings (size-0 buffers, unbound images),
  - each input image's contents just before the draw and each target's contents after it,
  - the PM4 context registers in effect (viewport, scissor, depth/stencil, blend, masks, ...)
    and SET_PREDICATION, when the capture's draw packets line up with the recorded draws.

The first divergence in draw order is printed first: that is where the bad run starts to differ.
"""

from __future__ import annotations

import argparse
import difflib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_bundle as ab  # noqa: E402
from pm4_draws import CTX_BASE, DRAW_OPS, SH_BASE, packets  # noqa: E402
from pm4_scan import OPCODES, destination  # noqa: E402

# GPU work that writes memory the CPU can read back: labels/fences, data writes, DMA, copies,
# and EVENT_WRITE (occlusion query ZPASS_DONE / sample counters write their results).
GPU_TO_CPU = {0x37, 0x40, 0x47, 0x49, 0x50, 0x46}

# Context registers holding guest addresses or per-run values: they differ between runs.
ADDRESS_REGS = {0xA010, 0xA011, 0xA012, 0xA013, 0xA014, 0xA015, 0xA016, 0xA017, 0xA005, 0xA006}
ADDRESS_REGS |= {0xA318 + 15 * i + k for i in range(8) for k in (0, 6, 7, 8, 9, 10, 11)}
REG_NAMES = {
    0xA00C: "PA_SC_WINDOW_SCISSOR_TL", 0xA00D: "PA_SC_WINDOW_SCISSOR_BR",
    0xA08E: "CB_TARGET_MASK", 0xA08F: "CB_SHADER_MASK", 0xA094: "PA_SC_VPORT_SCISSOR_0_TL",
    0xA095: "PA_SC_VPORT_SCISSOR_0_BR", 0xA0B4: "PA_SC_VPORT_ZMIN_0", 0xA0B5: "PA_SC_VPORT_ZMAX_0",
    0xA10B: "DB_STENCIL_CONTROL", 0xA10C: "DB_STENCILREFMASK", 0xA10D: "DB_STENCILREFMASK_BF",
    0xA10F: "PA_CL_VPORT_XSCALE", 0xA110: "PA_CL_VPORT_XOFFSET", 0xA111: "PA_CL_VPORT_YSCALE",
    0xA112: "PA_CL_VPORT_YOFFSET", 0xA113: "PA_CL_VPORT_ZSCALE", 0xA114: "PA_CL_VPORT_ZOFFSET",
    0xA1E0: "CB_BLEND0_CONTROL", 0xA200: "DB_DEPTH_CONTROL", 0xA202: "CB_COLOR_CONTROL",
    0xA203: "DB_SHADER_CONTROL", 0xA204: "PA_CL_CLIP_CNTL", 0xA205: "PA_SU_SC_MODE_CNTL",
    0xA000: "DB_RENDER_CONTROL", 0xA001: "DB_COUNT_CONTROL", 0xA003: "DB_RENDER_OVERRIDE",
}


def parse_target(text: str) -> tuple[int, int, str | None]:
    size, _, fmt = text.partition(":")
    w, h = size.split("x")
    return int(w), int(h), fmt or None


def chain(bundle: ab.Bundle, target: tuple[int, int, str | None]) -> tuple[int, set[int]]:
    """The last draw writing the target and every recorded draw/dispatch that feeds it."""
    w, h, fmt = target
    window = bundle.window()
    final = None
    for record in window:
        for ref in record["images"]:
            if (ref["role"] == "target" and ref["width"] == w and ref["height"] == h and
                    (fmt is None or ref["format"].startswith(fmt))):
                final = record
    if final is None:
        raise SystemExit(f"{bundle.dir}: no draw writes a {w}x{h} target in the capture")
    seqs = {final["seq"]}
    todo = [final]
    while todo:
        record = todo.pop()
        for ref in record["images"]:
            if ref["role"] == "target" and record is not final:
                continue
            writers = [r for r in bundle.writers(ref["uid"], before=record["seq"])
                       if r["seq"] >= bundle.start]
            # Every writer in the capture that lands before this draw can contribute
            # (accumulation, partial updates); walk them all.
            for writer in writers:
                if writer["seq"] not in seqs:
                    seqs.add(writer["seq"])
                    todo.append(writer)
    return final["seq"], seqs


def signature(record: dict) -> tuple:
    shaders = record.get("shaders", {})
    stages = tuple(sorted(shaders.items())) if isinstance(shaders, dict) else tuple(shaders)
    images = tuple((r["role"], r["width"], r["height"], r["format"]) for r in record["images"])
    return record.get("type", ""), stages, images


def last_frame(bundle: ab.Bundle, final_seq: int) -> list[dict]:
    frame = bundle.by_seq[final_seq]["frame"]
    return [r for r in bundle.window() if r["frame"] == frame]


def content(bundle: ab.Bundle, uid: int, seq: int, after: bool) -> str:
    snap = bundle.snapshot_after(uid, seq) if after else bundle.snapshot_before(uid, seq)
    if snap is None:
        return "?"
    stats = bundle.load(snap)
    return stats.summary() if stats else "?"


def coarse(summary: str) -> str:
    """Class of an image's contents, for comparisons that ignore small numeric differences."""
    if summary == "?":
        return "?"
    for word in ("NaN", "Inf", "all zero", "black", "constant"):
        if word in summary:
            return word
    return "content"


def empty_count(record: dict) -> tuple[int, int]:
    text = record.get("empty_bindings", "")
    return text.count("EMPTY buf"), text.count("EMPTY img")


def pm4_registers(bundle: ab.Bundle) -> list[tuple[dict[int, int], int]]:
    """Context registers and predication in effect at each graphics draw packet, in order."""
    index_path = bundle.dir / "pm4/index.jsonl"
    if not index_path.exists():
        return []
    index = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
    regs: dict[int, int] = {}
    predication = 0
    out = []
    for entry in index:
        if not entry["queue"].startswith("gfx"):
            continue
        raw = (bundle.dir / "pm4" / entry["first"]).read_bytes()
        dwords = list(struct.unpack(f"<{len(raw) // 4}I", raw))
        for opcode, body in packets(dwords):
            if opcode in (0x76, 0x69) and body:
                base = SH_BASE if opcode == 0x76 else CTX_BASE
                for k, value in enumerate(body[1:]):
                    regs[base + (body[0] & 0xFFFF) + k] = value
            elif opcode == 0x20 and len(body) >= 2:  # SET_PREDICATION
                predication = body[1]
            elif opcode in DRAW_OPS:
                out.append(({k: v for k, v in regs.items() if k >= CTX_BASE}, predication))
    return out


def gpu_to_cpu(bundle: ab.Bundle) -> list[tuple]:
    """(queue, opcode name, bytes, data/event summary) for each packet writing memory the CPU
    can read, in capture order. Addresses are left out: they differ between runs."""
    index_path = bundle.dir / "pm4/index.jsonl"
    if not index_path.exists():
        return []
    out = []
    for entry in (json.loads(l) for l in index_path.read_text().splitlines() if l.strip()):
        for part in ("first", "second"):
            name = entry.get(part)
            if not name:
                continue
            raw = (bundle.dir / "pm4" / name).read_bytes()
            dwords = list(struct.unpack(f"<{len(raw) // 4}I", raw))
            for opcode, body in packets(dwords):
                if opcode not in GPU_TO_CPU or not body:
                    continue
                op = OPCODES.get(opcode, "EVENT_WRITE" if opcode == 0x46 else hex(opcode))
                dest = destination(opcode, body)
                if opcode == 0x47:  # EOP: event, addr lo, addr hi | data_sel, data lo, data hi
                    detail = (f"event {body[0] & 0x3F:#x} data_sel {(body[2] >> 29) & 7} "
                              f"value {body[3] if len(body) > 3 else 0:#x}")
                elif opcode == 0x49:  # RELEASE_MEM
                    detail = (f"event {body[0] & 0x3F:#x} data_sel {(body[1] >> 29) & 7} "
                              f"value {body[4] if len(body) > 4 else 0:#x}")
                elif opcode == 0x37:  # WRITE_DATA payload
                    detail = "data " + " ".join(f"{d:#x}" for d in body[3:7])
                elif opcode == 0x46:  # EVENT_WRITE: event type (0x1 = ZPASS_DONE ...)
                    detail = f"event {body[0] & 0x3F:#x} index {(body[0] >> 8) & 0xF}"
                else:
                    detail = f"control {body[0]:#x}"
                out.append((entry["queue"], op, dest[1] if dest else 0, detail))
    return out


def register_diff(good: dict[int, int], bad: dict[int, int]) -> list[str]:
    lines = []
    for reg in sorted(set(good) | set(bad)):
        if reg in ADDRESS_REGS:
            continue
        a, b = good.get(reg), bad.get(reg)
        if a != b:
            name = REG_NAMES.get(reg, f"ctx {reg:#x}")
            lines.append(f"{name} good {a if a is None else hex(a)} bad "
                         f"{b if b is None else hex(b)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("good", type=Path)
    parser.add_argument("bad", type=Path)
    parser.add_argument("--target", default="800x450:R8G8B8A8")
    parser.add_argument("--max", type=int, default=40, help="divergences to print")
    args = parser.parse_args()
    target = parse_target(args.target)

    good, bad = ab.Bundle(args.good), ab.Bundle(args.bad)
    good_final, good_chain = chain(good, target)
    bad_final, bad_chain = chain(bad, target)
    good_frame, bad_frame = last_frame(good, good_final), last_frame(bad, bad_final)
    print(f"good: final draw #{good_final}, {len(good_chain)} draws in its chain, "
          f"{len(good_frame)} in its frame")
    print(f"bad:  final draw #{bad_final}, {len(bad_chain)} draws in its chain, "
          f"{len(bad_frame)} in its frame")

    # PM4 draw packets in capture order; recorded graphics draws in the same order.
    good_regs, bad_regs = pm4_registers(good), pm4_registers(bad)
    good_draws = [r["seq"] for r in good.window() if r.get("type") == "draw"]
    bad_draws = [r["seq"] for r in bad.window() if r.get("type") == "draw"]
    good_pm4 = dict(zip(good_draws, good_regs)) if len(good_regs) == len(good_draws) else {}
    bad_pm4 = dict(zip(bad_draws, bad_regs)) if len(bad_regs) == len(bad_draws) else {}
    if not good_pm4 or not bad_pm4:
        print(f"PM4 draw packets do not line up with the recorded draws (good "
              f"{len(good_regs)}/{len(good_draws)}, bad {len(bad_regs)}/{len(bad_draws)}); "
              f"register comparison skipped")

    matcher = difflib.SequenceMatcher(a=[signature(r) for r in good_frame],
                                      b=[signature(r) for r in bad_frame], autojunk=False)
    findings: list[tuple[int, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for g, b in zip(good_frame[i1:i2], bad_frame[j1:j2]):
                if g["seq"] not in good_chain and b["seq"] not in bad_chain:
                    continue
                where = f"good #{g['seq']} / bad #{b['seq']} {g.get('type')} {g.get('shaders')}"
                if g["count"] != b["count"]:
                    findings.append((g["seq"], f"{where}: count {g['count']} vs {b['count']}"))
                if empty_count(g) != empty_count(b):
                    findings.append((g["seq"], f"{where}: empty bindings (buf, img) "
                                     f"{empty_count(g)} vs {empty_count(b)}: {b['empty_bindings'][:300]}"))
                for k, (gr, br) in enumerate(zip(g["images"], b["images"])):
                    after = gr["role"] != "read"
                    gc = content(good, gr["uid"], g["seq"], after)
                    bc = content(bad, br["uid"], b["seq"], after)
                    if coarse(gc) != coarse(bc):
                        findings.append((g["seq"], f"{where}: {'output' if after else 'input'} "
                                         f"{k} {gr['role']} {gr['width']}x{gr['height']} "
                                         f"{gr['format']}: good uid {gr['uid']} {gc} | bad uid "
                                         f"{br['uid']} {bc}"))
                if g["seq"] in good_pm4 and b["seq"] in bad_pm4:
                    (gregs, gpred), (bregs, bpred) = good_pm4[g["seq"]], bad_pm4[b["seq"]]
                    diff = register_diff(gregs, bregs)
                    if gpred != bpred:
                        diff.append(f"SET_PREDICATION good {gpred:#x} bad {bpred:#x}")
                    if diff:
                        findings.append((g["seq"], f"{where}: registers: " + "; ".join(diff[:12])))
        else:
            for g in good_frame[i1:i2]:
                if g["seq"] in good_chain:
                    findings.append((g["seq"], f"only in good: #{g['seq']} {g.get('type')} "
                                     f"{g.get('shaders')} count {g['count']}"))
            for b in bad_frame[j1:j2]:
                if b["seq"] in bad_chain:
                    anchor = good_frame[min(i1, len(good_frame) - 1)]["seq"] if good_frame else 0
                    findings.append((anchor, f"only in bad: #{b['seq']} {b.get('type')} "
                                     f"{b.get('shaders')} count {b['count']}"))

    # GPU-to-CPU results: the CPU decides what to draw next from these (labels, queries,
    # readbacks). A packet the bad run lacks, or a different value, is a lead.
    good_g2c, bad_g2c = gpu_to_cpu(good), gpu_to_cpu(bad)
    print(f"\nGPU-to-CPU writes in the capture: good {len(good_g2c)}, bad {len(bad_g2c)}")
    for label, items in (("good", good_g2c), ("bad", bad_g2c)):
        kinds: dict[str, int] = {}
        for q, op, _, _ in items:
            kinds[f"{q} {op}"] = kinds.get(f"{q} {op}", 0) + 1
        print(f"  {label}: " + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())))
    g2c = difflib.SequenceMatcher(a=good_g2c, b=bad_g2c, autojunk=False)
    shown = 0
    for tag, i1, i2, j1, j2 in g2c.get_opcodes():
        if tag == "equal" or shown >= args.max:
            continue
        shown += 1
        print(f"  {tag}: good {good_g2c[i1:i2][:4]} | bad {bad_g2c[j1:j2][:4]}")

    findings.sort(key=lambda f: f[0])
    print(f"\n{len(findings)} divergences along the chain (draw order):")
    for _, text in findings[: args.max]:
        print(" -", text)
    if len(findings) > args.max:
        print(f" ... {len(findings) - args.max} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
