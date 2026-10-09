"""Replay register state from a bundle's PM4 capture and print the draws/dispatches that match.

    python tools/mcp/pm4_draws.py <bundle dir> [--rt 0x1009bc0000] [--cs-pgm 0x...] [--max 20]

Tracks SET_SH_REG / SET_CONTEXT_REG writes through the captured graphics indirect buffers (in
processing order) and, at each draw, reports the pixel shader program address, its user data
SGPRs decoded as T# / S# where they look like descriptors, and color target 0. --rt filters on
CB_COLOR0_BASE (guest address of the render target).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

SH_BASE, CTX_BASE = 0x2C00, 0xA000
PGM_LO_PS, USER_DATA_PS = 0x2C08, 0x2C0C
CB_COLOR0_BASE, CB_COLOR0_VIEW = 0xA318, 0xA31B
DRAW_OPS = {0x2D: "DRAW_INDEX_AUTO", 0x27: "DRAW_INDEX_2", 0x35: "DRAW_INDEX_OFFSET_2",
            0x25: "DRAW_INDIRECT", 0x26: "DRAW_INDEX_INDIRECT"}


def decode_tsharp(w: list[int]) -> dict:
    """GCN image descriptor (T#), 8 dwords (Southern/Sea Islands layout)."""
    base = (w[0] | ((w[1] & 0xFF) << 32)) << 8
    return {
        "address": hex(base),
        "data_format": (w[1] >> 20) & 0x3F,
        "num_format": (w[1] >> 26) & 0xF,
        "width": (w[2] & 0x3FFF) + 1,
        "height": ((w[2] >> 14) & 0x3FFF) + 1,
        "perf_mod": (w[2] >> 28) & 7,
        "dst_sel": [(w[3] >> s) & 7 for s in (0, 3, 6, 9)],
        "base_level": (w[3] >> 12) & 0xF,
        "last_level": (w[3] >> 16) & 0xF,
        "tiling_index": (w[3] >> 20) & 0x1F,
        "pow2pad": (w[3] >> 25) & 1,
        "type": (w[3] >> 28) & 0xF,
        "depth": (w[4] & 0x1FFF) + 1,
        "pitch": ((w[4] >> 13) & 0x3FFF) + 1,
        "base_array": w[5] & 0x1FFF,
        "last_array": (w[5] >> 13) & 0x1FFF,
        "min_lod_warn": w[6] & 0xFFF,
    }


def packets(dwords):
    i = 0
    while i < len(dwords):
        header = dwords[i]
        kind = header >> 30
        if kind == 3:
            count = ((header >> 16) & 0x3FFF) + 1
            yield (header >> 8) & 0xFF, dwords[i + 1:i + 1 + count]
            i += 1 + count
        elif kind == 0:
            i += 1 + ((header >> 16) & 0x3FFF) + 1
        else:
            i += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--rt", default=None, help="CB_COLOR0 guest address to match")
    parser.add_argument("--max", type=int, default=20)
    args = parser.parse_args()
    rt = int(args.rt, 16) if args.rt else None
    index = [json.loads(l) for l in (args.bundle / "pm4/index.jsonl").read_text().splitlines()
             if l.strip()]
    regs: dict[int, int] = {}
    shown = 0
    for entry in index:
        if not entry["queue"].startswith("gfx"):
            continue
        raw = (args.bundle / "pm4" / entry["first"]).read_bytes()
        dwords = list(struct.unpack(f"<{len(raw) // 4}I", raw))
        for opcode, body in packets(dwords):
            if opcode in (0x76, 0x69) and body:  # SET_SH_REG, SET_CONTEXT_REG
                base = SH_BASE if opcode == 0x76 else CTX_BASE
                for k, value in enumerate(body[1:]):
                    regs[base + (body[0] & 0xFFFF) + k] = value
            elif opcode in DRAW_OPS:
                cb = regs.get(CB_COLOR0_BASE, 0) << 8
                if rt is not None and cb != rt:
                    continue
                pgm = regs.get(PGM_LO_PS, 0) << 8
                ud = [regs.get(USER_DATA_PS + k, 0) for k in range(16)]
                view = regs.get(CB_COLOR0_VIEW, 0)
                pitch_tiles = (regs.get(0xA319, 0) & 0x7FF) + 1
                slice_tiles = (regs.get(0xA31A, 0) & 0x3FFFFF) + 1
                pitch = pitch_tiles * 8
                height = slice_tiles * 64 // pitch if pitch else 0
                info = regs.get(0xA31C, 0)
                print(f"submit {entry['n']} {DRAW_OPS[opcode]} ps_pgm {pgm:#x} RT0 {cb:#x} "
                      f"pitch {pitch} height {height} format {(info >> 2) & 0x1F} "
                      f"tile_index {regs.get(0xA31D, 0) & 0x1F} "
                      f"view slice {view & 0x7FF}..{(view >> 13) & 0x7FF} "
                      f"scissor {regs.get(0xA00C, 0):#x}..{regs.get(0xA00D, 0):#x}")
                print("  user data:", " ".join(f"{x:08x}" for x in ud))
                print("  T# s[0:7]:", decode_tsharp(ud[0:8]))
                shown += 1
                if shown >= args.max:
                    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
