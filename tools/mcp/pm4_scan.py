"""Find the PM4 packets in a diagnostic bundle that write memory around given addresses.

    python tools/mcp/pm4_scan.py <bundle dir> 0x20320d240 [0x203235400 ...] [--window 0x1000]

Walks every captured submission (pm4/index.jsonl: graphics DCB/CCB and compute ACBs) and lists,
in order, the packets that write guest memory: DUMP_CONST_RAM (constant engine RAM to memory),
WRITE_DATA, DMA_DATA, COPY_DATA, EVENT_WRITE_EOP / RELEASE_MEM fences, plus the CE/DE counter
packets (INCREMENT_CE_COUNTER, WAIT_ON_CE_COUNTER, ...) that order them. A destination within
--window bytes of an address is marked with <==.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

OPCODES = {
    0x10: "NOP", 0x37: "WRITE_DATA", 0x40: "COPY_DATA", 0x47: "EVENT_WRITE_EOP",
    0x49: "RELEASE_MEM", 0x50: "DMA_DATA", 0x81: "WRITE_CONST_RAM", 0x83: "DUMP_CONST_RAM",
    0x84: "INCREMENT_CE_COUNTER", 0x85: "INCREMENT_DE_COUNTER", 0x86: "WAIT_ON_CE_COUNTER",
    0x88: "WAIT_ON_DE_COUNTER_DIFF", 0x8b: "SWITCH_BUFFER", 0x3c: "WAIT_REG_MEM",
    0x2d: "DRAW_INDEX_AUTO", 0x27: "DRAW_INDEX_2", 0x15: "DISPATCH_DIRECT",
    0x3f: "INDIRECT_BUFFER", 0x33: "INDIRECT_BUFFER_CONST",
}
ORDERING = {"INCREMENT_CE_COUNTER", "INCREMENT_DE_COUNTER", "WAIT_ON_CE_COUNTER",
            "WAIT_ON_DE_COUNTER_DIFF", "SWITCH_BUFFER"}


def packets(dwords: list[int]):
    i = 0
    while i < len(dwords):
        header = dwords[i]
        kind = header >> 30
        if kind == 3:
            count = ((header >> 16) & 0x3FFF) + 1
            opcode = (header >> 8) & 0xFF
            yield i, opcode, dwords[i + 1:i + 1 + count]
            i += 1 + count
        elif kind == 2:
            i += 1
        elif kind == 0:
            count = ((header >> 16) & 0x3FFF) + 1
            i += 1 + count
        else:
            i += 1


def destination(opcode: int, body: list[int]) -> tuple[int, int] | None:
    """(address, bytes) written to guest memory, when known."""
    try:
        if opcode == 0x83:  # DUMP_CONST_RAM: offset, num_dw, addr_lo, addr_hi
            return body[2] | (body[3] << 32), (body[1] & 0x7FFF) * 4
        if opcode == 0x37:  # WRITE_DATA: control, addr_lo, addr_hi, data...
            return body[1] | (body[2] << 32), (len(body) - 3) * 4
        if opcode == 0x50:  # DMA_DATA: control, src_lo, src_hi, dst_lo, dst_hi, command
            return body[3] | (body[4] << 32), body[5] & 0x1FFFFF
        if opcode == 0x40:  # COPY_DATA: control, src_lo, src_hi, dst_lo, dst_hi
            return body[3] | (body[4] << 32), 8 if (body[0] >> 16) & 1 else 4
        if opcode in (0x47, 0x49):  # fences: address in dwords 2..3 (EOP) / 3..4 (RELEASE)
            lo, hi = (body[1], body[2]) if opcode == 0x47 else (body[2], body[3])
            return (lo | ((hi & 0xFFFF) << 32)) & ~3, 8
    except IndexError:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("addresses", nargs="+")
    parser.add_argument("--window", default="0x1000")
    parser.add_argument("--all", action="store_true", help="list every memory write")
    args = parser.parse_args()
    targets = [int(a, 16) for a in args.addresses]
    window = int(args.window, 16)
    index = [json.loads(l) for l in (args.bundle / "pm4/index.jsonl").read_text().splitlines()
             if l.strip()]
    for entry in index:
        for part in ("first", "second"):
            name = entry.get(part)
            if not name:
                continue
            raw = (args.bundle / "pm4" / name).read_bytes()
            dwords = list(struct.unpack(f"<{len(raw) // 4}I", raw))
            for offset, opcode, body in packets(dwords):
                op = OPCODES.get(opcode, f"op{opcode:#x}")
                dest = destination(opcode, body)
                near = dest and any(dest[0] - window <= t < dest[0] + dest[1] + window
                                    for t in targets)
                if near or (args.all and dest) or op in ORDERING and args.all:
                    where = f"{dest[0]:#x}+{dest[1]:#x}" if dest else ""
                    mark = " <==" if near else ""
                    print(f"submit {entry['n']} {entry['queue']} {part} frame {entry['frame']} "
                          f"@{offset:#x} {op} {where}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
