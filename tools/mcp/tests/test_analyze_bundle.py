"""Tests the bundle analyzer on a small synthetic bundle shaped like a real one.

Run: python tools/mcp/tests/test_analyze_bundle.py
"""

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_bundle  # noqa: E402


def write_bundle(folder: Path) -> None:
    (folder / "images").mkdir(parents=True)
    img = lambda uid, w, h, fmt, role: {"uid": uid, "slot": uid, "address": hex(0x1000 * uid),
                                        "width": w, "height": h, "format": fmt, "version": 1,
                                        "flags": 0, "role": role}
    records = [
        # A compute pass meant to fill the lighting image leaves it empty.
        {"seq": 1, "frame": 1, "type": "dispatch", "count": 64, "shaders": {"cs": "0xaaa"},
         "images": [img(10, 4, 4, "R16G16B16A16Sfloat", "storage")], "buffers": []},
        # The car draw reads it and renders a black car.
        {"seq": 2, "frame": 1, "type": "draw", "count": 300,
         "shaders": {"vs": "0xbbb", "fs": "0xccc"},
         "images": [img(20, 4, 4, "R8G8B8A8Unorm", "target"),
                    img(10, 4, 4, "R16G16B16A16Sfloat", "read")],
         "buffers": [{"address": "0x5000", "size": 256, "written": False}],
         "empty_bindings": " | EMPTY buf 3 of 0xccc"},
        # A later pass produces NaN.
        {"seq": 3, "frame": 2, "type": "draw", "count": 3,
         "shaders": {"vs": "0xddd", "fs": "0xeee"},
         "images": [img(30, 4, 4, "R32Sfloat", "target"),
                    img(20, 4, 4, "R8G8B8A8Unorm", "read")], "buffers": []},
    ]
    with open(folder / "draws.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    lighting = np.zeros((4, 4, 4), np.float16).tobytes()
    car = np.zeros((4, 4, 4), np.uint8)
    car[..., 3] = 255
    nan = np.full((4, 4), np.nan, np.float32)
    nan[0, 0] = 1.0
    (folder / "images/s1_uid10.bin").write_bytes(lighting)
    (folder / "images/s2_uid20.bin").write_bytes(car.tobytes())
    (folder / "images/s2_uid30.bin").write_bytes(nan.tobytes())
    entries = [
        {"snapshot": 1, "after_seq": 1, "uid": 10, "width": 4, "height": 4,
         "format": "R16G16B16A16Sfloat", "address": "0xa000", "file": "images/s1_uid10.bin"},
        {"snapshot": 2, "after_seq": 3, "uid": 20, "width": 4, "height": 4,
         "format": "R8G8B8A8Unorm", "address": "0x14000", "file": "images/s2_uid20.bin"},
        {"snapshot": 2, "after_seq": 3, "uid": 30, "width": 4, "height": 4,
         "format": "R32Sfloat", "address": "0x1e000", "file": "images/s2_uid30.bin"},
    ]
    with open(folder / "images.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    (folder / "manifest.json").write_text(json.dumps(
        {"reason": "test", "start_seq": 1, "end_seq": 3, "snapshots": 2, "images_dumped": 3,
         "shaders": 5}))
    (folder / "log.txt").write_text(
        "[Render_Vulkan] <Error> (shadGT:GpuCommandProcessor) vk_pipeline_cache.cpp:1220 "
        "CompileModule: SHADER-FAIL: fs shader 0x123 (permutation 0, 10 dwords) could not be "
        "translated: assertion failed; draws that use it are skipped\n"
        "[Lib.Pad] <Error> (Game:Main) pad.cpp:276 scePadInit: (STUBBED) called\n",
        encoding="utf-8")


class AnalyzerTest(unittest.TestCase):
    def test_findings_and_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "bundle"
            write_bundle(folder)
            report = analyze_bundle.analyze(folder, ["uid=30"])
            findings = report["findings"]
            self.assertEqual(len(findings["nan_inf"]), 1)
            self.assertIn("uid 30", findings["nan_inf"][0])
            self.assertIn("15 NaN", findings["nan_inf"][0])
            self.assertEqual(len(findings["reads_empty"]), 1)
            self.assertIn("uid 10", findings["reads_empty"][0])
            self.assertIn("read by 1 draws/dispatches", findings["reads_empty"][0])
            self.assertIn("written by GPU draws/dispatches #1", findings["reads_empty"][0])
            self.assertEqual(len(findings["empty_bindings"]), 1)
            self.assertEqual(len(findings["contained"]), 1)
            self.assertEqual(len(findings["stubs"]), 1)
            chain = "\n".join(report["chains"][0]["lines"])
            self.assertIn("#3 draw", chain)
            self.assertIn("#2 draw", chain)
            self.assertIn("<-- EMPTY", chain)
            self.assertIn("#1 dispatch", chain)
            self.assertTrue((folder / "report/report.md").exists())
            self.assertTrue(report["previews"])

    def test_decoders(self):
        # B10G11R11: R=1.0 (exp 15, mantissa 0), G=2.0, B=0.5
        r = (15 << 6)
        g = (16 << 6)
        b = (14 << 5)
        packed = struct.pack("<I", r | (g << 11) | (b << 22))
        array, is_float = analyze_bundle.decode(packed, 1, 1, "B10G11R11UfloatPack32")
        self.assertTrue(is_float)
        np.testing.assert_allclose(array[0, 0], [1.0, 2.0, 0.5])
        array, _ = analyze_bundle.decode(bytes([255, 0, 0, 255]), 1, 1, "B8G8R8A8Unorm")
        np.testing.assert_allclose(array[0, 0], [0, 0, 1, 1])
        array, _ = analyze_bundle.decode(np.float16([0.5, 1.5]).tobytes(), 1, 1, "R16G16Sfloat")
        np.testing.assert_allclose(array[0, 0], [0.5, 1.5])
        self.assertIsNone(analyze_bundle.decode(b"\0" * 16, 1, 1, "Bc1RgbaUnormBlock"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
