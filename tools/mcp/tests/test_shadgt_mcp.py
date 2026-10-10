"""Tests the shadGT MCP server end to end over MCP stdio against fake_shadgt.py (no game).

Run: python tools/mcp/tests/test_shadgt_mcp.py
"""

import asyncio
import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent / "shadgt_mcp.py"
FAKE = HERE / "fake_shadgt.py"

sys.path.insert(0, str(HERE.parent))
import shadgt_mcp  # noqa: E402


def attr(obj, *names):
    """mcp 2.x uses snake_case result fields, 1.x camelCase."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def is_error(result):
    return bool(attr(result, "is_error", "isError"))


class McpServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.profile = Path(self.tmp.name)
        (self.profile / "user/log").mkdir(parents=True)
        (self.profile / "user/config.json").write_text("{}", encoding="utf-8")
        self.record = self.profile / "record.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def commands(self):
        if not self.record.exists():
            return []
        return [json.loads(l) for l in self.record.read_text().splitlines()]

    def run_session(self, body, extra_env=None):
        env = dict(os.environ)
        env.update({
            "SHADGT_PROFILE_DIR": str(self.profile),
            "SHADGT_MCP_TEST_COMMAND": json.dumps([sys.executable, str(FAKE)]),
            "FAKE_SHADGT_RECORD": str(self.record),
        })
        env.update(extra_env or {})
        params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=env)

        async def main():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await body(session)

        return asyncio.run(asyncio.wait_for(main(), 120))

    @staticmethod
    def data(result):
        if is_error(result):
            raise AssertionError(result.content[0].text)
        structured = attr(result, "structured_content", "structuredContent")
        if structured is not None:
            return structured.get("result", structured)
        return json.loads(result.content[0].text)

    def test_tools_registered(self):
        async def body(s):
            return sorted(t.name for t in (await s.list_tools()).tools)

        names = self.run_session(body)
        self.assertEqual(names, sorted([
            "launch", "stop", "pause", "resume", "status", "read_log", "wait_for_log",
            "list_dumps", "screenshot", "press", "input_sequence", "capture_frame"]))

    def test_smoke_launch_screenshot_press_stop(self):
        """The smoke test Lance runs on the real game, against the stub."""
        async def body(s):
            launched = self.data(await s.call_tool("launch", {}))
            status = self.data(await s.call_tool("status", {}))
            shot = await s.call_tool("screenshot", {})
            pressed = self.data(await s.call_tool("press", {"button": "cross"}))
            combo = self.data(await s.call_tool(
                "press", {"button": "l1+lstick_left", "hold_frames": 3}))
            await s.call_tool("pause", {})
            await s.call_tool("resume", {})
            stopped = self.data(await s.call_tool("stop", {}))
            return launched, status, shot, pressed, combo, stopped

        launched, status, shot, pressed, combo, stopped = self.run_session(body)

        self.assertIn("ENABLE_TEST_AUTOMATION", launched["capabilities"])
        self.assertEqual(launched["input_mode"], "ipc")
        self.assertTrue(status["running"])
        self.assertEqual(status["serial"], "CUSA03220")
        self.assertEqual(status["title"], "Gran Turismo SPORT")
        self.assertGreater(status["fps"], 20)

        self.assertFalse(is_error(shot), shot.content)
        image = shot.content[0]
        self.assertEqual(image.type, "image")
        from PIL import Image
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(image.data))).size, (64, 36))

        self.assertEqual(pressed["mode"], "ipc")
        self.assertEqual(pressed["buttons"], "0x4000")
        self.assertEqual(combo["buttons"], "0x400")
        self.assertFalse(stopped["killed"])
        self.assertEqual(stopped["exit_code"], 0)

        cmds = self.commands()
        self.assertEqual(cmds[:2], [["RUN"], ["START"]])
        pads = [c for c in cmds if c[0] == "PAD"]
        # Time-based press: hold, then clear.
        self.assertEqual(pads[0], ["PAD", "0x4000", "128", "128", "128", "128", "0", "0",
                                   "1000000"])
        self.assertEqual(pads[1], ["PAD", "0x0", "128", "128", "128", "128", "0", "0", "0"])
        # Frame-based press with the left stick pushed left.
        self.assertEqual(pads[2], ["PAD", "0x400", "0", "128", "128", "128", "0", "0", "3"])
        self.assertIn(["PAUSE"], cmds)
        self.assertIn(["RESUME"], cmds)
        self.assertEqual(cmds[-1], ["STOP"])
        self.assertTrue(any(c[0] == "SCREENSHOT" for c in cmds))

    def test_input_sequence_and_logs(self):
        log = self.profile / "user/log/shad_log.txt"
        log.write_text("[Render] <Warning> DIAG-033 stall 1: game frame 5 took 80 ms.\n",
                       encoding="utf-8")

        async def body(s):
            await s.call_tool("launch", {})
            seq = self.data(await s.call_tool("input_sequence", {"steps": [
                {"press": "cross", "hold_ms": 50},
                {"wait_ms": 100},
                {"press": "circle", "hold_frames": 2},
                {"screenshot": True},
            ]}))

            async def append_later():
                await asyncio.sleep(0.5)
                with open(log, "a", encoding="utf-8") as f:
                    f.write("[Loader] <Info> Race started\n")

            task = asyncio.create_task(append_later())
            waited = self.data(await s.call_tool(
                "wait_for_log", {"pattern": "race started", "timeout_s": 10}))
            await task
            read = self.data(await s.call_tool("read_log", {"pattern": "DIAG-033"}))
            dumps = self.data(await s.call_tool("list_dumps", {}))
            await s.call_tool("stop", {})
            return seq, waited, read, dumps

        seq, waited, read, dumps = self.run_session(body)
        self.assertEqual(len(seq), 4)
        self.assertTrue(Path(seq[3]["screenshot"]).exists())
        self.assertTrue(waited["found"])
        self.assertIn("Race started", waited["line"])
        self.assertEqual(read["matches"], 1)
        self.assertEqual(dumps["stall_count"], 1)
        self.assertTrue(dumps["screenshots"])

    def test_old_build_falls_back_to_phase1(self):
        async def body(s):
            launched = self.data(await s.call_tool("launch", {}))
            status = self.data(await s.call_tool("status", {}))
            await s.call_tool("stop", {})
            return launched, status

        launched, status = self.run_session(body, {"FAKE_SHADGT_NO_AUTOMATION": "1"})
        self.assertNotIn("ENABLE_TEST_AUTOMATION", launched["capabilities"])
        self.assertEqual(launched["input_mode"], "keyboard (SendInput)")
        self.assertEqual(launched["screenshot_mode"], "window capture")
        self.assertNotIn("frames", status)
        self.assertNotIn(["STATUS"], self.commands())


class UnitTest(unittest.TestCase):
    def test_enabled_file_patches(self):
        # Only ticked byte patches of the game's file, as the Qt launcher sends them.
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "user" / "patches" / "shadPS4"
            repo.mkdir(parents=True)
            (repo / "files.json").write_text(
                json.dumps({"GranTurismoSport.xml": ["CUSA02168", "CUSA03220"],
                            "Other.xml": ["CUSA00001"]}), encoding="utf-8")
            (repo / "GranTurismoSport.xml").write_text(
                '<?xml version="1.0"?><Patch>'
                '<Metadata Name="Boot fix" AppElf="eboot.bin" isEnabled="true"><PatchList>'
                '<Line Type="bytes" Address="0x0211ab30" Value="e92ba3490090"/>'
                '</PatchList></Metadata>'
                '<Metadata Name="30 FPS lock (shadGT)" AppElf="eboot.bin" isEnabled="false">'
                '<PatchList><Line Type="bytes" Address="0x020748e2" Value="e8793cb500909090"/>'
                '</PatchList></Metadata></Patch>', encoding="utf-8")
            self.assertEqual(shadgt_mcp.enabled_file_patches(Path(tmp), "CUSA03220"),
                             [("Boot fix", "0x0211ab30", "e92ba3490090")])
            self.assertEqual(shadgt_mcp.enabled_file_patches(Path(tmp), "CUSA99999"), [])
            self.assertEqual(shadgt_mcp.enabled_file_patches(Path(tmp) / "none", "CUSA03220"), [])

    def test_parse_status(self):
        parsed = shadgt_mcp.parse_status(
            "STATUS frames=12 fps=59.8 paused=0 serial=CUSA03220 app_ver=01.69 "
            "title=Gran Turismo SPORT")
        self.assertEqual(parsed, {"frames": 12, "fps": 59.8, "paused": 0, "serial": "CUSA03220",
                                  "app_ver": "01.69", "title": "Gran Turismo SPORT"})

    def test_buttons(self):
        self.assertEqual(shadgt_mcp.normalize_button("X"), "cross")
        self.assertEqual(shadgt_mcp.normalize_button("pad_up"), "up")
        with self.assertRaises(ValueError):
            shadgt_mcp.normalize_button("select")

    def test_keyboard_mapping_from_profile_config(self):
        profile = shadgt_mcp._main_checkout() / "Build/gt-sport-fixed"
        if not (profile / "user/input_config/default.ini").exists():
            self.skipTest("GT Sport profile not present")
        bindings = shadgt_mcp.parse_input_config(profile)
        # cross = kp2 and cross = n: the non-keypad key wins.
        self.assertEqual(shadgt_mcp.keys_for("cross", bindings), ["n"])
        self.assertEqual(shadgt_mcp.keys_for("hotkey_capture_frame", bindings), ["f12"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
