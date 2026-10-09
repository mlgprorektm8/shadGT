"""Checkpoints: jump straight back to a test point by restoring save data and replaying inputs.

A checkpoint is a folder under <profile>/checkpoints/<name>/:

  route.json   the inputs since boot, each with the screen it was pressed on
  step-NNN.png the screen just before input NNN (what replay waits for)
  final.png    the screen when the checkpoint was saved (what replay verifies)
  savedata/    GT Sport's save data as it was when the recorded run was launched

Replay restores the save data, launches, and for every input waits until the screen matches
the recorded one (inside an optional box, so random backgrounds can be ignored) before
pressing. It stops at the first step whose screen never matches and reports both pictures.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROUTE_VERSION = 1
SAVEDATA_REL = Path("user/home/1000/savedata/CUSA03220")


def similarity(a, b, box: list[float] | None = None) -> float:
    """1.0 for identical pictures, lower as they differ (mean absolute grey difference).

    box is [x0, y0, x1, y1] as fractions of the picture (0..1), compared on both."""
    from PIL import ImageChops, ImageStat

    def prepare(image):
        image = image.convert("L")
        if box:
            w, h = image.size
            image = image.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w),
                                int(box[3] * h)))
        return image.resize((160, 90))

    diff = ImageChops.difference(prepare(a), prepare(b))
    return 1.0 - ImageStat.Stat(diff).mean[0] / 255.0


@dataclass
class Step:
    press: str
    hold_ms: int = 100
    hold_frames: int | None = None
    after_ms: int = 0          # time since the previous input (or launch) when recorded
    ref: str | None = None     # screen before this input
    box: list[float] | None = None
    min_similarity: float | None = None
    wait: bool = True          # wait for the reference screen before pressing


@dataclass
class Route:
    name: str
    description: str = ""
    steps: list[Step] = field(default_factory=list)
    final_ref: str | None = None
    final_box: list[float] | None = None
    min_similarity: float = 0.90
    launch: dict = field(default_factory=dict)
    version: int = ROUTE_VERSION

    def to_json(self) -> dict:
        return {
            "version": self.version, "name": self.name, "description": self.description,
            "min_similarity": self.min_similarity, "final_ref": self.final_ref,
            "final_box": self.final_box, "launch": self.launch,
            "steps": [s.__dict__ for s in self.steps],
        }

    @staticmethod
    def from_json(data: dict) -> "Route":
        steps = [Step(**s) for s in data.get("steps", [])]
        return Route(name=data["name"], description=data.get("description", ""), steps=steps,
                     final_ref=data.get("final_ref"), final_box=data.get("final_box"),
                     min_similarity=data.get("min_similarity", 0.90),
                     launch=data.get("launch", {}), version=data.get("version", 1))


class Recorder:
    """Records a run's inputs from launch, for saving as a checkpoint."""

    def __init__(self, work_dir: Path, profile_dir: Path, launch: dict):
        self.dir = work_dir
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        source = profile_dir / SAVEDATA_REL
        if source.exists():
            shutil.copytree(source, self.dir / "savedata")
        self.route = Route(name="", launch=launch)
        self.last = time.monotonic()

    def add(self, press: str, hold_ms: int, hold_frames: int | None,
            screenshot: Callable[[Path], None]) -> Step:
        now = time.monotonic()
        index = len(self.route.steps)
        ref = f"step-{index:03}.png"
        try:
            screenshot(self.dir / ref)
        except Exception:
            ref = None
        step = Step(press=press, hold_ms=hold_ms, hold_frames=hold_frames,
                    after_ms=int((now - self.last) * 1000), ref=ref)
        self.route.steps.append(step)
        self.last = time.monotonic()
        return step

    def save(self, target: Path, name: str, description: str,
             screenshot: Callable[[Path], None]) -> Route:
        screenshot(self.dir / "final.png")
        self.route.name = name
        self.route.description = description
        self.route.final_ref = "final.png"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(self.dir, target)
        (target / "route.json").write_text(json.dumps(self.route.to_json(), indent=2),
                                           encoding="utf-8")
        return self.route


def load_route(folder: Path) -> Route:
    return Route.from_json(json.loads((folder / "route.json").read_text(encoding="utf-8")))


def save_route(folder: Path, route: Route) -> None:
    (folder / "route.json").write_text(json.dumps(route.to_json(), indent=2), encoding="utf-8")


def restore_savedata(folder: Path, profile_dir: Path, backup_dir: Path) -> bool:
    """Replaces the profile's GT Sport save data with the checkpoint's (the current one is
    kept in backup_dir/last-before-checkpoint first)."""
    source = folder / "savedata"
    if not source.exists():
        return False
    target = profile_dir / SAVEDATA_REL
    backup = backup_dir / "last-before-checkpoint"
    if target.exists():
        if backup.exists():
            shutil.rmtree(backup)
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(target, backup)
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return True


class Diverged(RuntimeError):
    pass


def wait_for_screen(grab: Callable[[], object], ref, box, threshold: float, timeout_s: float,
                    alive: Callable[[], bool], poll_s: float = 0.4):
    """Grabs screens until one matches ref (similarity >= threshold). Returns (best score,
    last picture, matched)."""
    deadline = time.monotonic() + timeout_s
    best = -1.0
    picture = None
    while True:
        picture = grab()
        score = similarity(picture, ref, box)
        best = max(best, score)
        if score >= threshold:
            return score, picture, True
        if time.monotonic() >= deadline or not alive():
            return best, picture, False
        time.sleep(poll_s)


def replay(folder: Path, route: Route, press: Callable[[Step], None], grab: Callable[[], object],
           alive: Callable[[], bool], report_dir: Path, first_timeout_s: float = 180.0,
           step_timeout_s: float = 60.0, log: Callable[[str], None] = print) -> dict:
    """Replays a route on a running emulator. Raises Diverged at the first mismatch."""
    from PIL import Image

    report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, step in enumerate(route.steps):
        if step.wait and step.ref and (folder / step.ref).exists():
            ref = Image.open(folder / step.ref)
            threshold = step.min_similarity or route.min_similarity
            timeout = first_timeout_s if index == 0 else max(step_timeout_s,
                                                              step.after_ms / 1000 * 3)
            score, picture, matched = wait_for_screen(grab, ref, step.box, threshold, timeout,
                                                      alive)
            if not matched:
                actual = report_dir / f"diverged-step-{index:03}.png"
                picture.save(actual)
                raise Diverged(
                    f"step {index} ({step.press}): the screen never matched {step.ref} "
                    f"(best similarity {score:.3f} < {threshold}); see {actual} vs "
                    f"{folder / step.ref}")
            results.append({"step": index, "press": step.press, "similarity": round(score, 3)})
        else:
            time.sleep(step.after_ms / 1000)
            results.append({"step": index, "press": step.press, "similarity": None})
        press(step)
        log(f"step {index}: {step.press}")
    final = None
    if route.final_ref and (folder / route.final_ref).exists():
        ref = Image.open(folder / route.final_ref)
        score, picture, matched = wait_for_screen(grab, ref, route.final_box,
                                                  route.min_similarity, step_timeout_s, alive)
        if not matched:
            actual = report_dir / "diverged-final.png"
            picture.save(actual)
            raise Diverged(f"final screen never matched (best similarity {score:.3f}); see "
                           f"{actual} vs {folder / route.final_ref}")
        final = round(score, 3)
    return {"steps": results, "final_similarity": final}
