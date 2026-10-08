# GT Sport Graphics Baseline

## Current baseline: October 7, 2026 (read-ahead pipeline builds)

Set as the baseline by the player on October 7, 2026 (CUSA03220 update 1.69), after the
run below; graphics reported fine on the read-ahead builds. Commit `6219309e`, tag
`gt-sport-baseline-20261007-readahead`.

Measured (log `Build/gt-sport-fixed/runs/20261007-220322-6219309e.log`, 824 s): average
42.6 FPS, 10th percentile 25.5, 90th 60.0, while building 947 pipelines the cache did not
have yet. 1,171 of them were built ahead of their draw, 81 on demand; draws waited over
50 ms 233 times (worst 187 ms), mostly in the race preview. Read-ahead cost 14 s of the
command thread's 824 s.

Changes since the performance baseline below:

- Car thumbnails no longer stop the game with "BREAK! thumbnail_functions.ad:473" (FIX-012:
  the shader cache stores attribute loads/stores; linear images up to 256 KB are read back).
  Cache entries stored before FIX-012 still load.
- All user data stays in the `user` folder beside the executable on Windows;
  `scripts/Make-GTSportPortable.ps1` packages a portable folder.
- PERF-017: runtime pipelines are fast-linked from stage libraries; an optimized link
  replaces them later (PERF-021: only after 3 s without new pipelines).
- PERF-018: stage libraries shared between pipelines, the two shader libraries compiled at
  the same time.
- PERF-019 to PERF-023: on a pipeline miss (and, for 2 s after one, at the start of every
  command buffer) the command thread reads the following commands with a copy of the
  registers and queues every new pipeline on build workers (all cores but two). Draws use
  exactly the pipeline their real registers select; a wrong guess only costs a build.
- DIAG-013 compile timing and DIAG-026 (names a shader the translator rejects) in the log.

`-DisablePerf <ids>` switches any of PERF-017 to PERF-023 off. Reference copy:
`D:/Development/shadPS4-regression-baseline`, with a portable folder.

Still open: 60 FPS in races (the command thread spends about 14 ms per frame on draw setup
for about 1,450 draws; the GPU is busy about 35%).

## Previous baseline: October 7, 2026 (performance)

The player called this build "fantastic" on October 7, 2026 (CUSA03220 update 1.69): a
solid 30+ FPS throughout races and no exploded vertices when sparks fly. Lighting, sparks
and the leaderboard were last reported fine on the PERF-012 build ("just as stable as
before"); this build was not separately rechecked for them.

Measured in that race run (log `Build/gt-sport-fixed/runs/20261007-120935-6df44eb3.log`):
average 44.6 FPS, 10th percentile 29.5, 90th 60.5 (2-second windows at 5+ FPS). The
October 6 baseline averaged about 26 in races with dips to 11.

Changes since October 6 (details in `GT_SPORT_OPTIMIZATION_LOG.md`):

- PERF-009: GDS-to-memory copies read back asynchronously.
- PERF-012: pages the CPU rewrites constantly stay unprotected and are uploaded once per
  upload epoch (buffer binding about 25 -> 4 ms per frame).
- PERF-013: runtime shader permutations no longer overwrite stored ones (about 1,085 cached
  pipelines had been rejected and recompiled in every race).
- PERF-014: GPU-side waits proceed behind fences the command thread already processed.
- PERF-016: submits every 128 draws while CPU-access drains keep happening, so each drain
  waits for little (about 250 per 2 s costing about 120 ms, down from 30-40 costing up
  to 750 ms).
- Off: PERF-011 and PERF-015 (both produced exploded vertices; PERF-015 with sparks).
- Diagnostics in the build: performance monitor, frontend work and draw-step timing,
  wait and drain reports, every 2 s at warning level. Each run's log and summary are kept
  in `Build/gt-sport-fixed/runs`.

Tag: `gt-sport-baseline-20261007`. Release build, `Build/gt-sport-fixed` profile
(`readbacks_mode` 1 and `readback_linear_images_enabled` true are now the profile's own
settings), launched with `scripts/Run-GTSportPerformance.ps1` and no extra flags.
`-DisablePerf <ids>` switches individual changes off for comparisons. Reference copy:
`D:/Development/shadPS4-regression-baselines/GT-Sport-20261007`.

Still open: the items listed under the October 6 baseline below, and 60 FPS in races
(heavy stretches are now about 30; the command thread's per-draw work is the limit).

## Previous baseline: October 6, 2026

The player called this "the most accurate it's ever been" on October 6, 2026
(CUSA03220 update 1.69), after these changes (details in `GT_SPORT_OPTIMIZATION_LOG.md`):

- FIX-009: cropped render targets are written back into their full image (car on the
  loading and transmission screens).
- FIX-010: render targets on a mip level keep that mip when other images overlap
  (rental-car top-left square fixed, player confirmed).
- ACC-007: `V_CVT_PKRTZ_F16_F32` rounds toward zero (finite overflow is 65504, not Inf).
- DIAG-020/021 diagnostics are still in the build (log only; remove before the next
  performance work).
- Shader cache rebuilt from empty on October 6 (old cache kept as
  `user/cache/CUSA03220-before-cache-reset-20261006`).

Tag: `gt-sport-graphics-baseline-20261006`. Release build, `Build/gt-sport-fixed` profile,
launched with `scripts/Run-GTSportPerformance.ps1 -ReadbacksMode 1` (player confirmed;
the saved profile itself keeps `readbacks_mode` 0, so pass the flag). Reference copy:
`D:/Development/shadPS4-regression-baselines/GT-Sport-20261006` (Release binary,
profile config and a source bundle; no saves or captures).

Still open at this baseline: anti-aliasing history on the transmission screen,
"R" drawn as "9" in some headings, and the race-preview flicker (not rechecked on the
clean cache).

## Older baseline: October 5, 2026

Confirmed by the player on October 5, 2026, with CUSA03220 update 1.69:

- The frozen/glitchy top strip is gone.
- Track icons are white in the race.
- Track-map backgrounds no longer have the solid red panels.
- The bottom in-race track HUD remains cyan. Do not record it as fixed.
- Race loading has been stable in the latest checks, not proven over long sessions.

Reference source commit: `8021b5f27b920a4f0c2357ee435089a1eec5c2a8`.
The confirmed executable is the Clang Debug build; the optimized Clang Release
build uses the same renderer. Release speed and visual parity need a player check.

## Performance Run

```powershell
& ./scripts/Run-GTSportPerformance.ps1
```

This uses the isolated `Build/gt-sport-fixed` profile and enables the FPS counter.
It disables RenderDoc, validation, shader dumps, Tracy (Release build), and routine
info logging. Warnings and errors remain enabled. Temporary diagnostic settings
are restored when the game exits normally. Resolution, readbacks, buffer copying,
red-zone protection, installed patches, and saves are not changed.

Use `-FullLogging` for the profile's original log filter. `-CheckOnly` validates
the paths and required red-zone setting without changing files or launching.

## Separate Reference Copy

The reference copy is stored outside this repository at:

`D:/Development/shadPS4-regression-baselines/GT-Sport-20261005-8021b5f2`

It contains the confirmed Debug executable and symbols, optimized Release binary,
source archive/Git bundle, dependency source archive, closed isolated profile,
shader/cache data, captures, local replay investigations, launcher scripts, build
settings, test results, and SHA-256 file records. The manifest records exact scope.
This is a private local backup: the profile includes saves and local configuration.
Do not upload the profile, game-derived captures, or other private assets.

Keep this copy unchanged. Test future changes against a working copy, using the
same race, camera, lighting, and settings. Never overwrite current saves with old
reference saves just to compare rendering.

## Recovery

Open `README.md` and `manifest.json` in the reference directory first. The source
tag `gt-sport-graphics-baseline-20261005` identifies the checkpoint locally.
The standalone Git bundle preserves that tag even if this repository is changed.
Restore source into a new directory, not over a dirty working tree. The dependency
archive supplies the pinned library source without needing a fresh download.

For a binary comparison, copy the reference binary directory elsewhere and use
the launcher's `-BuildDirectory` option with the current isolated profile. To use
a copied reference profile, update its absolute `General.home_dir` and `font_dir`
paths to that working copy before launch. The external dumped system-module and
game paths must still exist; those game/system files are not duplicated here.
