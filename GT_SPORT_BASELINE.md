# GT Sport Graphics Baseline

## Current baseline: October 6, 2026

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

## Previous baseline: October 5, 2026

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
