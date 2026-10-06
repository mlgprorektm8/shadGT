# GT Sport Regression Reference

Saved October 5, 2026, for CUSA03220 update 1.69.
Source: `8021b5f27b920a4f0c2357ee435089a1eec5c2a8`.
Tag: `gt-sport-graphics-baseline-20261005`.

The player confirmed the Debug build removes the frozen top strip, restores white
in-race icons, and fixes the red map backgrounds. The bottom in-race track HUD
remains cyan. Race loading has been stable in recent checks; long sessions are
not proven. Release is optimized from the same source with Clang `/O2 /Ob2
/DNDEBUG` and Tracy disabled. Its speed and graphics parity still need a race test.

## Contents

- `binaries/Debug`: exact player-confirmed executable and symbols.
- `binaries/Release`: optimized executable, with any runtime libraries present.
- `source.bundle`: standalone Git history and annotated reference tag.
- `source-8021b5f2.zip`: source tree at the reference commit.
- `dependency-source.zip`: dependency sources, including submodule contents.
- `submodules.txt`: dependency revision identifiers.
- `profile-and-investigations`: closed profile, saves, caches, captures, replay
  experiments, logs, and the focused regression test report.
- `capture-tools`: local capture inspection and RenderDoc replay tools.
- `build-settings`: configuration caches and compilation commands for both builds.
- `launchers-and-notes`: performance/capture launchers and investigation notes.
- `local-notes.patch`: tracked documentation changes made after the source commit.
- `manifest.json`: file sizes, SHA-256 hashes, confirmed results, and limitations.

This copy is private. Do not upload saves, game-derived captures, or configuration.
The game/update files on E: and dumped system modules under
`D:/Emulators/shadPS4/sys_modules` are external prerequisites and are not copied.

## Binary Comparison

Keep this directory unchanged. Copy a binary directory to a separate working
directory before use. Run `launchers-and-notes/Run-GTSportPerformance.ps1` with
explicit `-BuildDirectory`, `-ProfileDirectory`, and `-GamePath` paths. Its defaults
are for the original development repository, not this archive.

Use a current working profile to avoid rolling saves back. If you copy the archived
profile, update the absolute `General.home_dir` and `font_dir` fields in
`user/config.json` to that working copy. Preserve red-zone protection, patches,
readbacks, resolution, and buffer copying. Compare the same race/camera/settings.

## Source Recovery

Restore to a new directory, never over an existing dirty working tree:

```powershell
git clone --branch gt-sport-graphics-baseline-20261005 ./source.bundle D:/Development/shadPS4-reference-work
Expand-Archive -LiteralPath ./dependency-source.zip -DestinationPath D:/Development/shadPS4-reference-work/externals -Force
```

The dependency archive provides the pinned files without requiring downloads.
Use LLVM 22 clang-cl, CMake/Ninja, and the Visual Studio x64 developer environment.
The archived `build-settings/Release/CMakeCache.txt` and `compile_commands.json`
record the original configuration. Select Release and set `TRACY_ENABLE=OFF`.
Do not blindly reuse the old cache: it contains absolute paths.

`manifest.json` hashes all saved files except the manifest itself. Verify hashes
before using this copy to diagnose a regression. The source bundle was verified
with `git bundle verify`; focused renderer regressions are recorded in
`profile-and-investigations/test-reference-focused.log`.
