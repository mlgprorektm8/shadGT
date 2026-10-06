# Gran Turismo Sport CUSA03220 implementation and validation

## Current status - October 5, 2026

Latest player confirmation: the frozen top strip is gone, in-race icons are white,
and map backgrounds are correct. The bottom in-race track HUD still renders cyan.
Recent race loading has been stable; long-term stability is not established.
The confirmed Debug build adds exact swizzled
source-alpha dual-source blending and a narrow read-only far-depth precision
workaround. Shader/cache versions are now binary 12, metadata 11, pipeline key 7.
The previous keep-destination-alpha branch did not match the captured blend and
has been replaced. The old bottom-origin strip pixel trace was incorrect.

Capture evidence: sky event 10614 at top-left pixel (200,100) maps z=0.99 through
viewport [0.9999995231628418,1.0] to shader depth 1.0, failing Less against clear
depth 1.0. The initial HDR target 16784 already contains engine artwork. Adjusting
the far endpoint to nextafter(1,0) in replay produces depth 0.9999999403953552,
passes the test, and replaces the stale background with sky. This workaround
changes endpoint semantics and is not exact hardware viewport emulation; it is
restricted to read-only float-depth quads without explicit depth exports, stencil,
depth bounds, or guest bias. Other scene defects remain in the old capture.
Results are under capture-far-depth-{experiment,pixel,final} and
capture-strip-sky-depth-range. The original captures are unchanged.

Debug build, CLI smoke check, diff check, and 69 focused regressions passed.
Source checkpoint: `8021b5f27b920a4f0c2357ee435089a1eec5c2a8`.
Clang Release now builds from the same renderer with /O2 /Ob2 /DNDEBUG and Tracy
disabled. Use `scripts/Run-GTSportPerformance.ps1` for the isolated profile without
RenderDoc, validation, dumps, or routine info logging. Graphics safety settings
are unchanged. Release speed and visual parity need a player race check.
See `GT_SPORT_BASELINE.md` for the separate regression reference and recovery.

The following sections preserve earlier investigation history, not current status.

The RenderDoc menu capture isolated both causes of the red map panel. The game
clears an `RGBA8` map target with packed value `0x000000ff` and a reversed shader
export mapping. Applying the export mapping to the already packed clear value moved
the physical alpha bit into the color channel, producing the solid red background.
Keeping the clear components in memory order removes that background in an edited
capture replay. The same replay showed valid cyan map geometry being rejected by
stencil because an emulated HTile metadata clear reset depth but preserved stale
stencil value 1. Replaying that clear over both valid aspects makes the geometry pass.

The user confirmed that the resulting October 5 Clang Debug build removes the red
track-layout background, but their first race test crashed. The crash log is in the
global AppData profile, not the isolated test profile: it records a guest `lock xadd`
write fault with `redzone_patches` disabled. The existing 1.69 boot patch was active.
This is the known Windows guest-red-zone failure mode and is separate from Vulkan.
`Build/Run-GT-Sport.ps1` now accepts `-BuildType Debug` so both configurations can
use the isolated profile, where red-zone patching is enabled. A fresh Debug rebuild
and visual race retest are pending below; neither the strip nor race stability is
claimed fixed yet.

The user confirmed that the October 4 22:26 full-image export build left both
artifacts unchanged. That run reached `race_gt_league` and ended at 22:32:30 with
no logged device loss or invalid T# warnings; validation was off. No new screenshots
were saved from it.

A bidirectional crop-copy experiment passed 45 focused tests and reached a race
at 03:12. Screenshots `CUSA03220_20261005_031320_528_game_000000.png` and
`CUSA03220_20261005_031324_078_game_000001.png` still show the frozen top strip
and both red map panels. The car appears overexposed compared with the October 4
baseline. The first 32 logged crop copies were all 960x200 -> 960x100 at
0x1009bc0000; they do not establish that the reverse path was exercised.

The experiment and its five tests have been backed out, with snapshots of the
five edited source/test files retained under `Build/gt-sport-crop-experiment-20261005`.
The shader-metadata and full-image buffer-preservation changes remain intact.
The baseline Release rebuild, CLI smoke check, and 40 focused tests pass.

Portable RenderDoc 1.46 is under `Build/tools/renderdoc-1.46/RenderDoc_1.46_64`.
The command-line tool and capture DLL have valid Authenticode signatures.
The next run will inject its capture DLL before Vulkan initialization; F12 should
then produce a GPU frame capture rather than a PNG screenshot. No global layer
registration or original profile configuration is being changed. Native UI
inspection remains unavailable (helper pipe error 2), so race navigation and
the capture key require the player. Neither artifact is claimed fixed.

### Full-image export - October 4, 2026, 22:26 Central

The 22:16 metadata-fix retest restored normal lighting according to the user. The
frozen top strip and red map panels remain. F12 screenshots at 22:20-22:21 show
solid red panels with car markers still drawn. The final log records 16 invalid T#
warnings in shader `f302c320`, but no device loss. Vulkan validation was off.

Another disconnected runtime path was found: buffer/image preservation tests used
`PlanImageAliasExports`, but the buffer cache still exported only same-base images
on formatted read-only requests. It never set `BufferCoherent` or tracked a full
preserved image through `MarkRegionAsGpuModified`. Those helpers now run before
raw and formatted buffer reads/writes. Entire eligible footprints become resident,
all mips are exported with padding preserved, and both buffer tracking and image
proof are updated. CPU shortcuts for small reads and DMA fills/copies cannot bypass
overlapping rendered images. Three new regressions cover render-after-export,
redundant export suppression, and evidence for buffer-authoritative dirty aliases.

Ambiguous competing image aliases remain guarded, including compatible crops with
separate rendered content. Unlike the earlier status claim below, the current planner
does not merge or sequentially export competing crops. Single-sample exact physical
formats without packed stencil are eligible. The Release build, CLI smoke check,
`git diff --check`, and 37 focused regressions pass. The new runtime behavior still
requires a race retest; passing helper tests alone does not establish correct visuals.

The 22:25:10 Release executable was launched at 22:26:14 (PID 26664) with the
existing version-10 cache retained and normal profile settings unchanged. The
previous log is saved as `log-before-full-image-export-20261004-222614.txt`. The
process is responding, and the log confirms actual full 1920x1080 image exports in
the new path. Race graphics and longer stability are still pending the player test.

### Metadata fix - October 4, 2026, evening

The October 4 22:09 fresh-cache retest used the 18:19 Release executable, readbacks
mode 0, and no Vulkan validation or shader dumps. The user reported lighting that
was too bright, a frozen top strip, and persistent red HUD/map squares. The run ended
at 22:10:56 with Vulkan device loss after 13 invalid T# warnings from shader
`aa3822a3`. The cache and log are being preserved before the next run.

Inspection found that the production pipeline cache still defined the old shared
`Program::info`, while the tests exercised a separate per-permutation container.
The renderer now includes that tested container. Live shader selection refreshes
and compares each module's own resource metadata; new permutations save and return
their own compilation metadata. Preload keeps the metadata at each original index
without replacing pointers already retained by pipelines. Metadata version 10 rejects
the previous cache entries. The shader-program test includes the production cache
header and checks newly active resource specialization. The Release build, CLI smoke
check, and 24 focused regressions pass. The subsequent player result is recorded above.

### Earlier status - October 4, 2026, 14:35 Central

The user's latest visual check before this build reported dark lighting, with the
frozen strip and red HUD squares unchanged. The Release build now accepts ordinary
`FormatInvalid` color-output placeholders and binds null descriptors for unsupported
T# image formats before image creation. It does not guess a Vulkan mapping for
`Format2_10_10_10`/`Ubint`. The build succeeded; a cache-disabled diagnostic passed
the old detiler fault point but was stopped before reaching the later format assertion.

With the previous isolated-profile cache enabled, startup faulted after preloading
924 pipelines and reporting 4,791 stale entries. I preserved that 525 MB cache at
`Build/gt-sport-fixed/user/cache/CUSA03220-pre-format-fix` and launched with a fresh
cache, normal speed, validation off, and readbacks mode 0. The emulator has remained
open for more than a minute past `StartProject:gtmode`, with no guest execution fault
or format assertion in the captured log so far. Visual confirmation from that run is
pending; this does not yet establish race stability or corrected graphics.

- Release build and CLI smoke check passed. The focused graphics regression set
  passed 50/50 tests, including compatible-crop export ordering and shared-memory
  indexing/barrier coverage.
- Shader permutations own their resource metadata; dynamic lighting data reads
  remain GPU-indexed. Auxiliary tessellation and depth attachment/copy fixes are
  included in the player-tested camera improvement build.
- Follow-up fixes preserve all four FMask read components, return valid FMask LOD
  values, order sampler LOD bounds, ignore array fields for non-array textures,
  skip empty buffer transfers, and resolve dependent host resource-table reads.
- Full unambiguous image footprints are preserved before partial buffer writes,
  including tiling padding. Verified same-base 2D crop aliases can be exported from
  older to newer content versions. Buffer-authoritative ranges and incompatible
  competitors still block export; multisampling and substituted formats remain
  unsupported. Interior writes invalidate refreshed aliases, and CPU/image writes
  and backing changes revoke the preservation proof.
- Buffers and DMA finish before texture refresh; targets and sampled/storage
  views refresh before rendering. Final layouts use each descriptor's physical
  Vulkan backing image and combine its shader and attachment access.
- Shader binary/metadata/pipeline cache versions are 11/9/6. Older binary entries
  are rejected; recompilation on first visit is expected.
- Visual comparisons use `Build/Run-GT-Sport.ps1` with no switches, which keeps
  Vulkan validation, shader dumps, and precise readbacks disabled while retaining
  pipeline caching. The current profile uses readbacks mode 0; a clean non-validation
  log still does not establish correct visuals.
- All 1,724 fresh shaders captured in that race passed offline Vulkan 1.3/scalar
  validation (`Build/validate-oct4-graphics-final-race.json`). Descriptor/layout/copy,
  sampler/module/interface VUIDs and SRT walker failures were absent. There were
  107 invalid T# warnings, all shader `aa3822a3` flattened offset 109, plus the
  image allocation assertion in `stdout-oct4-graphics-final.txt`.
- The race also found a new depth STORE synchronization fault. Its missing write
  access is corrected and all seven resource-binding tests pass again
  (`Build/test-oct4-depth-store.log`). The presentation synchronization warning
  remains; duplicate suppression prevents treating its count as a duration bound.

The current full-speed Release executable uses the restored image-version ordering
without runtime SRT readbacks. Run `Build/Run-GT-Sport.ps1` with no switches for the
normal-speed profile, then compare lighting, the top strip, and red map squares. Use
`scripts/Check-GTSportGraphics.ps1` to summarize later logs. Native desktop automation
is unavailable in this session, so the player provides the visual result.

## Earlier implementation and validation history

Tested game: update 1.69, emulator source revision 8e23388a with the local changes listed below.
The user confirmed that a campaign race runs and that rapidly cycling through camera views no
longer crashes with the latest build. The startup, command-buffer lifetime, and rendering fixes
below are implemented and built. Longer gameplay and error-free rendering are not established.

## Implemented

- Comparison-sampled textures get separate shader image bindings from ordinary sampled textures.
  Previously the same T# could be deduplicated into a color binding even when used by a comparison
  instruction. Validation caught an R8 color view being sampled with depth comparison in the race.
  Three regression tests cover comparison/ordinary separation, shared read/write usage, and mip
  fallback identity. Shader cache binary version is incremented to invalidate older bindings.
- Invalid/null texture slots reset their scratch descriptor before reuse and preserve the current
  shader's sampled/storage type. Validation had caught a sampled-image binding being pushed as
  STORAGE_IMAGE because an earlier draw left that type in the scratch slot.
- Successive pending layout transitions of the same image are flushed in order. Vulkan barriers
  in one dependency do not execute in sequence. Validation caught a depth image being transitioned
  from its intermediate sampled state without ordering the previous attachment store.
- GPU semaphore waits cover all command stages, including fragment sampling of a ready frame;
  the previous second wait was at color attachment output. The wait-stage array now covers the
  full supported semaphore count.
- Supported 8/16-bit uniform/storage buffer features are enabled on device creation. Validation
  previously rejected built-in shader modules requiring uniformAndStorageBuffer16BitAccess.
- CE/GFX packet advancement uses the packet size captured before execution/yielding. An indirect
  buffer can complete a fence and let the guest recycle its parent allocation before returning.
  Rereading the parent header then interpreted allocator pointers as counts/commands. EOS/EOP and
  compute RELEASE_MEM fence payloads are also copied before publishing completion.
- DeviceService initialization, termination, event polling, and empty peripheral enumeration.
  Enumeration initializes output counts and validates arguments. This backend exposes no Mbus
  peripherals; ordinary SDL controllers continue through libScePad.
- GCN stencil Ones selects reference 0xff for Vulkan Replace. Mask equivalence and reachable
  stencil/depth operations determine whether the translation is exact. Conflicting references
  still produce a warning; this is not a full implementation of arbitrary stencil operations.
- Buffer descriptor sizes and dword counts use 64-bit arithmetic. Previously, 64-byte stride
  times 0x04000000 records overflowed to zero, and raw 0xffffffff-byte dword rounding overflowed.
  Vertex ranges are clamped before merging and host bindings stay within the cached buffer.
  Maximum-record storage descriptors retain mapped-memory clamping without the previous error
  message. Other clamping diagnostics remain and now include the descriptor address and fields.
- PM4 type-0 consecutive register writes, with payload/register range validation and compute
  register synchronization. Invalid packets still stop execution, with surrounding command words
  logged for investigation. Packet layout follows the
  [AMD SI programming guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/programmer-references/si_programming_guide_v2.pdf), section 2.1.1.
- sceKernelGetOpenPsId returns a complete virtual console identity persisted in the profile's
  openpsid.bin, validates null output, and preserves output on failure.
- sceKernelGetSanitizerNewReplaceExternal returns null for the absent guest sanitizer runtime.
- Windows unhandled faults log the accessed address, registers, and faulting instruction.

## Validation

- Release built locally with Clang/Ninja; CLI --help runs.
- 30 targeted GTests passed: stencil translation, DeviceService lifecycle/output behavior,
  PM4 packet parsing/bounds, OpenPSID persistence/error handling, and buffer-size overflow.
  The counter tests cover prefetch limits, release when DE catches up, and counter wraparound.
  REWIND tests cover publication after a snapshot, tail refresh, live readiness checks, and bounds.
- The race retest after image-binding separation no longer reports invalid depth comparison on
  R8 textures. The next race run after ordered transitions no longer reports the prior depth-image
  WRITE_AFTER_WRITE hazards. These runs do not establish that every rendering error is resolved.
- The initial diagnostic run used validation and the crash layer's forced barriers. This slowed
  menus significantly. Forced barriers and the crash layer were subsequently disabled. Normal
  use should also disable validation. Pipeline caching is enabled in the isolated profile.
- git diff --check passed. compile_commands.json was refreshed for IDE indexing.
- Game tests use copies of saves/cache in Build/gt-sport-fixed/user. The original saves and
  user configuration were not edited.
- The isolated profile uses the existing enabled "Fix 1.69 update boot crash" patch copied from
  the user's patch directory. It also enables Windows red-zone protection. Without that existing
  boot patch, the game fails with the ProductBootScreen.ad:99 ADHOC nil error.
- With the boot patch, startup completes and shaders compile/render. The DeviceService,
  OpenPSID, sanitizer lookup, and stencil-op-2 stub/unsupported messages disappear in the fresh log.

## Remaining failures and dependencies

- WAIT_ON_DE_COUNTER_DIFF had its subtraction
  reversed (DE minus CE), causing unsigned underflow when CE leads DE. The difference now uses
  CE minus DE, as specified in the AMD SI programming guide, section 2.3.7. The build also logs
  GPU memory, video-output label, semaphore, REWIND, and CE/DE waits that exceed five seconds.
  The later test passed the menu and loaded a campaign race, confirmed by the user.
- The new wait diagnostics identified a persistent REWIND wait in the copied command buffer.
  Copying command buffers captured an unset readiness bit, which never changed when the guest
  published the original buffer. REWIND now polls the original buffer with an acquire load and
  refreshes the copied tail after publication. The same handling also polls direct/indirect live
  buffers. The user confirmed a campaign race ran after this fix.

- An unattended test stopped on an invalid PM4 type-0 packet: header 0x00c00060 declares 193 data
  dwords, but only two remain. This is malformed command data, not a missing valid packet decoder.
  Submission lifetime, GPU writes, indirect-buffer control flow, and guest corruption require
  further diagnosis. The latest experiment enables the existing copy_gpu_buffers option to test
  command-buffer lifetime; that option is not yet established as a fix.
  Two copy-buffer runs passed startup and ended without a logged critical exception, but did not
  establish successful race entry or a reliable fix. Their process exit status was unavailable.
- The original access violation at eboot offset 0x1cccdc8 occurred while opening a race, according
  to the user. The user subsequently confirmed a successful race-entry run. The original log lacks the
  accessed address/registers; new diagnostics capture these if the failure repeats.
- The camera-switch test ended with Vulkan device loss on the RTX 4070 Ti. The latest fixes
  remove specific observed rendering errors; the user confirmed that rapid camera switching
  no longer crashes. A WRITE_AFTER_PRESENT diagnostic still appears under synchronization
  validation, as does a tessellation interface mismatch in the menu. Some shader-resource-table
  offsets involving Phi/GetAttributeU32 remain unsupported.
- The next synchronization test later stopped on recycled indirect-buffer data at 0x202800040.
  This is preserved in log-camera-malformed-packet.txt. The captured-size/fence-payload changes
  address this lifetime hazard. The normal-performance retest reached race_gt_league and
  QuickRoot::onInitializeEnd without an immediate malformed-packet error, but subsequently lost
  the Vulkan device. This later failure is saved in log-device-lost-after-lifetime-fix.txt.
  After the null-descriptor fix, another validation run reached the race without the previous
  descriptorType mismatch, depth-image barrier hazards, or immediate device loss. Repeated camera
  cycling was subsequently confirmed working by the user. Longer driving stability remains untested.
- The null-binding validation run stayed active in the race for several minutes without those
  failures and was closed normally (exit code 0). Its log is log-null-binding-validation.txt.
  The latest run uses normal settings: validation and crash diagnostics disabled, pipeline cache
  enabled. The user confirmed that its rapid camera-switch retest no longer crashes.
- The final normal run reached campaign race loading (race_gt_league, QuickRoot::onInitializeEnd)
  and remained active for several minutes without a new device-loss, malformed-packet, or GPU-wait
  stall diagnostic. At the memory check, total GPU memory use was 4284 MiB of 12282 MiB. The exact
  rapid camera-button sequence was not independently automated; the user tested it and confirmed
  that the crash no longer occurs.
- SST-Roman.otf and SST-Bold.otf are missing from the dumped PS4 font directories. These assets
  cannot be implemented as emulator code. A local path to the user's dumped fonts is needed.
- BluetoothHid initialization/callback/device registration and thread atexit registration hooks
  still resolve to generic stubs. Other unrelated API/metadata diagnostics also remain.
- Trophy extraction requires the user's trophy decryption key; this was already missing in the
  original log. No key was generated, downloaded, or invented.

## Run the current build for race testing

From PowerShell:

```powershell
& 'D:/Development/shadPS4/Build/Run-GT-Sport.ps1'
```

This selects the isolated profile and the base game path; the emulator loads the installed 1.69
update automatically. The launcher disables validation/shader dumps for normal use and enables
pipeline caching. Add `-Diagnostic` to enable validation/dumps until the emulator closes.
The launch script disables implicit Vulkan overlay layers only for this process and restores the
caller's environment afterward. The first visit to a view can still compile shaders; the cache is
invalidated once because older image bindings are incompatible with this build.
Open the same race that previously crashed. The new log is at
Build/gt-sport-fixed/user/log/shad_log.txt. Build/gt-sport-fixed/log-malformed-command.txt preserves
the prior malformed-packet run. log-camera-device-lost.txt preserves the first camera-switch
failure; log-camera-validation-before.txt and log-camera-before-sync-fix.txt preserve the
Vulkan validation runs before the additional rendering fixes.

Both manual CMake builds and GitHub Actions builds are supported by this repository. The binary
and validation in this report are from a local manual build; no GitHub Action was triggered.
