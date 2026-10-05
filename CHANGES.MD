# Gran Turismo Sport compatibility changes

Updated: October 5, 2026. Original base revision: `8e23388af83fe31ec6372a29e3940e2b459dfc6d`.
October 4 work started from local revision `014635d7` with the preceding graphics edits present.
Tested title: Gran Turismo Sport, CUSA03220, update 1.69.

Latest user retest: map backgrounds are correct, track outlines and icons remain
cyan, and the frozen top strip remains. Race loading has not crashed in that test;
long-term stability is still unconfirmed.

The follow-up capture inspection corrected the earlier map-blend diagnosis. The
map uses alpha Zero / OneMinusSrcAlpha, not destination-alpha preservation. Its
reversed export mapping puts logical red in physical alpha, so applying the native
alpha equation removes red. The new path exports logical source alpha separately
and uses dual-source factors for the exact reversed-map blend equation. It is
limited to attachment 0 with no other enabled color targets or guest MRT1 export,
and requires dual-source support. Shader/cache versions reject the old interface.

The top-strip trace also corrected a bottom-origin coordinate assumption. At the
actual top pixel (200,100), old engine artwork is already in the initial scene
target. The sky draw's z=0.99 maps through [0.9999995231628418,1.0] to rounded
depth 1.0 and fails the strict Less test. A capture replay with the far endpoint
one float32 step below 1.0 makes the sky pass and replaces that stale background
with clouds. The runtime precision workaround is limited to very narrow far-depth,
read-only, strict-Less floating-depth quad passes without stencil, depth bounds,
explicit fragment depth, or polygon bias. This is a scoped workaround, not exact
emulation of guest viewport precision. Other captured scene artifacts remain.

The follow-up Clang Debug build, CLI smoke check, and 69 selected regressions pass,
including logical/physical blend equations and synthetic SPIR-V output checks.
Both changes still require a fresh in-game visual check. The capture launcher now
accepts Debug, retaining the isolated profile and its enabled red-zone protection.

The October 5 GPU capture traced the solid red map background to color fast-clear
channel handling. `CLEAR_WORD` is already packed in surface-memory order, so it must
not be remapped by the component swap used for shader exports. A replay with that
mapping removed eliminates the red background. The same capture found valid map
geometry rejected by stale stencil value 1 after an emulated HTile metadata clear;
metadata clears now clear every valid depth/stencil aspect. The user confirmed the
red background is gone in the first Clang Debug build. That direct-launch test then
hit a guest `lock xadd` memory fault because its global profile had Windows guest
red-zone patching disabled. The GT Sport launcher can now select Debug while retaining
the isolated profile's enabled red-zone workaround.

Campaign races now load in the user's test. The user also confirmed that rapidly
cycling through all camera views no longer crashes. These results do not establish
long-term gameplay stability or error-free rendering. In the October 4 race retest,
the user confirmed that camera darkness is largely fixed in all cameras. The frozen
top strip and red track maps remain in that tested build.
The next October 4 retest produced a much larger lighting improvement, confirmed
by the user. The strip remains, with the top mirror now visible, and red squares
remain. That run later stopped on an image allocation assertion.
The exact-range SRT readback trial remained too bright or closed before it could be
judged, and the red squares did not improve. The log shows invalid T# values in shader
`aa3822a3` followed by a `SurfaceFormat` assertion. Runtime SRT readback has been
removed after two unchanged visual trials; dynamic descriptor-index traversal stays.
The current build accepts `FormatInvalid` color-output placeholders and binds a null
image descriptor for unsupported T# surface formats instead of asserting. The invalid
`Format2_10_10_10`/`Ubint` pair is not mapped to a guessed Vulkan format. The Release
build passed. A cache-disabled diagnostic passed the earlier detiler fault point, but
was stopped before reaching the later format assertion, so that runtime path is not
yet confirmed.
The cached launch with the old isolated-profile cache faulted after preloading 924
pipelines and reporting 4,791 stale entries. That 525 MB cache was preserved under
`Build/gt-sport-fixed/user/cache/CUSA03220-pre-format-fix`; the fresh-cache launch has
remained open past the prior fault point. The user's last visual check still reported
dark lighting with the strip and red squares unchanged. In the October 4 22:09
fresh-cache retest, the user reported lighting that was too bright, with the frozen
strip and red squares still present. That run ended at 22:10:56 in Vulkan device loss
after 13 invalid texture-descriptor warnings; its log is preserved separately.

The follow-up found that the tested per-permutation `Program` container was no longer
used by the renderer's pipeline cache. Live compilation still returned the original
permutation's resource metadata, and preload shared or replaced that metadata across
modules. Both paths now use the tested container, retain each module's own stable
metadata, and serialize the metadata produced by that module's compilation. Shader
metadata version 10 rejects entries from the previous behavior. The shader-program
test now includes the production pipeline-cache header, and a new regression checks
that newly active textures must match the compiled specialization. Release build,
CLI smoke check, and 24 focused regressions pass. In the 22:16 retest, the user
confirmed normal lighting, but the frozen strip and red maps remained. F12 screenshots
confirm the solid red map panels. The final log has 16 invalid T# warnings in shader
`f302c320` and no recorded device loss; validation was disabled.

The buffer cache also still used its old same-base image export path, bypassing the
tested full-footprint preservation planner. Buffer reads and writes now invoke that
planner before CPU shortcuts, export all mips with tiling padding preserved, track
the entire exported range, and set the existing `BufferCoherent` proof. Interior GPU
writes can then invalidate and refresh the image without losing untouched pixels.
DMA fills and copies check overlapping rendered images before choosing host memory.
Previously preserved buffer data does not block a newer render; competing image or
buffer-authoritative aliases still block ambiguous exports. Multisampled, substituted
format, and packed-stencil exports are excluded. Release build, CLI smoke check, and
37 focused regressions pass. The user subsequently confirmed that both artifacts
remained unchanged. The October 4 22:26 run reached a race and recorded no device
loss or invalid T# warnings; validation was disabled.

On October 5, a bidirectional crop-copy experiment passed 45 focused tests but left
both artifacts visible in the 03:13 race screenshots. The car also appeared
overexposed. Only larger-to-smaller copies were observed in the bounded diagnostic
log, so the unobserved reverse path is not established as the cause of either issue.
The experiment and its five tests were backed out; a local snapshot remains under
`Build/gt-sport-crop-experiment-20261005`. The previous shader and full-image buffer
preservation changes remain. The baseline Release rebuild, CLI smoke check, and
40 focused tests pass. A portable RenderDoc launch is being prepared to inspect
the failed draws and their inputs; neither artifact is claimed fixed.

## October 4 graphics work

- Shader permutations own stable, separate resource metadata. Live compilation and
  cache preload select the metadata for the actual module, including additional
  flat-buffer bindings and mip fallback descriptors. Resource-count comparisons
  reject incompatible permutations before indexing their lists.
- Dynamic scalar data reads with a known user-data pointer now keep workgroup/loop
  indices on the GPU instead of substituting the first flat-buffer entry. Descriptor
  producers and their dependencies remain on the host resource-table path. The
  captured lighting shaders use this path; the camera improvement is player-confirmed.
- Auxiliary rectangle/quad tessellation interfaces carry only fragment inputs
  actually read, preserve export widths, and handle missing exports without declaring
  nonexistent vertex inputs. Clip-distance transport remains separate from ordinary
  fragment parameters and render-target layer transport.
- Depth attachments derive their format from valid backing addresses, match sampled
  feedback layouts and aspect masks, and use an intermediate buffer for incompatible
  depth format copies. Depth-plane transfer sizes exclude packed stencil bytes.
- The validation run loaded the Khronos layer, entered Sunday Cup Race 1 at Northern
  Isle Speedway, and continued without the previously observed descriptor-layout,
  depth attachment/layout/copy, or auxiliary interface validation errors. Longer
  running revealed further resource-table, sampler, and shader-module issues being
  addressed in the next build. Presentation synchronization warnings also remain.
- The first October 4 build passed 50 targeted tests. All 1,325 shader modules captured
  at the first offline check passed Vulkan 1.3 validation with scalar block layout;
  this check preceded the later invalid FMask permutation and is not a blanket
  claim about every subsequently generated module.
- Follow-up fixes make non-array textures ignore array depth fields, skip empty GPU
  buffer transfers, preserve the four-component FMask read result, and provide a
  valid floating-point FMask LOD result. Vulkan sampler LOD bounds remain ordered.
- The host resource-table walker retries reads after their index dependencies are
  fetched. Three regressions execute the generated walker, including the nested
  sibling-table dependency found in shader `aa3822a3` and a changed-index rerun.
  Truly GPU-dependent descriptor offsets still report unresolved groups.
- The SRT walker now calls a runtime synchronization hook before dereferencing nested
  guest pointers and descriptor-table entries. BufferCache downloads only when the
  accessed range is GPU-modified; repeated reads in its 512 KB download window are
  coalesced for that refresh. A generated-walker regression mutates the index at the
  callback boundary and confirms the fresh descriptor is flattened. Metadata version
  advances to 9 to reject cached walkers with the previous calling convention.
- Buffer/image alias preservation exports the full unambiguous single-sample image
  footprint, makes its buffer pages resident, preserves tiling padding, and tracks
  untouched pixels before partial writes. Competing aliases without known write
  order, multisampling, and substituted physical formats are skipped with bounded
  diagnostics; this does not solve every alias combination.
- Shader binary cache version 11 rejects earlier compiled FMask/resource-table
  behavior. Metadata and pipeline key versions are now 9 and 6.
- All stage buffers and DMA uploads now finish before texture and attachment
  refresh. Final descriptors use the actual Vulkan backing image and a shared
  layout/access state for its sampled, storage, and attachment uses.
- A full-image buffer-preservation flag permits interior shader writes to
  invalidate a refreshed texture. New CPU/image writes and backing changes revoke
  that proof; unrelated clean aliases retain the conservative invalidation guard.
- The combined Release build, CLI smoke check, and 85 focused regression tests
  passed. This build's frozen-strip and map improvements still require a race
  retest. The diagnostic launcher restores normal settings after emulator exit.
- Follow-up tracing found that same-address cropped 2D aliases could keep sampling
  stale pixels after a larger backing image was rendered. Image contents now carry
  monotonic versions, and compatible crop images are refreshed from a newer safe
  backing. The user reports the campaign race list background and race-start icon
  improved and the top mirror HUD became visible; the frozen strip and red map squares
  remained. The user later confirmed normal lighting after the shared-memory fix.
  A follow-up change that delayed content-version increments until after `UpdateImage`
  made the lighting brighter in the user's normal-speed check and left the strip and
  map unchanged. The experiment was reverted; the Release executable was rebuilt with
  the prior ordering. The experiment later ended in Vulkan device loss during submit.
- The follow-up race captured 1,724 fresh SPIR-V modules; all passed Vulkan 1.3
  validation with scalar block layout. It logged no earlier sampler, shader-module,
  descriptor/layout/copy VUIDs or resource-table walker failures. Invalid conditional
  texture entries and the image allocation assertion still need diagnosis.
- A new read-only depth attachment synchronization fault came from omitting the
  end-of-rendering STORE write in final access tracking. Final attachment access
  now retains that write; seven updated resource-binding regressions pass.
- Shared memory lowered to storage buffers now uses element indices scaled from byte
  addresses (16-, 32-, and 64-bit accesses), and control barriers include UniformMemory
  semantics when this lowering is active. The player says lighting looks normalized
  in the standard-speed build after this change; two shared-memory regressions pass.
- Buffer/image preservation now recognizes only verified same-base, same-layout 2D
  crops as sequentially exportable. Their full footprints are written oldest-to-newest
  by contents version; an overlapping buffer-authoritative range or incompatible
  image still blocks export. Two alias-ordering regressions were added.
- The combined release build and CLI smoke check pass, and all 50 focused graphics
  regressions pass. The current full-speed race run uses readbacks mode 0, no shader
  dumps, and no Vulkan validation. Its visual effect on the frozen strip and red map
  squares is pending; clean logs do not prove correct graphics.

## GPU command processing and race loading

- Corrected `WAIT_ON_DE_COUNTER_DIFF` to use CE minus DE, preventing an unsigned
  underflow that could leave the constant engine waiting indefinitely.
- REWIND now polls the readiness bit in the original guest command buffer with an
  acquire load. Once published, it refreshes the remaining copied commands. A
  snapshot taken before publication previously waited on a permanently stale bit.
- Added PM4 type-0 consecutive register writes with payload/register bounds checks
  and synchronization of overlapping compute registers. Malformed packets still
  fail validation rather than being silently skipped.
- Graphics and constant-engine packet advancement uses the size captured before
  yielding or executing nested buffers. Fence payloads are copied before publishing
  completion, so guest reuse of the allocation cannot change data still being read.
- Added diagnostics for GPU waits exceeding five seconds and surrounding command
  words when packet bounds are invalid. These diagnostics do not bypass waits.

## Rendering and camera switching

- Separated comparison-sampled image bindings from ordinary sampling of the same
  guest texture descriptor, avoiding an invalid color-image depth comparison.
- Reset null/invalid texture binding descriptors before reuse and select the current
  shader's sampled/storage descriptor type. Previously a scratch slot could retain
  a storage-image type from an earlier draw.
- Flush an earlier pending transition of an image before queuing its next transition,
  preserving the ordering of attachment writes and subsequent shader reads.
- Cover every supported wait semaphore with an all-command-stage wait, including
  fragment sampling of a completed frame.
- Enable supported 8-bit and 16-bit uniform/storage buffer access features during
  Vulkan device creation.
- Incremented the shader binary cache version from 6 to 7 to invalidate incompatible
  image bindings. The first visit to a view can compile shaders again.
- Translate GCN stencil Ones using Vulkan Replace with reference `0xff`, accounting
  for masks and reachable stencil/depth operations. Unrepresentable combinations
  still report a warning.
- Use 64-bit buffer size/dword calculations and clamp vertex ranges before merging.
  Host bindings remain within the cached allocation. Open-ended maximum-record
  descriptors retain mapped-memory clamping; other clamp diagnostics remain.

## Guest APIs and fault diagnostics

- Implemented DeviceService initialization, termination, event polling, and empty
  Mbus peripheral enumeration with argument validation and initialized output counts.
  SDL controllers continue through the existing Pad backend.
- Implemented `sceKernelGetOpenPsId` with a persistent 16-byte virtual console identity
  in the user profile. It validates output pointers and leaves output unchanged on
  failure. Moved the shared kernel UUID declaration into `uuid.h`.
- Registered `sceKernelGetSanitizerNewReplaceExternal` to return null when the guest
  sanitizer runtime is absent.
- Windows unhandled faults now report the accessed address, operation, CPU registers,
  flags, and faulting instruction when it can be read safely.
- Registered the new guest APIs and sources in the emulator's CMake build.

## Regression coverage and validation

Added eight test files and corresponding CMake targets, covering 30 cases:

| Area | Cases |
| --- | ---: |
| Stencil reference translation | 8 |
| DeviceService lifecycle and query outputs | 4 |
| PM4 type-0 parsing and bounds | 3 |
| OpenPSID persistence and failures | 4 |
| Buffer size overflow | 2 |
| Image binding identity | 3 |
| CE/DE counter waits and wraparound | 3 |
| REWIND publication, tail refresh, and bounds | 3 |

- All 30 targeted tests passed again during the repository cleanup.
- A fresh Clang/Ninja Release build passed after repairing submodule metadata.
  The CLI `--help` smoke check also passed.
- In-game validation runs reached a campaign race after the rendering fixes. The
  latest null-binding validation run remained active for several minutes and closed
  normally with exit code 0.
- The user confirmed successful campaign loading and rapid camera switching in the
  normal-performance build. The camera-button sequence was tested by the user.
- Normal test-profile settings disable validation/crash diagnostics and enable
  pipeline caching. Diagnostic forced barriers had significantly slowed menus.

The targeted tests can be rerun from the repository root:

```powershell
ctest --test-dir Build/x64-Clang-Tests -R 'StencilReference|DeviceServiceTest|PM4Type0|OpenPsIdTest|BufferResource|CeDeCounter|PM4Rewind|ImageResource' --output-on-failure
```

Both manual CMake builds and `.github/workflows/build.yml` are supported. Validation
here used local manual builds; no GitHub Actions workflow was triggered.

## Repository cleanup

- Existing ignore rules already exclude `Build/`, logs, IDE caches, and the root
  `compile_commands.json`. Added alternate `build-*`/`build_*` directories, the local
  `Install/` directory, CMake/Ninja generated files, CTest output, root emulator
  `user/` profiles, and crash dumps.
- Build binaries, downloaded SDKs, diagnostic logs, and isolated game profiles stay
  local under `Build/`. No saves, game assets, or original user configuration were
  deleted or edited for this cleanup.
- Missing submodule Git metadata was discovered while checking status and rebuilding.
  Restored and verified all 54 submodule repositories at their pinned revisions
  without checking out or overwriting dependency working files. Plain `git status`
  now works and reports no submodule changes. This is a local repair, not a source change.
- Verified ignore rules with `git check-ignore`; source files, regression tests, and
  this summary remain visible to Git. `git diff --check` passed.
- `CHANGES.MD` preserves this change summary outside the ignored build directory.
  More detailed local investigation notes remain in `Build/GT_SPORT_STATUS.md`.

## Remaining limitations

### October 3 graphics follow-up

- The user reports incorrect lighting, frozen frames at the top of the screen, and
  red squares replacing HUD elements such as the track map. These remain visually
  unverified after the following changes.
- Shader permutation compilation now saves the newly compiled permutation's metadata
  instead of the original program's metadata. Shader metadata version advances from
  6 to 7 so old entries are rejected and regenerated.
- The display frame's transition to shader-read layout now uses shader-read access
  and includes the fragment shader destination stage, matching its use by ImGui.
- Release build and CLI smoke check pass; the 30 existing targeted tests pass.
  These tests do not exercise cache preload or prove visual correctness.
- `scripts/Check-GTSportGraphics.ps1` summarizes graphics diagnostics from the local
  log and keeps player visual confirmation explicit. Log counts depend on run length
  and duplicate suppression.
- Recurring agent runs are not configured: scheduling tools are unavailable in the
  current session. Future checks should use the same race/camera/weather, inspect
  logs and any available screenshots, and ask the user about all three symptoms.

### Follow-up after the user reported no visual improvement

- Rectangle/quad emulation now forwards vertex render-target layer selection through
  auxiliary tessellation shaders and writes the final Layer builtin in evaluation.
  It previously discarded this guest export. A reserved varying shifts the intermediate
  parameter locations; fragment-facing parameter locations stay consistent.
- Corrected auxiliary clip-distance varying locations and kept them separate from
  the layer transport. Point size and viewport index forwarding remain unsupported.
- Shader binary and pipeline key versions advance to 9 and 5 respectively to reject
  previously generated shader interfaces, including local experimental variants.
- Release build, CLI smoke check, and 32 targeted tests pass. Nine auxiliary shader
  modules pass spirv-val for Vulkan 1.2; 630 game shader modules captured during the
  startup/menu run pass offline Vulkan 1.3 validation with scalar block layout.
- Actual runtime validation requires setting VK_LAYER_PATH to the local Vulkan SDK's
  Bin directory. An initial run requested validation without loading the layer;
  that run is not counted as runtime validation. The later runs loaded the layer.
- The dropped-layer warnings disappear. Cache-enabled validation still reports
  descriptor mismatches during preload. With pipeline caching disabled, later live draws
  still report descriptor mismatches, depth attachment format/layout errors, tessellation
  inputs missing from the preceding vertex shader, WRITE_AFTER_PRESENT hazards, and 17
  dynamic resource-offset failures involving Phi/GetAttributeU32. Diagnostic runs
  closed normally with exit code 0. These observations do not prove correct graphics.
- The current visual comparison run disables pipeline caching in the isolated profile
  to isolate the observed preload mismatches; this does not fix the live-draw errors.
  Validation and shader dumping are disabled
  again for normal performance. Cache files and original user settings are preserved;
  first-time shader compilation can be slower. The prior isolated config is backed up
  at Build/gt-sport-fixed/config-before-layer-validation.json.
- Player comparison of lighting, the frozen top strip, and red HUD/map squares remains
  pending. Logs for the layer changes are preserved under Build/gt-sport-fixed.

- Longer races and sustained gameplay stability still need testing.
- Synchronization validation still reports a `WRITE_AFTER_PRESENT` diagnostic, and
  the menu has a tessellation interface mismatch. Some shader resource table offsets
  involving `Phi`/`GetAttributeU32` remain unsupported.
- BluetoothHid and thread atexit hooks still have generic stubs; other unrelated
  diagnostics remain. The emulator is not claimed to be free of all errors.
- The game dump lacks `SST-Roman.otf` and `SST-Bold.otf`; trophy extraction also needs
  the user's trophy decryption key.
- Testing uses copied saves/cache and the existing enabled "Fix 1.69 update boot
  crash" patch in an isolated profile. No new game boot patch was authored.
