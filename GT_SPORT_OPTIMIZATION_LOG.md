# GT Sport (CUSA03220) optimization log

Started: October 5, 2026. Base: commit `8021b5f2` (the confirmed graphics checkpoint in
`GT_SPORT_BASELINE.md`) plus the uncommitted `CHANGES.MD` and `scripts/Run-GTSportPerformance.ps1`.

Each entry is written here **before** its code change. Its status is then updated to
Implemented, Built, Tested, or Reverted. Player confirmation is recorded separately: a
clean build or log does not prove correct graphics or better frame rate.

## Survey summary

Areas reviewed before planning: the prior GT Sport work (`CHANGES.MD`,
`Build/GT_SPORT_STATUS.md`), the latest Release performance-run log, the per-draw path
(`Rasterizer::Draw`, `BindResources`, `BindBuffers`, `BindTextures`, `RebindTextures`,
`FinalizeTextureLayouts`, `BeginRendering`), scheduler rendering and barriers, the stream
buffer, pipeline/program lookup, the swizzled-alpha blend path, the PM4 command processor
waits, and `libScePad`/`libSceVideoOut` calls seen in the log.

Findings from the October 5, 11:47 Release run (`Build/gt-sport-fixed/user/log/shad_log.txt`):

- `scePadOpenExt` logged 2,126 warnings plus thousands of suppressed duplicates. It is called
  repeatedly by the game's `tm_ffb_deviceEventThread` (wheel force-feedback polling) and a
  second thread. Pad handle maps are unordered maps with no lock, while `scePadRead` and others
  read them from other threads: a data race, not only log noise.
- `sceVideoOutSubmitFlip: flipmode = N` is logged on every flip (344 lines in the run).
- 1,817 stale pipeline-cache entries at startup (from earlier cache versions).
- No device loss, invalid T#, or GPU wait-stall diagnostics in that run.

## Planned and completed changes

### OPT-001: Restore render-pass merging between consecutive draws

- Status: Implemented; Release build passed (October 5)
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp` (`Rasterizer::Draw`)
- Change: Remove the `scheduler.EndRendering()` call added after every direct draw in local
  commit `8021b5f2`. Upstream (`8e23388a`) does not have it, and the commit and change notes
  give no reason for it.
- Why: `Scheduler::BeginRendering` reuses the active dynamic-rendering pass when the next draw
  has an identical `RenderState`. Ending after every draw turns each draw into its own
  `vkCmdBeginRendering`/`vkCmdEndRendering` pair with attachment load/store, which costs CPU
  driver time and GPU attachment traffic on GT Sport's thousands of draws per frame.
  `DrawIndirect` already omits the call.
- Safety: Every operation recorded between draws that is invalid inside a render pass already
  ends rendering itself: `Runtime::FlushBarriers` (when barriers exist), all runtime
  copy/fill/clear/upload/download paths, blit helper, tile manager, fault manager, compute
  dispatches, and session end. A layout or attachment change produces a different
  `RenderState` and starts a new pass.
- Risk: Medium. Restores upstream behavior, but the confirmed visuals were tested with the call
  present. Revert if any graphics regression appears.

### OPT-002: Stop CPU zero-filling emulated shared memory on every dispatch

- Status: Implemented; Release build passed (October 5)
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp` (`BindBuffers`, `SharedMemory`)
- Change: When workgroup shared memory (LDS) is lowered to a storage buffer, reserve its
  per-dispatch region in the stream buffer with `StreamBuffer::Reserve` instead of `Map` plus
  `memset`. Assert clearly if the request exceeds the stream buffer (`Map` previously returned
  null and `memset` wrote through it).
- Why: GT Sport's lighting compute shaders use this lowering (confirmed by the earlier
  lighting fix). The size is `LDS bytes x workgroups`, written over PCIe into host-visible
  device memory on the GPU command thread for each dispatch. The region is still separate for
  each dispatch and protected by the stream buffer's GPU-tick watches.
- Accuracy: GCN LDS has undefined contents when a workgroup starts, and native Vulkan
  `Workgroup` memory is also uninitialized, so zero-filling is not hardware behavior.
- Risk: Low. A shader that relied on zeroed LDS would already be nondeterministic on hardware.

### OPT-003: Make libScePad handle state thread-safe and stop redirect log spam

- Status: Implemented; Release build passed (October 5)
- File: `src/core/libraries/pad/pad.cpp`
- Change: Guard `pad_handle_map`, `handle_to_controller_map`, and the handle counter with a
  shared mutex (exclusive for open/close, shared for lookups). Log the `scePadOpenExt`
  redirect once instead of on every call.
- Why: Concurrent `unordered_map` insert and lookup is undefined behavior and can crash or
  corrupt lookups during the wheel-polling loop. Logging every poll costs formatting and log
  I/O on game threads.
- Risk: Low. Return values and handle assignment are unchanged.

### OPT-004: Log a non-default flip mode once per mode

- Status: Implemented; Release build passed (October 5)
- File: `src/core/libraries/videoout/video_out.cpp`
- Change: Warn once per distinct `flipMode` value rather than on every flip.
- Risk: Very low. Diagnostic output only.

### ACC-001: Generalize swizzled-alpha blend emulation

- Status: Implemented; Release build passed; 5 color-export and 4 color-blend tests pass (3 export and 2 blend tests are new)
- Files: `src/shader_recompiler/runtime_info.h`, `src/shader_recompiler/frontend/translate/export.cpp`,
  `src/video_core/renderer_vulkan/liverpool_to_vk.{h,cpp}`,
  `src/video_core/renderer_vulkan/vk_pipeline_cache.cpp`,
  `src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp`,
  `src/video_core/renderer_vulkan/vk_pipeline_serialization.cpp`, tests.
- Problem: When a color buffer's component swap moves logical alpha out of the physical alpha
  lane (for example the reversed RGBA8 mapping used by GT Sport's HUD), Vulkan applies the alpha
  equation to a logical color channel and resolves `SrcAlpha` from the wrong lane. The existing
  exact fix covers only color `SrcAlpha`/`OneMinusSrcAlpha`, alpha `Zero`/`OneMinusSrcAlpha`, Add.
  That fixed the in-race icons; the bottom in-race track HUD is still cyan, which is the same
  symptom (logical red lost in the physical alpha lane).
- Change: Keep the confirmed exact path unchanged. Add a general path for any enabled blend
  whose color and alpha functions match (Add, Subtract, or ReverseSubtract) and whose four
  factors depend only on the source (`Zero`, `One`, `SrcColor`, `OneMinusSrcColor`, `SrcAlpha`,
  `OneMinusSrcAlpha`). The shader exports each logical channel premultiplied by its guest
  source factor (`Fs(l) * S_l`, with the source clamped to the format's normalized range) and
  writes `1 - Fd(l)` to the secondary output in physical order. Vulkan then uses
  `One`/`OneMinusSrc1Color` (color) and `One`/`OneMinusSrc1Alpha` (alpha) with the guest
  function, so each physical lane gets its own logical source and destination factor.
  It applies only when lane placement changes the result: the color and alpha equations differ,
  or a factor reads source alpha. Equations identical for every channel without alpha factors
  (for example `One`/`One`) are already exact on the native path and stay there.
  Blending applies only to Unorm, Snorm, Srgb, and Float targets, so other number formats keep
  the native path.
- Gate: Same structural limits as the exact path (attachment 0, dual-source support, alpha
  exported, no blend bypass, no number conversion). The MRT check requires that no MRT1-7 is
  actually written: a target counts only if it has a bound color buffer, a nonzero
  `CB_TARGET_MASK`, and a nonzero shader export format. The old check (MRT1 format Zero and
  MRT1-7 target masks clear) is a subset, so every draw the confirmed exact path accepted is
  still accepted. While either emulation is active, the translator drops exports to other MRTs,
  which hardware does not write under this condition; they previously could add an unused
  attachment or be mistaken for the synthetic second source.
  (Corrected before testing: the first version required every MRT1-7 export format to be Zero,
  which could have rejected draws the confirmed path accepted.)
- Cache: Shader binary and pipeline key versions advance, so shaders recompile on first visit.
- Risk: Medium. It changes only draws that are currently blended wrongly, but the cyan HUD
  might use a configuration outside this set; ACC-002 identifies it if so.

### ACC-002: Report remaining unhandled swizzled blends once

- Status: Implemented; Release build passed
- File: `src/video_core/renderer_vulkan/vk_pipeline_cache.cpp`
- Change: When an enabled blend has logical alpha outside the physical alpha lane and neither
  emulation path applies, log one warning per distinct configuration (bounded count), naming
  the failed condition, the swizzle, and the blend factors and functions.
- Why: The cyan bottom HUD still needs its exact blend configuration. A normal-speed race
  run with `scripts/Run-GTSportPerformance.ps1` (warning filter) will record it.
- Risk: Very low. Diagnostic output only.

### ACC-003: Count the new blend diagnostics in the log summary script

- Status: Implemented
- File: `scripts/Check-GTSportGraphics.ps1`
- Change: Add `SwizzledFactorBlendEmulation` (the one-time info line when the general path is
  first used) and `UnhandledSwizzledBlend` (ACC-002 warnings) to the summarized patterns.
- Note: The factor-blend info line is logged at Info level, which the performance launcher's
  `*:Warning` filter hides; use `-FullLogging` to see it. ACC-002 warnings always appear.
- Risk: None. Script output only.

### PROF-001: Symbol-enabled Release build and a CPU profiling launcher

- Status: Implemented. `Build/x64-Clang-Profile` built (shadps4.exe and a 181 MB PDB, 14:09); script parses. Needs the player to run it (UAC prompt for WPR).
- Files: new build directory `Build/x64-Clang-Profile` (ignored by Git), new script
  `scripts/Profile-GTSport.ps1`. No emulator source change.
- Why: The player reports frame rate needs a large improvement. The bottleneck has not been
  measured; guessing at hot paths risks effort on the wrong code. Windows Performance Recorder
  and xperf are installed, but the current Release binary has no PDB, so samples cannot be
  attributed to emulator functions.
- Change: Configure a second Release build with identical optimization flags (`/O2 /Ob2
  /DNDEBUG`) plus CodeView symbols (`/Zi`) and linker `/DEBUG /OPT:REF /OPT:ICF`, so its code
  matches Release. The script launches it through `Run-GTSportPerformance.ps1
  -BuildDirectory`, starts a WPR CPU-sampling trace when the player presses Enter during a race,
  stops after a fixed duration, and writes per-thread and per-function summaries with xperf.
- Risk: None to the game profile beyond what the performance launcher already does.
- Fixes after the first run: xperf needs `-o` before `-a` (the summary was not written), and
  `OpenThread` must request `THREAD_QUERY_LIMITED_INFORMATION` (0x0800), not 0x1000
  (`THREAD_RESUME`), for thread names.

### LIGHT-001: Refresh GPU-written flattened shader constants on the GPU timeline

- Status: Implemented; Release (14:09) and profiling builds pass; 487/487 non-GPU-instruction tests pass, including source-address checks added to the three generated-walker tests. Runtime effect untested.
- Files: `src/shader_recompiler/ir/passes/flatten_extended_userdata_pass.cpp`,
  `src/shader_recompiler/ir/passes/srt.h`, `src/shader_recompiler/info.h`,
  `src/video_core/renderer_vulkan/vk_rasterizer.{h,cpp}`, shader-cache version.
- Problem: Every scalar constant a shader loads through a descriptor-table pointer
  (`s_load_dword*`, IR `ReadConst`) is copied into the flat user-data buffer by the JIT SRT
  walker on the CPU when the draw is recorded. With `readbacks_mode 0`, GPU writes never reach
  guest memory, and even with readbacks, a value produced earlier in the same frame has not been
  executed yet at record time. Such shaders read stale constants. A GPU-computed exposure or
  lighting parameter read this way would give lighting that is too bright or too dark depending
  on camera and area, matching the player's report. (The removed runtime-readback trial in
  `CHANGES.MD` could only provide previous-frame values.)
- Change: The walker also records the guest source address of each flattened dword (third
  argument; no change to the values it copies). At bind time, flat-buffer dwords whose source
  range is GPU-modified in the buffer cache are copied from the cached GPU buffer into the
  uploaded flat buffer with `vkCmdCopyBuffer`, followed by a transfer-to-shader-read barrier,
  so the shader sees the value in GPU execution order, as hardware scalar loads do.
  Pointers, offsets, and resource descriptors consumed by the CPU walker or binding code are
  unchanged; only the shader-visible copy is refreshed.
- Diagnostic: One bounded warning per shader that needs a refresh (shader hash, refreshed dword
  count), plus a running total at 1,000, 10,000, 100,000... refreshed bindings, so a race log
  shows how widespread and how frequent this is. `scripts/Check-GTSportGraphics.ps1` counts both.
- Cost: Only draws that read GPU-written constants pay for a copy and barrier (which ends the
  current render pass). Others add one GPU-modified range check per contiguous source run.
- Cache: Shader binary/metadata versions advance (walker calling convention changes).
- Risk: Medium. Untested hypothesis for the lighting symptom; the diagnostic will confirm
  whether the path is exercised.

## Verification so far (October 5)

- Release build and CLI `--help` smoke check pass with OPT-001 to ACC-002.
- 487 of 487 non-GPU-instruction tests pass (one skipped by design), including the new
  color-blend and color-export tests. 41 `GcnTest` GPU instruction tests fail on this machine
  both with and without these changes (identical failure set, compared by stashing the changes
  and rebuilding), so they are pre-existing and unrelated.
- Runtime check, 13:51 launch with `scripts/Run-GTSportPerformance.ps1 -FullLogging`: the game
  booted, compiled shaders into the new cache version, reached the menu, Arcade, and
  `race_timeattack`, and stayed responsive past 150 seconds. Both the exact and the new general
  swizzled blend paths were used. No unhandled swizzled blend, assertion, device loss, or
  critical errors were logged by that point. The redirect and flip-mode warnings each appeared
  once (flip mode 5). This run used the binary built before the MRT-gate correction above,
  which only widens which draws the emulation accepts; the corrected build must be relinked
  after that session closes.
- Not yet established: frame-rate change, whether the bottom race HUD is now white, and
  long-session stability. These need the player.
- The emulator closed at 13:56:49 after also loading a campaign race (`race_gt_league`). Log
  summary for that session: no device loss, validation, descriptor, or allocation diagnostics;
  `UnhandledSwizzledBlend` 0. Its log is saved as
  `Build/gt-sport-fixed/log-optimization-smoke-*.txt`. Release was then relinked (13:56) with
  the corrected MRT gate.

### Player result, October 5 session (binary before the MRT-gate correction)

- Bottom in-race track HUD: **white** (previously cyan). ACC-001 confirmed by the player.
- Frame rate: playable in Time Trial but needs a large improvement. No numbers recorded.
- Lighting: inconsistent; too bright or too dark in some cameras or areas.

### Player result, October 5 14:12 profiling session (LIGHT-001 build)

- Lighting: **great** (player). Correction: the session log has no LIGHT-001 refresh messages,
  so its refresh path never ran and did not cause the improvement. The likely cause is the
  13:56 swizzled-blend MRT gate correction (ACC-001) or a difference in tracks and conditions
  between sessions. LIGHT-001 stays because it is part of the approved build and only acts on
  GPU-written constants.
- Graphics accuracy: player has no complaints. **Accuracy is now frozen: later changes must not
  alter rendering results.** Performance work must keep identical output.
- Frame rate: about 30 FPS in a race; the goal is higher.
- Profile `Build/gt-sport-fixed/profiles/20261005-141240`: 20-second WPR CPU trace. Two
  emulator threads used 97% and 91% of a core, a third 32%, the rest about 2% or less. This
  points to a CPU-bound single thread rather than the GPU. Thread names were not captured by
  the script; `functions.txt` was not written, so the trace is processed separately.

### Profile findings, 14:12 trace (one-second slice, 3,727 emulator samples)

- GPU command processor thread (`Liverpool::Process`, about 91% of a core): 58% of its samples
  are in `VideoCore::Image::GetBarriers`, from `Runtime::Transit` in `BindTextures` (308) and
  `FinalizeTextureLayouts` (175). Barrier flushes and the driver are small, so this is
  bookkeeping, not GPU synchronization. A partial (view-range) transition that changes nothing
  still expands the image's state into one entry per mip and layer, and the next full
  transition walks and collapses it again, on every draw. Images here have up to 512 layers.
- Game thread 25068 (97% of a core): `sceUsbdHandleEventsTimeout` → libusb, mostly in the kernel.
- Game thread 6260 (32%): Windows device enumeration (`setupapi`/`cfgmgr32`/`devobj`).

### PERF-001: O(1) no-op image transitions with identical barrier semantics

- Status: Implemented; Release and profiling builds pass. New equivalence test: 1,500 random sequences x 40 transitions over five image shapes give identical per-subresource barriers, subresource states, and last_state versus the original algorithm; a deliberately injected stage-tracking bug was caught by it, then reverted. 490/490 non-GPU-instruction tests pass. No shader cache change. **Reverted from the default build by REG-001 pending A/B (dark lighting and device loss in the 14:34 session).**
- Files: `src/video_core/texture_cache/image.{h,cpp}`, new `src/video_core/texture_cache/image_barriers.h`,
  new test `tests/test_image_barriers.cpp`.
- Change: Move the barrier computation into a header-only function so it can be tested. Add a
  lazy "every subresource has this state" marker to each backing. A partial transition that
  would change no subresource records the marker instead of allocating and filling
  per-subresource states. A full transition of such an image emits at most one barrier covering
  all levels and layers. Real partial transitions still materialize per-subresource states.
  Adjacent layers of one mip with identical old state share one barrier instead of one each.
- Exactness: Vulkan barriers on a subresource range are equivalent to the same barrier on each
  subresource, and `last_state` (read for descriptor and attachment layouts and by
  `CopySubrect`) is updated exactly as before, including its pipeline stage. A randomized test
  runs the original algorithm and the new one side by side and requires identical
  per-subresource barriers, identical logical subresource states, and identical `last_state`
  after every step.
- Risk: Low; proven equivalent by the test before use in the game.

### PERF-002: Convert guest USB timeouts to the host timeval layout

- Status: Implemented; builds pass. Confirmed libusb on Windows uses the Winsock timeval ({long, long}) and computes timeouts from both fields, so the misread was real. No shader cache change. **Reverted from the default build by REG-001 pending A/B (dark lighting and device loss in the 14:34 session).**
- Files: `src/core/libraries/usbd/usbd.{h,cpp}`
- Problem: `sceUsbdHandleEventsTimeout`, `sceUsbdHandleEventsLocked`, and `sceUsbdWaitForEvent`
  pass the guest's `timeval` pointer straight to libusb. The guest layout is two 64-bit fields;
  Windows `struct timeval` has two 32-bit `long` fields. libusb therefore reads the low half of
  the guest seconds as seconds and the high half (normally 0) as microseconds, so a timeout
  such as 0 s + 100,000 µs becomes 0 and the game's wait loop busy-polls (thread 25068).
- Change: Read the guest `OrbisKernelTimeval` and pass a correctly converted host `timeval`
  (null stays null; negative or out-of-range values are clamped).
- Effect: The game's USB wait blocks for the timeout it requested, as on hardware. No rendering
  change. Linux and macOS already had matching layouts and are unaffected in practice.
- Risk: Low.

### Player result, October 5 14:34 profiling session (PERF-001 + PERF-002 build)

- Frame rate: about 30 FPS with some drops (unchanged).
- **Regression:** lighting dark. The session also ended with Vulkan device loss
  (`Device lost during submit`, present thread). The approved 14:12 session had no device loss.
- Profile `profiles/20261005-143435`: GPU command processor 59.7% of a core (was about 91%), and
  the 97% USB polling thread is gone (`tm_ffb_deviceEventThread` 29.6%). The emulator thread is
  no longer saturated, yet FPS did not change, which suggests frame pacing (a 60 Hz game
  dropping to every other vblank) or the GPU now limits the frame rate.
- The aa3822a3 invalid-descriptor context (source buffers and GPU-modified flags) is the same as
  in the approved session, so it does not explain the difference.

### REG-001: Restore the approved build; isolate PERF-001 and PERF-002 for A/B tests

- Status: Done (14:45). Patches in `Build/gt-sport-ab-20261005` (perf001.patch, perf002.patch, perf001-files). Source verified identical to the approved patch; Release (14:43) and profiling (14:45) builds rebuilt from it; 487/487 tests pass. A/B executables with PDB: `perf001-only` (14:44), `perf002-only` (14:44). PERF-001 and PERF-002 are reverted in the default build until the A/B result.
- Change: Save the PERF-001 and PERF-002 source changes as separate patches under
  `Build/gt-sport-ab-20261005`, then restore those files to the approved versions (the files they
  touched are unchanged in the approved patch). Rebuild Release so the default build matches
  the approved source again. Build two A/B executables from the profiling directory: approved +
  PERF-001 only, and approved + PERF-002 only.
- Test order for the player, on the same track and camera: (1) the saved approved binary, to
  check whether dark lighting also happens there (session-to-session variation rather than a
  code regression); (2) each A/B executable if (1) looks correct.
- Neither change returns to the default build until it passes the player's check.

### Player A/B result (after REG-001)

- Best lighting: the saved **approved Release** build. Its only inaccuracy: cockpit camera
  lighting.
- `perf001-only` and `perf002-only` both had worse lighting than the approved build.
- Interpretation: the two changes are independent, and PERF-002 (USB timeouts) cannot affect
  rendering directly. Both alter emulator timing (less CPU load), so lighting appears to be
  timing-sensitive. A likely mechanism is the game reading GPU-computed results (for example
  scene luminance for auto-exposure) on the CPU. With `readbacks_mode 0` the CPU never sees
  GPU writes, so such values are stale and depend on timing. The cockpit inaccuracy in the
  approved build fits the same pattern. Not yet proven.

### DIAG-001: Temporary readback-mode override in the performance launcher

- Status: Implemented; script parses and `-CheckOnly` accepts it; profile value (0) untouched. Awaiting the player run of the approved build with `-ReadbacksMode 2`.
- File: `scripts/Run-GTSportPerformance.ps1`
- Change: Add `-ReadbacksMode <0|1|2>`, applied like the other per-run overrides and restored
  when the emulator exits. Default unchanged (the profile's own value).
- Why: `readbacks_mode 2` (Precise) read-protects GPU-written pages so a CPU read downloads the
  GPU data; modes 0 and 1 never give CPU reads the GPU result. Running the approved build with
  mode 2 tests the readback hypothesis without any code change. Expect a frame-rate cost.
- Risk: None to the saved profile settings; it is a test-only override.

### Player result, DIAG-001 run (approved build, `-ReadbacksMode 2`, 15:17)

- Crashed. Track glitches (vertex explosions), lighting a little off, poor frame rate.
- Precise readbacks are not viable for GT Sport in their current form; the hypothesis is
  neither confirmed nor ruled out. The profile setting was restored to 0 by the launcher.
- Crash cause (log): `Unhandled exception: code is too big` on the GPU command processor, an
  Xbyak exception from the 32 MB SRT-walker JIT buffer (`g_srt_codegen`) filling up.

### FIX-001: Stop duplicating cached SRT walker code during pipeline-cache warm-up

- Status: Implemented; Release build passes (needed -j 6 after a compiler out-of-memory with full parallelism). Runtime check pending.
- Files: `src/shader_recompiler/ir/passes/srt.h`,
  `src/shader_recompiler/ir/passes/flatten_extended_userdata_pass.cpp`,
  `src/shader_recompiler/info.h`, `src/video_core/renderer_vulkan/vk_pipeline_serialization.cpp`.
- Problem: `LoadPipelineStage` runs for every stage of every cached pipeline. Each call
  deserializes the shader metadata, and deserialization appends that shader's walker code to
  the append-only 32 MB buffer, even when the shader permutation is already loaded and the new
  copy is discarded. A shader shared by N cached pipelines leaves N copies, so buffer use grows
  with the pipeline cache until it overflows. This affects every readback mode, including the
  approved build, as the cache grows.
- Change: When registering cached walker code, reuse an existing registration for the same
  permutation hash if its bytes are identical (`memcmp`), instead of appending a copy.
  Discarded duplicates never ran, and a permutation only ever runs its first registered copy,
  so execution is unchanged. Code compiled at runtime is unaffected.
- Risk: Low. No rendering change; reduces JIT memory use and warm-up work.

### Finding: runaway shader permutations with identical code

- The shader cache directory holds 619,507 files. At the current cache version (since 13:50),
  fragment shaders `88dad027`, `4dc8e34a`, and `d24c0889` gained 9,109, 8,833, and 4,985
  permutations; 1,468, 1,172, and 960 of them in the crashed 15:17 session alone.
- All permutations of each shader written in that session have byte-identical SPIR-V (one md5
  each). The specialization comparison reports false mismatches, so the same module is
  recompiled, a new pipeline is built, and a new cache entry is written again and again. This
  costs frame time (compile stutter), memory, and cache size, and grows JIT and cache memory
  without bound.

### DIAG-002: Report which specialization field causes each new permutation

- Status: Implemented; Release build passes. Unattended menu run in progress.
- Files: `src/shader_recompiler/specialization.h`, `src/video_core/renderer_vulkan/vk_pipeline_cache.cpp`.
- Change: Add a helper that names the first specialization component that differs (resource
  counts, runtime-info sub-fields, start bindings, buffer/image/sampler index, and so on). When
  a program that already has 4 or more permutations gets a new one, log the reason against
  each existing permutation in compact form, rate-limited (first 40 events, then every 500th).
- No behavior change; diagnostic only. It will be removed or reduced once the cause is fixed.

### DIAG-002 result (unattended menu run, 15:5x)

- Reproduced at the menus with no input: 32 new-permutation events within about a minute.
- Every rejection of the most recent permutation is `fs input count`, for fragment shaders
  `d24c0889`, `6ef0d49b`, `63d472ef`, `26b05d13`, `53121301`, `de092780`, and others.

### Player report during the DIAG-002 run (approved + FIX-001 + DIAG-002)

- FPS as low as 15; target 30 or higher. Lighting very inaccurate.

### FIX-002: Build new permutations' specialization from the unmodified runtime info

- Status: Implemented and verified unattended (4-minute menu run from an empty cache): highest permutation index 8, versus thousands before. The 30 remaining multi-permutation events are vertex shaders whose start bindings differ by paired fragment shader (upstream behavior, bounded) and aa3822a3 image types from its known invalid descriptors. Player check of FPS and lighting pending.
- File: `src/video_core/renderer_vulkan/vk_pipeline_cache.cpp` (`GetProgram`), shader metadata
  version in `vk_pipeline_serialization.cpp`, new regression test.
- Cause: `InjectClipDistanceAttributes` (used on the NVIDIA proprietary driver,
  `needs_clip_distance_emulation`) increments `runtime_info.hw.fs.num_inputs` during compilation.
  Commit `8021b5f2` changed `GetProgram` to build each new permutation's specialization after
  `CompileModule`, from that modified runtime info. Upstream built it before compiling. The stored
  key therefore has one more input than any freshly built key, never matches, and every lookup
  compiles another identical module (same SPIR-V) and pipeline, then walks the growing list.
- Change: Compile with a copy of the runtime info and build the specialization from the
  unmodified one, so the stored key matches the next lookup. Resource metadata still comes from
  the compiled module's own `Info`, as before.
- Exactness: The reused module was compiled from the same runtime info and is byte-identical to
  the modules being recompiled (verified by md5), so rendering is unchanged.
- Cache: Shader metadata version advances so the thousands of never-matching stored permutations
  are rejected. The old cache directory is moved aside (not deleted) as
  `cache/CUSA03220-pre-permutation-fix-20261005` so the next run starts clean; expect one
  round of shader compilation.
- Expected effect: no more compile stutter from repeated permutations, no per-lookup scan of
  thousands of permutations, bounded JIT and cache growth (removing the crash's main driver).

### Player result, 16:00 session (approved + FIX-001 + FIX-002)

- Crashed: Vulkan device loss during submit (same signature as 14:34). No code-buffer overflow;
  only 10 bounded new-permutation events, so FIX-002 held.
- FPS: 20-30 in a race, 40-60 in Time Trial (previously about 30 in Time Trial, drops to 15).
- Lighting: correction from the player: fine everywhere except the cockpit camera (same as
  the approved build), so FIX-001/FIX-002 did not regress lighting.

### DIAG-003: Crash-diagnostic layer switch in the performance launcher

- Status: Implemented. The 16:07 session with the layer did not crash; the player closed it. Lighting too dark; FPS unchanged.
- File: `scripts/Run-GTSportPerformance.ps1`
- Change: `-CrashDiagnostics` enables `Vulkan.vkcrash_diagnostic_enabled` for that run, points
  `VK_LAYER_PATH` at the local SDK `Build/tools/VulkanSDK/Bin`, and sets `CDL_OUTPUT_PATH` to
  `Build/gt-sport-fixed/crash-dumps`, so the layer writes a report naming the in-flight
  draw/dispatch and resources when the device is lost. Restored when the emulator exits.
- Why: the device loss is intermittent and the log does not identify the faulting command.
- Cost: the layer adds tracking and slows rendering; diagnostic runs only.

### Finding: lighting varies between sessions with identical code

- 16:00 session (approved + FIX-001 + FIX-002): lighting fine except cockpit camera.
- 16:07 session, same executable, only the crash-diagnostic layer added (timing change, no
  rendering change): lighting too dark.
- So dark lighting is nondeterministic and timing-sensitive, not caused by a specific code
  change. Single-session A/B comparisons of lighting are therefore unreliable. This likely also
  explains the earlier PERF-001/PERF-002 A/B results: both builds changed timing, and one
  session each cannot separate them from normal variation. PERF-001 is proven equivalent by its
  barrier test; PERF-002 does not touch rendering.
- The cockpit-camera error is consistent across sessions and is the better first target for a
  frame capture.

### Player report, 19:23 capture session (current Release: approved + FIX-001 + FIX-002)

- Exposure wrong per camera: third-person too bright, HUD camera too dark, cockpit bright,
  hood camera dark. This pattern points to auto-exposure (per-view brightness adaptation)
  rather than individual materials.
- Five RenderDoc captures saved 19:27-19:28 (`user/captures/CUSA03220_capture*.rdc`); being
  analyzed with `Build/tools/renderdoc-1.46/RenderDoc_1.46_64/capture-inspect.exe`.

### Capture analysis, 19:27 captures

- Capture 1: 3D scene completely black, HUD only. Capture 2: third-person, about normal.
  Capture 4: cockpit, about normal. Capture 5: same third-person view as 2, about 20 s later,
  heavily overexposed. Capture 3: 7 actions only (no scene).
- Sky pixel traced through the display copy to the tonemap pass: event 10103 (capture 2) and
  9785 (capture 5), same fragment shader `fs_0xb33ec4df` (matched by SPIR-V md5 to the cache).
- Tonemap inputs are identical except two words of its CPU-built flat buffer: **word 47 is
  0.0046 in the normal frame and -2.4e-7 in the overexposed one** (word 50 also differs). Word 47
  is almost certainly the exposure scale. So the exposure constant fed to the tonemap is wrong.
- LIGHT-001 never logged a refresh, so the buffer cache did not consider its source GPU-written.
  Hypothesis: the game's exposure pass writes the value through an image (storage image or
  render target) tracked by the texture cache, which the buffer-cache check does not see.

### DIAG-004: Trace the source of flattened constants for the tonemap shader

- Status: Implemented, but it never fired in the 20:02-20:21 race session (Release with the trace confirmed), so the tonemap does not run under hash b33ec4df at runtime or the md5 match was a twin. Superseded by DIAG-005.
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp` (`RefreshGpuWrittenConstants`).
- Change: For `fs_0xb33ec4df` only, log (rate-limited) flat-buffer words 40-55: value, guest
  source address, buffer-cache GPU-modified flag, and whether a GPU-written texture-cache image
  aliases the address (`HasGpuImageAlias`). No behavior change.

### Player report, 20:02-20:21 session

- Brightness depends on track and camera: Lago Maggiore correct except cockpit; Northern Isle
  dark in every camera. Green glitch: bright green blobs (glare/lens-flare shapes) in the
  cockpit view (screenshot `CUSA03220_20261005_202113_279_game_000014.png`).
- FPS dramatically better in this build (FIX-002 with a warm cache).

### DIAG-005: Report shaders whose flattened constants come from GPU-written memory

- Status: Implemented (Release 20:23). 20:24 and 20:33 capture sessions ran it: zero reports, and zero LIGHT-001 refreshes. Result: no flattened constant (including the tonemap exposure word) is read from GPU-written buffer or image memory. The game writes the exposure constant from the CPU, so the likely fault is upstream of it: the CPU computing exposure from GPU results it cannot see with readbacks disabled. The DIAG-001 Precise-readback crash was the JIT leak fixed by FIX-001/FIX-002, so a Precise retest is worthwhile; its vertex glitches remain unexplained.
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp` (`RefreshGpuWrittenConstants`).
- Change: Replace DIAG-004. For each contiguous source run, also check whether a GPU-written
  texture-cache image aliases it (`HasGpuImageAlias`), which the buffer-cache check misses.
  Log once per shader (bounded): stage and hash, flat-buffer size, dwords from GPU-modified
  buffers, and dwords from GPU image memory with the first such address. No behavior change.
- Why: tests whether exposure-like constants are produced by GPU image writes, the suspected
  reason LIGHT-001 never refreshed anything.

### Player result, Precise readbacks retest (after FIX-001/FIX-002)

- `-ReadbacksMode 2`: lighting still incorrect and unstable; vertex explosions back.
- Conclusion: GPU-to-CPU readbacks are not the cause of the exposure problem. Precise mode
  stays off. Next suspect: GPU-side exposure data that the tonemap samples as textures
  (the 8192x1 R32F tables and the 1x1 R8 image), which could be stale if the texture cache
  does not see the GPU writes that produce them.

### Capture analysis, 20:3x captures 5 (normal) and 6 (overexposed), Northern Isle

- Tonemap passes: event 9363 (capture 5) and 6550 (capture 6). Both 8192-entry tables and the
  tone curve are bit-identical (exported as DDS with an extended local capture tool).
- Correction: the 304-byte block compared earlier is not the emulator flat buffer (that is
  64 dwords) but one of the game's own constant buffers, copied from guest memory into the
  stream buffer when the draw is recorded. In the overexposed frame its contents are wrong:
  dwords 4-7 are `464c8266 43070000 0 0` (13088.6, 135.0) instead of floats near 0.95, 0.32,
  0.04, -0.19, and dwords 47, 50, 51, 58, 59 are zero instead of color-grading values.
- Conclusion: the game's per-frame constants are captured at the wrong time (before the game
  writes them for this frame, or after it starts writing the next frame). This explains the
  timing sensitivity, why readbacks do not help, and why DIAG-004/005 (watching the flat
  buffer) saw nothing.

### DIAG-006: Trace the tonemap constant buffer and when it is read

- Status: Done (21:13 build, player run until 21:38).
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp` (`BindBuffers`), diagnostic only.
- Change: Remove DIAG-004/DIAG-005. For every ordinary buffer bound to a shader, if the shader is
  a fragment shader with a 304-byte read-only buffer, log (rate-limited) the shader hash, guest
  address, the dwords 4-7 and 47, whether the range was CPU-modified since its last upload, and
  the current flip/frame counter, so normal and bad frames can be compared with timing. Also
  log each distinct fragment shader hash with such a buffer once, to confirm the tonemap's
  runtime hash.

### DIAG-006 result

- Tonemap runtime hash confirmed as `fs_0xb33ec4df`. Its 304-byte constant buffer moves to a new
  address every frame in the 0x20xxxxxxx range, consistent with constants embedded in the
  game's command-buffer memory (allocated from the command stream).
- Of 23 traced binds, most hold data that is clearly not this frame's constants (pointer pairs
  such as `0160278c 00000002`, small integers, zeros); only a few hold plausible floats. The
  memory has already been reused for other data when the emulator reads it.
- `GPU.copy_gpu_buffers` is enabled in the profile. It copies the command packets so the game
  can continue, but V# constant pointers still address the original command memory, which the
  game may reuse. It was turned on as a lifetime experiment before the captured-packet-size,
  fence-payload, and REWIND fixes.

### DIAG-007: One-run `-CopyGpuBuffers` override in the performance launcher

- Status: Implemented and run by the player with `-CopyGpuBuffers 0`: brightness no longer depends on camera or track, but gradually increases over time on both tracks; the race loaded without crashing; FPS about 2x worse.
- File: `scripts/Run-GTSportPerformance.ps1`
- Change: `-CopyGpuBuffers <0|1>` overrides `GPU.copy_gpu_buffers` for one run, restored on exit.
- Test: run with `-CopyGpuBuffers 0`. If exposure becomes stable, the copy mode exposes reused
  embedded constants. Watch for the old malformed-packet/race-loading failures.

### FIX-003: Read embedded constant buffers from the command-buffer copy

- Status: **Reverted** (Release rebuilt without it). The 22:06 build ran until 22:20: all 24 traced tonemap binds had `copied=false`, so the constant buffer is not inside the top-level command buffer and the fix never applied. Player saw Northern Isle gradually brighter and Lago Maggiore brighter/darker in some cameras, FPS a little worse; this is session-to-session variation, not an effect of FIX-003.
- Files: `src/video_core/amdgpu/liverpool.{h,cpp}`, `src/video_core/renderer_vulkan/vk_rasterizer.cpp`.
- Finding: with `copy_gpu_buffers` off, the per-camera/per-track exposure randomness disappears,
  so the copy mode is what exposes reused embedded constants; but the copy is also where the
  frame rate comes from (it lets the game continue while commands are processed).
- Change: While a graphics submission is processed from its copy, remember the original guest
  range of its command buffer and the copy. When a shader binds a read-only buffer that lies
  entirely inside that original range, upload the bytes from the copy (what the game wrote at
  submit time, which is what hardware reads) through the stream buffer, instead of from the
  original guest memory, which the game may already have reused. Bounded size; written buffers
  and ranges outside the command buffer are unchanged. A bounded diagnostic counts redirected
  binds and how many differed from guest memory.
- Exactness: constants embedded in a command buffer must stay valid until the GPU consumes
  them, so the submit-time copy is the hardware-visible value. Draws not using embedded
  constants are unaffected.
- Not covered: SRT tables embedded in the command buffer and read by the CPU walker, and
  indirect buffers (not copied). The gradual brightening is a separate issue (auto-exposure
  feedback) to investigate after this.

### Current understanding of the exposure bug (October 5, 22:2x)

- The tonemap (`fs_0xb33ec4df`) reads a 304-byte game constant buffer through a V# whose
  address changes every frame (0x20xxxxxxx). Its contents alternate between a few distinct,
  internally consistent patterns (for example `1.5, 0.5, -1, 1` and `0, -0.77, 0.77, 0`)
  rather than the expected exposure/grading floats. This suggests the V# points at the wrong
  constant block (descriptor read from guest memory at the wrong time or from the wrong
  table), not merely half-written data.
- `copy_gpu_buffers` off removes camera/track dependence but costs about 2x FPS, so timing of
  the emulator's reads relative to the game's CPU writes matters.
- Not the cause: GPU-to-CPU readbacks (Precise mode), GPU-written flattened constants or
  images (DIAG-005), constants embedded in the top-level command buffer (FIX-003).

### DIAG-008: Trace where the tonemap's buffer descriptor comes from, and processing lag

- Status: Done. The 01:3x session ended in the intermittent Vulkan device loss (not caused by the trace) after logging 22 traces.
- Files: `src/video_core/renderer_vulkan/vk_rasterizer.cpp` (extends DIAG-006),
  `src/video_core/amdgpu/liverpool.h` (read-only getter).
- Change: For the 304-byte tonemap buffer, also log the four flat-buffer offsets of its V#,
  whether each came from user-data registers (offset < 16) or was read by the SRT walker from
  guest memory (with that guest address), the V# words as flattened and as they are in guest
  memory at bind time, and the number of graphics submissions still queued behind the one being
  processed (how far the emulator lags the game). Diagnostic only.

### DIAG-008 result

- The tonemap V# is read by the SRT walker from a table in guest memory (0x2030xxxxx-0x2032xxxxx),
  is unchanged at bind time, and is well formed: base in 0x200-0x203 GB range, stride 16,
  19 records (304 bytes). The emulator is only one submission behind (pending_submits=1).
- So the address is right but the memory there does not hold this frame's constants yet.
  Candidate: the Constant Engine (CE) dumps constants from CE RAM into a per-frame ring with
  DUMP_CONST_RAM, and the draw engine (DE) must wait on CE/DE counters before reading them.
  The emulator implements the dump as a direct memcpy into guest memory on the GPU thread.

### DIAG-009: Correlate the tonemap buffer with recent CE constant dumps

- Status: Done (01:4x-01:50 run, 26 traces, no crash). Result: CE and DE counters stay 0 and no DUMP_CONST_RAM was recorded, so GT Sport does not use the Constant Engine here. CE/DE sync is ruled out.
- Files: `src/video_core/amdgpu/liverpool.{h,cpp}` (CE `DumpConstRam`), `vk_rasterizer.cpp`.
- Change: Record the last 256 DUMP_CONST_RAM destinations with CE/DE counters. At the traced
  tonemap bind, log whether its buffer lies in a recent dump, how many dumps ago, the dump's
  counters, and the current CE/DE counters. Diagnostic only.

### DIAG-010: Is the tonemap buffer inside an indirect command buffer?

- Status: Done (02:01 run, 22 traces, no crash). Result: neither the tonemap constant buffer nor its V# table lies inside any recent submitted or indirect command buffer. The constants live in a separate guest ring that the game CPU fills.
- Files: `src/video_core/amdgpu/liverpool.{h,cpp}` (INDIRECT_BUFFER), `vk_rasterizer.cpp`.
- Change: Record the last 256 graphics INDIRECT_BUFFER ranges (and the top-level submitted
  command buffers) with a sequence number. At the traced tonemap bind, report whether its
  constant buffer and its V# table lie inside a recent command buffer, which one, and how many
  buffers ago. Diagnostic only; replaces the DIAG-009 dump lookup.

### DIAG-011: Re-read the tonemap constants later to test "read too early"

- Status: Done (02:3x run, 15 checks). Result: in 13 of 15 checks the 304 bytes were unchanged one and two submissions after the draw (the other two were reused by later frames). The emulator does not read too early; the game itself wrote these values, so its CPU computed the bad exposure from some wrong input.
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp` (diagnostic only).
- Change: At each traced tonemap bind, keep the address and the 304 bytes the emulator used.
  At the end of that submission (`OnSubmit`) and again at the next one, re-read the guest memory
  and log whether it changed and the new dwords 4-7 and 47. If correct-looking floats appear
  later, the emulator read the constants before the game wrote them, which points to a missed
  or mis-evaluated GPU wait (for example WAIT_REG_MEM) that should hold the draw back.

### Finding: occlusion queries are faked

- `EventWrite` with ZPASS_DONE / PIXEL_PIPE_STAT_DUMP writes a fake, ever-increasing counter
  (+0x2FFFFFF per dump) into every counter pair, never the real number of passed samples. A
  game's visibility test (end - begin) therefore depends only on how many dumps happened in
  between, not on what is visible. Sun visibility for glare, lens flares, and possibly exposure
  adaptation would vary with the number of queries in a view, i.e. with camera and track.

### DIAG-012: Count occlusion-query dumps in races

- Status: Done and removed. The 02:38 race run (23 tonemap traces, so a race was reached) logged zero ZPASS_DONE dumps: GT Sport does not use occlusion queries here. Ruled out.
- File: `src/video_core/amdgpu/liverpool.cpp` (diagnostic only).
- Change: Log the first 20 ZPASS_DONE dumps (address, counter pairs) and the running total
  every 1,000 dumps, to confirm GT Sport issues occlusion queries during races before real
  query support is implemented.

### DIAG-013: One-run `-ReadbackLinearImages` override

- Status: Done. **Player result with `-ReadbackLinearImages 1`: brightness stable and correct on both tracks.** FPS somewhat worse in heavy races; the game crashed/froze (log ends without an error, consistent with a hang). Root cause of the exposure bug confirmed: the game CPU reads its luminance/exposure result from a small GPU-written linear image that the emulator did not copy back.
- File: `scripts/Run-GTSportPerformance.ps1`
- Why: the tonemap samples a 1x1 linear R8 image with a 6-level mip chain (address
  0x1000e32400), the usual shape of an average-luminance downsample. If the game CPU reads that
  GPU-rendered image to set exposure, buffer readbacks (modes 1/2) do not help, because they
  only download GPU-written buffers. `GPU.readback_linear_images_enabled` (off in the profile)
  downloads linear and tiny (<= 8 px wide) images written as render targets or storage after
  each submission.
- Change: `-ReadbackLinearImages <0|1>` overrides the setting for one run, restored on exit.
  Also removes the DIAG-012 counter from `liverpool.cpp`.

### FIX-004: Cheap, asynchronous readback of small GPU-written linear images

- Status: Implemented (02:53). Player: crashed while loading a race (02:55). The log ends without a message and stderr is empty. Cause: the asynchronous write-back ran on the scheduler priority thread and called `TryWriteBacking`, which asserts that the address is still mapped; race loading unmaps memory, so a delayed write can hit an unmapped range. That thread also freed staging memory, which the GPU thread otherwise owns.
- Correction (FIX-004b): defer the write-back with `DeferOperation` (runs on the GPU thread when the tick completes) and skip it if the destination is no longer a valid mapping. Implemented in `DownloadImageMemoryDeferred`; Release built 02:57; 487/487 tests pass. Awaiting player test.
- File: `src/video_core/texture_cache/texture_cache.{h,cpp}`.
- Problem with the existing option: `ProcessDownloadImages` downloads every linear (or <= 8 px
  wide) image written as a render target or storage, each with `scheduler.Finish()` (a full CPU
  wait for the GPU), on every submission and EOS fence. That stalls heavily in busy races and is
  a likely cause of the freezes.
- Change: When `readback_linear_images_enabled` is on, queue only images whose download is at
  most 64 KB (covers the 1x1 luminance chain and small tables), and download them
  asynchronously: the copy is recorded and the guest memory is written when that GPU work
  completes (`DeferPriorityOperation`), as a hardware memory write becomes visible to the CPU
  after the GPU finishes. No `Finish()` per image.
- Effect: keeps the correct exposure, removes the per-image GPU stalls. The game sees the
  luminance result once the producing work completes, which is also when hardware makes it
  visible.
- Risk: an image larger than 64 KB that the CPU reads would no longer be copied back; the
  bounded diagnostic logs each distinct size skipped so this can be reviewed.

### FIX-004c / DIAG-014: Defensive readback and synchronous logging for the crash

- Status: Done. Player run 03:01 with `-ReadbackLinearImages 1 -SyncLog`: the race loaded, brightness looked fine for a few seconds, then it crashed. The synchronous log shows `Device lost during submit` (Vulkan device loss), now triggered quickly when readback is on.
- Player: FIX-004b still crashes while loading a race (02:59); the log again ends without a
  message (asynchronous logging loses the last lines).
- Change (FIX-004c, `texture_cache.cpp`): in the readback path, skip images whose download size
  exceeds their guest size, depth images, and 3D images instead of asserting or downloading;
  these are not CPU-read luminance results.
- Change (DIAG-014, launcher): `-SyncLog` sets `Log.sync` for one run so the final messages and
  the emulator's fault report are written before a crash.

### FIX-004d: Restrict readback to tiny single-sample, single-layer color images

- Status: Implemented (03:04). Player run 03:05: crashed while loading a race. Synchronous log: the readback set is exactly the luminance chains (8x8, 4x4, 2x2, 1x1 linear images at 0x100af6a000..., several sets) and the 8192x1 table is skipped. The crash itself is `Unhandled exception: code is too big` (SRT-walker JIT buffer full), not the readback. Earlier race-loading crashes with empty logs were most likely the same overflow.
- File: `src/video_core/texture_cache/texture_cache.cpp` (`ShouldReadBack`).
- Why: with readback on, device loss follows within seconds. Copying a multisampled image
  directly to a buffer is invalid in Vulkan, and multi-layer or array images raise the same
  risk; the luminance result is one 1x1 single-sample color image.
- Change: read back only images with one sample, one layer, color aspect, and at most 4 KB
  (was 64 KB). Log each distinct image that is read back (bounded), so the set is visible.

### FIX-005: Enlarge the SRT walker code buffer and report its use

- Status: Implemented (03:07). Player run 03:1x-03:23: crashed on the second race (Vulkan device loss, separate issue). Walker code passed 32 MB during startup warm-up, so the old size could never last a session; FIX-005 is required.
- File: `src/shader_recompiler/ir/passes/flatten_extended_userdata_pass.cpp`.
- Why: the current cache is bounded (24,280 files, at most 12 permutations per shader), so this
  is not a runaway; GT Sport has on the order of ten thousand unique permutations, each with a
  walker of up to a few KB (slightly larger since LIGHT-001), which exceeds the fixed 32 MB
  executable buffer during long sessions.
- Change: reserve 256 MB for walker code (Xbyak allocates it up front as executable memory) and
  log buffer use whenever it crosses another 16 MB, so remaining headroom is visible.
- Risk: none to rendering; more reserved address space and committed memory as used.

### Player result 03:23 (FIX-004d + FIX-005, asynchronous tiny readback)

- Brightness less accurate than the synchronous all-image readback run: screenshots
  `CUSA03220_20261006_0321*`/`0323*` show heavy overexposure in the hood camera on both tracks.
  FPS 20-40 (with synchronous logging). Crash on the second race was a Vulkan device loss.
- Cause: the asynchronous write lands after the EOS/EOP fence that tells the game the GPU work
  finished, so the game reads the previous luminance value. The working run downloaded at the
  fence, before signaling (as hardware memory is complete when the fence is written).

### FIX-004e: Synchronous, batched readback before the fence

- Status: Implemented (03:26). **Player: brightness/lighting back to the good run and "pretty accurate".** But FPS 12-18 in a race and a freeze during the race (no crash). Cause: every fence with pending readbacks waits for the whole GPU (`scheduler.Finish`), several times per frame.
- File: `src/video_core/texture_cache/texture_cache.{h,cpp}`.
- Change: `ProcessDownloadImages` records all pending small readbacks, waits for the GPU once,
  and writes them to guest memory before returning (it runs before the fence is signaled and
  at submit). Nothing is done when no readback is pending. The size limit returns to 64 KB;
  the single-sample, single-layer, color filters stay. The per-image `Finish()` of the
  original option (the freeze/stall source) is gone.
- Writes are skipped if the destination is no longer mapped.

### FIX-006: Signal graphics fences after the GPU finishes instead of stalling

- Status: Implemented (03:34). Player: brightness still correct; 14 FPS in a race; froze at the start of a race. Log: the GPU thread stuck in `WAIT_REG_MEM` on 0x20289592c (waiting for 1, value 0) repeatedly.
- Diagnosis: (1) deadlock: the deferred operations run strictly in order on the scheduler priority thread; an earlier entry waits for a GPU tick that is not submitted while the GPU thread blocks in WAIT_REG_MEM, so neither proceeds. (2) Low FPS: once one fence was deferred, every later graphics fence was deferred too, serializing most of the frame behind the GPU.
- Correction (FIX-006b): defer only fences reached while readbacks are pending, and later writes to an address that already has a deferred write (ordering is per address). All other fences are signaled immediately as before. Before blocking in graphics WAIT_REG_MEM with deferred fences outstanding, submit pending GPU work so they can complete.
- FIX-006b status: implemented (Oct 6). `DeferFenceSignal(address, signal)` tracks per-address deferred counts under a mutex; `FlushForDeferredFences()` is called before blocking in graphics WAIT_REG_MEM and again on each wait-diagnostics report. Release build OK. Unit tests not run: the Release tree has no test target, and the change is outside the shader code they cover. Awaiting player test with `-ReadbackLinearImages 1`.
- Files: `src/video_core/amdgpu/liverpool.cpp` (graphics EOP, EOS, WRITE_DATA),
  `src/video_core/renderer_vulkan/vk_rasterizer.{h,cpp}`, `src/video_core/texture_cache/texture_cache.{h,cpp}`.
- Hardware behavior: an EOP/EOS fence is written when the GPU work before it has finished.
- Change: When a graphics fence is reached while readbacks are pending, record the readback
  copies, submit the GPU work without waiting, and queue an operation that runs when that work
  completes (scheduler priority thread): write the readback data to guest memory (skipping
  unmapped ranges), then write the fence value and raise its interrupt. The GPU thread continues
  immediately. While any fence is deferred, later graphics fences and WRITE_DATA writes are
  deferred in the same FIFO, so guest-visible writes keep their order (a later label can never
  be overwritten by an older one). With nothing pending and no deferred fence outstanding,
  fences are signaled immediately as before. EOS GDS stores and compute fences keep the current
  path. Staging memory from readbacks is released on the GPU thread.
- Expected: brightness as in FIX-004e, without the per-fence GPU drains (FPS back near the
  no-readback level), and no freezes from stalls.
- Risk: moderate (changes when the guest sees graphics fences, toward hardware behavior).

### REF-001: Save the accuracy-approved state as a local reference

- Status: Done. Release exe, profiling exe + PDB, a 1,448-line patch, scripts, and SHA256SUMS.txt saved; copies verified.
- Location: `Build/gt-sport-accuracy-approved-20261005` (ignored by Git; local only).
- Change: Copy the approved Release and profiling executables (plus PDB) and save the full
  uncommitted source diff against `8021b5f2` as a patch, with SHA-256 hashes. Nothing in the
  repository or the game profile is modified.
- Why: The player approved the graphics and asked that they not regress. Later performance
  builds can be A/B compared against this binary, or the source restored from the patch.

## Not changed: recommendations for player A/B tests

- `GPU.copy_gpu_buffers` is `true` in the isolated profile. It was enabled as a lifetime
  experiment before the captured-packet-size and fence-payload fixes, and copies every command
  buffer submission on the CPU. Turning it off may raise frame rate, but it should be tested
  separately for race-loading stability. The profile is not edited here.
- The 1,817 stale pipeline-cache entries come from earlier cache versions. ACC-001 advances the
  versions again, so the first race after this build will compile shaders.

### Player result, October 6 03:47 (FIX-006b)

- Brightness correct. FPS 14-20 in a race. Crashed shortly after starting a time trial that followed a race. Green blob visible again.
- Log: the crash is `Device lost during submit` (present thread). Of 20 runs on Oct 6, 8 ended in device loss, with and without readback, and the invalid-T# count for `cs_0xaa3822a3` does not track it (11 in one device-loss run, 410 in a clean one). Those descriptors are already rejected and bound as null images, so they are not the direct cause.

### DIAG-015: Measure deferred-fence latency

- Status: Implemented (Oct 6), Release build OK. Log line: `Deferred fences: N in F frames, latency avg X ms max Y ms`. Awaiting player run.
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp`
- Change: time each deferred fence from deferral to signal. Every 2 seconds, log the count, the average and maximum latency, and how many deferrals happened per frame.
- Why: the FPS drop with readback on looks like the GPU and the guest CPU running one after the other. The guest waits on a fence that only signals after the whole frame's GPU work, and that work is only submitted when the fence is reached. The numbers show whether that wait costs about one GPU frame (which would explain 30 FPS dropping to about 15) and whether earlier submission would help.
- Risk: none for rendering; one log line every 2 seconds.

### Player result, October 6 03:59 (DIAG-015)

- Crashed (`Device lost during submit`, GPU command processor thread).
- In races: about 2 deferred fences per frame, average latency about 12 ms, max about 64 ms, at about 13 frames per second by the frame counter. In one loading or menu stretch: 1 fence per frame at about 60 ms (5 FPS).
- Reading: when a fence is deferred, the frame's GPU work has only just been submitted (submission happens only at the end of a guest submit or at a deferral), so the guest waits for the GPU to render almost the whole frame from scratch. Recording and GPU execution do not overlap.

### PERF-003: Submit GPU work in chunks while readback is enabled

- Status: Implemented (Oct 6), Release build OK. `SubmitChunkIfNeeded()` runs at the start of Draw, DrawIndirect, DispatchDirect and DispatchIndirect. Awaiting player run; DIAG-015 latency lines will show the effect.
- Files: `src/video_core/renderer_vulkan/vk_rasterizer.cpp/.h`
- Change: when `readback_linear_images` is on, submit recorded work after every 128 draws or dispatches. The GPU then starts on the frame while the command processor is still recording it, so a deferred fence waits only for the tail of the frame.
- Accuracy: none expected. A submission boundary changes only when work reaches the GPU, not what it renders, and submissions already happen at arbitrary points (deferrals, waits, guest submits). With readback off, nothing changes.
- Risk: a little more submission overhead (a few hundred submits per second), and render passes end at each boundary.

### Player result, October 6 04:03 (PERF-003)

- Brightness correct. About 15 FPS in a race. Race previews about 15 FPS (were a locked 30). Crashed around lap 3 at Northern Isle Speedway (`Device lost during submit`, GPU command processor thread).
- DIAG-015: in races the deferred-fence latency dropped from about 12 ms to about 2.5 ms (2 per frame), yet FPS did not change. So deferred fences are no longer the bottleneck.
- Remaining full GPU drains while readbacks are pending: `ProcessDownloadImages()` calls `scheduler.Finish()` from `OnSubmit` (end of every guest submit) and from `OnFence` on paths that are not deferred: compute-queue WRITE_DATA and RELEASE_MEM, and the EOS GdsStore path.

### DIAG-016: Count and time synchronous readback drains by source

- Status: Implemented (Oct 6), Release build OK. Log line: `GPU drains in 2.0 s (count/total): submit=N/X ms ...`. The pre-existing GdsStore `Finish()` is timed as `gds_store`. Awaiting player run.
- Files: `vk_rasterizer.cpp/.h`, `liverpool.cpp`
- Change: `OnFence` takes a source tag (gfx EOS, gfx EOP, gfx WRITE_DATA, compute WRITE_DATA, compute RELEASE_MEM). `OnSubmit` counts as its own source. When a call actually drains (readbacks pending), record the count and the drain time per source. Log a summary every 2 seconds.
- Why: to find which drain costs the frame time before changing it.
- Risk: none for rendering; one log line every 2 seconds.

### Player result, October 6 04:09 (DIAG-016, same code as PERF-003 otherwise)

- Brightness correct. 22-36 FPS in a race (the previous run had about 15 with the same rendering code, so session-to-session variation is large). No severe freeze, and no crash: the log ends without a Critical message.
- DIAG-016: synchronous drains happened only at startup: 121 compute RELEASE_MEM drains costing 457 ms in one 2-second window, then 3 costing 11.5 ms over 16 seconds. None during races. Submit, gfx and GdsStore drains: none recorded while readbacks were pending.
- DIAG-015: in races about 3 deferred fences per frame, about 1 ms average latency.
- Conclusion: with readback on, the readback and fence path no longer limits the frame rate in races, and FPS is now about what it was with readback off (about 30). Further gains need a CPU profile of the current build to find the actual bottleneck.

### PROF-002: Profile with readback enabled

- Status: Done (Oct 6). `x64-Clang-Profile` rebuilt with FIX-006b, PERF-003 and DIAG-015/016.
- File: `scripts/Profile-GTSport.ps1`
- Change: new `-ReadbackLinearImages <-1|0|1>` parameter, passed through to `Run-GTSportPerformance.ps1`, so the profile runs the configuration the player uses.
- Risk: none (script only).

### Player result and profile, October 6 04:24 (PROF-002)

- Race previews still about 18 FPS (they were a locked 30 before the readback work).
- Profile `profiles/20261006-042412` (16 logical CPUs, so one core is 6.25% of machine usage):
  - `tm_ffb_eventHandleThread` (game force-feedback thread): 95% of a core. `tm_ffb_deviceEventThread`: 31%.
  - `shadPS4:GpuCommandProcessor`: 86% of a core, so it is the bottleneck. The game's job threads use about 3% each.
  - Exclusive time in the emulator process: `VideoCore::Image::GetBarriers` 1.16% of machine (about 19% of a core, mostly on the GPU thread). `Common::SpinLock::lock` 0.70% (the guest `sleepq` lock, used by game threads, not the GPU thread). `Liverpool::ProcessGraphics` 0.36%. Kernel time is 6.7% (about 45% of the process), largely the USB force-feedback threads.
  - Call stacks could not be unwound through the GPU thread (only `main` stacks resolved), so callers are inferred.
- Reading: the `GetBarriers` fast path is a handful of instructions, so this much time means the partial-transition path, which loops over every mip and layer and emits one barrier per subresource.

### PERF-004: Merge per-subresource image barriers and collapse uniform state

- Status: Superseded before testing. My first version set `last_state` differently from the original on collapse (`last_state` is read for descriptor and attachment layouts and by `CopySubrect`), so it was discarded. Instead PERF-001 is re-applied from `Build/gt-sport-ab-20261005/perf001.patch` plus `perf001-files`. PERF-001 already does this merging and the lazy uniform-state marker, keeps `last_state` exact, and has a randomized equivalence test against the original algorithm. Its Oct 5 revert came from a single dark-lighting session, and the later finding "lighting varies between sessions with identical code" (plus the readback root cause) explains that session. Player A/B on the same track still applies.
- File: `src/video_core/texture_cache/image.cpp` (`Image::GetBarriers`)
- Change: in the partial-transition path, emit one barrier per run of adjacent layers in a mip that share the same old state, instead of one per subresource. After the update, if every subresource has the same state, drop the per-subresource vector and go back to whole-image tracking, so later transitions take the fast path.
- Accuracy: none expected. A barrier covering adjacent layers with identical src/dst state is equivalent in Vulkan to the individual barriers, and collapsing a uniform state to whole-image tracking describes the same layout and access.
- Risk: low; an indexing mistake would show as validation errors or corruption, so test in the same tracks.

### PERF-001 re-applied (Oct 6)

- Status: Done. Release build OK. Tests: 548 run; the only failures are the 43 GcnTest GPU-instruction tests, the same count as the run before this change. All 505 others pass, including the `ImageBarriers` equivalence tests.
- Player check: same track and cameras as before. Lighting must match the current build (readback on). Report FPS in a race and in race previews.

### Player result, October 6 04:36 (PERF-001 re-applied)

- Lighting correct in every camera. FPS massively better: 40-50 in a race, 40-60 in time trial, race previews a locked 30 again (so the 15-18 FPS previews came from the barrier cost, not PERF-003).
- Crashes about 1 minute after starting a race (`Device lost during submit`).

### Finding: device loss follows a garbage descriptor address 0x3f80000000

- In all 15 device-loss runs on Oct 6, the last log line before `Device lost during submit` (2 lines earlier) is `UpdatePageWatchers: Tracking memory region 0x3f80000000 - 0x3f80000000 which is not fully GPU mapped`. The 3 runs that log it as their very last line end without a message (async log loss). Runs without it did not crash.
- 0x3f80000000 is the float 1.0 bit pattern (0x3f800000) in an address field. `cs_0xaa3822a3` reads its descriptor table from a GPU-written buffer (`gpu_modified=true`) whose CPU copy is stale, so its T#/V# words are garbage.
- `MemoryManager::IsValidGpuMapping` only checks the 40-bit limit, so the T# check accepts such addresses (an earlier log shows `Binding null image for unsupported T# format ... address=0x3f80000000`). `BindBuffers` does not check V# addresses at all. A garbage address then reaches the buffer or texture cache, which tracks and binds memory that is not GPU mapped.

### FIX-007: Reject descriptors whose base address is not GPU mapped

- Status: Implemented (Oct 6), Release build OK. New log line: `Rejecting V# outside GPU-mapped memory ...`; T# rejections reuse the existing `Rejecting invalid T#` line. Awaiting player run.
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp`
- Change: in `BindBuffers` (non-special V#) and `BindTextures`, if the base address page is not in the rasterizer's GPU-mapped ranges (`IsMapped(base, 1)`), bind a null descriptor (as already done for null and invalid sharps) and log the first 32 rejections with shader and address.
- Only the base page is checked, so ranges that run into unmapped memory (for example the 0xbf3d796000 region, which already logs "not fully GPU mapped" without crashing) behave as before.
- Accuracy: valid descriptors point into GPU-mapped memory, so they are unaffected. Only garbage descriptors change, from undefined to null.
- Risk: if the game legitimately uses a descriptor in memory that shadPS4 did not register as GPU mapped, that resource would read as null. The bounded log shows any such case.

### Player result, October 6 04:42 (FIX-007): regression, reverted

- No crash, but sun shading, interior lighting and car glare broken. Tokyo (night track) renders almost nothing (no textures or lighting). About 20 FPS.
- Log: about 2.25 million `Rejecting invalid T#` lines for textures at 0x29000000xx, from many fragment shaders. Plus V# rejections at base 0x2900000000 for `cs_0x8aa6e00c` (written, sizes 0x2d20000 and 0x13880000).
- Cause: 0x2900000000 is PRT aperture 0 (`sceKernelSetPrtAperture` id 0, 0x2900000000 + 0xc0000000). Aperture 1 is 0x3000000000 + 0x100000000. `SetPrtArea` registers each aperture as GPU mapped, but later `UnmapMemory` calls inside it (the game streams pages in and out) remove ranges from `mapped_ranges`. So `IsMapped` is false for legitimate PRT resources, and FIX-007 nulled them.
- The crash address 0x3f80000000 lies inside PRT aperture 1, so it is not necessarily garbage. FIX-007 probably stopped the crash only by nulling PRT resources.

### REG-002: Revert FIX-007

- Status: Done (Oct 6). Release rebuilt; no FIX-007 code remains in `vk_rasterizer.cpp`. Crash-diagnostic layer present in `Build/tools/VulkanSDK/Bin`.
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp`
- Change: remove the `IsMapped` checks that FIX-007 added to `BindTextures` and `BindBuffers`, restoring the PERF-001 build's behavior (approved lighting, about 40-60 FPS, crash about 1 minute into a race).
- Next: identify the faulting GPU command with `-CrashDiagnostics`, now that the crash reproduces within about a minute. Leading hypothesis: GPU access to a PRT page that is not backed.

### Player result, October 6 05:15 (REG-002 build, crash-diagnostics run)

- Crashed the same way: `Tracking memory region 0x3f80000000 - 0x3f80000000` two lines before `Failed to get master semaphore value: ErrorDeviceLost`.
- No crash-diagnostic report in `crash-dumps` or `user/log`, and the log has no "Requested layer ... not available" error. Either the layer was not active in this run, or it did not write a report before the emulator's assertion ended the process.
- The tracked region is empty (start == end), so the call that tracks 0x3f80000000 passes size 0.

### DIAG-017: Trace bindings on the 0x3f80000000 page

- Status: Implemented (Oct 6), Release build OK. Log lines: `Crash-page T# ...` and `Crash-page V# ...`. Awaiting a normal-speed player run (the crash reproduces in about a minute).
- File: `src/video_core/renderer_vulkan/vk_rasterizer.cpp`
- Change: in `BindTextures` (after the existing validity checks) and `BindBuffers` (non-special V#), log each binding whose base address is in [0x3f80000000, 0x3f80010000): shader, address, size or dimensions, format, tiling, and whether it is written. Log at most 64 lines, plus every 1000th after that.
- Why: identify the resource the GPU is given just before the device loss.
- Risk: none for rendering.

### Player result, October 6 09:35 (DIAG-017) and new screenshot report

- DIAG-017 caught the crash trigger: `Crash-page T# shader=cs_0xaa3822a3 address=0x3f80000000 type=Cube extent=1x15873x1 pitch=15361 layers=8 levels=1 tiling=24 data_format=9`, then `Device lost during submit`. It is a garbage descriptor from the stale table of `cs_0xaa3822a3` that passes every existing geometry check, yet Vulkan forbids it: cube images must be square, and a 1x15873 cube image is invalid.
- New player goal (verbatim): "we need graphics to be a 1:1 parity with the game on ps4, and then focus on performance". So accuracy work resumes; the Oct 5 accuracy freeze is replaced by a parity target.
- Screenshot issues (09:36-09:45, `user/screenshots`):
  1. Showroom (shots 3-6): hard-edged rectangular light and shadow regions in the upper-left area, with a vertical edge at x of about 960 (half of 1920) and horizontal edges near y 140-540. That suggests a half-resolution buffer read or written as a quarter of a full-resolution image (an aliasing or extent mismatch).
  2. Cockpit in the menu showroom (shot 2): interior almost black.
  3. Time trial at Tsukuba (shots 8-9): the sun is drawn as a cluster of rainbow pixels, a green haze spreads in the sky around it, and large red and green blobs sit over the HUD. That is the glare/bloom chain spreading corrupt bright pixels (the green glare issue seen before).
  4. Loading screen (shot 0): the car renders on pure black (needs a PS4 reference to confirm).

### FIX-008: Reject T# descriptors with an impossible guest size

- Status: Implemented (Oct 6), Release build OK. Tests: 550 run; only the 43 GcnTest GPU-instruction failures, as before. All others pass, including new `RejectsGuestSizeAboveFourGiB` (the exact crash descriptor) and `AcceptsLargeRealisticTextures` (16384x16384 RGBA8 and BC7). `tests/CMakeLists.txt`: the descriptor tests now compile `pixel_format.cpp`, with log and assert stubs.
- Files: `src/video_core/texture_cache/image_descriptor.h` (`CheckImageDescriptorGeometry`), `tests/test_image_descriptor.cpp`.
- Correction to the first plan: this renderer stores cube images as 2D arrays (no cube-compatible flag), so a non-square cube is not itself invalid in Vulkan. The real problem is size: pitch 15361 x 15873 rows x 8 layers x 4 bytes is about 7.8 GB, so uploading or detiling it runs far out of bounds.
- Change: new `GuestSizeLimit` error when pitch x height (in blocks for BC formats) x layers (or depth for 3D) x bytes per block exceeds 4 GiB. The descriptor is then bound as a null image, like other rejected geometry.
- Accuracy: no real PS4 texture comes near 4 GiB (the console has 8 GB in total), so valid descriptors are unaffected.
- Risk: low.

### DIAG-018: RenderDoc captures of the screenshot issues

- Status: Requested from player.
- Tool: `Build/Run-GT-Sport-Capture.ps1 -BuildType Release` (RenderDoc 1.46). The profile already has `readback_linear_images_enabled: true`, so captures match what the player sees.
- Captures wanted (F12 in game; files land in `Build/gt-sport-fixed/CUSA03220*.rdc`):
  1. Brand Central showroom (Toyota 2000GT) showing the rectangular light/shadow seams.
  2. Showroom cockpit view (dark interior).
  3. Tsukuba time trial in cockpit view with the rainbow sun and the red/green glare blobs.
- Analysis with `Build/tools/capture-inspect` follows: find the pass that writes the seam region, the extent and alias of the half-resolution target, and the first pass where the glare blobs or sun pixels go wrong.

### Player captures, October 6 09:55 (DIAG-018)

- 9 captures (`user/captures/CUSA03220_capture_25..33.rdc`). The run did not crash; FIX-008 rejected no descriptors in this session (`GuestSizeLimit` count 0).
- 25-26: car loading screen; the car does not render (the open loading-screen issue).
- 27-29: Tsukuba cockpit. Sun, glare and HUD look correct; no rainbow sun or red/green blobs. So the glare corruption does not reproduce under RenderDoc, which points to a timing-dependent cause (synchronization), not a deterministic shader error.
- 30: Tokyo Expressway event menu. 31: black. 32-33: Tokyo in race, 3D scene completely black (HUD only).
- Showroom seams and dark cockpit interior were not captured.
- Tool: `capture-inspect` gained `thumb` (embedded thumbnail, no replay) and `nan-minmax` (replays an event with the fragment shader's GLSL FMin/FMax/FClamp swapped for NMin/NMax/NClamp) modes.

### Finding: Tokyo is black because NaN persists in the HDR scene target

- Capture 33, target 62954 (1920x1080 R11G11B10, guest 0x1005000000). The scene pass begins with `loadOp = Load` (event 2943); nothing clears the target. At event 3000, 1,915,034 of 2,073,600 pixels are already NaN (carried over from previous frames), and 1,343,699 remain NaN after the opaque scene (event 3832).
- Event 6099 (atmospheric fog/sky full-screen pass; reads depth 75350, the HDR target itself and a 512x256 BC6H map) bilinearly samples the HDR target at texel corners and copies it through for sky pixels, so NaN neighbours spread every frame. A pixel that was 2.44 before 6099 becomes NaN after it. Every later pass inherits NaN, and the final image is black.
- The NaN seed predates the capture. Replaying event 6099 with NMin/NMax/NClamp does not clear existing NaN (expected, since the input is already NaN), so the capture cannot confirm the seed.
- Hardware never produces this NaN. Known emulation differences that can seed it: GCN `v_min_f32`/`v_max_f32` return the non-NaN operand (IEEE and non-IEEE mode alike), and the VOP3 clamp modifier maps NaN to 0. The recompiler emits SPIR-V `FMin`/`FMax`/`FClamp` (NaN result undefined), and these shaders declare `SignedZeroInfNanPreserve`, so the NVIDIA driver keeps NaN.

### ACC-005: GCN NaN semantics for float min, max, med3 and clamp

- Status: Implemented (Oct 6), Release build OK. Covers FPMin/FPMax (32/64), FPClamp (32/64), FPSaturate (32/64, the VOP3 clamp modifier), the MinTri/MaxTri/MedTri fallbacks, and the unorm/snorm pack clamps (`emit_spirv_bitwise_conversion.cpp`). Clamp is written as NMin(NMax(x, lo), hi), the GLSL NClamp definition (sirit has no OpNClamp). `ShaderBinaryVersion` 13 -> 14. Tests: 550 run; only the same 43 GcnTest GPU-instruction failures.
- Files: `src/shader_recompiler/backend/spirv/emit_spirv_floating_point.cpp`, `src/video_core/renderer_vulkan/vk_pipeline_serialization.cpp` (cache version bump).
- Change: emit GLSL `NMin`/`NMax`/`NClamp` instead of `FMin`/`FMax`/`FClamp` for FPMin/FPMax/FPClamp (32 and 64 bit) and for the non-AMD fallbacks of MinTri/MaxTri/MedTri. The `VK_AMD_shader_trinary_minmax` path is unchanged (not available on this GPU). Bump `ShaderBinaryVersion` so cached SPIR-V is rebuilt.
- Accuracy: identical results for all non-NaN inputs. With a NaN input, the result now matches GCN (the other operand, or 0 for clamp to [0,1]).
- Risk: low; possibly slightly more ALU on some drivers. Needs player check at Tokyo (night) and on the approved tracks for lighting regressions.

### REG-003: Restore the state of the ACC-005 hand-off (player request)

- Status: Done (Oct 6, 11:21).
- Player asked to restore the code to the state at the ACC-005 hand-off message. That state is commit 97e4157f: FIX-008 and ACC-005 in, `ShaderBinaryVersion` 14, log ending at ACC-005.
- Discarded the later uncommitted work, which was not made through this log's hand-off: TEST-001, ACC-006, ACC-005b, ENV-001, INV-001, DIAG-019/019b. Files: `src/common/logging/log.cpp`, `emit_spirv_floating_point.cpp`, `scalar_alu.cpp`, `vk_pipeline_serialization.cpp` (version 15 back to 14), `tests/gcn/gcn_test_runner.cpp`, `tests/gcn/test_gcn_instructions.cpp`, and their log entries. Saved first to `Build/post-message-changes-20261006-112148.patch` (`git apply` restores it).
- Release rebuilt from the restored source.

### Finding: Tokyo NaN lives in a lighting-probe feedback loop seeded by all-ones texels

- Captures 27-29 (Tsukuba): the HDR target 62954 and every probe texture (32x32 cube 62951, 256x256x8 cube array 75960, probe arrays 171549/171551/225279/232232) have no NaN or Inf at any sampled event.
- Capture 30 (Tokyo event menu) already starts with 62951 fully NaN and 75960 at 379,794 NaN texels (6 cube faces of mip 0 by capture 31). So the poison appears between leaving Tsukuba and the Tokyo menu, before any capture.
- Chain in capture 30 (`capture-inspect ... usage`, new mode): draws 1805-1895 render the environment cube 406792, sampling (among others) the Tokyo-only array 338678 and the previous ambient cube 62951; compute 1904-2016 filters 406792; compute 2023+ writes 75960; compute 2353/2359 rebuild 62951 from 406792. The ambient cube feeds its own next update, so one NaN persists forever. This also fits lighting going wrong when a car loads (probes are rebuilt then).
- 338678 is 64x64, 2048 layers, 7 mips, R11G11B10, guest 0x1081e4c400 size 0x2c00000, sampled by the env-cube pixel shader (binding 17, next to a 4x4x48 R32G32_UINT table at 0x201603c00 that looks like an indirection table). 7,688,439 of its 8,388,608 mip-0 dwords are 0xFFFFFFFF (NaN in all three channels); 532 layers fully, 1436 partly. Identical in captures 30 and 33. It is only ever read (PS_Resource) in these frames, so its contents come from a guest-memory upload, not GPU rendering.
- Unknown: whether PS4 memory at that address also holds 0xFFFFFFFF (and the game avoids sampling it), or whether the emulator misses the real data (stale image, DMA fill ordering, or layout). DIAG-020 is built to decide this.
- Ruled out: GT Sport's 5,027 dumped shaders contain no V_MUL/MAC/MAD/MIN/MAX_LEGACY, RCP/RSQ_CLAMP or LEGACY opcodes (scanned with a GCN decoder; 0 undecodable), so the IEEE translation of the legacy multiply-adds is not the cause.

### ACC-007: V_CVT_PKRTZ_F16_F32 rounds toward zero

- Status: Implemented (Oct 6), Release build OK. `ShaderBinaryVersion` 14 -> 15. The bit logic was checked against an exact round-toward-zero reference on 200,008 inputs (0 mismatches). New GcnTests `cvt_pkrtz_*` added, but the GcnTest GPU runner returns 0 for every test on this machine (even `add_f32`), so they cannot pass here.
- Evidence: LLVM's AMDGPU instcombine folds `llvm.amdgcn.cvt.pkrtz(65535.0)` to half 65504 (`amdgcn-intrinsics.ll`, `constant_rtz_pkrtz`), i.e. IEEE round-toward-zero: finite overflow saturates to +-65504.
- Before: the recompiler emitted GLSL `PackHalf2x16` (rounding undefined; NVIDIA rounds to nearest), so values >= 65520 became +-Inf. Also, `Unpack(Pack(x))` folds to `x` for compressed exports, so the attachment received the raw f32 and the host conversion decided (Vulkan allows Inf for out-of-range values).
- Change (`vector_alu.cpp`): truncate each f32 to the f16-representable value toward zero (normals keep 10 mantissa bits, f16 denormals fewer, below 2^-24 -> +-0, finite overflow -> +-65504, Inf/NaN unchanged), then pack. The pack is then exact on any host.
- Used by 2,963 shaders (VOP2) and 1,917 (VOP3) in the dump. Not shown to be the Tokyo seed; it removes one documented Inf source in HDR output.

### DIAG-020: where the all-ones lighting texels come from

- Status: Implemented (Oct 6), Release build OK. Warning-level log lines, bounded.
- `DIAG-020 upload`: each upload of a float image of 1 MB or more (B10G11R11, RGBA16F, RG16F, RGBA32F, E5B9G9R9): address, size, extent, layers, mips, format, tile mode, whether the data came from guest memory (`cpu`) or the buffer cache (`gpu`), and the share of 0xFFFFFFFF dwords in guest memory. First 400 lines without all-ones, up to 2000 with.
- `DIAG-020 stale`: every 5 s per CPU-sourced clean image, rehash guest memory; log once per image if it changed without an invalidation (the image would keep old data).
- `DIAG-020 fill`: every DMA fill of 64 KB or more: address, size, value, and whether it ran on the CPU fast path or the GPU (first 500).
- Files: `texture_cache.cpp`, `image.h`, `vk_rasterizer.cpp`. Remove after the run.
- Reading the result: upload `cpu` with high all-ones and no stale line means guest memory really holds 0xFFFFFFFF (then look at who should have written it). A stale line means the emulator missed the real data. A large `fill ... value 0xffffffff` covering the address shows the fill that wrote it.

### Player result, October 6 12:42 (ACC-007 + DIAG-020 build)

- Race lighting fine, including Tokyo (afternoon). The Tokyo 2048-layer lighting array was never uploaded in this session (no DIAG-020 line for it), so the Tokyo fix is not proven; the broken captures were at night.
- New reports: race preview flickers and the leaderboard rows change color; garage car not drawn on the transmission screen; square glitch top-left when picking a rental car; headings show "R" as "9" ("9ANKING BOA9D").

### FIX-009: write newer cropped render targets back into the full image

- Capture 25 (car loading screen): the car is drawn at event 3969 into 195763, a 1200x1080 crop of the 1920x1080 target 62957 (same address 0x1006bc8000, same pitch and size 0x7f8000). At 4113 the game blends depth of field over it through the full-width descriptor 62957. The emulator bound its own stale 62957 (no car), then at 4118 copied 62957 over 195763 (crop refresh), erasing the car. The composite at 4514 reads 195763, so the car is black.
- Change: `FindImage` now copies any crop of the requested image that is newer (`contents_version`) back into it, oldest first, before use; `Runtime::CopySubrect` copies in either direction and only the crop rectangle (the full image keeps its other pixels).
- Files: `texture_cache.cpp`, `vk_runtime.cpp`.

### FIX-010: keep the mip found for a render target when later overlaps keep the same image

- Capture 5 (rental car, Toyota 86): the top-left square comes from the depth-of-field chain 26691 (960x540, 8 mips, 0x1009e00000). Draws 3616/3629/3642 sample mip 0 and render the 480x270, 240x135 and 120x67 mips, but they are bound to a mip-0 view (the only views of 26691 are levels 0:7 and 0:0), so they paint shrinking copies over the top-left of mip 0. Mips 4-7 got their own images (26698-26704) at the end of the chain.
- Cause (upstream code): `FindImage` resets `view_mip`/`view_slice` to -1 for every overlapping image, so a later overlap that returns the same image drops the mip found by `MipOf`. Inferred from the code and the missing views; `DIAG-021 kept mip` logs each case the fix now keeps (first 50).
- Change: only replace the mip/slice when the overlap returns a different image or a subresource.
- Possibly also related (not shown): leaderboard panels changing color and preview flicker, if those sample blurred mips.

### Player result, October 6 13:35-16:30 (FIX-009 + FIX-010 build) and captures

- Rental car top-left square: fixed (player confirmed). Transmission screen: car area now flat grey instead of black. Race preview: still flickers (less), ranking rows still change color.
- Startup takes about 3.5 minutes: `PipelineCache::WarmUp` builds all 17,718 cached pipelines on the main thread, so the window does not respond. Not a hang.
- The Tokyo 2048-layer lighting array (now 131314 at 0x10782e1400) has no all-ones or NaN texels in this session's capture 8 (5.7% zero words, real data).

### Finding: ranking-board rows are a frosted-glass blur of the scene behind them

- Capture 9 (old numbering, 12:54 run), row draw 19728 samples a 142x48 crop of a 142x96 blur target in the 0x1009bc0000 scratch pool; the blur reads the display buffer behind the row. Row colors follow the scene, so they go flat grey when the scene is missing. Unknown: whether the PS4 tints them the same way.

### Finding: race preview grey/black frames do not reproduce in replay

- Captures 8 and 10 (16:29 run): thumbnails (the frame as presented live) are black/grey, but replaying the same captured commands gives a correct frame: scene, TAA, composite at 18249 and UI are all present in the display buffer, and the presenter reads that buffer. Same commands, different live result, so the cause is timing-dependent (missing synchronization or a race), not shader math. Same conclusion as the earlier glare finding.
- Next: a run with Vulkan synchronization validation (`Build/Run-GT-Sport.ps1 -Diagnostic`) to list hazards on the preview screen.

### Finding: transmission screen grey comes from a stuck TAA history

- Captures 1-5: the car renders into 26082 and its 1200x1080 crop 57860 (copy 3742) correctly. The TAA pass (fs_0x184b619c, viewport x 720-1920) outputs one constant value over the whole car area in both ping-pong parities (captures 1/3/5 write 0x10083b0000, capture 4 writes 0x10073c0000), while the same shader works for the cockpit view.
- Lead (unconfirmed): the TAA shader's output channel order differs from the morning build (z and w swapped: old `(s.w, s.z, s.x, s.y)`, new `(s.w, s.z, s.y, s.x)` in the copy-through path). GCN code: `v_cvt_pkrtz v0, v0, v1; v_cvt_pkrtz_e64 v1, v2, 1.0; exp mrt1 v0 v1; v_cvt_pkrtz v2, v2, v3; exp mrt0 v0 v2`. Not resolved which order is correct.

### FIX-011: startup precompile with progress, persistent driver pipeline cache

- Player request (Oct 6): cleaner window title and no long startup delay; precompile shaders before the game starts, with a visible notice.
- Cause of the slow, frozen startup: `PipelineCache::WarmUp` built every cached pipeline on the window thread with no event pumping, and it ran before the `VkPipelineCache` was created, which was never saved anyway, so the driver recompiled everything on each launch (3.5 minutes for 17,718 pipelines; about 65 ms each on the fresh cache).
- Changes:
  - `vk_pipeline_cache.cpp`: create the `VkPipelineCache` before `WarmUp`, from `user/cache/<serial>.vkpipelinecache` when its header (version, vendor, device, pipeline cache UUID) matches this GPU and driver. Save it (temp file + rename, on a background thread) after the precompile, every 1,000 pipelines during it, every 64 new pipelines (at most every 30 s) during play, and on shutdown.
  - `vk_pipeline_serialization.cpp`: graphics pipelines found in the cache are built on up to 13 worker threads (inputs copied per pipeline); the window title shows "Compiling shaders N / M (P%)" and window events are pumped, so the window stays responsive.
  - `emulator.cpp`: fork builds use the title "<game> - Current Build" (revision still logged).
- Measured on the player's PC (6,241 cached pipelines): first launch about 50 s, second launch 10 s with the saved driver cache; window responsive throughout (0 not-responding samples).

## Performance work, October 7, 2026 (target: 60 FPS without losing accuracy)

### Measurements

- Profile (Time Trial, 30-40% CPU and GPU use): no thread saturated. The GPU command thread was about 74% of a core; the game waits on it.
- Flip pacing (PERF-DIAG-002): submit-to-flip about 9 ms, Present under 0.5 ms, no waits for free presentation frames. Most vblanks simply had no new frame, so display pacing is not the limit.
- Ceiling run with `-ReadbacksMode 0 -ReadbackLinearImages 0`: 55-60 FPS for long stretches (log frame counts near 120 per 2 s). Readbacks off but linear images on: 26-40. Normal settings: 30-45. The readback paths cost the frames.
- GPU waits by source (PERF-DIAG-001): in races nearly all are `buffer_cache.cpp` downloads, about 250 per second. They come from 5 small regions the game CPU writes (4-8 byte stores) about 25 times per frame in total, in pages the GPU also writes. Each store forces the GPU thread to drain the GPU and download a 512 KB window. GT Sport runs 63 job threads; the emulator does not cause that (the game never queries a CPU count).

### Kept (committed)

- DIAG-020/021 removed (DIAG-020 copied and scanned large float images on every upload).
- PERF-002 re-applied: guest USB `timeval` converted to the host layout (the Thrustmaster wheel thread no longer busy-polls a core). Earlier "lighting regression" attributed to it was later traced to readback timing.
- PERF-006: `FindImage` checks images at the same base address first and walks the whole range (every 256 KB page of large targets) only when there is no perfect match. Exact: a perfect match and all of its crops share the base address. Its self time was about 10% of the GPU thread.
- PERF-007: with linear-image readbacks on, chunked submits (every 128 draws) only happen while a readback or deferred fence is pending; otherwise they were pure overhead (submits were about 22% of the GPU thread). Menus and pre-race now hold 60 FPS; races about 30-40 (13-55).
- PERF-DIAG-001/002 left in: GPU waits per call site, flip pacing and render-frame waits, each reported every 2 s at warning level.

### Tried and reverted

- PERF-005 (deferred compute-queue ReleaseMem fences): no gain; its test run also ran with readbacks 0 by accident (the earlier `-ReadbacksMode 0` run had not restored the profile), which removed the sparks.
- PERF-008 (emulate small CPU stores into GPU-modified pages and update the GPU copy inline): about 550 stores per second took the path, each becoming a GPU `updateBuffer` that ended render passes; races dropped to 4 FPS. Removed.

### Next (experimental branch)

- Track GPU writes precisely enough that CPU stores into GPU-owned pages do not force a full drain.

## Experimental branch `experimental/precise-gpu-write-tracking` (October 7, continued)

Lance chose (October 7) to keep this branch separate from `main` until it reaches adequate speed, then to work toward 60 FPS in races through a multi-core redesign of command processing. Each run's log is kept under `Build/gt-sport-fixed/runs` with a summary from `scripts/Analyze-GTSportRun.ps1`.

### Measurements (20-car race)

- Per frame: about 4,000 draws, 25,000-38,000 PM4 packets, 15,000-50,000 texture lookups, 5,000-13,000 buffer binds (PERF-DIAG-010).
- GPU busy 20-35% (PERF-DIAG-007, sampled from the scheduler's timeline semaphore). The GPU command thread is the limit; with `-ReadbacksMode 0` races were also 20-30 FPS, so readback waits are not the main cost.
- Draw steps on the command thread (PERF-DIAG-011, before PERF-012): buffer binding about 25 of 38 ms per frame; Vulkan command recording (descriptors, dynamic state, draw) about 3 ms. Moving recording to another thread would gain little, so the redesign starts with buffer binding.

### Kept

- PERF-009: GDS-to-memory copies are read back asynchronously with the next deferred fence.
- PERF-012: pages the CPU rewrites constantly (8 write faults) stay unprotected and CPU-modified, uploaded at most once per upload epoch. Epochs start at guest submissions, after GPU waits on memory, and after command processor writes to guest memory. GPU writes return such pages to normal tracking. Race: buffer binding 25 -> 5 ms per frame, protection calls 6,800 -> 40 per frame, average FPS 26 -> 36 (10th percentile 11 -> 22). No visual changes reported.
- PERF-013: runtime shader permutations start after the highest stored index for their program, so they no longer overwrite stored shaders. About 1,085 cached pipelines had been rejected at every start and recompiled in races; after the fix about 20-30 pipelines compile per run.
- Diagnostics PERF-DIAG-006 to 012 (fault pages, monitor, frontend waits, work counters, draw step timing, drain writers).

### Tried and disabled

- PERF-010 fence deferral for hot pages: deferring fences until page readbacks landed made races slower (14-18 vs 20-35 FPS). Page readbacks now only ride along with fences.
- PERF-011 (skip GPU-written bytes in uploads instead of draining on CPU write faults next to them): fine with 64-byte granularity, but with exact store sizes (PERF-011b) it fired every frame and produced blue, exploded vertices around the track. Without byte-level CPU write tracking it can lose CPU stores to excluded bytes, so it is disabled.

## Shader precompilation removed (October 9, branch `mcp-server`)

Lance asked (October 9) for the shader precompilation to be deleted completely. Removed:

- `PipelineCache::WarmUp` (upstream's preload of every stored pipeline at startup) and its
  loaders `LoadGraphicsPipeline`, `LoadComputePipeline` and `LoadPipelineStage`.
- FIX-011's worker-thread precompile (`PreloadQueue`) and the "Compiling shaders N / M"
  window title.
- PERF-024's background build of stored pipelines: the low-priority worker queue, the
  `preloading` builds and their promotion by the read-ahead, and the `-DisablePerf 24` switch.
- The `preloading` constructor argument of graphics and compute pipelines (only used for
  pipelines built from stored data).

The shader database (`user/cache`) is no longer opened, so it is neither read nor written and
the files stay as they are (cache format versions unchanged). Every pipeline compiles the
first time a draw needs it.

Kept: the driver pipeline cache (`<serial>.vkpipelinecache`, loaded at start and saved
during play and on exit), the read-ahead builds (PERF-019 to 023) and the shader cache's
serialization code.

## Car thumbnail BREAK! (October 9, branch `diag-rework`)

### Finding: a shader-compile stall loses the thumbnail script's only wait

`THUMBNAIL.ADC` calls `iconShotImage`, sleeps 0.1 s (`Thread::Sleep`, measured with
`sceKernelGetProcessTimeCounter` / `Frequency`), then asks `checkTickEntry` whether the shot is
pending. The request only becomes pending at the game loop's next scene update, and the game loop
runs three frames ahead of the flips. When the GPU thread stalls for 100 ms or more right then,
the check sees nothing pending. `waitTickEntry` returns at once and the mask commands go out
before the colour capture, so the PNG is about 4 KB and the debug build shows BREAK!.

On a PS4, where shaders are precompiled, 0.1 s is about six game-loop updates. All 9 bad thumbnails
on October 9 have a 583–4900 ms frame 5–7 frames after the 800x450 render starts. That stall is
mostly about 127 shaders translated again as new permutations (only translation counted; the
time base itself is correct). The good one (13:06) has no frame of 100 ms or more there.

### FIX-043 (perf id 44): permutations keyed only on what translation read

The resource patching pass records, per translated shader:
- which buffers' strides entered an address (indexed, thread-id or swizzled addressing);
- which images' sRGB format decided a forced degamma (the only use of `is_srgb`).

`StageSpecialization` then:
- leaves the other strides and sRGB flags out of the key;
- records the V# translation saw for an unbound buffer instead of zeros;
- adds thread-id addressing and coherence (mtype 3), which the code reads but the key missed.

Shaders loaded from storage carry no usage flags and keep the full key. An in-session diagnostic
counts new permutations whose SPIR-V repeats an existing one.

Offline check against the October 6–9 shader store (58,707 permutations): 8,237 repeat another
permutation's SPIR-V, and 100 pairs differ only in sRGB without a degamma sampler, all identical.
For the 127 shaders of the thumbnail stall, 178 of their 924 stored permutations would have
matched an older one. The in-session repeats are not measurable offline (the store is no longer
written), so the new log line measures them in the next run.

### FIX-044 (perf id 45): guest clocks held while the GPU thread compiles

These stand still while the GPU command thread translates a shader, builds a pipeline or waits
for a pipeline build:
- `sceKernelReadTsc`, `sceKernelGetProcessTimeCounter` and `sceKernelGetProcessTime`;
- the monotonic and uptime `clock_gettime` clocks;
- the vblank count, vblank events and flips.

Real time is unchanged (`gettimeofday`, REALTIME, the network clock), and so are host waits
(`usleep`, `nanosleep`, timed condition waits, equeue timeouts).

Replaying the logged stalls through the held clock: 8 of the 9 bad thumbnails fall to 28–91 ms in
the critical window and the good one stays good. The 13:08 one still has a 102 ms frame. That
frame comes right after its 4.9 s compile and is spent waiting for the host GPU, which is not held.

Risks checked:
- The SDL/OpenAL audio backends paced the host device with `sceKernelGetProcessTime`. They now
  pace with host time (`GetHostProcessTime`); the output time the game reads stays guest time.
- AvPlayer runs on host time, so a movie keeps playing through a hold, as before. Its video and
  audio stay together.
- Guest threads that sleep in host time and then read a held clock see no time pass. That is
  harmless for polling loops, but code dividing by elapsed time would see zero.
- `DisablePerf 45` restores the old clocks.

The shader database was disabled at Lance's request (precompile removal above), so every session
translates every permutation again, including ones an earlier session already built for the same
car. Re-enabling it (on demand, without the startup precompile) would remove those. That is
Lance's decision.

## PERF-031: draw pipe, decoding and recording on two threads (October 9, branch `perf-parallel-gpu`)

Lance asked (October 9) for 60 FPS work following the Bloodborne PC port's parallel GPU design
(bbport, GPL-2.0-or-later, `docs/parallel_gpu.md` at https://github.com/deadinside28/bloodborne_pc).
Its two-stage pipeline gained 18-19% there. The design is borrowed; the code is written for shadGT.

### Starting point (race, October 8, 39 FPS)

- About 1,530 draws per frame. The GPU is about 34% busy; the command thread is the limit.
- The command thread spends 17 ms per frame on draws: buffers 6.5 ms, pipeline 3.9, textures
  2.3, vertex/index 1.8, and render passes, descriptors and recording about 2.3.
- On top of that: PM4 decoding (about 22,600 packets per frame) and waits.

### GT Sport's packet mix (two dealership bundles, 3 frames each)

- No constant-engine packets.
- About 650 DMA_DATA per frame, one per draw. They are memory writes, so they run in order on
  the recorder rather than forcing drains.
- 16 WAIT_REG_MEM per frame on the graphics queue and 11 on compute; about 18 EVENT_WRITE,
  12 EOP and 5 EOS.
- Between WAIT_REG_MEMs there are runs of about 37 draws in which the two threads overlap.

### Design

- **Stage A, the command thread:** keeps PM4 decoding and the register file. Every register
  write marks its 32-dword block (`RegsDelta`).
- **Hand-off:** each draw and dispatch sends the blocks written since the previous one, plus
  the CB/DB extents and, for dispatches, the queue's compute registers.
- **Stage B, the recorder thread (`shadGT:GpuRecorder`):** applies them to its own register
  copy and runs the unchanged `Rasterizer::Draw`/`Dispatch*`. Rasterizer and pipeline-cache code
  reads registers through `Liverpool::DrawRegs()`/`DrawCsRegs()`/`DrawCbExtent()`.
- **In order on the recorder:**
  - DMA_DATA and WRITE_DATA;
  - EOP, EOS and RELEASE_MEM fences;
  - MEM_SEMAPHORE signals and occlusion results;
  - debug markers and flip IRQs;
  - submit-done flushes, the GPU-idle IRQ, and the end of a submission (`num_submits`), so
    "idle" still means recorded. The command thread counts its own decoding in `num_tasks`.
- **Drain first, then run on the command thread:**
  - WAIT_REG_MEM, COND_EXEC and MEM_SEMAPHORE waits, only when not already met;
  - CE dumps;
  - commands other threads send to the GPU thread (flips, CPU fault flushes, unmaps);
  - a fault on the command thread;
  - copies or writes into a command buffer still being decoded.
- **Thread identity:** the recorder counts as a GPU thread for fault handling.
- **Off while pipelined:** the pipeline read-ahead (PERF-019 to 023), because it reads the
  command thread's state.
- **FIX-044:** the recorder is marked as a GPU command thread, so its compiles hold the guest
  clocks.

### Switches and checks

- `SHADGT_DRAW_PIPE=1` turns it on (off by default until measured). `-DisablePerf 46` forces it
  off.
- `SHADGT_DRAW_PIPE_VERIFY=N` sends the full register file with every Nth draw and logs any
  register the recorder had stale (a write that was not marked).
- Every 2 s, a `PERF-031 draw pipe` line reports:
  - jobs, recorder busy % and maximum queue;
  - drains by reason, with the time the command thread waited.
- Unit tests (`shadps4_draw_pipe_test`):
  - register deltas reproduce the register file over 2,000 random draws;
  - work runs in order on the recorder thread;
  - drains wait for running work;
  - backpressure keeps order;
  - results written before a drain are visible after it.

### Expected and open

- Recording stays the larger stage, so the first gain is the overlap of decoding, waits and
  pipeline-cache hits with recording. That is roughly what bbport saw from this step (+18-19%).
- bbport's later steps go further:
  - pipeline selection and descriptor reads on stage A;
  - texture and render-state memo caches.

  They need stage A to read guest memory ahead of recorded writes, with pending-write tracking.
- bbport reported a heap-corruption crash with its pipe enabled (open there).
- Diagnostics that read the command thread's history from the recorder (DIAG-009/010 const-dump
  and command-buffer lookups) can be inaccurate while pipelined.

## PERF-032: shader store used again, without the startup precompile (October 9, branch `shader-cache`)

Lance asked (October 9) for the shader cache back on without the startup precompile he had removed.

- **Loading on demand:** the store (`user/cache/CUSA03220`) is opened when the pipeline cache
  starts, and nothing else happens at startup. The first time a draw uses a program, its stored
  permutations are listed from their `.meta` files at their original indices; stale versions and
  ones translated before a needed FIX-018/020 fix are skipped. A listed permutation's SPIR-V is
  read and its module created only when a draw matches its key.
- **Storing:** new translations are stored as before (`.meta`, `.spv`, pipeline `.key`).
- **Meta version 16:** adds FIX-043's resource usage after the shader info. Without it a stored
  permutation, whose key leaves unused strides and sRGB flags out, would never match the key
  rebuilt from the loaded shader. Versions 14 and 15 still load as translations of unknown usage,
  keyed on every property, as they were stored.
- **Damaged files:** stored SPIR-V must be whole (magic, instruction word counts ending exactly
  at the end, OpFunctionEnd last). A cut-short `.meta` is read under FIX-034's recoverable scope.
  Either kind is translated again instead. All 58,707 stored modules of October 9 pass; their
  metas are versions 14 (13,447) and 15 (45,260).
- **Switch and diagnostics:** `-DisablePerf 47` leaves the store closed. Only the directory form
  is used (`pipeline_cache_archived` off). `PERF-032` lines report programs looked up,
  permutations listed and modules loaded instead of translated.

## Race measurements of PERF-031/032, and PERF-033 (October 9, branch `perf-parallel-gpu`)

Lance's runs (16:44 pipe + cache, 16:54 cache) ran races at about 24 FPS in both.

### Steady race windows (more than 1,200 draws per frame, no compiles)

| Run | FPS | µs per draw | Buffers, µs per draw | Uploads per frame |
|---|---|---|---|---|
| October 8, fast race | 45 | 8.8 | 2.2 | 24 MB |
| October 9, both builds | 24 | 15-16 | 7.4-8.2 | 127-134 MB |

- Upload volume decides race FPS. Across all runs since October 7 it ranges from 1 to 155 MB
  per frame with the same code, depending on the race, and FPS follows it.
- In today's races: about 14 upload epochs per frame of about 9.5 MB each.
- CPU write faults are only about 50 per second, so nearly all of it is PERF-012 hot pages
  uploaded again whole in every epoch.

### Draw pipe in the race

- The recorder was 70-99% busy.
- The command thread spent 1.3-2.0 s of every 2 s waiting at WAIT_REG_MEM: 3,000-9,000 drains
  per 2 s. Those waits are on fence labels that the recorder writes later, so almost every one
  found them unmet and drained.

### PERF-031 follow-up: recorder-side waits

- A fence or WRITE_DATA job queued to the recorder notes its address and value until it runs.
- A WAIT_REG_MEM that such a pending value satisfies becomes an in-order recorder job:
  - it waits as the command thread did (PERF-014's pending-deferred-fence check, then a flush
    and polling until the scheduler's completion thread writes the label);
  - decoding goes on instead of draining.
- Waits on anything else still drain.

### PERF-033 (perf id 48): unchanged hot pages are not uploaded again

- Each hot page gets a generation that changes every time it becomes hot. Leaving the hot set,
  which a GPU write causes, voids earlier records.
- After an upload, the hash of the uploaded bytes is kept.
- A hot page is left out of an upload when it is in the same generation and its bytes hash the
  same.
- Hashes are taken before and after the copy, and a page written during the copy gets no record.
  Pages partly kept from guest bytes (PERF-011/015) are always uploaded.
- XXH3 of 4 KB pages runs at 45-49 GB/s here, about 3 ms per frame for a 10 MB hot set
  checked 14 times. The upload path costs about 13 ms per frame for 130 MB.
- A 2 s `PERF-033 hot pages` line reports pages not uploaded and the MB saved.
