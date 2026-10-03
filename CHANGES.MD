# Gran Turismo Sport compatibility changes

Updated: October 3, 2026. Base revision: `8e23388af83fe31ec6372a29e3940e2b459dfc6d`.
Tested title: Gran Turismo Sport, CUSA03220, update 1.69.

Campaign races now load in the user's test. The user also confirmed that rapidly
cycling through all camera views no longer crashes. These results do not establish
long-term gameplay stability or error-free rendering.

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
