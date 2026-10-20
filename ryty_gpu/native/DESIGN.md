# Native renderer for GT Sport (experimental)

Goal: render GT Sport without emulating the PS4's command processor. Today the game writes PS4
GPU commands (PM4), submits them, and the renderer decodes them on one thread and records Vulkan
commands on a second one: about 13 us per draw for ~3,800 draws per race frame, all serial. The
native renderer records each draw where the game issues it, on the game's own threads, and keeps
the backend (Vulkan first, DX12 second) behind one interface.

## What the game looks like (eboot v1.69, static analysis)

Addresses are eboot.bin addresses; GTSport.exe RVA = address + 0x10000.

- **Layers.** GTRender / PDIScene (scene) -> PDIGraphics (Texture, TextureSet, FrameBuffer,
  ModelSet) -> **PGL** (Polyphony's OpenGL-like library, 0x1c50060-0x1c7806c, 417 functions)
  -> Sony's static Gnm/Gnmx library (0x1b0000-0x1e0000, 1,367 functions) -> libSceGnmDriver
  (our HLE).
- **Every draw** goes through one of five PGL functions, which tail-jump into Gnm's
  drawIndexAuto (0x1b4890) or drawIndex (0x1b4aa0) wrapper; nothing else reaches those:
  drawArrays 0x1c5a750, drawArraysInstanced 0x1c5a890, drawTriangles32 0x1c5a9e0,
  drawElements 0x1c5ab00, immediate-mode draw 0x1c621b0. Indirect draws are never used.
- **No hand-written commands.** Scanning all 111,918 functions for PM4 packet headers finds them
  only in the Gnm library, plus PGL's INDIRECT_BUFFER links between command chunks. Nothing is
  prerecorded and replayed. So the Gnm library's ~110 externally used command functions plus
  the PGL entry points are a complete input boundary, and every one is a real function.
- **Compute** goes through four Gnm dispatch wrappers (0x1b53c0 graphics ring, 0x1d97d0 compute
  rings, 0x1cafa0, 0x1db650); compute rings are kicked with DingDong (0x1d8b90).
- **Submission.** One graphics submit per frame, from PGL 0x1c56610 (called by the submit thread
  0x1834210).
- **Job tree (found with the check races, NAT-005..009).** Jobs are forked: PGL 0x1c56770
  clones the current context into a child (so the child assumes the parent's GPU state), ends
  the parent's command run with a 16-byte link slot that will lead into the child's commands,
  and the parent goes on right after the slot (or in a fresh chunk), linked after the child's
  last chunk through a link object (word handshake, xchg). The GPU runs the tree depth-first:
  parent part 1, child, parent part 2, ... The frame's root context (0x1431cd240, not a job)
  writes the frame setup and forks the frame's jobs from 0x1833d43. A job's own commands are
  therefore read by skipping each fork's slot, starting from the parent's state at the fork;
  jobs never moved between threads in the races, but ~1-2% of job ends are not the innermost
  open job on the thread.
- **Render jobs.** A job takes a PGL context from a pool (0x1c572c0), makes it current on its
  thread (TLS slot fs:[0]-8) with a fresh command chunk and state copied from a template, records,
  and ends (0x1c57860), linking its chunk after the previous job's. Jobs run in parallel on the
  game's worker threads, and each starts from known state, so each job can be recorded into its
  own native command list independently.
- **PGL context** (selected fields): DrawCommandBuffer +0x780 (+8 end, +0x10 write pointer),
  compute stream +0x7d0, primitive +0x590, index size +0x594, index offset +0x5ac, instances
  +0x5b0, vertex attribute object +0x838, vertex buffers +0x640 (16 B each), per-stage program
  state +0x820 (0x478 each). State is committed lazily before each draw (0x1c61460, 0x1c585d0;
  programs 0x1c65d00 -> 0x1c65a70). pglAllocate (0x1c55a10) hands out per-draw data from the top
  of the command chunk.
- Outside PGL the engine calls Gnm directly mostly for cache flushes and waits (ACQUIRE_MEM
  0x1b6980, 137 sites), labels and EOP events, async compute, depth and stencil clear values,
  queries and predication.

## Architecture

```
game thread (render job)                     submit (PGL 0x1c56610 / sceGnmSubmit HLE)
  Gnm/PGL hooks -> JobRecorder ------------>  Executor: walk the chunk chain in GPU order,
     register state, user data                 run each job's command list, insert the
     on draw: pipeline + bindings              game's syncs (labels, EOP, waits)
     -> backend command list                         |
                                                     v
                       shared, thread-safe:   Backend (RHI): Vulkan, later DX12
                       ShaderCache, PipelineCache, ImageRegistry, guest memory import
```

- **Hooks (NAT-001/002, done).** Entry hooks with trampolines; counting stubs that touch no
  registers. RYGT_NATIVE_HOOKS=1.
- **JobRecorder.** One per job/command buffer. Mirrors the registers the Gnm functions set
  (reusing AmdGpu::Liverpool::Regs layouts), the bound shaders and user data. On a draw it builds
  a pipeline key and resolves resources, then records backend commands. No PM4 is decoded.
- **Ordering.** Every recorded draw is tagged with the address its PM4 packet got in the chunk.
  At submit a light walker follows the chunk chain reading packet headers only (~1 ms a frame)
  and runs the recorded command lists in exactly the GPU's order. Sync packets (EOP/EOS labels,
  WAIT_REG_MEM, ACQUIRE_MEM) become fences, label writes and barriers.
- **Barriers.** Images live in GENERAL layout; the game's own cache flushes and waits
  (ACQUIRE_MEM, waitForGraphicsWrites) become global memory barriers, as on the PS4.
- **Buffers.** Read in place from imported guest memory (the direct renderer's page table / buffer
  device address): no copies or tracking per draw.
- **Images.** ImageRegistry maps (address, format, size, tiling) to backend images with
  read-mostly lookups. CPU-written textures are uploaded once and watched for writes; render
  targets are GPU-only.
- **Shaders.** shadGT's GCN recompiler (SPIR-V) is reused; DX12 converts SPIR-V to DXIL.
  Resources are bindless: per draw, the shader's resource slots map to heap indices.
- **Backend interface (RHI).** Device, Queue, CommandList, Buffer, Image, Pipeline, Fence,
  bindless heap. Vulkan: descriptor indexing, dynamic rendering, timeline semaphores. DX12:
  SM 6.6 descriptor heaps, enhanced barriers.

## Milestones

1. **N1 Hook check** (built, not run yet): race run with RYGT_NATIVE_HOOKS=1, game draws =
   renderer draws.
2. **N2 Backend interface + Vulkan** (done, NAT-010..012): rhi.h, rhi_vulkan.cpp. Self test
   outside the game (RytyNativeRhiSelfTest, loaded from Python): pictures, four threads
   recording at once, secondary lists with dynamic state, 40 bindings through a descriptor set,
   compute with specialization constants. Recording costs about 0.24 us per draw.
3. **Image registry** (done, NAT-020/021): image_registry.cpp. Textures and render targets by
   guest address and layout, shared by all job threads; uploads untiled on the GPU with
   shadGT's tiling shaders; downloads re-tiled. Self test (RytyNativeImageSelfTest): exact round
   trips for Thin1DThin, Thin2DThin and Display2DThin with mips.
4. **N3 Shadow mode** (done, NAT-003..009, five check races on 2026-10-19). job_walker.cpp's
   JobCursor walks each job's own commands on its thread, up to each fork (handing the state
   at that point to the child) and to the job's end; the frame's root context is walked the
   same way. shadow.cpp compares, at every draw, the registers the renderer reads with the
   command processor's (enabled stages' programs and the user data they take, written color
   targets and blending, depth/stencil when used, raster state; not stale values in unused
   registers, sample positions, HTILE details). Results: hooks see every draw; decoded draws
   equal the game's draws; walks of all jobs succeed; menus, car list and light scenes 99.8 to
   100% of draws the same where it matters (light scenes 100% on every register). The rest
   (~0.1%) are draws whose pixel shader comes from earlier work outside the job's ancestry
   (the job sees no pixel shader): detectable on the job thread, so such draws can fall back
   to the command processor.
5. **N4 Native mode.** Lessons from N3 and the direct renderer's profile: the serial
   bottleneck is the recorder (resources: BindTextures/FindImage and friends, ~9 ms per
   submit), not the decoder; selecting pipelines elsewhere alone does not raise the frame rate.
   The old texture and buffer caches are single-threaded and keep images in step with guest
   memory as the command processor sees it, so job threads can't use them. Native mode
   therefore uses its own thread-safe caches for all of a frame's GPU work:
   - Job threads record each job's draws into secondary command lists (split at the job's
     flushes, events, waits and writes): pipeline from shadGT's cache (SetThreadDrawState +
     lookup lock, later a lock-free cache), buffers read in place from imported guest memory,
     textures and targets as ImageRegistry handles. Work that must happen at the GPU's point
     in time (uploads from guest memory, invalidation, layout/metadata operations) is queued
     with the segment, not done at record time.
   - The command processor keeps walking packets in GPU order: it runs each segment's list
     after its queued work, and does syncs (labels, EOP, waits), dispatches, clears, copies and
     the root context's draws through the same native rasterizer. The old rasterizer is not
     used in native mode; the presenter shows the registry's display images.
   Steps: **N4.0 (NAT-030)** the job-thread recorder in shadow mode (record, don't run):
   pipelines, bindings and lists per draw, checked against what the current renderer binds for
   the same draw packet, with the cost per draw. **N4.1** native mode for whole frames, boot
   screen and menus first. **N4.2** races, screenshots compared with the current renderer.

   N4.0 so far (RYGT_NATIVE_PIPELINES=1 with shadow mode):
   - NAT-030: job threads look each draw's pipeline up in shadGT's cache
     (PipelineCache::LookupBuiltPipeline: only pipelines already built; nothing translated,
     loaded, built or waited for) and the command processor compares with its own selection.
   - NAT-031: the per-call key state (key, infos, modules, runtime infos, fetch shader, ...)
     moved into KeyScratch; the threads above use the cache's own, a job thread its own. Job
     threads match permutations in their own copies of the cached infos (GetProgramForJob,
     read-only) under a shared lock that yields to the threads above (LookupLock).
   - Eighth race: whenever a job thread found a pipeline it was the renderer's (rare
     differences when two permutations match the same inputs, or programs change); the misses
     were quad lists (now looked up too: their triangle-list pipeline) and new programs or
     pipelines in loading screens. 1.2 us per lookup in light scenes; 9-16 us in races while the
     command processor still takes the lock for its own selection of every draw (native mode
     drops that).
   - NAT-032: the V#/T#/S# each draw reads, hashed on the job thread and again by the recorder
     as it binds the draw (keyed by the draw packet), to see whether the game rewrites resource
     descriptors between a job's end and the GPU's turn.
6. **N5 Speed**: lock-free pipeline lookups for repeated draws, pipelines built ahead from the
   shader pack.
7. **N6 DX12**: the backend interface on D3D12, SPIR-V to DXIL, and pipelines built natively
   (the reused Vulkan pipeline cache does not carry over).
