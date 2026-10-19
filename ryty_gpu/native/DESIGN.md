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

1. **N1 Hook check** (built): race run with RYGT_NATIVE_HOOKS=1, game draws = renderer draws.
2. **N2 Backend interface + Vulkan device** in the renderer library, off by default.
3. **N3 Shadow mode**: JobRecorders run for every job but render nothing; each draw's state is
   compared with what the PM4 decoder sees at the same packet address. This builds the whole
   front end against a correctness oracle with zero risk to the picture.
4. **N4 Native mode**: the executor replaces the command processor (boot screen, menus, race).
5. **N5 Speed**: parallel recording tuned, pipelines built ahead from the shader pack.
6. **N6 DX12 backend.**
