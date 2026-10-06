# Build tools setup

The scripts in `Build/` expect a few tools under `Build/tools/`. They aren't in git
because they're several GB. Install them yourself using the steps below. Everything
under `Build/` that isn't a script, a doc or `tools/capture-inspect` source is ignored.

## 1. Compiler and build (needed for everything)

- Visual Studio 2022 (or Build Tools) with the **Desktop development with C++** workload.
  Run all commands below from the **x64 Native Tools Command Prompt / Developer PowerShell**.
- LLVM 22 (`clang-cl`), installed to `C:\Program Files\LLVM`: https://github.com/llvm/llvm-project/releases
- CMake and Ninja (both ship with Visual Studio, or https://cmake.org and https://ninja-build.org).
- Submodules: `git submodule update --init --recursive`

Configure and build with the presets. The scripts look for `Build/x64-Clang-<Type>/shadps4.exe`:

```powershell
cmake --preset x64-Clang-Release
cmake --build Build/x64-Clang-Release
```

## 2. Vulkan SDK 1.4.363.0 → `Build/tools/VulkanSDK`

`Run-GT-Sport.ps1 -Diagnostic` / `-QuickDiagnostic` loads the validation layers from
`Build/tools/VulkanSDK/Bin`.

1. Download the 1.4.363.0 Windows installer from https://vulkan.lunarg.com/sdk/home#windows
   (older versions are under "SDK archive"). Save it as `Build/tools/vulkan-sdk.exe`.
2. Install it into the tools folder without touching your system install:

```powershell
Build/tools/vulkan-sdk.exe --root D:/path/to/shadPS4/Build/tools/VulkanSDK `
    --accept-licenses --default-answer --confirm-command install copy_only=1
```

## 3. RenderDoc 1.46 → `Build/tools/renderdoc-1.46/RenderDoc_1.46_64`

`Run-GT-Sport-Capture.ps1` launches the game through `renderdoccmd.exe` from that folder.

1. Download the **portable zip** of RenderDoc 1.46 (64-bit) from https://renderdoc.org/builds
   (older versions are listed further down that page).
2. Extract it so `Build/tools/renderdoc-1.46/RenderDoc_1.46_64/renderdoccmd.exe` exists.

## 4. capture-inspect (optional, for analysing `.rdc` captures)

The source is tracked in `Build/tools/capture-inspect/` (`main.cpp`, `renderdoc.def`).
It needs the RenderDoc replay API headers and the RenderDoc install from step 3.

1. Copy the replay headers from the RenderDoc v1.46 source tag
   (https://github.com/baldurk/renderdoc/tree/v1.46/renderdoc/api/replay) into
   `Build/tools/renderdoc-replay-api-1.46/`.
2. Build it from the repo root:

```powershell
cd Build/tools/capture-inspect
lib /nologo /def:renderdoc.def /machine:x64 /out:renderdoc.lib
& 'C:/Program Files/LLVM/bin/clang-cl.exe' /nologo /std:c++17 /MD /EHsc /O2 `
    /I ../renderdoc-replay-api-1.46 `
    /I ../../../externals/json/single_include `
    /I ../../../externals/sirit/externals/SPIRV-Headers/include `
    main.cpp renderdoc.lib `
    /Fe:../renderdoc-1.46/RenderDoc_1.46_64/capture-inspect.exe
```

The exe goes next to `renderdoc.dll` so it can load it.

## 5. Game path

The scripts default to `E:/Console Games/PS4 Games/CUSA03220/eboot.bin`. Pass
`-GamePath` to `Run-GT-Sport.ps1`, or edit `$game` in `Run-GT-Sport-Capture.ps1`, to point at your own dump.
