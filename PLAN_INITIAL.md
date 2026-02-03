# Metal Backend Skeleton — Implementation Plan

## Goal

Create a Metal rendering backend skeleton that mirrors the DX12 architecture: a shared library (`Renderer`) implementing all ~40 C API functions from `Renderer.h`, with stub implementations that compile and run on macOS. The skeleton will initialize a Metal device and clear the screen, proving the pipeline works end-to-end.

## Known Issue: SDL 1.2 + macOS Window Handle

The project uses SDL 1.2 (vendored in `sdl-master/`). On macOS, SDL 1.2's `SDL_SysWMinfo` doesn't expose the `NSWindow` — it falls through to a generic struct with just `version` and `data`. The Quartz video driver (`SDL_QuartzVideo.h:104`) stores `NSWindow *window` internally but doesn't expose it via `SDL_GetWMInfo`.

**Approach:** Patch `sdl-master/sdl/SDL_syswm.h` to add a `SDL_VIDEO_DRIVER_QUARTZ` case that exposes the `NSWindow*`. This is a minimal, targeted change to the vendored SDL.

---

## Step 1: Platform Portability — RT/Core

Make core utilities compile on macOS. These changes are additive (`#ifdef`) and won't break Windows.

### 1a. `RT/Core/VirtualMemory.c`
- Add `#ifdef _WIN32` / `#else` POSIX implementation using `mmap`/`mprotect`/`munmap`
- Note: `RT_ReleaseVirtualMemory` doesn't take a size param but `munmap` requires one. Use `mmap` with `MAP_ANON` and track reserved size via a small static hash table, or pass 0 to munmap (which will fail) and document the limitation. For the skeleton, a simple approach: reserve a fixed max size so we know the extent.

### 1b. `RT/Core/Common.c`
- Add `#ifdef _WIN32` / `#else` macOS implementation:
  - `RT_FATAL_ERROR_`: Use `fprintf(stderr, ...)` + `abort()` instead of `MessageBoxA` + `ExitProcess`
  - `RT_GetHighResTime`: Use `mach_absolute_time()`
  - `RT_SecondsElapsed`: Use `mach_timebase_info` to convert

### 1c. `RT/Renderer/Backend/Common/include/ApiTypes.h`
- Make `RT_API` / `RT_EXPORT` cross-platform:
  - Windows: keep `__declspec(dllexport)`
  - macOS: use `__attribute__((visibility("default")))`
- Make `thread_local` cross-platform:
  - Windows: keep `__declspec(thread)`
  - macOS: use `_Thread_local` (C11)

---

## Step 2: Patch SDL 1.2 for macOS Window Handle

### 2a. `sdl-master/sdl/SDL_syswm.h`
- Add a `#elif defined(SDL_VIDEO_DRIVER_QUARTZ)` case before the `#else` fallback, exposing:
  ```c
  typedef struct SDL_SysWMinfo {
      SDL_version version;
      NSWindow *nswindow;
  } SDL_SysWMinfo;
  ```

### 2b. `sdl-master/sdl/src/video/quartz/SDL_QuartzWM.c` (or wherever `SDL_GetWMInfo` is implemented for Quartz)
- Implement the Quartz case to populate `info->nswindow` from the internal `qz_window` field

### 2c. `sdl-master/sdl/SDL_config_macosx.h`
- Remove `SDL_VIDEO_DRIVER_X11` and related X11 defines (lines 125-135) since Metal builds don't need X11 and it causes `SDL_syswm.h` to pick the X11 struct instead of Quartz

---

## Step 3: CMake Integration

### 3a. Root `CMakeLists.txt`
- Add `GRAPHICS_API=Metal` recognition with `RT_METAL` compile definition
- Guard Windows-specific defines (`_WIN32`, `WINDOWS_IGNORE_PACKING_MISMATCH`, etc.) behind `if(WIN32)`
- Add macOS-specific defines behind `if(APPLE)`
- Remove MSVC-specific compiler ID settings when on macOS

### 3b. `CMakePresets.json`
- Add `metal-mac-debug` and `metal-mac-release` presets with `GRAPHICS_API=Metal`

### 3c. `RT/CMakeLists.txt`
- Make DX12 backend inclusion conditional: only include `Renderer/Backend/DX12` when `GRAPHICS_API=DirectX12`
- Add `Renderer/Backend/Metal` when `GRAPHICS_API=Metal`

### 3d. `d1/CMakeLists.txt`
- Add `Metal` case in the `GRAPHICS_API` conditional, including the same RT source files as DX12 (RTgr.c, Game/Level.c, Game/Lights.c, RTmaterials.c, polymodel_viewer.cpp, material_viewer.cpp) but with `metal_bridge.c` instead of `dx12.c`
- Add platform-specific link libraries for macOS (Cocoa, IOKit, etc.) instead of Windows libs (dxguid, Winmm, Ws2_32, dinput8)

### 3e. Create `RT/Renderer/Backend/Metal/CMakeLists.txt`
- Build `Renderer` as a shared library (.dylib)
- Source files: `Renderer.mm`, `RenderBackend.mm`, `ImageReadWrite.cpp`, `GLTFLoader.cpp`, `mikktspace.c`, `MeshTracker.cpp`, ImGui sources
- Link frameworks: Metal, MetalKit, QuartzCore, AppKit, Foundation
- Link: `RT_CORE`, `Renderer_Common`

---

## Step 4: Create Metal Backend Files

### Directory: `RT/Renderer/Backend/Metal/`

### 4a. `src/MetalIncludes.h`
- Metal/MetalKit/QuartzCore imports (guarded by `__OBJC__`)
- Forward declarations for non-ObjC contexts
- `MTL_STUB(name)` logging macro

### 4b. `src/GlobalMetal.h`
- `MetalState` struct: `id<MTLDevice>`, `id<MTLCommandQueue>`, `CAMetalLayer*`, `NSWindow*`, frame data, resolution, IO, scene camera, default resource handles
- SlotMap for mesh/texture resources
- Global extern `g_mtl`

### 4c. `src/RenderBackend.h`
- Identical interface to DX12's `RenderBackend.h` — same `RenderBackend::` namespace, same function signatures
- Declares `g_rt_material_edges[]` and `g_rt_material_indices[]` externs

### 4d. `src/RenderBackend.mm` — Core implementation
- **`Init`**: Create `MTLDevice`, `MTLCommandQueue`, set up `CAMetalLayer` on the SDL window's `NSView`, init frame semaphore, init IO struct
- **`Exit`**: Release Metal objects
- **`BeginFrame`**: Wait on frame semaphore
- **`BeginScene`**: Store camera/scene settings
- **`EndScene`**: Call `RaytraceRender()` + `RasterRenderDebugLines()`
- **`EndFrame`**: Acquire drawable, clear to solid color, present, signal semaphore
- **`SwapBuffers`**: No-op (present done in EndFrame on Metal)
- **All other functions**: Stub implementations that log `MTL_STUB("FuncName")` and return safe defaults (`RT_RESOURCE_HANDLE_NULL`, zero values, valid pointers for arrays)

### 4e. `src/Renderer.mm` — C-to-C++ wrapper
- Copy from DX12's `Renderer.cpp`, same pass-through pattern
- Each `RT_Xxx` function calls `RenderBackend::Xxx`
- Include mikktspace tangent generation (no platform dependency)

### 4f. Copy platform-independent files from DX12
- `mikktspace.h` / `mikktspace.c` — pure C
- `MeshTracker.hpp` / `MeshTracker.cpp` — only depends on RT/Core
- `ImageReadWrite.cpp` — uses STB; strip DirectXTK12 DDS dependency for skeleton (keep PNG/JPG loading via STB, stub DDS)
- `GLTFLoader.cpp` — uses CGLTF, no D3D12 dependency

### 4g. Third-party libraries (full copy from DX12)
- `cimgui/` — Copy from DX12, then replace `imgui_impl_dx12.*` and `imgui_impl_win32.*` with `imgui_impl_metal.*` and `imgui_impl_osx.*` from upstream ImGui
- `implot/` — Full copy, pure C++, no changes needed
- `STB/` — Full copy, pure C, no changes needed
- `CGLTF/` — Full copy, pure C, no changes needed

---

## Step 5: Game-Side Integration

### 5a. Create `RT/metal_bridge.h`
- Same structure as `dx12.h`: `dx_texture` typedef (it's actually API-agnostic, just holds `RT_ResourceHandle` and dimensions), function declarations for `metal_start_frame`, `metal_end_frame`, `metal_set_render_target`, `metal_init_texture`, etc.

### 5b. Create `RT/metal_bridge.c`
- Same logic as `dx12.c` — these functions only call through the `RT_*` C API (RT_RasterRender, RT_RasterSetViewport, etc.), they don't touch D3D12 directly
- Copy `dx12.c` and rename functions from `dx12_*` to `metal_*`

### 5c. Modify `RT/RTgr.c`
- Add `#ifdef RT_METAL` / `#elif defined(RT_DX12)` guards for:
  - Header include: `dx12.h` vs `metal_bridge.h`
  - Window handle extraction: `info.window` (HWND) vs `info.nswindow` (NSWindow*)
  - ImGui init: `igInitWin32` vs Metal/macOS ImGui init
  - Backend function calls: `dx12_start_frame` vs `metal_start_frame` (etc.)

### 5d. Modify `RT/RTgr.h`
- Make the cimgui include path conditional:
  - `RT_DX12`: `"../../RT/Renderer/Backend/DX12/cimgui/cimgui.h"`
  - `RT_METAL`: `"../../RT/Renderer/Backend/Metal/cimgui/cimgui.h"`

### 5e. Modify `d1/include/gr.h`
- Add `RT_METAL` alongside `RT_DX12` for the `dxtexture` field in `grs_bitmap` (reuse the same `dx_texture` struct since it's API-agnostic)

---

## Step 6: Update All RT_DX12 Guards in Game Code

Update ALL ~44 files that use `#ifdef RT_DX12` to also include `|| defined(RT_METAL)`, done in one pass. This is mechanical: find every `#ifdef RT_DX12` and `#if defined(RT_DX12)` and add the Metal condition. Key files include:
- `d1/include/gr.h` — bitmap struct, BM_OGL define
- `d1/main/render.c` — rendering integration
- `d1/main/gamerend.c` — game rendering
- `d1/main/gamefont.c`, `d1/2d/bitmap.c`, `d1/2d/bitblt.c`, `d1/3d/draw.c`, `d1/3d/rod.c`
- All other files with RT_DX12 guards

---

## Verification

After implementation:
1. `cmake --preset metal-mac-debug` should configure successfully
2. `cmake --build out/build/metal-mac-debug` should compile without errors
3. Running the built executable should:
   - Create an SDL window
   - Initialize the Metal device (logged to stderr)
   - Clear the screen to a solid color each frame
   - Log stub messages for unimplemented functions
   - Not crash

---

## Files Created (new)

| File | Description |
|---|---|
| `RT/Renderer/Backend/Metal/CMakeLists.txt` | Build config for Metal backend |
| `RT/Renderer/Backend/Metal/src/MetalIncludes.h` | Metal framework imports |
| `RT/Renderer/Backend/Metal/src/GlobalMetal.h` | Global Metal state struct |
| `RT/Renderer/Backend/Metal/src/RenderBackend.h` | C++ namespace declarations |
| `RT/Renderer/Backend/Metal/src/RenderBackend.mm` | Core Metal implementation + stubs |
| `RT/Renderer/Backend/Metal/src/Renderer.mm` | C-to-C++ API wrapper |
| `RT/metal_bridge.h` | Game-side bridge header (like dx12.h) |
| `RT/metal_bridge.c` | Game-side bridge impl (like dx12.c) |

## Files Modified (existing)

| File | Change |
|---|---|
| `CMakeLists.txt` | Add Metal API recognition, platform guards |
| `CMakePresets.json` | Add metal-mac-debug/release presets |
| `RT/CMakeLists.txt` | Conditional backend selection |
| `d1/CMakeLists.txt` | Metal source files and macOS link libs |
| `RT/Core/VirtualMemory.c` | Add POSIX mmap implementation |
| `RT/Core/Common.c` | Add macOS timer/error implementation |
| `RT/Renderer/Backend/Common/include/ApiTypes.h` | Cross-platform macros |
| `RT/RTgr.c` | RT_METAL ifdef guards |
| `RT/RTgr.h` | Conditional cimgui include |
| `d1/include/gr.h` | Add RT_METAL to bitmap struct guards |
| `sdl-master/sdl/SDL_syswm.h` | Add Quartz/macOS SysWMinfo case |
| `sdl-master/sdl/SDL_config_macosx.h` | Remove X11 driver for Metal builds |

## Files Copied from DX12 (full copies, not symlinks)

| Source | Destination | Notes |
|---|---|---|
| `DX12/src/mikktspace.h/.c` | `Metal/src/mikktspace.h/.c` | Pure C, no changes |
| `DX12/src/MeshTracker.hpp/.cpp` | `Metal/src/MeshTracker.hpp/.cpp` | Only depends on Core/Arena |
| `DX12/src/ImageReadWrite.cpp` | `Metal/src/ImageReadWrite.cpp` | Modified: stub DDS loading |
| `DX12/src/GLTFLoader.cpp` | `Metal/src/GLTFLoader.cpp` | No D3D12 dependency |
| `DX12/cimgui/` | `Metal/cimgui/` | Full copy, swap DX12/Win32 backends for Metal/OSX |
| `DX12/implot/` | `Metal/implot/` | Full copy, no changes |
| `DX12/STB/` | `Metal/STB/` | Full copy, no changes |
| `DX12/CGLTF/` | `Metal/CGLTF/` | Full copy, no changes |
