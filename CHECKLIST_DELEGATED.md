# Metal Backend Skeleton — Delegated Implementation Checklist

---

## Model Assignment Legend

| Agent | When to use | Strengths |
|-------|-------------|-----------|
| **Gemini** | Mechanical / boilerplate / copy-rename / simple guards | Cheapest. Perfect for repetitive edits, find-and-replace, JSON, trivial preprocessor changes. |
| **Codex** | Moderate complexity / well-specified coding tasks | Good C/C++/CMake generation. Follows detailed specs reliably. Handles POSIX APIs, SDL internals, header creation. |
| **Claude** | Architectural / Metal+ObjC expertise / debugging | Reserve for tasks requiring Metal API knowledge, Objective-C++, creative problem-solving, or build troubleshooting. |

---

## Execution Order & Parallelism

Steps 1, 2, and 6 are independent of each other and can run in parallel.
Step 3 depends on Step 1 (portability macros) being done.
Step 4 depends on Step 3 (CMake must know about Metal dir).
Step 5 depends on Step 4 (backend files must exist).
Step 7 depends on everything.

**Suggested parallel batches:**
1. **Batch A** (parallel): Step 1 (Codex), Step 2 (Codex), Step 6 (Gemini)
2. **Batch B** (sequential): Step 3 (Codex/Gemini)
3. **Batch C** (parallel where noted): Step 4 (Claude + Codex + Gemini)
4. **Batch D**: Step 5 (Gemini + Codex)
5. **Batch E**: Step 7 (Claude)

---

## Step 1: Platform Portability — RT/Core

### 1.1 `RT/Core/VirtualMemory.c` — Add POSIX implementation — `Codex`
> **Why Codex:** POSIX memory APIs (mmap/mprotect/munmap) are well-documented and the spec is detailed. Requires understanding of memory management patterns and implementing a small hash table for size tracking. Too much logic for Gemini, not complex enough for Claude.

- [ ] Wrap existing Windows code in `#ifdef _WIN32`
- [ ] Add `#else` block with POSIX implementation:
  - [ ] `RT_ReserveVirtualMemory`: Use `mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0)`
  - [ ] `RT_CommitVirtualMemory`: Use `mprotect(address, size, PROT_READ | PROT_WRITE)`
  - [ ] `RT_DecommitVirtualMemory`: Use `madvise(address, size, MADV_DONTNEED)` + `mprotect(address, size, PROT_NONE)`
  - [ ] `RT_ReleaseVirtualMemory`: Needs size tracking since `munmap` requires size. Add a small static hash table mapping address -> size, populated in `RT_ReserveVirtualMemory`
- [ ] Include `<sys/mman.h>` and `<unistd.h>` in the POSIX path
- [ ] Verify Windows build still works (no changes to Windows code path)

### 1.2 `RT/Core/Common.c` — Add macOS implementation — `Codex`
> **Why Codex:** macOS timer APIs (mach_absolute_time) and error handling are standard patterns. Spec provides exact function names and headers. Moderate complexity — right in Codex's sweet spot.

- [ ] Wrap existing Windows code in `#ifdef _WIN32`
- [ ] Add `#elif defined(__APPLE__)` block:
  - [ ] `RT_FATAL_ERROR_`: Use `fprintf(stderr, ...)` for the error message, `__builtin_trap()` for debugger break, `exit(1)` to terminate. Use `basename()` from `<libgen.h>` instead of `PathStripPathA`
  - [ ] `RT_GetHighResTime`: Use `mach_absolute_time()` from `<mach/mach_time.h>`
  - [ ] `RT_SecondsElapsed`: Use `mach_timebase_info` to convert ticks to nanoseconds, then divide by 1e9
- [ ] Remove Windows-specific includes (`<Windows.h>`, `<Shlwapi.h>`, `#pragma comment`) from the macOS path

### 1.3 `RT/Renderer/Backend/Common/include/ApiTypes.h` — Cross-platform macros — `Gemini`
> **Why Gemini:** Pure preprocessor #ifdef/#else blocks with exact values specified. No logic, just conditional defines. Mechanical.

- [ ] Make `RT_API` cross-platform:
  - C mode: `#ifdef _WIN32` -> `extern` / else -> `extern`
  - C++ mode: `#ifdef _WIN32` -> `extern "C"` / else -> `extern "C"`
- [ ] Make `RT_EXPORT` cross-platform:
  - Windows: `extern __declspec(dllexport)` (existing)
  - macOS: `extern __attribute__((visibility("default")))`
- [ ] Make `thread_local` cross-platform:
  - Windows: `__declspec(thread)` (existing)
  - macOS/Clang: `_Thread_local` (C11 keyword)
- [ ] Verify the `#pragma pack` directives work with Clang (they do)

---

## Step 2: Patch SDL 1.2 for macOS Window Handle

### 2.1 `sdl-master/sdl/SDL_syswm.h` — Add Quartz SysWMinfo — `Codex`
> **Why Codex:** Requires reading existing SDL struct patterns and adding a new platform case. Moderate — needs to understand SDL conventions and place the new code correctly relative to existing #elif chains.

- [ ] Add `#elif defined(SDL_VIDEO_DRIVER_QUARTZ)` case before the `#else` generic fallback (before line 184)
- [ ] Define macOS-specific `SDL_SysWMinfo` struct with `SDL_version version` and `void *nswindow` (use `void*` to avoid requiring ObjC in the header)
- [ ] Define macOS-specific `SDL_SysWMmsg` struct (can be minimal/generic)

### 2.2 SDL Quartz WM implementation — Populate nswindow — `Codex`
> **Why Codex:** Requires exploring SDL source to find the Quartz WM implementation, understanding the driver internals, and adding a line to populate the new field. Needs codebase navigation + moderate C knowledge.

- [ ] Find where `SDL_GetWMInfo` is implemented for the Quartz driver (likely `sdl-master/sdl/src/video/quartz/SDL_QuartzWM.c` or similar)
- [ ] Add code to populate `info->nswindow` from the Quartz driver's internal `qz_window` field (declared in `SDL_QuartzVideo.h:104` as `NSWindow *window`)

### 2.3 `sdl-master/sdl/SDL_config_macosx.h` — Fix driver priority — `Gemini`
> **Why Gemini:** Removing/undefining a few #define lines. Completely mechanical — just comment out or #undef specific lines.

- [ ] Remove or `#undef` the `SDL_VIDEO_DRIVER_X11` define (line 125) and related X11 defines (lines 125-135)
  - Reason: `SDL_syswm.h` checks `#if defined(SDL_VIDEO_DRIVER_X11)` first, so having both X11 and Quartz defined causes the X11 struct to be used instead of our new Quartz struct
  - Only needed for Metal builds; X11 is not useful on modern macOS

---

## Step 3: CMake Integration

### 3.1 Root `CMakeLists.txt` — Add Metal recognition — `Codex`
> **Why Codex:** Requires understanding the existing CMake structure and adding conditional blocks in the right places. The `if(WIN32)` guard wrapping needs judgment about which definitions are Windows-only.

- [ ] Add `elseif(${GRAPHICS_API} STREQUAL "Metal")` block (after DX12 check, before OpenGL check) that sets `add_compile_definitions(PUBLIC RT_METAL)`
- [ ] Wrap Windows-specific compile definitions (lines 38-44: `_WIN32`, `WINDOWS_IGNORE_PACKING_MISMATCH`, `WIN32_LEAN_AND_MEAN`, `NOMINMAX`) in `if(WIN32)` guard
- [ ] Wrap or conditionally skip MSVC compiler ID settings (lines 8-9: `CMAKE_C_COMPILER_ID`, `CMAKE_CXX_COMPILER_ID`) for non-Windows

### 3.2 `CMakePresets.json` — Add Metal presets — `Gemini`
> **Why Gemini:** Adding JSON objects to an existing array. Exact field values are specified. Pure data entry.

- [ ] Add `metal-mac-debug` preset:
  - Inherits: `default`
  - `CMAKE_BUILD_TYPE`: `Debug`
  - `GRAPHICS_API`: `Metal`
  - Optional condition: `hostSystemName == Darwin`
- [ ] Add `metal-mac-release` preset:
  - Inherits: `default`
  - `CMAKE_BUILD_TYPE`: `RelWithDebInfo`
  - `GRAPHICS_API`: `Metal`
  - Optional condition: `hostSystemName == Darwin`

### 3.3 `RT/CMakeLists.txt` — Conditional backend selection — `Gemini`
> **Why Gemini:** Replacing one line with a 5-line if/elseif/endif block. Exact code is given in the spec.

- [ ] Change unconditional `add_subdirectory("Renderer/Backend/DX12")` (line 13) to:
  ```cmake
  if(${GRAPHICS_API} STREQUAL "DirectX12")
      add_subdirectory("Renderer/Backend/DX12")
  elseif(${GRAPHICS_API} STREQUAL "Metal")
      add_subdirectory("Renderer/Backend/Metal")
  endif()
  ```

### 3.4 `d1/CMakeLists.txt` — Add Metal source files and link libs — `Codex`
> **Why Codex:** Moderate CMake complexity — needs to add a new elseif block with correct source file lists, platform-aware linking, and executable flag handling. Requires understanding CMake patterns in the existing file.

- [ ] Add `elseif(${GRAPHICS_API} STREQUAL "Metal")` in the `GRAPHICS_API` conditional (after DX12, before OpenGL):
  - Set `Graphic_Includes` to:
    - `$ENV{RT_EXT_CODE_DIR}/RTgr.c`, `RTgr.h`
    - `$ENV{RT_EXT_CODE_DIR}/Game/Level.h`, `Level.c`
    - `$ENV{RT_EXT_CODE_DIR}/Game/Lights.h`, `Lights.c`
    - `$ENV{RT_EXT_CODE_DIR}/metal_bridge.c`, `metal_bridge.h` (NOT dx12.c)
    - `$ENV{RT_EXT_CODE_DIR}/polymodel_viewer.cpp`, `polymodel_viewer.h`
    - `$ENV{RT_EXT_CODE_DIR}/material_viewer.cpp`, `material_viewer.h`
    - `$ENV{RT_EXT_CODE_DIR}/RTmaterials.c`, `RTmaterials.h`
  - Set `Graphic_Libs` to `Renderer`
- [ ] Make `target_link_libraries` for descent1 platform-aware:
  - Windows: `dxguid`, `Winmm`, `Ws2_32`, `dinput8` (existing)
  - macOS: `"-framework Cocoa"`, `"-framework IOKit"` (replace Windows libs)
- [ ] Handle the `WIN32` flag on `add_executable` — on macOS, don't use `WIN32` subsystem flag

### 3.5 Create `RT/Renderer/Backend/Metal/CMakeLists.txt` — `Codex`
> **Why Codex:** New CMake file with ObjC compilation flags, framework linking, and shared library setup. Well-specified but needs CMake + Apple toolchain knowledge that Gemini may get wrong.

- [ ] `cmake_minimum_required(VERSION 3.13)`
- [ ] Guard with `if(NOT APPLE)` fatal error
- [ ] Set C++17 standard
- [ ] Define `IMGUI_SOURCE_FILES` list with ImGui core + Metal/OSX backends
- [ ] Define `IMPLOT_SOURCE_FILES` list
- [ ] Define `Renderer_SOURCE` list: `Renderer.mm`, `RenderBackend.mm`, `ImageReadWrite.cpp`, `GLTFLoader.cpp`, `mikktspace.c`, `MeshTracker.cpp`, ImGui sources, ImPlot sources
- [ ] `add_library(Renderer SHARED ${Renderer_SOURCE})`
- [ ] Set target properties: output directory, C++17, ObjC ARC
- [ ] `target_include_directories` for cimgui, implot, STB, CGLTF
- [ ] `target_link_libraries`: `Renderer_Common`, `RT_CORE`, `-framework Metal`, `-framework MetalKit`, `-framework QuartzCore`, `-framework AppKit`, `-framework Foundation`

---

## Step 4: Create Metal Backend Files

### 4.1 Create directory structure — `Gemini`
> **Why Gemini:** Two mkdir commands. Trivial.

- [ ] Create `RT/Renderer/Backend/Metal/`
- [ ] Create `RT/Renderer/Backend/Metal/src/`

### 4.2 Copy third-party libraries from DX12 (full copies) — `Gemini`
> **Why Gemini:** File copy operations (cp -r) plus deleting a few files. The ImGui Metal/OSX backend files need to be downloaded from upstream, but the URLs and filenames are well-known. Mechanical.

- [ ] Copy `RT/Renderer/Backend/DX12/STB/` -> `RT/Renderer/Backend/Metal/STB/`
- [ ] Copy `RT/Renderer/Backend/DX12/CGLTF/` -> `RT/Renderer/Backend/Metal/CGLTF/`
- [ ] Copy `RT/Renderer/Backend/DX12/implot/` -> `RT/Renderer/Backend/Metal/implot/`
- [ ] Copy `RT/Renderer/Backend/DX12/cimgui/` -> `RT/Renderer/Backend/Metal/cimgui/`
  - [ ] Remove `cimgui/imgui/backends/imgui_impl_dx12.cpp` and `imgui_impl_dx12.h`
  - [ ] Remove `cimgui/imgui/backends/imgui_impl_win32.cpp` and `imgui_impl_win32.h`
  - [ ] Add `imgui_impl_metal.h`, `imgui_impl_metal.mm` from upstream ImGui
  - [ ] Add `imgui_impl_osx.h`, `imgui_impl_osx.mm` from upstream ImGui

### 4.3 Copy platform-independent source files from DX12 — `Codex`
> **Why Codex:** Mostly file copies, but ImageReadWrite.cpp needs DDS/DirectXTK12 dependencies surgically removed while keeping STB-based loading intact. Requires reading the file and understanding what to cut. Too much judgment for Gemini.

- [ ] Copy `RT/Renderer/Backend/DX12/src/mikktspace.h` -> `RT/Renderer/Backend/Metal/src/mikktspace.h`
- [ ] Copy `RT/Renderer/Backend/DX12/src/mikktspace.c` -> `RT/Renderer/Backend/Metal/src/mikktspace.c`
- [ ] Copy `RT/Renderer/Backend/DX12/src/MeshTracker.hpp` -> `RT/Renderer/Backend/Metal/src/MeshTracker.hpp`
- [ ] Copy `RT/Renderer/Backend/DX12/src/MeshTracker.cpp` -> `RT/Renderer/Backend/Metal/src/MeshTracker.cpp`
- [ ] Copy `RT/Renderer/Backend/DX12/src/GLTFLoader.cpp` -> `RT/Renderer/Backend/Metal/src/GLTFLoader.cpp`
- [ ] Copy `RT/Renderer/Backend/DX12/src/ImageReadWrite.cpp` -> `RT/Renderer/Backend/Metal/src/ImageReadWrite.cpp`
  - [ ] Remove/stub the DirectXTK12 DDSc.h dependency (DDS loading) — keep STB-based PNG/JPG/BMP loading
  - [ ] Remove any `#include` of D3D12-specific headers

### 4.4 Create `RT/Renderer/Backend/Metal/src/MetalIncludes.h` — `Codex`
> **Why Codex:** Small utility header, but needs correct `#ifdef __OBJC__` guarding pattern for Metal imports and proper forward declarations for mixed C/ObjC compilation. Spec is detailed enough for Codex.

- [ ] Add `#pragma once`
- [ ] Add `#ifdef __OBJC__` guard with Metal/MetalKit/QuartzCore imports
- [ ] Add forward declarations for non-ObjC contexts (`typedef void* id;`)
- [ ] Include common headers: `<assert.h>`, `<stdint.h>`, `"Core/Common.h"`, `"Core/Arena.h"`
- [ ] Define `MTL_LOG(fmt, ...)` macro: `fprintf(stderr, "[Metal] " fmt "\n", ##__VA_ARGS__)`
- [ ] Define `MTL_STUB(name)` macro: `MTL_LOG("STUB: %s not yet implemented", name)`

### 4.5 Create `RT/Renderer/Backend/Metal/src/GlobalMetal.h` — `Claude`
> **Why Claude:** This is the architectural backbone of the Metal backend. Requires understanding Metal API types (`id<MTLDevice>`, `id<MTLCommandQueue>`, `CAMetalLayer`, `dispatch_semaphore_t`), how they interact, correct ObjC guarding for a C++ header, and how `SlotMap` templates work with Metal resource types. Getting the struct layout wrong here cascades to every other file.

- [ ] Add `#pragma once`
- [ ] Include `MetalIncludes.h`, Core headers, `Renderer.h`
- [ ] Guard Objective-C types behind `#ifdef __OBJC__`
- [ ] Define constants: `BACK_BUFFER_COUNT = 3`, `MAX_INSTANCES = 1000`, `MAX_RASTER_TRIANGLES = 10000`, `MAX_RASTER_LINES = 5000`, `MAX_DEBUG_LINES_WORLD = 5000`
- [ ] Define `MeshResource` struct: `id<MTLBuffer> vertex_buffer`, `uint32_t triangle_count`
- [ ] Define `TextureResource` struct: `RT_ResourceHandle handle`, `id<MTLTexture> texture`
- [ ] Define `FrameData` struct: `dispatch_semaphore_t semaphore`, `id<MTLCommandBuffer> command_buffer`
- [ ] Define `MetalState` struct with fields:
  - [ ] Window/Layer: `NSWindow *window`, `CAMetalLayer *metal_layer`
  - [ ] Device: `id<MTLDevice> device`, `id<MTLCommandQueue> command_queue`
  - [ ] IO: `RT_RendererIO io`, `RT_Arena *arena`
  - [ ] Resolution: `output_width`, `output_height`, `render_width`, `render_height`
  - [ ] Frame: `frame_index`, `current_back_buffer_index`, `FrameData frame_data[BACK_BUFFER_COUNT]`
  - [ ] Default resources: `RT_ResourceHandle white_texture_handle`, `black_texture_handle`, `billboard_quad`, `cube`
  - [ ] Scene: camera, prev_camera, render_blit
  - [ ] Sync: `dispatch_semaphore_t frame_semaphore`
  - [ ] Misc: `queued_screenshot`, `viewport_offset_y`
- [ ] Declare extern globals: `MetalState g_mtl`, `SlotMap<MeshResource>`, `SlotMap<TextureResource>`

### 4.6 Create `RT/Renderer/Backend/Metal/src/RenderBackend.h` — `Codex`
> **Why Codex:** Header with function declarations — can be closely modeled on the DX12 version. Requires reading the DX12 RenderBackend.h and adapting includes. No Metal-specific logic, just signatures.

- [ ] Add `#pragma once`
- [ ] Include `"ApiTypes.h"`, `"Renderer.h"`
- [ ] Declare extern arrays: `g_rt_material_edges[RT_MAX_MATERIAL_EDGES]`, `g_rt_material_indices[RT_MAX_MATERIALS]`
- [ ] Declare `namespace RenderBackend` with all functions (identical signatures to DX12 version):
  - [ ] Init/Exit: `Init`, `Exit`, `Flush`
  - [ ] Frame: `BeginFrame`, `BeginScene`, `EndScene`, `EndFrame`, `SwapBuffers`
  - [ ] IO/Debug: `GetIO`, `CheckWindowMinimized`, `DoDebugMenus`
  - [ ] Resources: `UploadTexture`, `UploadMesh`, `ReleaseTexture`, `ReleaseMesh`, `UpdateMaterial`
  - [ ] Defaults: `GetDefaultWhiteTexture`, `GetDefaultBlackTexture`, `GetBillboardMesh`, `GetCubeMesh`
  - [ ] Raytracing: `RaytraceSubmitLights`, `RaytraceSetVerticalOffset`, `RaytraceGetVerticalOffset`, `RaytraceGetCurrentLightCount`, `RaytraceMesh`, `RaytraceBillboardColored`, `RaytraceRod`, `RaytraceRender`, `RaytraceSetSkyColors`
  - [ ] Rasterization: `RasterSetViewport`, `RasterSetRenderTarget`, `RasterTriangles`, `RasterLines`, `RasterLinesWorld`, `RasterRender`, `RasterRenderDebugLines`, `RasterBlitScene`, `RasterBlit`
  - [ ] ImGui: `RenderImGuiTexture`, `RenderImGui`
  - [ ] Utility: `QueueScreenshot`

### 4.7 Create `RT/Renderer/Backend/Metal/src/RenderBackend.mm` — `Claude`
> **Why Claude:** **This is the hardest task in the entire checklist.** Objective-C++ file implementing Metal initialization (MTLCreateSystemDefaultDevice, CAMetalLayer setup, command queue), triple-buffered frame management with dispatch semaphores, render pass descriptors, command buffer lifecycle, and drawable presentation. Even though most functions are stubs, the Init and EndFrame implementations require real Metal API expertise, correct @autoreleasepool usage, and understanding of the Metal render loop. Errors here mean a black screen or crashes with no useful diagnostics.

- [ ] Include headers: `RenderBackend.h`, `GlobalMetal.h`, Metal/QuartzCore/AppKit imports
- [ ] Define global arrays: `g_rt_material_edges`, `g_rt_material_indices`
- [ ] Define global state: `MetalState g_mtl`, slot maps
- [ ] Implement `RenderBackend::Init`:
  - [ ] Store arena from params
  - [ ] Extract NSWindow from `params->window_handle` (cast from `void*`)
  - [ ] Create Metal device via `MTLCreateSystemDefaultDevice()`
  - [ ] Log device name
  - [ ] Create command queue
  - [ ] Set up `CAMetalLayer` on window's content view (`setWantsLayer:YES`, assign layer)
  - [ ] Configure layer: pixel format `BGRAUnorm`, framebuffer only, drawable size
  - [ ] Store output dimensions
  - [ ] Create frame semaphore: `dispatch_semaphore_create(BACK_BUFFER_COUNT)`
  - [ ] Initialize IO struct
- [ ] Implement `RenderBackend::Exit`:
  - [ ] Release Metal objects (nil assignments, ARC handles the rest)
  - [ ] Log shutdown
- [ ] Implement `RenderBackend::Flush`: stub
- [ ] Implement `RenderBackend::BeginFrame`:
  - [ ] `dispatch_semaphore_wait(g_mtl.frame_semaphore, DISPATCH_TIME_FOREVER)`
- [ ] Implement `RenderBackend::BeginScene`:
  - [ ] Store prev_camera = camera
  - [ ] Store new camera from scene_settings
  - [ ] Store render_blit flag
- [ ] Implement `RenderBackend::EndScene`:
  - [ ] Call `RaytraceRender()` and `RasterRenderDebugLines()`
- [ ] Implement `RenderBackend::EndFrame`:
  - [ ] `@autoreleasepool` block
  - [ ] Acquire drawable from `metal_layer`
  - [ ] Create render pass descriptor, clear to solid color (cornflower blue)
  - [ ] Create command buffer, render encoder, end encoding
  - [ ] Present drawable
  - [ ] Add completion handler to signal frame semaphore
  - [ ] Commit command buffer
  - [ ] Increment frame_index
- [ ] Implement `RenderBackend::SwapBuffers`: no-op (present done in EndFrame)
- [ ] Implement `RenderBackend::GetIO`: return `&g_mtl.io`
- [ ] Implement `RenderBackend::CheckWindowMinimized`: return `[g_mtl.window isMiniaturized]`
- [ ] Implement all remaining functions as stubs:
  - [ ] `DoDebugMenus` — stub
  - [ ] `UploadTexture` — stub, return `RT_RESOURCE_HANDLE_NULL`
  - [ ] `UploadMesh` — stub, return `RT_RESOURCE_HANDLE_NULL`
  - [ ] `ReleaseTexture` — stub
  - [ ] `ReleaseMesh` — stub
  - [ ] `UpdateMaterial` — stub, return `material_index`
  - [ ] `GetDefaultWhiteTexture` — return `RT_RESOURCE_HANDLE_NULL`
  - [ ] `GetDefaultBlackTexture` — return `RT_RESOURCE_HANDLE_NULL`
  - [ ] `GetBillboardMesh` — return `RT_RESOURCE_HANDLE_NULL`
  - [ ] `GetCubeMesh` — return `RT_RESOURCE_HANDLE_NULL`
  - [ ] `RaytraceSubmitLights` — stub
  - [ ] `RaytraceSetVerticalOffset` — store in `g_mtl.viewport_offset_y`
  - [ ] `RaytraceGetVerticalOffset` — return `g_mtl.viewport_offset_y`
  - [ ] `RaytraceGetCurrentLightCount` — return 0
  - [ ] `RaytraceMesh` — stub
  - [ ] `RaytraceBillboardColored` — stub
  - [ ] `RaytraceRod` — stub
  - [ ] `RaytraceRender` — stub
  - [ ] `RaytraceSetSkyColors` — stub
  - [ ] `RasterSetViewport` — stub
  - [ ] `RasterSetRenderTarget` — stub
  - [ ] `RasterTriangles` — stub
  - [ ] `RasterLines` — stub
  - [ ] `RasterLinesWorld` — stub
  - [ ] `RasterRender` — stub
  - [ ] `RasterRenderDebugLines` — stub
  - [ ] `RasterBlitScene` — stub
  - [ ] `RasterBlit` — stub
  - [ ] `RenderImGuiTexture` — stub
  - [ ] `RenderImGui` — stub
  - [ ] `QueueScreenshot` — stub

### 4.8 Create `RT/Renderer/Backend/Metal/src/Renderer.mm` — `Codex`
> **Why Codex:** Mostly a copy of the DX12 Renderer.cpp with include path changes. The C API wrappers and mikktspace code are platform-independent. Codex can read the DX12 version and adapt it. No Metal-specific logic — just plumbing.

- [ ] Copy from `RT/Renderer/Backend/DX12/src/Renderer.cpp`
- [ ] Keep all the C API wrapper functions (they just call `RenderBackend::Xxx`)
- [ ] Keep the `RT_GenerateTangents` implementation with mikktspace (no platform dependency)
- [ ] Keep the `RT_RaytraceSetRenderFlagsOverride` static state
- [ ] Keep the helper functions that build `RT_RenderMeshParams` from simpler arguments
- [ ] Update includes to reference Metal backend headers instead of DX12

---

## Step 5: Game-Side Integration

### 5.1 Create `RT/metal_bridge.h` — `Gemini`
> **Why Gemini:** Copy dx12.h, rename `dx12_` to `metal_` in declarations, change include guard. Mechanical find-and-replace on a header file.

- [ ] Define `dx_texture` struct (same as in `dx12.h` — it's API-agnostic: `RT_ResourceHandle handle`, `int w, h, tw, th, lw`, `float u, v`)
- [ ] Declare functions (same signatures as `dx12.h` but `metal_` prefix):
  - [ ] `metal_start_frame()`
  - [ ] `metal_end_frame()`
  - [ ] `metal_set_render_target(RT_ResourceHandle)`
  - [ ] `metal_urect(int left, int top, int right, int bot)`
  - [ ] `metal_init_texture(grs_bitmap*)`
  - [ ] `metal_internal_string(int x, int y, const char* s)`
  - [ ] `metal_init_font(grs_font*)`
  - [ ] `metal_load_bitmap_pixel_data(RT_Arena*, grs_bitmap*)`
  - [ ] `metal_ubitmapm_cs(int x, int y, int dw, int dh, grs_bitmap*, int c, int scale)`
  - [ ] `metal_ubitblt(int dw, int dh, int dx, int dy, int sw, int sh, int sx, int sy, grs_bitmap*, grs_bitmap*, int texfilt)`
- [ ] Include guards, includes for `ApiTypes.h`, `Renderer.h`, `gr.h`

### 5.2 Create `RT/metal_bridge.c` — `Gemini`
> **Why Gemini:** Copy dx12.c, sed `dx12_` to `metal_` globally, change one #include. Pure mechanical rename. This is the poster child for a cheap model task.

- [ ] Copy from `RT/dx12.c`
- [ ] Rename all `dx12_` functions to `metal_`
- [ ] Replace `#include "dx12.h"` with `#include "metal_bridge.h"`
- [ ] All logic stays the same — these functions only call `RT_*` API functions, not D3D12 directly
- [ ] Key functions to rename:
  - [ ] `dx12_start_frame` -> `metal_start_frame`
  - [ ] `dx12_end_frame` -> `metal_end_frame`
  - [ ] `dx12_set_render_target` -> `metal_set_render_target`
  - [ ] `dx12_init_texture` -> `metal_init_texture`
  - [ ] `dx12_loadbmtexture_f` -> `metal_loadbmtexture_f`
  - [ ] `dx12_init_font` -> `metal_init_font`
  - [ ] `dx12_urect` -> `metal_urect`
  - [ ] `dx12_ulinec` -> `metal_ulinec`
  - [ ] `dx12_upixelc` -> `metal_upixelc`
  - [ ] `dx12_drawcircle` -> `metal_drawcircle`
  - [ ] `dx12_internal_string` -> `metal_internal_string`
  - [ ] `dx12_load_bitmap_pixel_data` -> `metal_load_bitmap_pixel_data`
  - [ ] `dx12_ubitmapm_cs` -> `metal_ubitmapm_cs`
  - [ ] `dx12_ubitblt` -> `metal_ubitblt`
  - [ ] `dx12_font_choose_size` -> `metal_font_choose_size`

### 5.3 Modify `RT/RTgr.c` — Add RT_METAL guards — `Codex`
> **Why Codex:** Multiple #ifdef blocks need to be added throughout a large file, each with the correct Metal alternative. Requires reading existing code to understand where dx12_ calls are and what context they're in. More judgment than Gemini can reliably handle — needs to understand the #elif pattern and not break the existing DX12 path.

- [ ] Add `#include "metal_bridge.h"` under `#elif defined(RT_METAL)` (alongside existing `#include "dx12.h"` under `#ifdef RT_DX12`)
- [ ] Window handle extraction (~line 485): Add `#elif defined(RT_METAL)` that sets `initParams.window_handle = info.nswindow` (from patched SDL)
- [ ] ImGui init (~line 487): Add `#elif defined(RT_METAL)` for Metal/macOS ImGui init (or defer to renderer backend)
- [ ] Replace all `dx12_` function calls with conditional macros or `#ifdef` blocks:
  - [ ] `dx12_start_frame()` -> `metal_start_frame()` under `RT_METAL`
  - [ ] `dx12_end_frame()` -> `metal_end_frame()` under `RT_METAL`
  - [ ] `dx12_init_texture()` -> `metal_init_texture()` under `RT_METAL`
  - [ ] `dx12_loadbmtexture_f()` -> `metal_loadbmtexture_f()` under `RT_METAL`
  - [ ] `dx12_set_render_target()` -> `metal_set_render_target()` under `RT_METAL`
  - [ ] `dx12_ubitmapm_cs()` -> `metal_ubitmapm_cs()` under `RT_METAL`
  - [ ] `dx12_ubitblt()` -> `metal_ubitblt()` under `RT_METAL`
  - [ ] `dx12_urect()` -> `metal_urect()` under `RT_METAL`
  - [ ] `dx12_internal_string()` -> `metal_internal_string()` under `RT_METAL`
  - [ ] `dx12_init_font()` -> `metal_init_font()` under `RT_METAL`
  - [ ] Any other `dx12_*` calls

### 5.4 Modify `RT/RTgr.h` — Conditional cimgui include — `Gemini`
> **Why Gemini:** Change one #include to a 5-line #ifdef block. Exact code is in the spec. Trivial.

- [ ] Change hardcoded `#include "../../RT/Renderer/Backend/DX12/cimgui/cimgui.h"` (line 88) to:
  ```c
  #ifdef RT_DX12
  #include "../../RT/Renderer/Backend/DX12/cimgui/cimgui.h"
  #elif defined(RT_METAL)
  #include "../../RT/Renderer/Backend/Metal/cimgui/cimgui.h"
  #endif
  ```

### 5.5 Modify `d1/include/gr.h` — Add RT_METAL to bitmap struct — `Gemini`
> **Why Gemini:** Changing `#ifdef RT_DX12` to `#if defined(RT_DX12) || defined(RT_METAL)` on 1-2 lines. Identical to Step 6 pattern.

- [ ] Change `#ifdef RT_DX12` (line 101) to `#if defined(RT_DX12) || defined(RT_METAL)` for the `dxtexture` field
- [ ] Also update the `BM_OGL` / `BM_RGBA8` defines if they're guarded by `RT_DX12`

---

## Step 6: Update All RT_DX12 Guards in Game Code — `Gemini`

> **Why Gemini for ALL of Step 6:** This is the single most mechanical task in the checklist. Every change is identical: `#ifdef RT_DX12` -> `#if defined(RT_DX12) || defined(RT_METAL)` and `#if defined(RT_DX12)` -> `#if defined(RT_DX12) || defined(RT_METAL)`. No judgment required. A regex could do it. Gemini can batch all 35+ files in one session. **This is the highest-value Gemini task — many files, zero ambiguity.**

Mechanical find-and-replace: every `#ifdef RT_DX12` becomes `#if defined(RT_DX12) || defined(RT_METAL)`, and every `#if defined(RT_DX12)` gets `|| defined(RT_METAL)` appended.

### d1/main/ files (24 files):
- [ ] `d1/main/automap.c`
- [ ] `d1/main/bm.c`
- [ ] `d1/main/console.c`
- [ ] `d1/main/credits.c`
- [ ] `d1/main/endlevel.c`
- [ ] `d1/main/game.c`
- [ ] `d1/main/gamecntl.c`
- [ ] `d1/main/gamefont.c`
- [ ] `d1/main/gamerend.c`
- [ ] `d1/main/gameseq.c`
- [ ] `d1/main/gauges.c`
- [ ] `d1/main/inferno.c`
- [ ] `d1/main/laser.c`
- [ ] `d1/main/laser.h`
- [ ] `d1/main/lighting.c`
- [ ] `d1/main/menu.c`
- [ ] `d1/main/menu.h`
- [ ] `d1/main/morph.c`
- [ ] `d1/main/newmenu.c`
- [ ] `d1/main/object.c`
- [ ] `d1/main/physics.c`
- [ ] `d1/main/polyobj.c`
- [ ] `d1/main/polyobj.h`
- [ ] `d1/main/render.c`
- [ ] `d1/main/render.h`
- [ ] `d1/main/state.c`
- [ ] `d1/main/terrain.c`
- [ ] `d1/main/titles.c`

### d1/include/ files (1 file):
- [ ] `d1/include/gr.h` (already covered in Step 5.5 but include here for completeness)

### d1/arch/ files (2 files):
- [ ] `d1/arch/sdl/event.c`
- [ ] `d1/arch/sdl/mouse.c`

### d1/3d/ files (2 files):
- [ ] `d1/3d/draw.c`
- [ ] `d1/3d/rod.c`

### d1/2d/ files (5 files):
- [ ] `d1/2d/bitmap.c`
- [ ] `d1/2d/bitblt.c`
- [ ] `d1/2d/circle.c`
- [ ] `d1/2d/disc.c`
- [ ] `d1/2d/font.c`
- [ ] `d1/2d/pcx.c`

### RT/ files (1 file):
- [ ] `RT/RText.h`

---

## Step 7: Verification — `Claude`

> **Why Claude:** Build troubleshooting requires understanding compiler errors in context, diagnosing linker failures across the ObjC++/C/C++ boundary, and making judgment calls about what to fix. If something goes wrong, Claude can reason about the full dependency graph. This is the wrong place to save tokens.

- [ ] Run `cmake --preset metal-mac-debug` — should configure without errors
- [ ] Run `cmake --build out/build/metal-mac-debug` — should compile without errors
- [ ] Run the built `descent1` executable:
  - [ ] SDL window appears
  - [ ] Metal device name logged to stderr
  - [ ] Screen clears to solid color (cornflower blue)
  - [ ] Stub messages appear in stderr for unimplemented functions
  - [ ] No crashes or hangs
- [ ] Verify Windows DX12 build is NOT broken:
  - [ ] Run `cmake --preset directx12-win-debug` (if on Windows) or verify no DX12 code was changed

---

## Summary: Task Distribution

| Agent | Tasks | Estimated % of work |
|-------|-------|---------------------|
| **Gemini** | 1.3, 2.3, 3.2, 3.3, 4.1, 4.2, 5.1, 5.2, 5.4, 5.5, **all of Step 6** (35+ files) | ~45% of files touched, ~15% of difficulty |
| **Codex** | 1.1, 1.2, 2.1, 2.2, 3.1, 3.4, 3.5, 4.3, 4.4, 4.6, 4.8, 5.3 | ~35% of files touched, ~40% of difficulty |
| **Claude** | 4.5, **4.7**, 7 | ~20% of files touched, ~45% of difficulty |

### Key principle
Claude touches only 3 tasks but they carry the most architectural risk. Gemini handles the bulk of file count (especially Step 6's 35+ files) with zero ambiguity. Codex handles everything in between — well-specified but requiring real coding judgment.
