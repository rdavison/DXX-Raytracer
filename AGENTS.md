# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

DXX-Raytracer is a raytracing fork of DXX-Retro that modernizes the graphics of Descent (1995) using DirectX 12 raytracing. It's built by a team at Breda University Games. The codebase mixes legacy C game code (from DXX-Retro/Descent) with a modern C++17 raytracing renderer.

## Build Commands

The project uses CMake with Ninja and requires MSVC (Windows-only for DX12):

```bash
# Configure (pick a preset)
cmake --preset directx12-win-debug
cmake --preset directx12-win-release    # RelWithDebInfo
cmake --preset directx12-win-ship       # Release/shipping
cmake --preset openGL-x64-debug
cmake --preset openGL-x64-release

# Build
cmake --build out/build/<preset-name>

# Quick-start presets (skip main menu, go to level select)
cmake --preset directx12-win-debug-quick
cmake --preset directx12-win-release-quick
```

There are no automated tests. CI (GitHub Actions on `windows-latest`) builds all four main presets: `directx12-win-debug`, `directx12-win-release`, `openGL-x64-debug`, `openGL-x64-release`.

The `EDITOR` environment variable enables the level editor build. `QUICK_START` env var skips the main menu.

## Architecture

The codebase has two major layers connected through a C API boundary:

### d1/ — Game Engine (C)
The original Descent game code. Contains gameplay logic (AI, physics, weapons, objects), 2D/3D graphics primitives, input handling (SDL2), audio, networking (UDP multiplayer), HUD/menus, and the level editor. This is mostly legacy C code with fixed-point math and original data formats.

- `d1/main/` — Core game logic (AI, physics, rendering integration, game state, save/load)
- `d1/arch/sdl/` — SDL2 platform layer (window, input, audio, timers)
- `d1/arch/ogl/` — OpenGL renderer (alternative to DX12 raytracing)
- `d1/2d/`, `d1/3d/` — 2D/3D graphics primitives, bitmaps, matrices
- `d1/maths/` — Fixed-point math, trigonometry, vectors
- `d1/editor/` — Level editor (built when `EDITOR` env var is set)

### RT/ — Raytracing Renderer (C/C++)
The modern rendering layer. Organized as:

- **RT/Core/** — Foundational utilities: arena allocator, config system, file I/O, math (`MiniMath.h`), string handling, memory management, vault system
- **RT/Game/** — Bridge between game and renderer: level geometry (`Level.c/h`), dynamic lighting (`Lights.c/h`)
- **RT/Renderer/Backend/Common/include/** — Renderer public API (`Renderer.h`, `ApiTypes.h`). This is the C-compatible interface the game calls into. Defines key constants (max textures, materials, segments, lights) and all renderer data structures.
- **RT/Renderer/Backend/DX12/** — DirectX 12 implementation:
  - `src/` — Core DX12 code: `Renderer.cpp` (main impl), `RenderBackend.cpp` (device/swapchain), `CommandQueue/CommandList` (GPU work), `Resource.cpp` (GPU memory), `ShaderTable.cpp` (raytracing dispatch), `GPUProfiler.cpp`, `FSR2.cpp` (AMD upscaling)
  - `assets/shaders/` — HLSL shaders for the raytracing pipeline
  - `assets/shaders/include_shared/` — `.hlsl.h` headers shared between C++ and HLSL (shared type definitions via `#define` aliasing between HLSL and C++ types)

### RT root files — Integration glue
- `RTgr.c/h` — Graphics interface bridging game's rendering calls to the RT renderer
- `RTmaterials.c/h` — Material system (PBR material definitions, loading, management)
- `dx12.c/h` — DX12 integration entry points called from game code
- `material_viewer.cpp/h`, `polymodel_viewer.cpp/h` — Debug tools (ImGui-based)

### External Dependencies (vendored)
- `sdl-master/` — SDL 2.0 (windowing, input)
- `physfs-main/` — PhysicsFS (virtual filesystem for game assets)
- `sgv-archiver/` — Seekable Game Vault format (custom asset archive)
- DX12 subdirectory has: CImGui, ImPlot, DXC (shader compiler), FSR2, DirectXTK12, CGLTF, STB, MikkTSpace

## Rendering Pipeline

The DX12 raytracing pipeline follows this approximate flow:
1. **Primary rays** (`primary_ray.hlsl`) — Ray generation from camera
2. **Direct lighting** (`direct_lighting_inline.hlsl`) — Light sampling with ReSTIR (`restir/gen_candidates.hlsl`)
3. **Indirect lighting** (`indirect_lighting_inline.hlsl`) — Path-traced global illumination
4. **Denoising** (`denoise/` — multi-pass: prepass, resample, post_resample, blur, history_fix)
5. **TAA** (`taa.hlsl`) — Temporal anti-aliasing
6. **Composite** (`composite.hlsl`) — Bloom, motion blur, post-processing
7. **Resolve** (`resolve_final_color.hlsl`) — Final color space conversion

BRDF calculations are in `brdf.hlsl`. Shared definitions live in `include/common.hlsl` and `include_shared/*.hlsl.h`.

## Key Patterns

- **C/C++ boundary**: The renderer API in `RT/Renderer/Backend/Common/include/Renderer.h` is a pure C interface (structs and function declarations) so the C game code can call into the C++ renderer. Keep this interface C-compatible.
- **Shared shader/CPU headers**: Files in `assets/shaders/include_shared/` use preprocessor tricks (`shared_defines.hlsl.h`) to define types that work in both HLSL and C++. When modifying shared structures, update both sides.
- **Material constants**: Material indices, triangle flags, and limits are defined in `Renderer.h` and must stay in sync with shader expectations.
- **Graphics API abstraction**: The build supports both DirectX12 (`RT_DX12` define) and OpenGL (`OGL` define), selected via the `GRAPHICS_API` CMake variable. The DX12 path includes raytracing; the OpenGL path is legacy rasterization.

## Texture Pipeline

PBR textures follow naming conventions: `*_basecolor.png`, `*_normal.png`, `*_metallic.png`, `*_roughness.png`, `*_emissive.png`. Use `PNGToDDS.py` with NVIDIA Texture Tools to convert PNG to DDS (with mips). Emissive textures must be pre-multiplied by alpha.
