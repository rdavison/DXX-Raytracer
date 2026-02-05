# Phase 3 — Metal Raster MVP (Main Menu + HUD + Some Geometry)

## Goal
Render the **main menu**, **HUD**, and **some basic geometry** using the Metal backend. This phase focuses on the raster path (not raytracing), enabling enough functionality for on-screen visuals and interaction.

## Success Criteria
- Main menu is visible and interactive (mouse/keyboard).
- HUD renders with textures (cockpit/HUD elements).
- At least one simple geometry path renders (e.g., a few triangles/segments).
- No crashes; Metal validation enabled run has no **new** errors.

## Scope (In)
- Metal raster pipeline for triangles and blits.
- Texture upload for 2D assets (UI/HUD).
- Basic render target management for UI passes.
- Debug lines (optional but recommended if it unblocks troubleshooting).

## Scope (Out)
- Full raytracing path.
- Advanced post-processing and denoising.
- Material system parity with DX12.

## Implementation Strategy
1. **Raster core**: create a minimal Metal pipeline for textured triangles with basic vertex/index buffers.
2. **Texture upload**: upload 2D RGBA textures and bind them to the raster pipeline.
3. **UI/HUD**: wire the UI paths to use raster output (RenderImGuiTexture + HUD blits).
4. **Geometry**: implement a minimal mesh path for a small set of static triangles.

## Validation
- Run in Metal with validation enabled (MTL_DEBUG_LAYER=1).
- Capture logs showing menu/HUD rendering, and at least one geometry draw.
