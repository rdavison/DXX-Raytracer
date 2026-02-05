# Phase 3 Checklist — Metal Raster MVP

## Ordered Task Sequence

### 1) Raster Pipeline Core
1.1. Implement `RasterSetViewport` to set Metal viewport/scissor.
1.2. Implement a minimal `RasterTriangles` path: create vertex/index buffers, encode draw calls.
1.3. Implement `RasterRender` to submit the raster command list into the swapchain pass.

### 2) Texture Upload + Binding
2.1. Implement `UploadTexture` for RGBA8 textures (CPU -> MTLTexture).
2.2. Implement a simple sampler and bind texture/sampler in the raster pipeline.
2.3. Implement `RenderImGuiTexture` using a basic textured quad.

### 3) Render Targets + Blit
3.1. Implement `RasterSetRenderTarget` for offscreen targets.
3.2. Implement `RasterBlit` to copy a texture to the swapchain.
3.3. Implement `RasterBlitScene` for UI/HUD composites.

### 4) HUD + Menu Validation
4.1. Verify main menu renders (text + background).
4.2. Verify HUD textures render (cockpit/HUD elements).
4.3. Verify a simple geometry path renders (any visible triangles).

### 5) Debug + Validation
5.1. (Optional) Implement `RasterRenderDebugLines` for overlays.
5.2. Run with Metal validation enabled; fix any new errors.
5.3. Capture a short log demonstrating menu/HUD/geometry rendering.
