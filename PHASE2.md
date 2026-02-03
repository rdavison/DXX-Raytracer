# Phase 2: MVP C (Tight) — ImGui Rendering on Metal

## Acceptance Checklist
- Widgets render correctly (text, buttons, sliders, checkboxes).
- Input parity: click, drag, scroll, keyboard text input.
- DPI correctness on Retina (no size mismatch or blur).
- No ImGui asserts.
- No Metal validation errors.

## Plan
1. Wire Metal ImGui renderer: build font atlas texture, create Metal pipeline/state, and render ImGui draw data into the swapchain render pass.
2. Implement ImGui input plumbing for SDL on macOS Metal: mouse/keyboard/scroll, text input, and per-frame DeltaTime updates.
3. Handle DPI/scale and display size consistently (Retina scaling, viewport/scissor).
4. Verify in-app debug UI (RTgr debug menus) renders and interacts; run with Metal validation enabled and fix issues.

## Audit Notes (1.1)
- Metal backend stubs in `RT/Renderer/Backend/Metal/src/RenderBackend.mm` include:
  - ImGui: `RenderImGuiTexture`, `RenderImGui`
  - Raster: `RasterSetViewport`, `RasterSetRenderTarget`, `RasterTriangles`, `RasterLines`, `RasterLinesWorld`, `RasterRender`, `RasterRenderDebugLines`, `RasterBlitScene`, `RasterBlit`
  - Resource: `UploadTexture`, `UploadMesh`, `ReleaseTexture`, `ReleaseMesh`, `UpdateMaterial`
  - Raytracing: `RaytraceSubmitLights`, `RaytraceMesh`, `RaytraceBillboardColored`, `RaytraceRod`, `RaytraceRender`, `RaytraceSetSkyColors`
  - Debug: `DoDebugMenus`, `QueueScreenshot`
- `EndFrame()` currently clears to cornflower blue and presents via `CAMetalLayer` with no draw calls.
- `OnWindowResize()` only updates `CAMetalLayer.drawableSize` and output dimensions.
