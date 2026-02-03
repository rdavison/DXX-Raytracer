# Phase 2 Checklist — MVP C (Tight)

## Ordered Task Sequence

### 1) ImGui Metal Renderer (Core)
1.1. Audit current Metal backend ImGui stubs in `RT/Renderer/Backend/Metal/src/RenderBackend.mm`.
1.2. Decide render pass placement for ImGui overlay (after scene, before present) and wire it.
1.3. Implement/port ImGui Metal shaders (vertex + fragment) and add to build.
1.4. Create Metal pipeline state for ImGui (vertex + fragment shaders, blending enabled).
1.5. Create/resize per-frame vertex & index buffers for ImGui draw data.
1.6. Build and upload ImGui font atlas texture (keep CPU-side pixels for rebuilds).
1.7. Render ImGui draw lists into the active swapchain render pass.

### 2) Input + Timing (Interactivity)
2.1. Map SDL mouse position/buttons into ImGui IO each frame.
2.2. Map SDL scroll wheel to ImGui IO.
2.3. Map keyboard press/release + modifiers to ImGui IO.
2.4. Feed text input events (SDL_TEXTINPUT) to ImGui IO.
2.5. Respect `io.WantCaptureMouse/Keyboard` to avoid double input.
2.6. Update ImGui IO `DeltaTime` using a high‑res timer.

### 3) DPI + Display Size (Correctness)
3.1. Set `io.DisplaySize` from the logical window size each frame.
3.2. Set `io.DisplayFramebufferScale` for Retina (drawable size / window size).
3.3. Ensure Metal viewport/scissor uses drawable pixel size.
3.4. Handle font atlas rebuild + texture recreation on DPI changes.
3.5. Validate no scaling mismatch between ImGui and rendered scene.

### 4) Validation + Verification (Tight MVP)
4.1. Display RT debug menus (from `RT/RTgr.c`) and interact (buttons, sliders).
4.2. Confirm no ImGui asserts at runtime.
4.3. Run with Metal validation enabled; resolve warnings/errors.
4.4. Capture a short log proving: UI renders, input works, and no validation issues.
