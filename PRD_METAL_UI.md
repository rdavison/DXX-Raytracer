# PRD — Metal UI Scaling + Flicker Stabilization

## Summary
Fix UI scaling and flicker in the Metal backend so the main menu and HUD render at the correct size on Retina displays, with stable background compositing and visible popup backgrounds.

## Problem Statement
On Metal (macOS), the UI appears too small and flickers between the background image and the blue clear color. Logs show a mismatch between logical viewport size (e.g., 1280x720) and drawable size (e.g., 2560x1440). The UI raster batches are normalized against the logical resolution, but presentation occurs at Retina resolution, causing a 2x scale mismatch and likely contributing to flicker when raster batches are missing or cleared between frames.

## Goals
- Correctly scale UI (menus, HUD, popups) to match the window’s actual drawable size on Retina displays.
- Eliminate flicker between the menu background and the clear color.
- Ensure menu popups render their background (no unintended transparency).
- Keep behavior consistent with DX12/OpenGL UI sizing expectations.

## Non-Goals
- Full Metal raytracing parity.
- Implementing missing 3D mesh/material features unrelated to UI scaling.
- Addressing DDS loading or material updates (unless required to fix UI).

## User Impact
- Main menu renders at correct size (not small or centered only).
- UI background is stable and does not flicker.
- Popup background is visible (opaque where expected).

## Requirements
### Functional
1. UI raster triangles should map 1:1 to the drawable size on Retina (scale factor applied correctly).
2. Menu background must render every frame without flicker.
3. Popup background must render (no missing draw call or incorrect alpha).
4. HUD elements respect the expected screen resolution (logical size) but are presented correctly at Retina resolution.

### Technical
1. Determine and apply a consistent UI scale factor (e.g., `drawable_size / logical_size`).
2. Scale viewport or vertex positions so that logical UI coordinates match drawable pixels.
3. Avoid per-frame clearing that wipes UI if no batches are submitted.
4. Preserve existing raster batching and texture sampling behavior.

## Proposed Approach
- Derive UI scale from `drawable_size` and `grd_curscreen->sc_w/sc_h` (or `last_width/last_height`).
- Apply scale in one place:
  - Option A: adjust Metal raster viewport to `drawable_size` and scale vertices for UI;
  - Option B (preferred): scale viewport to drawable size using `scale_x/scale_y` while keeping vertex normalization based on logical resolution.
- Ensure background is drawn before clear or disable unnecessary clears when batches exist.
- Add lightweight runtime diagnostics to verify scaling and batch counts.

## Acceptance Criteria
1. On a Retina display, menu text and background fill the correct screen area with no visible scaling mismatch.
2. No flicker between menu background and blue clear screen over a 15s run.
3. Name entry popup background is visible and opaque where expected.
4. Logs show consistent viewport and target sizes, with scale applied.

## Risks
- Fixing scale may require careful interaction with existing viewport/canvas logic in `metal_bridge.c`.
- Flicker may also be influenced by missing render target usage or batch flushing order.
- Over-correcting scale could distort non-UI rendering paths.

## Instrumentation
- Log once per second:
  - `last_width/last_height`, `sc_w/sc_h`, canvas size, viewport size, drawable size, batch count.
- Optional: a toggle to enable/disable logging.

## Milestones
1. Apply scale correction and validate in menu (visual check).
2. Remove flicker by stabilizing raster submission and render pass clearing.
3. Validate HUD and popup backgrounds.
4. Clean up instrumentation (or keep behind a debug flag).

## Test Plan
- Run `./out/build/metal-mac-debug/d1/descent1` for 15s on Retina display.
- Verify UI size, background stability, and popup background visibility.
- Compare behavior to DX12 path for parity.
