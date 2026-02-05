# Metal Raytracing: Acceleration Structures & Lighting

## Current State

The Metal backend traces primary rays via a compute shader with brute-force triangle intersection (~2K triangles, O(pixels * triangles)). Texture mapping works via dynamic per-frame remapping. Lighting is a hardcoded directional light with ambient term. No acceleration structure, no shadow rays, no light submission.

## Implementation Order

1. Metal Acceleration Structures (performance foundation)
2. Light Submission (get light data to GPU)
3. Point Light Shading + Shadow Rays (real lighting)
4. Emissive Materials (self-illuminating surfaces)

---

## Phase 1: Metal Acceleration Structures

### Goal
Replace brute-force ray-triangle intersection with Metal's native `MTLAccelerationStructure` API, enabling O(log n) ray queries.

### Architecture

Metal uses a two-level acceleration structure, same as DX12:
- **BLAS (Bottom-Level)**: Per-mesh triangle geometry, built once at mesh upload
- **TLAS (Top-Level)**: Per-frame instance transforms pointing to BLASes, rebuilt every frame

### Data Flow

```
UploadMesh() -> Build BLAS per mesh, store in MeshResource
RaytraceMesh() -> Collect instance descriptors (transform + BLAS ref)
RaytraceRender() -> Build TLAS from instances, dispatch intersection shader
```

### Step 1.1: BLAS Creation in UploadMesh()

When a mesh is uploaded, build a `MTLPrimitiveAccelerationStructure`:

```objc
// In UploadMesh(), after creating triangle_buffer:
MTLAccelerationStructureTriangleGeometryDescriptor *geomDesc =
    [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
geomDesc.vertexBuffer = positions_buffer;  // Extract float3 positions from RT_Triangle
geomDesc.vertexStride = sizeof(float) * 3;
geomDesc.triangleCount = triangle_count;

MTLPrimitiveAccelerationStructureDescriptor *blasDesc =
    [MTLPrimitiveAccelerationStructureDescriptor descriptor];
blasDesc.geometryDescriptors = @[geomDesc];

id<MTLAccelerationStructure> blas = [device newAccelerationStructureWithDescriptor:blasDesc];
// Build with command buffer encoder
```

**Key detail**: `RT_Triangle` stores positions interleaved with normals/UVs/etc (stride = `sizeof(RT_Triangle)`). We need to either:
- (a) Extract positions into a separate compact buffer (3 * float3 per triangle), or
- (b) Use `vertexStride = sizeof(RT_Triangle)` with proper offsets and `indexBuffer` for vertex indices

Option (a) is simpler. Build a positions-only buffer at upload time.

**MeshResource changes**:
```cpp
struct MeshResource {
    id<MTLBuffer> triangle_buffer;
    id<MTLBuffer> position_buffer;           // float3 positions only (for BLAS)
    id<MTLAccelerationStructure> blas;       // Bottom-level acceleration structure
    uint32_t triangle_count;
};
```

### Step 1.2: TLAS Creation in RaytraceRender()

Each frame, build a `MTLInstanceAccelerationStructure` from submitted instances:

```objc
MTLInstanceAccelerationStructureDescriptor *tlasDesc =
    [MTLInstanceAccelerationStructureDescriptor descriptor];
tlasDesc.instanceDescriptorBuffer = instance_descriptor_buffer;
tlasDesc.instanceCount = instance_count;
tlasDesc.instancedAccelerationStructures = @[blas1, blas2, ...];
```

**Instance descriptor buffer**: Array of `MTLAccelerationStructureInstanceDescriptor`:
```c
typedef struct {
    MTLPackedFloat4x3 transformationMatrix;  // 3x4 row-major transform
    uint32_t options;                         // MTLAccelerationStructureInstanceOptions
    uint32_t mask;                            // Intersection mask
    uint32_t intersectionFunctionTableOffset;
    uint32_t accelerationStructureIndex;      // Index into instancedAccelerationStructures
} MTLAccelerationStructureInstanceDescriptor;
```

### Step 1.3: Shader Changes

Replace the manual ray-triangle loop with Metal's intersection query:

```metal
#include <metal_raytracing>
using raytracing::intersector;
using raytracing::intersection_result;

// In kernel signature:
instance_acceleration_structure accel_struct [[buffer(8)]]

// In kernel body:
raytracing::ray ray;
ray.origin = ro;
ray.direction = rd;
ray.min_distance = 0.001;
ray.max_distance = 1e30;

intersector<triangle_data> intersector;
auto result = intersector.intersect(ray, accel_struct);

if (result.type == intersection_type::triangle) {
    uint instance_id = result.instance_id;
    uint prim_id = result.primitive_id;
    float2 barycentrics = result.triangle_barycentric_coord;
    float t = result.distance;

    // Look up triangle data from combined buffer using instance_id + prim_id
    // Interpolate normals, UVs using barycentrics
    // Sample texture, compute lighting
}
```

### Step 1.4: Retain Triangle Data for Shading

The acceleration structure handles intersection but we still need triangle attributes (normals, UVs, material indices) for shading. Keep the combined triangle buffer and instance buffer bound to the shader. Use `instance_id` to find the instance, then `primitive_id` to index into that instance's triangle range.

### Files to Modify

- **GlobalMetal.h**: Add BLAS to `MeshResource`, TLAS-related buffers to `MetalState`
- **RenderBackend.mm**:
  - `UploadMesh()`: Build position buffer + BLAS
  - `RaytraceRender()`: Build TLAS, bind to shader
  - Shader: Replace manual loop with `intersector`
- **RenderBackend.h**: No changes needed

### Risks / Considerations

- `MTLAccelerationStructure` requires macOS 13+ / Apple Silicon. Verify min deployment target.
- BLAS build is async via command buffer - need to ensure it completes before first use.
- The shader must switch from compute kernel to a Metal ray tracing pipeline, or use `intersection_query` within the compute kernel (available on Apple Silicon with Metal 3).
- Scratch buffers needed for AS builds - size queried from descriptor.

---

## Phase 2: Light Submission

### Goal
Accept lights from the game and upload them to the GPU.

### Game API

The game calls `RT_RaytraceSubmitLights(count, lights)` with an array of `RT_Light`:
```c
typedef struct RT_Light {
    uint8_t kind;           // 0=Area_Sphere, 1=Area_Rect
    uint8_t spot_angle;
    uint8_t spot_softness;
    uint8_t spot_vignette;
    uint32_t emission;      // RGBE packed color
    RT_Mat34 transform;     // Position + orientation (3x4 matrix)
} RT_Light;
```

### Implementation

1. **Add light buffer to MetalState**:
   ```cpp
   id<MTLBuffer> raytrace_light_buffer;  // RT_Light[RT_MAX_LIGHTS]
   uint32_t raytrace_light_count;
   ```

2. **Implement RaytraceSubmitLights()**:
   ```cpp
   void RaytraceSubmitLights(size_t count, const RT_Light* lights) {
       if (g_mtl.raytrace_light_count + count > RT_MAX_LIGHTS)
           count = RT_MAX_LIGHTS - g_mtl.raytrace_light_count;
       RT_Light* dst = (RT_Light*)[g_mtl.raytrace_light_buffer contents];
       memcpy(dst + g_mtl.raytrace_light_count, lights, sizeof(RT_Light) * count);
       g_mtl.raytrace_light_count += count;
   }
   ```

3. **Add light_count to SceneConstants**, bind light buffer to shader.

4. **Reset light count** at start of each frame.

### Files to Modify

- **GlobalMetal.h**: Add light buffer + count to MetalState, add light_count to SceneConstants
- **RenderBackend.mm**: Implement RaytraceSubmitLights, create buffer in Init(), bind in RaytraceRender()

---

## Phase 3: Point Light Shading + Shadow Rays

### Goal
Evaluate submitted lights at hit points with distance attenuation and trace shadow rays.

### Light Evaluation (Shader)

For each hit point, loop through lights:
```metal
float3 total_light = float3(0.05); // Small ambient

for (uint i = 0; i < scene.light_count; i++) {
    // Decode light position from transform column 3
    float3 light_pos = float3(lights[i].transform[3], lights[i].transform[7], lights[i].transform[11]);

    // Decode RGBE emission
    float3 emission = decode_rgbe(lights[i].emission);

    float3 to_light = light_pos - hit_pos;
    float dist_sq = dot(to_light, to_light);
    float dist = sqrt(dist_sq);
    float3 L = to_light / dist;

    float NdotL = max(0.0, dot(hit_normal, L));
    float attenuation = 1.0 / dist_sq;

    // Shadow ray (Phase 1 acceleration structure required)
    raytracing::ray shadow_ray;
    shadow_ray.origin = hit_pos + hit_normal * 0.001; // Bias to avoid self-intersection
    shadow_ray.direction = L;
    shadow_ray.min_distance = 0.001;
    shadow_ray.max_distance = dist - 0.001;

    auto shadow_result = intersector.intersect(shadow_ray, accel_struct);
    float visibility = (shadow_result.type == intersection_type::none) ? 1.0 : 0.0;

    total_light += emission * NdotL * attenuation * visibility;
}

color = albedo * total_light;
```

### RGBE Decoding
```metal
float3 decode_rgbe(uint rgbe) {
    float r = float(rgbe & 0xFF) / 255.0;
    float g = float((rgbe >> 8) & 0xFF) / 255.0;
    float b = float((rgbe >> 16) & 0xFF) / 255.0;
    float e = float((rgbe >> 24)) - 128.0;
    float scale = exp2(e) * RT_LIGHT_SCALE;
    return float3(r, g, b) * scale;
}
```

### RT_Mat34 Layout
The `RT_Mat34` transform stores the light position in the translation column. Need to verify the exact memory layout (row-major vs column-major) by checking `RT_Mat34` definition and how the DX12 shader reads it.

### Files to Modify
- **RenderBackend.mm** (shader): Add light loop, shadow ray tracing, RGBE decode
- Light evaluation can be a separate function in the shader for clarity

---

## Phase 4: Emissive Materials

### Goal
Surfaces with `emissive_strength > 0` should contribute light to the scene.

### Implementation
After resolving the material in the shader, check emissive properties:
```metal
if (mat.emissive_factor > 0) {
    float strength = float(mat.emissive_factor) / 255.0;
    // Emissive surfaces glow regardless of lighting
    color.rgb += albedo.rgb * strength;
}
```

This is simple additive emission - emissive surfaces appear bright but don't cast light onto other surfaces (that would require global illumination / bounce rays, which is a future enhancement).

### Files to Modify
- **RenderBackend.mm** (shader): Add emissive check after material resolution

---

## Dependency Graph

```
Phase 1: Acceleration Structures
    ├── Step 1.1: BLAS in UploadMesh
    ├── Step 1.2: TLAS in RaytraceRender  (depends on 1.1)
    ├── Step 1.3: Shader intersection     (depends on 1.2)
    └── Step 1.4: Attribute lookup         (depends on 1.3)

Phase 2: Light Submission                  (independent of Phase 1)
    └── Buffer creation + data upload

Phase 3: Shadow Rays + Lighting            (depends on Phase 1 + Phase 2)
    ├── Light evaluation loop
    ├── Shadow ray tracing
    └── RGBE decoding

Phase 4: Emissive Materials                (independent, can be done anytime)
```

## Performance Expectations

| Stage | Rays/pixel | Expected Performance |
|-------|-----------|---------------------|
| Current (brute force primary) | 1 | Slow, ~2K tri tests/pixel |
| After Phase 1 (accel struct) | 1 | Fast, ~20 tri tests/pixel |
| After Phase 3 (shadow rays) | 1 + N_lights | Fast per ray, but N_lights shadow rays each |

With `RT_MAX_LIGHTS = 100`, worst case is 101 rays per pixel. In practice, lights can be culled by distance to reduce shadow ray count.

## Future Enhancements (Not in Scope)

- **Global Illumination**: Bounce rays for indirect lighting
- **ReSTIR**: Importance sampling for many-light scenes (DX12 backend has this)
- **Denoising**: Temporal/spatial denoising for noisy stochastic lighting
- **Motion Vectors**: Previous-frame transform tracking for temporal effects
