# Metal Raytracing Implementation Plan (Phase 3, Step 4.3)

## Goal
Implement basic brute-force raytracing to verify geometry renders. This is a stepping stone to full raytracing, not a dead end.

## Files to Modify

1. **RT/Renderer/Backend/Metal/src/GlobalMetal.h** - Add raytracing data structures
2. **RT/Renderer/Backend/Metal/src/RenderBackend.mm** - Implement raytracing functions

## Implementation Steps

### Step 1: Add Data Structures (GlobalMetal.h)

Add after line 46 (after MeshResource):

```cpp
// GPU-compatible instance data for raytracing
struct RaytraceInstance
{
    RT_Mat4 object_to_world;
    RT_Mat4 world_to_object;
    uint32_t triangle_buffer_idx;  // Index into mesh slotmap
    uint32_t triangle_count;
    uint32_t color;
    uint32_t _pad;
};

// Scene constants for compute shader
struct RaytraceSceneConstants
{
    RT_Vec3 camera_position;  float _pad0;
    RT_Vec3 camera_forward;   float _pad1;
    RT_Vec3 camera_right;     float _pad2;
    RT_Vec3 camera_up;        float _pad3;
    float vfov_radians;
    float aspect_ratio;
    uint32_t render_width;
    uint32_t render_height;
    uint32_t instance_count;
    uint32_t total_triangles;
    float _pad4[2];
};
```

Add to MetalState struct (after line 150, before mesh_tracker):

```cpp
// Raytracing state
#ifdef __OBJC__
    id<MTLComputePipelineState> raytrace_pipeline;
    id<MTLBuffer> raytrace_instance_buffer;
    id<MTLBuffer> raytrace_scene_buffer;
    id<MTLTexture> raytrace_output_texture;
#else
    id raytrace_pipeline;
    id raytrace_instance_buffer;
    id raytrace_scene_buffer;
    id raytrace_output_texture;
#endif
    uint32_t raytrace_instance_count;
    std::vector<RT_ResourceHandle> raytrace_pending_meshes;
```

Rename `vertex_buffer` to `triangle_buffer` in MeshResource (line 41).

### Step 2: Add Compute Shader (RenderBackend.mm)

Add after kRasterLineShader (around line 183):

```metal
static const char* kRaytraceShader = R"metal(
#include <metal_stdlib>
using namespace metal;

struct Triangle {
    float3 pos0, pos1, pos2;
    float3 normal0, normal1, normal2;
    float4 tangent0, tangent1, tangent2;
    float2 uv0, uv1, uv2;
    uint color;
    uint material_edge_index;
};

struct Instance {
    float4x4 object_to_world;
    float4x4 world_to_object;
    uint triangle_buffer_idx;
    uint triangle_count;
    uint color;
    uint _pad;
};

struct SceneConstants {
    float3 camera_position;  float _pad0;
    float3 camera_forward;   float _pad1;
    float3 camera_right;     float _pad2;
    float3 camera_up;        float _pad3;
    float vfov_radians;
    float aspect_ratio;
    uint render_width;
    uint render_height;
    uint instance_count;
    uint total_triangles;
    float _pad4[2];
};

bool ray_tri_intersect(float3 ro, float3 rd, float3 v0, float3 v1, float3 v2,
                       thread float& t, thread float& u, thread float& v) {
    float3 e1 = v1 - v0, e2 = v2 - v0;
    float3 h = cross(rd, e2);
    float a = dot(e1, h);
    if (abs(a) < 1e-7) return false;
    float f = 1.0 / a;
    float3 s = ro - v0;
    u = f * dot(s, h);
    if (u < 0 || u > 1) return false;
    float3 q = cross(s, e1);
    v = f * dot(rd, q);
    if (v < 0 || u + v > 1) return false;
    t = f * dot(e2, q);
    return t > 0.001;
}

kernel void raytrace_main(
    texture2d<float, access::write> output [[texture(0)]],
    constant SceneConstants& scene [[buffer(0)]],
    constant Instance* instances [[buffer(1)]],
    constant Triangle* triangles [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= scene.render_width || gid.y >= scene.render_height) return;

    float2 ndc = float2(
        (float(gid.x) + 0.5) / float(scene.render_width) * 2.0 - 1.0,
        1.0 - (float(gid.y) + 0.5) / float(scene.render_height) * 2.0
    );

    float half_h = tan(scene.vfov_radians * 0.5);
    float3 rd = normalize(scene.camera_forward +
                          scene.camera_right * (ndc.x * half_h * scene.aspect_ratio) +
                          scene.camera_up * (ndc.y * half_h));
    float3 ro = scene.camera_position;

    float closest_t = 1e30;
    float3 hit_normal = float3(0);
    float4 hit_color = float4(0);

    uint tri_offset = 0;
    for (uint i = 0; i < scene.instance_count; i++) {
        constant Instance& inst = instances[i];
        float3 obj_ro = (inst.world_to_object * float4(ro, 1)).xyz;
        float3 obj_rd = normalize((inst.world_to_object * float4(rd, 0)).xyz);

        for (uint j = 0; j < inst.triangle_count; j++) {
            constant Triangle& tri = triangles[tri_offset + j];
            float t, u, v;
            if (ray_tri_intersect(obj_ro, obj_rd, tri.pos0, tri.pos1, tri.pos2, t, u, v)) {
                float3 hit_obj = obj_ro + obj_rd * t;
                float3 hit_world = (inst.object_to_world * float4(hit_obj, 1)).xyz;
                float world_t = length(hit_world - ro);
                if (world_t < closest_t) {
                    closest_t = world_t;
                    float w = 1.0 - u - v;
                    float3 n = normalize(tri.normal0*w + tri.normal1*u + tri.normal2*v);
                    float3x3 nm = float3x3(inst.object_to_world[0].xyz,
                                           inst.object_to_world[1].xyz,
                                           inst.object_to_world[2].xyz);
                    hit_normal = normalize(nm * n);
                    uint c = tri.color ? tri.color : inst.color;
                    hit_color = float4(float(c&0xFF)/255.0, float((c>>8)&0xFF)/255.0,
                                       float((c>>16)&0xFF)/255.0, 1.0);
                }
            }
        }
        tri_offset += inst.triangle_count;
    }

    float4 color;
    if (closest_t < 1e29) {
        float ndotl = max(0.0f, dot(hit_normal, normalize(float3(0.5,1,0.3))));
        color = float4(hit_color.rgb * (0.2 + 0.8*ndotl), 1);
    } else {
        float sky_t = rd.y * 0.5 + 0.5;
        color = float4(mix(float3(0.1,0.1,0.2), float3(0.5,0.7,1.0), sky_t), 1);
    }
    output.write(color, gid);
}
)metal";
```

### Step 3: Create Pipeline in Init()

Add after raster pipeline creation (around line 483):

```cpp
// Create raytracing compute pipeline
{
    NSError* error = nil;
    id<MTLLibrary> lib = [g_mtl.device newLibraryWithSource:
        [NSString stringWithUTF8String:kRaytraceShader] options:nil error:&error];
    if (lib) {
        id<MTLFunction> fn = [lib newFunctionWithName:@"raytrace_main"];
        if (fn) {
            g_mtl.raytrace_pipeline = [g_mtl.device newComputePipelineStateWithFunction:fn error:&error];
            if (g_mtl.raytrace_pipeline) MTL_LOG("Raytracing pipeline created");
        }
    }
    if (!g_mtl.raytrace_pipeline) MTL_LOG("Failed to create raytrace pipeline: %s",
        error ? error.localizedDescription.UTF8String : "unknown");

    g_mtl.raytrace_instance_buffer = [g_mtl.device newBufferWithLength:
        sizeof(RaytraceInstance) * MAX_INSTANCES options:MTLResourceStorageModeShared];
    g_mtl.raytrace_scene_buffer = [g_mtl.device newBufferWithLength:
        sizeof(RaytraceSceneConstants) options:MTLResourceStorageModeShared];
    g_mtl.raytrace_instance_count = 0;
}
```

### Step 4: Implement UploadMesh (replace stub at line 816)

```cpp
RT_ResourceHandle UploadMesh(const RT_UploadMeshParams& mesh_params)
{
    if (!mesh_params.triangles || mesh_params.triangle_count == 0)
        return RT_RESOURCE_HANDLE_NULL;

    size_t size = sizeof(RT_Triangle) * mesh_params.triangle_count;
    id<MTLBuffer> buf = [g_mtl.device newBufferWithBytes:mesh_params.triangles
                                                  length:size
                                                 options:MTLResourceStorageModeShared];
    if (!buf) return RT_RESOURCE_HANDLE_NULL;

    if (mesh_params.name)
        buf.label = [NSString stringWithUTF8String:mesh_params.name];

    MeshResource res = {};
    res.triangle_buffer = buf;
    res.triangle_count = (uint32_t)mesh_params.triangle_count;

    RT_ResourceHandle handle = g_mesh_slotmap.Insert(res);
    MTL_LOG("UploadMesh: %zu triangles -> handle %llu", mesh_params.triangle_count, handle.value);
    return handle;
}
```

### Step 5: Implement RaytraceMesh (replace stub at line 848)

```cpp
void RaytraceMesh(const RT_RenderMeshParams& params)
{
    if (g_mtl.raytrace_instance_count >= MAX_INSTANCES) return;

    MeshResource* mesh = g_mesh_slotmap.Find(params.mesh_handle);
    if (!mesh || !mesh->triangle_buffer) return;

    RaytraceInstance* instances = (RaytraceInstance*)[g_mtl.raytrace_instance_buffer contents];
    RaytraceInstance& inst = instances[g_mtl.raytrace_instance_count];

    inst.object_to_world = params.transform ? *params.transform : RT_Mat4Identity();
    inst.world_to_object = RT_Mat4Inverse(inst.object_to_world);
    inst.triangle_buffer_idx = params.mesh_handle.index;
    inst.triangle_count = mesh->triangle_count;
    inst.color = params.color ? params.color : 0xFFFFFFFF;

    g_mtl.raytrace_pending_meshes.push_back(params.mesh_handle);
    g_mtl.raytrace_instance_count++;
}
```

### Step 6: Implement RaytraceRender (replace stub at line 851)

```cpp
void RaytraceRender()
{
    if (g_mtl.raytrace_instance_count == 0 || !g_mtl.raytrace_pipeline) {
        g_mtl.raytrace_instance_count = 0;
        g_mtl.raytrace_pending_meshes.clear();
        return;
    }

    @autoreleasepool {
        uint32_t w = g_mtl.render_width > 0 ? g_mtl.render_width :
                     (uint32_t)g_mtl.metal_layer.drawableSize.width;
        uint32_t h = g_mtl.render_height > 0 ? g_mtl.render_height :
                     (uint32_t)g_mtl.metal_layer.drawableSize.height;
        if (w == 0 || h == 0) { g_mtl.raytrace_instance_count = 0; return; }

        // Ensure output texture
        if (!g_mtl.raytrace_output_texture ||
            g_mtl.raytrace_output_texture.width != w ||
            g_mtl.raytrace_output_texture.height != h) {
            MTLTextureDescriptor* desc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
            desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModeShared;
            g_mtl.raytrace_output_texture = [g_mtl.device newTextureWithDescriptor:desc];
        }

        // Build combined triangle buffer
        uint32_t total_tris = 0;
        for (uint32_t i = 0; i < g_mtl.raytrace_instance_count; i++) {
            MeshResource* m = g_mesh_slotmap.Find(g_mtl.raytrace_pending_meshes[i]);
            if (m) total_tris += m->triangle_count;
        }

        id<MTLBuffer> combined = [g_mtl.device newBufferWithLength:
            sizeof(RT_Triangle) * total_tris options:MTLResourceStorageModeShared];
        RT_Triangle* dst = (RT_Triangle*)[combined contents];
        uint32_t offset = 0;

        RaytraceInstance* instances = (RaytraceInstance*)[g_mtl.raytrace_instance_buffer contents];
        for (uint32_t i = 0; i < g_mtl.raytrace_instance_count; i++) {
            MeshResource* m = g_mesh_slotmap.Find(g_mtl.raytrace_pending_meshes[i]);
            if (m && m->triangle_buffer) {
                memcpy(dst + offset, [m->triangle_buffer contents],
                       sizeof(RT_Triangle) * m->triangle_count);
                offset += m->triangle_count;
            }
        }

        // Fill scene constants
        RaytraceSceneConstants* scene = (RaytraceSceneConstants*)[g_mtl.raytrace_scene_buffer contents];
        scene->camera_position = g_mtl.scene.camera.position;
        scene->camera_forward = g_mtl.scene.camera.forward;
        scene->camera_right = g_mtl.scene.camera.right;
        scene->camera_up = g_mtl.scene.camera.up;
        scene->vfov_radians = g_mtl.scene.camera.vfov * 3.14159f / 180.0f;
        scene->aspect_ratio = (float)w / (float)h;
        scene->render_width = w;
        scene->render_height = h;
        scene->instance_count = g_mtl.raytrace_instance_count;
        scene->total_triangles = total_tris;

        // Dispatch
        id<MTLCommandBuffer> cmd = [g_mtl.command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_mtl.raytrace_pipeline];
        [enc setTexture:g_mtl.raytrace_output_texture atIndex:0];
        [enc setBuffer:g_mtl.raytrace_scene_buffer offset:0 atIndex:0];
        [enc setBuffer:g_mtl.raytrace_instance_buffer offset:0 atIndex:1];
        [enc setBuffer:combined offset:0 atIndex:2];

        MTLSize tpg = MTLSizeMake(8, 8, 1);
        MTLSize groups = MTLSizeMake((w+7)/8, (h+7)/8, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        MTL_LOG("RaytraceRender: %u instances, %u triangles at %ux%u",
                g_mtl.raytrace_instance_count, total_tris, w, h);
    }

    g_mtl.raytrace_instance_count = 0;
    g_mtl.raytrace_pending_meshes.clear();
}
```

### Step 7: Display Output in PresentFrame

In PresentFrame(), after getting drawable (around line 645), add before raster batch encoding:

```cpp
// Blit raytraced output if available
if (g_mtl.raytrace_output_texture && g_mtl.scene.render_blit) {
    // Use existing RasterBlit mechanism - store texture for blit
    g_mtl.raster_render_target = g_mtl.raytrace_output_texture;
}
```

### Step 8: Add #include for vector

At top of RenderBackend.mm, ensure `<vector>` is included (already present at line 18).

## Verification

1. Build with `cmake --build out/build/metal-mac-debug`
2. Run game and load a level
3. Expected output in console:
   - "Raytracing pipeline created"
   - "UploadMesh: X triangles -> handle Y"
   - "RaytraceRender: N instances, M triangles at WxH"
4. Expected visual: Level geometry visible with simple lighting

## Potential Issues

- **Black screen**: Camera position/direction wrong - check camera vectors in scene constants
- **Sky only**: No instances queued - check UploadMesh returns valid handle
- **Crash**: Buffer size mismatch - verify RT_Triangle struct matches shader

## Future Improvements (Out of Scope)

- Metal acceleration structures for performance
- Proper material/texture sampling
- Motion vectors for TAA
