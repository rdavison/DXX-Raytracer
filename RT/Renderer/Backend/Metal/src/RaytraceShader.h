#pragma once

// Metal raytrace compute shader source (compiled at runtime from string)
// Extracted from RenderBackend.mm for maintainability

static const char* kRaytraceShader = R"metal(
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// Constants matching Renderer.h
#define RT_TRIANGLE_HOLDS_MATERIAL_EDGE  (1u << 31)
#define RT_TRIANGLE_HOLDS_MATERIAL_INDEX (1u << 30)
#define RT_MAX_TEXTURES 6030

struct Triangle {
    packed_float3 pos0;
    packed_float3 pos1;
    packed_float3 pos2;
    packed_float3 normal0;
    packed_float3 normal1;
    packed_float3 normal2;
    // Use float arrays instead of float4 to avoid 16-byte alignment padding
    float tangent0[4];
    float tangent1[4];
    float tangent2[4];
    packed_float2 uv0;
    packed_float2 uv1;
    packed_float2 uv2;
    uint color;
    uint material_edge_index;
};

struct Instance {
    float4x4 object_to_world;
    float4x4 world_to_object;
    uint triangle_buffer_idx;
    uint triangle_count;
    uint color;
    uint material_override;
    uint triangle_offset;
    uint _pad[3];
};

struct SceneConstants {
    packed_float3 camera_position; float _pad0;
    packed_float3 camera_forward;  float _pad1;
    packed_float3 camera_right;    float _pad2;
    packed_float3 camera_up;       float _pad3;
    float vfov_radians;
    float aspect_ratio;
    uint render_width;
    uint render_height;
    uint instance_count;
    uint total_triangles;
    float _pad4[2];
    uint debug_mode;
    uint texture_count;
    uint use_accel;
    uint light_count;
};

#define RT_LIGHT_SCALE 1000.0f

// GPU Light struct (matches RT_Light in C++)
struct Light {
    uint kind_packed;     // kind, spot_angle, spot_softness, spot_vignette packed as 4 bytes
    uint emission;        // RGBE packed color
    float transform[12];  // 3x4 row-major matrix
};

// Decode RGBE packed emission to float3 color
// Format: 9-bit R [0:8], 9-bit G [9:17], 9-bit B [18:26], 5-bit exponent [27:31] (bias 20)
// Matches RT_UnpackRGBE in MiniMath.h and UnpackRGBE in common.hlsl
float3 decode_rgbe(uint rgbe) {
    int exponent = int(rgbe >> 27) - 20;
    float scale = exp2(float(exponent)) / 256.0;
    float r = float((rgbe >>  0) & 0x1FFu) * scale;
    float g = float((rgbe >>  9) & 0x1FFu) * scale;
    float b = float((rgbe >> 18) & 0x1FFu) * scale;
    return float3(r, g, b) * RT_LIGHT_SCALE;
}

// GPU Material struct (matches GPUMaterial in C++)
struct Material {
    uint albedo_index;
    uint normal_index;
    uint metalness_index;
    uint roughness_index;
    uint emissive_index;
    uint height_index;
    uint flags;
    float metalness_factor;
    float roughness_factor;
    uint emissive_factor;
    uint _pad[2];
};

// Get material index from material_indices array (packed uint16s)
uint get_material_index(constant uint* material_indices, uint material_edge) {
    uint word_idx = material_edge >> 1;
    uint word = material_indices[word_idx];
    if (material_edge & 1) {
        return (word >> 16) & 0xFFFF;
    } else {
        return word & 0xFFFF;
    }
}

// Decode material_edge_index to get material index
uint resolve_material_index(uint material_edge_index, uint material_override,
                            constant uint* material_edges,
                            constant uint* material_indices) {
    if (material_override != 0) {
        return material_override;
    }

    if (material_edge_index & RT_TRIANGLE_HOLDS_MATERIAL_INDEX) {
        return material_edge_index & ~RT_TRIANGLE_HOLDS_MATERIAL_INDEX;
    }

    if (material_edge_index & RT_TRIANGLE_HOLDS_MATERIAL_EDGE) {
        uint edge_idx = material_edge_index & ~RT_TRIANGLE_HOLDS_MATERIAL_EDGE;
        return get_material_index(material_indices, edge_idx);
    }

    // Normal case: look up in material_edges, then material_indices
    uint edge = material_edges[material_edge_index];
    uint mat1 = edge & 0xFFFF;
    return get_material_index(material_indices, mat1);
}

// ---- Tone mapping (matches DX12 post_process.hlsl) ----

float NaturalShoulder(float x) {
    return 1.0 - exp(-x);
}

float NaturalShoulderLinear(float x, float t) {
    if (x <= t) return x;
    return t + (1.0 - t) * NaturalShoulder((x - t) / (1.0 - t));
}

float3 NaturalShoulderLinear3(float3 x, float t) {
    return float3(
        NaturalShoulderLinear(x.x, t),
        NaturalShoulderLinear(x.y, t),
        NaturalShoulderLinear(x.z, t)
    );
}

float3 ApplyTonemappingCurve(float3 color) {
    float linear_section = 0.25;
    float whitepoint = 8.0;
    return NaturalShoulderLinear3(color, linear_section) / NaturalShoulderLinear(whitepoint, linear_section);
}

float LinearTosRGB(float x) {
    return x <= 0.0031308 ? x * 12.92 : pow(1.055 * x, 1.0 / 2.4) - 0.055;
}

float3 LinearTosRGB3(float3 x) {
    return float3(LinearTosRGB(x.x), LinearTosRGB(x.y), LinearTosRGB(x.z));
}

// Use 31 textures max (Metal compute shader limit without argument buffers)
#define MAX_BOUND_TEXTURES 31

kernel void raytrace_main(
    texture2d<float, access::write> output [[texture(0)]],
    array<texture2d<float>, MAX_BOUND_TEXTURES> textures [[texture(1)]],
    sampler tex_sampler [[sampler(0)]],
    constant SceneConstants& scene [[buffer(0)]],
    constant Instance* instances [[buffer(1)]],
    constant Triangle* triangles [[buffer(2)]],
    device atomic_uint* hit_counter [[buffer(3)]],
    constant Material* materials [[buffer(4)]],
    constant uint* material_edges [[buffer(5)]],
    constant uint* material_indices [[buffer(6)]],
    constant uint* texture_remap [[buffer(7)]],
    instance_acceleration_structure accel_struct [[buffer(8)]],
    constant Light* lights [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= scene.render_width || gid.y >= scene.render_height) return;

    float2 ndc = float2(
        (float(gid.x) + 0.5) / float(scene.render_width) * 2.0 - 1.0,
        1.0 - (float(gid.y) + 0.5) / float(scene.render_height) * 2.0
    );

    float half_h = tan(scene.vfov_radians * 0.5);
    if (half_h < 0.0001f) {
        half_h = 1.0f;
    }
    float3 rd = normalize(scene.camera_forward +
                          scene.camera_right * (ndc.x * half_h * scene.aspect_ratio) +
                          scene.camera_up * (ndc.y * half_h));
    float3 ro = scene.camera_position;

    float closest_t = 1e30;
    float3 hit_normal = float3(0);
    float4 hit_color = float4(0);
    float2 hit_uv = float2(0);
    uint hit_material_index = 0;
    uint hit_instance_material_override = 0;

    if (scene.use_accel) {
        // ============================================================
        // Acceleration structure path: O(log n) intersection
        // ============================================================
        ray r;
        r.origin = ro;
        r.direction = rd;
        r.min_distance = 0.001;
        r.max_distance = 1e30;

        intersector<triangle_data, instancing> i;
        i.accept_any_intersection(false);  // We want closest hit
        auto result = i.intersect(r, accel_struct);

        if (result.type == intersection_type::triangle) {
            closest_t = result.distance;
            uint inst_id = result.instance_id;
            uint prim_id = result.primitive_id;
            float2 bary = result.triangle_barycentric_coord;
            float u = bary.x;
            float v = bary.y;
            float w = 1.0 - u - v;

            constant Instance& inst = instances[inst_id];
            uint global_tri_idx = inst.triangle_offset + prim_id;
            constant Triangle& tri = triangles[global_tri_idx];

            // Interpolate normal in object space, transform to world space
            float3 n0 = float3(tri.normal0);
            float3 n1 = float3(tri.normal1);
            float3 n2 = float3(tri.normal2);
            float3 obj_normal = normalize(n0 * w + n1 * u + n2 * v);
            hit_normal = normalize((inst.world_to_object * float4(obj_normal, 0.0)).xyz);

            // Interpolate UV
            float2 uv0 = float2(tri.uv0);
            float2 uv1 = float2(tri.uv1);
            float2 uv2 = float2(tri.uv2);
            hit_uv = uv0 * w + uv1 * u + uv2 * v;

            // Material info
            hit_material_index = tri.material_edge_index;
            hit_instance_material_override = inst.material_override;

            // Vertex/instance color
            uint c = tri.color ? tri.color : inst.color;
            hit_color = float4(float(c & 0xFF) / 255.0, float((c >> 8) & 0xFF) / 255.0,
                               float((c >> 16) & 0xFF) / 255.0, 1.0);
        }
    } else {
        // ============================================================
        // Brute-force fallback path: O(n) intersection
        // ============================================================
        for (uint i = 0; i < scene.instance_count; i++) {
            constant Instance& inst = instances[i];

            float4x4 w2o = transpose(inst.world_to_object);
            float4x4 o2w = transpose(inst.object_to_world);

            float3 obj_ro = (w2o * float4(ro, 1.0)).xyz;
            float3 obj_rd = normalize((w2o * float4(rd, 0.0)).xyz);

            uint tri_off = inst.triangle_offset;
            for (uint j = 0; j < inst.triangle_count; j++) {
                constant Triangle& tri = triangles[tri_off + j];
                float3 p0 = float3(tri.pos0);
                float3 p1 = float3(tri.pos1);
                float3 p2 = float3(tri.pos2);

                // Inline ray-triangle intersection (Moller-Trumbore)
                float3 e1 = p1 - p0, e2 = p2 - p0;
                float3 h = cross(obj_rd, e2);
                float a = dot(e1, h);
                if (abs(a) < 1e-7) continue;
                float f = 1.0 / a;
                float3 s = obj_ro - p0;
                float u = f * dot(s, h);
                if (u < 0 || u > 1) continue;
                float3 q = cross(s, e1);
                float v = f * dot(obj_rd, q);
                if (v < 0 || u + v > 1) continue;
                float t = f * dot(e2, q);
                if (t <= 0.001) continue;

                float3 obj_hit = obj_ro + obj_rd * t;
                float3 world_hit = (o2w * float4(obj_hit, 1.0)).xyz;
                float world_t = length(world_hit - ro);

                if (world_t < closest_t) {
                    closest_t = world_t;
                    float w_bary = 1.0 - u - v;
                    float3 n0 = float3(tri.normal0);
                    float3 n1 = float3(tri.normal1);
                    float3 n2 = float3(tri.normal2);
                    float3 obj_normal = normalize(n0 * w_bary + n1 * u + n2 * v);
                    hit_normal = normalize((inst.world_to_object * float4(obj_normal, 0.0)).xyz);
                    hit_uv = float2(tri.uv0) * w_bary + float2(tri.uv1) * u + float2(tri.uv2) * v;
                    hit_material_index = tri.material_edge_index;
                    hit_instance_material_override = inst.material_override;
                    uint c = tri.color ? tri.color : inst.color;
                    hit_color = float4(float(c & 0xFF) / 255.0, float((c >> 8) & 0xFF) / 255.0,
                                       float((c >> 16) & 0xFF) / 255.0, 1.0);
                }
            }
        }
    }

    float4 color;
    if (closest_t < 1e29) {
        atomic_fetch_add_explicit(hit_counter, 1, memory_order_relaxed);

        // Resolve material and sample texture
        uint mat_idx = resolve_material_index(hit_material_index, hit_instance_material_override,
                                              material_edges, material_indices);
        mat_idx = min(mat_idx, (uint)(RT_MAX_TEXTURES - 1));

        constant Material& mat = materials[mat_idx];
        uint tex_idx = mat.albedo_index;

        float4 albedo = hit_color;

        if (scene.debug_mode == 1) {
            albedo = float4(fract(hit_uv.x), fract(hit_uv.y), 0.5, 1.0);
        }
        else if (scene.debug_mode == 2) {
            float hue = fract(float(mat_idx) * 0.1);
            albedo = float4(hue, 1.0 - hue, fract(hue * 3.0), 1.0);
        }
        else if (tex_idx > 0 && tex_idx < RT_MAX_TEXTURES) {
            uint slot = texture_remap[tex_idx];
            if (slot > 0 && slot < MAX_BOUND_TEXTURES) {
                albedo = textures[slot].sample(tex_sampler, hit_uv);
                albedo.rgb *= hit_color.rgb;
            }
        }

        // Lighting
        float3 hit_pos = ro + rd * closest_t;
        float3 total_light = float3(0.05);  // Small ambient

        if (scene.light_count > 0 && scene.use_accel) {
            // Evaluate submitted lights with shadow rays
            // We sum all lights (DX12 importance-samples one via ReSTIR).
            // To compensate, we use a smoothed distance falloff that prevents
            // nearby lights from dominating, similar to a radius-based model.
            for (uint li = 0; li < scene.light_count; li++) {
                // Extract light position from transform (column 3 of 3x4 row-major matrix)
                float3 light_pos = float3(
                    lights[li].transform[3],
                    lights[li].transform[7],
                    lights[li].transform[11]);

                float3 light_emission = decode_rgbe(lights[li].emission);

                float3 to_light = light_pos - hit_pos;
                float dist_sq = dot(to_light, to_light);
                float dist = sqrt(dist_sq);
                if (dist < 0.001) continue;
                float3 L = to_light / dist;

                float NdotL = max(0.0f, dot(hit_normal, L));
                if (NdotL <= 0.0) continue;

                // Smooth distance attenuation with minimum radius to prevent
                // singularity near lights. radius=1.0 means lights at dist<1
                // don't get brighter than at dist=1.
                float min_radius = 1.0;
                float attenuation = 1.0 / max(dist_sq, min_radius * min_radius);

                // Shadow ray using acceleration structure
                ray shadow_ray;
                shadow_ray.origin = hit_pos + hit_normal * 0.002;
                shadow_ray.direction = L;
                shadow_ray.min_distance = 0.001;
                shadow_ray.max_distance = dist - 0.002;

                intersector<triangle_data, instancing> shadow_i;
                shadow_i.accept_any_intersection(true);  // Any hit = occluded
                auto shadow_result = shadow_i.intersect(shadow_ray, accel_struct);
                float visibility = (shadow_result.type == intersection_type::none) ? 1.0 : 0.0;

                total_light += light_emission * NdotL * attenuation * visibility;
            }
        } else {
            // Fallback: hardcoded directional light when no lights submitted
            float ndotl = max(0.0f, dot(hit_normal, normalize(float3(0.5, 1, 0.3))));
            total_light = float3(0.2 + 0.8 * ndotl);
        }

        // Apply Lambertian BRDF (energy conservation: diffuse = albedo / pi)
        float3 hdr = (albedo.rgb / 3.14159265) * total_light;

        // Exposure adjustment (matching DX12 default of 0.1)
        hdr *= exp2(0.1);

        // Tone map: NaturalShoulder with linear section + whitepoint (matches DX12)
        float3 ldr = ApplyTonemappingCurve(hdr);

        // Linear to sRGB
        ldr = LinearTosRGB3(ldr);

        color = float4(ldr, 1);
    } else {
        float sky_t = rd.y * 0.5 + 0.5;
        color = float4(mix(float3(0.1, 0.1, 0.2), float3(0.5, 0.7, 1.0), sky_t), 1);
    }
    output.write(color, gid);
}
)metal";
