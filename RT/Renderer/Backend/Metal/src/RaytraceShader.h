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
#define RT_TRIANGLE_ALPHA_CUTOUT         (1u << 29)
#define RT_TRIANGLE_MEI_MASK             0x1FFFFFFFu  // mask out flag bits 29-31
#define RT_MAX_TEXTURES 6030
#define RT_MATERIAL_FLAG_ALPHA_CUTOUT    0x10

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

    // Normal case: mask out flag bits, look up in material_edges, then material_indices
    uint edge = material_edges[material_edge_index & RT_TRIANGLE_MEI_MASK];
    uint mat1 = edge & 0xFFFF;
    return get_material_index(material_indices, mat1);
}

// Rotate UVs based on overlay orientation (matches DX12 GetRotatedUVs)
float2 rotate_overlay_uv(float2 uv, uint orient) {
    switch (orient) {
        case 1:  return float2(1.0 - uv.y, uv.x);
        case 2:  return float2(1.0 - uv.x, 1.0 - uv.y);
        case 3:  return float2(uv.y, 1.0 - uv.x);
        default: return uv;
    }
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

        // Loop to handle overlay transparency (grate see-through)
        for (int bounce = 0; bounce < 8; bounce++) {
            intersector<triangle_data, instancing> i;
            i.accept_any_intersection(false);
            auto result = i.intersect(r, accel_struct);

            if (result.type != intersection_type::triangle) break;

            uint inst_id = result.instance_id;
            uint prim_id = result.primitive_id;
            float2 bary = result.triangle_barycentric_coord;
            float u = bary.x;
            float v = bary.y;
            float w = 1.0 - u - v;

            constant Instance& inst = instances[inst_id];
            uint global_tri_idx = inst.triangle_offset + prim_id;
            constant Triangle& tri = triangles[global_tri_idx];

            float2 uv = float2(tri.uv0) * w + float2(tri.uv1) * u + float2(tri.uv2) * v;

            // Check overlay (mat2) for transparency — only on WALL_CLOSED sides (grates)
            bool skip_hit = false;
            uint mei_raw = tri.material_edge_index;
            bool is_cutout_side = (mei_raw & RT_TRIANGLE_ALPHA_CUTOUT) != 0;
            if (is_cutout_side &&
                !(mei_raw & (RT_TRIANGLE_HOLDS_MATERIAL_INDEX | RT_TRIANGLE_HOLDS_MATERIAL_EDGE))) {
                uint mei = mei_raw & RT_TRIANGLE_MEI_MASK;
                uint edge = material_edges[mei];
                uint mat2_tmap = (edge >> 16) & 0x3FFF;
                if (mat2_tmap != 0) {
                    // Has overlay (mat2): check overlay alpha (grates, overlay-animated doors)
                    uint mat2_idx = get_material_index(material_indices, mat2_tmap);
                    mat2_idx = min(mat2_idx, (uint)(RT_MAX_TEXTURES - 1));
                    constant Material& mat2 = materials[mat2_idx];
                    uint tex2 = mat2.albedo_index;
                    if (tex2 > 0 && tex2 < RT_MAX_TEXTURES) {
                        uint slot2 = texture_remap[tex2];
                        if (slot2 > 0 && slot2 < MAX_BOUND_TEXTURES) {
                            uint orient = (edge >> 30) & 3;
                            float2 uv_rot = rotate_overlay_uv(uv, orient);
                            float overlay_alpha = textures[slot2].sample(tex_sampler, uv_rot).a;
                            // alpha=0.0: grate hole (flood fill couldn't reach from edge)
                            // alpha~0.25: border area (flood fill marked from edge)
                            // alpha=1.0: opaque bar
                            if (overlay_alpha < 0.1) {
                                skip_hit = true; // Grate hole — see through
                            }
                        }
                    }
                } else {
                    // No overlay: check base texture alpha (tmap1-animated doors)
                    uint mat1 = edge & 0xFFFF;
                    uint mat1_idx = get_material_index(material_indices, mat1);
                    mat1_idx = min(mat1_idx, (uint)(RT_MAX_TEXTURES - 1));
                    constant Material& mat1_mat = materials[mat1_idx];
                    uint tex1 = mat1_mat.albedo_index;
                    if (tex1 > 0 && tex1 < RT_MAX_TEXTURES) {
                        uint slot1 = texture_remap[tex1];
                        if (slot1 > 0 && slot1 < MAX_BOUND_TEXTURES) {
                            float base_alpha = textures[slot1].sample(tex_sampler, uv).a;
                            if (base_alpha < 0.5) {
                                skip_hit = true;
                            }
                        }
                    }
                }
            }

            if (skip_hit && scene.debug_mode != 4) {
                r.origin = r.origin + r.direction * (result.distance + 0.002);
                continue;
            }

            // Solid hit — record it
            closest_t = result.distance + length(r.origin - ro);

            float3 n0 = float3(tri.normal0);
            float3 n1 = float3(tri.normal1);
            float3 n2 = float3(tri.normal2);
            float3 obj_normal = normalize(n0 * w + n1 * u + n2 * v);
            hit_normal = normalize((inst.world_to_object * float4(obj_normal, 0.0)).xyz);

            hit_uv = uv;
            hit_material_index = tri.material_edge_index;
            hit_instance_material_override = inst.material_override;

            uint c = tri.color ? tri.color : inst.color;
            hit_color = float4(float(c & 0xFF) / 255.0, float((c >> 8) & 0xFF) / 255.0,
                               float((c >> 16) & 0xFF) / 255.0, 1.0);
            break;
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

        // Check for overlay texture (mat2) for rendering
        uint overlay_mat_idx = 0;
        float2 overlay_uv = hit_uv;
        if (!(hit_instance_material_override) &&
            !(hit_material_index & (RT_TRIANGLE_HOLDS_MATERIAL_INDEX | RT_TRIANGLE_HOLDS_MATERIAL_EDGE))) {
            uint edge = material_edges[hit_material_index & RT_TRIANGLE_MEI_MASK];
            uint mat2_tmap = (edge >> 16) & 0x3FFF;
            if (mat2_tmap != 0) {
                uint orient = (edge >> 30) & 3;
                overlay_mat_idx = get_material_index(material_indices, mat2_tmap);
                overlay_mat_idx = min(overlay_mat_idx, (uint)(RT_MAX_TEXTURES - 1));
                overlay_uv = rotate_overlay_uv(hit_uv, orient);
            }
        }

        if (scene.debug_mode == 1) {
            albedo = float4(fract(hit_uv.x), fract(hit_uv.y), 0.5, 1.0);
        }
        else if (scene.debug_mode == 2) {
            float hue = fract(float(mat_idx) * 0.1);
            albedo = float4(hue, 1.0 - hue, fract(hue * 3.0), 1.0);
        }
        else if (scene.debug_mode == 3) {
            // Overlay debug visualization:
            // DARK RED = special tri (holds_mat_idx/holds_mat_edge)
            // RED = no overlay, MAGENTA = overlay but no cutout flag (screens etc)
            // GREEN = cutout side with overlay alpha (bright=opaque bar, dark=hole)
            // BLUE = overlay not in remap, YELLOW = overlay no albedo
            uint dbg_mei_raw = hit_material_index;
            bool dbg_cutout = (dbg_mei_raw & RT_TRIANGLE_ALPHA_CUTOUT) != 0;
            if (hit_instance_material_override != 0 ||
                (dbg_mei_raw & (RT_TRIANGLE_HOLDS_MATERIAL_INDEX | RT_TRIANGLE_HOLDS_MATERIAL_EDGE))) {
                albedo = float4(0.5, 0.0, 0.0, 1.0); // Dark red = special triangle type
            } else {
                uint dbg_mei = dbg_mei_raw & RT_TRIANGLE_MEI_MASK;
                uint dbg_edge = material_edges[dbg_mei];
                uint dbg_mat2_tmap = (dbg_edge >> 16) & 0x3FFF;
                if (dbg_mat2_tmap == 0) {
                    albedo = float4(0.3, 0.0, 0.0, 1.0); // Dark red = no overlay
                } else if (!dbg_cutout) {
                    albedo = float4(0.8, 0.0, 0.8, 1.0); // Magenta = overlay but NOT cutout side
                } else {
                    uint dbg_mat2_idx = get_material_index(material_indices, dbg_mat2_tmap);
                    constant Material& dbg_mat2 = materials[min(dbg_mat2_idx, (uint)(RT_MAX_TEXTURES - 1))];
                    uint dbg_tex2 = dbg_mat2.albedo_index;
                    if (dbg_tex2 == 0 || dbg_tex2 >= RT_MAX_TEXTURES) {
                        albedo = float4(1.0, 1.0, 0.0, 1.0); // Yellow = overlay but no albedo texture
                    } else {
                        uint dbg_slot2 = texture_remap[dbg_tex2];
                        if (dbg_slot2 == 0 || dbg_slot2 >= MAX_BOUND_TEXTURES) {
                            albedo = float4(0.0, 0.0, float(dbg_tex2 % 256) / 255.0, 1.0); // Blue = not in remap
                        } else {
                            uint dbg_orient = (dbg_edge >> 30) & 3;
                            float2 dbg_uv_rot = rotate_overlay_uv(hit_uv, dbg_orient);
                            float4 dbg_sample = textures[dbg_slot2].sample(tex_sampler, dbg_uv_rot);
                            albedo = float4(0.0, dbg_sample.a, dbg_sample.a * 0.5, 1.0); // Green=alpha
                        }
                    }
                }
            }
        }
        else if (scene.debug_mode == 4) {
            // Show overlay alpha for cutout sides, normal base texture for others
            bool dbg4_cutout = (hit_material_index & RT_TRIANGLE_ALPHA_CUTOUT) != 0;
            if (dbg4_cutout && !(hit_material_index & (RT_TRIANGLE_HOLDS_MATERIAL_INDEX | RT_TRIANGLE_HOLDS_MATERIAL_EDGE))) {
                uint dbg4_mei = hit_material_index & RT_TRIANGLE_MEI_MASK;
                uint dbg4_edge = material_edges[dbg4_mei];
                uint dbg4_mat2_tmap = (dbg4_edge >> 16) & 0x3FFF;
                if (dbg4_mat2_tmap != 0) {
                    uint dbg4_mat2_idx = get_material_index(material_indices, dbg4_mat2_tmap);
                    constant Material& dbg4_mat2 = materials[min(dbg4_mat2_idx, (uint)(RT_MAX_TEXTURES - 1))];
                    uint dbg4_tex2 = dbg4_mat2.albedo_index;
                    uint dbg4_slot2 = (dbg4_tex2 > 0 && dbg4_tex2 < RT_MAX_TEXTURES) ? texture_remap[dbg4_tex2] : 0;
                    if (dbg4_slot2 > 0 && dbg4_slot2 < MAX_BOUND_TEXTURES) {
                        uint dbg4_orient = (dbg4_edge >> 30) & 3;
                        float2 dbg4_uv = rotate_overlay_uv(hit_uv, dbg4_orient);
                        float4 dbg4_sample = textures[dbg4_slot2].sample(tex_sampler, dbg4_uv);
                        // RED channel = overlay alpha, GREEN = base texture present
                        uint dbg4_base_slot = (tex_idx > 0 && tex_idx < RT_MAX_TEXTURES) ? texture_remap[tex_idx] : 0;
                        float base_ok = (dbg4_base_slot > 0) ? 1.0 : 0.0;
                        albedo = float4(dbg4_sample.a, base_ok * 0.3, 0.0, 1.0);
                    } else {
                        albedo = float4(0.0, 0.0, 1.0, 1.0); // Blue = overlay not in remap
                    }
                } else {
                    // Cutout side but no overlay — show in cyan
                    albedo = float4(0.0, 1.0, 1.0, 1.0);
                }
            } else {
                // Non-cutout: render normally
                if (tex_idx > 0 && tex_idx < RT_MAX_TEXTURES) {
                    uint slot = texture_remap[tex_idx];
                    if (slot > 0 && slot < MAX_BOUND_TEXTURES) {
                        albedo = textures[slot].sample(tex_sampler, hit_uv);
                        albedo.rgb *= hit_color.rgb;
                    }
                }
                if (overlay_mat_idx != 0) {
                    constant Material& mat2 = materials[overlay_mat_idx];
                    uint tex2_idx = mat2.albedo_index;
                    if (tex2_idx > 0 && tex2_idx < RT_MAX_TEXTURES) {
                        uint slot2 = texture_remap[tex2_idx];
                        if (slot2 > 0 && slot2 < MAX_BOUND_TEXTURES) {
                            float4 overlay_color = textures[slot2].sample(tex_sampler, overlay_uv);
                            if (overlay_color.a >= 0.5) {
                                albedo.rgb = overlay_color.rgb * hit_color.rgb;
                            }
                        }
                    }
                }
            }
        }
        else {
            // Sample base texture first
            if (tex_idx > 0 && tex_idx < RT_MAX_TEXTURES) {
                uint slot = texture_remap[tex_idx];
                if (slot > 0 && slot < MAX_BOUND_TEXTURES) {
                    albedo = textures[slot].sample(tex_sampler, hit_uv);
                    albedo.rgb *= hit_color.rgb;
                }
            }
            // If overlay exists and pixel is opaque, render overlay on top
            if (overlay_mat_idx != 0) {
                constant Material& mat2 = materials[overlay_mat_idx];
                uint tex2_idx = mat2.albedo_index;
                if (tex2_idx > 0 && tex2_idx < RT_MAX_TEXTURES) {
                    uint slot2 = texture_remap[tex2_idx];
                    if (slot2 > 0 && slot2 < MAX_BOUND_TEXTURES) {
                        float4 overlay_color = textures[slot2].sample(tex_sampler, overlay_uv);
                        if (overlay_color.a >= 0.5) {
                            albedo.rgb = overlay_color.rgb * hit_color.rgb;
                        }
                    }
                }
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

        // Emissive contribution (self-illuminating surfaces glow regardless of lighting)
        if (mat.emissive_factor > 0) {
            float strength = float(mat.emissive_factor) / 255.0;
            hdr += albedo.rgb * strength;
        }

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
