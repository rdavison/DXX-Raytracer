//! Metal compute shader source for raytracing.
//!
//! Inline MSL source compiled at runtime. Casts primary rays,
//! intersects via hardware TLAS or brute-force fallback,
//! shades with point lights + shadow rays (or directional fallback).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLLibrary};

pub const RAYTRACE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// ============================================================================
// GPU structs — must match Rust #[repr(C)] layouts exactly
// ============================================================================

struct SceneConstants {
    packed_float3 camera_position;  float _pad0;
    packed_float3 camera_forward;   float _pad1;
    packed_float3 camera_right;     float _pad2;
    packed_float3 camera_up;        float _pad3;
    float  vfov_radians;
    float  aspect_ratio;
    uint   render_width;
    uint   render_height;
    uint   instance_count;
    uint   total_triangles;
    float  _pad4[2];
    uint   debug_mode;
    uint   texture_count;
    uint   use_accel;
    uint   light_count;
};

struct Instance {
    float4x4 object_to_world;
    float4x4 world_to_object;
    uint   triangle_buffer_idx;
    uint   triangle_count;
    uint   color;
    uint   material_override;
    uint   triangle_offset;
    uint   _pad[3];
};

// Triangle layout: 152 bytes matching Rust Triangle struct
struct Triangle {
    packed_float3 pos0;       // 12 bytes
    packed_float3 pos1;       // 12 bytes
    packed_float3 pos2;       // 12 bytes
    packed_float3 norm0;      // 12 bytes
    packed_float3 norm1;      // 12 bytes
    packed_float3 norm2;      // 12 bytes
    float tangent0[4];        // 16 bytes
    float tangent1[4];        // 16 bytes
    float tangent2[4];        // 16 bytes
    packed_float2 uv0;        // 8 bytes
    packed_float2 uv1;        // 8 bytes
    packed_float2 uv2;        // 8 bytes
    uint   color;             // 4 bytes
    uint   material_edge_index; // 4 bytes
};

// Light struct: 56 bytes matching Rust Light layout
// kind(u8) + spot_angle(u8) + spot_softness(u8) + spot_vignette(u8) = 4 bytes packed as uint
// emission(u32) = 4 bytes RGBE packed
// transform = Mat34 = float[12] = 48 bytes (3 rows × 4 cols, row-major)
struct Light {
    uint   kind_packed;      // 4 bytes: kind(8) | spot_angle(8) | spot_softness(8) | vignette(8)
    uint   emission;         // 4 bytes: RGBE packed
    float  transform[12];    // 48 bytes: 3x4 row-major matrix
};

// ============================================================================
// Alpha cutout constants and material structs
// ============================================================================

constant uint RT_TRIANGLE_ALPHA_CUTOUT = (1u << 29);
constant uint RT_TRIANGLE_MEI_MASK     = 0x1FFFFFFFu;
constant uint RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE = 0xFFFFu;
constant uint RT_MAT2_DOOR_SHIFT       = 10;
constant uint RT_MAT2_DOOR_MASK        = 0x3C00u;  // bits 10-13
constant uint RT_MAT2_ORIENT_SHIFT     = 14;
constant uint RT_MAT2_ORIENT_MASK      = 0xC000u;  // bits 14-15

struct GPUMaterial {
    uint albedo_index;
    uint flags;
    uint _pad[2];
};

// Look up a u16 material index from the packed u16 array
uint get_material_index(uint tmap_num, constant ushort* material_indices) {
    return uint(material_indices[tmap_num]);
}

// Rotate UV by orientation (0=0°, 1=90°, 2=180°, 3=270°)
float2 rotate_uv(float2 uv, uint orient) {
    float2 c = float2(0.5, 0.5);
    float2 p = uv - c;
    switch (orient) {
        case 1: p = float2(-p.y, p.x); break;   // 90°
        case 2: p = float2(-p.x, -p.y); break;   // 180°
        case 3: p = float2(p.y, -p.x); break;    // 270°
        default: break;
    }
    return p + c;
}

// Check if a ray should skip this triangle (transparent cutout).
// Returns true if the ray should pass through (transparent pixel or opening door).
bool should_skip_cutout(
    uint mei_raw,
    float2 hit_uv,
    constant uint* material_edges,
    constant ushort* material_indices,
    constant GPUMaterial* gpu_materials,
    constant uint* texture_remap,
    constant uint& remap_table_size,
    array<texture2d<float, access::sample>, 32> alpha_textures,
    sampler tex_sampler)
{
    // Not an alpha cutout triangle
    if ((mei_raw & RT_TRIANGLE_ALPHA_CUTOUT) == 0) return false;

    uint mei = mei_raw & RT_TRIANGLE_MEI_MASK;
    uint edge = material_edges[mei];
    uint mat1_tex = edge & 0xFFFFu;
    uint mat2_raw = (edge >> 16) & 0xFFFFu;
    uint mat2_tex = mat2_raw & 0x03FFu;  // bits 0-9
    uint door_openness = (mat2_raw & RT_MAT2_DOOR_MASK) >> RT_MAT2_DOOR_SHIFT;
    uint orientation = (mat2_raw & RT_MAT2_ORIENT_MASK) >> RT_MAT2_ORIENT_SHIFT;

    // Opening/closing door — fully transparent
    if (door_openness > 0) return true;

    // Determine which texture to sample (overlay preferred over base)
    uint tmap_num = (mat2_tex > 0) ? mat2_tex : mat1_tex;
    if (tmap_num == 0) return false;

    // Resolve: tmap_num → material_slot → gpu_material → albedo_index
    uint mat_slot = get_material_index(tmap_num, material_indices);
    uint albedo_idx = gpu_materials[mat_slot].albedo_index;
    if (albedo_idx == 0) return false;

    // Look up texture remap slot
    if (albedo_idx >= remap_table_size) return false;
    uint slot = texture_remap[albedo_idx];
    if (slot == 0) return false;

    // Apply UV orientation for overlay textures
    float2 sample_uv = hit_uv;
    if (mat2_tex > 0 && orientation > 0) {
        sample_uv = rotate_uv(sample_uv, orientation);
    }

    // Sample texture alpha
    // Holes have alpha=0, border pixels have alpha~0.25 (flood-filled),
    // grate bars have alpha=1.0. Only treat actual holes as transparent.
    float alpha = alpha_textures[slot].sample(tex_sampler, sample_uv).a;
    return alpha < 0.1;
}

// Check if a ray should skip this triangle (billboard/rod with transparent pixel).
// Returns true if the ray should pass through.
bool should_skip_material_override(
    uint mei_raw,
    float2 hit_uv,
    uint material_override_val,
    constant GPUMaterial* gpu_materials,
    constant uint* texture_remap,
    constant uint& remap_table_size,
    array<texture2d<float, access::sample>, 32> alpha_textures,
    sampler tex_sampler)
{
    if (mei_raw != RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE) return false;
    if (material_override_val == 0u) return false;

    uint albedo_idx = gpu_materials[material_override_val].albedo_index;
    if (albedo_idx == 0u || albedo_idx >= remap_table_size) return false;

    uint slot = texture_remap[albedo_idx];
    if (slot == 0u) return false;

    float4 sampled = alpha_textures[slot].sample(tex_sampler, hit_uv);
    return sampled.a < 0.1;
}

// ============================================================================
// Helpers
// ============================================================================

float4 decode_color(uint packed) {
    float r = float(packed & 0xFF) / 255.0;
    float g = float((packed >> 8) & 0xFF) / 255.0;
    float b = float((packed >> 16) & 0xFF) / 255.0;
    float a = float((packed >> 24) & 0xFF) / 255.0;
    return float4(r, g, b, a);
}

// Decode RGBE emission: 9-bit R, 9-bit G, 9-bit B, 5-bit exponent (bias 20)
float3 decode_rgbe(uint rgbe) {
    int exponent = int(rgbe >> 27) - 20;
    float scale = exp2(float(exponent)) / 256.0;
    float r = float((rgbe >>  0) & 0x1FFu) * scale;
    float g = float((rgbe >>  9) & 0x1FFu) * scale;
    float b = float((rgbe >> 18) & 0x1FFu) * scale;
    return float3(r, g, b) * 1000.0; // RT_LIGHT_SCALE
}

// Tone mapping: Reinhard-style shoulder curve
float3 ApplyTonemappingCurve(float3 x) {
    return x / (1.0 + x);
}

// Linear to sRGB gamma
float3 LinearTosRGB(float3 c) {
    return pow(clamp(c, 0.0, 1.0), float3(1.0 / 2.2));
}

// ============================================================================
// Brute-force ray-triangle intersection (Moller-Trumbore)
// ============================================================================
struct HitResult {
    float t;
    float u;
    float v;
    int   instance_id;
    int   primitive_id;
};

bool ray_triangle_intersect(
    float3 origin, float3 dir,
    float3 v0, float3 v1, float3 v2,
    thread float &t, thread float &u, thread float &v)
{
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(dir, e2);
    float det = dot(e1, pvec);
    if (abs(det) < 1e-8) return false;
    float inv_det = 1.0 / det;
    float3 tvec = origin - v0;
    u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) return false;
    float3 qvec = cross(tvec, e1);
    v = dot(dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) return false;
    t = dot(e2, qvec) * inv_det;
    return t > 0.001;
}

HitResult brute_force_intersect(
    float3 origin, float3 dir,
    constant Instance* instances,
    constant Triangle* triangles,
    uint instance_count)
{
    HitResult best;
    best.t = INFINITY;
    best.instance_id = -1;
    best.primitive_id = -1;

    for (uint i = 0; i < instance_count; i++) {
        constant Instance& inst = instances[i];
        float4 o4 = inst.world_to_object * float4(origin, 1.0);
        float4 d4 = inst.world_to_object * float4(dir, 0.0);
        float3 local_origin = o4.xyz;
        float3 local_dir = d4.xyz;

        uint offset = inst.triangle_offset;
        for (uint j = 0; j < inst.triangle_count; j++) {
            constant Triangle& tri = triangles[offset + j];
            float t, u, v;
            if (ray_triangle_intersect(local_origin, local_dir,
                                       tri.pos0, tri.pos1, tri.pos2,
                                       t, u, v)) {
                if (t < best.t) {
                    best.t = t;
                    best.u = u;
                    best.v = v;
                    best.instance_id = i;
                    best.primitive_id = j;
                }
            }
        }
    }
    return best;
}

// ============================================================================
// Main raytracing kernel
// ============================================================================
kernel void raytrace_main(
    texture2d<float, access::write> output [[texture(0)]],
    constant SceneConstants& scene [[buffer(0)]],
    constant Instance* instances [[buffer(1)]],
    constant Triangle* triangles [[buffer(2)]],
    constant uint* material_edges [[buffer(5)]],
    constant ushort* material_indices [[buffer(6)]],
    constant GPUMaterial* gpu_materials [[buffer(7)]],
    instance_acceleration_structure accel_struct [[buffer(8)]],
    constant Light* lights [[buffer(9)]],
    constant uint* texture_remap [[buffer(11)]],
    constant uint& remap_table_size [[buffer(12)]],
    array<texture2d<float, access::sample>, 32> alpha_textures [[texture(16)]],
    sampler tex_sampler [[sampler(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= scene.render_width || gid.y >= scene.render_height)
        return;

    // --- Ray generation ---
    float2 pixel = float2(gid) + 0.5;
    float2 ndc = pixel / float2(scene.render_width, scene.render_height) * 2.0 - 1.0;
    ndc.y = -ndc.y; // flip Y

    float half_h = tan(scene.vfov_radians * 0.5);
    float3 forward = float3(scene.camera_forward);
    float3 right   = float3(scene.camera_right);
    float3 up      = float3(scene.camera_up);
    float3 dir = normalize(forward + right * (ndc.x * half_h * scene.aspect_ratio) + up * (ndc.y * half_h));
    float3 origin = float3(scene.camera_position);

    // --- Intersection ---
    float4 final_color = float4(0.0, 0.0, 0.0, 1.0);

    if (scene.use_accel != 0) {
        // Hardware-accelerated intersection via TLAS with door transparency bounce loop
        ray r;
        r.origin = origin;
        r.direction = dir;
        r.min_distance = 0.001;
        r.max_distance = INFINITY;

        int hit_inst_id = -1;
        int hit_prim_id = -1;
        float2 hit_bary;
        float hit_dist_total = 0.0;

        for (int bounce = 0; bounce < 8; bounce++) {
            intersector<triangle_data, instancing> isect;
            isect.assume_geometry_type(geometry_type::triangle);
            isect.force_opacity(forced_opacity::opaque);

            auto intersection = isect.intersect(r, accel_struct, 0xFF);

            if (intersection.type == intersection_type::none) break;

            uint inst_id = intersection.instance_id;
            uint prim_id = intersection.primitive_id;
            constant Instance& inst = instances[inst_id];
            uint tri_idx = inst.triangle_offset + prim_id;
            constant Triangle& tri = triangles[tri_idx];

            // Interpolate UV at hit point
            float2 bary = intersection.triangle_barycentric_coord;
            float bw0 = 1.0 - bary.x - bary.y;
            float2 hit_uv = float2(tri.uv0) * bw0 + float2(tri.uv1) * bary.x + float2(tri.uv2) * bary.y;

            if (should_skip_cutout(tri.material_edge_index, hit_uv,
                    material_edges, material_indices, gpu_materials,
                    texture_remap, remap_table_size, alpha_textures, tex_sampler)) {
                // Advance ray past this transparent surface
                r.origin = r.origin + r.direction * (intersection.distance + 0.002);
                continue;
            }

            // Skip transparent pixels on billboard/rod overrides
            if (should_skip_material_override(tri.material_edge_index, hit_uv,
                    inst.material_override, gpu_materials,
                    texture_remap, remap_table_size, alpha_textures, tex_sampler)) {
                r.origin = r.origin + r.direction * (intersection.distance + 0.002);
                continue;
            }

            // Solid hit
            hit_inst_id = inst_id;
            hit_prim_id = prim_id;
            hit_bary = intersection.triangle_barycentric_coord;
            hit_dist_total = intersection.distance + length(r.origin - origin);
            break;
        }

        if (hit_inst_id >= 0) {
            constant Instance& inst = instances[hit_inst_id];
            uint tri_idx = inst.triangle_offset + hit_prim_id;
            constant Triangle& tri = triangles[tri_idx];

            // Barycentric interpolation for normal
            float w0 = 1.0 - hit_bary.x - hit_bary.y;
            float3 normal_local = normalize(tri.norm0 * w0 + tri.norm1 * hit_bary.x + tri.norm2 * hit_bary.y);

            // Transform normal to world space
            float3x3 normal_mat = float3x3(
                inst.world_to_object[0].xyz,
                inst.world_to_object[1].xyz,
                inst.world_to_object[2].xyz
            );
            float3 normal_world = normalize(transpose(normal_mat) * normal_local);

            // Flip back-facing normals
            if (dot(normal_world, dir) > 0.0)
                normal_world = -normal_world;

            // Hit position in world space
            float3 hit_pos = origin + dir * hit_dist_total;

            // Decode vertex color as albedo
            float4 vert_color = decode_color(tri.color);
            float4 inst_color = decode_color(inst.color);
            float3 albedo = (vert_color * inst_color).rgb;

            // Billboard/rod overrides are emissive (self-lit, no lighting)
            if (tri.material_edge_index == RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE) {
                // Interpolate UV at hit point
                float2 uv = float2(tri.uv0) * w0 + float2(tri.uv1) * hit_bary.x + float2(tri.uv2) * hit_bary.y;

                float3 tex_color = albedo;  // fallback: vertex * instance color
                if (inst.material_override > 0u) {
                    // material_override is a bitmap index — use directly into gpu_materials[]
                    uint albedo_idx = gpu_materials[inst.material_override].albedo_index;
                    if (albedo_idx > 0u && albedo_idx < remap_table_size) {
                        uint slot = texture_remap[albedo_idx];
                        if (slot > 0u) {
                            float4 sampled = alpha_textures[slot].sample(tex_sampler, uv);
                            tex_color = sampled.rgb * inst_color.rgb;
                        }
                    }
                }
                float3 ldr = LinearTosRGB(tex_color);
                final_color = float4(ldr, 1.0);
            } else {
            float3 total_light = float3(0.0);

            if (scene.light_count > 0) {
                // Point light loop with shadow rays (transparency-aware)
                for (uint li = 0; li < scene.light_count; li++) {
                    // Extract position from transform column 3 (row-major: indices 3, 7, 11)
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

                    float NdotL = max(0.0f, dot(normal_world, L));
                    if (NdotL <= 0.0) continue;

                    // Inverse-square attenuation with min radius clamp
                    float min_radius = 1.0;
                    float attenuation = 1.0 / max(dist_sq, min_radius * min_radius);

                    // Shadow ray with door transparency bounce loop
                    float visibility = 1.0;
                    ray shadow_ray;
                    shadow_ray.origin = hit_pos + normal_world * 0.002;
                    shadow_ray.direction = L;
                    shadow_ray.min_distance = 0.001;
                    shadow_ray.max_distance = dist - 0.002;

                    for (int sbounce = 0; sbounce < 4; sbounce++) {
                        intersector<triangle_data, instancing> shadow_i;
                        shadow_i.accept_any_intersection(false);
                        shadow_i.force_opacity(forced_opacity::opaque);
                        auto shadow_hit = shadow_i.intersect(shadow_ray, accel_struct, 0xFF);

                        if (shadow_hit.type == intersection_type::none) break;

                        uint s_inst_id = shadow_hit.instance_id;
                        uint s_prim_id = shadow_hit.primitive_id;
                        constant Instance& s_inst = instances[s_inst_id];
                        uint s_tri_idx = s_inst.triangle_offset + s_prim_id;
                        constant Triangle& s_tri = triangles[s_tri_idx];

                        // Interpolate UV at shadow hit point
                        float2 s_bary = shadow_hit.triangle_barycentric_coord;
                        float s_w0 = 1.0 - s_bary.x - s_bary.y;
                        float2 s_hit_uv = float2(s_tri.uv0) * s_w0 + float2(s_tri.uv1) * s_bary.x + float2(s_tri.uv2) * s_bary.y;

                        if (should_skip_cutout(s_tri.material_edge_index, s_hit_uv,
                                material_edges, material_indices, gpu_materials,
                                texture_remap, remap_table_size, alpha_textures, tex_sampler)) {
                            shadow_ray.origin = shadow_ray.origin + shadow_ray.direction * (shadow_hit.distance + 0.002);
                            continue;
                        }

                        // Skip transparent pixels on billboard/rod overrides for shadow rays
                        if (should_skip_material_override(s_tri.material_edge_index, s_hit_uv,
                                s_inst.material_override, gpu_materials,
                                texture_remap, remap_table_size, alpha_textures, tex_sampler)) {
                            shadow_ray.origin = shadow_ray.origin + shadow_ray.direction * (shadow_hit.distance + 0.002);
                            continue;
                        }

                        visibility = 0.0;
                        break;
                    }

                    total_light += light_emission * NdotL * attenuation * visibility;
                }
            } else {
                // Fallback: simple directional light when no lights submitted
                float3 light_dir = normalize(float3(0.3, 1.0, 0.5));
                float NdotL = max(dot(normal_world, light_dir), 0.0);
                total_light = float3(NdotL * 0.85 + 0.15);
            }

            // Lambertian BRDF + tone mapping + gamma
            float3 hdr = (albedo / 3.14159265) * total_light;
            hdr *= exp2(0.1); // exposure
            float3 ldr = ApplyTonemappingCurve(hdr);
            ldr = LinearTosRGB(ldr);
            final_color = float4(ldr, 1.0);
            } // end else (non-emissive lighting)
        }
    } else {
        // Brute-force fallback (no shadow rays without TLAS)
        HitResult hit = brute_force_intersect(origin, dir, instances, triangles, scene.instance_count);

        if (hit.instance_id >= 0) {
            constant Instance& inst = instances[hit.instance_id];
            uint tri_idx = inst.triangle_offset + hit.primitive_id;
            constant Triangle& tri = triangles[tri_idx];

            float w0 = 1.0 - hit.u - hit.v;
            float3 normal_local = normalize(tri.norm0 * w0 + tri.norm1 * hit.u + tri.norm2 * hit.v);

            float3x3 normal_mat = float3x3(
                inst.world_to_object[0].xyz,
                inst.world_to_object[1].xyz,
                inst.world_to_object[2].xyz
            );
            float3 normal_world = normalize(transpose(normal_mat) * normal_local);

            // Flip back-facing normals
            if (dot(normal_world, dir) > 0.0)
                normal_world = -normal_world;

            float4 vert_color = decode_color(tri.color);
            float4 inst_color = decode_color(inst.color);
            float4 base_color = vert_color * inst_color;

            // Billboard/rod overrides are emissive (self-lit, no lighting)
            if (tri.material_edge_index == RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE) {
                float2 uv = float2(tri.uv0) * w0 + float2(tri.uv1) * hit.u + float2(tri.uv2) * hit.v;

                float3 tex_color = base_color.rgb;  // fallback: vertex * instance color
                if (inst.material_override > 0u) {
                    // material_override is a bitmap index — use directly into gpu_materials[]
                    uint albedo_idx = gpu_materials[inst.material_override].albedo_index;
                    if (albedo_idx > 0u && albedo_idx < remap_table_size) {
                        uint slot = texture_remap[albedo_idx];
                        if (slot > 0u) {
                            float4 sampled = alpha_textures[slot].sample(tex_sampler, uv);
                            tex_color = sampled.rgb * inst_color.rgb;
                        }
                    }
                }
                float3 ldr = LinearTosRGB(tex_color);
                final_color = float4(ldr, 1.0);
            } else {
                // Directional light fallback
                float3 light_dir = normalize(float3(0.3, 1.0, 0.5));
                float NdotL = max(dot(normal_world, light_dir), 0.0);
                float ambient = 0.15;

                final_color = float4(base_color.rgb * (ambient + 0.85 * NdotL), 1.0);
            }
        }
    }

    output.write(final_color, gid);
}
"#;

/// Compile the raytracing compute shader into a Metal library.
pub fn compile_raytrace_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLLibrary>> {
    let source = NSString::from_str(RAYTRACE_SHADER_SOURCE);

    // Use Metal compile options to enable Metal 3.0 for raytracing
    let options = objc2_metal::MTLCompileOptions::new();
    options.setLanguageVersion(objc2_metal::MTLLanguageVersion::Version3_0);

    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .expect("[Rust Metal] Failed to compile raytrace shader library");
    eprintln!("[Rust Metal] Raytrace shader library compiled successfully");
    library
}
