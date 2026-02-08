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
    uint   shadow_mode;
    uint   frame_number;
    uint   debug_mode;
    uint   texture_count;
    uint   use_accel;
    uint   light_count;
    float  billboard_opacity_threshold;
    float  billboard_emissive_boost;
    uint   _pad4[2];
};

struct Instance {
    float4x4 object_to_world;
    float4x4 world_to_object;
    uint   triangle_buffer_idx;
    uint   triangle_count;
    uint   color;
    uint   material_override;
    uint   triangle_offset;
    uint   object_type;
    uint   _pad[2];
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
constant uint RT_TRIANGLE_HOLDS_MATERIAL_INDEX = (1u << 30);
constant uint RT_TRIANGLE_HOLDS_MATERIAL_EDGE  = (1u << 31);
constant uint RT_TRIANGLE_MEI_MASK     = 0x1FFFFFFFu;
constant uint RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE = 0xFFFFu;
constant uint RT_MAT2_DOOR_SHIFT       = 10;
constant uint RT_MAT2_DOOR_MASK        = 0x3C00u;  // bits 10-13
constant uint RT_MAT2_ORIENT_SHIFT     = 14;
constant uint RT_MAT2_ORIENT_MASK      = 0xC000u;  // bits 14-15

struct GPUMaterial {
    uint albedo_index;
    uint flags;
    uint emissive_factor;
    uint _pad;
};

// Bindless texture array via Metal Argument Buffer (Tier 2)
constant uint RT_MAX_TEXTURES = 6030;

struct TextureArray {
    array<texture2d<float>, 6030> textures [[id(0)]];
};

// Sample a texture from the bindless array by index
float4 sample_bindless(uint tex_idx, float2 uv,
                       constant TextureArray& tex_array,
                       sampler tex_sampler) {
    if (tex_idx == 0 || tex_idx >= RT_MAX_TEXTURES) return float4(0);
    return tex_array.textures[tex_idx].sample(tex_sampler, uv);
}

// Look up a u16 material index from the packed u16 array
uint get_material_index(uint tmap_num, constant ushort* material_indices) {
    return uint(material_indices[tmap_num]);
}

// Resolve material index for both polymodel and level geometry triangles.
// Polymodels set RT_TRIANGLE_HOLDS_MATERIAL_EDGE (bit 31) or
// RT_TRIANGLE_HOLDS_MATERIAL_INDEX (bit 30) in material_edge_index.
// Level geometry uses material_edges[] indirection.
uint resolve_material_index(uint material_edge_index,
                            constant uint* material_edges,
                            constant ushort* material_indices) {
    // Polymodel: direct material index via material_indices lookup
    if (material_edge_index & RT_TRIANGLE_HOLDS_MATERIAL_EDGE) {
        uint edge_idx = material_edge_index & ~RT_TRIANGLE_HOLDS_MATERIAL_EDGE;
        return get_material_index(edge_idx, material_indices);
    }
    // Polymodel: direct material index (no indirection)
    if (material_edge_index & RT_TRIANGLE_HOLDS_MATERIAL_INDEX) {
        return material_edge_index & ~RT_TRIANGLE_HOLDS_MATERIAL_INDEX;
    }
    // Level geometry: material_edges → material_indices
    uint mei = material_edge_index & RT_TRIANGLE_MEI_MASK;
    uint edge = material_edges[mei];
    uint mat1 = edge & 0xFFFFu;
    return get_material_index(mat1, material_indices);
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
    return float3(r, g, b) * 250.0; // RT_LIGHT_SCALE
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
// Soft shadow helpers
// ============================================================================

// PCG hash for cheap pseudo-random numbers
uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand_float(uint seed) {
    return float(pcg_hash(seed)) / 4294967295.0;
}

// Light radius from transform scale (set by RT_MakeSphericalLight)
// Row-major 3x4: row 0 = [transform[0], transform[1], transform[2], tx]
float get_light_radius(constant Light& l) {
    return length(float3(l.transform[0], l.transform[1], l.transform[2]));
}

// Uniform sphere surface sampling from two random values in [0,1]
float3 sample_sphere_point(float u, float v) {
    float z = 1.0 - 2.0 * u;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * 3.14159265 * v;
    return float3(r * cos(phi), r * sin(phi), z);
}

// ============================================================================
// Shadow tracing helper (transparency-aware bounce loop)
// ============================================================================

struct ShadowResult {
    float visibility;
    float opaque_hit_dist;  // distance to first opaque blocker, -1 if none
};

ShadowResult trace_shadow_visibility(
    float3 shadow_origin,
    float3 shadow_dir,
    float max_dist,
    instance_acceleration_structure accel_struct,
    constant Triangle* triangles,
    constant Instance* instances,
    constant uint* material_edges,
    constant ushort* material_indices,
    constant GPUMaterial* gpu_materials,
    constant uint* texture_remap,
    constant uint& remap_table_size,
    array<texture2d<float, access::sample>, 32> alpha_textures,
    sampler tex_sampler)
{
    ShadowResult result;
    result.visibility = 1.0;
    result.opaque_hit_dist = -1.0;

    ray shadow_ray;
    shadow_ray.origin = shadow_origin;
    shadow_ray.direction = shadow_dir;
    shadow_ray.min_distance = 0.001;
    shadow_ray.max_distance = max_dist;

    float dist_traveled = 0.0;

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
            dist_traveled += shadow_hit.distance + 0.002;
            continue;
        }

        // Billboard/rod: transparent to shadow rays (gas/plasma doesn't cast shadows)
        if (s_tri.material_edge_index == RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE) {
            shadow_ray.origin = shadow_ray.origin + shadow_ray.direction * (shadow_hit.distance + 0.002);
            dist_traveled += shadow_hit.distance + 0.002;
            continue;
        }

        // Opaque blocker
        result.opaque_hit_dist = dist_traveled + shadow_hit.distance;
        result.visibility = 0.0;
        break;
    }

    return result;
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
// Tile-based light culling constants
// ============================================================================

constant uint TILE_SIZE = 16;
constant uint MAX_LIGHTS_PER_TILE = 64;
constant uint TILE_STRIDE = 1 + MAX_LIGHTS_PER_TILE;  // count + indices

// ============================================================================
// Tile light culling kernel (separate pass, one thread per tile)
// ============================================================================

kernel void tile_cull_lights(
    constant SceneConstants& scene [[buffer(0)]],
    constant Light* lights [[buffer(9)]],
    device uint* tile_data [[buffer(13)]],
    uint2 tile_id [[thread_position_in_grid]])
{
    uint tiles_x = (scene.render_width + TILE_SIZE - 1) / TILE_SIZE;
    uint tiles_y = (scene.render_height + TILE_SIZE - 1) / TILE_SIZE;
    if (tile_id.x >= tiles_x || tile_id.y >= tiles_y) return;

    uint tile_idx = tile_id.y * tiles_x + tile_id.x;
    uint base = tile_idx * TILE_STRIDE;

    float half_h = tan(scene.vfov_radians * 0.5);
    float3 cam_pos = float3(scene.camera_position);
    float3 cam_fwd = float3(scene.camera_forward);
    float3 cam_right = float3(scene.camera_right);
    float3 cam_up = float3(scene.camera_up);

    // Tile screen bounds in NDC [-1, 1]
    float tile_ndc_l = float(tile_id.x * TILE_SIZE) / float(scene.render_width) * 2.0 - 1.0;
    float tile_ndc_r = float(min((tile_id.x + 1) * TILE_SIZE, scene.render_width)) / float(scene.render_width) * 2.0 - 1.0;
    float tile_ndc_t = -(float(tile_id.y * TILE_SIZE) / float(scene.render_height) * 2.0 - 1.0);
    float tile_ndc_b = -(float(min((tile_id.y + 1) * TILE_SIZE, scene.render_height)) / float(scene.render_height) * 2.0 - 1.0);

    // Tile center direction (for rough spatial test)
    float ndc_cx = (tile_ndc_l + tile_ndc_r) * 0.5;
    float ndc_cy = (tile_ndc_t + tile_ndc_b) * 0.5;
    float3 tile_center_dir = normalize(cam_fwd + cam_right * (ndc_cx * half_h * scene.aspect_ratio) + cam_up * (ndc_cy * half_h));

    // Half-angle of tile frustum (conservative)
    float tile_half_angle = length(float2(tile_ndc_r - tile_ndc_l, tile_ndc_t - tile_ndc_b)) * 0.5 * half_h * max(scene.aspect_ratio, 1.0);

    uint count = 0;
    for (uint li = 0; li < scene.light_count && count < MAX_LIGHTS_PER_TILE; li++) {
        float3 light_pos = float3(
            lights[li].transform[3],
            lights[li].transform[7],
            lights[li].transform[11]);
        float3 light_emission = decode_rgbe(lights[li].emission);
        float emission_mag = max(max(light_emission.r, light_emission.g), light_emission.b);
        float max_dist = sqrt(emission_mag * 100.0);

        float3 to_light = light_pos - cam_pos;
        float depth = dot(to_light, cam_fwd);

        // Light behind camera: include if within max influence range
        if (depth < -max_dist) continue;

        // Distance cutoff: skip if too far regardless of tile
        float light_dist = length(to_light);
        if (light_dist > max_dist + 500.0) continue;  // generous buffer for tile depth range

        if (depth > 0.1) {
            // Project light to NDC and test overlap with tile
            float ndc_x = dot(to_light, cam_right) / (depth * half_h * scene.aspect_ratio);
            float ndc_y = dot(to_light, cam_up) / (depth * half_h);
            float ndc_radius = max_dist / (depth * half_h);

            // Circle-rect overlap: closest point on tile rect to light center
            float closest_x = clamp(ndc_x, tile_ndc_l, tile_ndc_r);
            float closest_y = clamp(ndc_y, tile_ndc_b, tile_ndc_t);
            float dx = ndc_x - closest_x;
            float dy = ndc_y - closest_y;
            if (dx * dx + dy * dy > ndc_radius * ndc_radius) continue;
        }
        // else: light near camera plane, include conservatively

        tile_data[base + 1 + count] = li;
        count++;
    }
    tile_data[base] = count;
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
    constant TextureArray& tex_array [[buffer(7)]],
    instance_acceleration_structure accel_struct [[buffer(8)]],
    constant Light* lights [[buffer(9)]],
    constant GPUMaterial* gpu_materials [[buffer(10)]],
    constant uint* texture_remap [[buffer(11)]],
    constant uint& remap_table_size [[buffer(12)]],
    constant uint* tile_light_data [[buffer(13)]],
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

        // Accumulated translucent billboard color (composited during bounce loop)
        float3 accum_color = float3(0.0);
        float accum_alpha = 0.0;

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
                r.origin = r.origin + r.direction * (intersection.distance + 0.002);
                continue;
            }

            // Billboard/rod: two-tier rendering (opaque subject + glow halo)
            if (tri.material_edge_index == RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE) {
                float4 bb_inst_color = decode_color(inst.color);
                float bb_alpha = 1.0;
                float3 bb_color = decode_color(tri.color).rgb * bb_inst_color.rgb;

                if (inst.material_override > 0u) {
                    uint albedo_idx = gpu_materials[inst.material_override].albedo_index;
                    bool sampled_tex = false;
                    // Try remap table first (alpha textures for cutout)
                    if (albedo_idx > 0u && albedo_idx < remap_table_size) {
                        uint slot = texture_remap[albedo_idx];
                        if (slot > 0u) {
                            float4 sampled = alpha_textures[slot].sample(tex_sampler, hit_uv);
                            bb_alpha = sampled.a;
                            bb_color = sampled.rgb * bb_inst_color.rgb;
                            sampled_tex = true;
                        }
                    }
                    // Fallback: bindless texture sampling (for rods/lasers not in remap)
                    if (!sampled_tex) {
                        float4 sampled = sample_bindless(albedo_idx, hit_uv, tex_array, tex_sampler);
                        if (sampled.a > 0.0) {
                            bb_alpha = sampled.a;
                            bb_color = sampled.rgb * bb_inst_color.rgb;
                        }
                    }
                }

                if (bb_alpha >= scene.billboard_opacity_threshold) {
                    // Opaque billboard/rod: self-lit with HDR boost + tonemapping
                    float3 boosted = bb_color * scene.billboard_emissive_boost;
                    float3 hdr = boosted + accum_color;
                    float3 tonemapped = ApplyTonemappingCurve(hdr);
                    final_color = float4(LinearTosRGB(tonemapped), 1.0);
                    break;
                } else if (bb_alpha >= 0.1) {
                    // Halo glow: translucent additive with emissive boost, continue ray
                    accum_color += bb_color * bb_alpha * scene.billboard_emissive_boost;
                    r.origin = r.origin + r.direction * (intersection.distance + 0.002);
                    continue;
                } else {
                    // Hole: skip entirely
                    r.origin = r.origin + r.direction * (intersection.distance + 0.002);
                    continue;
                }
            }

            // Solid hit (non-billboard geometry)
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

            // Barycentric interpolation for smooth normal (diffuse)
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

            // Interpolate UV at hit point for texture sampling
            float2 solid_uv = float2(tri.uv0) * w0 + float2(tri.uv1) * hit_bary.x + float2(tri.uv2) * hit_bary.y;

            // Sample texture via bindless texture array
            if (tri.material_edge_index != RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE) {
                uint mei_raw = tri.material_edge_index;
                bool is_polymodel = (mei_raw & (RT_TRIANGLE_HOLDS_MATERIAL_EDGE | RT_TRIANGLE_HOLDS_MATERIAL_INDEX)) != 0;

                // Resolve base material (works for both polymodels and level geometry)
                uint mat_slot = resolve_material_index(mei_raw, material_edges, material_indices);
                uint base_tex = gpu_materials[mat_slot].albedo_index;
                float4 base_sample = sample_bindless(base_tex, solid_uv, tex_array, tex_sampler);
                if (base_sample.a > 0.0) {
                    albedo = base_sample.rgb * inst_color.rgb;
                }

                // Overlay texture (level geometry only — polymodels don't have overlays)
                if (!is_polymodel) {
                    uint mei = mei_raw & RT_TRIANGLE_MEI_MASK;
                    uint edge = material_edges[mei];
                    uint mat2_raw = (edge >> 16) & 0xFFFFu;
                    uint mat2_tex = mat2_raw & 0x03FFu;
                    uint orientation = (mat2_raw & RT_MAT2_ORIENT_MASK) >> RT_MAT2_ORIENT_SHIFT;
                    if (mat2_tex > 0u) {
                        uint mat2_slot = get_material_index(mat2_tex, material_indices);
                        uint overlay_tex = gpu_materials[mat2_slot].albedo_index;
                        float2 overlay_uv = (orientation > 0u) ? rotate_uv(solid_uv, orientation) : solid_uv;
                        float4 overlay_sample = sample_bindless(overlay_tex, overlay_uv, tex_array, tex_sampler);
                        if (overlay_sample.a >= 0.5) {
                            albedo = overlay_sample.rgb * inst_color.rgb;
                        }
                    }
                }
            }

            // Detail noise overlay: subtle world-space variation to break up texture tiling
            {
                uint mei_raw = tri.material_edge_index;
                bool is_billboard = (mei_raw == RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE);
                bool is_polymodel = (mei_raw & (RT_TRIANGLE_HOLDS_MATERIAL_EDGE | RT_TRIANGLE_HOLDS_MATERIAL_INDEX)) != 0;
                if (!is_billboard && !is_polymodel) {
                    // Two octaves of value noise from world position
                    float2 hp = float2(dot(hit_pos, float3(127.1, 311.7, 74.7)),
                                       dot(hit_pos, float3(269.5, 183.3, 246.1)));
                    float n1 = fract(sin(dot(floor(hp * 0.3), float2(12.9898, 78.233))) * 43758.5453);
                    float n2 = fract(sin(dot(floor(hp * 1.2), float2(63.7264, 10.873))) * 43758.5453);
                    float noise = n1 * 0.6 + n2 * 0.4; // [0, 1]
                    // Modulate albedo brightness: range ~0.88 to 1.12
                    albedo *= 0.88 + noise * 0.24;
                }
            }

            // Hemisphere ambient: cool blue from above, warm brown from below
            float hemisphere_blend = normal_world.y * 0.5 + 0.5; // 0=down, 1=up
            float3 sky_color   = float3(0.06, 0.08, 0.12); // cool blue
            float3 ground_color = float3(0.05, 0.04, 0.03); // warm brown
            float3 total_light = mix(ground_color, sky_color, hemisphere_blend);
            float3 total_specular = float3(0.0);
            float3 V = -dir; // view direction (toward camera)

            if (scene.light_count > 0) {
                // Tile-based light loop: read culled light list for this pixel's tile
                uint tile_x = gid.x / TILE_SIZE;
                uint tile_y = gid.y / TILE_SIZE;
                uint tiles_per_row = (scene.render_width + TILE_SIZE - 1) / TILE_SIZE;
                uint tile_idx = tile_y * tiles_per_row + tile_x;
                uint tile_base = tile_idx * TILE_STRIDE;
                uint tile_count = min(tile_light_data[tile_base], MAX_LIGHTS_PER_TILE);

                for (uint tli = 0; tli < tile_count; tli++) {
                    uint li = tile_light_data[tile_base + 1 + tli];
                    // Extract position from transform column 3 (row-major: indices 3, 7, 11)
                    float3 light_pos = float3(
                        lights[li].transform[3],
                        lights[li].transform[7],
                        lights[li].transform[11]);
                    float3 light_emission = decode_rgbe(lights[li].emission);

                    float3 to_light = light_pos - hit_pos;
                    float dist_sq = dot(to_light, to_light);

                    // Distance cutoff: skip if max contribution is negligible
                    // emission / dist_sq < 0.01 → dist_sq > emission * 100
                    float emission_mag = max(max(light_emission.r, light_emission.g), light_emission.b);
                    if (dist_sq > emission_mag * 100.0) continue;

                    float dist = sqrt(dist_sq);
                    if (dist < 0.001) continue;
                    float3 L = to_light / dist;

                    float NdotL = max(0.0f, dot(normal_world, L));
                    if (NdotL <= 0.0) continue;

                    // Inverse-square attenuation with min radius clamp
                    float min_radius = 1.0;
                    float attenuation = 1.0 / max(dist_sq, min_radius * min_radius);

                    // Shadow visibility — branched on shadow_mode
                    float3 shadow_origin = hit_pos + normal_world * 0.002;
                    float visibility = 1.0;

                    if (scene.shadow_mode == 1) {
                        // Mode 1: Multi-sample soft shadows (8 jittered rays)
                        float light_radius = get_light_radius(lights[li]);
                        float vis_sum = 0.0;
                        for (uint si = 0; si < 8; si++) {
                            uint seed = (gid.y * scene.render_width + gid.x) * 997u
                                      + scene.frame_number * 17u + li * 31u + si * 7u;
                            float u = rand_float(seed);
                            float v = rand_float(seed + 1u);
                            float3 offset = sample_sphere_point(u, v) * light_radius;
                            float3 sample_pos = light_pos + offset;
                            float3 to_sample = sample_pos - hit_pos;
                            float sample_dist = length(to_sample);
                            if (sample_dist < 0.001) { vis_sum += 1.0; continue; }
                            float3 sample_L = to_sample / sample_dist;

                            ShadowResult sr = trace_shadow_visibility(
                                shadow_origin, sample_L, sample_dist - 0.002,
                                accel_struct, triangles, instances,
                                material_edges, material_indices, gpu_materials,
                                texture_remap, remap_table_size, alpha_textures, tex_sampler);
                            vis_sum += sr.visibility;
                        }
                        visibility = vis_sum / 8.0;
                    } else if (scene.shadow_mode == 2) {
                        // Mode 2: Analytic soft shadows (1 ray + PCSS-style penumbra)
                        ShadowResult sr = trace_shadow_visibility(
                            shadow_origin, L, dist - 0.002,
                            accel_struct, triangles, instances,
                            material_edges, material_indices, gpu_materials,
                            texture_remap, remap_table_size, alpha_textures, tex_sampler);

                        if (sr.opaque_hit_dist >= 0.0) {
                            // Opaque blocker found — estimate penumbra
                            float light_radius = get_light_radius(lights[li]);
                            float angular_size = light_radius / dist;
                            float t = sr.opaque_hit_dist / dist;
                            visibility = smoothstep(0.0, max(angular_size, 0.001), t * angular_size);
                        } else {
                            visibility = sr.visibility;
                        }
                    } else {
                        // Mode 0: Hard shadows (single ray, current behavior)
                        ShadowResult sr = trace_shadow_visibility(
                            shadow_origin, L, dist - 0.002,
                            accel_struct, triangles, instances,
                            material_edges, material_indices, gpu_materials,
                            texture_remap, remap_table_size, alpha_textures, tex_sampler);
                        visibility = sr.visibility;
                    }

                    total_light += light_emission * NdotL * attenuation * visibility;

                    // Phong specular: use interpolated normal for smooth blending across triangles
                    float3 R = reflect(-L, normal_world);
                    float RdotV = max(dot(R, V), 0.0);
                    float spec = pow(RdotV, 128.0);
                    total_specular += light_emission * spec * attenuation * visibility;
                }
            } else {
                // Fallback: simple directional light when no lights submitted
                float3 light_dir = normalize(float3(0.3, 1.0, 0.5));
                float NdotL = max(dot(normal_world, light_dir), 0.0);
                total_light = float3(NdotL * 0.85 + 0.15);
                float3 R = reflect(-light_dir, normal_world);
                float spec = pow(max(dot(R, V), 0.0), 128.0);
                total_specular = float3(spec * 0.5);
            }

            // Lambertian BRDF + Phong specular (shininess=128, ks=0.5)
            float3 hdr = (albedo / 2.0) * total_light + total_specular * 0.5;

            // Emissive contribution (self-illuminating surfaces glow regardless of lighting)
            {
                uint mei_raw = tri.material_edge_index;
                if (mei_raw != RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE) {
                    uint mat_slot = resolve_material_index(mei_raw, material_edges, material_indices);
                    uint ef = gpu_materials[mat_slot].emissive_factor;
                    if (ef > 0u) {
                        float strength = float(ef) / 255.0;
                        if (strength >= 0.9) {
                            hdr = albedo * strength;  // blackbody: albedo IS the light
                        } else {
                            hdr += albedo * strength; // partial emissive: additive
                        }
                    }
                }
            }

            // Weapon polymodels glow self-lit (OBJ_WEAPON == 5)
            if (inst.object_type == 5u) {
                hdr = albedo * scene.billboard_emissive_boost;
            }

            // Single-bounce specular reflection (nearby geometry bleeds into surfaces)
            if (scene.use_accel != 0) {
                float3 refl_dir = reflect(dir, normal_world);
                ray refl_ray;
                refl_ray.origin = hit_pos + normal_world * 0.01;
                refl_ray.direction = refl_dir;
                refl_ray.min_distance = 0.001;
                refl_ray.max_distance = 200.0;

                intersector<triangle_data, instancing> refl_i;
                refl_i.assume_geometry_type(geometry_type::triangle);
                refl_i.force_opacity(forced_opacity::opaque);
                auto refl_hit = refl_i.intersect(refl_ray, accel_struct, 0xFF);

                if (refl_hit.type != intersection_type::none) {
                    uint r_inst_id = refl_hit.instance_id;
                    uint r_prim_id = refl_hit.primitive_id;
                    constant Instance& r_inst = instances[r_inst_id];
                    uint r_tri_idx = r_inst.triangle_offset + r_prim_id;
                    constant Triangle& r_tri = triangles[r_tri_idx];

                    // Interpolate reflection hit normal + UV
                    float2 r_bary = refl_hit.triangle_barycentric_coord;
                    float r_w0 = 1.0 - r_bary.x - r_bary.y;
                    float3 r_normal_local = normalize(r_tri.norm0 * r_w0 + r_tri.norm1 * r_bary.x + r_tri.norm2 * r_bary.y);
                    float3x3 r_nmat = float3x3(r_inst.world_to_object[0].xyz, r_inst.world_to_object[1].xyz, r_inst.world_to_object[2].xyz);
                    float3 r_normal = normalize(transpose(r_nmat) * r_normal_local);
                    if (dot(r_normal, refl_dir) > 0.0) r_normal = -r_normal;

                    float2 r_uv = float2(r_tri.uv0) * r_w0 + float2(r_tri.uv1) * r_bary.x + float2(r_tri.uv2) * r_bary.y;
                    float3 r_hit_pos = refl_ray.origin + refl_dir * refl_hit.distance;

                    // Reflection albedo from texture
                    float4 r_inst_color = decode_color(r_inst.color);
                    float3 r_albedo = (decode_color(r_tri.color) * r_inst_color).rgb;
                    if (r_tri.material_edge_index != RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE) {
                        uint r_mat = resolve_material_index(r_tri.material_edge_index, material_edges, material_indices);
                        uint r_tex = gpu_materials[r_mat].albedo_index;
                        float4 r_samp = sample_bindless(r_tex, r_uv, tex_array, tex_sampler);
                        if (r_samp.a > 0.0) r_albedo = r_samp.rgb * r_inst_color.rgb;
                    }

                    // Basic diffuse shading at reflection hit (no shadows for perf)
                    float3 r_light = float3(0.05);
                    uint r_tile_x = gid.x / TILE_SIZE;
                    uint r_tile_y = gid.y / TILE_SIZE;
                    uint r_tile_idx = r_tile_y * ((scene.render_width + TILE_SIZE - 1) / TILE_SIZE) + r_tile_x;
                    uint r_tile_base = r_tile_idx * TILE_STRIDE;
                    uint r_tile_count = min(tile_light_data[r_tile_base], MAX_LIGHTS_PER_TILE);
                    for (uint rli = 0; rli < r_tile_count; rli++) {
                        uint li = tile_light_data[r_tile_base + 1 + rli];
                        float3 lp = float3(lights[li].transform[3], lights[li].transform[7], lights[li].transform[11]);
                        float3 to_l = lp - r_hit_pos;
                        float d2 = dot(to_l, to_l);
                        float3 le = decode_rgbe(lights[li].emission);
                        float em = max(max(le.r, le.g), le.b);
                        if (d2 > em * 100.0) continue;
                        float d = sqrt(d2);
                        if (d < 0.001) continue;
                        float ndl = max(dot(r_normal, to_l / d), 0.0);
                        r_light += le * ndl / max(d2, 1.0);
                    }

                    float3 r_color = (r_albedo / 3.14159) * r_light;

                    // Schlick Fresnel: more reflection at grazing angles
                    float cos_theta = max(dot(-dir, normal_world), 0.0);
                    float f0 = 0.04; // dielectric base reflectivity
                    float fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);

                    hdr = mix(hdr, r_color, fresnel);
                }
            }

            hdr *= exp2(0.1); // exposure
            float3 shaded = ApplyTonemappingCurve(hdr);

            // Additive billboard glow on top of shaded background
            float3 blended = shaded + accum_color;
            final_color = float4(LinearTosRGB(blended), 1.0);
        } else if (accum_color.r + accum_color.g + accum_color.b > 0.0) {
            // Only billboard glow, no solid background
            final_color = float4(LinearTosRGB(accum_color), 1.0);
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

// ============================================================================
// Bloom post-processing kernels
// ============================================================================

// Threshold + downsample: full-res raytrace output → half-res bloom texture
kernel void bloom_threshold(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint out_w = output.get_width();
    uint out_h = output.get_height();
    if (gid.x >= out_w || gid.y >= out_h) return;

    // 2x2 box downsample from full-res input
    uint2 src = gid * 2;
    float4 s0 = input.read(src + uint2(0, 0));
    float4 s1 = input.read(src + uint2(1, 0));
    float4 s2 = input.read(src + uint2(0, 1));
    float4 s3 = input.read(src + uint2(1, 1));
    float3 avg = (s0.rgb + s1.rgb + s2.rgb + s3.rgb) * 0.25;

    // Luminance-based soft threshold
    float lum = dot(avg, float3(0.2126, 0.7152, 0.0722));
    float knee = smoothstep(0.6, 1.0, lum);
    float3 bloom = avg * knee;

    output.write(float4(bloom, 1.0), gid);
}

// Horizontal 9-tap Gaussian blur
kernel void bloom_blur_h(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = output.get_width();
    uint h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;

    const float weights[5] = { 0.227027, 0.194946, 0.121621, 0.054054, 0.016216 };

    float3 result = input.read(gid).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
        uint lx = uint(max(int(gid.x) - i, 0));
        uint rx = min(gid.x + uint(i), w - 1);
        result += input.read(uint2(lx, gid.y)).rgb * weights[i];
        result += input.read(uint2(rx, gid.y)).rgb * weights[i];
    }
    output.write(float4(result, 1.0), gid);
}

// Vertical 9-tap Gaussian blur
kernel void bloom_blur_v(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = output.get_width();
    uint h = output.get_height();
    if (gid.x >= w || gid.y >= h) return;

    const float weights[5] = { 0.227027, 0.194946, 0.121621, 0.054054, 0.016216 };

    float3 result = input.read(gid).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
        uint ly = uint(max(int(gid.y) - i, 0));
        uint ry = min(gid.y + uint(i), h - 1);
        result += input.read(uint2(gid.x, ly)).rgb * weights[i];
        result += input.read(uint2(gid.x, ry)).rgb * weights[i];
    }
    output.write(float4(result, 1.0), gid);
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
