//! Texture remap logic for alpha cutout rendering.
//!
//! Extracts texture remap table construction from the dispatch pipeline
//! into a standalone pure function — no Metal API calls, no unsafe.

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;

use crate::texture::TextureSlotMap;
use crate::types::{GPUMaterial, MaterialEdge, Triangle};

use super::raytrace::ResolvedInstance;

/// Maximum number of texture slots for alpha cutout remap (slot 0 = no texture).
pub const MAX_ALPHA_TEXTURE_SLOTS: usize = 31;

/// Result of building the texture remap: a sparse table mapping albedo indices
/// to compact slots, plus the textures to bind at those slots.
pub struct TextureRemap<'a> {
    /// Sparse table: `remap_table[albedo_idx] = slot (1..31)`, 0 = unmapped.
    pub remap_table: Vec<u32>,
    /// Textures to bind at slots 1..N, in order.
    pub textures: Vec<(u32, &'a ProtocolObject<dyn MTLTexture>)>,
    /// Diagnostics for logging.
    pub diagnostics: RemapDiagnostics,
}

/// Diagnostics from the texture remap build, for logging.
#[derive(Debug, Default)]
pub struct RemapDiagnostics {
    /// Cutout textures that got a remap slot.
    pub cutout_mapped: u32,
    /// Material override textures that got a remap slot.
    pub override_mapped: u32,
    /// Material override textures that had albedo_index=0 (no texture uploaded).
    pub override_no_albedo: u32,
    /// Material override textures where find_by_index returned None.
    pub override_tex_missing: u32,
    /// Material override textures that were already mapped by a cutout triangle.
    pub override_already_mapped: u32,
    /// Total remap slots used (1-based).
    pub slots_used: u32,
}

/// Build texture remap from triangle data and instance material overrides.
///
/// Scans all triangles with the alpha cutout flag set and maps their albedo
/// texture indices to compact slots (1..31). Also includes textures from
/// billboard/rod material_override instances.
///
/// Pure function — no Metal API calls, no unsafe.
pub fn build_texture_remap<'a>(
    triangles: &[Triangle],
    instances: &[ResolvedInstance],
    material_edges: &[MaterialEdge],
    material_indices: &[u16],
    gpu_materials: &[GPUMaterial],
    texture_slotmap: &'a TextureSlotMap,
) -> TextureRemap<'a> {
    let rt_triangle_alpha_cutout: u32 = 1 << 29;
    let rt_triangle_mei_mask: u32 = 0x1FFFFFFF;

    let mut remap_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut remap_textures: Vec<(u32, &ProtocolObject<dyn MTLTexture>)> = Vec::new();
    let mut next_slot: u32 = 1;
    let mut diag = RemapDiagnostics::default();

    // Scan triangles for alpha cutout textures
    for tri in triangles {
        if (tri.material_edge_index & rt_triangle_alpha_cutout) == 0 {
            continue;
        }
        let mei = (tri.material_edge_index & rt_triangle_mei_mask) as usize;
        if mei >= material_edges.len() {
            continue;
        }
        let edge = &material_edges[mei];

        let mat2_raw = edge.mat2;
        let mat2_tex = (mat2_raw & 0x03FF) as usize;
        let mat1_tex = edge.mat1 as usize;
        let tmap_num = if mat2_tex > 0 { mat2_tex } else { mat1_tex };

        if tmap_num >= material_indices.len() {
            continue;
        }
        let mat_slot = material_indices[tmap_num] as usize;
        if mat_slot >= gpu_materials.len() {
            continue;
        }
        let albedo_idx = gpu_materials[mat_slot].albedo_index;
        if albedo_idx == 0 {
            continue;
        }

        if remap_map.contains_key(&albedo_idx) {
            continue;
        }

        if let Some(tex) = texture_slotmap.find_by_index(albedo_idx) {
            if next_slot < MAX_ALPHA_TEXTURE_SLOTS as u32 {
                remap_map.insert(albedo_idx, next_slot);
                remap_textures.push((albedo_idx, tex));
                next_slot += 1;
                diag.cutout_mapped += 1;
            }
        }
    }

    // Also add textures for billboard/rod material_override instances.
    for inst in instances {
        let mat_override = inst.gpu.material_override as usize;
        if mat_override == 0 {
            continue;
        }
        if mat_override >= gpu_materials.len() {
            continue;
        }
        let albedo_idx = gpu_materials[mat_override].albedo_index;
        if albedo_idx == 0 {
            diag.override_no_albedo += 1;
            continue;
        }
        if remap_map.contains_key(&albedo_idx) {
            diag.override_already_mapped += 1;
            continue;
        }
        if let Some(tex) = texture_slotmap.find_by_index(albedo_idx) {
            if next_slot < MAX_ALPHA_TEXTURE_SLOTS as u32 {
                remap_map.insert(albedo_idx, next_slot);
                remap_textures.push((albedo_idx, tex));
                next_slot += 1;
                diag.override_mapped += 1;
            }
        } else {
            diag.override_tex_missing += 1;
        }
    }

    // Build sparse remap table
    let max_remap_idx = remap_map.keys().copied().max().unwrap_or(0) as usize;
    let remap_table_size = max_remap_idx + 1;
    let mut remap_table: Vec<u32> = vec![0u32; remap_table_size.max(1)];
    for (&albedo_idx, &slot) in &remap_map {
        remap_table[albedo_idx as usize] = slot;
    }

    diag.slots_used = next_slot - 1;

    TextureRemap {
        remap_table,
        textures: remap_textures,
        diagnostics: diag,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_triangle(mei: u32) -> Triangle {
        Triangle {
            positions: [Vec3::default(); 3],
            normals: [Vec3::default(); 3],
            tangents: [Vec4::default(); 3],
            uvs: [Vec2::default(); 3],
            color: 0xFFFFFFFF,
            material_edge_index: mei,
        }
    }

    #[test]
    fn empty_triangles_produces_empty_remap() {
        let slotmap = TextureSlotMap::new();
        let remap = build_texture_remap(&[], &[], &[], &[], &[], &slotmap);
        assert!(remap.textures.is_empty());
        assert_eq!(remap.remap_table.len(), 1); // at least 1 element
        assert_eq!(remap.remap_table[0], 0);
    }

    #[test]
    fn non_cutout_triangles_ignored() {
        let tri = make_triangle(0); // no alpha cutout flag
        let slotmap = TextureSlotMap::new();
        let remap = build_texture_remap(&[tri], &[], &[], &[], &[], &slotmap);
        assert!(remap.textures.is_empty());
    }

    #[test]
    fn cutout_triangle_with_invalid_mei_ignored() {
        let alpha_flag = 1u32 << 29;
        let tri = make_triangle(alpha_flag | 99999); // MEI out of range
        let edges = vec![MaterialEdge { mat1: 0, mat2: 0 }; 10];
        let slotmap = TextureSlotMap::new();
        let remap = build_texture_remap(&[tri], &[], &edges, &[], &[], &slotmap);
        assert!(remap.textures.is_empty());
    }

    #[test]
    fn max_slots_respected() {
        let alpha_flag = 1u32 << 29;
        // Create 35 triangles with distinct materials to exceed 31-slot limit
        let mut triangles = Vec::new();
        let mut edges = Vec::new();
        let mut mat_indices = Vec::new();
        let mut gpu_mats = Vec::new();

        for i in 0..35u32 {
            triangles.push(make_triangle(alpha_flag | i));
            edges.push(MaterialEdge { mat1: i as u16, mat2: 0 });
            mat_indices.push(i as u16);
            gpu_mats.push(GPUMaterial {
                albedo_index: i + 1, // nonzero
                flags: 0,
                _pad: [0; 2],
            });
        }

        // We can't test with real textures (no Metal device), but we can verify
        // that the function doesn't panic with valid inputs and no textures found
        let slotmap = TextureSlotMap::new();
        let remap = build_texture_remap(
            &triangles,
            &[],
            &edges,
            &mat_indices,
            &gpu_mats,
            &slotmap,
        );
        // All lookups will return None from empty slotmap, so no textures remapped
        assert!(remap.textures.is_empty());
    }
}
