//! Per-frame raytracing orchestration (Phase 3A).
//!
//! Builds combined triangle buffer, TLAS, and dispatches compute shader.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::mesh::MeshSlotMap;
use crate::state::FrameBuffers;
use crate::texture::TextureSlotMap;
use crate::types::*;

// ============================================================================
// GPU-compatible structs matching shader layout
// ============================================================================

/// GPU instance data — 160 bytes, matches shader Instance struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RaytraceInstance {
    pub object_to_world: Mat4,      // 64 bytes
    pub world_to_object: Mat4,      // 64 bytes
    pub triangle_buffer_idx: u32,   // 4
    pub triangle_count: u32,        // 4
    pub color: u32,                 // 4
    pub material_override: u32,     // 4
    pub triangle_offset: u32,       // 4
    pub _pad: [u32; 3],             // 12
}

/// Scene constants — 112 bytes, matches shader SceneConstants struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RaytraceSceneConstants {
    pub camera_position: Vec3,
    pub _pad0: f32,
    pub camera_forward: Vec3,
    pub _pad1: f32,
    pub camera_right: Vec3,
    pub _pad2: f32,
    pub camera_up: Vec3,
    pub _pad3: f32,
    pub vfov_radians: f32,
    pub aspect_ratio: f32,
    pub render_width: u32,
    pub render_height: u32,
    pub instance_count: u32,
    pub total_triangles: u32,
    pub _pad4: [f32; 2],
    pub debug_mode: u32,
    pub texture_count: u32,
    pub use_accel: u32,
    pub light_count: u32,
}

/// Pending mesh instance queued for this frame.
pub struct PendingInstance {
    pub instance: RaytraceInstance,
    pub mesh_handle: ResourceHandle,
}

// ============================================================================
// Per-frame functions
// ============================================================================

/// Clear the instance list for a new frame.
pub fn begin_scene(instances: &mut Vec<PendingInstance>) {
    instances.clear();
}

/// Queue a mesh instance for raytracing this frame.
pub fn queue_mesh_instance(
    instances: &mut Vec<PendingInstance>,
    mesh_slotmap: &MeshSlotMap,
    mesh_handle: ResourceHandle,
    transform: &Mat4,
    color: u32,
    material_override: u16,
) {
    let mesh = match mesh_slotmap.find(mesh_handle) {
        Some(m) => m,
        None => return,
    };

    let world_to_object = transform.inverse();

    instances.push(PendingInstance {
        instance: RaytraceInstance {
            object_to_world: *transform,
            world_to_object,
            triangle_buffer_idx: mesh_handle.index,
            triangle_count: mesh.triangle_count as u32,
            color,
            material_override: material_override as u32,
            triangle_offset: 0, // set during build_combined_triangle_buffer
            _pad: [0; 3],
        },
        mesh_handle,
    });
}

/// Build a single combined triangle buffer from all queued instances.
/// Uses pre-allocated buffer from FrameBuffers, growing only when needed.
/// Updates triangle_offset for each instance. Returns total_triangle_count, or None if empty.
pub fn build_combined_triangle_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    mesh_slotmap: &MeshSlotMap,
    instances: &mut [PendingInstance],
    frame_buffers: &mut FrameBuffers,
) -> Option<u32> {
    let tri_size = std::mem::size_of::<Triangle>();
    let mut total_tris: usize = 0;
    for inst in instances.iter() {
        if let Some(mesh) = mesh_slotmap.find(inst.mesh_handle) {
            total_tris += mesh.triangle_count;
        }
    }

    if total_tris == 0 {
        return None;
    }

    // Ensure buffer is large enough
    if !crate::state::ensure_buffer_capacity(
        device,
        &mut frame_buffers.combined_tri_buf,
        &mut frame_buffers.combined_tri_capacity,
        total_tris,
        tri_size,
    ) {
        return None;
    }

    let combined = frame_buffers.combined_tri_buf.as_ref().unwrap();
    let dst_base = combined.contents().as_ptr() as *mut u8;
    let mut offset: usize = 0;

    for inst in instances.iter_mut() {
        if let Some(mesh) = mesh_slotmap.find(inst.mesh_handle) {
            let byte_count = mesh.triangle_count * tri_size;
            inst.instance.triangle_offset = (offset / tri_size) as u32;

            let src = mesh.triangle_buffer.contents().as_ptr() as *const u8;
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst_base.add(offset), byte_count);
            }
            offset += byte_count;
        }
    }

    Some(total_tris as u32)
}

/// Build or refit the top-level acceleration structure from queued instances.
/// When the set of instances (same mesh handles, same count) hasn't changed,
/// refits in-place instead of doing a full rebuild — much faster for transform-only updates.
pub fn build_tlas(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    mesh_slotmap: &MeshSlotMap,
    instances: &[PendingInstance],
    tlas_store: &mut [Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>; 2],
    scratch_store: &mut [Option<Retained<ProtocolObject<dyn MTLBuffer>>>; 2],
    buffer_index: &mut usize,
    prev_instance_keys: &mut Vec<u32>,
) -> Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> {
    if instances.is_empty() {
        prev_instance_keys.clear();
        return None;
    }

    // Collect unique BLAS references and build instance descriptors
    let mut blas_list: Vec<(u32, &ProtocolObject<dyn MTLAccelerationStructure>)> = Vec::new();
    let mut blas_index_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for inst in instances {
        let mesh = match mesh_slotmap.find(inst.mesh_handle) {
            Some(m) => m,
            None => continue,
        };
        let key = inst.mesh_handle.index;
        if !blas_index_map.contains_key(&key) {
            let idx = blas_list.len() as u32;
            blas_index_map.insert(key, idx);
            blas_list.push((key, &mesh.blas));
        }
    }

    if blas_list.is_empty() {
        prev_instance_keys.clear();
        return None;
    }

    // Determine if we can refit: same instance keys as last frame
    let current_keys: Vec<u32> = instances.iter().map(|i| i.mesh_handle.index).collect();
    let can_refit = current_keys.len() == prev_instance_keys.len()
        && current_keys.iter().zip(prev_instance_keys.iter()).all(|(a, b)| a == b)
        && tlas_store[*buffer_index].is_some();

    // Create instance descriptor buffer
    let inst_desc_size = std::mem::size_of::<MTLAccelerationStructureInstanceDescriptor>();
    let inst_desc_buf = device.newBufferWithLength_options(
        instances.len() * inst_desc_size,
        MTLResourceOptions::StorageModeShared,
    )?;

    // Fill instance descriptors
    let desc_ptr = inst_desc_buf.contents().as_ptr() as *mut MTLAccelerationStructureInstanceDescriptor;
    for (i, inst) in instances.iter().enumerate() {
        let blas_idx = match blas_index_map.get(&inst.mesh_handle.index) {
            Some(&idx) => idx,
            None => continue,
        };

        let m = &inst.instance.object_to_world.e;
        let transform = MTLPackedFloat4x3 {
            columns: [
                MTLPackedFloat3 { x: m[0][0], y: m[1][0], z: m[2][0] },
                MTLPackedFloat3 { x: m[0][1], y: m[1][1], z: m[2][1] },
                MTLPackedFloat3 { x: m[0][2], y: m[1][2], z: m[2][2] },
                MTLPackedFloat3 { x: m[0][3], y: m[1][3], z: m[2][3] },
            ],
        };

        let desc = MTLAccelerationStructureInstanceDescriptor {
            transformationMatrix: transform,
            options: MTLAccelerationStructureInstanceOptions::Opaque,
            mask: 0xFF,
            intersectionFunctionTableOffset: 0,
            accelerationStructureIndex: blas_idx,
        };

        unsafe {
            desc_ptr.add(i).write(desc);
        }
    }

    // Create NSArray of BLAS references
    let blas_array: Vec<&ProtocolObject<dyn MTLAccelerationStructure>> =
        blas_list.iter().map(|(_, blas)| *blas).collect();
    let blas_ns_array = objc2_foundation::NSArray::from_slice(&blas_array);

    // Create instance acceleration structure descriptor
    let tlas_desc = MTLInstanceAccelerationStructureDescriptor::descriptor();
    tlas_desc.setInstanceCount(instances.len());
    tlas_desc.setInstanceDescriptorBuffer(Some(&inst_desc_buf));
    unsafe {
        tlas_desc.setInstanceDescriptorBufferOffset(0);
    }
    tlas_desc.setInstancedAccelerationStructures(Some(&blas_ns_array));

    // Set Refit usage so the TLAS is built in a refit-compatible layout
    tlas_desc.setUsage(MTLAccelerationStructureUsage::Refit);

    if can_refit {
        // Refit in-place: update transforms without rebuilding
        let idx = *buffer_index;
        let tlas = tlas_store[idx].as_ref()?;

        // Refit needs scratch buffer sized for refit (usually smaller than build)
        let sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
        let scratch_size = sizes.refitScratchBufferSize.max(64);
        let need_scratch = match &scratch_store[idx] {
            Some(existing) => existing.length() < scratch_size,
            None => true,
        };
        if need_scratch {
            scratch_store[idx] = device.newBufferWithLength_options(
                scratch_size,
                MTLResourceOptions::StorageModePrivate,
            );
        }
        let scratch = scratch_store[idx].as_ref()?;

        let cmd_buf = command_queue.commandBuffer()?;
        let encoder = cmd_buf.accelerationStructureCommandEncoder()?;
        unsafe {
            encoder.refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset(
                tlas,
                &tlas_desc,
                None, // in-place refit
                Some(scratch),
                0,
            );
        }
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Keys unchanged, no need to update prev_instance_keys
        Some(tlas.clone())
    } else {
        // Full rebuild
        let sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
        if sizes.accelerationStructureSize == 0 {
            prev_instance_keys.clear();
            return None;
        }

        // Alternate buffer index for double-buffering
        let idx = *buffer_index;
        *buffer_index = 1 - idx;

        // Allocate/reuse TLAS
        let need_realloc = match &tlas_store[idx] {
            Some(existing) => existing.size() < sizes.accelerationStructureSize,
            None => true,
        };
        if need_realloc {
            tlas_store[idx] = device.newAccelerationStructureWithSize(sizes.accelerationStructureSize);
        }
        let tlas = tlas_store[idx].as_ref()?;

        // Allocate/reuse scratch buffer
        let scratch_size = sizes.buildScratchBufferSize.max(64);
        let need_scratch = match &scratch_store[idx] {
            Some(existing) => existing.length() < scratch_size,
            None => true,
        };
        if need_scratch {
            scratch_store[idx] = device.newBufferWithLength_options(
                scratch_size,
                MTLResourceOptions::StorageModePrivate,
            );
        }
        let scratch = scratch_store[idx].as_ref()?;

        let cmd_buf = command_queue.commandBuffer()?;
        let encoder = cmd_buf.accelerationStructureCommandEncoder()?;
        encoder.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
            tlas,
            &tlas_desc,
            scratch,
            0,
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Update keys for next frame's refit check
        *prev_instance_keys = current_keys;

        Some(tlas.clone())
    }
}

/// Ensure the raytrace output texture exists and matches the requested size.
pub fn ensure_output_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    texture: &mut Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    stored_w: &mut u32,
    stored_h: &mut u32,
    width: u32,
    height: u32,
) {
    if *stored_w == width && *stored_h == height && texture.is_some() {
        return;
    }

    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::RGBA8Unorm,
            width as usize,
            height as usize,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage(MTLTextureUsage::ShaderWrite.0 | MTLTextureUsage::ShaderRead.0));
    desc.setStorageMode(MTLStorageMode::Shared);

    *texture = device.newTextureWithDescriptor(&desc);
    *stored_w = width;
    *stored_h = height;
    eprintln!("[Rust Metal] Raytrace output texture created: {}x{}", width, height);
}

/// Maximum number of texture slots for alpha cutout remap (slot 0 = no texture).
const MAX_ALPHA_TEXTURE_SLOTS: usize = 31;

/// Dispatch the raytracing compute shader.
/// Uses pre-allocated FrameBuffers to avoid per-frame GPU allocations.
pub fn dispatch_rays(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    mesh_slotmap: &MeshSlotMap,
    instances: &mut Vec<PendingInstance>,
    camera: &Camera,
    render_width: u32,
    render_height: u32,
    compute_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    output_texture: &ProtocolObject<dyn MTLTexture>,
    tlas_store: &mut [Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>; 2],
    scratch_store: &mut [Option<Retained<ProtocolObject<dyn MTLBuffer>>>; 2],
    tlas_buffer_index: &mut usize,
    prev_instance_keys: &mut Vec<u32>,
    lights: &[Light],
    material_edges: &[MaterialEdge],
    material_indices: &[u16],
    gpu_materials: &[GPUMaterial],
    texture_slotmap: &TextureSlotMap,
    white_texture: Option<&ProtocolObject<dyn MTLTexture>>,
    frame_buffers: &mut FrameBuffers,
    raytrace_sampler: Option<&ProtocolObject<dyn MTLSamplerState>>,
) {
    if instances.is_empty() {
        return;
    }

    // 1. Build combined triangle buffer (pre-allocated)
    let total_tris = match build_combined_triangle_buffer(device, mesh_slotmap, instances, frame_buffers) {
        Some(count) => count,
        None => return,
    };

    // 2. Build/refit TLAS
    let mut use_accel: u32 = 0;
    let tlas = build_tlas(
        device,
        command_queue,
        mesh_slotmap,
        instances,
        tlas_store,
        scratch_store,
        tlas_buffer_index,
        prev_instance_keys,
    );
    if tlas.is_some() {
        use_accel = 1;
    }

    // 3. Build texture remap for alpha cutout triangles
    let rt_triangle_alpha_cutout: u32 = 1 << 29;
    let rt_triangle_mei_mask: u32 = 0x1FFFFFFF;

    let mut remap_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut remap_textures: Vec<(u32, &ProtocolObject<dyn MTLTexture>)> = Vec::new();
    let mut next_slot: u32 = 1;

    let combined_tri_buf = frame_buffers.combined_tri_buf.as_ref().unwrap();
    let combined_ptr = combined_tri_buf.contents().as_ptr() as *const Triangle;
    for i in 0..total_tris as usize {
        let tri = unsafe { &*combined_ptr.add(i) };
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
            }
        }
    }

    let max_remap_idx = remap_map.keys().copied().max().unwrap_or(0) as usize;
    let remap_table_size = max_remap_idx + 1;
    let mut remap_table: Vec<u32> = vec![0u32; remap_table_size.max(1)];
    for (&albedo_idx, &slot) in &remap_map {
        remap_table[albedo_idx as usize] = slot;
    }

    // 4. Fill scene constants
    let vfov_rad = camera.vfov * std::f32::consts::PI / 180.0;
    let scene = RaytraceSceneConstants {
        camera_position: camera.position,
        _pad0: 0.0,
        camera_forward: camera.forward,
        _pad1: 0.0,
        camera_right: camera.right,
        _pad2: 0.0,
        camera_up: camera.up,
        _pad3: 0.0,
        vfov_radians: vfov_rad,
        aspect_ratio: render_width as f32 / render_height as f32,
        render_width,
        render_height,
        instance_count: instances.len() as u32,
        total_triangles: total_tris,
        _pad4: [0.0; 2],
        debug_mode: 0,
        texture_count: remap_textures.len() as u32,
        use_accel,
        light_count: lights.len() as u32,
    };

    // 5. Write data into pre-allocated buffers

    // Scene constants (fixed size — allocate once)
    let scene_size = std::mem::size_of::<RaytraceSceneConstants>();
    if !crate::state::ensure_fixed_buffer(device, &mut frame_buffers.scene_buf, scene_size) {
        return;
    }
    let scene_buf = frame_buffers.scene_buf.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            &scene as *const RaytraceSceneConstants as *const u8,
            scene_buf.contents().as_ptr() as *mut u8,
            scene_size,
        );
    }

    // Instance buffer (variable size)
    let instance_data: Vec<RaytraceInstance> = instances.iter().map(|p| p.instance).collect();
    let inst_size = std::mem::size_of::<RaytraceInstance>();
    if !crate::state::ensure_buffer_capacity(
        device,
        &mut frame_buffers.instance_buf,
        &mut frame_buffers.instance_capacity,
        instance_data.len(),
        inst_size,
    ) {
        return;
    }
    let inst_buf = frame_buffers.instance_buf.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            instance_data.as_ptr() as *const u8,
            inst_buf.contents().as_ptr() as *mut u8,
            instance_data.len() * inst_size,
        );
    }

    // Material edges (fixed max size — allocate once)
    let me_byte_len = material_edges.len() * std::mem::size_of::<MaterialEdge>();
    if !crate::state::ensure_fixed_buffer(device, &mut frame_buffers.material_edges_buf, me_byte_len) {
        return;
    }
    let me_buf = frame_buffers.material_edges_buf.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            material_edges.as_ptr() as *const u8,
            me_buf.contents().as_ptr() as *mut u8,
            me_byte_len,
        );
    }

    // Material indices (fixed max size — allocate once)
    let mi_byte_len = material_indices.len() * std::mem::size_of::<u16>();
    if !crate::state::ensure_fixed_buffer(device, &mut frame_buffers.material_indices_buf, mi_byte_len) {
        return;
    }
    let mi_buf = frame_buffers.material_indices_buf.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            material_indices.as_ptr() as *const u8,
            mi_buf.contents().as_ptr() as *mut u8,
            mi_byte_len,
        );
    }

    // GPU materials (fixed max size — allocate once)
    let gm_byte_len = gpu_materials.len() * std::mem::size_of::<GPUMaterial>();
    if !crate::state::ensure_fixed_buffer(device, &mut frame_buffers.gpu_materials_buf, gm_byte_len) {
        return;
    }
    let gm_buf = frame_buffers.gpu_materials_buf.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            gpu_materials.as_ptr() as *const u8,
            gm_buf.contents().as_ptr() as *mut u8,
            gm_byte_len,
        );
    }

    // Light buffer (variable size)
    if !lights.is_empty() {
        let light_elem_size = std::mem::size_of::<Light>();
        if !crate::state::ensure_buffer_capacity(
            device,
            &mut frame_buffers.light_buf,
            &mut frame_buffers.light_capacity,
            lights.len(),
            light_elem_size,
        ) {
            return;
        }
        let light_buf = frame_buffers.light_buf.as_ref().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                lights.as_ptr() as *const u8,
                light_buf.contents().as_ptr() as *mut u8,
                lights.len() * light_elem_size,
            );
        }
    }

    // Remap table (variable size)
    let remap_elem_size = std::mem::size_of::<u32>();
    if !crate::state::ensure_buffer_capacity(
        device,
        &mut frame_buffers.remap_buf,
        &mut frame_buffers.remap_capacity,
        remap_table.len(),
        remap_elem_size,
    ) {
        return;
    }
    let remap_buf = frame_buffers.remap_buf.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            remap_table.as_ptr() as *const u8,
            remap_buf.contents().as_ptr() as *mut u8,
            remap_table.len() * remap_elem_size,
        );
    }

    // Remap size (fixed 4 bytes — allocate once)
    if !crate::state::ensure_fixed_buffer(device, &mut frame_buffers.remap_size_buf, std::mem::size_of::<u32>()) {
        return;
    }
    let rs_buf = frame_buffers.remap_size_buf.as_ref().unwrap();
    let remap_size = remap_table.len() as u32;
    unsafe {
        std::ptr::copy_nonoverlapping(
            &remap_size as *const u32 as *const u8,
            rs_buf.contents().as_ptr() as *mut u8,
            std::mem::size_of::<u32>(),
        );
    }

    // 6. Dispatch compute
    let cmd_buf = match command_queue.commandBuffer() {
        Some(b) => b,
        None => return,
    };

    let encoder = match cmd_buf.computeCommandEncoder() {
        Some(e) => e,
        None => return,
    };

    encoder.setComputePipelineState(compute_pipeline);
    unsafe {
        encoder.setTexture_atIndex(Some(output_texture), 0);
    }
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(scene_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(inst_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(combined_tri_buf), 0, 2);
    }

    // Bind TLAS at buffer index 8
    if let Some(ref tlas) = tlas {
        unsafe {
            encoder.setAccelerationStructure_atBufferIndex(Some(tlas.as_ref()), 8);
        }
    }

    // Bind light buffer at index 9
    if !lights.is_empty() {
        if let Some(ref buf) = frame_buffers.light_buf {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, 9);
            }
        }
    }

    // Bind material edges at index 5
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(me_buf), 0, 5);
    }

    // Bind material indices at index 6
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(mi_buf), 0, 6);
    }

    // Bind gpu_materials at index 7
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gm_buf), 0, 7);
    }

    // Bind remap table at index 11
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(remap_buf), 0, 11);
    }

    // Bind remap size at index 12
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(rs_buf), 0, 12);
    }

    // Bind cached sampler
    if let Some(sampler) = raytrace_sampler {
        unsafe {
            encoder.setSamplerState_atIndex(Some(sampler), 0);
        }
    }

    // Bind alpha cutout textures
    if let Some(white_tex) = white_texture {
        unsafe {
            encoder.setTexture_atIndex(Some(white_tex), 16);
        }
    }
    for (i, (_albedo_idx, tex)) in remap_textures.iter().enumerate() {
        unsafe {
            encoder.setTexture_atIndex(Some(*tex), 17 + i);
        }
    }

    // Dispatch threadgroups
    let threads_per_group = MTLSize { width: 8, height: 8, depth: 1 };
    let threadgroups = MTLSize {
        width: (render_width as usize + 7) / 8,
        height: (render_height as usize + 7) / 8,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_group);
    encoder.endEncoding();

    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
}
