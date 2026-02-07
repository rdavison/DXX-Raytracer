//! Per-frame raytracing orchestration.
//!
//! Builds combined triangle buffer, TLAS, and dispatches compute shader.
//! Works directly with domain types — no PendingInstance bridge.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::domain::SceneInstance;
use crate::gpu::accel::TlasState;
use crate::mesh::MeshSlotMap;
use crate::state::FrameBuffers;
use crate::texture::TextureSlotMap;
use crate::types::*;

use super::texture_remap;

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

/// Scene constants — 128 bytes, matches shader SceneConstants struct.
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
    pub shadow_mode: u32,    // 0=hard, 1=multi-sample, 2=analytic
    pub frame_number: u32,   // RNG seed, incremented each frame
    pub debug_mode: u32,
    pub texture_count: u32,
    pub use_accel: u32,
    pub light_count: u32,
    pub billboard_opacity_threshold: f32,
    pub billboard_emissive_boost: f32,
    pub _pad4: [u32; 2],  // pad to 128 bytes (16-byte aligned)
}

/// Instance resolved against the mesh slotmap — ready for GPU upload.
/// Created at dispatch time from SceneInstance + mesh lookup.
pub(crate) struct ResolvedInstance {
    pub gpu: RaytraceInstance,
    pub mesh_handle: ResourceHandle,
}

// ============================================================================
// Resolve domain instances into GPU-ready instances
// ============================================================================

/// Diagnostics from instance resolution.
pub(crate) struct ResolveDiagnostics {
    pub level_meshes: u32,
    pub billboards: u32,
    pub rods: u32,
    pub failed: u32,
}

/// Convert domain `SceneInstance`s into `ResolvedInstance`s by looking up
/// mesh data in the slotmap. Instances with missing meshes are filtered out.
fn resolve_instances(
    instances: &[SceneInstance],
    mesh_slotmap: &MeshSlotMap,
) -> (Vec<ResolvedInstance>, ResolveDiagnostics) {
    let mut diag = ResolveDiagnostics {
        level_meshes: 0, billboards: 0, rods: 0, failed: 0,
    };
    let resolved = instances.iter().filter_map(|si| {
        let handle = si.mesh().raw();
        let mesh = match mesh_slotmap.find(handle) {
            Some(m) => m,
            None => {
                diag.failed += 1;
                return None;
            }
        };
        match si {
            SceneInstance::LevelMesh { .. } => diag.level_meshes += 1,
            SceneInstance::Billboard { .. } => diag.billboards += 1,
            SceneInstance::Rod { .. } => diag.rods += 1,
        }
        Some(ResolvedInstance {
            gpu: RaytraceInstance {
                object_to_world: *si.transform(),
                world_to_object: si.transform().inverse(),
                triangle_buffer_idx: handle.index,
                triangle_count: mesh.triangle_count as u32,
                color: si.color().packed(),
                material_override: si.material_override()
                    .map(|m| m.as_u16() as u32).unwrap_or(0),
                triangle_offset: 0,
                _pad: [0; 3],
            },
            mesh_handle: handle,
        })
    }).collect();
    (resolved, diag)
}

// ============================================================================
// Per-frame functions
// ============================================================================

/// Build a single combined triangle buffer from all resolved instances.
/// Uses pre-allocated buffer from FrameBuffers, growing only when needed.
/// Updates triangle_offset for each instance. Returns total_triangle_count, or None if empty.
fn build_combined_triangle_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    mesh_slotmap: &MeshSlotMap,
    instances: &mut [ResolvedInstance],
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
            inst.gpu.triangle_offset = (offset / tri_size) as u32;

            let src = mesh.triangle_buffer.contents().as_ptr() as *const u8;
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst_base.add(offset), byte_count);
            }
            offset += byte_count;
        }
    }

    Some(total_tris as u32)
}

/// Build the top-level acceleration structure from resolved instances.
///
/// Always performs a full rebuild (matching the ObjC Metal backend).
/// Since we waitUntilCompleted after both the TLAS build and the compute
/// dispatch, there's no GPU/CPU overlap that would require double-buffering,
/// so we reuse a single TLAS slot and only reallocate when it's too small.
fn build_tlas(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    mesh_slotmap: &MeshSlotMap,
    instances: &[ResolvedInstance],
    tlas_state: &mut TlasState,
) -> Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> {
    if instances.is_empty() {
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
        return None;
    }

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

        let m = &inst.gpu.object_to_world.e;
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

    // Full rebuild every frame
    let sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
    if sizes.accelerationStructureSize == 0 {
        return None;
    }

    // Reuse slot 0; only reallocate when too small
    let need_realloc = match &tlas_state.tlas[0] {
        Some(existing) => existing.size() < sizes.accelerationStructureSize,
        None => true,
    };
    if need_realloc {
        if let Some(new_accel) = device.newAccelerationStructureWithSize(sizes.accelerationStructureSize) {
            tlas_state.tlas[0] = Some(crate::gpu::accel::GpuAccelStructure::from_retained(new_accel));
        } else {
            return None;
        }
    }
    let tlas_accel = tlas_state.tlas[0].as_ref()?;
    let tlas = tlas_accel.metal_accel();

    let scratch_size = sizes.buildScratchBufferSize.max(64);
    let need_scratch = match &tlas_state.scratch[0] {
        Some(existing) => existing.length() < scratch_size,
        None => true,
    };
    if need_scratch {
        tlas_state.scratch[0] = device.newBufferWithLength_options(
            scratch_size,
            MTLResourceOptions::StorageModePrivate,
        );
    }
    let scratch = tlas_state.scratch[0].as_ref()?;

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

    Some(tlas_accel.clone_retained())
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

/// Dispatch the raytracing compute shader.
/// Uses pre-allocated FrameBuffers to avoid per-frame GPU allocations.
pub fn dispatch(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    mesh_slotmap: &MeshSlotMap,
    instances: &[SceneInstance],
    camera: &Camera,
    render_width: u32,
    render_height: u32,
    compute_pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    tile_cull_pipeline: Option<&ProtocolObject<dyn MTLComputePipelineState>>,
    output_texture: &ProtocolObject<dyn MTLTexture>,
    tlas_state: &mut TlasState,
    lights: &[Light],
    material_edges: &[MaterialEdge],
    material_indices: &[u16],
    gpu_materials: &[GPUMaterial],
    texture_slotmap: &TextureSlotMap,
    white_texture: Option<&ProtocolObject<dyn MTLTexture>>,
    frame_buffers: &mut FrameBuffers,
    raytrace_sampler: Option<&ProtocolObject<dyn MTLSamplerState>>,
    arg_buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
) {
    if instances.is_empty() {
        return;
    }

    // Frame counter for periodic diagnostics
    static DISPATCH_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let frame = DISPATCH_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // 1. Resolve domain instances into GPU-ready instances
    let (mut resolved, resolve_diag) = resolve_instances(instances, mesh_slotmap);
    if resolved.is_empty() {
        return;
    }

    // 2. Build combined triangle buffer (pre-allocated)
    let total_tris = match build_combined_triangle_buffer(device, mesh_slotmap, &mut resolved, frame_buffers) {
        Some(count) => count,
        None => return,
    };

    // 3. Build/refit TLAS
    let mut use_accel: u32 = 0;
    let tlas = build_tlas(device, command_queue, mesh_slotmap, &resolved, tlas_state);
    if tlas.is_some() {
        use_accel = 1;
    }

    // 4. Build texture remap for alpha cutout triangles
    let combined_tri_buf = frame_buffers.combined_tri_buf.as_ref().unwrap();
    let combined_ptr = combined_tri_buf.contents().as_ptr() as *const Triangle;
    let triangles_slice = unsafe {
        std::slice::from_raw_parts(combined_ptr, total_tris as usize)
    };

    let remap = texture_remap::build_texture_remap(
        triangles_slice,
        &resolved,
        material_edges,
        material_indices,
        gpu_materials,
        texture_slotmap,
    );

    // Periodic diagnostics (every 60 frames)
    if frame % 60 == 0 {
        let rd = &resolve_diag;
        let td = &remap.diagnostics;
        eprintln!(
            "[Rust Metal] dispatch #{}: instances: {} level + {} billboard + {} rod ({} failed) | \
             remap: {} cutout + {} override mapped, {} no_albedo, {} tex_missing, {} already_mapped, {}/{} slots",
            frame,
            rd.level_meshes, rd.billboards, rd.rods, rd.failed,
            td.cutout_mapped, td.override_mapped,
            td.override_no_albedo, td.override_tex_missing, td.override_already_mapped,
            td.slots_used, super::texture_remap::MAX_ALPHA_TEXTURE_SLOTS,
        );
    }


    // 5. Fill scene constants
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
        instance_count: resolved.len() as u32,
        total_triangles: total_tris,
        shadow_mode: 2,           // analytic soft shadows by default
        frame_number: frame as u32,
        debug_mode: 0,
        texture_count: remap.textures.len() as u32,
        use_accel,
        light_count: lights.len() as u32,
        billboard_opacity_threshold: 0.8,
        billboard_emissive_boost: 2.5,
        _pad4: [0; 2],
    };

    // 6. Write data into pre-allocated buffers

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
    let instance_data: Vec<RaytraceInstance> = resolved.iter().map(|r| r.gpu).collect();
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
        remap.remap_table.len(),
        remap_elem_size,
    ) {
        return;
    }
    let remap_buf = frame_buffers.remap_buf.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            remap.remap_table.as_ptr() as *const u8,
            remap_buf.contents().as_ptr() as *mut u8,
            remap.remap_table.len() * remap_elem_size,
        );
    }

    // Remap size (fixed 4 bytes — allocate once)
    if !crate::state::ensure_fixed_buffer(device, &mut frame_buffers.remap_size_buf, std::mem::size_of::<u32>()) {
        return;
    }
    let rs_buf = frame_buffers.remap_size_buf.as_ref().unwrap();
    let remap_size = remap.remap_table.len() as u32;
    unsafe {
        std::ptr::copy_nonoverlapping(
            &remap_size as *const u32 as *const u8,
            rs_buf.contents().as_ptr() as *mut u8,
            std::mem::size_of::<u32>(),
        );
    }

    // 7. Tile-based light culling pass
    const TILE_SIZE: u32 = 16;
    const MAX_LIGHTS_PER_TILE: usize = 64;
    const TILE_STRIDE: usize = 1 + MAX_LIGHTS_PER_TILE;

    let tiles_x = (render_width + TILE_SIZE - 1) / TILE_SIZE;
    let tiles_y = (render_height + TILE_SIZE - 1) / TILE_SIZE;
    let total_tiles = (tiles_x * tiles_y) as usize;
    let tile_data_elems = total_tiles * TILE_STRIDE;

    if !crate::state::ensure_buffer_capacity(
        device,
        &mut frame_buffers.tile_data_buf,
        &mut frame_buffers.tile_data_capacity,
        tile_data_elems,
        std::mem::size_of::<u32>(),
    ) {
        return;
    }
    let tile_data_buf = frame_buffers.tile_data_buf.as_ref().unwrap();

    if let Some(tile_pipeline) = tile_cull_pipeline {
        let cull_cmd = match command_queue.commandBuffer() {
            Some(b) => b,
            None => return,
        };
        let cull_enc = match cull_cmd.computeCommandEncoder() {
            Some(e) => e,
            None => return,
        };
        cull_enc.setComputePipelineState(tile_pipeline);
        unsafe {
            cull_enc.setBuffer_offset_atIndex(Some(scene_buf), 0, 0);
        }
        if !lights.is_empty() {
            if let Some(ref buf) = frame_buffers.light_buf {
                unsafe {
                    cull_enc.setBuffer_offset_atIndex(Some(buf), 0, 9);
                }
            }
        }
        unsafe {
            cull_enc.setBuffer_offset_atIndex(Some(tile_data_buf), 0, 13);
        }
        let tile_threads_per_group = MTLSize { width: 16, height: 16, depth: 1 };
        let tile_threadgroups = MTLSize {
            width: (tiles_x as usize + 15) / 16,
            height: (tiles_y as usize + 15) / 16,
            depth: 1,
        };
        cull_enc.dispatchThreadgroups_threadsPerThreadgroup(tile_threadgroups, tile_threads_per_group);
        cull_enc.endEncoding();
        cull_cmd.commit();
        cull_cmd.waitUntilCompleted();
    }

    // 8. Dispatch main raytrace compute
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

    // Bind gpu_materials at index 10
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(gm_buf), 0, 10);
    }

    // Bind argument buffer (bindless textures) at index 7
    if let Some(ab) = arg_buffer {
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(ab), 0, 7);
        }
        // Make all active textures GPU-resident
        texture_slotmap.for_each_active(|_idx, tex| {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(tex);
            unsafe {
                encoder.useResource_usage(resource, MTLResourceUsage::Read);
            }
        });
        // Also make white texture resident (fallback slots reference it)
        if let Some(wt) = white_texture {
            let resource: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(wt);
            unsafe {
                encoder.useResource_usage(resource, MTLResourceUsage::Read);
            }
        }
    }

    // Bind remap table at index 11
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(remap_buf), 0, 11);
    }

    // Bind remap size at index 12
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(rs_buf), 0, 12);
    }

    // Bind tile light data at index 13
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(tile_data_buf), 0, 13);
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
    for (i, (_albedo_idx, tex)) in remap.textures.iter().enumerate() {
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
