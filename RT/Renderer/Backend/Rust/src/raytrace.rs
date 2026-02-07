//! Per-frame raytracing orchestration (Phase 3A).
//!
//! Builds combined triangle buffer, TLAS, and dispatches compute shader.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::mesh::MeshSlotMap;
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
/// Updates triangle_offset for each instance. Returns (buffer, total_triangle_count).
pub fn build_combined_triangle_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    mesh_slotmap: &MeshSlotMap,
    instances: &mut [PendingInstance],
) -> Option<(Retained<ProtocolObject<dyn MTLBuffer>>, u32)> {
    // Calculate total size
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

    let total_bytes = total_tris * tri_size;
    let combined = device.newBufferWithLength_options(
        total_bytes,
        MTLResourceOptions::StorageModeShared,
    )?;

    // Copy triangle data and set offsets
    let dst_base = combined.contents().as_ptr() as *mut u8;
    let mut offset: usize = 0;

    for inst in instances.iter_mut() {
        if let Some(mesh) = mesh_slotmap.find(inst.mesh_handle) {
            let byte_count = mesh.triangle_count * tri_size;
            inst.instance.triangle_offset = (offset / tri_size) as u32;

            // Copy from mesh's triangle buffer
            let src = mesh.triangle_buffer.contents().as_ptr() as *const u8;
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst_base.add(offset), byte_count);
            }
            offset += byte_count;
        }
    }

    Some((combined, total_tris as u32))
}

/// Build the top-level acceleration structure from queued instances.
pub fn build_tlas(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    mesh_slotmap: &MeshSlotMap,
    instances: &[PendingInstance],
    tlas_store: &mut [Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>; 2],
    scratch_store: &mut [Option<Retained<ProtocolObject<dyn MTLBuffer>>>; 2],
    buffer_index: &mut usize,
) -> Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> {
    if instances.is_empty() {
        return None;
    }

    // Collect unique BLAS references and build instance descriptors
    // Map: mesh_handle.index -> (blas_index, &BLAS)
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

        // Convert row-major Mat4 to Metal's column-major MTLPackedFloat4x3
        // Metal wants 4 columns, each with 3 rows (x,y,z)
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

    // Query sizes
    let sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
    if sizes.accelerationStructureSize == 0 {
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

    // Build TLAS
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

    Some(tlas.clone())
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
    lights: &[Light],
    material_edges: &[MaterialEdge],
    material_indices: &[u16],
    gpu_materials: &[GPUMaterial],
    texture_slotmap: &TextureSlotMap,
    white_texture: Option<&ProtocolObject<dyn MTLTexture>>,
) {
    if instances.is_empty() {
        return;
    }

    // 1. Build combined triangle buffer
    let (combined_tri_buf, total_tris) = match build_combined_triangle_buffer(device, mesh_slotmap, instances) {
        Some(result) => result,
        None => return,
    };

    // 2. Build TLAS
    let mut use_accel: u32 = 0;
    let tlas = build_tlas(
        device,
        command_queue,
        mesh_slotmap,
        instances,
        tlas_store,
        scratch_store,
        tlas_buffer_index,
    );
    if tlas.is_some() {
        use_accel = 1;
    }

    // 3. Build texture remap for alpha cutout triangles
    // Scan triangles for ALPHA_CUTOUT flag, resolve their textures, assign remap slots
    let rt_triangle_alpha_cutout: u32 = 1 << 29;
    let rt_triangle_mei_mask: u32 = 0x1FFFFFFF;

    // albedo_index -> remap slot (1-based; 0 = no texture)
    let mut remap_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    // Ordered list of (albedo_index, &MTLTexture) for binding
    let mut remap_textures: Vec<(u32, &ProtocolObject<dyn MTLTexture>)> = Vec::new();
    let mut next_slot: u32 = 1;

    // Read combined triangle data to find cutout triangles
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

        // Resolve overlay texture (mat2), fall back to base (mat1)
        let mat2_raw = edge.mat2;
        let mat2_tex = (mat2_raw & 0x03FF) as usize; // bits 0-9
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

        // Check if already remapped
        if remap_map.contains_key(&albedo_idx) {
            continue;
        }

        // Look up actual texture
        if let Some(tex) = texture_slotmap.find_by_index(albedo_idx) {
            if next_slot < MAX_ALPHA_TEXTURE_SLOTS as u32 {
                remap_map.insert(albedo_idx, next_slot);
                remap_textures.push((albedo_idx, tex));
                next_slot += 1;
            }
        }
    }

    // Build remap table: indexed by albedo_index → slot number
    // We need a buffer large enough for all albedo indices we encountered
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

    // Create GPU buffers for scene constants and instance data
    let scene_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new_unchecked(&scene as *const RaytraceSceneConstants as *mut c_void),
            std::mem::size_of::<RaytraceSceneConstants>(),
            MTLResourceOptions::StorageModeShared,
        )
    };
    let scene_buf = match scene_buf {
        Some(b) => b,
        None => return,
    };

    // Instance buffer
    let instance_data: Vec<RaytraceInstance> = instances.iter().map(|p| p.instance).collect();
    let inst_buf = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new_unchecked(instance_data.as_ptr() as *mut c_void),
            instance_data.len() * std::mem::size_of::<RaytraceInstance>(),
            MTLResourceOptions::StorageModeShared,
        )
    };
    let inst_buf = match inst_buf {
        Some(b) => b,
        None => return,
    };

    // 4. Dispatch compute
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
        encoder.setBuffer_offset_atIndex(Some(&scene_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&inst_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&combined_tri_buf), 0, 2);
    }

    // Bind TLAS at buffer index 8
    if let Some(ref tlas) = tlas {
        unsafe {
            encoder.setAccelerationStructure_atBufferIndex(Some(tlas.as_ref()), 8);
        }
    }

    // Bind light buffer at index 9
    if !lights.is_empty() {
        let light_buf = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(lights.as_ptr() as *mut c_void),
                lights.len() * std::mem::size_of::<Light>(),
                MTLResourceOptions::StorageModeShared,
            )
        };
        if let Some(ref buf) = light_buf {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, 9);
            }
        }
    }

    // Bind material edges buffer at index 5
    if !material_edges.is_empty() {
        let me_buf = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(material_edges.as_ptr() as *mut c_void),
                material_edges.len() * std::mem::size_of::<MaterialEdge>(),
                MTLResourceOptions::StorageModeShared,
            )
        };
        if let Some(ref buf) = me_buf {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, 5);
            }
        }
    }

    // Bind material_indices buffer at index 6 (packed u16 pairs)
    if !material_indices.is_empty() {
        let mi_buf = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(material_indices.as_ptr() as *mut c_void),
                material_indices.len() * std::mem::size_of::<u16>(),
                MTLResourceOptions::StorageModeShared,
            )
        };
        if let Some(ref buf) = mi_buf {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, 6);
            }
        }
    }

    // Bind gpu_materials buffer at index 7
    if !gpu_materials.is_empty() {
        let gm_buf = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(gpu_materials.as_ptr() as *mut c_void),
                gpu_materials.len() * std::mem::size_of::<GPUMaterial>(),
                MTLResourceOptions::StorageModeShared,
            )
        };
        if let Some(ref buf) = gm_buf {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, 7);
            }
        }
    }

    // Bind texture remap table at index 11
    {
        let remap_buf = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(remap_table.as_ptr() as *mut c_void),
                remap_table.len() * std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
        };
        if let Some(ref buf) = remap_buf {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, 11);
            }
        }

        // Bind remap table size at index 12 (so shader knows bounds)
        let remap_size = remap_table.len() as u32;
        let rs_buf = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(&remap_size as *const u32 as *mut c_void),
                std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
        };
        if let Some(ref buf) = rs_buf {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf), 0, 12);
            }
        }
    }

    // Bind sampler for texture sampling
    {
        let sampler_desc = MTLSamplerDescriptor::new();
        sampler_desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sampler_desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sampler_desc.setSAddressMode(MTLSamplerAddressMode::Repeat);
        sampler_desc.setTAddressMode(MTLSamplerAddressMode::Repeat);
        if let Some(sampler) = device.newSamplerStateWithDescriptor(&sampler_desc) {
            unsafe {
                encoder.setSamplerState_atIndex(Some(&sampler), 0);
            }
        }
    }

    // Bind alpha cutout textures: slot 0 = white fallback at texture index 16,
    // slots 1-30 = remapped textures at texture indices 17-46
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
