//! Mesh storage with BLAS construction for raytracing (Phase 3A).
//!
//! Each uploaded mesh gets a triangle buffer (full Triangle data),
//! a position-only buffer (3 × packed_float3 per triangle for BLAS),
//! and a bottom-level acceleration structure.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::types::{ResourceHandle, Triangle};

/// A single mesh slot containing GPU buffers and BLAS.
pub struct MeshSlot {
    pub triangle_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    #[allow(dead_code)] // Retained to keep BLAS valid
    position_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub blas: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
    pub triangle_count: usize,
    generation: u32,
}

/// Slotmap for mesh resources indexed by ResourceHandle.
pub struct MeshSlotMap {
    slots: Vec<Option<MeshSlot>>,
    next_generation: u32,
}

impl MeshSlotMap {
    pub fn new() -> Self {
        Self {
            // Slot 0 reserved (NULL handle)
            slots: vec![None],
            next_generation: 1,
        }
    }

    /// Look up a mesh by handle, validating generation.
    pub fn find(&self, handle: ResourceHandle) -> Option<&MeshSlot> {
        let slot = self.slots.get(handle.index as usize)?.as_ref()?;
        if slot.generation == handle.generation {
            Some(slot)
        } else {
            None
        }
    }

    /// Remove a mesh by handle.
    pub fn remove(&mut self, handle: ResourceHandle) {
        if let Some(slot) = self.slots.get_mut(handle.index as usize) {
            if let Some(s) = slot.as_ref() {
                if s.generation == handle.generation {
                    *slot = None;
                }
            }
        }
    }

    /// Insert a mesh slot, returning a ResourceHandle.
    fn insert(&mut self, slot: MeshSlot) -> ResourceHandle {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);
        if self.next_generation == 0 {
            self.next_generation = 1;
        }

        // Find first free slot (skip index 0)
        for (i, s) in self.slots.iter_mut().enumerate().skip(1) {
            if s.is_none() {
                *s = Some(MeshSlot { generation, ..slot });
                return ResourceHandle { index: i as u32, generation };
            }
        }

        // No free slot — append
        let index = self.slots.len() as u32;
        self.slots.push(Some(MeshSlot { generation, ..slot }));
        ResourceHandle { index, generation }
    }
}

/// Upload mesh triangles to the GPU, build BLAS, return handle.
pub fn upload_mesh(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    slotmap: &mut MeshSlotMap,
    triangles: &[Triangle],
) -> ResourceHandle {
    if triangles.is_empty() {
        eprintln!("[Rust Metal] WARNING: upload_mesh called with 0 triangles");
        return ResourceHandle::NULL;
    }

    let tri_count = triangles.len();

    // 1. Create triangle data buffer (full Triangle structs)
    let tri_byte_len = tri_count * std::mem::size_of::<Triangle>();
    let triangle_buffer = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new_unchecked(triangles.as_ptr() as *mut c_void),
            tri_byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let triangle_buffer = match triangle_buffer {
        Some(b) => b,
        None => {
            eprintln!("[Rust Metal] ERROR: Failed to create triangle buffer");
            return ResourceHandle::NULL;
        }
    };

    // 2. Extract position-only buffer: 3 × packed_float3 per triangle
    //    packed_float3 = 12 bytes, so 36 bytes per triangle
    let pos_stride: usize = 12; // sizeof(packed_float3)
    let pos_byte_len = tri_count * 3 * pos_stride;
    let mut positions = Vec::<f32>::with_capacity(tri_count * 9);
    for tri in triangles {
        // vertex 0
        positions.push(tri.positions[0].x);
        positions.push(tri.positions[0].y);
        positions.push(tri.positions[0].z);
        // vertex 1
        positions.push(tri.positions[1].x);
        positions.push(tri.positions[1].y);
        positions.push(tri.positions[1].z);
        // vertex 2
        positions.push(tri.positions[2].x);
        positions.push(tri.positions[2].y);
        positions.push(tri.positions[2].z);
    }

    let position_buffer = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new_unchecked(positions.as_ptr() as *mut c_void),
            pos_byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let position_buffer = match position_buffer {
        Some(b) => b,
        None => {
            eprintln!("[Rust Metal] ERROR: Failed to create position buffer");
            return ResourceHandle::NULL;
        }
    };

    // 3. Build BLAS
    let blas = match build_blas(device, command_queue, &position_buffer, tri_count) {
        Some(b) => b,
        None => {
            eprintln!("[Rust Metal] ERROR: Failed to build BLAS for {} triangles", tri_count);
            return ResourceHandle::NULL;
        }
    };

    eprintln!("[Rust Metal] Mesh uploaded: {} triangles, BLAS built", tri_count);

    // 4. Insert into slotmap
    slotmap.insert(MeshSlot {
        triangle_buffer,
        position_buffer,
        blas,
        triangle_count: tri_count,
        generation: 0, // will be overwritten by insert()
    })
}

/// Build a bottom-level acceleration structure from a position buffer.
fn build_blas(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    position_buffer: &ProtocolObject<dyn MTLBuffer>,
    triangle_count: usize,
) -> Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> {
    // Create geometry descriptor
    let geo_desc = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
    geo_desc.setVertexBuffer(Some(position_buffer));
    unsafe { geo_desc.setVertexBufferOffset(0); }
    geo_desc.setVertexStride(12); // packed_float3 = 12 bytes
    geo_desc.setTriangleCount(triangle_count);
    geo_desc.setOpaque(true);

    // Create primitive acceleration structure descriptor
    let prim_desc = MTLPrimitiveAccelerationStructureDescriptor::descriptor();

    // Create NSArray of geometry descriptors
    // MTLAccelerationStructureTriangleGeometryDescriptor inherits from
    // MTLAccelerationStructureGeometryDescriptor, so we need to cast
    let geo_descs: Retained<objc2_foundation::NSArray<MTLAccelerationStructureGeometryDescriptor>> = unsafe {
        // Cast the triangle geometry descriptor to the base class array
        let raw_ptr: *const MTLAccelerationStructureTriangleGeometryDescriptor = &*geo_desc;
        let base_ptr: *const MTLAccelerationStructureGeometryDescriptor = raw_ptr.cast();
        objc2_foundation::NSArray::from_slice(&[&*base_ptr])
    };
    prim_desc.setGeometryDescriptors(Some(&geo_descs));

    // Query sizes
    let sizes = device.accelerationStructureSizesWithDescriptor(&prim_desc);

    if sizes.accelerationStructureSize == 0 {
        eprintln!("[Rust Metal] WARNING: BLAS size is 0");
        return None;
    }

    // Allocate acceleration structure
    let blas = device.newAccelerationStructureWithSize(sizes.accelerationStructureSize)?;

    // Create scratch buffer
    let scratch_size = sizes.buildScratchBufferSize.max(64);
    let scratch = device.newBufferWithLength_options(
        scratch_size,
        MTLResourceOptions::StorageModePrivate,
    )?;

    // Build via command encoder
    let cmd_buf = command_queue.commandBuffer()?;
    let encoder = cmd_buf.accelerationStructureCommandEncoder()?;
    encoder.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
        &blas,
        &prim_desc,
        &scratch,
        0,
    );
    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    Some(blas)
}
