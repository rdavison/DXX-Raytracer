//! Safe wrappers around Metal acceleration structures (BLAS/TLAS).
//!
//! Contains `GpuAccelStructure` (thin wrapper) and builder helpers
//! for bottom-level (BLAS) and top-level (TLAS) acceleration structures.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use super::encoder::GpuCommandBuffer;

/// Safe wrapper around a Metal acceleration structure.
pub struct GpuAccelStructure {
    accel: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
}

impl GpuAccelStructure {
    /// Wrap an existing retained acceleration structure.
    pub fn from_retained(
        accel: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
    ) -> Self {
        Self { accel }
    }

    /// Size in bytes of the acceleration structure.
    pub fn size(&self) -> usize {
        self.accel.size()
    }

    /// Access the underlying `MTLAccelerationStructure` for encoder binding.
    pub fn metal_accel(&self) -> &ProtocolObject<dyn MTLAccelerationStructure> {
        &self.accel
    }

    /// Clone the inner retained reference (bumps the reference count).
    pub fn clone_retained(&self) -> Retained<ProtocolObject<dyn MTLAccelerationStructure>> {
        self.accel.clone()
    }

    /// Consume and return the inner retained acceleration structure.
    pub fn into_retained(self) -> Retained<ProtocolObject<dyn MTLAccelerationStructure>> {
        self.accel
    }
}

/// Build a bottom-level acceleration structure from a triangle position buffer,
/// then compact it to reduce memory usage.
///
/// `position_buffer` must contain `triangle_count * 3` packed float3 vertices
/// (12 bytes per vertex, 36 bytes per triangle).
///
/// Returns `None` if any allocation or build step fails.
pub fn build_blas(
    device: &ProtocolObject<dyn MTLDevice>,
    command_queue: &ProtocolObject<dyn MTLCommandQueue>,
    position_buffer: &ProtocolObject<dyn MTLBuffer>,
    triangle_count: usize,
) -> Option<GpuAccelStructure> {
    // Create geometry descriptor
    let geo_desc = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
    geo_desc.setVertexBuffer(Some(position_buffer));
    unsafe {
        geo_desc.setVertexBufferOffset(0);
    }
    geo_desc.setVertexStride(12); // packed_float3 = 12 bytes
    geo_desc.setTriangleCount(triangle_count);
    geo_desc.setOpaque(true);

    // Create primitive AS descriptor
    let prim_desc = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    let geo_descs: Retained<
        objc2_foundation::NSArray<MTLAccelerationStructureGeometryDescriptor>,
    > = unsafe {
        let raw_ptr: *const MTLAccelerationStructureTriangleGeometryDescriptor = &*geo_desc;
        let base_ptr: *const MTLAccelerationStructureGeometryDescriptor = raw_ptr.cast();
        objc2_foundation::NSArray::from_slice(&[&*base_ptr])
    };
    prim_desc.setGeometryDescriptors(Some(&geo_descs));

    // Query sizes
    let sizes = device.accelerationStructureSizesWithDescriptor(&prim_desc);
    if sizes.accelerationStructureSize == 0 {
        return None;
    }

    // Allocate uncompacted structure
    let blas = device.newAccelerationStructureWithSize(sizes.accelerationStructureSize)?;

    // Create scratch buffer
    let scratch_size = sizes.buildScratchBufferSize.max(64);
    let scratch = device
        .newBufferWithLength_options(scratch_size, MTLResourceOptions::StorageModePrivate)?;

    // Build
    let cmd_buf = GpuCommandBuffer::new(command_queue.commandBuffer()?);
    if let Some(encoder) = cmd_buf.accel_encoder() {
        encoder.build(&blas, &prim_desc, &scratch);
        encoder.end_encoding();
    }
    let committed = cmd_buf.commit();
    committed.wait_until_completed();

    // Compact: query compacted size
    let size_buf = device.newBufferWithLength_options(
        std::mem::size_of::<u32>(),
        MTLResourceOptions::StorageModeShared,
    )?;

    let cmd_buf2 = GpuCommandBuffer::new(command_queue.commandBuffer()?);
    if let Some(encoder) = cmd_buf2.accel_encoder() {
        encoder.write_compacted_size(&blas, &size_buf);
        encoder.end_encoding();
    }
    let committed2 = cmd_buf2.commit();
    committed2.wait_until_completed();

    let compacted_size =
        unsafe { *(size_buf.contents().as_ptr() as *const u32) as usize };

    if compacted_size == 0 || compacted_size >= sizes.accelerationStructureSize {
        return Some(GpuAccelStructure::from_retained(blas));
    }

    // Compact into smaller structure
    let compacted_blas = device.newAccelerationStructureWithSize(compacted_size)?;

    let cmd_buf3 = GpuCommandBuffer::new(command_queue.commandBuffer()?);
    if let Some(encoder) = cmd_buf3.accel_encoder() {
        encoder.copy_and_compact(&blas, &compacted_blas);
        encoder.end_encoding();
    }
    let committed3 = cmd_buf3.commit();
    committed3.wait_until_completed();

    Some(GpuAccelStructure::from_retained(compacted_blas))
}

// ============================================================================
// TlasState — TLAS management
// ============================================================================

/// Top-level acceleration structure state.
///
/// Since we waitUntilCompleted after both the TLAS build and the compute
/// dispatch, there's no GPU/CPU overlap — a single slot suffices.
/// Rebuilt every frame (matching the ObjC Metal backend).
pub struct TlasState {
    pub tlas: [Option<GpuAccelStructure>; 2],
    pub scratch: [Option<Retained<ProtocolObject<dyn MTLBuffer>>>; 2],
    /// Cached instance descriptor buffer for TLAS builds (avoids per-frame allocation).
    pub inst_desc_buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub inst_desc_capacity: usize,
}

impl TlasState {
    pub fn new() -> Self {
        Self {
            tlas: [None, None],
            scratch: [None, None],
            inst_desc_buf: None,
            inst_desc_capacity: 0,
        }
    }
}
