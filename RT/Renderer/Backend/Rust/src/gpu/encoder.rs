//! Safe wrappers around Metal command buffers and command encoders.
//!
//! `GpuComputeEncoder` and `GpuRenderEncoder` provide typed buffer/texture
//! binding and consume `self` on `end_encoding()` to prevent use-after-end.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use super::buffer::GpuBuffer;
use super::texture::GpuTexture;

// ============================================================================
// GpuCommandBuffer
// ============================================================================

/// Safe wrapper around `MTLCommandBuffer`.
///
/// Provides methods to create encoders and manage the command buffer lifecycle.
pub struct GpuCommandBuffer {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl GpuCommandBuffer {
    pub(super) fn new(cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>) -> Self {
        Self { cmd_buf }
    }

    /// Create a compute command encoder.
    pub fn compute_encoder(&self) -> Option<GpuComputeEncoder> {
        let encoder = self.cmd_buf.computeCommandEncoder()?;
        Some(GpuComputeEncoder { encoder })
    }

    /// Create a render command encoder with the given pass descriptor.
    pub fn render_encoder(
        &self,
        pass_desc: &MTLRenderPassDescriptor,
    ) -> Option<GpuRenderEncoder> {
        let encoder = self.cmd_buf.renderCommandEncoderWithDescriptor(pass_desc)?;
        Some(GpuRenderEncoder { encoder })
    }

    /// Create an acceleration structure command encoder.
    pub fn accel_encoder(&self) -> Option<GpuAccelEncoder> {
        let encoder = self.cmd_buf.accelerationStructureCommandEncoder()?;
        Some(GpuAccelEncoder { encoder })
    }

    /// Commit the command buffer for execution.
    pub fn commit(self) -> CommittedCommandBuffer {
        self.cmd_buf.commit();
        CommittedCommandBuffer {
            cmd_buf: self.cmd_buf,
        }
    }

    /// Present a drawable before committing.
    pub fn present_drawable(&self, drawable: &ProtocolObject<dyn MTLDrawable>) {
        self.cmd_buf.presentDrawable(drawable);
    }

    /// Add a completion handler that runs when the GPU finishes this command buffer.
    ///
    /// # Safety
    /// The handler block must be valid for the lifetime of the GPU execution.
    pub unsafe fn add_completed_handler(
        &self,
        block: *mut block2::Block<dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>,
    ) {
        self.cmd_buf.addCompletedHandler(block);
    }

    /// Access the underlying `MTLCommandBuffer`.
    pub fn metal_command_buffer(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.cmd_buf
    }
}

/// A command buffer that has been committed. Can only wait for completion.
pub struct CommittedCommandBuffer {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommittedCommandBuffer {
    /// Block until the GPU has finished executing this command buffer.
    pub fn wait_until_completed(&self) {
        self.cmd_buf.waitUntilCompleted();
    }
}

// ============================================================================
// GpuComputeEncoder
// ============================================================================

/// Safe wrapper around `MTLComputeCommandEncoder`.
///
/// Provides typed buffer/texture binding. Consumes `self` on `end_encoding()`
/// to prevent use after the encoding pass has ended.
pub struct GpuComputeEncoder {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl GpuComputeEncoder {
    /// Set the compute pipeline state.
    pub fn set_pipeline(&self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.encoder.setComputePipelineState(pipeline);
    }

    /// Bind a typed buffer at the given index.
    pub fn set_buffer<T: Copy>(&self, buffer: &GpuBuffer<T>, index: usize) {
        unsafe {
            self.encoder
                .setBuffer_offset_atIndex(Some(buffer.metal_buffer()), 0, index);
        }
    }

    /// Bind a raw `MTLBuffer` at the given index (for buffers not wrapped in `GpuBuffer`).
    pub fn set_raw_buffer(&self, buffer: &ProtocolObject<dyn MTLBuffer>, index: usize) {
        unsafe {
            self.encoder
                .setBuffer_offset_atIndex(Some(buffer), 0, index);
        }
    }

    /// Bind a texture at the given index.
    pub fn set_texture(&self, texture: &GpuTexture, index: usize) {
        unsafe {
            self.encoder
                .setTexture_atIndex(Some(texture.metal_texture()), index);
        }
    }

    /// Bind a raw `MTLTexture` at the given index.
    pub fn set_raw_texture(&self, texture: &ProtocolObject<dyn MTLTexture>, index: usize) {
        unsafe {
            self.encoder.setTexture_atIndex(Some(texture), index);
        }
    }

    /// Bind an acceleration structure at the given buffer index.
    pub fn set_accel_structure(
        &self,
        accel: &ProtocolObject<dyn MTLAccelerationStructure>,
        buffer_index: usize,
    ) {
        unsafe {
            self.encoder
                .setAccelerationStructure_atBufferIndex(Some(accel), buffer_index);
        }
    }

    /// Bind a sampler at the given index.
    pub fn set_sampler(&self, sampler: &ProtocolObject<dyn MTLSamplerState>, index: usize) {
        unsafe {
            self.encoder.setSamplerState_atIndex(Some(sampler), index);
        }
    }

    /// Dispatch threadgroups.
    pub fn dispatch(
        &self,
        threadgroups: [usize; 3],
        threads_per_group: [usize; 3],
    ) {
        let groups = MTLSize {
            width: threadgroups[0],
            height: threadgroups[1],
            depth: threadgroups[2],
        };
        let threads = MTLSize {
            width: threads_per_group[0],
            height: threads_per_group[1],
            depth: threads_per_group[2],
        };
        self.encoder
            .dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    }

    /// End encoding. Consumes `self` — the encoder cannot be used after this.
    pub fn end_encoding(self) {
        self.encoder.endEncoding();
    }

    /// Access the underlying `MTLComputeCommandEncoder`.
    pub fn metal_encoder(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
        &self.encoder
    }
}

// ============================================================================
// GpuRenderEncoder
// ============================================================================

/// Safe wrapper around `MTLRenderCommandEncoder`.
///
/// Consumes `self` on `end_encoding()` to prevent use after the pass has ended.
pub struct GpuRenderEncoder {
    encoder: Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
}

impl GpuRenderEncoder {
    /// Set the viewport.
    pub fn set_viewport(&self, x: f64, y: f64, width: f64, height: f64) {
        self.encoder.setViewport(MTLViewport {
            originX: x,
            originY: y,
            width,
            height,
            znear: 0.0,
            zfar: 1.0,
        });
    }

    /// Set the scissor rectangle.
    pub fn set_scissor(&self, x: usize, y: usize, width: usize, height: usize) {
        self.encoder.setScissorRect(MTLScissorRect {
            x,
            y,
            width,
            height,
        });
    }

    /// Set the render pipeline state.
    pub fn set_pipeline(&self, pipeline: &ProtocolObject<dyn MTLRenderPipelineState>) {
        self.encoder.setRenderPipelineState(pipeline);
    }

    /// Bind a texture to the fragment shader at the given index.
    pub fn set_fragment_texture(&self, texture: &GpuTexture, index: usize) {
        unsafe {
            self.encoder
                .setFragmentTexture_atIndex(Some(texture.metal_texture()), index);
        }
    }

    /// Bind a raw `MTLTexture` to the fragment shader at the given index.
    pub fn set_fragment_raw_texture(
        &self,
        texture: &ProtocolObject<dyn MTLTexture>,
        index: usize,
    ) {
        unsafe {
            self.encoder
                .setFragmentTexture_atIndex(Some(texture), index);
        }
    }

    /// Bind a sampler to the fragment shader at the given index.
    pub fn set_fragment_sampler(
        &self,
        sampler: &ProtocolObject<dyn MTLSamplerState>,
        index: usize,
    ) {
        unsafe {
            self.encoder
                .setFragmentSamplerState_atIndex(Some(sampler), index);
        }
    }

    /// Bind a typed buffer to the vertex shader at the given index.
    pub fn set_vertex_buffer<T: Copy>(&self, buffer: &GpuBuffer<T>, index: usize) {
        unsafe {
            self.encoder
                .setVertexBuffer_offset_atIndex(Some(buffer.metal_buffer()), 0, index);
        }
    }

    /// Bind a raw `MTLBuffer` to the vertex shader at the given index.
    pub fn set_vertex_raw_buffer(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        index: usize,
    ) {
        unsafe {
            self.encoder
                .setVertexBuffer_offset_atIndex(Some(buffer), 0, index);
        }
    }

    /// Draw primitives.
    pub fn draw_primitives(
        &self,
        primitive_type: MTLPrimitiveType,
        vertex_start: usize,
        vertex_count: usize,
    ) {
        unsafe {
            self.encoder
                .drawPrimitives_vertexStart_vertexCount(primitive_type, vertex_start, vertex_count);
        }
    }

    /// End encoding. Consumes `self` — the encoder cannot be used after this.
    pub fn end_encoding(self) {
        self.encoder.endEncoding();
    }

    /// Access the underlying `MTLRenderCommandEncoder`.
    pub fn metal_encoder(&self) -> &ProtocolObject<dyn MTLRenderCommandEncoder> {
        &self.encoder
    }
}

// ============================================================================
// GpuAccelEncoder
// ============================================================================

/// Safe wrapper around `MTLAccelerationStructureCommandEncoder`.
///
/// Consumes `self` on `end_encoding()`.
pub struct GpuAccelEncoder {
    encoder: Retained<ProtocolObject<dyn MTLAccelerationStructureCommandEncoder>>,
}

impl GpuAccelEncoder {
    /// Build an acceleration structure.
    pub fn build(
        &self,
        accel: &ProtocolObject<dyn MTLAccelerationStructure>,
        descriptor: &MTLAccelerationStructureDescriptor,
        scratch: &ProtocolObject<dyn MTLBuffer>,
    ) {
        self.encoder
            .buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                accel,
                descriptor,
                scratch,
                0,
            );
    }

    /// Refit an acceleration structure in-place (update transforms without full rebuild).
    pub fn refit(
        &self,
        source: &ProtocolObject<dyn MTLAccelerationStructure>,
        descriptor: &MTLAccelerationStructureDescriptor,
        scratch: &ProtocolObject<dyn MTLBuffer>,
    ) {
        unsafe {
            self.encoder
                .refitAccelerationStructure_descriptor_destination_scratchBuffer_scratchBufferOffset(
                    source,
                    descriptor,
                    None, // in-place refit
                    Some(scratch),
                    0,
                );
        }
    }

    /// Write the compacted size of an acceleration structure to a buffer.
    pub fn write_compacted_size(
        &self,
        accel: &ProtocolObject<dyn MTLAccelerationStructure>,
        to_buffer: &ProtocolObject<dyn MTLBuffer>,
    ) {
        self.encoder
            .writeCompactedAccelerationStructureSize_toBuffer_offset(accel, to_buffer, 0);
    }

    /// Copy and compact an acceleration structure.
    pub fn copy_and_compact(
        &self,
        source: &ProtocolObject<dyn MTLAccelerationStructure>,
        destination: &ProtocolObject<dyn MTLAccelerationStructure>,
    ) {
        self.encoder
            .copyAndCompactAccelerationStructure_toAccelerationStructure(source, destination);
    }

    /// End encoding. Consumes `self`.
    pub fn end_encoding(self) {
        self.encoder.endEncoding();
    }
}
