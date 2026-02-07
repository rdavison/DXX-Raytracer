//! Raster batch accumulation and GPU encoding for 2D UI rendering.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::texture::TextureSlotMap;
use crate::types::{RasterLineVertex, RasterTriVertex, ResourceHandle};

/// A batch of triangles sharing the same texture.
pub struct RasterBatch {
    pub texture: ResourceHandle,
    pub vertices: Vec<RasterTriVertex>,
}

/// Push a triangle batch into the accumulator.
pub fn push_tri_batch(
    batches: &mut Vec<RasterBatch>,
    texture: ResourceHandle,
    vertices: &[RasterTriVertex],
) {
    batches.push(RasterBatch {
        texture,
        vertices: vertices.to_vec(),
    });
}

/// Push line vertices into the accumulator.
pub fn push_lines(line_buf: &mut Vec<RasterLineVertex>, vertices: &[RasterLineVertex]) {
    line_buf.extend_from_slice(vertices);
}

/// Encode all accumulated raster batches into the given render command encoder.
///
/// Draws triangle batches first (each with its own texture binding), then lines.
/// Uses transient vertex buffers created from the device.
pub fn encode_batches(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    device: &ProtocolObject<dyn MTLDevice>,
    target_w: f64,
    target_h: f64,
    batches: &[RasterBatch],
    lines: &[RasterLineVertex],
    slotmap: &TextureSlotMap,
    white_tex: &ProtocolObject<dyn MTLTexture>,
    tri_pipeline: &ProtocolObject<dyn MTLRenderPipelineState>,
    line_pipeline: &ProtocolObject<dyn MTLRenderPipelineState>,
    sampler: &ProtocolObject<dyn MTLSamplerState>,
) {
    // Set viewport to full drawable size
    encoder.setViewport(MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: target_w,
        height: target_h,
        znear: 0.0,
        zfar: 1.0,
    });

    // Set scissor to full target
    encoder.setScissorRect(MTLScissorRect {
        x: 0,
        y: 0,
        width: target_w as usize,
        height: target_h as usize,
    });

    // ---- Triangle batches ----
    if !batches.is_empty() {
        encoder.setRenderPipelineState(tri_pipeline);
        unsafe {
            encoder.setFragmentSamplerState_atIndex(Some(sampler), 0);
        }

        for batch in batches {
            if batch.vertices.is_empty() {
                continue;
            }

            // Resolve texture: look up in slotmap, fall back to white
            let tex = if batch.texture.is_valid() {
                slotmap.find(batch.texture).unwrap_or(white_tex)
            } else {
                white_tex
            };

            unsafe {
                encoder.setFragmentTexture_atIndex(Some(tex), 0);
            }

            // Create transient vertex buffer
            let byte_len = batch.vertices.len() * std::mem::size_of::<RasterTriVertex>();
            let vb = unsafe {
                device.newBufferWithBytes_length_options(
                    NonNull::new_unchecked(batch.vertices.as_ptr() as *mut c_void),
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
            };

            if let Some(ref buf) = vb {
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(Some(buf), 0, 0);
                    encoder.drawPrimitives_vertexStart_vertexCount(
                        MTLPrimitiveType::Triangle,
                        0,
                        batch.vertices.len(),
                    );
                }
            }
        }
    }

    // ---- Lines ----
    if !lines.is_empty() {
        encoder.setRenderPipelineState(line_pipeline);

        let byte_len = lines.len() * std::mem::size_of::<RasterLineVertex>();
        let vb = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(lines.as_ptr() as *mut c_void),
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        };

        if let Some(ref buf) = vb {
            unsafe {
                encoder.setVertexBuffer_offset_atIndex(Some(buf), 0, 0);
                encoder.drawPrimitives_vertexStart_vertexCount(
                    MTLPrimitiveType::Line,
                    0,
                    lines.len(),
                );
            }
        }
    }
}
