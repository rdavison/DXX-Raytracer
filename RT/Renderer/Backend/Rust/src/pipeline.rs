//! Render pipeline and sampler creation for the 2D raster pass.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

/// Create the triangle raster pipeline.
///
/// Vertex layout (stride=40):
///   attr0: Float3  @ offset 0   (position)
///   attr1: Float2  @ offset 12  (uv)
///   attr2: Float4  @ offset 20  (color)
///   attr3: UInt    @ offset 36  (texture_index, unused in shader but part of vertex struct)
///
/// Alpha blend: src=SourceAlpha, dst=OneMinusSourceAlpha, op=Add
pub fn create_tri_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
    let vs_name = NSString::from_str("tri_vertex");
    let fs_name = NSString::from_str("tri_fragment");
    let vs_fn = library
        .newFunctionWithName(&vs_name)
        .expect("[Rust Metal] tri_vertex function not found");
    let fs_fn = library
        .newFunctionWithName(&fs_name)
        .expect("[Rust Metal] tri_fragment function not found");

    // Vertex descriptor
    let vd = MTLVertexDescriptor::new();
    let attrs = vd.attributes();
    let layouts = vd.layouts();

    unsafe {
        // attr0: position Float3 @ offset 0
        let a0 = attrs.objectAtIndexedSubscript(0);
        a0.setFormat(MTLVertexFormat::Float3);
        a0.setOffset(0);
        a0.setBufferIndex(0);

        // attr1: uv Float2 @ offset 12
        let a1 = attrs.objectAtIndexedSubscript(1);
        a1.setFormat(MTLVertexFormat::Float2);
        a1.setOffset(12);
        a1.setBufferIndex(0);

        // attr2: color Float4 @ offset 20
        let a2 = attrs.objectAtIndexedSubscript(2);
        a2.setFormat(MTLVertexFormat::Float4);
        a2.setOffset(20);
        a2.setBufferIndex(0);

        // attr3: texture_index UInt @ offset 36
        let a3 = attrs.objectAtIndexedSubscript(3);
        a3.setFormat(MTLVertexFormat::UInt);
        a3.setOffset(36);
        a3.setBufferIndex(0);

        // layout0: stride 40, per-vertex
        let l0 = layouts.objectAtIndexedSubscript(0);
        l0.setStride(40);
        l0.setStepFunction(MTLVertexStepFunction::PerVertex);
    }

    // Pipeline descriptor
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vs_fn));
    desc.setFragmentFunction(Some(&fs_fn));
    desc.setVertexDescriptor(Some(&vd));

    // Color attachment 0: BGRA8Unorm with alpha blending
    let color_attachments = desc.colorAttachments();
    unsafe {
        let ca0 = color_attachments.objectAtIndexedSubscript(0);
        ca0.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        ca0.setBlendingEnabled(true);
        ca0.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca0.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca0.setRgbBlendOperation(MTLBlendOperation::Add);
        ca0.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        ca0.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca0.setAlphaBlendOperation(MTLBlendOperation::Add);
    }

    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .expect("[Rust Metal] Failed to create triangle pipeline");

    eprintln!("[Rust Metal] Triangle pipeline created");
    pipeline
}

/// Create the line raster pipeline.
///
/// Vertex layout (stride=28):
///   attr0: Float3  @ offset 0   (position)
///   attr1: Float4  @ offset 12  (color)
///
/// Same alpha blend as triangles.
pub fn create_line_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
    let vs_name = NSString::from_str("line_vertex");
    let fs_name = NSString::from_str("line_fragment");
    let vs_fn = library
        .newFunctionWithName(&vs_name)
        .expect("[Rust Metal] line_vertex function not found");
    let fs_fn = library
        .newFunctionWithName(&fs_name)
        .expect("[Rust Metal] line_fragment function not found");

    // Vertex descriptor
    let vd = MTLVertexDescriptor::new();
    let attrs = vd.attributes();
    let layouts = vd.layouts();

    unsafe {
        // attr0: position Float3 @ offset 0
        let a0 = attrs.objectAtIndexedSubscript(0);
        a0.setFormat(MTLVertexFormat::Float3);
        a0.setOffset(0);
        a0.setBufferIndex(0);

        // attr1: color Float4 @ offset 12
        let a1 = attrs.objectAtIndexedSubscript(1);
        a1.setFormat(MTLVertexFormat::Float4);
        a1.setOffset(12);
        a1.setBufferIndex(0);

        // layout0: stride 28, per-vertex
        let l0 = layouts.objectAtIndexedSubscript(0);
        l0.setStride(28);
        l0.setStepFunction(MTLVertexStepFunction::PerVertex);
    }

    // Pipeline descriptor
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vs_fn));
    desc.setFragmentFunction(Some(&fs_fn));
    desc.setVertexDescriptor(Some(&vd));

    // Color attachment 0: BGRA8Unorm with alpha blending
    let color_attachments = desc.colorAttachments();
    unsafe {
        let ca0 = color_attachments.objectAtIndexedSubscript(0);
        ca0.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        ca0.setBlendingEnabled(true);
        ca0.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca0.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca0.setRgbBlendOperation(MTLBlendOperation::Add);
        ca0.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        ca0.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca0.setAlphaBlendOperation(MTLBlendOperation::Add);
    }

    let pipeline = device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .expect("[Rust Metal] Failed to create line pipeline");

    eprintln!("[Rust Metal] Line pipeline created");
    pipeline
}

/// Create a sampler with linear min/mag filtering and clamp-to-edge addressing.
pub fn create_sampler(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLSamplerState>> {
    let desc = MTLSamplerDescriptor::new();
    desc.setMinFilter(MTLSamplerMinMagFilter::Linear);
    desc.setMagFilter(MTLSamplerMinMagFilter::Linear);
    desc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
    desc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);

    let sampler = device
        .newSamplerStateWithDescriptor(&desc)
        .expect("[Rust Metal] Failed to create sampler");

    eprintln!("[Rust Metal] Sampler created");
    sampler
}
