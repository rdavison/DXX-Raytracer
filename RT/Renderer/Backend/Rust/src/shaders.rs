//! Metal shader source and compilation.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLLibrary};

/// Metal shading language source for the 2D raster pipeline.
///
/// Triangle shaders: passthrough vertex (pos3, uv2, color4, texture_index uint),
/// fragment samples texture and multiplies by vertex color.
///
/// Line shaders: passthrough vertex (pos3, color4), fragment returns color directly.
pub const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- Triangle shaders ----

struct TriVertexIn {
    float3 position [[attribute(0)]];
    float2 uv       [[attribute(1)]];
    float4 color    [[attribute(2)]];
    uint   texIdx   [[attribute(3)]];
};

struct TriVertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex TriVertexOut tri_vertex(TriVertexIn in [[stage_in]]) {
    TriVertexOut out;
    out.position = float4(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

fragment float4 tri_fragment(TriVertexOut in [[stage_in]],
                             texture2d<float> tex [[texture(0)]],
                             texture2d<float> bloom_tex [[texture(1)]],
                             sampler samp [[sampler(0)]]) {
    float4 texColor = tex.sample(samp, in.uv);
    float3 bloom = bloom_tex.sample(samp, in.uv).rgb;
    return float4(texColor.rgb * (1.0 + bloom * 0.35), texColor.a) * in.color;
}

// ---- Line shaders ----

struct LineVertexIn {
    float3 position [[attribute(0)]];
    float4 color    [[attribute(1)]];
};

struct LineVertexOut {
    float4 position [[position]];
    float4 color;
};

vertex LineVertexOut line_vertex(LineVertexIn in [[stage_in]]) {
    LineVertexOut out;
    out.position = float4(in.position, 1.0);
    out.color = in.color;
    return out;
}

fragment float4 line_fragment(LineVertexOut in [[stage_in]]) {
    return in.color;
}
"#;

/// Compile Metal shader source into a library.
pub fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLLibrary>> {
    let source = NSString::from_str(SHADER_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("[Rust Metal] Failed to compile shader library");
    eprintln!("[Rust Metal] Shader library compiled successfully");
    library
}
