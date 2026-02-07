//! Safe wrappers around Metal API calls.
//!
//! All `unsafe` Metal FFI is contained within this module. Code outside `gpu/`
//! interacts with Metal exclusively through these safe abstractions.

pub mod accel;
pub mod buffer;
pub mod device;
pub mod encoder;
pub mod shader_types;
pub mod texture;

pub use accel::GpuAccelStructure;
pub use buffer::GpuBuffer;
pub use device::GpuDevice;
pub use encoder::{GpuCommandBuffer, GpuComputeEncoder, GpuRenderEncoder};
pub use shader_types::{GpuInstance, GpuLight, GpuMaterial, GpuMaterialEdge, GpuSceneConstants};
pub use texture::GpuTexture;
