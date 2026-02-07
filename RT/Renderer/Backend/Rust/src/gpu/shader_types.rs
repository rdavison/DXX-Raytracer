//! GPU-compatible struct type aliases.
//!
//! These re-export the existing `#[repr(C)]` structs that match the MSL shader
//! layout. They're the "serialization format" for the GPU — pure data, no behavior.
//!
//! Domain types from `domain/` convert INTO these before GPU upload.
//! In a future phase, these may become standalone definitions and the originals
//! in `types.rs`/`raytrace.rs` can be removed.

/// GPU instance data (160 bytes, matches shader `Instance` struct).
pub type GpuInstance = crate::renderer::raytrace::RaytraceInstance;

/// GPU scene constants (112 bytes, matches shader `SceneConstants` struct).
pub type GpuSceneConstants = crate::renderer::raytrace::RaytraceSceneConstants;

/// GPU triangle data (matches shader `Triangle` struct).
pub type GpuTriangle = crate::types::Triangle;

/// GPU material edge (4 bytes: mat1 + mat2).
pub type GpuMaterialEdge = crate::types::MaterialEdge;

/// GPU material data (16 bytes: albedo_index + flags + padding).
pub type GpuMaterial = crate::types::GPUMaterial;

/// GPU light data (matches shader `Light` struct).
pub type GpuLight = crate::types::Light;
