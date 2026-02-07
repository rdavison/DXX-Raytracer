//! Rendering pipeline orchestration.
//!
//! Contains raytracing dispatch, raster batch encoding, and texture remap logic.
//! Works directly with domain types from `domain/`.

pub mod raster;
pub mod raytrace;
pub mod texture_remap;
