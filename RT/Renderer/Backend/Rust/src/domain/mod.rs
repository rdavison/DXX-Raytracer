//! Domain types for the raytracer.
//!
//! These are pure safe Rust types with no `unsafe` code and no Metal dependencies.
//! Raw C data is parsed into these types at the FFI boundary in `lib.rs`.
//! Once parsed, the types make illegal states unrepresentable.

pub mod color;
pub mod indices;
pub mod instance;
pub mod light;
pub mod material;
pub mod scene;
pub mod triangle_ref;

pub use color::RGBA8;
pub use indices::{BitmapIndex, TextureSlotId, TmapNum};
pub use instance::{MeshHandle, SceneInstance};
pub use light::ParsedLight;
pub use material::{OverlayMaterial, ParsedMaterialEdge};
pub use scene::SceneConfig;
pub use triangle_ref::TriangleMaterialRef;
