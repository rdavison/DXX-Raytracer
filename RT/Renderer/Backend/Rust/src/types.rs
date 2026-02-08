//! C-compatible type definitions matching ApiTypes.h and Renderer.h
//!
//! All structs use #[repr(C)] to match the C layout exactly.
//! Field names match the C originals for easy cross-referencing.

use std::ffi::c_void;

// ============================================================================
// Math types (ApiTypes.h)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Mat4 {
    pub e: [[f32; 4]; 4],
}

impl Vec3 {
    pub fn scale(self, s: f32) -> Self {
        Vec3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn add(self, other: Self) -> Self {
        Vec3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(self, other: Self) -> Self {
        Vec3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn cross(self, other: Self) -> Self {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            return self;
        }
        self.scale(1.0 / len)
    }

    pub fn negate(self) -> Self {
        Vec3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mat4 {
    pub fn identity() -> Self {
        let mut m = Mat4 { e: [[0.0; 4]; 4] };
        m.e[0][0] = 1.0;
        m.e[1][1] = 1.0;
        m.e[2][2] = 1.0;
        m.e[3][3] = 1.0;
        m
    }

    /// General 4×4 matrix inverse via cofactor expansion.
    /// Returns identity if the matrix is singular.
    pub fn inverse(&self) -> Self {
        let m = &self.e;
        let mut inv = [0.0f32; 16];
        let flat = [
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3],
        ];

        inv[0] = flat[5]*flat[10]*flat[15] - flat[5]*flat[11]*flat[14]
               - flat[9]*flat[6]*flat[15] + flat[9]*flat[7]*flat[14]
               + flat[13]*flat[6]*flat[11] - flat[13]*flat[7]*flat[10];

        inv[4] = -flat[4]*flat[10]*flat[15] + flat[4]*flat[11]*flat[14]
               + flat[8]*flat[6]*flat[15] - flat[8]*flat[7]*flat[14]
               - flat[12]*flat[6]*flat[11] + flat[12]*flat[7]*flat[10];

        inv[8] = flat[4]*flat[9]*flat[15] - flat[4]*flat[11]*flat[13]
               - flat[8]*flat[5]*flat[15] + flat[8]*flat[7]*flat[13]
               + flat[12]*flat[5]*flat[11] - flat[12]*flat[7]*flat[9];

        inv[12] = -flat[4]*flat[9]*flat[14] + flat[4]*flat[10]*flat[13]
                + flat[8]*flat[5]*flat[14] - flat[8]*flat[6]*flat[13]
                - flat[12]*flat[5]*flat[10] + flat[12]*flat[6]*flat[9];

        inv[1] = -flat[1]*flat[10]*flat[15] + flat[1]*flat[11]*flat[14]
               + flat[9]*flat[2]*flat[15] - flat[9]*flat[3]*flat[14]
               - flat[13]*flat[2]*flat[11] + flat[13]*flat[3]*flat[10];

        inv[5] = flat[0]*flat[10]*flat[15] - flat[0]*flat[11]*flat[14]
               - flat[8]*flat[2]*flat[15] + flat[8]*flat[3]*flat[14]
               + flat[12]*flat[2]*flat[11] - flat[12]*flat[3]*flat[10];

        inv[9] = -flat[0]*flat[9]*flat[15] + flat[0]*flat[11]*flat[13]
               + flat[8]*flat[1]*flat[15] - flat[8]*flat[3]*flat[13]
               - flat[12]*flat[1]*flat[11] + flat[12]*flat[3]*flat[9];

        inv[13] = flat[0]*flat[9]*flat[14] - flat[0]*flat[10]*flat[13]
                - flat[8]*flat[1]*flat[14] + flat[8]*flat[2]*flat[13]
                + flat[12]*flat[1]*flat[10] - flat[12]*flat[2]*flat[9];

        inv[2] = flat[1]*flat[6]*flat[15] - flat[1]*flat[7]*flat[14]
               - flat[5]*flat[2]*flat[15] + flat[5]*flat[3]*flat[14]
               + flat[13]*flat[2]*flat[7] - flat[13]*flat[3]*flat[6];

        inv[6] = -flat[0]*flat[6]*flat[15] + flat[0]*flat[7]*flat[14]
               + flat[4]*flat[2]*flat[15] - flat[4]*flat[3]*flat[14]
               - flat[12]*flat[2]*flat[7] + flat[12]*flat[3]*flat[6];

        inv[10] = flat[0]*flat[5]*flat[15] - flat[0]*flat[7]*flat[13]
                - flat[4]*flat[1]*flat[15] + flat[4]*flat[3]*flat[13]
                + flat[12]*flat[1]*flat[7] - flat[12]*flat[3]*flat[5];

        inv[14] = -flat[0]*flat[5]*flat[14] + flat[0]*flat[6]*flat[13]
                + flat[4]*flat[1]*flat[14] - flat[4]*flat[2]*flat[13]
                - flat[12]*flat[1]*flat[6] + flat[12]*flat[2]*flat[5];

        inv[3] = -flat[1]*flat[6]*flat[11] + flat[1]*flat[7]*flat[10]
               + flat[5]*flat[2]*flat[11] - flat[5]*flat[3]*flat[10]
               - flat[9]*flat[2]*flat[7] + flat[9]*flat[3]*flat[6];

        inv[7] = flat[0]*flat[6]*flat[11] - flat[0]*flat[7]*flat[10]
               - flat[4]*flat[2]*flat[11] + flat[4]*flat[3]*flat[10]
               + flat[8]*flat[2]*flat[7] - flat[8]*flat[3]*flat[6];

        inv[11] = -flat[0]*flat[5]*flat[11] + flat[0]*flat[7]*flat[9]
                + flat[4]*flat[1]*flat[11] - flat[4]*flat[3]*flat[9]
                - flat[8]*flat[1]*flat[7] + flat[8]*flat[3]*flat[5];

        inv[15] = flat[0]*flat[5]*flat[10] - flat[0]*flat[6]*flat[9]
                - flat[4]*flat[1]*flat[10] + flat[4]*flat[2]*flat[9]
                + flat[8]*flat[1]*flat[6] - flat[8]*flat[2]*flat[5];

        let det = flat[0]*inv[0] + flat[1]*inv[4] + flat[2]*inv[8] + flat[3]*inv[12];
        if det.abs() < 1e-12 {
            return Self::identity();
        }
        let inv_det = 1.0 / det;

        let mut result = Mat4 { e: [[0.0; 4]; 4] };
        for r in 0..4 {
            for c in 0..4 {
                result.e[r][c] = inv[r * 4 + c] * inv_det;
            }
        }
        result
    }

    /// Build a 4x4 matrix from three basis vectors (placed as columns of the upper-left 3x3).
    pub fn from_basis_vectors(x: Vec3, y: Vec3, z: Vec3) -> Self {
        let mut m = Mat4 { e: [[0.0; 4]; 4] };
        m.e[0][0] = x.x; m.e[0][1] = y.x; m.e[0][2] = z.x;
        m.e[1][0] = x.y; m.e[1][1] = y.y; m.e[1][2] = z.y;
        m.e[2][0] = x.z; m.e[2][1] = y.z; m.e[2][2] = z.z;
        m.e[3][3] = 1.0;
        m
    }

    /// Build a translation matrix.
    pub fn from_translation(t: Vec3) -> Self {
        let mut m = Self::identity();
        m.e[0][3] = t.x;
        m.e[1][3] = t.y;
        m.e[2][3] = t.z;
        m
    }

    /// 4x4 matrix multiplication.
    pub fn mul(self, other: Self) -> Self {
        let mut result = Mat4 { e: [[0.0; 4]; 4] };
        for r in 0..4 {
            for c in 0..4 {
                result.e[r][c] = self.e[r][0] * other.e[0][c]
                    + self.e[r][1] * other.e[1][c]
                    + self.e[r][2] * other.e[2][c]
                    + self.e[r][3] * other.e[3][c];
            }
        }
        result
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Mat34 {
    pub e: [[f32; 4]; 3],
}

// ============================================================================
// Resource handle (ApiTypes.h)
// ============================================================================

/// Matches C `RT_ResourceHandle` which contains a `union { ...; uint64_t value; }`,
/// giving it 8-byte alignment. Without `align(8)`, Rust would use 4-byte alignment
/// (two u32 fields), causing struct layout mismatches in any containing struct
/// (e.g. RenderMeshParams fields shift by 4 bytes, garbling mesh handles).
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct ResourceHandle {
    pub index: u32,
    pub generation: u32,
}

impl ResourceHandle {
    pub const NULL: Self = ResourceHandle {
        index: 0,
        generation: 0,
    };

    pub fn is_valid(&self) -> bool {
        self.index != 0
    }
}

impl Default for ResourceHandle {
    fn default() -> Self {
        Self::NULL
    }
}

// ============================================================================
// Image (ApiTypes.h)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Image {
    pub format: u32, // RT_TextureFormat enum
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub mip_count: u32,
    pub pixels: *mut c_void,
}

// ============================================================================
// Raster vertices (ApiTypes.h)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RasterTriVertex {
    pub pos: Vec3,
    pub uv: Vec2,
    pub color: Vec4,
    pub texture_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RasterLineVertex {
    pub pos: Vec3,
    pub color: Vec4,
}

// ============================================================================
// Light (ApiTypes.h)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub kind: u8,
    pub spot_angle: u8,
    pub spot_softness: u8,
    pub spot_vignette: u8,
    pub emission: u32,
    pub transform: Mat34,
}

// ============================================================================
// Renderer types (Renderer.h)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MaterialEdge {
    pub mat1: u16,
    pub mat2: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub vfov: f32,
    pub near_plane: f32,
    pub far_plane: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            up: Vec3 { x: 0.0, y: 1.0, z: 0.0 },
            forward: Vec3 { x: 0.0, y: 0.0, z: -1.0 },
            right: Vec3 { x: 1.0, y: 0.0, z: 0.0 },
            vfov: 90.0,
            near_plane: 0.1,
            far_plane: 1000.0,
        }
    }
}

#[repr(C)]
pub struct RendererInitParams {
    pub arena: *mut c_void,
    pub window_handle: *mut c_void,
}

#[repr(C)]
pub struct SceneSettings {
    pub camera: *mut Camera,
    pub render_width_override: u32,
    pub render_height_override: u32,
    pub render_blit: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub positions: [Vec3; 3],
    pub normals: [Vec3; 3],
    pub tangents: [Vec4; 3],
    pub uvs: [Vec2; 3],
    pub color: u32,
    pub material_edge_index: u32,
}

#[repr(C)]
pub struct UploadTextureParams {
    pub image: Image,
    pub flags: u8,
    pub name: *const i8,
}

#[repr(C)]
pub struct UploadMeshParams {
    pub triangle_count: usize,
    pub triangles: *mut Triangle,
    pub name: *const i8,
}

#[repr(C)]
pub struct RenderMeshParams {
    pub key_signature: i32,
    pub key_submodel_index: i32,
    pub flags: u32,
    pub mesh_handle: ResourceHandle,
    pub transform: *const Mat4,
    pub prev_transform: *const Mat4,
    pub color: u32,
    pub material_override: u16,
    pub object_type: u8,
}

#[repr(C)]
pub struct RasterTrianglesParams {
    pub texture_handle: ResourceHandle,
    pub vertices: *mut RasterTriVertex,
    pub num_vertices: u32,
}

#[repr(C)]
pub struct DoRendererDebugMenuParams {
    pub ui_has_cursor_focus: bool,
}

#[repr(C)]
pub struct RendererIO {
    pub scene_transition: bool,
    pub debug_line_depth_enabled: bool,
    pub screen_overlay_color: Vec4,
    pub delta_time: f32,
    pub debug_render_mode: i32,
    pub config: *mut c_void,
    pub frame_frozen: bool,
}

impl Default for RendererIO {
    fn default() -> Self {
        Self {
            scene_transition: false,
            debug_line_depth_enabled: false,
            screen_overlay_color: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
            delta_time: 0.0,
            debug_render_mode: 0,
            config: std::ptr::null_mut(),
            frame_frozen: false,
        }
    }
}

// ============================================================================
// GPU Material (uploaded to GPU for shader texture lookup)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GPUMaterial {
    pub albedo_index: u32,
    pub flags: u32,
    pub emissive_factor: u32,
    pub _pad: u32,
}

// ============================================================================
// Material (Renderer.h)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub textures: [ResourceHandle; 6], // RT_MaterialTextureSlot_COUNT = 6
    pub metalness: f32,
    pub roughness: f32,
    pub emissive_color: Vec3,
    pub emissive_strength: f32,
    pub flags: u32,
    pub always_load_texture: bool,
    pub texture_load_state: u32,
    pub texture_load_state_next: u32,
}

// ============================================================================
// Layout assertions — catch C/Rust struct mismatches at compile time
// ============================================================================

const _: () = {
    assert!(std::mem::size_of::<ResourceHandle>() == 8);
    assert!(std::mem::align_of::<ResourceHandle>() == 8);

    // RenderMeshParams field offsets must match C layout:
    //   key(8) + flags(4) + pad(4) + mesh_handle(8) + transform(8) + prev_transform(8) + color(4) + material_override(2) + pad(2) = 48
    assert!(std::mem::size_of::<RenderMeshParams>() == 48);
};
