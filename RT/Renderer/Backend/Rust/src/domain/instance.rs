//! Scene instance types.
//!
//! Represents the different kinds of objects that can be submitted for
//! raytracing in a frame: level geometry, billboards, and rods.

use super::color::RGBA8;
use super::indices::BitmapIndex;
use crate::types::{Camera, Mat4, ObjectType, ResourceHandle, Vec2, Vec3};

/// Validated mesh handle — wraps a `ResourceHandle` that is known to exist
/// in the `MeshSlotMap` at the time of creation.
#[derive(Clone, Copy, Debug)]
pub struct MeshHandle(ResourceHandle);

impl MeshHandle {
    /// Create from a `ResourceHandle` that has been validated against the mesh slotmap.
    /// Callers must ensure the handle exists in the slotmap.
    pub fn from_validated(handle: ResourceHandle) -> Self {
        debug_assert!(handle.is_valid(), "MeshHandle created from invalid ResourceHandle");
        Self(handle)
    }

    /// Try to create from a `ResourceHandle`, returning None if it's NULL.
    pub fn try_new(handle: ResourceHandle) -> Option<Self> {
        if handle.is_valid() {
            Some(Self(handle))
        } else {
            None
        }
    }

    pub fn raw(self) -> ResourceHandle {
        self.0
    }
}

/// A scene instance queued for rendering this frame.
#[derive(Clone, Debug)]
pub enum SceneInstance {
    /// Level mesh — uses material edges from the triangle data.
    LevelMesh {
        mesh: MeshHandle,
        transform: Mat4,
        color: RGBA8,
        material_override: Option<BitmapIndex>,
        object_type: ObjectType,
    },
    /// Camera-facing billboard sprite.
    Billboard {
        mesh: MeshHandle,
        transform: Mat4,
        prev_transform: Mat4,
        color: RGBA8,
        material: BitmapIndex,
    },
    /// Rod connecting two points (laser beams, etc).
    Rod {
        mesh: MeshHandle,
        transform: Mat4,
        color: RGBA8,
        material: BitmapIndex,
    },
}

impl SceneInstance {
    /// Create a camera-facing billboard instance.
    ///
    /// Computes a transform that orients a unit quad to face the camera,
    /// scaled by `dim` and positioned at `pos`.
    pub fn billboard(
        camera: &Camera,
        mesh: MeshHandle,
        material: BitmapIndex,
        color: RGBA8,
        dim: Vec2,
        pos: Vec3,
        prev_pos: Vec3,
    ) -> Self {
        let basis = billboard_basis(camera, dim);
        let transform = Mat4::from_translation(pos).mul(basis);
        let prev_transform = Mat4::from_translation(prev_pos).mul(basis);

        Self::Billboard {
            mesh,
            transform,
            prev_transform,
            color,
            material,
        }
    }

    /// Create a rod instance connecting two points.
    ///
    /// Computes a transform that orients a unit quad along the axis from
    /// `bot_p` to `top_p`, with the given `width`.
    pub fn rod(
        camera: &Camera,
        mesh: MeshHandle,
        material: BitmapIndex,
        bot_p: Vec3,
        top_p: Vec3,
        width: f32,
    ) -> Self {
        let transform = rod_transform(camera, bot_p, top_p, width);

        Self::Rod {
            mesh,
            transform,
            color: RGBA8::WHITE,
            material,
        }
    }

    pub fn mesh(&self) -> MeshHandle {
        match self {
            Self::LevelMesh { mesh, .. } => *mesh,
            Self::Billboard { mesh, .. } => *mesh,
            Self::Rod { mesh, .. } => *mesh,
        }
    }

    pub fn transform(&self) -> &Mat4 {
        match self {
            Self::LevelMesh { transform, .. } => transform,
            Self::Billboard { transform, .. } => transform,
            Self::Rod { transform, .. } => transform,
        }
    }

    pub fn color(&self) -> RGBA8 {
        match self {
            Self::LevelMesh { color, .. } => *color,
            Self::Billboard { color, .. } => *color,
            Self::Rod { color, .. } => *color,
        }
    }

    /// The material override index for this instance (billboards/rods always,
    /// level meshes only when explicitly overridden).
    pub fn material_override(&self) -> Option<BitmapIndex> {
        match self {
            Self::LevelMesh { material_override, .. } => *material_override,
            Self::Billboard { material, .. } => Some(*material),
            Self::Rod { material, .. } => Some(*material),
        }
    }

    pub fn object_type(&self) -> ObjectType {
        match self {
            Self::LevelMesh { object_type, .. } => *object_type,
            Self::Billboard { .. } | Self::Rod { .. } => ObjectType::None,
        }
    }
}

/// Compute a camera-facing basis matrix scaled by billboard dimensions.
fn billboard_basis(camera: &Camera, dim: Vec2) -> Mat4 {
    let x = camera.right.scale(dim.x);
    let y = camera.up.scale(dim.y);
    let z = camera.forward.negate();
    Mat4::from_basis_vectors(x, y, z)
}

/// Compute a rod transform oriented along the axis from bot_p to top_p.
fn rod_transform(camera: &Camera, bot_p: Vec3, top_p: Vec3, width: f32) -> Mat4 {
    let z_cam = camera.forward.negate();
    let y = top_p.sub(bot_p).scale(0.5);
    let x = y.cross(z_cam).normalize().scale(width);
    let z = x.cross(y).normalize();
    let pos = bot_p.add(top_p).scale(0.5);
    Mat4::from_translation(pos).mul(Mat4::from_basis_vectors(x, y, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> Camera {
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

    #[test]
    fn billboard_transform_position() {
        let cam = test_camera();
        let mesh = MeshHandle::from_validated(ResourceHandle { index: 1, generation: 1 });
        let mat = BitmapIndex::new(10).unwrap();
        let pos = Vec3 { x: 5.0, y: 10.0, z: -20.0 };
        let prev_pos = Vec3 { x: 4.0, y: 10.0, z: -20.0 };
        let dim = Vec2 { x: 2.0, y: 3.0 };

        let inst = SceneInstance::billboard(&cam, mesh, mat, RGBA8::WHITE, dim, pos, prev_pos);

        // Translation should be in column 3 (row-major e[row][col])
        let t = inst.transform();
        assert!((t.e[0][3] - 5.0).abs() < 0.001);
        assert!((t.e[1][3] - 10.0).abs() < 0.001);
        assert!((t.e[2][3] - (-20.0)).abs() < 0.001);

        // Basis X should be camera.right * dim.x = (2, 0, 0)
        assert!((t.e[0][0] - 2.0).abs() < 0.001);
        // Basis Y should be camera.up * dim.y = (0, 3, 0)
        assert!((t.e[1][1] - 3.0).abs() < 0.001);
        // Basis Z should be -camera.forward = (0, 0, 1)
        assert!((t.e[2][2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn billboard_preserves_material() {
        let cam = test_camera();
        let mesh = MeshHandle::from_validated(ResourceHandle { index: 1, generation: 1 });
        let mat = BitmapIndex::new(42).unwrap();
        let pos = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

        let inst = SceneInstance::billboard(
            &cam, mesh, mat, RGBA8::WHITE,
            Vec2 { x: 1.0, y: 1.0 }, pos, pos,
        );
        assert_eq!(inst.material_override().unwrap().as_u16(), 42);
    }

    #[test]
    fn rod_transform_midpoint() {
        let cam = test_camera();
        let mesh = MeshHandle::from_validated(ResourceHandle { index: 1, generation: 1 });
        let mat = BitmapIndex::new(5).unwrap();
        let bot = Vec3 { x: 0.0, y: 0.0, z: -10.0 };
        let top = Vec3 { x: 0.0, y: 10.0, z: -10.0 };

        let inst = SceneInstance::rod(&cam, mesh, mat, bot, top, 0.5);

        // Position should be midpoint: (0, 5, -10)
        let t = inst.transform();
        assert!((t.e[0][3] - 0.0).abs() < 0.001);
        assert!((t.e[1][3] - 5.0).abs() < 0.001);
        assert!((t.e[2][3] - (-10.0)).abs() < 0.001);
    }

    #[test]
    fn level_mesh_no_override() {
        let mesh = MeshHandle::from_validated(ResourceHandle { index: 1, generation: 1 });
        let inst = SceneInstance::LevelMesh {
            mesh,
            transform: Mat4::identity(),
            color: RGBA8::WHITE,
            material_override: None,
            object_type: ObjectType::None,
        };
        assert!(inst.material_override().is_none());
    }

    #[test]
    fn level_mesh_with_override() {
        let mesh = MeshHandle::from_validated(ResourceHandle { index: 1, generation: 1 });
        let mat = BitmapIndex::new(99).unwrap();
        let inst = SceneInstance::LevelMesh {
            mesh,
            transform: Mat4::identity(),
            color: RGBA8::WHITE,
            material_override: Some(mat),
            object_type: ObjectType::None,
        };
        assert_eq!(inst.material_override().unwrap().as_u16(), 99);
    }
}
