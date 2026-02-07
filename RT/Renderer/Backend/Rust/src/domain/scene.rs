//! Scene configuration for a frame.

use crate::types::Camera;

/// Per-frame scene configuration parsed from `SceneSettings`.
#[derive(Clone, Debug)]
pub struct SceneConfig {
    pub camera: Camera,
    pub render_width: u32,
    pub render_height: u32,
    pub render_blit: bool,
}

impl SceneConfig {
    /// Parse scene settings from raw C values.
    ///
    /// Applies defaults: vfov clamped to >= 1.0, render size falls back to
    /// `current_width`/`current_height` if overrides are zero.
    pub fn new(
        mut camera: Camera,
        width_override: u32,
        height_override: u32,
        render_blit: bool,
        current_width: u32,
        current_height: u32,
    ) -> Self {
        if camera.vfov < 1.0 {
            camera.vfov = 72.0;
        }
        let (render_width, render_height) = if width_override > 0 && height_override > 0 {
            (width_override, height_override)
        } else {
            (current_width, current_height)
        };
        Self {
            camera,
            render_width,
            render_height,
            render_blit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Vec3;

    fn default_camera() -> Camera {
        Camera::default()
    }

    #[test]
    fn vfov_clamped_to_default() {
        let mut cam = default_camera();
        cam.vfov = 0.0;
        let config = SceneConfig::new(cam, 1920, 1080, false, 640, 480);
        assert!((config.camera.vfov - 72.0).abs() < 0.001);
    }

    #[test]
    fn override_dimensions_used() {
        let cam = default_camera();
        let config = SceneConfig::new(cam, 1920, 1080, true, 640, 480);
        assert_eq!(config.render_width, 1920);
        assert_eq!(config.render_height, 1080);
        assert!(config.render_blit);
    }

    #[test]
    fn zero_override_falls_back_to_current() {
        let cam = default_camera();
        let config = SceneConfig::new(cam, 0, 0, false, 640, 480);
        assert_eq!(config.render_width, 640);
        assert_eq!(config.render_height, 480);
    }
}
