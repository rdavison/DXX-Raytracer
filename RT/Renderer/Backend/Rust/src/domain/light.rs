//! Parsed light types.
//!
//! The raw `Light` struct from C packs kind, spot parameters, and emission
//! into a compact format. These types parse that into a typed enum.

use crate::types::{Light, LightKind, Mat34, Vec3};

/// A parsed light with typed variant.
#[derive(Clone, Debug)]
pub enum ParsedLight {
    Point {
        position: Vec3,
        emission: [f32; 3],
    },
    Spot {
        position: Vec3,
        direction: Vec3,
        emission: [f32; 3],
        angle: f32,
        softness: f32,
    },
}

impl ParsedLight {
    /// Parse from a raw `Light` C struct.
    ///
    /// `kind == 0` → Point light, `kind == 1` → Spot light.
    /// Emission is packed as RGB332-style in a u32, but currently just
    /// extracted as-is since the shader interprets it.
    pub fn from_raw(raw: &Light) -> Self {
        let position = position_from_transform(&raw.transform);

        // Unpack emission: the C side packs R, G, B into the u32.
        // For now, pass the raw packed value through — the shader unpacks it.
        let emission_packed = raw.emission;
        let r = (emission_packed & 0xFF) as f32 / 255.0;
        let g = ((emission_packed >> 8) & 0xFF) as f32 / 255.0;
        let b = ((emission_packed >> 16) & 0xFF) as f32 / 255.0;
        let emission = [r, g, b];

        match raw.kind {
            LightKind::AreaRect => {
                let direction = forward_from_transform(&raw.transform);
                let angle = raw.spot_angle as f32 / 255.0 * std::f32::consts::PI;
                let softness = raw.spot_softness as f32 / 255.0;
                ParsedLight::Spot {
                    position,
                    direction,
                    emission,
                    angle,
                    softness,
                }
            }
            LightKind::AreaSphere => ParsedLight::Point { position, emission },
        }
    }
}

/// Extract position (translation column) from a 3×4 matrix.
fn position_from_transform(t: &Mat34) -> Vec3 {
    Vec3 {
        x: t.e[0][3],
        y: t.e[1][3],
        z: t.e[2][3],
    }
}

/// Extract forward direction (-Z axis) from a 3×4 matrix.
fn forward_from_transform(t: &Mat34) -> Vec3 {
    Vec3 {
        x: -t.e[0][2],
        y: -t.e[1][2],
        z: -t.e[2][2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_transform() -> Mat34 {
        Mat34 {
            e: [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
            ],
        }
    }

    #[test]
    fn point_light_parsing() {
        let raw = Light {
            kind: LightKind::AreaSphere,
            spot_angle: 0,
            spot_softness: 0,
            spot_vignette: 0,
            emission: 0x00FF80FF, // R=255, G=128, B=255 (but packed B|G|R)
            transform: make_identity_transform(),
        };
        let parsed = ParsedLight::from_raw(&raw);
        match parsed {
            ParsedLight::Point { position, .. } => {
                assert!((position.x - 10.0).abs() < 0.001);
                assert!((position.y - 20.0).abs() < 0.001);
                assert!((position.z - 30.0).abs() < 0.001);
            }
            _ => panic!("Expected Point light"),
        }
    }

    #[test]
    fn spot_light_parsing() {
        let raw = Light {
            kind: LightKind::AreaRect,
            spot_angle: 128,
            spot_softness: 64,
            spot_vignette: 0,
            emission: 0x00FFFFFF,
            transform: make_identity_transform(),
        };
        let parsed = ParsedLight::from_raw(&raw);
        match parsed {
            ParsedLight::Spot { angle, softness, .. } => {
                assert!(angle > 0.0);
                assert!(softness > 0.0);
            }
            _ => panic!("Expected Spot light"),
        }
    }
}
