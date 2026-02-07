//! Packed RGBA8 color type.

/// Packed RGBA8 color (R in low byte, A in high byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct RGBA8(pub u32);

impl RGBA8 {
    pub const WHITE: Self = Self(0xFFFFFFFF);
    pub const TRANSPARENT: Self = Self(0);

    /// Pack RGB floats (0.0–1.0) into RGBA8 with alpha = 255.
    pub fn from_rgb_f32(r: f32, g: f32, b: f32) -> Self {
        Self::from_rgba_f32(r, g, b, 1.0)
    }

    /// Pack RGBA floats (0.0–1.0) into RGBA8.
    pub fn from_rgba_f32(r: f32, g: f32, b: f32, a: f32) -> Self {
        let r = (r * 255.0).clamp(0.0, 255.0) as u32;
        let g = (g * 255.0).clamp(0.0, 255.0) as u32;
        let b = (b * 255.0).clamp(0.0, 255.0) as u32;
        let a = (a * 255.0).clamp(0.0, 255.0) as u32;
        Self(r | (g << 8) | (b << 16) | (a << 24))
    }

    pub fn packed(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_is_0xffffffff() {
        assert_eq!(RGBA8::WHITE.packed(), 0xFFFFFFFF);
    }

    #[test]
    fn transparent_is_0() {
        assert_eq!(RGBA8::TRANSPARENT.packed(), 0);
    }

    #[test]
    fn from_rgb_packs_correctly() {
        let c = RGBA8::from_rgb_f32(1.0, 0.0, 0.0);
        // R=255, G=0, B=0, A=255
        assert_eq!(c.packed(), 0xFF0000FF);
    }

    #[test]
    fn from_rgba_packs_correctly() {
        let c = RGBA8::from_rgba_f32(0.0, 1.0, 0.0, 0.5);
        let r = c.packed() & 0xFF;
        let g = (c.packed() >> 8) & 0xFF;
        let b = (c.packed() >> 16) & 0xFF;
        let a = (c.packed() >> 24) & 0xFF;
        assert_eq!(r, 0);
        assert_eq!(g, 255);
        assert_eq!(b, 0);
        assert!((a as i32 - 127).abs() <= 1); // 0.5 * 255 ≈ 127
    }

    #[test]
    fn clamps_out_of_range() {
        let c = RGBA8::from_rgba_f32(2.0, -1.0, 0.5, 1.0);
        let r = c.packed() & 0xFF;
        let g = (c.packed() >> 8) & 0xFF;
        assert_eq!(r, 255);
        assert_eq!(g, 0);
    }
}
