//! Parsed material types.
//!
//! `MaterialEdge` from C contains two packed u16 values encoding base and overlay
//! texture map numbers, plus animation state. These types parse that into a
//! structured representation.

use super::indices::TmapNum;

/// Parsed material edge — two material references with optional overlay.
#[derive(Clone, Debug)]
pub struct ParsedMaterialEdge {
    pub base: TmapNum,
    pub overlay: Option<OverlayMaterial>,
}

/// Overlay material (mat2) with animation state.
#[derive(Clone, Debug)]
pub struct OverlayMaterial {
    pub tmap_num: TmapNum,
    /// Door openness encoded in bits 10-13 (0–15).
    pub door_openness: u8,
    /// UV orientation encoded in bits 14-15 (0–3).
    pub orientation: u8,
}

impl ParsedMaterialEdge {
    /// Parse from raw `MaterialEdge` fields.
    ///
    /// `mat1` is the base texture map number.
    /// `mat2` packs: bits 0-9 = overlay tmap_num, bits 10-13 = door openness,
    /// bits 14-15 = orientation. If the overlay tmap_num is 0, there is no overlay.
    pub fn from_raw(mat1: u16, mat2: u16) -> Self {
        let base = TmapNum::new(mat1).unwrap_or_else(|| {
            // Out-of-range base — clamp to 0 as a safe fallback
            TmapNum::new(0).unwrap()
        });

        let overlay_tmap_raw = mat2 & 0x03FF; // bits 0-9
        let overlay = if overlay_tmap_raw > 0 {
            TmapNum::new(overlay_tmap_raw).map(|tmap_num| {
                let door_openness = ((mat2 >> 10) & 0x0F) as u8;
                let orientation = ((mat2 >> 14) & 0x03) as u8;
                OverlayMaterial {
                    tmap_num,
                    door_openness,
                    orientation,
                }
            })
        } else {
            None
        };

        ParsedMaterialEdge { base, overlay }
    }

    /// The active texture map number — overlay preferred over base.
    pub fn active_tmap(&self) -> TmapNum {
        match &self.overlay {
            Some(o) => o.tmap_num,
            None => self.base,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_only() {
        let edge = ParsedMaterialEdge::from_raw(42, 0);
        assert_eq!(edge.base.as_u16(), 42);
        assert!(edge.overlay.is_none());
        assert_eq!(edge.active_tmap().as_u16(), 42);
    }

    #[test]
    fn with_overlay() {
        // mat2: tmap=100 (bits 0-9), door_openness=5 (bits 10-13), orientation=2 (bits 14-15)
        let mat2 = 100 | (5 << 10) | (2 << 14);
        let edge = ParsedMaterialEdge::from_raw(42, mat2);
        assert_eq!(edge.base.as_u16(), 42);
        let overlay = edge.overlay.as_ref().unwrap();
        assert_eq!(overlay.tmap_num.as_u16(), 100);
        assert_eq!(overlay.door_openness, 5);
        assert_eq!(overlay.orientation, 2);
        assert_eq!(edge.active_tmap().as_u16(), 100);
    }

    #[test]
    fn out_of_range_base_clamps_to_zero() {
        let edge = ParsedMaterialEdge::from_raw(u16::MAX, 0);
        assert_eq!(edge.base.as_u16(), 0);
    }
}
