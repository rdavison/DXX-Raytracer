//! Parsed triangle material reference.
//!
//! The `material_edge_index` field in `Triangle` packs flags in the top bits
//! and the actual index in the bottom 29 bits. The special value `0xFFFF` means
//! "use the instance's material_override instead".

use super::indices::MaterialEdgeIndex;

/// What a triangle references for its material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangleMaterialRef {
    /// Standard level geometry — index into `material_edges[]`.
    MaterialEdge {
        index: MaterialEdgeIndex,
        alpha_cutout: bool,
    },
    /// Billboard/rod — uses the instance's `material_override` instead.
    InstanceOverride,
}

/// Raw flag constants matching `Renderer.h`.
const RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE: u32 = 0xFFFF;
const RT_TRIANGLE_ALPHA_CUTOUT: u32 = 1 << 29;
const RT_TRIANGLE_MEI_MASK: u32 = 0x1FFFFFFF;

impl TriangleMaterialRef {
    /// Parse from the raw `material_edge_index` field of a `Triangle`.
    pub fn parse(raw: u32) -> Self {
        if raw == RT_TRIANGLE_MATERIAL_INSTANCE_OVERRIDE {
            return Self::InstanceOverride;
        }

        let masked = raw & RT_TRIANGLE_MEI_MASK;
        let alpha_cutout = (raw & RT_TRIANGLE_ALPHA_CUTOUT) != 0;

        match MaterialEdgeIndex::new(masked) {
            Some(index) => Self::MaterialEdge {
                index,
                alpha_cutout,
            },
            None => {
                // Out-of-range index — treat as instance override to avoid OOB reads
                Self::InstanceOverride
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_override() {
        let r = TriangleMaterialRef::parse(0xFFFF);
        assert_eq!(r, TriangleMaterialRef::InstanceOverride);
    }

    #[test]
    fn material_edge_no_cutout() {
        let r = TriangleMaterialRef::parse(42);
        match r {
            TriangleMaterialRef::MaterialEdge { index, alpha_cutout } => {
                assert_eq!(index.as_u32(), 42);
                assert!(!alpha_cutout);
            }
            _ => panic!("Expected MaterialEdge"),
        }
    }

    #[test]
    fn material_edge_with_cutout() {
        let raw = 42 | (1 << 29);
        let r = TriangleMaterialRef::parse(raw);
        match r {
            TriangleMaterialRef::MaterialEdge { index, alpha_cutout } => {
                assert_eq!(index.as_u32(), 42);
                assert!(alpha_cutout);
            }
            _ => panic!("Expected MaterialEdge"),
        }
    }

    #[test]
    fn out_of_range_becomes_instance_override() {
        // An index >= RT_MAX_MATERIAL_EDGES (54000) should be treated as override
        let raw = 0x1FFFFFFF; // max 29-bit value, way above 54000
        let r = TriangleMaterialRef::parse(raw);
        assert_eq!(r, TriangleMaterialRef::InstanceOverride);
    }
}
