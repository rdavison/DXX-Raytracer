//! Newtype indices that validate at construction.
//!
//! Once you have one of these types, the index is known to be in range.

use crate::{RT_MAX_MATERIALS, RT_MAX_MATERIAL_EDGES};

/// Bitmap index into `gpu_materials[]` — range `[0, RT_MAX_MATERIALS)`.
/// Value 0 is valid (unlike `ResourceHandle` where 0 = NULL).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitmapIndex(u16);

impl BitmapIndex {
    pub fn new(raw: u16) -> Option<Self> {
        if (raw as usize) < RT_MAX_MATERIALS {
            Some(Self(raw))
        } else {
            None
        }
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    pub fn as_u16(self) -> u16 {
        self.0
    }

    pub fn as_u32(self) -> u32 {
        self.0 as u32
    }
}

/// Texture map number — index into `material_indices[]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TmapNum(u16);

impl TmapNum {
    pub fn new(raw: u16) -> Option<Self> {
        if (raw as usize) < RT_MAX_MATERIALS {
            Some(Self(raw))
        } else {
            None
        }
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    pub fn as_u16(self) -> u16 {
        self.0
    }
}

/// Material edge index — index into `material_edges[]`, already masked to 29 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialEdgeIndex(u32);

impl MaterialEdgeIndex {
    pub fn new(raw: u32) -> Option<Self> {
        if (raw as usize) < RT_MAX_MATERIAL_EDGES {
            Some(Self(raw))
        } else {
            None
        }
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

/// Texture slot index — index into `TextureSlotMap`. 0 = no texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureSlotId(u32);

impl TextureSlotId {
    pub const NONE: Self = Self(0);

    pub fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub fn is_some(self) -> bool {
        self.0 != 0
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitmap_index_valid_range() {
        assert!(BitmapIndex::new(0).is_some());
        assert!(BitmapIndex::new(100).is_some());
        assert!(BitmapIndex::new(2109).is_some()); // RT_MAX_MATERIALS - 1
        assert!(BitmapIndex::new(2110).is_none()); // RT_MAX_MATERIALS
        assert!(BitmapIndex::new(u16::MAX).is_none());
    }

    #[test]
    fn tmap_num_valid_range() {
        assert!(TmapNum::new(0).is_some());
        assert!(TmapNum::new(2109).is_some());
        assert!(TmapNum::new(2110).is_none());
    }

    #[test]
    fn material_edge_index_valid_range() {
        assert!(MaterialEdgeIndex::new(0).is_some());
        assert!(MaterialEdgeIndex::new(53999).is_some()); // RT_MAX_MATERIAL_EDGES - 1
        assert!(MaterialEdgeIndex::new(54000).is_none()); // RT_MAX_MATERIAL_EDGES
    }

    #[test]
    fn texture_slot_id_none() {
        let slot = TextureSlotId::NONE;
        assert!(!slot.is_some());
        assert_eq!(slot.as_u32(), 0);
    }

    #[test]
    fn texture_slot_id_some() {
        let slot = TextureSlotId::new(5);
        assert!(slot.is_some());
        assert_eq!(slot.as_u32(), 5);
    }
}
