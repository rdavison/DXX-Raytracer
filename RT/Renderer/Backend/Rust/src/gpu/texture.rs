//! Safe wrapper around `MTLTexture`.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLOrigin, MTLRegion, MTLSize, MTLTexture};

/// Safe wrapper around a Metal texture.
pub struct GpuTexture {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
}

impl GpuTexture {
    /// Wrap an existing retained texture.
    pub fn from_retained(texture: Retained<ProtocolObject<dyn MTLTexture>>) -> Self {
        Self { texture }
    }

    /// Write pixel data to a region of the texture.
    ///
    /// `data` must contain enough bytes to fill the region at the given `bytes_per_row`.
    pub fn write_pixels(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        data: &[u8],
        bytes_per_row: usize,
    ) {
        let region = MTLRegion {
            origin: MTLOrigin { x, y, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        };
        let expected_bytes = bytes_per_row * height;
        assert!(
            data.len() >= expected_bytes,
            "GpuTexture::write_pixels: data.len() ({}) < expected ({})",
            data.len(),
            expected_bytes
        );
        unsafe {
            self.texture
                .replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region,
                    0,
                    NonNull::new_unchecked(data.as_ptr() as *mut c_void),
                    bytes_per_row,
                );
        }
    }

    /// Write raw pixel data from a void pointer (for C interop).
    ///
    /// # Safety
    /// `pixels` must point to at least `bytes_per_row * height` valid bytes.
    pub unsafe fn write_pixels_raw(
        &self,
        width: usize,
        height: usize,
        pixels: *mut c_void,
        bytes_per_row: usize,
    ) {
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        };
        self.texture
            .replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                NonNull::new_unchecked(pixels),
                bytes_per_row,
            );
    }

    pub fn width(&self) -> usize {
        self.texture.width()
    }

    pub fn height(&self) -> usize {
        self.texture.height()
    }

    /// Access the underlying `MTLTexture` for encoder binding.
    pub fn metal_texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.texture
    }

    /// Consume and return the inner retained texture.
    pub fn into_retained(self) -> Retained<ProtocolObject<dyn MTLTexture>> {
        self.texture
    }
}
