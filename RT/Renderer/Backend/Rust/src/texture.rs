//! Texture upload, slotmap storage, and white fallback texture.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;

use crate::types::{ResourceHandle, UploadTextureParams};

// Texture format enum values matching RT_TextureFormat in ApiTypes.h
const RT_TEXTURE_FORMAT_RGBA8: u32 = 0;
const RT_TEXTURE_FORMAT_RGBA8_SRGB: u32 = 1;
const RT_TEXTURE_FORMAT_R8: u32 = 2;

/// A single slot in the texture slotmap.
struct TextureSlot {
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
    generation: u32,
}

/// Simple slotmap for managing textures indexed by ResourceHandle.
pub struct TextureSlotMap {
    slots: Vec<Option<TextureSlot>>,
    next_generation: u32,
}

impl TextureSlotMap {
    pub fn new() -> Self {
        // Slot 0 is reserved (NULL handle has index=0)
        Self {
            slots: vec![None],
            next_generation: 1,
        }
    }

    /// Insert a texture, returning a ResourceHandle.
    pub fn insert(
        &mut self,
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
    ) -> ResourceHandle {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);
        if self.next_generation == 0 {
            self.next_generation = 1;
        }

        // Find first free slot (skip index 0)
        for (i, slot) in self.slots.iter_mut().enumerate().skip(1) {
            if slot.is_none() {
                *slot = Some(TextureSlot {
                    texture,
                    generation,
                });
                return ResourceHandle {
                    index: i as u32,
                    generation,
                };
            }
        }

        // No free slot — append
        let index = self.slots.len() as u32;
        self.slots.push(Some(TextureSlot {
            texture,
            generation,
        }));
        ResourceHandle {
            index,
            generation,
        }
    }

    /// Look up a texture by handle. Returns None if handle is invalid or stale.
    pub fn find(
        &self,
        handle: ResourceHandle,
    ) -> Option<&ProtocolObject<dyn MTLTexture>> {
        let slot = self.slots.get(handle.index as usize)?.as_ref()?;
        if slot.generation == handle.generation {
            Some(&slot.texture)
        } else {
            None
        }
    }

    /// Look up a texture by slot index only (ignores generation).
    /// Used for GPU material lookups where we only have the index.
    pub fn find_by_index(
        &self,
        index: u32,
    ) -> Option<&ProtocolObject<dyn MTLTexture>> {
        let slot = self.slots.get(index as usize)?.as_ref()?;
        Some(&slot.texture)
    }

    /// Iterate over all active (index, texture) pairs.
    pub fn for_each_active<F: FnMut(u32, &ProtocolObject<dyn MTLTexture>)>(&self, mut f: F) {
        for (i, slot) in self.slots.iter().enumerate().skip(1) {
            if let Some(s) = slot {
                f(i as u32, &s.texture);
            }
        }
    }

    /// Remove a texture by handle.
    pub fn remove(&mut self, handle: ResourceHandle) {
        if let Some(slot) = self.slots.get_mut(handle.index as usize) {
            if let Some(s) = slot.as_ref() {
                if s.generation == handle.generation {
                    *slot = None;
                }
            }
        }
    }
}

/// Upload a texture to the GPU and return a handle.
pub fn upload_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    slotmap: &mut TextureSlotMap,
    params: &UploadTextureParams,
) -> ResourceHandle {
    let image = &params.image;

    // Map format
    let (pixel_format, bpp) = match image.format {
        RT_TEXTURE_FORMAT_RGBA8 => (MTLPixelFormat::RGBA8Unorm, 4u32),
        RT_TEXTURE_FORMAT_RGBA8_SRGB => (MTLPixelFormat::RGBA8Unorm_sRGB, 4),
        RT_TEXTURE_FORMAT_R8 => (MTLPixelFormat::R8Unorm, 1),
        _ => {
            eprintln!(
                "[Rust Metal] WARNING: Unsupported texture format {}",
                image.format
            );
            return ResourceHandle::NULL;
        }
    };

    if image.pixels.is_null() || image.width == 0 || image.height == 0 {
        return ResourceHandle::NULL;
    }

    // Create texture descriptor
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            pixel_format,
            image.width as usize,
            image.height as usize,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage::ShaderRead);
    desc.setStorageMode(MTLStorageMode::Shared);

    // Create texture
    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("[Rust Metal] Failed to create texture");

    // Upload pixel data
    let bytes_per_row = image.width as usize * bpp as usize;
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: image.width as usize,
            height: image.height as usize,
            depth: 1,
        },
    };

    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region,
            0,
            NonNull::new_unchecked(image.pixels),
            bytes_per_row,
        );
    }

    slotmap.insert(texture)
}

/// Create a 1×1 white RGBA8 texture for use as a fallback when no texture is bound.
pub fn create_white_texture(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLTexture>> {
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::RGBA8Unorm,
            1,
            1,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage::ShaderRead);
    desc.setStorageMode(MTLStorageMode::Shared);

    let texture = device
        .newTextureWithDescriptor(&desc)
        .expect("[Rust Metal] Failed to create white texture");

    let white_pixel: u32 = 0xFFFFFFFF;
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    };

    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region,
            0,
            NonNull::new_unchecked(&white_pixel as *const u32 as *mut c_void),
            4,
        );
    }

    eprintln!("[Rust Metal] White fallback texture created");
    texture
}
