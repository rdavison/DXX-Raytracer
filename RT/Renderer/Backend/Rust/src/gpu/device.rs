//! Safe wrapper around `MTLDevice` + `MTLCommandQueue`.
//!
//! Provides factory methods that return safe wrapper types from this module.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandQueue, MTLDevice, MTLPixelFormat, MTLResourceOptions, MTLStorageMode,
    MTLTextureDescriptor, MTLTextureUsage,
};

use super::buffer::GpuBuffer;
use super::encoder::GpuCommandBuffer;
use super::texture::GpuTexture;

/// Safe wrapper around a Metal device and its command queue.
pub struct GpuDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl GpuDevice {
    /// Create from existing retained Metal objects.
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Self {
        Self {
            device,
            command_queue,
        }
    }

    /// Create a typed GPU buffer with the given capacity (in elements).
    pub fn create_buffer<T: Copy>(&self, capacity: usize) -> Option<GpuBuffer<T>> {
        GpuBuffer::new(&self.device, capacity)
    }

    /// Create a typed GPU buffer initialized with the given data.
    pub fn create_buffer_with_data<T: Copy>(&self, data: &[T]) -> Option<GpuBuffer<T>> {
        GpuBuffer::with_data(&self.device, data)
    }

    /// Create a 2D texture with the given parameters.
    pub fn create_texture_2d(
        &self,
        format: MTLPixelFormat,
        width: usize,
        height: usize,
        usage: MTLTextureUsage,
        storage: MTLStorageMode,
    ) -> Option<GpuTexture> {
        let desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format, width, height, false,
            )
        };
        desc.setUsage(usage);
        desc.setStorageMode(storage);

        let texture = self.device.newTextureWithDescriptor(&desc)?;
        Some(GpuTexture::from_retained(texture))
    }

    /// Create a command buffer for encoding GPU work.
    pub fn command_buffer(&self) -> Option<GpuCommandBuffer> {
        let cmd_buf = self.command_queue.commandBuffer()?;
        Some(GpuCommandBuffer::new(cmd_buf))
    }

    /// Create a private-storage buffer (GPU-only, for scratch data).
    /// Returns the raw `MTLBuffer` since private buffers aren't CPU-accessible.
    pub fn create_private_buffer(
        &self,
        byte_len: usize,
    ) -> Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> {
        if byte_len == 0 {
            return None;
        }
        self.device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModePrivate)
    }

    /// Access the underlying `MTLDevice` for operations not yet wrapped.
    pub fn metal_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Access the underlying `MTLCommandQueue` for operations not yet wrapped.
    pub fn metal_command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.command_queue
    }
}
