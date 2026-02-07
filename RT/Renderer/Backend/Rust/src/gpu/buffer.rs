//! Typed Metal buffer with safe read/write access.
//!
//! `GpuBuffer<T>` wraps an `MTLBuffer` and provides bounds-checked writes.
//! The buffer always uses `StorageModeShared` for CPU+GPU access.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// A typed Metal buffer with safe read/write access.
///
/// `T` must be `Copy` (plain-old-data suitable for GPU upload).
/// The buffer uses `StorageModeShared` so both CPU and GPU can access it.
pub struct GpuBuffer<T: Copy> {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    capacity: usize, // in elements of T
    _marker: PhantomData<T>,
}

impl<T: Copy> GpuBuffer<T> {
    /// Allocate an uninitialized buffer with room for `capacity` elements.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, capacity: usize) -> Option<Self> {
        let byte_len = capacity.checked_mul(std::mem::size_of::<T>())?;
        if byte_len == 0 {
            return None;
        }
        let buffer =
            device.newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)?;
        Some(Self {
            buffer,
            capacity,
            _marker: PhantomData,
        })
    }

    /// Allocate a buffer initialized with the contents of `data`.
    pub fn with_data(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        let byte_len = data.len() * std::mem::size_of::<T>();
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(data.as_ptr() as *mut c_void),
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        }?;
        Some(Self {
            buffer,
            capacity: data.len(),
            _marker: PhantomData,
        })
    }

    /// Write `data` into the buffer starting at element offset 0.
    ///
    /// # Panics
    /// Panics if `data.len() > self.capacity`.
    pub fn write(&self, data: &[T]) {
        assert!(
            data.len() <= self.capacity,
            "GpuBuffer::write: data.len() ({}) exceeds capacity ({})",
            data.len(),
            self.capacity
        );
        let byte_len = data.len() * std::mem::size_of::<T>();
        if byte_len == 0 {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.buffer.contents().as_ptr() as *mut u8,
                byte_len,
            );
        }
    }

    /// Write `data` into the buffer starting at the given element offset.
    ///
    /// # Panics
    /// Panics if `offset + data.len() > self.capacity`.
    pub fn write_at(&self, offset: usize, data: &[T]) {
        assert!(
            offset + data.len() <= self.capacity,
            "GpuBuffer::write_at: offset ({}) + len ({}) exceeds capacity ({})",
            offset,
            data.len(),
            self.capacity
        );
        let byte_offset = offset * std::mem::size_of::<T>();
        let byte_len = data.len() * std::mem::size_of::<T>();
        if byte_len == 0 {
            return;
        }
        unsafe {
            let dst = (self.buffer.contents().as_ptr() as *mut u8).add(byte_offset);
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, dst, byte_len);
        }
    }

    /// Ensure the buffer has room for at least `needed` elements.
    /// If the current capacity is sufficient, does nothing.
    /// Otherwise, allocates a new buffer (old contents are lost).
    ///
    /// Returns `true` if the buffer was reallocated, `false` if capacity was already sufficient.
    /// Returns `false` (no reallocation) if allocation fails — caller should check `capacity()`.
    pub fn ensure_capacity(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        needed: usize,
    ) -> bool {
        if self.capacity >= needed {
            return false;
        }
        let byte_len = match needed.checked_mul(std::mem::size_of::<T>()) {
            Some(n) if n > 0 => n,
            _ => return false,
        };
        if let Some(buf) =
            device.newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
        {
            self.buffer = buf;
            self.capacity = needed;
            true
        } else {
            false
        }
    }

    /// Current capacity in elements of `T`.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Byte length of the underlying Metal buffer.
    pub fn byte_length(&self) -> usize {
        self.buffer.length()
    }

    /// Raw pointer to the buffer contents. Use within `gpu/` module only.
    pub(super) fn contents_ptr(&self) -> *mut u8 {
        self.buffer.contents().as_ptr() as *mut u8
    }

    /// Access the underlying `MTLBuffer` for encoder binding.
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}
