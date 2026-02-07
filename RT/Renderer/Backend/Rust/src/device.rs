//! Metal device initialization and CAMetalLayer setup.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::{NSView, NSWindow};
use objc2_metal::{
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat,
};
use objc2_quartz_core::CAMetalLayer;

/// Holds the core Metal objects created during init.
pub struct DeviceState {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub metal_layer: Retained<CAMetalLayer>,
}

/// Create the Metal device, command queue, and attach a CAMetalLayer to the window.
///
/// `window_handle` is an `NSWindow*` passed from SDL via the C init params.
/// Returns `None` if device creation fails.
pub fn create_device_and_layer(window_handle: *mut c_void) -> Option<DeviceState> {
    // 1. Create system default Metal device
    let device = MTLCreateSystemDefaultDevice()?;
    eprintln!("[Rust Metal] Device: {:?}", device.name());

    // 2. Create command queue
    let command_queue = device.newCommandQueue()?;

    // 3. Get NSWindow from the raw pointer
    if window_handle.is_null() {
        eprintln!("[Rust Metal] ERROR: window_handle is null");
        return None;
    }
    let window: &NSWindow = unsafe { &*(window_handle as *const NSWindow) };

    // 4. Get the content view
    let view: Retained<NSView> = window.contentView()?;

    // 5. Create CAMetalLayer and configure it
    let layer = CAMetalLayer::new();
    layer.setDevice(Some(&device));
    layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    layer.setFramebufferOnly(true);
    layer.setDisplaySyncEnabled(true);

    // Set layer size to match view bounds
    let bounds = view.bounds();
    layer.setFrame(bounds);

    // 6. Attach layer to view
    view.setWantsLayer(true);
    view.setLayer(Some(&layer));

    eprintln!(
        "[Rust Metal] Layer attached: {}x{}",
        bounds.size.width, bounds.size.height
    );

    Some(DeviceState {
        device,
        command_queue,
        metal_layer: layer,
    })
}
