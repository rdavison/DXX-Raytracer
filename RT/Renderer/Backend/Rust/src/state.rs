//! Global renderer state management.
//!
//! Uses `static mut` with single-threaded access — same invariant as the C++ backend.
//! The game only calls renderer functions from the main thread.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAccelerationStructure, MTLBuffer, MTLCommandQueue, MTLComputePipelineState,
    MTLDevice, MTLRenderPipelineState, MTLSamplerState, MTLTexture,
};
use objc2_quartz_core::CAMetalLayer;

use crate::device::DeviceState;
use crate::mesh::MeshSlotMap;
use crate::raster::RasterBatch;
use crate::raytrace::PendingInstance;
use crate::texture::TextureSlotMap;
use crate::types::*;
use crate::types::Light;
use crate::{RT_MAX_MATERIAL_EDGES, RT_MAX_MATERIALS};

// ============================================================================
// Dispatch semaphore FFI (libdispatch)
// ============================================================================

/// Opaque type for dispatch_semaphore_t (pointer-sized).
pub type DispatchSemaphore = *mut c_void;

/// Timeout value meaning "wait forever".
pub const DISPATCH_TIME_FOREVER: u64 = !0; // ~0ULL

extern "C" {
    pub fn dispatch_semaphore_create(value: isize) -> DispatchSemaphore;
    pub fn dispatch_semaphore_wait(dsema: DispatchSemaphore, timeout: u64) -> isize;
    pub fn dispatch_semaphore_signal(dsema: DispatchSemaphore) -> isize;
}

// ============================================================================
// Triple buffering constant
// ============================================================================

pub const BACK_BUFFER_COUNT: isize = 3;

// ============================================================================
// MetalState
// ============================================================================

pub struct MetalState {
    // Core Metal objects (device must be retained for Metal lifetime even if not read directly)
    #[allow(dead_code)]
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub metal_layer: Retained<CAMetalLayer>,

    // C-accessible IO struct
    pub io: RendererIO,

    // Material system (pre-allocated, written to by C code via pointer)
    pub material_edges: Vec<MaterialEdge>,
    pub material_indices: Vec<u16>,

    // Frame synchronization
    pub frame_semaphore: DispatchSemaphore,
    pub frame_begun: bool,
    pub frame_index: u64,

    // 2D raster pipeline state
    pub tri_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub line_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub raster_sampler: Option<Retained<ProtocolObject<dyn MTLSamplerState>>>,
    pub white_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub white_texture_handle: ResourceHandle,
    pub texture_slotmap: TextureSlotMap,

    // Per-frame raster batch accumulators
    pub raster_batches: Vec<RasterBatch>,
    pub raster_lines: Vec<RasterLineVertex>,
    pub viewport: (f32, f32, f32, f32),

    // Raytracing (Phase 3A)
    pub mesh_slotmap: MeshSlotMap,
    pub raytrace_instances: Vec<PendingInstance>,
    pub raytrace_output_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub raytrace_output_w: u32,
    pub raytrace_output_h: u32,
    pub raytrace_output_handle: ResourceHandle,
    pub compute_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub camera: Camera,
    pub tlas: [Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>; 2],
    pub tlas_scratch: [Option<Retained<ProtocolObject<dyn MTLBuffer>>>; 2],
    pub tlas_buffer_index: usize,
    pub render_width: u32,
    pub render_height: u32,
    pub render_blit: bool,

    // Lighting (Phase 3B)
    pub lights: Vec<Light>,

    // GPU material data (indexed by material_index, stores albedo texture index + flags)
    pub gpu_materials: Vec<GPUMaterial>,
}

// SAFETY: Renderer is only called from the game's main thread.
unsafe impl Send for MetalState {}
unsafe impl Sync for MetalState {}

// ============================================================================
// Global state
// ============================================================================

static mut STATE: Option<MetalState> = None;

/// Initialize global state from a DeviceState.
/// Must be called exactly once, from the main thread.
pub fn init(device_state: DeviceState) {
    let semaphore = unsafe { dispatch_semaphore_create(BACK_BUFFER_COUNT) };
    assert!(!semaphore.is_null(), "Failed to create dispatch semaphore");

    let state = MetalState {
        device: device_state.device,
        command_queue: device_state.command_queue,
        metal_layer: device_state.metal_layer,
        io: RendererIO::default(),
        material_edges: vec![MaterialEdge { mat1: 0, mat2: 0 }; RT_MAX_MATERIAL_EDGES],
        material_indices: vec![0u16; RT_MAX_MATERIALS],
        frame_semaphore: semaphore,
        frame_begun: false,
        frame_index: 0,
        tri_pipeline: None,
        line_pipeline: None,
        raster_sampler: None,
        white_texture: None,
        white_texture_handle: ResourceHandle::NULL,
        texture_slotmap: TextureSlotMap::new(),
        raster_batches: Vec::new(),
        raster_lines: Vec::new(),
        viewport: (0.0, 0.0, 640.0, 480.0),

        // Raytracing (Phase 3A)
        mesh_slotmap: MeshSlotMap::new(),
        raytrace_instances: Vec::new(),
        raytrace_output_texture: None,
        raytrace_output_w: 0,
        raytrace_output_h: 0,
        raytrace_output_handle: ResourceHandle::NULL,
        compute_pipeline: None,
        camera: Camera::default(),
        tlas: [None, None],
        tlas_scratch: [None, None],
        tlas_buffer_index: 0,
        render_width: 640,
        render_height: 480,
        render_blit: false,

        // Lighting (Phase 3B)
        lights: Vec::with_capacity(100),

        // GPU materials
        gpu_materials: vec![GPUMaterial::default(); RT_MAX_MATERIALS],
    };

    unsafe {
        *&raw mut STATE = Some(state);
    }
}

/// Get a mutable reference to the global state.
/// Panics if not initialized.
#[inline]
pub fn get() -> &'static mut MetalState {
    unsafe {
        (*&raw mut STATE)
            .as_mut()
            .expect("[Rust Metal] State not initialized — call RT_RendererInit first")
    }
}

/// Check if state is initialized.
#[inline]
pub fn is_initialized() -> bool {
    unsafe { (*&raw const STATE).is_some() }
}

/// Tear down global state, releasing all Metal resources.
pub fn teardown() {
    unsafe {
        *&raw mut STATE = None;
    }
    eprintln!("[Rust Metal] State torn down");
}
