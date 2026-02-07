//! Rust Metal Renderer — C API exports
//!
//! Implements all functions from Renderer.h as `extern "C"` exports.
//! Phase 1: Device init, black frame, frame lifecycle with GPU sync.
//! Phase 2: 2D raster pipeline, texture upload, menu/HUD rendering.

mod device;
mod mesh;
mod pipeline;
mod raster;
mod raytrace;
mod raytrace_shader;
mod shaders;
mod state;
mod texture;
mod types;

use std::ffi::c_int;
use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice,
    MTLDrawable, MTLLibrary, MTLLoadAction, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassDescriptor, MTLResourceOptions,
    MTLStoreAction, MTLTexture, MTLViewport,
};
use objc2_quartz_core::CAMetalDrawable;

use state::{dispatch_semaphore_signal, dispatch_semaphore_wait, DISPATCH_TIME_FOREVER};
use types::*;

// ============================================================================
// Renderer constants (matching Renderer.h)
// ============================================================================

const RT_MAX_SEGMENTS: usize = 9000;
const RT_SIDES_PER_SEGMENT: usize = 6;
const RT_MAX_MATERIAL_EDGES: usize = RT_MAX_SEGMENTS * RT_SIDES_PER_SEGMENT;

const RT_MAX_BITMAP_FILES: usize = 1800;
const RT_MAX_OBJ_BITMAPS: usize = 210;
const RT_EXTRA_BITMAP_COUNT: usize = 100;
const RT_MAX_MATERIALS: usize = RT_MAX_BITMAP_FILES + RT_MAX_OBJ_BITMAPS + RT_EXTRA_BITMAP_COUNT;

// ============================================================================
// Initialization & Lifecycle
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_RendererInit(params: *const RendererInitParams) {
    if params.is_null() {
        eprintln!("[Rust Metal] ERROR: RT_RendererInit called with null params");
        return;
    }
    let params = unsafe { &*params };
    eprintln!("[Rust Metal] RT_RendererInit called");

    // Create Metal device and layer, attach to window
    let device_state = device::create_device_and_layer(params.window_handle)
        .expect("[Rust Metal] Failed to create Metal device and layer");

    // Initialize global state
    state::init(device_state);

    // Phase 2: Create shader library, pipelines, sampler, and white texture
    let s = state::get();
    let library = shaders::compile_library(&s.device);
    s.tri_pipeline = Some(pipeline::create_tri_pipeline(&s.device, &library));
    s.line_pipeline = Some(pipeline::create_line_pipeline(&s.device, &library));
    s.raster_sampler = Some(pipeline::create_sampler(&s.device));
    s.white_texture = Some(texture::create_white_texture(&s.device));
    // Also create a slotmap entry so RT_GetDefaultWhiteTexture can return a handle
    let white_tex_copy = texture::create_white_texture(&s.device);
    s.white_texture_handle = s.texture_slotmap.insert(white_tex_copy);

    // Phase 3A: Compile raytrace shader and create compute pipeline
    let rt_library = raytrace_shader::compile_raytrace_library(&s.device);
    let func_name = objc2_foundation::NSString::from_str("raytrace_main");
    if let Some(func) = rt_library.newFunctionWithName(&func_name) {
        match s.device.newComputePipelineStateWithFunction_error(&func) {
            Ok(pipeline) => {
                s.compute_pipeline = Some(pipeline);
                eprintln!("[Rust Metal] Raytrace compute pipeline created");
            }
            Err(e) => {
                eprintln!("[Rust Metal] ERROR: Failed to create compute pipeline: {:?}", e);
            }
        }
    } else {
        eprintln!("[Rust Metal] ERROR: raytrace_main function not found in library");
    }

    eprintln!("[Rust Metal] Initialization complete");
}

#[no_mangle]
pub extern "C" fn RT_GetRendererIO() -> *mut RendererIO {
    if !state::is_initialized() {
        // State not yet initialized — allocate lazily so early callers don't crash.
        // This can happen if the game calls RT_GetRendererIO before RT_RendererInit.
        eprintln!("[Rust Metal] WARNING: RT_GetRendererIO called before init");
        static mut EARLY_IO: RendererIO = RendererIO {
            scene_transition: false,
            debug_line_depth_enabled: false,
            screen_overlay_color: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
            delta_time: 0.0,
            debug_render_mode: 0,
            config: std::ptr::null_mut(),
            frame_frozen: false,
        };
        return &raw mut EARLY_IO;
    }
    &mut state::get().io as *mut RendererIO
}

#[no_mangle]
pub extern "C" fn RT_RendererExit() {
    eprintln!("[Rust Metal] RT_RendererExit called");
    state::teardown();
}

// ============================================================================
// Frame Lifecycle
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_BeginFrame() {
    let s = state::get();
    if s.frame_begun {
        // Already inside a frame — don't double-wait on the semaphore
        return;
    }
    // Wait for a free back buffer slot (triple buffering)
    unsafe {
        dispatch_semaphore_wait(s.frame_semaphore, DISPATCH_TIME_FOREVER);
    }
    s.frame_begun = true;
}

#[no_mangle]
pub extern "C" fn RT_BeginScene(settings: *const SceneSettings) {
    if settings.is_null() {
        return;
    }
    let settings = unsafe { &*settings };
    let s = state::get();

    // Read camera from settings
    if !settings.camera.is_null() {
        s.camera = unsafe { *settings.camera };
    }

    // If vfov is 0 or tiny (config not loaded), use a sensible default
    if s.camera.vfov < 1.0 {
        s.camera.vfov = 72.0;
    }

    // Read render size
    if settings.render_width_override > 0 && settings.render_height_override > 0 {
        s.render_width = settings.render_width_override;
        s.render_height = settings.render_height_override;
    }

    s.render_blit = settings.render_blit;

    if s.frame_index % 60 == 0 {
        eprintln!(
            "[Rust Metal] BeginScene: camera=({:.1},{:.1},{:.1}) vfov={:.1} render={}x{} blit={}",
            s.camera.position.x, s.camera.position.y, s.camera.position.z,
            s.camera.vfov,
            s.render_width, s.render_height, settings.render_blit
        );
    }

    // Clear instance list and lights for new frame
    raytrace::begin_scene(&mut s.raytrace_instances);
    s.lights.clear();
}

#[no_mangle]
pub extern "C" fn RT_EndScene() {
    // Dispatch raytracing at end of scene (same as C++ Metal backend)
    RT_RaytraceRender();
}

#[no_mangle]
pub extern "C" fn RT_EndFrame() {
    // Phase 1: No-op (matches C++ backend).
}

#[no_mangle]
pub extern "C" fn RT_SwapBuffers() {
    let s = state::get();
    if !s.frame_begun {
        return;
    }
    s.frame_begun = false;

    present_frame();

    s.frame_index += 1;
}

/// Present frame: clear to black, encode 2D raster batches, present drawable.
fn present_frame() {
    let s = state::get();

    // 1. Get next drawable from the layer
    let drawable = match s.metal_layer.nextDrawable() {
        Some(d) => d,
        None => {
            // No drawable available — signal semaphore so we don't deadlock
            unsafe { dispatch_semaphore_signal(s.frame_semaphore); }
            eprintln!("[Rust Metal] WARNING: No drawable available");
            s.raster_batches.clear();
            s.raster_lines.clear();
            return;
        }
    };

    // 2. Create render pass descriptor — clear to black
    let pass_desc = MTLRenderPassDescriptor::renderPassDescriptor();
    let color0 = unsafe { pass_desc.colorAttachments().objectAtIndexedSubscript(0) };
    let draw_texture = drawable.texture();
    color0.setTexture(Some(&draw_texture));
    color0.setLoadAction(MTLLoadAction::Clear);
    color0.setStoreAction(MTLStoreAction::Store);
    color0.setClearColor(MTLClearColor {
        red: 0.0,
        green: 0.0,
        blue: 0.0,
        alpha: 1.0,
    });

    // 3. Create command buffer
    let cmd_buf = match s.command_queue.commandBuffer() {
        Some(buf) => buf,
        None => {
            unsafe { dispatch_semaphore_signal(s.frame_semaphore); }
            eprintln!("[Rust Metal] ERROR: Failed to create command buffer");
            s.raster_batches.clear();
            s.raster_lines.clear();
            return;
        }
    };

    // 4. Create render command encoder
    if let Some(encoder) = cmd_buf.renderCommandEncoderWithDescriptor(&pass_desc) {
        let target_w = draw_texture.width() as f64;
        let target_h = draw_texture.height() as f64;

        // 4a. Blit raytrace output as fullscreen quad (before raster UI)
        if let (Some(rt_tex), Some(tri_pipe), Some(samp)) = (
            s.raytrace_output_texture.as_ref(),
            s.tri_pipeline.as_ref(),
            s.raster_sampler.as_ref(),
        ) {
            let white = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
            let fullscreen_verts = [
                RasterTriVertex { pos: Vec3 { x: -1.0, y: -1.0, z: 0.0 }, uv: Vec2 { x: 0.0, y: 1.0 }, color: white, texture_index: 0 },
                RasterTriVertex { pos: Vec3 { x:  1.0, y: -1.0, z: 0.0 }, uv: Vec2 { x: 1.0, y: 1.0 }, color: white, texture_index: 0 },
                RasterTriVertex { pos: Vec3 { x: -1.0, y:  1.0, z: 0.0 }, uv: Vec2 { x: 0.0, y: 0.0 }, color: white, texture_index: 0 },
                RasterTriVertex { pos: Vec3 { x:  1.0, y: -1.0, z: 0.0 }, uv: Vec2 { x: 1.0, y: 1.0 }, color: white, texture_index: 0 },
                RasterTriVertex { pos: Vec3 { x:  1.0, y:  1.0, z: 0.0 }, uv: Vec2 { x: 1.0, y: 0.0 }, color: white, texture_index: 0 },
                RasterTriVertex { pos: Vec3 { x: -1.0, y:  1.0, z: 0.0 }, uv: Vec2 { x: 0.0, y: 0.0 }, color: white, texture_index: 0 },
            ];

            encoder.setViewport(MTLViewport {
                originX: 0.0, originY: 0.0,
                width: target_w, height: target_h,
                znear: 0.0, zfar: 1.0,
            });
            encoder.setRenderPipelineState(tri_pipe);
            unsafe { encoder.setFragmentSamplerState_atIndex(Some(samp), 0); }
            unsafe { encoder.setFragmentTexture_atIndex(Some(rt_tex), 0); }

            let byte_len = fullscreen_verts.len() * std::mem::size_of::<RasterTriVertex>();
            let vb = unsafe {
                s.device.newBufferWithBytes_length_options(
                    NonNull::new_unchecked(fullscreen_verts.as_ptr() as *mut std::ffi::c_void),
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            if let Some(ref buf) = vb {
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(Some(buf), 0, 0);
                    encoder.drawPrimitives_vertexStart_vertexCount(
                        MTLPrimitiveType::Triangle,
                        0,
                        6,
                    );
                }
            }
        }

        // 4b. Encode 2D raster batches (HUD/menus on top)
        if let (Some(tri_pipe), Some(line_pipe), Some(samp), Some(white_tex)) = (
            s.tri_pipeline.as_ref(),
            s.line_pipeline.as_ref(),
            s.raster_sampler.as_ref(),
            s.white_texture.as_ref(),
        ) {
            raster::encode_batches(
                &encoder,
                &s.device,
                target_w,
                target_h,
                &s.raster_batches,
                &s.raster_lines,
                &s.texture_slotmap,
                white_tex,
                tri_pipe,
                line_pipe,
                samp,
            );
        }
        encoder.endEncoding();
    }

    // Clear batch accumulators
    s.raster_batches.clear();
    s.raster_lines.clear();

    // 5. Present drawable
    let mtl_drawable: &ProtocolObject<dyn MTLDrawable> =
        ProtocolObject::from_ref(&*drawable);
    cmd_buf.presentDrawable(mtl_drawable);

    // 6. Signal semaphore on completion
    let sem = s.frame_semaphore;
    let block = block2::RcBlock::new(
        move |_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            unsafe { dispatch_semaphore_signal(sem); }
        },
    );
    unsafe {
        let ptr: *mut block2::Block<dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>)> =
            &*block as *const _ as *mut _;
        cmd_buf.addCompletedHandler(ptr);
    }

    // 7. Commit
    cmd_buf.commit();
}

// ============================================================================
// Material System
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_GetMaterialEdgesArray() -> *mut MaterialEdge {
    state::get().material_edges.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn RT_GetMaterialIndicesArray() -> *mut u16 {
    state::get().material_indices.as_mut_ptr()
}

// ============================================================================
// Debug UI
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_DoRendererDebugMenus(_params: *const DoRendererDebugMenuParams) {
    // ImGui debug menus — stub
}

// ============================================================================
// Resource Management
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_UploadTexture(params: *const UploadTextureParams) -> ResourceHandle {
    if params.is_null() {
        return ResourceHandle::NULL;
    }
    let params = unsafe { &*params };
    let s = state::get();
    texture::upload_texture(&s.device, &mut s.texture_slotmap, params)
}

#[no_mangle]
pub extern "C" fn RT_UpdateMaterial(material_index: u16, material: *const Material) -> u16 {
    if material_index as usize >= RT_MAX_MATERIALS {
        return u16::MAX;
    }
    if !material.is_null() {
        let mat = unsafe { &*material };
        let s = state::get();
        let gpu_mat = &mut s.gpu_materials[material_index as usize];
        gpu_mat.albedo_index = mat.textures[0].index; // slot 0 = albedo
        gpu_mat.flags = mat.flags;
    }
    material_index
}

#[no_mangle]
pub extern "C" fn RT_UploadMesh(params: *const UploadMeshParams) -> ResourceHandle {
    if params.is_null() {
        return ResourceHandle::NULL;
    }
    let params = unsafe { &*params };

    if params.triangles.is_null() || params.triangle_count == 0 {
        return ResourceHandle::NULL;
    }

    let triangles = unsafe {
        std::slice::from_raw_parts(params.triangles, params.triangle_count)
    };

    let s = state::get();
    mesh::upload_mesh(&s.device, &s.command_queue, &mut s.mesh_slotmap, triangles)
}

#[no_mangle]
pub extern "C" fn RT_ReleaseTexture(handle: ResourceHandle) {
    if handle.is_valid() {
        state::get().texture_slotmap.remove(handle);
    }
}

#[no_mangle]
pub extern "C" fn RT_ReleaseMesh(handle: ResourceHandle) {
    if handle.is_valid() {
        state::get().mesh_slotmap.remove(handle);
    }
}

#[no_mangle]
pub extern "C" fn RT_GenerateTangents(_triangles: *mut Triangle, _triangle_count: usize) -> bool {
    // TODO: MikkTSpace tangent generation
    true
}

// ============================================================================
// Default Resources
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_GetDefaultWhiteTexture() -> ResourceHandle {
    state::get().white_texture_handle
}

#[no_mangle]
pub extern "C" fn RT_GetDefaultBlackTexture() -> ResourceHandle {
    // TODO: Return handle to 1x1 black texture
    ResourceHandle::NULL
}

#[no_mangle]
pub extern "C" fn RT_GetBillboardMesh() -> ResourceHandle {
    // TODO: Return handle to billboard quad mesh
    ResourceHandle::NULL
}

#[no_mangle]
pub extern "C" fn RT_GetCubeMesh() -> ResourceHandle {
    // TODO: Return handle to unit cube mesh
    ResourceHandle::NULL
}

// ============================================================================
// Window State
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_CheckWindowMinimized() -> c_int {
    0
}

// ============================================================================
// Raytracing Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_RaytraceSetRenderFlagsOverride(_flags: u32) -> u32 {
    0
}

#[no_mangle]
pub extern "C" fn RT_RaytraceMeshEx(params: *mut RenderMeshParams) {
    if params.is_null() {
        return;
    }
    let params = unsafe { &*params };

    if !params.mesh_handle.is_valid() || params.transform.is_null() {
        return;
    }

    let transform = unsafe { &*params.transform };
    let s = state::get();

    raytrace::queue_mesh_instance(
        &mut s.raytrace_instances,
        &s.mesh_slotmap,
        params.mesh_handle,
        transform,
        params.color,
        params.material_override,
    );
}

#[no_mangle]
pub extern "C" fn RT_RaytraceMeshColor(
    mesh: ResourceHandle,
    color: Vec4,
    transform: *const Mat4,
    prev_transform: *const Mat4,
) {
    let r = ((color.x * 255.0).clamp(0.0, 255.0)) as u32;
    let g = ((color.y * 255.0).clamp(0.0, 255.0)) as u32;
    let b = ((color.z * 255.0).clamp(0.0, 255.0)) as u32;
    let a = ((color.w * 255.0).clamp(0.0, 255.0)) as u32;
    let packed = r | (g << 8) | (b << 16) | (a << 24);

    let mut params = RenderMeshParams {
        key_signature: 0,
        key_submodel_index: 0,
        flags: 0,
        mesh_handle: mesh,
        transform,
        prev_transform,
        color: packed,
        material_override: 0,
    };
    RT_RaytraceMeshEx(&mut params);
}

#[no_mangle]
pub extern "C" fn RT_RaytraceMesh(
    mesh: ResourceHandle,
    transform: *const Mat4,
    prev_transform: *const Mat4,
) {
    let mut params = RenderMeshParams {
        key_signature: 0,
        key_submodel_index: 0,
        flags: 0,
        mesh_handle: mesh,
        transform,
        prev_transform,
        color: 0xFFFFFFFF,
        material_override: 0,
    };
    RT_RaytraceMeshEx(&mut params);
}

#[no_mangle]
pub extern "C" fn RT_RaytraceMeshOverrideMaterial(
    mesh: ResourceHandle,
    material_override: u16,
    transform: *const Mat4,
    prev_transform: *const Mat4,
) {
    let mut params = RenderMeshParams {
        key_signature: 0,
        key_submodel_index: 0,
        flags: 0,
        mesh_handle: mesh,
        transform,
        prev_transform,
        color: 0xFFFFFFFF,
        material_override,
    };
    RT_RaytraceMeshEx(&mut params);
}

#[no_mangle]
pub extern "C" fn RT_RaytraceBillboard(
    _material_index: u16,
    _dim: Vec2,
    _pos: Vec3,
    _prev_pos: Vec3,
) {
    // TODO: Billboard rendering
}

#[no_mangle]
pub extern "C" fn RT_RaytraceBillboardColored(
    _material_index: u16,
    _color: Vec3,
    _dim: Vec2,
    _pos: Vec3,
    _prev_pos: Vec3,
) {
    // TODO: Billboard rendering with color
}

#[no_mangle]
pub extern "C" fn RT_RaytraceRod(
    _material_index: u16,
    _bot_p: Vec3,
    _top_p: Vec3,
    _width: f32,
) {
    // TODO: Rod rendering
}

#[no_mangle]
pub extern "C" fn RT_RaytraceRender() {
    let s = state::get();

    if s.raytrace_instances.is_empty() {
        return;
    }

    let pipeline = match s.compute_pipeline.as_ref() {
        Some(p) => p.clone(),
        None => return,
    };

    let render_w = s.render_width;
    let render_h = s.render_height;

    // Ensure output texture exists and matches render size
    raytrace::ensure_output_texture(
        &s.device,
        &mut s.raytrace_output_texture,
        &mut s.raytrace_output_w,
        &mut s.raytrace_output_h,
        render_w,
        render_h,
    );

    let output_tex = match s.raytrace_output_texture.as_ref() {
        Some(t) => t.clone(),
        None => return,
    };

    // Collect owned copies needed to avoid borrow conflicts
    let camera = s.camera;
    let lights: Vec<Light> = s.lights.clone();

    raytrace::dispatch_rays(
        &s.device,
        &s.command_queue,
        &s.mesh_slotmap,
        &mut s.raytrace_instances,
        &camera,
        render_w,
        render_h,
        &pipeline,
        &output_tex,
        &mut s.tlas,
        &mut s.tlas_scratch,
        &mut s.tlas_buffer_index,
        &lights,
        &s.material_edges,
        &s.material_indices,
        &s.gpu_materials,
        &s.texture_slotmap,
        s.white_texture.as_deref(),
    );

    if s.frame_index % 120 == 0 {
        eprintln!(
            "[Rust Metal] Raytraced frame: {} instances, {} lights, {}x{}",
            s.raytrace_instances.len(),
            lights.len(),
            render_w,
            render_h
        );
    }
}

#[no_mangle]
pub extern "C" fn RT_RaytraceSubmitLights(light_count: usize, lights: *const Light) {
    if lights.is_null() || light_count == 0 {
        return;
    }
    let lights_slice = unsafe { std::slice::from_raw_parts(lights, light_count) };
    let s = state::get();
    let remaining = 100_usize.saturating_sub(s.lights.len());
    let to_copy = light_count.min(remaining);
    s.lights.extend_from_slice(&lights_slice[..to_copy]);
}

#[no_mangle]
pub extern "C" fn RT_RaytraceGetCurrentLightCount() -> u32 {
    state::get().lights.len() as u32
}

#[no_mangle]
pub extern "C" fn RT_RaytraceSetVerticalOffset(_new_offset: f32) {
    // TODO: Store vertical offset for camera
}

#[no_mangle]
pub extern "C" fn RT_RaytraceGetVerticalOffset() -> f32 {
    0.0
}

#[no_mangle]
pub extern "C" fn RT_RaytraceSetSkyColors(_sky_top: Vec3, _sky_bottom: Vec3) {
    // TODO: Store sky gradient colors
}

// ============================================================================
// Rasterizer Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_RasterSetViewport(x: f32, y: f32, width: f32, height: f32) {
    let s = state::get();
    s.viewport = (x, y, width, height);
}

#[no_mangle]
pub extern "C" fn RT_RasterSetRenderTarget(_texture: ResourceHandle) {
    // TODO: Set render target for raster pass
}

#[no_mangle]
pub extern "C" fn RT_RasterTriangles(params: *mut RasterTrianglesParams, num_params: u32) {
    if params.is_null() || num_params == 0 {
        return;
    }
    let params_slice = unsafe { std::slice::from_raw_parts(params, num_params as usize) };
    let s = state::get();
    for p in params_slice {
        if p.vertices.is_null() || p.num_vertices == 0 {
            continue;
        }
        let vertices =
            unsafe { std::slice::from_raw_parts(p.vertices, p.num_vertices as usize) };
        raster::push_tri_batch(&mut s.raster_batches, p.texture_handle, vertices);
    }
}

#[no_mangle]
pub extern "C" fn RT_RasterLines(vertices: *mut RasterLineVertex, num_vertices: u32) {
    if vertices.is_null() || num_vertices == 0 {
        return;
    }
    let verts = unsafe { std::slice::from_raw_parts(vertices, num_vertices as usize) };
    let s = state::get();
    raster::push_lines(&mut s.raster_lines, verts);
}

#[no_mangle]
pub extern "C" fn RT_RasterLineWorld(_a: Vec3, _b: Vec3, _color: Vec4) {
    // TODO: World-space debug line
}

#[no_mangle]
pub extern "C" fn RT_RasterLinesWorld(_vertices: *mut RasterLineVertex, _num_vertices: u32) {
    // TODO: World-space debug lines
}

#[no_mangle]
pub extern "C" fn RT_RasterBlitScene(
    top_left: *const Vec2,
    bottom_right: *const Vec2,
    _blit_blend: bool,
) {
    let s = state::get();

    // If we have a raytrace output texture, blit it into the raster pipeline
    let rt_tex = match s.raytrace_output_texture.as_ref() {
        Some(t) => t.clone(),
        None => return,
    };

    // Insert texture into slotmap if needed (reuse handle across frames)
    if !s.raytrace_output_handle.is_valid() {
        s.raytrace_output_handle = s.texture_slotmap.insert(rt_tex);
    } else {
        // Update the existing slot with the current texture
        s.texture_slotmap.remove(s.raytrace_output_handle);
        let tex_clone = s.raytrace_output_texture.as_ref().unwrap().clone();
        s.raytrace_output_handle = s.texture_slotmap.insert(tex_clone);
    }

    // Compute NDC coordinates for the blit quad
    let (tl, br) = if !top_left.is_null() && !bottom_right.is_null() {
        let tl = unsafe { &*top_left };
        let br = unsafe { &*bottom_right };
        (*tl, *br)
    } else {
        // Fullscreen fallback
        (
            Vec2 { x: 0.0, y: 0.0 },
            Vec2 { x: s.viewport.2, y: s.viewport.3 },
        )
    };

    let (_, _, vw, vh) = s.viewport;
    if vw <= 0.0 || vh <= 0.0 {
        return;
    }

    let x0 = (tl.x / vw) * 2.0 - 1.0;
    let y0 = 1.0 - (tl.y / vh) * 2.0;
    let x1 = (br.x / vw) * 2.0 - 1.0;
    let y1 = 1.0 - (br.y / vh) * 2.0;

    let white = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
    let vertices = [
        RasterTriVertex { pos: Vec3 { x: x1, y: y0, z: 0.0 }, uv: Vec2 { x: 1.0, y: 0.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x1, y: y1, z: 0.0 }, uv: Vec2 { x: 1.0, y: 1.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x0, y: y1, z: 0.0 }, uv: Vec2 { x: 0.0, y: 1.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x1, y: y0, z: 0.0 }, uv: Vec2 { x: 1.0, y: 0.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x0, y: y1, z: 0.0 }, uv: Vec2 { x: 0.0, y: 1.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x0, y: y0, z: 0.0 }, uv: Vec2 { x: 0.0, y: 0.0 }, color: white, texture_index: 0 },
    ];

    let handle = s.raytrace_output_handle;
    raster::push_tri_batch(&mut s.raster_batches, handle, &vertices);
}

#[no_mangle]
pub extern "C" fn RT_RasterBlit(
    src: ResourceHandle,
    top_left: *const Vec2,
    bottom_right: *const Vec2,
    _blit_blend: bool,
) {
    if top_left.is_null() || bottom_right.is_null() {
        return;
    }
    let tl = unsafe { &*top_left };
    let br = unsafe { &*bottom_right };

    // Convert pixel coordinates to NDC (-1..1)
    let s = state::get();
    let (_, _, vw, vh) = s.viewport;
    if vw <= 0.0 || vh <= 0.0 {
        return;
    }

    let x0 = (tl.x / vw) * 2.0 - 1.0;
    let y0 = 1.0 - (tl.y / vh) * 2.0;
    let x1 = (br.x / vw) * 2.0 - 1.0;
    let y1 = 1.0 - (br.y / vh) * 2.0;

    let white = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
    let vertices = [
        RasterTriVertex { pos: Vec3 { x: x1, y: y0, z: 0.0 }, uv: Vec2 { x: 1.0, y: 0.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x1, y: y1, z: 0.0 }, uv: Vec2 { x: 1.0, y: 1.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x0, y: y1, z: 0.0 }, uv: Vec2 { x: 0.0, y: 1.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x1, y: y0, z: 0.0 }, uv: Vec2 { x: 1.0, y: 0.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x0, y: y1, z: 0.0 }, uv: Vec2 { x: 0.0, y: 1.0 }, color: white, texture_index: 0 },
        RasterTriVertex { pos: Vec3 { x: x0, y: y0, z: 0.0 }, uv: Vec2 { x: 0.0, y: 0.0 }, color: white, texture_index: 0 },
    ];

    raster::push_tri_batch(&mut s.raster_batches, src, &vertices);
}

#[no_mangle]
pub extern "C" fn RT_RasterRender() {
    // Batches are encoded during present_frame() — this is intentionally a no-op.
    // The C bridge calls this to "flush" but we accumulate and draw everything at present time.
}

// ============================================================================
// Dear ImGui Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_RenderImGuiTexture(
    _texture_handle: ResourceHandle,
    _width: f32,
    _height: f32,
) {
    // ImGui texture rendering — stub
}

#[no_mangle]
pub extern "C" fn RT_RenderImGui() {
    // ImGui rendering — stub
}

// ============================================================================
// Utility Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn RT_QueueScreenshot(_file_name: *const i8) {
    // TODO: Queue screenshot capture
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_handle_null_is_invalid() {
        assert!(!ResourceHandle::NULL.is_valid());
    }

    #[test]
    fn resource_handle_nonzero_is_valid() {
        let h = ResourceHandle { index: 1, generation: 0 };
        assert!(h.is_valid());
    }

    #[test]
    fn mat4_identity() {
        let m = Mat4::identity();
        assert_eq!(m.e[0][0], 1.0);
        assert_eq!(m.e[1][1], 1.0);
        assert_eq!(m.e[2][2], 1.0);
        assert_eq!(m.e[3][3], 1.0);
        assert_eq!(m.e[0][1], 0.0);
    }

    #[test]
    fn renderer_io_default() {
        let io = RendererIO::default();
        assert!(!io.scene_transition);
        assert_eq!(io.delta_time, 0.0);
        assert!(io.config.is_null());
    }
}
