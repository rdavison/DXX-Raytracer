#include "RenderBackend.h"
#include "GlobalMetal.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>
#import <AppKit/AppKit.h>
#import <dispatch/dispatch.h>

#include "Core/MiniMath.h"

#include <stdio.h>
#include <string.h>

// ------------------------------------------------------------------
// Global arrays accessible through Renderer.h

RT_MaterialEdge g_rt_material_edges  [RT_MAX_MATERIAL_EDGES];
uint16_t        g_rt_material_indices[RT_MAX_MATERIALS];

// ------------------------------------------------------------------
// Global state

RT::MetalState  RT::g_mtl;
RT::TweakVars   RT::tweak_vars;

using namespace RT;

SlotMap<MeshResource>   RT::g_mesh_slotmap(MAX_INSTANCES);
SlotMap<TextureResource> RT::g_texture_slotmap(RT_MAX_TEXTURES);

// ------------------------------------------------------------------
// Init / Exit

void RenderBackend::Init(const RT_RendererInitParams *params)
{
	memset(&g_mtl, 0, sizeof(g_mtl));

	g_mtl.arena = params->arena;

	// Extract NSWindow from the platform window handle
	g_mtl.window = (__bridge NSWindow *)params->window_handle;
	MTL_LOG("Window: %s", [[g_mtl.window title] UTF8String]);

	// Create Metal device
	g_mtl.device = MTLCreateSystemDefaultDevice();
	if (!g_mtl.device)
	{
		MTL_LOG("FATAL: MTLCreateSystemDefaultDevice returned nil — no Metal-capable GPU found");
		abort();
	}
	MTL_LOG("Device: %s", [[g_mtl.device name] UTF8String]);

	// Create command queue
	g_mtl.command_queue = [g_mtl.device newCommandQueue];

	// Set up CAMetalLayer on the window's content view
	NSView *contentView = [g_mtl.window contentView];
	[contentView setWantsLayer:YES];

	g_mtl.metal_layer = [CAMetalLayer layer];
	g_mtl.metal_layer.device = g_mtl.device;
	g_mtl.metal_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
	g_mtl.metal_layer.framebufferOnly = YES;

	CGSize viewSize = [contentView bounds].size;
	CGFloat backingScaleFactor = [g_mtl.window backingScaleFactor];
	g_mtl.metal_layer.drawableSize = CGSizeMake(viewSize.width * backingScaleFactor,
	                                             viewSize.height * backingScaleFactor);
	g_mtl.metal_layer.contentsScale = backingScaleFactor;

	[contentView setLayer:g_mtl.metal_layer];

	// Store output dimensions
	g_mtl.output_width  = (uint32_t)(viewSize.width * backingScaleFactor);
	g_mtl.output_height = (uint32_t)(viewSize.height * backingScaleFactor);
	g_mtl.render_width  = g_mtl.output_width;
	g_mtl.render_height = g_mtl.output_height;

	MTL_LOG("Resolution: %u x %u", g_mtl.output_width, g_mtl.output_height);

	// Create frame semaphore for triple buffering
	g_mtl.frame_semaphore = dispatch_semaphore_create(BACK_BUFFER_COUNT);

	// Initialize frame index
	g_mtl.frame_index = 0;
	g_mtl.current_back_buffer_index = 0;

	// Initialize IO struct
	memset(&g_mtl.io, 0, sizeof(g_mtl.io));

	// Initialize mesh tracker
	g_mtl.mesh_tracker.Init(g_mtl.arena);

	MTL_LOG("Metal backend initialized successfully");
}

void RenderBackend::Exit()
{
	MTL_LOG("Shutting down Metal backend");

	// Wait for all in-flight frames to complete
	for (uint32_t i = 0; i < BACK_BUFFER_COUNT; i++)
	{
		dispatch_semaphore_wait(g_mtl.frame_semaphore, DISPATCH_TIME_FOREVER);
	}
	for (uint32_t i = 0; i < BACK_BUFFER_COUNT; i++)
	{
		dispatch_semaphore_signal(g_mtl.frame_semaphore);
	}

	// ARC handles release of ObjC objects when we nil them
	g_mtl.command_queue = nil;
	g_mtl.device = nil;
	g_mtl.metal_layer = nil;
	g_mtl.window = nil;

	MTL_LOG("Metal backend shut down");
}

void RenderBackend::Flush()
{
	MTL_STUB("Flush");
}

// ------------------------------------------------------------------
// Frame lifecycle

void RenderBackend::BeginFrame()
{
	dispatch_semaphore_wait(g_mtl.frame_semaphore, DISPATCH_TIME_FOREVER);
}

void RenderBackend::BeginScene(const RT_SceneSettings *scene_settings)
{
	g_mtl.render_width_override  = scene_settings->render_width_override;
	g_mtl.render_height_override = scene_settings->render_height_override;

	g_mtl.scene.render_blit = scene_settings->render_blit;
	g_mtl.scene.prev_camera = g_mtl.scene.camera;
	g_mtl.scene.camera = *scene_settings->camera;

	g_mtl.scene.camera.forward  = RT_Vec3Normalize(g_mtl.scene.camera.forward);
	g_mtl.scene.camera.right    = RT_Vec3Normalize(g_mtl.scene.camera.right);
	g_mtl.scene.camera.up       = RT_Vec3Normalize(g_mtl.scene.camera.up);
	g_mtl.scene.camera.near_plane = 0.001f;
	g_mtl.scene.camera.far_plane  = 10000.0f;
}

void RenderBackend::EndScene()
{
	RaytraceRender();
	RasterRenderDebugLines();
}

void RenderBackend::EndFrame()
{
	@autoreleasepool
	{
		id<CAMetalDrawable> drawable = [g_mtl.metal_layer nextDrawable];
		if (!drawable)
		{
			MTL_LOG("WARNING: nextDrawable returned nil, skipping frame");
			dispatch_semaphore_signal(g_mtl.frame_semaphore);
			return;
		}

		// Create render pass descriptor — clear to cornflower blue
		MTLRenderPassDescriptor *passDesc = [MTLRenderPassDescriptor renderPassDescriptor];
		passDesc.colorAttachments[0].texture     = [drawable texture];
		passDesc.colorAttachments[0].loadAction  = MTLLoadActionClear;
		passDesc.colorAttachments[0].storeAction = MTLStoreActionStore;
		passDesc.colorAttachments[0].clearColor  = MTLClearColorMake(0.392, 0.584, 0.929, 1.0);

		id<MTLCommandBuffer> commandBuffer = [g_mtl.command_queue commandBuffer];
		commandBuffer.label = @"Frame Command Buffer";

		id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:passDesc];
		encoder.label = @"Clear Pass";
		[encoder endEncoding];

		[commandBuffer presentDrawable:drawable];

		// Signal frame semaphore when GPU finishes this command buffer
		dispatch_semaphore_t semaphore = g_mtl.frame_semaphore;
		[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
			dispatch_semaphore_signal(semaphore);
		}];

		[commandBuffer commit];
	}

	g_mtl.current_back_buffer_index = (g_mtl.current_back_buffer_index + 1) % BACK_BUFFER_COUNT;
	g_mtl.frame_index++;

	g_mtl.mesh_tracker.PruneOldEntries(g_mtl.frame_index);
}

void RenderBackend::SwapBuffers()
{
	// No-op on Metal — presentation is handled in EndFrame
}

void RenderBackend::OnWindowResize(uint32_t width, uint32_t height)
{
	if (width == 0) width = 1;
	if (height == 0) height = 1;

	g_mtl.output_width  = width;
	g_mtl.output_height = height;
	g_mtl.render_width  = width;
	g_mtl.render_height = height;

	if (g_mtl.metal_layer)
	{
		g_mtl.metal_layer.drawableSize = CGSizeMake(width, height);
	}

	MTL_LOG("Window resized to %u x %u", width, height);
}

// ------------------------------------------------------------------
// IO / Debug

RT_RendererIO *RenderBackend::GetIO()
{
	return &g_mtl.io;
}

int RenderBackend::CheckWindowMinimized()
{
	if (g_mtl.window && [g_mtl.window isMiniaturized])
		return 1;

	if (g_mtl.render_width <= 1 && g_mtl.render_height <= 1)
		return 1;

	return 0;
}

void RenderBackend::DoDebugMenus(const RT_DoRendererDebugMenuParams *params)
{
	MTL_STUB("DoDebugMenus");
	(void)params;
}

// ------------------------------------------------------------------
// Resource management

RT_ResourceHandle RenderBackend::UploadTexture(const RT_UploadTextureParams &texture_params)
{
	MTL_STUB("UploadTexture");
	(void)texture_params;
	return RT_RESOURCE_HANDLE_NULL;
}

RT_ResourceHandle RenderBackend::UploadMesh(const RT_UploadMeshParams &mesh_params)
{
	MTL_STUB("UploadMesh");
	(void)mesh_params;
	return RT_RESOURCE_HANDLE_NULL;
}

void RenderBackend::ReleaseTexture(const RT_ResourceHandle texture_handle)
{
	MTL_STUB("ReleaseTexture");
	(void)texture_handle;
}

void RenderBackend::ReleaseMesh(const RT_ResourceHandle mesh_handle)
{
	MTL_STUB("ReleaseMesh");
	(void)mesh_handle;
}

uint16_t RenderBackend::UpdateMaterial(uint16_t material_index, const RT_Material *material)
{
	MTL_STUB("UpdateMaterial");
	(void)material;
	return material_index;
}

// ------------------------------------------------------------------
// Default resources

RT_ResourceHandle RenderBackend::GetDefaultWhiteTexture()
{
	return g_mtl.white_texture_handle;
}

RT_ResourceHandle RenderBackend::GetDefaultBlackTexture()
{
	return g_mtl.black_texture_handle;
}

RT_ResourceHandle RenderBackend::GetBillboardMesh()
{
	return g_mtl.billboard_quad;
}

RT_ResourceHandle RenderBackend::GetCubeMesh()
{
	return g_mtl.cube;
}

// ------------------------------------------------------------------
// Raytracing

void RenderBackend::RaytraceSubmitLights(size_t count, const RT_Light *lights)
{
	MTL_STUB("RaytraceSubmitLights");
	(void)count;
	(void)lights;
}

void RenderBackend::RaytraceSetVerticalOffset(float new_offset)
{
	g_mtl.viewport_offset_y = new_offset;
}

float RenderBackend::RaytraceGetVerticalOffset()
{
	return g_mtl.viewport_offset_y;
}

uint32_t RenderBackend::RaytraceGetCurrentLightCount()
{
	return 0;
}

void RenderBackend::RaytraceMesh(const RT_RenderMeshParams &render_mesh_params)
{
	MTL_STUB("RaytraceMesh");
	(void)render_mesh_params;
}

void RenderBackend::RaytraceBillboardColored(uint16_t material_index, RT_Vec3 color, RT_Vec2 dim, RT_Vec3 pos, RT_Vec3 prev_pos)
{
	MTL_STUB("RaytraceBillboardColored");
	(void)material_index;
	(void)color;
	(void)dim;
	(void)pos;
	(void)prev_pos;
}

void RenderBackend::RaytraceRod(uint16_t material_index, RT_Vec3 bot_p, RT_Vec3 top_p, float width)
{
	MTL_STUB("RaytraceRod");
	(void)material_index;
	(void)bot_p;
	(void)top_p;
	(void)width;
}

void RenderBackend::RaytraceRender()
{
	// Stub — will dispatch raytracing compute work here
}

void RenderBackend::RaytraceSetSkyColors(RT_Vec3 top, RT_Vec3 bottom)
{
	MTL_STUB("RaytraceSetSkyColors");
	(void)top;
	(void)bottom;
}

// ------------------------------------------------------------------
// Rasterization

void RenderBackend::RasterSetViewport(float x, float y, float width, float height)
{
	MTL_STUB("RasterSetViewport");
	(void)x;
	(void)y;
	(void)width;
	(void)height;
}

void RenderBackend::RasterSetRenderTarget(RT_ResourceHandle texture)
{
	MTL_STUB("RasterSetRenderTarget");
	(void)texture;
}

void RenderBackend::RasterTriangles(RT_RasterTrianglesParams *params, uint32_t num_params)
{
	MTL_STUB("RasterTriangles");
	(void)params;
	(void)num_params;
}

void RenderBackend::RasterLines(RT_RasterLineVertex *vertices, uint32_t num_vertices)
{
	MTL_STUB("RasterLines");
	(void)vertices;
	(void)num_vertices;
}

void RenderBackend::RasterLinesWorld(RT_RasterLineVertex *vertices, uint32_t num_vertices)
{
	MTL_STUB("RasterLinesWorld");
	(void)vertices;
	(void)num_vertices;
}

void RenderBackend::RasterRender()
{
	// Stub — will flush raster geometry here
}

void RenderBackend::RasterRenderDebugLines()
{
	// Stub — will render debug lines here
}

void RenderBackend::RasterBlitScene(const RT_Vec2 *top_left, const RT_Vec2 *bottom_right, bool blit_blend)
{
	MTL_STUB("RasterBlitScene");
	(void)top_left;
	(void)bottom_right;
	(void)blit_blend;
}

void RenderBackend::RasterBlit(RT_ResourceHandle src, const RT_Vec2 *top_left, const RT_Vec2 *bottom_right, bool blit_blend)
{
	MTL_STUB("RasterBlit");
	(void)src;
	(void)top_left;
	(void)bottom_right;
	(void)blit_blend;
}

// ------------------------------------------------------------------
// Dear ImGui

void RenderBackend::RenderImGuiTexture(RT_ResourceHandle texture_handle, float width, float height)
{
	MTL_STUB("RenderImGuiTexture");
	(void)texture_handle;
	(void)width;
	(void)height;
}

void RenderBackend::RenderImGui()
{
	MTL_STUB("RenderImGui");
}

// ------------------------------------------------------------------
// Utility

void RenderBackend::QueueScreenshot(const char *file_name)
{
	MTL_STUB("QueueScreenshot");
	g_mtl.queued_screenshot = true;
	size_t len = strlen(file_name);
	if (len >= sizeof(g_mtl.queued_screenshot_name))
		len = sizeof(g_mtl.queued_screenshot_name) - 1;
	memcpy(g_mtl.queued_screenshot_name, file_name, len);
	g_mtl.queued_screenshot_name[len] = '\0';
}
