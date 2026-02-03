#include "RenderBackend.h"

#ifdef defer
#undef defer
#endif

#include "GlobalMetal.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "cimgui.h"
#include "imgui_impl_metal.h"

RT_MaterialEdge g_rt_material_edges[RT_MAX_MATERIAL_EDGES];
uint16_t        g_rt_material_indices[RT_MAX_MATERIALS];

namespace RT
{
	MetalState g_mtl;
	SlotMap<MeshResource> g_mesh_slotmap(MAX_BOTTOM_LEVELS);
	SlotMap<TextureResource> g_texture_slotmap(RT_MAX_TEXTURES);
}

using namespace RT;

namespace RenderBackend
{
	void Init(const RT_RendererInitParams* params)
	{
		g_mtl.arena = params->arena;

		NSWindow* window = (__bridge NSWindow*)params->window_handle;
		g_mtl.window = window;

		g_mtl.device = MTLCreateSystemDefaultDevice();
		if (g_mtl.device)
		{
			MTL_LOG("Metal Device: %s", [g_mtl.device.name UTF8String]);
		}
		else
		{
			MTL_LOG("Failed to create Metal device!");
			return; // Should probably fatal error here
		}

		g_mtl.command_queue = [g_mtl.device newCommandQueue];
		ImGui_ImplMetal_Init(g_mtl.device);
		ImGui_ImplMetal_CreateDeviceObjects(g_mtl.device);

		NSView* view = [window contentView];
		[view setWantsLayer:YES];
		
		g_mtl.metal_layer = [CAMetalLayer layer];
		g_mtl.metal_layer.device = g_mtl.device;
		g_mtl.metal_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
		g_mtl.metal_layer.framebufferOnly = YES;
		// g_mtl.metal_layer.drawableSize will be set in Resize or automatically if not set? 
		// Usually good to set it to match the view's pixel size.
		// For now we'll rely on layout or set it if we have dimensions.
		
		[view setLayer:g_mtl.metal_layer];

		// We assume params->width/height might be useful?
		// But usually we get it from the window.
		// Let's store what we have.
		// g_mtl.render_width = ...

		g_mtl.frame_semaphore = dispatch_semaphore_create(BACK_BUFFER_COUNT);
		g_mtl.imgui_render_requested = false;
		g_mtl.imgui_draw_data = nullptr;
		
		// Init IO
		// g_mtl.io.config = ... (similar to DX12?) 
		// The plan says "Initialize IO struct". 
		// DX12 does RT_InitializeConfig etc.
		// For skeleton we might skip complex config for now or do minimal.
		// I will do minimal InitTweakVars equivalent if needed or just leave it blank for skeleton.
	}

	void Exit()
	{
		MTL_LOG("Shutting down Metal backend");
		ImGui_ImplMetal_Shutdown();
		g_mtl.metal_layer = nil;
		g_mtl.command_queue = nil;
		g_mtl.device = nil;
	}

	void Flush()
	{
		// Stub
	}

	void BeginFrame()
	{
		dispatch_semaphore_wait(g_mtl.frame_semaphore, DISPATCH_TIME_FOREVER);
	}

	void BeginScene(const RT_SceneSettings* scene_settings)
	{
		g_mtl.scene.prev_camera = g_mtl.scene.camera;
		if (scene_settings->camera)
		{
			g_mtl.scene.camera = *scene_settings->camera;
		}
		g_mtl.scene.render_blit = scene_settings->render_blit;
	}

	void EndScene()
	{
		RaytraceRender();
		RasterRenderDebugLines();
	}

	void EndFrame()
	{
		@autoreleasepool {
			id<CAMetalDrawable> drawable = [g_mtl.metal_layer nextDrawable];
			if (drawable)
			{
				MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
				passDescriptor.colorAttachments[0].texture = drawable.texture;
				passDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
				passDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
				passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.392, 0.584, 0.929, 1.0); // Cornflower Blue

				id<MTLCommandBuffer> commandBuffer = [g_mtl.command_queue commandBuffer];
				
				if (g_mtl.imgui_render_requested && g_mtl.imgui_draw_data)
				{
					ImGui_ImplMetal_NewFrame(passDescriptor);
				}

				id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
				if (g_mtl.imgui_render_requested && g_mtl.imgui_draw_data)
				{
					// ImGui overlay pass (after scene, before present).
					ImGui_ImplMetal_RenderDrawData(g_mtl.imgui_draw_data, commandBuffer, renderEncoder);
				}
				[renderEncoder endEncoding];

				[commandBuffer presentDrawable:drawable];
				
				__block dispatch_semaphore_t block_semaphore = g_mtl.frame_semaphore;
				[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
					dispatch_semaphore_signal(block_semaphore);
				}];

				[commandBuffer commit];
			}
			else
			{
				// Failed to get drawable, signal semaphore to avoid deadlock?
				// Or just skip frame.
				dispatch_semaphore_signal(g_mtl.frame_semaphore);
			}
		}

		g_mtl.frame_index++;
		g_mtl.current_back_buffer_index = (g_mtl.current_back_buffer_index + 1) % BACK_BUFFER_COUNT;
		g_mtl.imgui_render_requested = false;
		g_mtl.imgui_draw_data = nullptr;
	}

	void SwapBuffers()
	{
		// No-op
	}

	void OnWindowResize(uint32_t width, uint32_t height)
	{
		// Update metal layer drawable size?
		if (g_mtl.metal_layer)
		{
			g_mtl.metal_layer.drawableSize = CGSizeMake(width, height);
		}
		g_mtl.output_width = width;
		g_mtl.output_height = height;
		// Update render targets if any
	}

	RT_RendererIO *GetIO()
	{
		return &g_mtl.io;
	}

	int CheckWindowMinimized()
	{
		return [g_mtl.window isMiniaturized];
	}

	void DoDebugMenus(const RT_DoRendererDebugMenuParams *params)
	{
		MTL_STUB("DoDebugMenus");
	}

	RT_ResourceHandle UploadTexture(const RT_UploadTextureParams& texture_params)
	{
		MTL_STUB("UploadTexture");
		return RT_RESOURCE_HANDLE_NULL;
	}

	RT_ResourceHandle UploadMesh(const RT_UploadMeshParams& mesh_params)
	{
		MTL_STUB("UploadMesh");
		return RT_RESOURCE_HANDLE_NULL;
	}

	void ReleaseTexture(const RT_ResourceHandle texture_handle)
	{
		MTL_STUB("ReleaseTexture");
	}

	void ReleaseMesh(const RT_ResourceHandle mesh_handle)
	{
		MTL_STUB("ReleaseMesh");
	}

	uint16_t UpdateMaterial(uint16_t material_index, const RT_Material *material)
	{
		MTL_STUB("UpdateMaterial");
		return material_index;
	}

	RT_ResourceHandle GetDefaultWhiteTexture() { return RT_RESOURCE_HANDLE_NULL; }
	RT_ResourceHandle GetDefaultBlackTexture() { return RT_RESOURCE_HANDLE_NULL; }
	RT_ResourceHandle GetBillboardMesh() { return RT_RESOURCE_HANDLE_NULL; }
	RT_ResourceHandle GetCubeMesh() { return RT_RESOURCE_HANDLE_NULL; }

	// Raytracing stubs
	void RaytraceSubmitLights(size_t size, const RT_Light* lights) { MTL_STUB("RaytraceSubmitLights"); }
	void RaytraceSetVerticalOffset(float new_offset) { g_mtl.viewport_offset_y = new_offset; }
	float RaytraceGetVerticalOffset() { return g_mtl.viewport_offset_y; }
	uint32_t RaytraceGetCurrentLightCount() { return 0; }
	void RaytraceMesh(const RT_RenderMeshParams& render_mesh_params) { MTL_STUB("RaytraceMesh"); }
	void RaytraceBillboardColored(uint16_t material_index, RT_Vec3 color, RT_Vec2 dim, RT_Vec3 pos, RT_Vec3 prev_pos) { MTL_STUB("RaytraceBillboardColored"); }
	void RaytraceRod(uint16_t material_index, RT_Vec3 bot_p, RT_Vec3 top_p, float width) { MTL_STUB("RaytraceRod"); }
	void RaytraceRender() { MTL_STUB("RaytraceRender"); }
	void RaytraceSetSkyColors(RT_Vec3 top, RT_Vec3 bottom) { MTL_STUB("RaytraceSetSkyColors"); }

	// Rasterization stubs
	void RasterSetViewport(float x, float y, float width, float height) { MTL_STUB("RasterSetViewport"); }
	void RasterSetRenderTarget(RT_ResourceHandle texture) { MTL_STUB("RasterSetRenderTarget"); }
	void RasterTriangles(RT_RasterTrianglesParams* params, uint32_t num_params) { MTL_STUB("RasterTriangles"); }
	void RasterLines(RT_RasterLineVertex* vertices, uint32_t num_vertices) { MTL_STUB("RasterLines"); }
	void RasterLinesWorld(RT_RasterLineVertex* vertices, uint32_t num_vertices) { MTL_STUB("RasterLinesWorld"); }
	void RasterRender() { MTL_STUB("RasterRender"); }
	void RasterRenderDebugLines() { MTL_STUB("RasterRenderDebugLines"); }
	void RasterBlitScene(const RT_Vec2* top_left, const RT_Vec2* bottom_right, bool blit_blend) { MTL_STUB("RasterBlitScene"); }
	void RasterBlit(RT_ResourceHandle src, const RT_Vec2* top_left, const RT_Vec2* bottom_right, bool blit_blend) { MTL_STUB("RasterBlit"); }

	// ImGui stubs
	void RenderImGuiTexture(RT_ResourceHandle texture_handle, float width, float height) { MTL_STUB("RenderImGuiTexture"); }
	void RenderImGui()
	{
		g_mtl.imgui_draw_data = igGetDrawData();
		g_mtl.imgui_render_requested = (g_mtl.imgui_draw_data != nullptr);
	}

	void QueueScreenshot(const char *file_name) { MTL_STUB("QueueScreenshot"); }
}
