#include "RenderBackend.h"

#ifdef defer
#undef defer
#endif

#include "GlobalMetal.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "cimgui.h"
#include "imgui_impl_metal.h"
#include <math.h>
#include "Core/Config.h"
#include <vector>

struct RasterBatch
{
	RT_ResourceHandle texture;
	std::vector<RT_RasterTriVertex> vertices;
};

static std::vector<RasterBatch> g_raster_batches;

#define RT_RENDER_SETTINGS_CONFIG_FILE "render_settings.vars"

RT_MaterialEdge g_rt_material_edges[RT_MAX_MATERIAL_EDGES];
uint16_t        g_rt_material_indices[RT_MAX_MATERIALS];

namespace RT
{
	MetalState g_mtl;
	SlotMap<MeshResource> g_mesh_slotmap(MAX_BOTTOM_LEVELS);
	SlotMap<TextureResource> g_texture_slotmap(RT_MAX_TEXTURES);
}

using namespace RT;

static void LogImGuiRenderProof()
{
	FILE* log_file = fopen("RTLogger.txt", "a");
	if (log_file)
	{
		fprintf(log_file, "[Metal] ImGui draw data rendered\n");
		fclose(log_file);
	}
}

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
		g_mtl.imgui_last_scale_x = 0.0f;
		g_mtl.imgui_last_scale_y = 0.0f;
		g_mtl.viewport_x = 0.0f;
		g_mtl.viewport_y = 0.0f;
		g_mtl.viewport_width = 0.0f;
		g_mtl.viewport_height = 0.0f;

		g_mtl.io.config = RT_ArenaAllocStructNoZero(g_mtl.arena, RT_Config);
		RT_InitializeConfig(g_mtl.io.config, g_mtl.arena);
		RT_DeserializeConfigFromFile(g_mtl.io.config, RT_RENDER_SETTINGS_CONFIG_FILE);
		
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
			static bool logged_imgui_render = false;
			{
				NSView* view = [g_mtl.window contentView];
				if (view)
				{
					NSSize view_size = [view bounds].size;
					CGFloat scale = 1.0;
					if ([view.window respondsToSelector:@selector(backingScaleFactor)])
					{
						scale = view.window.backingScaleFactor;
					}
					CGSize drawable_size = CGSizeMake(view_size.width * scale, view_size.height * scale);
					if (drawable_size.width > 0 && drawable_size.height > 0)
					{
						g_mtl.metal_layer.drawableSize = drawable_size;
						ImGuiIO& io = ImGui::GetIO();
						if (io.DisplaySize.x > 0.0f && io.DisplaySize.y > 0.0f)
						{
							float scale_x = (float)(drawable_size.width / io.DisplaySize.x);
							float scale_y = (float)(drawable_size.height / io.DisplaySize.y);
							io.DisplayFramebufferScale.x = scale_x;
							io.DisplayFramebufferScale.y = scale_y;

							if (g_mtl.imgui_last_scale_x > 0.0f &&
								(fabsf(scale_x - g_mtl.imgui_last_scale_x) > 0.01f ||
								 fabsf(scale_y - g_mtl.imgui_last_scale_y) > 0.01f))
							{
								ImGui_ImplMetal_DestroyDeviceObjects();
								ImGui_ImplMetal_CreateDeviceObjects(g_mtl.device);
							}
							{
								static float last_logged_width = 0.0f;
								static float last_logged_height = 0.0f;
								float expected_width = io.DisplaySize.x * io.DisplayFramebufferScale.x;
								float expected_height = io.DisplaySize.y * io.DisplayFramebufferScale.y;
								if (fabsf(expected_width - (float)drawable_size.width) > 1.0f ||
									fabsf(expected_height - (float)drawable_size.height) > 1.0f)
								{
									if (last_logged_width != (float)drawable_size.width ||
										last_logged_height != (float)drawable_size.height)
									{
										MTL_LOG("ImGui scale mismatch: display %.0fx%.0f, drawable %.0fx%.0f",
											io.DisplaySize.x, io.DisplaySize.y,
											(float)drawable_size.width, (float)drawable_size.height);
										last_logged_width = (float)drawable_size.width;
										last_logged_height = (float)drawable_size.height;
									}
								}
							}
							g_mtl.imgui_last_scale_x = scale_x;
							g_mtl.imgui_last_scale_y = scale_y;
						}
					}
				}
			}
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
					// ImGui overlay pass (after scene, before present) into the swapchain render pass.
					ImGui_ImplMetal_RenderDrawData(g_mtl.imgui_draw_data, commandBuffer, renderEncoder);
					if (!logged_imgui_render)
					{
						MTL_LOG("ImGui draw data rendered");
						LogImGuiRenderProof();
						logged_imgui_render = true;
					}
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
		if (!params || !params->ui_has_cursor_focus)
			return;

		if (ImGui::Begin("Render Settings"))
		{
			if (ImGui::Button("Load Settings"))
			{
				if (g_mtl.io.config)
					RT_DeserializeConfigFromFile(g_mtl.io.config, RT_RENDER_SETTINGS_CONFIG_FILE);
			}

			ImGui::SameLine();

			if (ImGui::Button("Save Settings"))
			{
				if (g_mtl.io.config)
					RT_SerializeConfigToFile(g_mtl.io.config, (char *)RT_RENDER_SETTINGS_CONFIG_FILE);
			}

			ImGui::Separator();
			ImGui::Checkbox("Debug Line Depth", &g_mtl.io.debug_line_depth_enabled);
			ImGui::SliderInt("Debug Render Mode", &g_mtl.io.debug_render_mode, 0, 3);
			ImGui::Text("Delta Time: %.3f", g_mtl.io.delta_time);
		}
		ImGui::End();
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
	void RasterSetViewport(float x, float y, float width, float height)
	{
		g_mtl.viewport_x = x;
		g_mtl.viewport_y = y;
		g_mtl.viewport_width = width;
		g_mtl.viewport_height = height;
	}
	void RasterSetRenderTarget(RT_ResourceHandle texture) { MTL_STUB("RasterSetRenderTarget"); }
	void RasterTriangles(RT_RasterTrianglesParams* params, uint32_t num_params)
	{
		if (!params || num_params == 0)
			return;

		for (uint32_t i = 0; i < num_params; i++)
		{
			RT_RasterTrianglesParams* batch_params = &params[i];
			if (!batch_params->vertices || batch_params->num_vertices == 0)
				continue;

			RasterBatch batch = {};
			batch.texture = batch_params->texture_handle;
			batch.vertices.assign(batch_params->vertices, batch_params->vertices + batch_params->num_vertices);
			g_raster_batches.emplace_back(std::move(batch));
		}
	}
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
	static bool logged_imgui_submit = false;
	g_mtl.imgui_draw_data = igGetDrawData();
	g_mtl.imgui_render_requested = (g_mtl.imgui_draw_data != nullptr);
	if (!logged_imgui_submit)
	{
		int list_count = 0;
		if (g_mtl.imgui_draw_data)
			list_count = g_mtl.imgui_draw_data->CmdListsCount;
		FILE* log_file = fopen("RTLogger.txt", "a");
		if (log_file)
		{
			fprintf(log_file, "[Metal] ImGui RenderImGui called (draw_data=%p, cmd_lists=%d)\n",
				(void*)g_mtl.imgui_draw_data, list_count);
			fclose(log_file);
		}
		logged_imgui_submit = true;
	}
}

	void QueueScreenshot(const char *file_name) { MTL_STUB("QueueScreenshot"); }
}
