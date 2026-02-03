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
#include <CoreFoundation/CoreFoundation.h>
#include <cstdlib>
#include "Core/Config.h"
#include <vector>
#include <stddef.h>

struct RasterBatch
{
	RT_ResourceHandle texture;
	std::vector<RT_RasterTriVertex> vertices;
};

static std::vector<RasterBatch> g_raster_batches;
static std::vector<RT_RasterLineVertex> g_raster_lines;
static double g_last_raster_log_time = 0.0;
static double g_last_frame_log_time = 0.0;
static double g_last_line_log_time = 0.0;
static uint64_t g_raster_line_calls = 0;
static uint64_t g_raster_line_vertices = 0;
static bool g_raster_fullscreen_cover = false;

static bool ComputeRasterFullscreenCover()
{
	if (g_raster_batches.empty())
		return false;

	float min_x = 1.0f;
	float max_x = -1.0f;
	float min_y = 1.0f;
	float max_y = -1.0f;
	for (const RasterBatch& batch : g_raster_batches)
	{
		for (const RT_RasterTriVertex& v : batch.vertices)
		{
			min_x = (v.pos.x < min_x) ? v.pos.x : min_x;
			max_x = (v.pos.x > max_x) ? v.pos.x : max_x;
			min_y = (v.pos.y < min_y) ? v.pos.y : min_y;
			max_y = (v.pos.y > max_y) ? v.pos.y : max_y;
		}
	}
	return (min_x <= -0.98f) && (max_x >= 0.98f) &&
		(min_y <= -0.98f) && (max_y >= 0.98f);
}

static id<MTLRenderPipelineState> CreateRasterTriPipeline(id<MTLDevice> device)
{
	static const char* kRasterTriShader = R"metal(
#include <metal_stdlib>
using namespace metal;

struct VertexIn
{
	float3 pos [[attribute(0)]];
	float2 uv [[attribute(1)]];
	float4 color [[attribute(2)]];
	uint texture_index [[attribute(3)]];
};

struct VertexOut
{
	float4 position [[position]];
	float2 uv;
	float4 color;
};

vertex VertexOut raster_vs(VertexIn in [[stage_in]])
{
	VertexOut out;
	out.position = float4(in.pos, 1.0);
	out.uv = in.uv;
	out.color = in.color;
	return out;
}

fragment float4 raster_fs(VertexOut in [[stage_in]],
						  texture2d<float> tex [[texture(0)]],
						  sampler samp [[sampler(0)]])
{
	float4 sampled = tex.sample(samp, in.uv);
	return sampled * in.color;
}
)metal";

	NSError* error = nil;
	NSString* source = [NSString stringWithUTF8String:kRasterTriShader];
	MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
	id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
	if (!library)
	{
		MTL_LOG("Failed to compile raster shader: %s", error.localizedDescription.UTF8String);
		return nil;
	}

	id<MTLFunction> vs = [library newFunctionWithName:@"raster_vs"];
	id<MTLFunction> fs = [library newFunctionWithName:@"raster_fs"];
	if (!vs || !fs)
	{
		MTL_LOG("Failed to find raster shader functions");
		return nil;
	}

	MTLVertexDescriptor* vertex_desc = [[MTLVertexDescriptor alloc] init];
	vertex_desc.attributes[0].format = MTLVertexFormatFloat3;
	vertex_desc.attributes[0].offset = offsetof(RT_RasterTriVertex, pos);
	vertex_desc.attributes[0].bufferIndex = 0;
	vertex_desc.attributes[1].format = MTLVertexFormatFloat2;
	vertex_desc.attributes[1].offset = offsetof(RT_RasterTriVertex, uv);
	vertex_desc.attributes[1].bufferIndex = 0;
	vertex_desc.attributes[2].format = MTLVertexFormatFloat4;
	vertex_desc.attributes[2].offset = offsetof(RT_RasterTriVertex, color);
	vertex_desc.attributes[2].bufferIndex = 0;
	vertex_desc.attributes[3].format = MTLVertexFormatUInt;
	vertex_desc.attributes[3].offset = offsetof(RT_RasterTriVertex, texture_index);
	vertex_desc.attributes[3].bufferIndex = 0;
	vertex_desc.layouts[0].stride = sizeof(RT_RasterTriVertex);
	vertex_desc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

	MTLRenderPipelineDescriptor* pipeline_desc = [[MTLRenderPipelineDescriptor alloc] init];
	pipeline_desc.vertexFunction = vs;
	pipeline_desc.fragmentFunction = fs;
	pipeline_desc.vertexDescriptor = vertex_desc;
	pipeline_desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	pipeline_desc.colorAttachments[0].blendingEnabled = YES;
	pipeline_desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
	pipeline_desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
	pipeline_desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
	pipeline_desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
	pipeline_desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	pipeline_desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

	id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pipeline_desc error:&error];
	if (!pipeline)
	{
		MTL_LOG("Failed to create raster pipeline: %s", error.localizedDescription.UTF8String);
	}
	return pipeline;
}

static id<MTLRenderPipelineState> CreateRasterLinePipeline(id<MTLDevice> device)
{
	static const char* kRasterLineShader = R"metal(
#include <metal_stdlib>
using namespace metal;

struct VertexIn
{
	float3 pos [[attribute(0)]];
	float4 color [[attribute(1)]];
};

struct VertexOut
{
	float4 position [[position]];
	float4 color;
};

vertex VertexOut raster_line_vs(VertexIn in [[stage_in]])
{
	VertexOut out;
	out.position = float4(in.pos, 1.0);
	out.color = in.color;
	return out;
}

fragment float4 raster_line_fs(VertexOut in [[stage_in]])
{
	return in.color;
}
)metal";

	NSError* error = nil;
	NSString* source = [NSString stringWithUTF8String:kRasterLineShader];
	MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
	id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
	if (!library)
	{
		MTL_LOG("Failed to compile raster line shader: %s", error.localizedDescription.UTF8String);
		return nil;
	}

	id<MTLFunction> vs = [library newFunctionWithName:@"raster_line_vs"];
	id<MTLFunction> fs = [library newFunctionWithName:@"raster_line_fs"];
	if (!vs || !fs)
	{
		MTL_LOG("Failed to find raster line shader functions");
		return nil;
	}

	MTLVertexDescriptor* vertex_desc = [[MTLVertexDescriptor alloc] init];
	vertex_desc.attributes[0].format = MTLVertexFormatFloat3;
	vertex_desc.attributes[0].offset = offsetof(RT_RasterLineVertex, pos);
	vertex_desc.attributes[0].bufferIndex = 0;
	vertex_desc.attributes[1].format = MTLVertexFormatFloat4;
	vertex_desc.attributes[1].offset = offsetof(RT_RasterLineVertex, color);
	vertex_desc.attributes[1].bufferIndex = 0;
	vertex_desc.layouts[0].stride = sizeof(RT_RasterLineVertex);
	vertex_desc.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

	MTLRenderPipelineDescriptor* pipeline_desc = [[MTLRenderPipelineDescriptor alloc] init];
	pipeline_desc.vertexFunction = vs;
	pipeline_desc.fragmentFunction = fs;
	pipeline_desc.vertexDescriptor = vertex_desc;
	pipeline_desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	pipeline_desc.colorAttachments[0].blendingEnabled = YES;
	pipeline_desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
	pipeline_desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
	pipeline_desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
	pipeline_desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
	pipeline_desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
	pipeline_desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

	id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pipeline_desc error:&error];
	if (!pipeline)
	{
		MTL_LOG("Failed to create raster line pipeline: %s", error.localizedDescription.UTF8String);
	}
	return pipeline;
}

static id<MTLTexture> CreateWhiteTexture(id<MTLDevice> device)
{
	MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
																					width:1
																				   height:1
																				mipmapped:NO];
	id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
	if (texture)
	{
		uint32_t white = 0xFFFFFFFFu;
		MTLRegion region = { {0, 0, 0}, {1, 1, 1} };
		[texture replaceRegion:region mipmapLevel:0 withBytes:&white bytesPerRow:4];
	}
	return texture;
}

static void EncodeRasterBatches(id<MTLRenderCommandEncoder> renderEncoder,
								NSUInteger target_width,
								NSUInteger target_height,
								bool use_viewport,
								bool scale_viewport_to_target)
{
	static bool logged_raster_batches = false;
	if (!renderEncoder || (g_raster_batches.empty() && g_raster_lines.empty()))
		return;

	MTLViewport viewport;
	if (use_viewport && RT::g_mtl.viewport_width > 0.0f && RT::g_mtl.viewport_height > 0.0f)
	{
		if (scale_viewport_to_target && target_width > 0 && target_height > 0)
		{
			const float logical_w = RT::g_mtl.viewport_width;
			const float logical_h = RT::g_mtl.viewport_height;
			const float scale_x = (float)target_width / logical_w;
			const float scale_y = (float)target_height / logical_h;

			viewport.originX = RT::g_mtl.viewport_x * scale_x;
			viewport.originY = RT::g_mtl.viewport_y * scale_y;
			viewport.width = logical_w * scale_x;
			viewport.height = logical_h * scale_y;
		}
		else
		{
			viewport.originX = RT::g_mtl.viewport_x;
			viewport.originY = RT::g_mtl.viewport_y;
			viewport.width = RT::g_mtl.viewport_width;
			viewport.height = RT::g_mtl.viewport_height;
		}
	}
	else
	{
		viewport.originX = 0.0;
		viewport.originY = 0.0;
		viewport.width = target_width;
		viewport.height = target_height;
	}
	viewport.znear = 0.0;
	viewport.zfar = 1.0;
	[renderEncoder setViewport:viewport];

	MTLScissorRect scissor;
	scissor.x = 0;
	scissor.y = 0;
	scissor.width = target_width;
	scissor.height = target_height;
	[renderEncoder setScissorRect:scissor];

	if (!g_raster_batches.empty() && RT::g_mtl.raster_tri_pipeline)
	{
		[renderEncoder setRenderPipelineState:RT::g_mtl.raster_tri_pipeline];
		[renderEncoder setFragmentSamplerState:RT::g_mtl.raster_sampler atIndex:0];

		for (const RasterBatch& batch : g_raster_batches)
		{
			id<MTLTexture> texture = RT::g_mtl.raster_white_texture;
			RT::TextureResource* tex_res = RT::g_texture_slotmap.Find(batch.texture);
			if (tex_res && tex_res->texture)
				texture = tex_res->texture;

			id<MTLBuffer> vb = [RT::g_mtl.device newBufferWithBytes:batch.vertices.data()
														 length:batch.vertices.size() * sizeof(RT_RasterTriVertex)
														options:MTLResourceStorageModeShared];
			[renderEncoder setVertexBuffer:vb offset:0 atIndex:0];
			[renderEncoder setFragmentTexture:texture atIndex:0];
			[renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
							   vertexStart:0
							   vertexCount:(NSUInteger)batch.vertices.size()];
		}
	}

	if (!g_raster_lines.empty() && RT::g_mtl.raster_line_pipeline)
	{
		[renderEncoder setRenderPipelineState:RT::g_mtl.raster_line_pipeline];
		const size_t vertex_count = g_raster_lines.size() - (g_raster_lines.size() % 2);
		if (vertex_count > 0)
		{
			id<MTLBuffer> vb = [RT::g_mtl.device newBufferWithBytes:g_raster_lines.data()
														 length:vertex_count * sizeof(RT_RasterLineVertex)
														options:MTLResourceStorageModeShared];
			[renderEncoder setVertexBuffer:vb offset:0 atIndex:0];
			[renderEncoder drawPrimitives:MTLPrimitiveTypeLine
							   vertexStart:0
							   vertexCount:(NSUInteger)vertex_count];
		}
	}

	if (!logged_raster_batches && !g_raster_batches.empty())
	{
		MTL_LOG("Raster batches rendered (%zu)", g_raster_batches.size());
		logged_raster_batches = true;
	}

	double now = CFAbsoluteTimeGetCurrent();
	{
		float min_alpha = 1.0f;
		float max_alpha = 0.0f;
		size_t textured_batches = 0;
		size_t total_vertices = 0;
		float min_x = 1.0f;
		float max_x = -1.0f;
		float min_y = 1.0f;
		float max_y = -1.0f;
		for (const RasterBatch& batch : g_raster_batches)
		{
			if (batch.texture.value != 0)
				textured_batches++;
			for (const RT_RasterTriVertex& v : batch.vertices)
			{
				min_alpha = (v.color.w < min_alpha) ? v.color.w : min_alpha;
				max_alpha = (v.color.w > max_alpha) ? v.color.w : max_alpha;
				min_x = (v.pos.x < min_x) ? v.pos.x : min_x;
				max_x = (v.pos.x > max_x) ? v.pos.x : max_x;
				min_y = (v.pos.y < min_y) ? v.pos.y : min_y;
				max_y = (v.pos.y > max_y) ? v.pos.y : max_y;
				total_vertices++;
			}
		}
		g_raster_fullscreen_cover =
			(min_x <= -0.98f) && (max_x >= 0.98f) &&
			(min_y <= -0.98f) && (max_y >= 0.98f);
		if (now - g_last_raster_log_time >= 1.0)
		{
			size_t untextured_batches = (textured_batches <= g_raster_batches.size()) ? (g_raster_batches.size() - textured_batches) : 0;
			MTL_LOG("Raster batches this frame: %zu (verts=%zu textured=%zu untextured=%zu alpha[%.2f..%.2f] bounds x[%.2f..%.2f] y[%.2f..%.2f] fullscreen=%s viewport %.0fx%.0f target %lux%lu)",
				g_raster_batches.size(),
				total_vertices,
				textured_batches,
				untextured_batches,
				min_alpha,
				max_alpha,
				min_x,
				max_x,
				min_y,
				max_y,
				g_raster_fullscreen_cover ? "yes" : "no",
				RT::g_mtl.viewport_width,
				RT::g_mtl.viewport_height,
				(unsigned long)target_width,
				(unsigned long)target_height);
			g_last_raster_log_time = now;
		}
	}
}

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
		g_mtl.raster_render_requested = false;
		g_mtl.raster_render_target_handle = RT_RESOURCE_HANDLE_NULL;
		g_mtl.raster_tri_pipeline = CreateRasterTriPipeline(g_mtl.device);
		g_mtl.raster_line_pipeline = CreateRasterLinePipeline(g_mtl.device);

		MTLSamplerDescriptor* sampler_desc = [[MTLSamplerDescriptor alloc] init];
		sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
		sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
		sampler_desc.mipFilter = MTLSamplerMipFilterNearest;
		sampler_desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
		sampler_desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
		g_mtl.raster_sampler = [g_mtl.device newSamplerStateWithDescriptor:sampler_desc];
		g_mtl.raster_white_texture = CreateWhiteTexture(g_mtl.device);
		g_mtl.raster_render_target = nil;

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
		static double start_time_seconds = 0.0;
		if (start_time_seconds == 0.0)
		{
			start_time_seconds = CFAbsoluteTimeGetCurrent();
		}
		if (CFAbsoluteTimeGetCurrent() - start_time_seconds >= 15.0)
		{
			exit(0);
		}

		@autoreleasepool {
			static bool logged_imgui_render = false;
			bool had_raster_batches = !g_raster_batches.empty();
			bool had_imgui = g_mtl.imgui_render_requested && g_mtl.imgui_draw_data;
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
					// Use logical drawable size to keep UI sizing consistent; macOS will upscale on Retina.
					CGSize drawable_size = CGSizeMake(view_size.width, view_size.height);
					if (drawable_size.width > 0 && drawable_size.height > 0)
					{
						g_mtl.metal_layer.drawableSize = drawable_size;
						ImGuiIO& io = ImGui::GetIO();
						if (io.DisplaySize.x > 0.0f && io.DisplaySize.y > 0.0f)
						{
							float scale_x = 1.0f;
							float scale_y = 1.0f;
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
			double now = CFAbsoluteTimeGetCurrent();
			bool should_clear = true;
			const bool fullscreen_cover_now = ComputeRasterFullscreenCover();
			if (had_raster_batches)
				should_clear = fullscreen_cover_now;
			else if (had_imgui)
				should_clear = false;

			if (now - g_last_frame_log_time >= 1.0)
			{
				NSView* view = [g_mtl.window contentView];
				NSSize view_size = view ? [view bounds].size : NSMakeSize(0, 0);
				CGSize drawable_size = g_mtl.metal_layer ? g_mtl.metal_layer.drawableSize : CGSizeMake(0, 0);
				MTL_LOG("Frame: batches=%zu imgui=%d rt=%s rt_handle=%llu clear=%s view=%.0fx%.0f drawable=%.0fx%.0f",
					g_raster_batches.size(),
					had_imgui ? 1 : 0,
					g_mtl.raster_render_target ? "yes" : "no",
					(unsigned long long)g_mtl.raster_render_target_handle.value,
					should_clear ? "yes" : "no",
					view_size.width,
					view_size.height,
					drawable_size.width,
					drawable_size.height);
				g_last_frame_log_time = now;
			}
			if (now - g_last_line_log_time >= 1.0)
			{
				if (g_raster_line_calls > 0)
				{
					MTL_LOG("Raster lines: calls=%llu verts=%llu",
						(unsigned long long)g_raster_line_calls,
						(unsigned long long)g_raster_line_vertices);
				}
				g_raster_line_calls = 0;
				g_raster_line_vertices = 0;
				g_last_line_log_time = now;
			}
			id<CAMetalDrawable> drawable = [g_mtl.metal_layer nextDrawable];
			if (drawable)
			{
				MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
				passDescriptor.colorAttachments[0].texture = drawable.texture;
				passDescriptor.colorAttachments[0].loadAction = should_clear ? MTLLoadActionClear : MTLLoadActionLoad;
				passDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
				passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.392, 0.584, 0.929, 1.0); // Cornflower Blue

				id<MTLCommandBuffer> commandBuffer = [g_mtl.command_queue commandBuffer];
				
				if (had_imgui)
				{
					ImGui_ImplMetal_NewFrame(passDescriptor);
				}

				id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
				if (!g_raster_batches.empty() && g_mtl.raster_tri_pipeline)
				{
					EncodeRasterBatches(renderEncoder, drawable.texture.width, drawable.texture.height, true, true);
				}
				if (had_imgui)
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
		g_mtl.raster_render_requested = false;
		g_raster_batches.clear();
		g_raster_lines.clear();
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
		if (!texture_params.image.pixels || texture_params.image.width == 0 || texture_params.image.height == 0)
			return RT_RESOURCE_HANDLE_NULL;

		MTLPixelFormat pixel_format = MTLPixelFormatRGBA8Unorm;
		uint32_t bytes_per_pixel = 4;
		switch (texture_params.image.format)
		{
			case RT_TextureFormat_RGBA8:
				pixel_format = MTLPixelFormatRGBA8Unorm;
				bytes_per_pixel = 4;
				break;
			case RT_TextureFormat_RGBA8_SRGB:
				pixel_format = MTLPixelFormatRGBA8Unorm_sRGB;
				bytes_per_pixel = 4;
				break;
			case RT_TextureFormat_R8:
				pixel_format = MTLPixelFormatR8Unorm;
				bytes_per_pixel = 1;
				break;
			default:
				MTL_LOG("UploadTexture: unsupported format %u", texture_params.image.format);
				return RT_RESOURCE_HANDLE_NULL;
		}

		MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
																					width:texture_params.image.width
																				   height:texture_params.image.height
																				mipmapped:NO];
		desc.usage = MTLTextureUsageShaderRead;
		desc.storageMode = MTLStorageModeShared;
		id<MTLTexture> texture = [g_mtl.device newTextureWithDescriptor:desc];
		if (!texture)
			return RT_RESOURCE_HANDLE_NULL;

		uint32_t pitch = texture_params.image.pitch;
		if (pitch == 0)
			pitch = texture_params.image.width * bytes_per_pixel;

		MTLRegion region = { {0, 0, 0}, {texture_params.image.width, texture_params.image.height, 1} };
		[texture replaceRegion:region mipmapLevel:0 withBytes:texture_params.image.pixels bytesPerRow:pitch];

		TextureResource res = {};
		res.texture = texture;
		RT_ResourceHandle handle = g_texture_slotmap.Insert(res);
		return handle;
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
	void RasterSetRenderTarget(RT_ResourceHandle texture)
	{
		static bool logged_set_target = false;
		if (RT_RESOURCE_HANDLE_VALID(texture))
		{
			TextureResource* tex_res = g_texture_slotmap.Find(texture);
			if (tex_res && tex_res->texture)
			{
				if (!(tex_res->texture.usage & MTLTextureUsageRenderTarget))
				{
					MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:tex_res->texture.pixelFormat
																										width:tex_res->texture.width
																									   height:tex_res->texture.height
																									mipmapped:NO];
					desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
					desc.storageMode = MTLStorageModeShared;
					id<MTLTexture> rt_texture = [g_mtl.device newTextureWithDescriptor:desc];
					tex_res->texture = rt_texture;
				}
				g_mtl.raster_render_target = tex_res->texture;
				g_mtl.raster_render_target_handle = texture;
				if (!logged_set_target)
				{
					MTL_LOG("RasterSetRenderTarget: handle=%llu size=%lux%lu",
						(unsigned long long)texture.value,
						(unsigned long)tex_res->texture.width,
						(unsigned long)tex_res->texture.height);
					logged_set_target = true;
				}
				return;
			}
		}

		if (!logged_set_target)
		{
			MTL_LOG("RasterSetRenderTarget: reset to swapchain");
			logged_set_target = true;
		}
		g_mtl.raster_render_target = nil;
		g_mtl.raster_render_target_handle = RT_RESOURCE_HANDLE_NULL;
	}
	void RasterTriangles(RT_RasterTrianglesParams* params, uint32_t num_params)
	{
		static bool logged_raster_bounds = false;
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

			if (!logged_raster_bounds)
			{
				float min_x = batch_params->vertices[0].pos.x;
				float max_x = batch_params->vertices[0].pos.x;
				float min_y = batch_params->vertices[0].pos.y;
				float max_y = batch_params->vertices[0].pos.y;
				for (uint32_t v = 1; v < batch_params->num_vertices; v++)
				{
					float x = batch_params->vertices[v].pos.x;
					float y = batch_params->vertices[v].pos.y;
					min_x = x < min_x ? x : min_x;
					max_x = x > max_x ? x : max_x;
					min_y = y < min_y ? y : min_y;
					max_y = y > max_y ? y : max_y;
				}
				MTL_LOG("Raster batch bounds: x[%.2f..%.2f] y[%.2f..%.2f] verts=%u", min_x, max_x, min_y, max_y, batch_params->num_vertices);
				logged_raster_bounds = true;
			}
		}
	}
	void RasterLines(RT_RasterLineVertex* vertices, uint32_t num_vertices)
	{
		if (!vertices || num_vertices == 0)
			return;
		g_raster_lines.insert(g_raster_lines.end(), vertices, vertices + num_vertices);
		g_raster_line_calls++;
		g_raster_line_vertices += num_vertices;
	}
	void RasterLinesWorld(RT_RasterLineVertex* vertices, uint32_t num_vertices)
	{
		if (!vertices || num_vertices == 0)
			return;
		g_raster_lines.insert(g_raster_lines.end(), vertices, vertices + num_vertices);
		g_raster_line_calls++;
		g_raster_line_vertices += num_vertices;
	}
	void RasterRender()
	{
		if (g_mtl.raster_render_target && !g_raster_batches.empty() && g_mtl.raster_tri_pipeline)
		{
			@autoreleasepool {
				id<MTLCommandBuffer> commandBuffer = [g_mtl.command_queue commandBuffer];
				MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
				passDescriptor.colorAttachments[0].texture = g_mtl.raster_render_target;
				passDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
				passDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
				passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);

				id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
				EncodeRasterBatches(renderEncoder, g_mtl.raster_render_target.width, g_mtl.raster_render_target.height, true, false);
				[renderEncoder endEncoding];

				[commandBuffer commit];
				[commandBuffer waitUntilCompleted];
			}
			g_raster_batches.clear();
			g_raster_lines.clear();
			g_mtl.raster_render_requested = false;
		}
		else
		{
			g_mtl.raster_render_requested = true;
		}
	}
	void RasterRenderDebugLines() { MTL_STUB("RasterRenderDebugLines"); }
	void RasterBlitScene(const RT_Vec2* top_left, const RT_Vec2* bottom_right, bool blit_blend)
	{
		static bool logged_missing_target = false;
		if (RT_RESOURCE_HANDLE_VALID(g_mtl.raster_render_target_handle))
		{
			static bool logged_blit_scene = false;
			if (!logged_blit_scene)
			{
				MTL_LOG("RasterBlitScene: blitting render target");
				logged_blit_scene = true;
			}
			RasterBlit(g_mtl.raster_render_target_handle, top_left, bottom_right, blit_blend);
		}
		else if (!logged_missing_target)
		{
			MTL_LOG("RasterBlitScene skipped (no render target)");
			logged_missing_target = true;
		}
	}
	void RasterBlit(RT_ResourceHandle src, const RT_Vec2* top_left, const RT_Vec2* bottom_right, bool blit_blend)
	{
		(void)blit_blend;
		if (!top_left || !bottom_right)
			return;

		float width = (g_mtl.output_width > 0) ? (float)g_mtl.output_width : (float)g_mtl.metal_layer.drawableSize.width;
		float height = (g_mtl.output_height > 0) ? (float)g_mtl.output_height : (float)g_mtl.metal_layer.drawableSize.height;
		if (width <= 0.0f || height <= 0.0f)
			return;

		float x0 = (top_left->x / width) * 2.0f - 1.0f;
		float y0 = 1.0f - (top_left->y / height) * 2.0f;
		float x1 = (bottom_right->x / width) * 2.0f - 1.0f;
		float y1 = 1.0f - (bottom_right->y / height) * 2.0f;

		RT_Vec4 color = { 1.0f, 1.0f, 1.0f, 1.0f };
		RT_RasterTriVertex vertices[6] = {
			{ .pos = { x1, y0, 0.0f }, .uv = { 1.0f, 0.0f }, .color = color, .texture_index = 0 },
			{ .pos = { x1, y1, 0.0f }, .uv = { 1.0f, 1.0f }, .color = color, .texture_index = 0 },
			{ .pos = { x0, y1, 0.0f }, .uv = { 0.0f, 1.0f }, .color = color, .texture_index = 0 },
			{ .pos = { x1, y0, 0.0f }, .uv = { 1.0f, 0.0f }, .color = color, .texture_index = 0 },
			{ .pos = { x0, y1, 0.0f }, .uv = { 0.0f, 1.0f }, .color = color, .texture_index = 0 },
			{ .pos = { x0, y0, 0.0f }, .uv = { 0.0f, 0.0f }, .color = color, .texture_index = 0 },
		};

		RT_RasterTrianglesParams params = {};
		params.texture_handle = src;
		params.vertices = vertices;
		params.num_vertices = 6;
		RasterTriangles(&params, 1);
	}

	// ImGui stubs
	void RenderImGuiTexture(RT_ResourceHandle texture_handle, float width, float height)
	{
		TextureResource* res = g_texture_slotmap.Find(texture_handle);
		id<MTLTexture> texture = g_mtl.raster_white_texture;
		if (res && res->texture)
			texture = res->texture;

		ImGui::Image((ImTextureID)(__bridge void*)texture, ImVec2(width, height));
	}
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
