#include "RenderBackend.h"

#ifdef defer
#undef defer
#endif

#include "GlobalMetal.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "cimgui.h"
#include "imgui_impl_metal.h"
#include <stdint.h>

#include <math.h>
#include <CoreFoundation/CoreFoundation.h>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
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

// Raytrace shader source is in RaytraceShader.h
#include "RaytraceShader.h"

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
RT_Material     g_rt_materials[RT_MAX_TEXTURES];

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

static void MetalFileLog(const char* fmt, ...)
{
	static FILE* file = nullptr;
	if (!file)
	{
		file = fopen("metal_rt.log", "a");
		if (!file)
			return;
	}
	va_list args;
	va_start(args, fmt);
	vfprintf(file, fmt, args);
	fprintf(file, "\n");
	fflush(file);
	va_end(args);
}

namespace RenderBackend
{
	void Init(const RT_RendererInitParams* params)
	{
		g_mtl.arena = params->arena;
		MetalFileLog("[Metal] Init: start");

		NSWindow* window = (__bridge NSWindow*)params->window_handle;
		g_mtl.window = window;

		g_mtl.device = MTLCreateSystemDefaultDevice();
		if (g_mtl.device)
		{
			MTL_LOG("Metal Device: %s", [g_mtl.device.name UTF8String]);
			MetalFileLog("[Metal] Device: %s", [g_mtl.device.name UTF8String]);
		}
		else
		{
			MTL_LOG("Failed to create Metal device!");
			MetalFileLog("[Metal] Init: failed to create device");
			return; // Should probably fatal error here
		}

		g_mtl.command_queue = [g_mtl.device newCommandQueue];
		if (!g_mtl.command_queue)
		{
			MetalFileLog("[Metal] Init: failed to create command queue");
			return;
		}
		ImGui_ImplMetal_Init(g_mtl.device);
		ImGui_ImplMetal_CreateDeviceObjects(g_mtl.device);

		NSView* view = [window contentView];
		[view setWantsLayer:YES];
		
		g_mtl.metal_layer = [CAMetalLayer layer];
		g_mtl.metal_layer.device = g_mtl.device;
		g_mtl.metal_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
		g_mtl.metal_layer.framebufferOnly = YES;
		g_mtl.metal_layer.displaySyncEnabled = YES; // Enable V-Sync to prevent tearing/flickering

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
		MetalFileLog("[Metal] Init: raster pipelines tri=%s line=%s",
			g_mtl.raster_tri_pipeline ? "yes" : "no",
			g_mtl.raster_line_pipeline ? "yes" : "no");

		MTLSamplerDescriptor* sampler_desc = [[MTLSamplerDescriptor alloc] init];
		sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
		sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
		sampler_desc.mipFilter = MTLSamplerMipFilterNearest;
		sampler_desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
		sampler_desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
		g_mtl.raster_sampler = [g_mtl.device newSamplerStateWithDescriptor:sampler_desc];
		g_mtl.raster_white_texture = CreateWhiteTexture(g_mtl.device);
		g_mtl.raster_render_target = nil;

		// Create raytracing compute pipeline
		{
			NSError* error = nil;
			MTLCompileOptions* rtOptions = [[MTLCompileOptions alloc] init];
			rtOptions.languageVersion = MTLLanguageVersion3_0;
			id<MTLLibrary> lib = [g_mtl.device newLibraryWithSource:
				[NSString stringWithUTF8String:kRaytraceShader] options:rtOptions error:&error];
			if (lib) {
				id<MTLFunction> fn = [lib newFunctionWithName:@"raytrace_main"];
				if (fn) {
					// Use linked pipeline descriptor to support acceleration structures
					MTLComputePipelineDescriptor *pipeDesc = [[MTLComputePipelineDescriptor alloc] init];
					pipeDesc.computeFunction = fn;
					pipeDesc.linkedFunctions = [[MTLLinkedFunctions alloc] init];
					g_mtl.raytrace_pipeline = [g_mtl.device newComputePipelineStateWithDescriptor:pipeDesc
						options:0 reflection:nil error:&error];
					if (g_mtl.raytrace_pipeline)
					{
						MTL_LOG("Raytracing pipeline created (Metal 3.0 with accel struct support)");
						MetalFileLog("[Metal] Init: raytrace pipeline created with acceleration structure support");
					}
				}
			}
			if (!g_mtl.raytrace_pipeline) {
				MTL_LOG("Failed to create raytrace pipeline: %s",
					error ? error.localizedDescription.UTF8String : "unknown");
				MetalFileLog("[Metal] Init: raytrace pipeline FAILED: %s",
					error ? error.localizedDescription.UTF8String : "unknown");
			}

			g_mtl.raytrace_instance_buffer = [g_mtl.device newBufferWithLength:
				sizeof(RaytraceInstance) * MAX_INSTANCES options:MTLResourceStorageModeShared];
			g_mtl.raytrace_scene_buffer = [g_mtl.device newBufferWithLength:
				sizeof(RaytraceSceneConstants) options:MTLResourceStorageModeShared];
			g_mtl.raytrace_stats_buffer = [g_mtl.device newBufferWithLength:
				sizeof(uint32_t) options:MTLResourceStorageModeShared];
			g_mtl.raytrace_instance_count = 0;

			// Material system buffers
			g_mtl.raytrace_material_buffer = [g_mtl.device newBufferWithLength:
				sizeof(GPUMaterial) * RT_MAX_MATERIALS options:MTLResourceStorageModeShared];
			g_mtl.raytrace_material_edges_buffer = [g_mtl.device newBufferWithLength:
				sizeof(RT_MaterialEdge) * RT_MAX_MATERIAL_EDGES options:MTLResourceStorageModeShared];
			g_mtl.raytrace_material_indices_buffer = [g_mtl.device newBufferWithLength:
				sizeof(uint16_t) * RT_MAX_MATERIALS options:MTLResourceStorageModeShared];
			// Texture remap buffer: maps original texture indices to bounded shader slots (0-30)
			g_mtl.raytrace_texture_remap_buffer = [g_mtl.device newBufferWithLength:
				sizeof(uint32_t) * RT_MAX_TEXTURES options:MTLResourceStorageModeShared];
			g_mtl.raytrace_materials_dirty = true;

			// Light buffer
			g_mtl.raytrace_light_buffer = [g_mtl.device newBufferWithLength:
				sizeof(RT_Light) * RT_MAX_LIGHTS options:MTLResourceStorageModeShared];
			g_mtl.raytrace_light_count = 0;

			// Create texture sampler for raytracing
			MTLSamplerDescriptor* sampler_desc = [[MTLSamplerDescriptor alloc] init];
			sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
			sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
			sampler_desc.sAddressMode = MTLSamplerAddressModeRepeat;
			sampler_desc.tAddressMode = MTLSamplerAddressModeRepeat;
			g_mtl.raytrace_sampler = [g_mtl.device newSamplerStateWithDescriptor:sampler_desc];

			MetalFileLog("[Metal] Init: raytrace buffers instances=%s scene=%s materials=%s",
				g_mtl.raytrace_instance_buffer ? "yes" : "no",
				g_mtl.raytrace_scene_buffer ? "yes" : "no",
				g_mtl.raytrace_material_buffer ? "yes" : "no");
		}

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

	static bool g_frame_begun = false;

	void BeginFrame()
	{
		if (!g_frame_begun)
		{
			MetalFileLog("[Metal] BeginFrame");
			dispatch_semaphore_wait(g_mtl.frame_semaphore, DISPATCH_TIME_FOREVER);
			g_frame_begun = true;
		}
	}

	void BeginScene(const RT_SceneSettings* scene_settings)
	{
		if (!scene_settings)
		{
			MetalFileLog("[Metal] BeginScene: null scene_settings");
			return;
		}
		MetalFileLog("[Metal] BeginScene: overrides=%ux%u",
			scene_settings->render_width_override,
			scene_settings->render_height_override);
		g_mtl.scene.prev_camera = g_mtl.scene.camera;
		if (scene_settings->camera)
		{
			g_mtl.scene.camera = *scene_settings->camera;
			const RT_Vec3 cp = g_mtl.scene.camera.position;
			const RT_Vec3 cf = g_mtl.scene.camera.forward;
			const RT_Vec3 cr = g_mtl.scene.camera.right;
			const RT_Vec3 cu = g_mtl.scene.camera.up;
			MetalFileLog("[Metal] BeginScene: cam pos(%.2f %.2f %.2f) f(%.2f %.2f %.2f) r(%.2f %.2f %.2f) u(%.2f %.2f %.2f) vfov=%.2f",
				cp.x, cp.y, cp.z, cf.x, cf.y, cf.z, cr.x, cr.y, cr.z, cu.x, cu.y, cu.z, g_mtl.scene.camera.vfov);
		}
		g_mtl.render_width_override = scene_settings->render_width_override;
		g_mtl.render_height_override = scene_settings->render_height_override;
		if (g_mtl.render_width_override > 0 && g_mtl.render_height_override > 0)
		{
			g_mtl.render_width = g_mtl.render_width_override;
			g_mtl.render_height = g_mtl.render_height_override;
		}
		if (g_mtl.render_width == 0 || g_mtl.render_height == 0)
		{
			if (g_mtl.output_width > 0 && g_mtl.output_height > 0)
			{
				g_mtl.render_width = g_mtl.output_width;
				g_mtl.render_height = g_mtl.output_height;
			}
			else
			{
				g_mtl.render_width = BASE_SCREEN_SIZE_X;
				g_mtl.render_height = BASE_SCREEN_SIZE_Y;
			}
		}
		g_mtl.scene.render_blit = scene_settings->render_blit;
	}

	void EndScene()
	{
		MetalFileLog("[Metal] EndScene");
		RaytraceRender();
		RasterRenderDebugLines();
	}

	void EndFrame()
	{
		// Don't present here - presentation happens in SwapBuffers/PresentFrame
		// This allows multiple window handlers to accumulate their drawing
	}

	void PresentFrame()
	{
		@autoreleasepool {
			MetalFileLog("[Metal] PresentFrame");
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
			// Always clear for raster/UI frames to avoid undefined content showing through.
			// Only skip clear for imgui-only frames (imgui handles its own background).
			if (had_raster_batches)
				should_clear = true;
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
			// Ensure raytrace output exists for the frame if the pipeline is ready.
			if (!g_mtl.raytrace_output_texture && g_mtl.raytrace_pipeline)
			{
				MetalFileLog("[Metal] PresentFrame: triggering RaytraceRender (no output)");
				RaytraceRender();
			}

			// Blit raytraced output if available
			if (g_mtl.raytrace_output_texture && g_mtl.scene.render_blit) {
				// Use existing RasterBlit mechanism - store texture for blit
				g_mtl.raster_render_target = g_mtl.raytrace_output_texture;
			}

			id<CAMetalDrawable> drawable = [g_mtl.metal_layer nextDrawable];
			if (drawable)
			{
				MTLRenderPassDescriptor* passDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
				passDescriptor.colorAttachments[0].texture = drawable.texture;
				passDescriptor.colorAttachments[0].loadAction = should_clear ? MTLLoadActionClear : MTLLoadActionLoad;
				passDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
				passDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0); // Black

				id<MTLCommandBuffer> commandBuffer = [g_mtl.command_queue commandBuffer];
				
				if (had_imgui)
				{
					ImGui_ImplMetal_NewFrame(passDescriptor);
				}

				id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:passDescriptor];
				if (g_mtl.raytrace_output_texture && g_mtl.raster_tri_pipeline)
				{
					RT_RasterTriVertex verts[6] = {};
					verts[0].pos = { -1.0f, -1.0f, 0.0f };
					verts[1].pos = {  1.0f, -1.0f, 0.0f };
					verts[2].pos = { -1.0f,  1.0f, 0.0f };
					verts[3].pos = {  1.0f, -1.0f, 0.0f };
					verts[4].pos = {  1.0f,  1.0f, 0.0f };
					verts[5].pos = { -1.0f,  1.0f, 0.0f };
					verts[0].uv = { 0.0f, 1.0f };
					verts[1].uv = { 1.0f, 1.0f };
					verts[2].uv = { 0.0f, 0.0f };
					verts[3].uv = { 1.0f, 1.0f };
					verts[4].uv = { 1.0f, 0.0f };
					verts[5].uv = { 0.0f, 0.0f };
					for (int i = 0; i < 6; ++i)
					{
						verts[i].color = { 1.0f, 1.0f, 1.0f, 1.0f };
						verts[i].texture_index = 0;
					}

					id<MTLBuffer> vb = [g_mtl.device newBufferWithBytes:verts
															 length:sizeof(verts)
															options:MTLResourceStorageModeShared];
					[renderEncoder setRenderPipelineState:g_mtl.raster_tri_pipeline];
					[renderEncoder setFragmentSamplerState:g_mtl.raster_sampler atIndex:0];
					[renderEncoder setVertexBuffer:vb offset:0 atIndex:0];
					[renderEncoder setFragmentTexture:g_mtl.raytrace_output_texture atIndex:0];
					[renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
				}
				if ((!g_raster_batches.empty() || !g_raster_lines.empty()) &&
					(g_mtl.raster_tri_pipeline || g_mtl.raster_line_pipeline))
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
		// Present the accumulated frame content
		if (g_frame_begun)
		{
			PresentFrame();
			g_frame_begun = false;
		}
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
		// Store handle back into the resource so ForEach can access it
		TextureResource* stored = g_texture_slotmap.Find(handle);
		if (stored) stored->handle = handle;
		return handle;
	}

	RT_ResourceHandle UploadMesh(const RT_UploadMeshParams& mesh_params)
	{
		if (!mesh_params.triangles || mesh_params.triangle_count == 0)
			return RT_RESOURCE_HANDLE_NULL;

		size_t size = sizeof(RT_Triangle) * mesh_params.triangle_count;
		id<MTLBuffer> buf = [g_mtl.device newBufferWithBytes:mesh_params.triangles
													  length:size
													 options:MTLResourceStorageModeShared];
		if (!buf) return RT_RESOURCE_HANDLE_NULL;

		if (mesh_params.name)
			buf.label = [NSString stringWithUTF8String:mesh_params.name];

		// Build position-only buffer for BLAS (3 float3 per triangle)
		uint32_t tri_count = (uint32_t)mesh_params.triangle_count;
		size_t pos_buf_size = tri_count * 3 * sizeof(float) * 3;
		id<MTLBuffer> pos_buf = [g_mtl.device newBufferWithLength:pos_buf_size
														 options:MTLResourceStorageModeShared];
		float* pos_dst = (float*)[pos_buf contents];
		for (uint32_t i = 0; i < tri_count; i++) {
			const RT_Triangle& tri = mesh_params.triangles[i];
			// Vertex 0
			pos_dst[i * 9 + 0] = tri.pos0.x;
			pos_dst[i * 9 + 1] = tri.pos0.y;
			pos_dst[i * 9 + 2] = tri.pos0.z;
			// Vertex 1
			pos_dst[i * 9 + 3] = tri.pos1.x;
			pos_dst[i * 9 + 4] = tri.pos1.y;
			pos_dst[i * 9 + 5] = tri.pos1.z;
			// Vertex 2
			pos_dst[i * 9 + 6] = tri.pos2.x;
			pos_dst[i * 9 + 7] = tri.pos2.y;
			pos_dst[i * 9 + 8] = tri.pos2.z;
		}

		// Build BLAS (bottom-level acceleration structure)
		id<MTLAccelerationStructure> blas = nil;
		MTLAccelerationStructureTriangleGeometryDescriptor *geomDesc =
			[MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
		geomDesc.vertexBuffer = pos_buf;
		geomDesc.vertexStride = sizeof(float) * 3;
		geomDesc.triangleCount = tri_count;

		MTLPrimitiveAccelerationStructureDescriptor *blasDesc =
			[MTLPrimitiveAccelerationStructureDescriptor descriptor];
		blasDesc.geometryDescriptors = @[geomDesc];

		MTLAccelerationStructureSizes sizes =
			[g_mtl.device accelerationStructureSizesWithDescriptor:blasDesc];

		blas = [g_mtl.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
		id<MTLBuffer> scratch = [g_mtl.device newBufferWithLength:sizes.buildScratchBufferSize
														 options:MTLResourceStorageModePrivate];

		id<MTLCommandBuffer> cmdBuf = [g_mtl.command_queue commandBuffer];
		id<MTLAccelerationStructureCommandEncoder> asEncoder =
			[cmdBuf accelerationStructureCommandEncoder];
		[asEncoder buildAccelerationStructure:blas
								  descriptor:blasDesc
							   scratchBuffer:scratch
						 scratchBufferOffset:0];
		[asEncoder endEncoding];
		[cmdBuf commit];
		[cmdBuf waitUntilCompleted];

		if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
			MetalFileLog("[Metal] UploadMesh: BLAS build failed status=%ld", (long)cmdBuf.status);
			blas = nil;
		}

		MeshResource res = {};
		res.triangle_buffer = buf;
		res.position_buffer = pos_buf;
		res.blas = blas;
		res.triangle_count = tri_count;

		RT_ResourceHandle handle = g_mesh_slotmap.Insert(res);
		MTL_LOG("UploadMesh: %zu triangles -> handle %llu (blas=%s)",
			mesh_params.triangle_count, handle.value, blas ? "yes" : "no");
		return handle;
	}

	void ReleaseTexture(const RT_ResourceHandle texture_handle)
	{
		MTL_STUB("ReleaseTexture");
	}

	void ReleaseMesh(const RT_ResourceHandle mesh_handle)
	{
		if (RT_RESOURCE_HANDLE_VALID(mesh_handle))
		{
			MTL_LOG("ReleaseMesh: releasing handle %llu", (unsigned long long)mesh_handle.value);
			g_mesh_slotmap.Remove(mesh_handle);
		}
	}

	uint16_t UpdateMaterial(uint16_t material_index, const RT_Material *material)
	{
		if (material_index < RT_MAX_MATERIALS && material)
		{
			g_rt_materials[material_index] = *material;
			g_mtl.raytrace_materials_dirty = true;
		}
		return material_index;
	}

	RT_ResourceHandle GetDefaultWhiteTexture() { return RT_RESOURCE_HANDLE_NULL; }
	RT_ResourceHandle GetDefaultBlackTexture() { return RT_RESOURCE_HANDLE_NULL; }
	RT_ResourceHandle GetBillboardMesh() { return RT_RESOURCE_HANDLE_NULL; }
	RT_ResourceHandle GetCubeMesh() { return RT_RESOURCE_HANDLE_NULL; }

	// Raytracing
	void RaytraceSubmitLights(size_t count, const RT_Light* lights)
	{
		if (!lights || count == 0 || !g_mtl.raytrace_light_buffer) return;
		if (g_mtl.raytrace_light_count + count > RT_MAX_LIGHTS)
			count = RT_MAX_LIGHTS - g_mtl.raytrace_light_count;
		if (count == 0) return;
		RT_Light* dst = (RT_Light*)[g_mtl.raytrace_light_buffer contents];
		memcpy(dst + g_mtl.raytrace_light_count, lights, sizeof(RT_Light) * count);

		// Debug: log first light's emission RGBE values (once)
		static bool logged_lights = false;
		if (!logged_lights && count > 0) {
			logged_lights = true;
			for (size_t i = 0; i < count && i < 5; i++) {
				uint32_t rgbe = lights[i].emission;
				int exponent = (int)(rgbe >> 27) - 20;
				float scale = powf(2.0f, (float)exponent) / 256.0f;
				float fr = (float)((rgbe >>  0) & 0x1FF) * scale * 1000.0f;
				float fg = (float)((rgbe >>  9) & 0x1FF) * scale * 1000.0f;
				float fb = (float)((rgbe >> 18) & 0x1FF) * scale * 1000.0f;
				MetalFileLog("[Metal] Light[%zu]: rgbe=0x%08X exp=%d decoded=(%.2f, %.2f, %.2f) kind=%u pos=(%.2f, %.2f, %.2f)",
					i, rgbe, exponent, fr, fg, fb, lights[i].kind,
					lights[i].transform.e[0][3], lights[i].transform.e[1][3], lights[i].transform.e[2][3]);
			}
		}

		g_mtl.raytrace_light_count += (uint32_t)count;
	}
	void RaytraceSetVerticalOffset(float new_offset) { g_mtl.viewport_offset_y = new_offset; }
	float RaytraceGetVerticalOffset() { return g_mtl.viewport_offset_y; }
	uint32_t RaytraceGetCurrentLightCount() { return g_mtl.raytrace_light_count; }
	void RaytraceMesh(const RT_RenderMeshParams& params)
	{
		if (g_mtl.raytrace_instance_count >= MAX_INSTANCES) return;

		MeshResource* mesh = g_mesh_slotmap.Find(params.mesh_handle);
		if (!mesh || !mesh->triangle_buffer) return;

		RaytraceInstance* instances = (RaytraceInstance*)[g_mtl.raytrace_instance_buffer contents];
		RaytraceInstance& inst = instances[g_mtl.raytrace_instance_count];

		inst.object_to_world = params.transform ? *params.transform : RT_Mat4Identity();
		inst.world_to_object = RT_Mat4Inverse(inst.object_to_world);
		inst.triangle_buffer_idx = params.mesh_handle.index;
		inst.triangle_count = mesh->triangle_count;
		inst.color = params.color ? params.color : 0xFFFFFFFF;
		inst.material_override = params.material_override;
		inst.triangle_offset = 0;  // Will be filled in RaytraceRender
		inst._pad[0] = inst._pad[1] = inst._pad[2] = 0;

		g_mtl.raytrace_pending_meshes.push_back(params.mesh_handle);
		g_mtl.raytrace_instance_count++;
		MetalFileLog("[Metal] RaytraceMesh: queued instance %u (handle=%llu tris=%u)",
			g_mtl.raytrace_instance_count,
			(unsigned long long)params.mesh_handle.value,
			mesh->triangle_count);
		static bool logged_first_instance = false;
		if (!logged_first_instance)
		{
			MTL_LOG("RaytraceMesh: first instance queued (count=%u tris=%u)",
			g_mtl.raytrace_instance_count, mesh->triangle_count);
		logged_first_instance = true;
	}
}
	void RaytraceBillboardColored(uint16_t material_index, RT_Vec3 color, RT_Vec2 dim, RT_Vec3 pos, RT_Vec3 prev_pos) { MTL_STUB("RaytraceBillboardColored"); }
	void RaytraceRod(uint16_t material_index, RT_Vec3 bot_p, RT_Vec3 top_p, float width) { MTL_STUB("RaytraceRod"); }
	void RaytraceRender()
	{
		static bool logged_once = false;
		if (!logged_once)
		{
			MTL_LOG("RaytraceRender: enter (instances=%u pipeline=%s)",
				g_mtl.raytrace_instance_count,
				g_mtl.raytrace_pipeline ? "yes" : "no");
			logged_once = true;
		}
		// Skip rendering if no pipeline or no instances queued
		if (g_mtl.raytrace_instance_count == 0 || !g_mtl.raytrace_pipeline) {
			MetalFileLog("[Metal] RaytraceRender: skip (instances=%u pipeline=%s)",
				g_mtl.raytrace_instance_count, g_mtl.raytrace_pipeline ? "yes" : "no");
			g_mtl.raytrace_instance_count = 0;
			g_mtl.raytrace_pending_meshes.clear();
			return;
		}

		@autoreleasepool {
			uint32_t w = g_mtl.render_width > 0 ? g_mtl.render_width :
						 (uint32_t)g_mtl.metal_layer.drawableSize.width;
			uint32_t h = g_mtl.render_height > 0 ? g_mtl.render_height :
						 (uint32_t)g_mtl.metal_layer.drawableSize.height;
			if (w == 0 || h == 0) {
				MetalFileLog("[Metal] RaytraceRender: invalid size %ux%u", w, h);
				g_mtl.raytrace_instance_count = 0;
				g_mtl.raytrace_pending_meshes.clear();
				return;
			}

			// Ensure output texture
			if (!g_mtl.raytrace_output_texture ||
				g_mtl.raytrace_output_texture.width != w ||
				g_mtl.raytrace_output_texture.height != h) {
				MTLTextureDescriptor* desc = [MTLTextureDescriptor
					texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
				desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
				desc.storageMode = MTLStorageModeShared;
				g_mtl.raytrace_output_texture = [g_mtl.device newTextureWithDescriptor:desc];
				MetalFileLog("[Metal] RaytraceRender: created output texture %ux%u", w, h);
			}

			// Build combined triangle buffer and set per-instance triangle offsets
			uint32_t total_tris = 0;
			for (uint32_t i = 0; i < g_mtl.raytrace_instance_count; i++) {
				MeshResource* m = g_mesh_slotmap.Find(g_mtl.raytrace_pending_meshes[i]);
				if (m) total_tris += m->triangle_count;
			}

			uint32_t alloc_tris = (total_tris > 0) ? total_tris : 1;
			id<MTLBuffer> combined = [g_mtl.device newBufferWithLength:
				sizeof(RT_Triangle) * alloc_tris options:MTLResourceStorageModeShared];
			RT_Triangle* dst = (RT_Triangle*)[combined contents];
			uint32_t offset = 0;

			RaytraceInstance* instances = (RaytraceInstance*)[g_mtl.raytrace_instance_buffer contents];
			for (uint32_t i = 0; i < g_mtl.raytrace_instance_count; i++) {
				MeshResource* m = g_mesh_slotmap.Find(g_mtl.raytrace_pending_meshes[i]);
				if (m && m->triangle_buffer) {
					instances[i].triangle_offset = offset;
					memcpy(dst + offset, [m->triangle_buffer contents],
						   sizeof(RT_Triangle) * m->triangle_count);
					offset += m->triangle_count;
				}
			}

			// ============================================================
			// Build TLAS (top-level acceleration structure)
			// ============================================================
			// Collect unique BLASes and build instance descriptors
			NSMutableArray<id<MTLAccelerationStructure>> *uniqueBLASes = [NSMutableArray array];
			std::vector<uint32_t> blas_index_map; // instance index -> index into uniqueBLASes

			for (uint32_t i = 0; i < g_mtl.raytrace_instance_count; i++) {
				MeshResource* m = g_mesh_slotmap.Find(g_mtl.raytrace_pending_meshes[i]);
				if (m && m->blas) {
					// Find or add this BLAS to the unique list
					uint32_t blas_idx = (uint32_t)[uniqueBLASes count];
					for (uint32_t j = 0; j < [uniqueBLASes count]; j++) {
						if (uniqueBLASes[j] == m->blas) {
							blas_idx = j;
							break;
						}
					}
					if (blas_idx == (uint32_t)[uniqueBLASes count]) {
						[uniqueBLASes addObject:m->blas];
					}
					blas_index_map.push_back(blas_idx);
				} else {
					blas_index_map.push_back(0); // Fallback
				}
			}

			bool tlas_built = false;
			if ([uniqueBLASes count] > 0) {
				// Allocate instance descriptor buffer
				size_t desc_buf_size = sizeof(MTLAccelerationStructureInstanceDescriptor) * g_mtl.raytrace_instance_count;
				if (!g_mtl.raytrace_instance_desc_buffer ||
					g_mtl.raytrace_instance_desc_buffer.length < desc_buf_size) {
					g_mtl.raytrace_instance_desc_buffer = [g_mtl.device newBufferWithLength:desc_buf_size
						options:MTLResourceStorageModeShared];
				}

				MTLAccelerationStructureInstanceDescriptor *inst_descs =
					(MTLAccelerationStructureInstanceDescriptor*)[g_mtl.raytrace_instance_desc_buffer contents];

				for (uint32_t i = 0; i < g_mtl.raytrace_instance_count; i++) {
					MTLAccelerationStructureInstanceDescriptor &desc = inst_descs[i];
					const RaytraceInstance &inst = instances[i];

					// Convert 4x4 object_to_world to MTLPackedFloat4x3 (top 3 rows, column-major)
					// MTLPackedFloat4x3 has 4 columns of 3 floats each
					desc.transformationMatrix.columns[0] = MTLPackedFloat3Make(
						inst.object_to_world.e[0][0], inst.object_to_world.e[1][0], inst.object_to_world.e[2][0]);
					desc.transformationMatrix.columns[1] = MTLPackedFloat3Make(
						inst.object_to_world.e[0][1], inst.object_to_world.e[1][1], inst.object_to_world.e[2][1]);
					desc.transformationMatrix.columns[2] = MTLPackedFloat3Make(
						inst.object_to_world.e[0][2], inst.object_to_world.e[1][2], inst.object_to_world.e[2][2]);
					desc.transformationMatrix.columns[3] = MTLPackedFloat3Make(
						inst.object_to_world.e[0][3], inst.object_to_world.e[1][3], inst.object_to_world.e[2][3]);

					desc.options = MTLAccelerationStructureInstanceOptionOpaque;
					desc.mask = 0xFF;
					desc.intersectionFunctionTableOffset = 0;
					desc.accelerationStructureIndex = blas_index_map[i];
				}

				// Create TLAS descriptor
				MTLInstanceAccelerationStructureDescriptor *tlasDesc =
					[MTLInstanceAccelerationStructureDescriptor descriptor];
				tlasDesc.instanceDescriptorBuffer = g_mtl.raytrace_instance_desc_buffer;
				tlasDesc.instanceCount = g_mtl.raytrace_instance_count;
				tlasDesc.instancedAccelerationStructures = uniqueBLASes;

				// Query TLAS sizes
				MTLAccelerationStructureSizes tlas_sizes =
					[g_mtl.device accelerationStructureSizesWithDescriptor:tlasDesc];

				// Allocate or reallocate TLAS
				if (!g_mtl.raytrace_tlas ||
					g_mtl.raytrace_tlas.size < tlas_sizes.accelerationStructureSize) {
					g_mtl.raytrace_tlas = [g_mtl.device
						newAccelerationStructureWithSize:tlas_sizes.accelerationStructureSize];
				}

				// Allocate or reallocate scratch buffer
				if (!g_mtl.raytrace_tlas_scratch_buffer ||
					g_mtl.raytrace_tlas_scratch_buffer.length < tlas_sizes.buildScratchBufferSize) {
					g_mtl.raytrace_tlas_scratch_buffer = [g_mtl.device
						newBufferWithLength:tlas_sizes.buildScratchBufferSize
						options:MTLResourceStorageModePrivate];
				}

				// Build TLAS
				id<MTLCommandBuffer> tlasCmdBuf = [g_mtl.command_queue commandBuffer];
				id<MTLAccelerationStructureCommandEncoder> asEncoder =
					[tlasCmdBuf accelerationStructureCommandEncoder];
				[asEncoder buildAccelerationStructure:g_mtl.raytrace_tlas
										  descriptor:tlasDesc
									   scratchBuffer:g_mtl.raytrace_tlas_scratch_buffer
								 scratchBufferOffset:0];
				[asEncoder endEncoding];
				[tlasCmdBuf commit];
				[tlasCmdBuf waitUntilCompleted];

				if (tlasCmdBuf.status == MTLCommandBufferStatusCompleted) {
					tlas_built = true;
				} else {
					MetalFileLog("[Metal] RaytraceRender: TLAS build failed status=%ld error=%s",
						(long)tlasCmdBuf.status,
						tlasCmdBuf.error ? tlasCmdBuf.error.localizedDescription.UTF8String : "none");
				}

				static bool logged_tlas = false;
				if (!logged_tlas && tlas_built) {
					MetalFileLog("[Metal] TLAS built: %u instances, %lu unique BLASes",
						g_mtl.raytrace_instance_count, (unsigned long)[uniqueBLASes count]);
					logged_tlas = true;
				}
			}

			// Copy material data to GPU buffers
			memcpy([g_mtl.raytrace_material_edges_buffer contents], g_rt_material_edges,
				   sizeof(RT_MaterialEdge) * RT_MAX_MATERIAL_EDGES);
			memcpy([g_mtl.raytrace_material_indices_buffer contents], g_rt_material_indices,
				   sizeof(uint16_t) * RT_MAX_MATERIALS);

			// Build GPU material array from g_rt_materials
			GPUMaterial* gpu_materials = (GPUMaterial*)[g_mtl.raytrace_material_buffer contents];
			for (uint32_t i = 0; i < RT_MAX_MATERIALS; i++) {
				const RT_Material& src = g_rt_materials[i];
				GPUMaterial& dst = gpu_materials[i];
				dst.albedo_index = src.albedo_texture.index;
				dst.normal_index = src.normal_texture.index;
				dst.metalness_index = src.metalness_texture.index;
				dst.roughness_index = src.roughness_texture.index;
				dst.emissive_index = src.emissive_texture.index;
				dst.height_index = src.height_texture.index;
				dst.flags = src.flags;
				dst.metalness_factor = src.metalness;
				dst.roughness_factor = src.roughness;
				dst.emissive_factor = (uint32_t)(src.emissive_strength * 255.0f);
			}

			// ============================================================
			// Dynamic Texture Remapping: Build per-frame texture mapping
			// ============================================================
			// Step 1: Collect unique texture indices used by triangles this frame
			constexpr uint32_t MAX_BOUND_TEXTURES = 31;
			std::vector<uint32_t> used_texture_indices;
			used_texture_indices.reserve(MAX_BOUND_TEXTURES);

			// Helper lambda to get material index from material_indices (same logic as shader)
			auto get_material_index = [](uint32_t material_edge) -> uint32_t {
				// g_rt_material_indices is uint16_t[], so direct index access works
				return g_rt_material_indices[material_edge];
			};

			// Scan triangles to find material_edge_index values and resolve to materials
			for (uint32_t tri_idx = 0; tri_idx < total_tris; tri_idx++) {
				uint32_t mat_edge_idx = dst[tri_idx].material_edge_index;
				uint32_t mat_idx = 0;

				// Resolve material index using the same logic as shader
				if (mat_edge_idx & RT_TRIANGLE_HOLDS_MATERIAL_INDEX) {
					mat_idx = mat_edge_idx & ~RT_TRIANGLE_HOLDS_MATERIAL_INDEX;
				} else if (mat_edge_idx & RT_TRIANGLE_HOLDS_MATERIAL_EDGE) {
					uint32_t edge_idx = mat_edge_idx & ~RT_TRIANGLE_HOLDS_MATERIAL_EDGE;
					mat_idx = get_material_index(edge_idx);
				} else {
					// Normal case: look up in material_edges, then material_indices
					if (mat_edge_idx < RT_MAX_MATERIAL_EDGES) {
						RT_MaterialEdge edge = g_rt_material_edges[mat_edge_idx];
						uint16_t mat1 = edge.mat1;
						mat_idx = get_material_index(mat1);
					}
				}

				// Clamp material index and get albedo texture index
				if (mat_idx < RT_MAX_MATERIALS) {
					uint32_t tex_idx = g_rt_materials[mat_idx].albedo_texture.index;
					if (tex_idx > 0 && tex_idx < RT_MAX_TEXTURES) {
						// Check if already in our list
						bool found = false;
						for (uint32_t k = 0; k < used_texture_indices.size(); k++) {
							if (used_texture_indices[k] == tex_idx) {
								found = true;
								break;
							}
						}
						if (!found && used_texture_indices.size() < MAX_BOUND_TEXTURES - 1) {
							used_texture_indices.push_back(tex_idx);
						}
					}
				}
			}

			// Step 2: Build remap table
			// texture_remap[original_index] = remapped_slot (1-30), 0 = not mapped
			uint32_t* texture_remap = (uint32_t*)[g_mtl.raytrace_texture_remap_buffer contents];
			memset(texture_remap, 0, sizeof(uint32_t) * RT_MAX_TEXTURES);

			// Step 3: Build texture array and assign slots
			id<MTLTexture> texture_array[MAX_BOUND_TEXTURES];
			for (uint32_t i = 0; i < MAX_BOUND_TEXTURES; i++) {
				texture_array[i] = g_mtl.raster_white_texture; // Default to white
			}

			uint32_t next_slot = 1; // Start at slot 1 (slot 0 reserved for fallback white)
			for (uint32_t i = 0; i < used_texture_indices.size() && next_slot < MAX_BOUND_TEXTURES; i++) {
				uint32_t tex_idx = used_texture_indices[i];

				// Find the texture in slotmap by matching handle.index
				id<MTLTexture> found_tex = nil;

				g_texture_slotmap.ForEach([&](const TextureResource& tex_res) {
					if (tex_res.handle.index == tex_idx && tex_res.texture) {
						found_tex = tex_res.texture;
					}
				});

				if (found_tex) {
					texture_array[next_slot] = found_tex;
					texture_remap[tex_idx] = next_slot;
					next_slot++;
				}
			}

			uint32_t texture_count = next_slot;

			// Log texture remapping info
			static bool logged_remap = false;
			if (!logged_remap && used_texture_indices.size() > 0) {
				MetalFileLog("[Metal] Texture remap: %zu unique textures used, %u slots assigned",
					used_texture_indices.size(), texture_count - 1);
				for (uint32_t i = 0; i < used_texture_indices.size() && i < 10; i++) {
					uint32_t tex_idx = used_texture_indices[i];
					MetalFileLog("[Metal]   tex_idx %u -> slot %u", tex_idx, texture_remap[tex_idx]);
				}
				logged_remap = true;
			}

			// Fill scene constants
			RaytraceSceneConstants* scene = (RaytraceSceneConstants*)[g_mtl.raytrace_scene_buffer contents];
			scene->camera_position = g_mtl.scene.camera.position;
			scene->camera_forward = g_mtl.scene.camera.forward;
			scene->camera_right = g_mtl.scene.camera.right;
			scene->camera_up = g_mtl.scene.camera.up;
			// Use vfov from camera, fallback to 60 degrees if not set
			float vfov_degrees = g_mtl.scene.camera.vfov > 1.0f ? g_mtl.scene.camera.vfov : 60.0f;
			scene->vfov_radians = vfov_degrees * 3.14159f / 180.0f;
			scene->aspect_ratio = (float)w / (float)h;
			scene->render_width = w;
			scene->render_height = h;
			scene->instance_count = g_mtl.raytrace_instance_count;
			scene->total_triangles = total_tris;
			scene->debug_mode = 0;  // 0=normal, 1=UVs, 2=material index
			scene->texture_count = texture_count;
			scene->use_accel = tlas_built ? 1 : 0;
			scene->light_count = g_mtl.raytrace_light_count;
			MetalFileLog("[Metal] RaytraceRender: scene cam pos(%.2f %.2f %.2f) f(%.2f %.2f %.2f) vfov=%.2f textures=%u lights=%u accel=%u",
				scene->camera_position.x, scene->camera_position.y, scene->camera_position.z,
				scene->camera_forward.x, scene->camera_forward.y, scene->camera_forward.z,
				scene->vfov_radians, scene->texture_count, scene->light_count, scene->use_accel);

			if (g_mtl.raytrace_stats_buffer)
			{
				uint32_t* stats = (uint32_t*)[g_mtl.raytrace_stats_buffer contents];
				*stats = 0;
			}

			// Dispatch
			id<MTLCommandBuffer> cmd = [g_mtl.command_queue commandBuffer];
			id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
			[enc setComputePipelineState:g_mtl.raytrace_pipeline];
			[enc setTexture:g_mtl.raytrace_output_texture atIndex:0];
			// Bind texture array (textures 1-31)
			for (uint32_t i = 0; i < MAX_BOUND_TEXTURES; i++) {
				[enc setTexture:texture_array[i] atIndex:1 + i];
			}
			[enc setSamplerState:g_mtl.raytrace_sampler atIndex:0];
			[enc setBuffer:g_mtl.raytrace_scene_buffer offset:0 atIndex:0];
			[enc setBuffer:g_mtl.raytrace_instance_buffer offset:0 atIndex:1];
			[enc setBuffer:combined offset:0 atIndex:2];
			[enc setBuffer:g_mtl.raytrace_stats_buffer offset:0 atIndex:3];
			[enc setBuffer:g_mtl.raytrace_material_buffer offset:0 atIndex:4];
			[enc setBuffer:g_mtl.raytrace_material_edges_buffer offset:0 atIndex:5];
			[enc setBuffer:g_mtl.raytrace_material_indices_buffer offset:0 atIndex:6];
			[enc setBuffer:g_mtl.raytrace_texture_remap_buffer offset:0 atIndex:7];
			if (tlas_built && g_mtl.raytrace_tlas) {
				[enc setAccelerationStructure:g_mtl.raytrace_tlas atBufferIndex:8];
			}
			[enc setBuffer:g_mtl.raytrace_light_buffer offset:0 atIndex:9];

			MTLSize tpg = MTLSizeMake(8, 8, 1);
			MTLSize groups = MTLSizeMake((w+7)/8, (h+7)/8, 1);
			[enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
			[enc endEncoding];
			[cmd commit];
			[cmd waitUntilCompleted];
			if (cmd.status != MTLCommandBufferStatusCompleted)
			{
				MetalFileLog("[Metal] RaytraceRender: command buffer status=%ld error=%s",
					(long)cmd.status,
					cmd.error ? cmd.error.localizedDescription.UTF8String : "none");
			}

			MTL_LOG("RaytraceRender: %u instances, %u triangles at %ux%u",
					g_mtl.raytrace_instance_count, total_tris, w, h);
			MetalFileLog("[Metal] RaytraceRender: %u instances, %u triangles at %ux%u",
				g_mtl.raytrace_instance_count, total_tris, w, h);
			if (g_mtl.raytrace_stats_buffer)
			{
				uint32_t* stats = (uint32_t*)[g_mtl.raytrace_stats_buffer contents];
				uint32_t hit_count = *stats;
				uint32_t total_pixels = w * h;
				MTL_LOG("RaytraceRender: hits=%u of %u pixels", hit_count, total_pixels);
				MetalFileLog("[Metal] RaytraceRender: hits=%u of %u pixels", hit_count, total_pixels);
			}
			{
				static bool logged_sample_pixel = false;
				if (!logged_sample_pixel && g_mtl.raytrace_output_texture && g_mtl.raytrace_instance_count > 0)
				{
					uint32_t sample_x = w / 2;
					uint32_t sample_y = h / 2;
					uint8_t pixel[4] = {0, 0, 0, 0};
					MTLRegion region = MTLRegionMake2D(sample_x, sample_y, 1, 1);
					[g_mtl.raytrace_output_texture getBytes:pixel
												 bytesPerRow:4
												  fromRegion:region
												 mipmapLevel:0];
					MetalFileLog("[Metal] RaytraceRender: sample pixel (%u,%u) = %u %u %u %u",
						sample_x, sample_y, pixel[0], pixel[1], pixel[2], pixel[3]);
					logged_sample_pixel = true;
				}
			}
			{
				const RT_Vec3 cp = g_mtl.scene.camera.position;
				const RT_Vec3 cf = g_mtl.scene.camera.forward;
				const RT_Vec3 cr = g_mtl.scene.camera.right;
				const RT_Vec3 cu = g_mtl.scene.camera.up;
				MTL_LOG("RaytraceRender: cam pos(%.2f %.2f %.2f) f(%.2f %.2f %.2f) r(%.2f %.2f %.2f) u(%.2f %.2f %.2f)",
					cp.x, cp.y, cp.z, cf.x, cf.y, cf.z, cr.x, cr.y, cr.z, cu.x, cu.y, cu.z);
			}
		}

		g_mtl.raytrace_instance_count = 0;
		g_mtl.raytrace_light_count = 0;
		g_mtl.raytrace_pending_meshes.clear();
	}
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
		if (g_mtl.raster_render_target &&
			(!g_raster_batches.empty() || !g_raster_lines.empty()) &&
			(g_mtl.raster_tri_pipeline || g_mtl.raster_line_pipeline))
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
