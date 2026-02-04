#pragma once

#include "MetalIncludes.h"

#include "Core/MemoryScope.hpp"
#include "Core/SlotMap.hpp"
#include "Core/Config.h"
#include "Core/String.h"

#include "Renderer.h"
#include "MeshTracker.hpp"

#include <cstddef>
#include <vector>

#ifdef __OBJC__
#include <dispatch/dispatch.h>
#endif

struct ImDrawData;

namespace RT
{

	// Minimal stub — the DX12 backend generates this from shared shader headers.
	// Metal will get its own tweak vars once shaders are implemented.
	struct TweakVars
	{
		int placeholder;
	};

	extern TweakVars tweak_vars;

	constexpr uint32_t BACK_BUFFER_COUNT      = 3;
	constexpr uint32_t MAX_INSTANCES          = 1000;
	constexpr uint32_t MAX_RASTER_TRIANGLES   = 10000;
	constexpr uint32_t MAX_RASTER_LINES       = 5000;
	constexpr uint32_t MAX_DEBUG_LINES_WORLD  = 5000;
	constexpr uint32_t MAX_BOTTOM_LEVELS      = 1000;

	struct MeshResource
	{
#ifdef __OBJC__
		id<MTLBuffer> triangle_buffer;
#else
		id triangle_buffer;
#endif
		uint32_t triangle_count;
	};

	// GPU-compatible instance data for raytracing
	struct RaytraceInstance
	{
		RT_Mat4 object_to_world;
		RT_Mat4 world_to_object;
		uint32_t triangle_buffer_idx;  // Index into mesh slotmap
		uint32_t triangle_count;
		uint32_t color;
		uint32_t _pad;
	};

	// Scene constants for compute shader
	struct RaytraceSceneConstants
	{
		RT_Vec3 camera_position;  float _pad0;
		RT_Vec3 camera_forward;   float _pad1;
		RT_Vec3 camera_right;     float _pad2;
		RT_Vec3 camera_up;        float _pad3;
		float vfov_radians;
		float aspect_ratio;
		uint32_t render_width;
		uint32_t render_height;
		uint32_t instance_count;
		uint32_t total_triangles;
		float _pad4[2];
		uint32_t debug_mode;
		uint32_t _pad5[3];
	};
	static_assert(sizeof(RaytraceSceneConstants) == 112, "RaytraceSceneConstants size mismatch");
	static_assert(offsetof(RaytraceSceneConstants, render_width) == 72, "RaytraceSceneConstants render_width offset mismatch");
	static_assert(offsetof(RaytraceSceneConstants, instance_count) == 80, "RaytraceSceneConstants instance_count offset mismatch");
	static_assert(offsetof(RaytraceSceneConstants, debug_mode) == 96, "RaytraceSceneConstants debug_mode offset mismatch");

	struct TextureResource
	{
		RT_ResourceHandle handle;
#ifdef __OBJC__
		id<MTLTexture> texture;
#else
		id texture;
#endif
	};

	struct FrameData
	{
#ifdef __OBJC__
		dispatch_semaphore_t semaphore;
		id<MTLCommandBuffer> command_buffer;
#else
		void *semaphore;
		id command_buffer;
#endif
	};

	struct MetalState
	{
		// Window / layer
#ifdef __OBJC__
		NSWindow     *window;
		CAMetalLayer *metal_layer;
#else
		void *window;
		void *metal_layer;
#endif

		// Device
#ifdef __OBJC__
		id<MTLDevice>       device;
		id<MTLCommandQueue> command_queue;
#else
		id device;
		id command_queue;
#endif

		// IO
		RT_RendererIO io;
		RT_Arena     *arena;

		// Resolution
		uint32_t output_width;
		uint32_t output_height;
		uint32_t render_width;
		uint32_t render_height;
		uint32_t render_width_override;
		uint32_t render_height_override;

		// Frame
		uint64_t  frame_index;
		uint32_t  current_back_buffer_index;
		FrameData frame_data[BACK_BUFFER_COUNT];

		// Sync
#ifdef __OBJC__
		dispatch_semaphore_t frame_semaphore;
#else
		void *frame_semaphore;
#endif

		// Default resources
		RT_ResourceHandle white_texture_handle;
		RT_ResourceHandle black_texture_handle;
		RT_ResourceHandle billboard_quad;
		RT_ResourceHandle cube;

		// Scene
		struct SceneData
		{
			RT_Camera camera;
			RT_Camera prev_camera;
			bool render_blit;
		} scene;

		// Misc
		bool  queued_screenshot;
		char  queued_screenshot_name[1024];
		float viewport_offset_y;
		float viewport_x;
		float viewport_y;
		float viewport_width;
		float viewport_height;
		bool  raster_render_requested;
		RT_ResourceHandle raster_render_target_handle;

#ifdef __OBJC__
		id<MTLRenderPipelineState> raster_tri_pipeline;
		id<MTLRenderPipelineState> raster_line_pipeline;
		id<MTLSamplerState> raster_sampler;
		id<MTLTexture> raster_white_texture;
		id<MTLTexture> raster_render_target;
#else
		id raster_tri_pipeline;
		id raster_line_pipeline;
		id raster_sampler;
		id raster_white_texture;
		id raster_render_target;
#endif
		bool  imgui_render_requested;
		::ImDrawData *imgui_draw_data;
		float imgui_last_scale_x;
		float imgui_last_scale_y;

	// Raytracing state
#ifdef __OBJC__
	id<MTLComputePipelineState> raytrace_pipeline;
	id<MTLBuffer> raytrace_instance_buffer;
	id<MTLBuffer> raytrace_scene_buffer;
	id<MTLBuffer> raytrace_stats_buffer;
	id<MTLTexture> raytrace_output_texture;
#else
	id raytrace_pipeline;
	id raytrace_instance_buffer;
	id raytrace_scene_buffer;
	id raytrace_stats_buffer;
	id raytrace_output_texture;
#endif
		uint32_t raytrace_instance_count;
		std::vector<RT_ResourceHandle> raytrace_pending_meshes;

		// Mesh tracking
		MeshTracker mesh_tracker;
	};

	extern MetalState g_mtl;

	extern RT::SlotMap<MeshResource>   g_mesh_slotmap;
	extern RT::SlotMap<TextureResource> g_texture_slotmap;

	// ------------------------------------------------------------------
	// Helper functions

	inline FrameData *CurrentFrameData()
	{
		return &g_mtl.frame_data[g_mtl.current_back_buffer_index];
	}
}
