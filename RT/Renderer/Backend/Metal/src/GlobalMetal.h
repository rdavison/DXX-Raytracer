#pragma once

#include "MetalIncludes.h"

#include "Core/MemoryScope.hpp"
#include "Core/SlotMap.hpp"
#include "Core/Config.h"
#include "Core/String.h"

#include "Renderer.h"
#include "MeshTracker.hpp"

#ifdef __OBJC__
#include <dispatch/dispatch.h>
#endif

namespace RT
{
	struct ImDrawData;

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
		id<MTLBuffer> vertex_buffer;
#else
		id vertex_buffer;
#endif
		uint32_t triangle_count;
	};

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
		bool  imgui_render_requested;
		ImDrawData *imgui_draw_data;

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
