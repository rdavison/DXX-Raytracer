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
		id<MTLBuffer> position_buffer;              // float3 positions only (3 per tri, for BLAS)
		id<MTLAccelerationStructure> blas;           // Bottom-level acceleration structure
#else
		id triangle_buffer;
		id position_buffer;
		id blas;
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
		uint32_t material_override;    // Material index override (0 = use triangle's material)
		uint32_t triangle_offset;      // Offset into combined triangle buffer
		uint32_t _pad[3];              // Pad to 16-byte alignment (160 bytes total)
	};

	// GPU-compatible material data (matches DX12 Material struct)
	struct GPUMaterial
	{
		uint32_t albedo_index;
		uint32_t normal_index;
		uint32_t metalness_index;
		uint32_t roughness_index;
		uint32_t emissive_index;
		uint32_t height_index;
		uint32_t flags;
		float    metalness_factor;
		float    roughness_factor;
		uint32_t emissive_factor;
		uint32_t _pad[2];  // Pad to 48 bytes for alignment
	};
	static_assert(sizeof(GPUMaterial) == 48, "GPUMaterial size mismatch");

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
		uint32_t texture_count;
		uint32_t use_accel;         // 1 = use acceleration structure, 0 = brute force
		uint32_t light_count;
	};
	static_assert(sizeof(RaytraceSceneConstants) == 112, "RaytraceSceneConstants size mismatch");
	static_assert(offsetof(RaytraceSceneConstants, render_width) == 72, "RaytraceSceneConstants render_width offset mismatch");
	static_assert(offsetof(RaytraceSceneConstants, instance_count) == 80, "RaytraceSceneConstants instance_count offset mismatch");
	static_assert(offsetof(RaytraceSceneConstants, debug_mode) == 96, "RaytraceSceneConstants debug_mode offset mismatch");
	static_assert(offsetof(RaytraceSceneConstants, texture_count) == 100, "RaytraceSceneConstants texture_count offset mismatch");

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
	// Material system buffers
	id<MTLBuffer> raytrace_material_buffer;        // GPUMaterial array
	id<MTLBuffer> raytrace_material_edges_buffer;  // RT_MaterialEdge array
	id<MTLBuffer> raytrace_material_indices_buffer; // uint16_t array
	id<MTLBuffer> raytrace_texture_remap_buffer;   // Texture index remapping table
	id<MTLSamplerState> raytrace_sampler;          // Texture sampler
	// Acceleration structure state
	id<MTLAccelerationStructure> raytrace_tlas;    // Top-level acceleration structure
	id<MTLBuffer> raytrace_instance_desc_buffer;   // MTLAccelerationStructureInstanceDescriptor array
	id<MTLBuffer> raytrace_tlas_scratch_buffer;    // Scratch buffer for TLAS builds
	// Light buffer
	id<MTLBuffer> raytrace_light_buffer;           // RT_Light array
	// Argument buffer (Tier 2): holds all textures by natural index, eliminates 31-slot limit
	id<MTLBuffer> raytrace_argument_buffer;        // Encoded texture array for bindless access
	id<MTLArgumentEncoder> raytrace_arg_encoder;   // Encoder for the argument buffer
#else
	id raytrace_pipeline;
	id raytrace_instance_buffer;
	id raytrace_scene_buffer;
	id raytrace_stats_buffer;
	id raytrace_output_texture;
	id raytrace_material_buffer;
	id raytrace_material_edges_buffer;
	id raytrace_material_indices_buffer;
	id raytrace_texture_remap_buffer;
	id raytrace_sampler;
	id raytrace_tlas;
	id raytrace_instance_desc_buffer;
	id raytrace_tlas_scratch_buffer;
	id raytrace_light_buffer;
	id raytrace_argument_buffer;
	id raytrace_arg_encoder;
#endif
		uint32_t raytrace_instance_count;
		uint32_t raytrace_light_count;
		std::vector<RT_ResourceHandle> raytrace_pending_meshes;
		bool raytrace_materials_dirty;  // Flag to rebuild material buffer
		bool use_argument_buffers;      // True if device supports Tier 2 argument buffers

		// Async pipeline state
		uint32_t tlas_buffer_index;     // Double-buffered TLAS: alternates 0/1
#ifdef __OBJC__
		id<MTLAccelerationStructure> raytrace_tlas_double[2];  // Two TLAS buffers
		id<MTLBuffer> raytrace_tlas_scratch_double[2];
		dispatch_semaphore_t compute_semaphore;                // Frame-pacing semaphore for async dispatch
#else
		id raytrace_tlas_double[2];
		id raytrace_tlas_scratch_double[2];
		void* compute_semaphore;
#endif
		bool texture_remap_dirty;       // Rescan triangles for texture remap only when mesh changes

		// Deferred resource deletion (freed after GPU finishes using them)
		std::vector<RT_ResourceHandle> pending_mesh_releases;
		std::vector<RT_ResourceHandle> pending_texture_releases;

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
