#include "ImageReadWrite.h"

#include "Core/Arena.h"
#include "Core/String.h"
#include "Core/FileIO.h"
#include "Core/Vault.h"

// TODO(daniel): These external libraries are in a weird place... RT/Renderer/Backend/DX12? This has nothing to do with DX12!!!!!!!!

#define STBI_ASSERT(x) RT_ASSERT(x)
#define STBI_MALLOC(size) RT_ArenaAllocNoZero(&g_thread_arena, size, 16)
#define STBI_REALLOC_SIZED(pointer, old_size, new_size) RT_ArenaResize(&g_thread_arena, pointer, old_size, new_size, 16)
#define STBI_FREE(pointer) (void)(0) /* no-op */

#define STB_IMAGE_IMPLEMENTATION

#pragma warning(push, 0)
#include "../STB/stb_image.h"
#pragma pop

#define STBIW_ASSERT(x) RT_ASSERT(x)
#define STBIW_MALLOC(size) RT_ArenaAllocNoZero(&g_thread_arena, size, 16)
#define STBIW_REALLOC_SIZED(pointer, old_size, new_size) RT_ArenaResize(&g_thread_arena, pointer, old_size, new_size, 16)
#define STBIW_FREE(pointer) (void)(0) /* no-op */

#define STB_IMAGE_WRITE_IMPLEMENTATION

#pragma warning(push, 0)
#include "../STB/stb_image_write.h"
#pragma pop

RT_Image RT_LoadImageFromDisk(RT_Arena *arena, const char *path_c, int required_channel_count, bool is_srgb)
{
	RT_Image result = {};

	RT_ArenaMarker marker = RT_ArenaGetMarker(&g_thread_arena);

	RT_String path = RT_StringFromCString(path_c);
	RT_String ext  = RT_StringFindExtension(path);

	if (RT_StringsAreEqualNoCase(ext, RT_StringLiteral(".dds")))
	{
		RT_String memory;
		if (RT_ReadEntireFile(arena, path, &memory))
		{
			result = RT_LoadDDSFromMemory(memory);
		}
	}
	else
	{
		
		int w=0, h=0, channel_count=0;

		RT_String file_buffer;
		file_buffer.bytes = nullptr;
		file_buffer.count = 0;

		// first try to load from vault
		if (RT_GetFileFromVaults(path, file_buffer))
		{
			result.pixels = stbi_load_from_memory((unsigned char*)(file_buffer.bytes),file_buffer.count,&w,&h, &channel_count, required_channel_count);
		}

		// try to load from disk
		if (!result.pixels)
		{
			result.pixels = stbi_load(path_c, &w, &h, &channel_count, required_channel_count);
		}

		if (result.pixels)
		{
			result.width           = (uint32_t)w;
			result.height          = (uint32_t)h;
			result.pitch           = required_channel_count*result.width;
			result.mip_count       = 1;

			switch (required_channel_count)
			{
				case 4: result.format = RT_TextureFormat_RGBA8; break;
				case 1: result.format = RT_TextureFormat_R8;    break;
				RT_INVALID_DEFAULT_CASE;
			}

			if (arena != &g_thread_arena)
			{
				// stbi_load will always use the thread arena, but if we wanted the result to be in a different arena
				// we'll just copy it. This means that the destination arena isn't polluted by intermediate allocations
				// that stbi_load did to load the image.
				result.pixels = RT_ArenaCopy(arena, result.pixels, w*h*required_channel_count, 16);
				// and then we know we don't need the thread arena stuff anymore so free it
				RT_ArenaResetToMarker(&g_thread_arena, marker);
			}
		}
	}

	if (is_srgb)
	{
		result.format = RT_TextureFormatToSRGB(result.format);
	}

	return result;
}

RT_Image RT_LoadImageFromMemory(RT_Arena *arena, const void *memory, size_t memory_size, int required_channel_count, bool is_srgb)
{
	RT_Image result = {};

	RT_ArenaMarker marker = RT_ArenaGetMarker(&g_thread_arena);

	int w, h, channel_count;
	result.pixels = stbi_load_from_memory((const stbi_uc *)memory, memory_size, &w, &h, &channel_count, required_channel_count);

	result.width     = (uint32_t)w;
	result.height    = (uint32_t)h;
	result.pitch     = required_channel_count*result.width;
	result.mip_count = 1;

	switch (required_channel_count)
	{
		case 4: result.format = RT_TextureFormat_RGBA8; break;
		case 1: result.format = RT_TextureFormat_R8;    break;
		RT_INVALID_DEFAULT_CASE;
	}

	if (is_srgb)
	{
		result.format = RT_TextureFormatToSRGB(result.format);
	}

	if (arena != &g_thread_arena)
	{
		result.pixels = RT_ArenaCopy(arena, result.pixels, w*h*required_channel_count, 16);
		RT_ArenaResetToMarker(&g_thread_arena, marker);
	}

	return result;
}

void RT_WritePNGToDisk(const char *path, int w, int h, int channel_count, const void *pixels, int stride_in_bytes)
{
	RT_ArenaMemoryScope(&g_thread_arena)
	{
		int result = stbi_write_png(path, w, h, channel_count, pixels, stride_in_bytes);

		if (!result)
		{
			fprintf(stderr, "[RT_WritePNGToDisk]: ERROR: Failed to write png: '%s'\n", path);
		}
	}
}

RT_Image RT_LoadDDSFromMemory(RT_String memory)
{
	(void)memory;
	fprintf(stderr, "[RT_LoadDDSFromMemory]: DDS loading not supported on Metal backend yet.\n");
	return {};
}

RT_Image RT_LoadDDSFromDisk(RT_Arena *arena, RT_String path)
{
	(void)arena;
	(void)path;
	fprintf(stderr, "[RT_LoadDDSFromDisk]: DDS loading not supported on Metal backend yet.\n");
	return {};
}
