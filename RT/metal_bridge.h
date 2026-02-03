#ifndef _METAL_BRIDGE_H
#define _METAL_BRIDGE_H

#include "ApiTypes.h"
#include "Renderer.h"

#include "gr.h"

typedef struct _dx_texture
{
	RT_ResourceHandle handle;
	int w, h, tw, th, lw;
	float u, v;
} dx_texture;

void metal_start_frame();
void metal_end_frame();

void metal_set_render_target(RT_ResourceHandle hud_texture);
void metal_urect(int left, int top, int right, int bot);

void metal_init_texture(grs_bitmap* bm);
int metal_internal_string(int x, int y, const char* s);
void metal_init_font(grs_font* font);
uint32_t* metal_load_bitmap_pixel_data(RT_Arena* arena, grs_bitmap* bitmap);
bool metal_ubitmapm_cs(int x, int y, int dw, int dh, grs_bitmap* bm, int c, int scale);
bool metal_ubitblt(int dw, int dh, int dx, int dy, int sw, int sh, int sx, int sy, grs_bitmap* src, grs_bitmap* dst, int texfilt);

#endif //_METAL_BRIDGE_H
