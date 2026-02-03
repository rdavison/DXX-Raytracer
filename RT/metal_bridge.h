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
void metal_loadbmtexture_f(grs_bitmap* bm, int texfilt);
int metal_internal_string(int x, int y, const char* s);
void metal_init_font(grs_font* font);
void metal_font_choose_size(grs_font* font, int gap, int* rw, int* rh);
uint32_t* metal_load_bitmap_pixel_data(RT_Arena* arena, grs_bitmap* bitmap);
void metal_ulinec(int left, int top, int right, int bot, int c);
void metal_upixelc(int x, int y, int c);
void metal_drawcircle(int nsides, RT_Mat4* transform, RT_Vec4* col);
bool metal_ubitmapm_cs(int x, int y, int dw, int dh, grs_bitmap* bm, int c, int scale);
bool metal_ubitblt(int dw, int dh, int dx, int dy, int sw, int sh, int sx, int sy, grs_bitmap* src, grs_bitmap* dst, int texfilt);

#ifdef RT_METAL
#define dx12_start_frame         metal_start_frame
#define dx12_end_frame           metal_end_frame
#define dx12_set_render_target   metal_set_render_target
#define dx12_urect               metal_urect
#define dx12_ulinec              metal_ulinec
#define dx12_upixelc             metal_upixelc
#define dx12_drawcircle          metal_drawcircle
#define dx12_init_texture        metal_init_texture
#define dx12_loadbmtexture_f     metal_loadbmtexture_f
#define dx12_internal_string     metal_internal_string
#define dx12_init_font           metal_init_font
#define dx12_font_choose_size    metal_font_choose_size
#define dx12_load_bitmap_pixel_data metal_load_bitmap_pixel_data
#define dx12_ubitmapm_cs          metal_ubitmapm_cs
#define dx12_ubitblt              metal_ubitblt
#endif

#endif //_METAL_BRIDGE_H
