/*
THE COMPUTER CODE CONTAINED HEREIN IS THE SOLE PROPERTY OF PARALLAX
SOFTWARE CORPORATION ("PARALLAX").  PARALLAX, IN DISTRIBUTING THE CODE TO
END-USERS, AND SUBJECT TO ALL OF THE TERMS AND CONDITIONS HEREIN, GRANTS A
ROYALTY-FREE, PERPETUAL LICENSE TO SUCH END-USERS FOR USE BY SUCH END-USERS
IN USING, DISPLAYING,  AND CREATING DERIVATIVE WORKS THEREOF, SO LONG AS
SUCH USE, DISPLAY OR CREATION IS FOR NON-COMMERCIAL, ROYALTY OR REVENUE
FREE PURPOSES.  IN NO EVENT SHALL THE END-USER USE THE COMPUTER CODE
CONTAINED HEREIN FOR REVENUE-BEARING PURPOSES.  THE END-USER UNDERSTANDS
AND AGREES TO THE TERMS HEREIN AND ACCEPTS THE SAME BY USE OF THIS FILE.
COPYRIGHT 1993-1998 PARALLAX SOFTWARE CORPORATION.  ALL RIGHTS RESERVED.
*/
/*
 *
 * Graphical routines for drawing rectangles.
 *
 */

#include "u_mem.h"

#include "gr.h"
#include "grdef.h"
#include <stdio.h>

#if defined(RT_METAL)
static int g_gr_rect_log_count = 0;
static int g_gr_urect_log_count = 0;
#endif

#ifdef OGL
#include "ogl_init.h"
#endif


void gr_urect(int left,int top,int right,int bot)
{
#if defined(RT_METAL)
	if (g_gr_urect_log_count < 50) {
		int w = right - left + 1;
		int h = bot - top + 1;
		fprintf(stderr, "[Metal] gr_urect: w=%d h=%d type=%d\n", w, h, (int)TYPE);
		++g_gr_urect_log_count;
	}
#endif
#ifdef OGL
	if (TYPE == BM_OGL) {
		ogl_urect(left,top,right,bot);
		return;
	}
#elif defined(RT_DX12) || defined(RT_METAL)
	if (TYPE == BM_OGL || TYPE == BM_RTDX12) {
		dx12_urect(left, top, right, bot);
		return;
	}
#else
	int i;

	for ( i=top; i<=bot; i++ )
		gr_uscanline( left, right, i );
#endif
}

void gr_rect(int left,int top,int right,int bot)
{
	int i;

#if defined(RT_METAL)
	if (g_gr_rect_log_count < 50) {
		int w = right - left + 1;
		int h = bot - top + 1;
		fprintf(stderr, "[Metal] gr_rect: w=%d h=%d type=%d\n", w, h, (int)TYPE);
		++g_gr_rect_log_count;
	}
#endif
#ifdef OGL
	if (TYPE == BM_OGL) {
		ogl_urect(left,top,right,bot);
		return;
	}
#elif defined(RT_DX12) || defined(RT_METAL)
	if (TYPE == BM_OGL || TYPE == BM_RTDX12) {
		dx12_urect(left, top, right, bot);
		return;
	}
#endif
	for ( i=top; i<=bot; i++ )
		gr_scanline( left, right, i );
}
