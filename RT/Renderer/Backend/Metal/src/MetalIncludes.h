#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>
#ifdef defer
#define RT_DEFER_WAS_DEFINED 1
#undef defer
#endif
#import <AppKit/AppKit.h>
#ifdef RT_DEFER_WAS_DEFINED
#undef RT_DEFER_WAS_DEFINED
#define defer const auto RT_PASTE(defer_, __LINE__) = DeferDoodadHelp() + [&]()
#endif
#else
typedef void *id;
#endif

#include "Core/Common.h"
#include "Core/Arena.h"

#define MTL_LOG(fmt, ...) fprintf(stderr, "[Metal] " fmt "\n", ##__VA_ARGS__)
#define MTL_STUB(name) MTL_LOG("STUB: %s not yet implemented", name)
