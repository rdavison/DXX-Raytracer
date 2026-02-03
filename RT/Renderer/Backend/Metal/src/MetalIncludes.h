#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "Core/Common.h"
#include "Core/Arena.h"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/QuartzCore.h>
#import <AppKit/AppKit.h>
#else
typedef void *id;
#endif

#define MTL_LOG(fmt, ...) fprintf(stderr, "[Metal] " fmt "\n", ##__VA_ARGS__)
#define MTL_STUB(name) MTL_LOG("STUB: %s not yet implemented", name)
