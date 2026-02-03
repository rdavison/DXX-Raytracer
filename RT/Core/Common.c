#ifdef _WIN32
#include <Windows.h>
#include <wchar.h>
#include <Shlwapi.h>

#pragma comment(lib, "shlwapi")
#elif defined(__APPLE__)
#include <libgen.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#endif

#include "Core/Common.h"
#include "Core/Arena.h"
#include "Core/String.h"

#define SUPPORT_STRING "For support, screenshot this message and visit the #support channel in our discord server:\n https://discord.gg/9dm93hKrnp\nOr create an issue on our GitHub:\nhttps://github.com/BredaUniversityGames/DXX-Raytracer"

void RT_FATAL_ERROR_(const char *explanation, const char *title, const char *file, int line)
{
#ifdef _WIN32
    char file_stripped[256];
    RT_SaneStrncpy(file_stripped, file, RT_ARRAY_COUNT(file_stripped));

    PathStripPathA(file_stripped);

    char *message = RT_ArenaPrintF(&g_thread_arena, "Location:\n%s:%d\n\nExplanation:\n%s\n\n" SUPPORT_STRING, file_stripped, line, explanation);
    MessageBoxA(NULL, message, title, MB_OK|MB_ICONERROR);

    __debugbreak(); // If you're using a debugger, now is your chance to debug.

    ExitProcess(1); // goodbye forever
#elif defined(__APPLE__)
    char file_stripped[256];
    RT_SaneStrncpy(file_stripped, file, RT_ARRAY_COUNT(file_stripped));

    const char *base = basename(file_stripped);
    char *message = RT_ArenaPrintF(&g_thread_arena, "%s\n\nLocation:\n%s:%d\n\nExplanation:\n%s\n\n" SUPPORT_STRING, title, base, line, explanation);
    fprintf(stderr, "%s\n", message);

    __builtin_trap(); // If you're using a debugger, now is your chance to debug.

    exit(1); // goodbye forever
#endif
}

#ifdef _WIN32
static LARGE_INTEGER perf_freq;
#endif

RT_HighResTime RT_GetHighResTime(void)
{
#ifdef _WIN32
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);

    RT_HighResTime result;
    result.value = value.QuadPart;
    return result;
#elif defined(__APPLE__)
    RT_HighResTime result;
    result.value = (int64_t)mach_absolute_time();
    return result;
#endif
}

double RT_SecondsElapsed(RT_HighResTime start, RT_HighResTime end)
{
#ifdef _WIN32
    if (!perf_freq.QuadPart)
    {
        QueryPerformanceFrequency(&perf_freq);
    }

    double result = (double)(end.value - start.value) / (double)perf_freq.QuadPart;
    return result;
#elif defined(__APPLE__)
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0)
    {
        mach_timebase_info(&timebase);
    }

    uint64_t delta = (uint64_t)(end.value - start.value);
    double nanos = (double)delta * (double)timebase.numer / (double)timebase.denom;
    return nanos / 1e9;
#endif
}
