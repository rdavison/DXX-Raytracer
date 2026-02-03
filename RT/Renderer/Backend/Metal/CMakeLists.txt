cmake_minimum_required(VERSION 3.13)

if(NOT APPLE)
    message(FATAL_ERROR "Metal backend is only supported on Apple platforms")
endif()

project(Renderer_Metal)
set(CMAKE_CXX_STANDARD 17)

set(IMGUI_SOURCE_FILES
    "cimgui/cimgui.cpp"
    "cimgui/imgui/imgui.cpp"
    "cimgui/imgui/imgui_demo.cpp"
    "cimgui/imgui/imgui_draw.cpp"
    "cimgui/imgui/imgui_tables.cpp"
    "cimgui/imgui/imgui_widgets.cpp"
    "cimgui/imgui/backends/imgui_impl_metal.mm"
    "cimgui/imgui/backends/imgui_impl_osx.mm"
)

set(IMPLOT_SOURCE_FILES
    "implot/implot.cpp"
    "implot/implot_demo.cpp"
    "implot/implot_items.cpp"
)

set(Renderer_SOURCE
    "src/Renderer.mm"
    "src/RenderBackend.mm"
    "src/ImageReadWrite.cpp"
    "src/GLTFLoader.cpp"
    "src/mikktspace.c"
    "src/MeshTracker.cpp"
    ${IMGUI_SOURCE_FILES}
    ${IMPLOT_SOURCE_FILES}
)

add_library(Renderer SHARED ${Renderer_SOURCE})

set_target_properties(Renderer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED YES
    MACOSX_RPATH YES
)

target_compile_options(Renderer PRIVATE "-fobjc-arc")

target_include_directories(Renderer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/cimgui"
    "${CMAKE_CURRENT_SOURCE_DIR}/cimgui/imgui"
    "${CMAKE_CURRENT_SOURCE_DIR}/implot"
    "${CMAKE_CURRENT_SOURCE_DIR}/STB"
    "${CMAKE_CURRENT_SOURCE_DIR}/CGLTF"
    "${CMAKE_SOURCE_DIR}/RT/Renderer/Backend/Common/include"
    "${CMAKE_SOURCE_DIR}/RT/Core"
)

target_link_libraries(Renderer
    Renderer_Common
    RT_CORE
    "-framework Metal"
    "-framework MetalKit"
    "-framework QuartzCore"
    "-framework AppKit"
    "-framework Foundation"
)
