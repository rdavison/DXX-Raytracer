cmake_minimum_required(VERSION 3.13)

if(NOT APPLE)
    message(FATAL_ERROR "Rust backend (Metal) is only supported on Apple platforms")
endif()

project(Renderer_Rust)
set(CMAKE_CXX_STANDARD 17)

# Find Rust toolchain
find_program(CARGO_COMMAND cargo REQUIRED)
message(STATUS "Found Cargo: ${CARGO_COMMAND}")

# Set paths
set(RUST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RUST_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/target)

# Map CMake build type to Cargo profile
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_PROFILE "debug")
    set(CARGO_BUILD_FLAGS "")
else()
    set(CARGO_PROFILE "release")
    set(CARGO_BUILD_FLAGS "--release")
endif()

set(RUST_LIB_OUTPUT ${RUST_TARGET_DIR}/${CARGO_PROFILE}/librt_renderer.a)

# Custom command to build Rust library
add_custom_command(
    OUTPUT ${RUST_LIB_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${RUST_TARGET_DIR}
        ${CARGO_COMMAND} build ${CARGO_BUILD_FLAGS}
    WORKING_DIRECTORY ${RUST_SOURCE_DIR}
    DEPENDS
        ${RUST_SOURCE_DIR}/Cargo.toml
        ${RUST_SOURCE_DIR}/src/lib.rs
        ${RUST_SOURCE_DIR}/src/types.rs
        ${RUST_SOURCE_DIR}/src/device.rs
        ${RUST_SOURCE_DIR}/src/state.rs
        ${RUST_SOURCE_DIR}/src/shaders.rs
        ${RUST_SOURCE_DIR}/src/pipeline.rs
        ${RUST_SOURCE_DIR}/src/texture.rs
        ${RUST_SOURCE_DIR}/src/mesh.rs
        ${RUST_SOURCE_DIR}/src/raytrace_shader.rs
        ${RUST_SOURCE_DIR}/src/renderer/mod.rs
        ${RUST_SOURCE_DIR}/src/renderer/raytrace.rs
        ${RUST_SOURCE_DIR}/src/renderer/raster.rs
        ${RUST_SOURCE_DIR}/src/renderer/texture_remap.rs
    COMMENT "Building Rust Renderer library (${CARGO_PROFILE})"
)

# Create custom target for Rust build
add_custom_target(Renderer_Rust_Build DEPENDS ${RUST_LIB_OUTPUT})

# ImGui/ImPlot and utility sources (same as Metal/Zig backends)
set(METAL_BACKEND_DIR "${CMAKE_SOURCE_DIR}/RT/Renderer/Backend/Metal")

set(IMGUI_SOURCE_FILES
    "${METAL_BACKEND_DIR}/cimgui/cimgui.cpp"
    "${METAL_BACKEND_DIR}/cimgui/imgui/imgui.cpp"
    "${METAL_BACKEND_DIR}/cimgui/imgui/imgui_demo.cpp"
    "${METAL_BACKEND_DIR}/cimgui/imgui/imgui_draw.cpp"
    "${METAL_BACKEND_DIR}/cimgui/imgui/imgui_tables.cpp"
    "${METAL_BACKEND_DIR}/cimgui/imgui/imgui_widgets.cpp"
    "${METAL_BACKEND_DIR}/cimgui/imgui/backends/imgui_impl_metal.mm"
)

set(IMPLOT_SOURCE_FILES
    "${METAL_BACKEND_DIR}/implot/implot.cpp"
    "${METAL_BACKEND_DIR}/implot/implot_demo.cpp"
    "${METAL_BACKEND_DIR}/implot/implot_items.cpp"
)

# Utility sources from Metal backend (image loading, GLTF, mikktspace)
set(UTILITY_SOURCE_FILES
    "${METAL_BACKEND_DIR}/src/ImageReadWrite.cpp"
    "${METAL_BACKEND_DIR}/src/GLTFLoader.cpp"
    "${METAL_BACKEND_DIR}/src/mikktspace.c"
    "${METAL_BACKEND_DIR}/src/MeshTracker.cpp"
)

# Create ImGui/ImPlot support library
add_library(Renderer_ImGui STATIC ${IMGUI_SOURCE_FILES} ${IMPLOT_SOURCE_FILES})

set_target_properties(Renderer_ImGui PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED YES
)

target_compile_options(Renderer_ImGui PRIVATE "-fobjc-arc")

target_include_directories(Renderer_ImGui PRIVATE
    "${METAL_BACKEND_DIR}/cimgui"
    "${METAL_BACKEND_DIR}/cimgui/imgui"
    "${METAL_BACKEND_DIR}/cimgui/imgui/backends"
    "${METAL_BACKEND_DIR}/implot"
    "${CMAKE_SOURCE_DIR}/RT/Renderer/Backend/Common/include"
    "${CMAKE_SOURCE_DIR}/RT"
)

# Create Utilities library (image loading, GLTF, mikktspace)
add_library(Renderer_Utilities STATIC ${UTILITY_SOURCE_FILES})

set_target_properties(Renderer_Utilities PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

target_include_directories(Renderer_Utilities PRIVATE
    "${METAL_BACKEND_DIR}/src"
    "${METAL_BACKEND_DIR}/STB"
    "${METAL_BACKEND_DIR}/CGLTF"
    "${CMAKE_SOURCE_DIR}/RT/Renderer/Backend/Common/include"
    "${CMAKE_SOURCE_DIR}/RT"
)

# Create imported library for the Rust output
add_library(Renderer_Rust STATIC IMPORTED GLOBAL)
set_target_properties(Renderer_Rust PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB_OUTPUT}
)
add_dependencies(Renderer_Rust Renderer_Rust_Build)

# Create interface library that combines Rust renderer + ImGui + Utilities
add_library(Renderer INTERFACE)
target_link_libraries(Renderer INTERFACE
    Renderer_Rust
    Renderer_ImGui
    Renderer_Utilities
    Renderer_Common
    RT_CORE
    "-framework Metal"
    "-framework MetalKit"
    "-framework QuartzCore"
    "-framework AppKit"
    "-framework Foundation"
)
