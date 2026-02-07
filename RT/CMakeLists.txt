# CMakeList.txt : Cmake project for including the RT code
cmake_minimum_required (VERSION 3.10)

set(ENV{RT_EXT_CODE_DIR}
"${CMAKE_CURRENT_LIST_DIR}"
)

message("RT Dir is: $ENV{RT_EXT_CODE_DIR}")

add_subdirectory("Core")
target_include_directories(RT_CORE INTERFACE ${CMAKE_CURRENT_LIST_DIR})
add_subdirectory("Renderer/Backend/Common")
if(${GRAPHICS_API} STREQUAL "DirectX12")
    add_subdirectory("Renderer/Backend/DX12")
elseif(${GRAPHICS_API} STREQUAL "Metal")
    add_subdirectory("Renderer/Backend/Metal")
elseif(${GRAPHICS_API} STREQUAL "Zig")
    add_subdirectory("Renderer/Backend/Zig")
elseif(${GRAPHICS_API} STREQUAL "Rust")
    add_subdirectory("Renderer/Backend/Rust")
endif()
