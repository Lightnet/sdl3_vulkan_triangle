cmake_minimum_required(VERSION 3.22)
project(sdl3_vulkan_square C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(APP_NAME sdl_vulkan_square)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find SDL3
include(FetchContent)
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.2.14 # Changed to 'main' as 'release-3.2.14' may not exist
    GIT_SHALLOW TRUE
)
set(SDL_STATIC ON CACHE BOOL "Build SDL3 static library" FORCE)
set(SDL_SHARED OFF CACHE BOOL "Build SDL3 shared library" FORCE)
set(SDL_TEST OFF CACHE BOOL "Build SDL3 tests" FORCE)
FetchContent_MakeAvailable(SDL3)

# Shader compilation
set(VULKAN_SDK "C:/VulkanSDK/1.4.313.0" CACHE PATH "Path to Vulkan SDK")
find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator HINTS "${VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin")
if (NOT GLSLANG_VALIDATOR_EXECUTABLE)
    message(FATAL_ERROR "glslangValidator not found. Please install the Vulkan SDK at C:/VulkanSDK/1.4.313.0 or set VULKAN_SDK environment variable.")
endif()

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_FILES
    ${SHADER_DIR}/shader2d.vert
    ${SHADER_DIR}/shader2d.frag
)
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# Validate shader files
foreach(SHADER ${SHADER_FILES})
    if (NOT EXISTS ${SHADER})
        message(FATAL_ERROR "Shader file ${SHADER} does not exist.")
    endif()
endforeach()

set(SHADER_OUTPUT_FILES)
foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    get_filename_component(SHADER_BASE ${SHADER} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER} EXT)
    string(SUBSTRING ${SHADER_EXT} 1 -1 SHADER_TYPE) # Remove '.' from extension
    set(SHADER_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_BASE}_${SHADER_TYPE}_spv.h)
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} -V --vn ${SHADER_BASE}_${SHADER_TYPE}_spv ${SHADER} -o ${SHADER_OUTPUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME} to ${SHADER_OUTPUT}"
        VERBATIM
    )
    list(APPEND SHADER_OUTPUT_FILES ${SHADER_OUTPUT})
endforeach()

# Source files
set(SOURCE_FILES
    src/main.c
)

# Add executable
add_executable(${APP_NAME} ${SOURCE_FILES} ${SHADER_OUTPUT_FILES})

# Include directories
target_include_directories(${APP_NAME} PRIVATE
    ${CMAKE_BINARY_DIR}/include # Updated to match SHADER_OUTPUT_DIR
    ${SDL3_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${APP_NAME} PRIVATE
    SDL3::SDL3
    Vulkan::Vulkan
)

# Ensure shader headers are generated
add_custom_target(shaders DEPENDS ${SHADER_OUTPUT_FILES})
add_dependencies(${APP_NAME} shaders)