
# Minimal version of CMake
cmake_minimum_required(VERSION 3.1)
set(CMAKE_BUILD_TYPE "Release")
# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()

# Define project name
project(GPU_KAKTWOOS C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/")

find_package(OpenCL REQUIRED)

include_directories(${OPENCL_INCLUDE_DIR})

# Source code of application
set(opencl_src main.c)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")

# Set up executable
add_executable(gpu ${opencl_src})
target_link_libraries(gpu ${OpenCL_LIBRARY})

add_custom_command(
        TARGET gpu PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/kaktwoos.cl
        ${CMAKE_CURRENT_BINARY_DIR}/kaktwoos.cl)

