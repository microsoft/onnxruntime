# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

project(kernel_explorer)
cmake_minimum_required(VERSION 3.21)

include(../../../../cmake/external/pybind11.cmake)

# Memory error with Release build (-o3 -DNDEBUG), AMD is currently investigating
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip)
find_package(PythonLibs 3.8 EXACT REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(.)

add_library(timer timer.cpp)
add_library(vector_add kernels/vector_add.cpp)
add_library(kernel_explorer SHARED kernel_explorer.cpp)
target_link_libraries(kernel_explorer timer vector_add ${PYTHON_LIBRARIES})
set_target_properties(kernel_explorer PROPERTIES PREFIX "")

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME test_vector_add COMMAND ${Python_EXECUTABLE} -m pytest ..)
