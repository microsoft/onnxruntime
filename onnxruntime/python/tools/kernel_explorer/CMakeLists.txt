# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

project(kernel_explorer)
cmake_minimum_required(VERSION 3.21)

set(onnxruntime_DIR ../../../../onnxruntime/)
include(${onnxruntime_DIR}/../cmake/external/pybind11.cmake)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH ${onnxruntime_ROCM_HOME} ${onnxruntime_ROCM_HOME}/hip)

find_package(hip)
find_package(PythonLibs 3.8 EXACT REQUIRED)

set(onnxruntime_BERT_DIR ${onnxruntime_DIR}/contrib_ops/rocm/bert/)
set(onnxruntime_KE_DIR ${onnxruntime_DIR}/python/tools/kernel_explorer/)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${onnxruntime_DIR})

FILE(GLOB kernel_srcs ${onnxruntime_KE_DIR}/kernels/*.cc)
add_library(kernel_explorer SHARED kernel_explorer.cc ${onnxruntime_BERT_DIR}/util.cc ${kernel_srcs})
target_link_libraries(kernel_explorer ${PYTHON_LIBRARIES})
set_target_properties(kernel_explorer PROPERTIES PREFIX "")

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME test_kernels COMMAND ${Python_EXECUTABLE} -m pytest ..)
