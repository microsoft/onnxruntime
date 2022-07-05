# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

project(kernel_explorer)
cmake_minimum_required(VERSION 3.21)

if(CMAKE_CXX_COMPILER MATCHES ".*hipcc$")
  message(FATAL_ERROR "don't use hipcc!")
endif()

if(CMAKE_PROJECT_NAME STREQUAL "kernel_explorer")
  message(FATAL_ERROR "kernel explorer cannot be built as a top-level project at the moment")
endif()

if(NOT onnxruntime_ENABLE_PYTHON)
  message(FATAL_ERROR "python is required but is not enabled")
endif()

if(NOT HIP_FOUND)
  message(FATAL_ERROR "hip is required but is not found")
endif()

set(BERT_DIR ${REPO_ROOT}/onnxruntime/contrib_ops/rocm/bert/)

file(GLOB kernel_srcs CONFIGURE_DEPENDS "kernels/*.cc")
add_library(kernel_explorer SHARED kernel_explorer.cc ${BERT_DIR}/timer.cc ${kernel_srcs})
set_target_properties(kernel_explorer PROPERTIES PREFIX "")
target_include_directories(kernel_explorer PUBLIC $<TARGET_PROPERTY:onnxruntime_pybind11_state,INCLUDE_DIRECTORIES> ".")
target_link_libraries(kernel_explorer
  PRIVATE
    $<TARGET_PROPERTY:onnxruntime_pybind11_state,LINK_LIBRARIES>
    ${HIP_LIB})
target_compile_definitions(kernel_explorer
  PUBLIC ROCM_USE_FLOAT16
  PRIVATE $<TARGET_PROPERTY:onnxruntime_pybind11_state,COMPILE_DEFINITIONS>)
target_compile_options(kernel_explorer PRIVATE "-xhip")
# TODO: use predefined AMDGPU_TARGETS
target_compile_options(kernel_explorer PRIVATE "--offload-arch=gfx906" "--offload-arch=gfx908" "--offload-arch=gfx90a")

add_dependencies(kernel_explorer onnxruntime_pybind11_state)

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME test_kernels COMMAND ${Python_EXECUTABLE} -m pytest ..)
