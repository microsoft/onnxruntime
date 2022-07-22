# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(CMAKE_CXX_COMPILER MATCHES ".*hipcc$")
  message(FATAL_ERROR "don't use hipcc!")
endif()

if(NOT onnxruntime_ENABLE_PYTHON)
  message(FATAL_ERROR "python is required but is not enabled")
endif()

if(NOT HIP_FOUND)
  message(FATAL_ERROR "hip is required but is not found")
endif()

set(KERNEL_EXPLORER_ROOT ${ONNXRUNTIME_ROOT}/python/tools/kernel_explorer)
set(BERT_DIR ${ONNXRUNTIME_ROOT}/contrib_ops/rocm/bert)

file(GLOB kernel_explorer_srcs CONFIGURE_DEPENDS "${KERNEL_EXPLORER_ROOT}/*.cc")
file(GLOB kernel_explorer_kernel_srcs CONFIGURE_DEPENDS "${KERNEL_EXPLORER_ROOT}/kernels/*.cc")

onnxruntime_add_shared_library_module(kernel_explorer
  ${kernel_explorer_srcs}
  ${kernel_explorer_kernel_srcs}
  ${BERT_DIR}/util.cc)
set_target_properties(kernel_explorer PROPERTIES PREFIX "_")
target_include_directories(kernel_explorer PUBLIC
  $<TARGET_PROPERTY:onnxruntime_pybind11_state,INCLUDE_DIRECTORIES>
  ${KERNEL_EXPLORER_ROOT})
target_link_libraries(kernel_explorer
  PRIVATE
    $<TARGET_PROPERTY:onnxruntime_pybind11_state,LINK_LIBRARIES>
    ${HIP_LIB})
target_compile_definitions(kernel_explorer
  PUBLIC ROCM_USE_FLOAT16
  PRIVATE $<TARGET_PROPERTY:onnxruntime_pybind11_state,COMPILE_DEFINITIONS>)

# handle kernel_explorer sources as hip language
target_compile_options(kernel_explorer PRIVATE "-xhip")
# TODO: use predefined AMDGPU_TARGETS
target_compile_options(kernel_explorer PRIVATE "--offload-arch=gfx906" "--offload-arch=gfx908" "--offload-arch=gfx90a")
# https://github.com/ROCm-Developer-Tools/HIP/blob/4514f350849b1090954295f8f87a5f8d78bd781b/hip-lang-config.cmake.in
target_link_libraries(kernel_explorer PRIVATE ${CLANGRT_BUILTINS})

add_dependencies(kernel_explorer onnxruntime_pybind11_state)

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME test_kernels COMMAND ${Python_EXECUTABLE} -m pytest ..)
