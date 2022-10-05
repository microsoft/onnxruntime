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
enable_language(HIP)

include(composable_kernel)

set(KERNEL_EXPLORER_ROOT ${ONNXRUNTIME_ROOT}/python/tools/kernel_explorer)
set(BERT_DIR ${ONNXRUNTIME_ROOT}/contrib_ops/rocm/bert)

file(GLOB kernel_explorer_srcs CONFIGURE_DEPENDS "${KERNEL_EXPLORER_ROOT}/*.cc")
# NOTE: This should not be necessary, but hip* symbols are hiding by some ifdef in LANGUAGE CXX mode, weird...
set_source_files_properties(${kernel_explorer_srcs} PROPERTIES LANGUAGE HIP)

file(GLOB kernel_explorer_kernel_srcs CONFIGURE_DEPENDS "${KERNEL_EXPLORER_ROOT}/kernels/*.cc")
set_source_files_properties(${kernel_explorer_kernel_srcs} PROPERTIES LANGUAGE HIP)

onnxruntime_add_shared_library_module(kernel_explorer
  ${kernel_explorer_srcs}
  ${kernel_explorer_kernel_srcs})
set_target_properties(kernel_explorer PROPERTIES PREFIX "_")
target_include_directories(kernel_explorer PUBLIC
  $<TARGET_PROPERTY:onnxruntime_pybind11_state,INCLUDE_DIRECTORIES>
  ${KERNEL_EXPLORER_ROOT})
target_link_libraries(kernel_explorer
  PRIVATE
    $<TARGET_PROPERTY:onnxruntime_pybind11_state,LINK_LIBRARIES>
    onnxruntime_composable_kernel_includes
    # Currently we shall not use composablekernels::device_operations, the target includes all conv dependencies, which
    # are extremely slow to compile. Instead, we only link all gemm related objects. See the following link on updating.
    # https://github.com/ROCmSoftwarePlatform/composable_kernel/blob/85978e0201/library/src/tensor_operation_instance/gpu/CMakeLists.txt#L33-L54
    device_gemm_instance
    ${HIP_LIB})
target_compile_definitions(kernel_explorer
  PUBLIC ROCM_USE_FLOAT16
  PRIVATE $<TARGET_PROPERTY:onnxruntime_pybind11_state,COMPILE_DEFINITIONS>)
target_compile_options(kernel_explorer PRIVATE -Wno-sign-compare -D__HIP_PLATFORM_HCC__=1)

add_dependencies(kernel_explorer onnxruntime_pybind11_state)

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME test_kernels COMMAND ${Python_EXECUTABLE} -m pytest ..)
