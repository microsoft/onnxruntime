# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(CheckLanguage)

if(NOT onnxruntime_ENABLE_PYTHON)
  message(FATAL_ERROR "python is required but is not enabled")
endif()

set(KERNEL_EXPLORER_ROOT ${ONNXRUNTIME_ROOT}/python/tools/kernel_explorer)

if (onnxruntime_USE_CUDA)
  check_language(CUDA)
  set(LANGUAGE CUDA)
  set(BERT_DIR ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert)
elseif(onnxruntime_USE_ROCM)
  check_language(HIP)
  set(LANGUAGE HIP)
  if (onnxruntime_USE_COMPOSABLE_KERNEL)
    include(composable_kernel)
  endif()
  if (onnxruntime_USE_HIPBLASLT)
    find_package(hipblaslt REQUIRED)
  endif()
  set(BERT_DIR ${ONNXRUNTIME_ROOT}/contrib_ops/rocm/bert)
endif()

file(GLOB kernel_explorer_srcs CONFIGURE_DEPENDS
  "${KERNEL_EXPLORER_ROOT}/*.cc"
  "${KERNEL_EXPLORER_ROOT}/*.h"
)

file(GLOB kernel_explorer_kernel_srcs CONFIGURE_DEPENDS
  "${KERNEL_EXPLORER_ROOT}/kernels/*.cc"
  "${KERNEL_EXPLORER_ROOT}/kernels/*.h"
  "${KERNEL_EXPLORER_ROOT}/kernels/*.cu"
  "${KERNEL_EXPLORER_ROOT}/kernels/*.cuh"
)

onnxruntime_add_shared_library_module(kernel_explorer ${kernel_explorer_srcs} ${kernel_explorer_kernel_srcs})
set_target_properties(kernel_explorer PROPERTIES PREFIX "_")
target_include_directories(kernel_explorer PUBLIC
  $<TARGET_PROPERTY:onnxruntime_pybind11_state,INCLUDE_DIRECTORIES>
  ${KERNEL_EXPLORER_ROOT})
target_link_libraries(kernel_explorer PRIVATE $<TARGET_PROPERTY:onnxruntime_pybind11_state,LINK_LIBRARIES>)
target_compile_definitions(kernel_explorer PRIVATE $<TARGET_PROPERTY:onnxruntime_pybind11_state,COMPILE_DEFINITIONS>)
target_compile_options(kernel_explorer PRIVATE -Wno-sign-compare)

if (onnxruntime_USE_CUDA)
  file(GLOB kernel_explorer_cuda_kernel_srcs CONFIGURE_DEPENDS
    "${KERNEL_EXPLORER_ROOT}/kernels/cuda/*.cc"
    "${KERNEL_EXPLORER_ROOT}/kernels/cuda/*.h"
    "${KERNEL_EXPLORER_ROOT}/kernels/cuda/*.cu"
    "${KERNEL_EXPLORER_ROOT}/kernels/cuda/*.cuh"
  )
  target_sources(kernel_explorer PRIVATE ${kernel_explorer_cuda_kernel_srcs})
  target_include_directories(kernel_explorer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
elseif (onnxruntime_USE_ROCM)
  file(GLOB kernel_explorer_rocm_kernel_srcs CONFIGURE_DEPENDS
    "${KERNEL_EXPLORER_ROOT}/kernels/rocm/*.cc"
    "${KERNEL_EXPLORER_ROOT}/kernels/rocm/*.h"
    "${KERNEL_EXPLORER_ROOT}/kernels/rocm/*.cu"
    "${KERNEL_EXPLORER_ROOT}/kernels/rocm/*.cuh"
  )
  auto_set_source_files_hip_language(${kernel_explorer_kernel_srcs} ${kernel_explorer_rocm_kernel_srcs})
  target_sources(kernel_explorer PRIVATE ${kernel_explorer_rocm_kernel_srcs})
  target_compile_definitions(kernel_explorer PRIVATE __HIP_PLATFORM_AMD__=1 __HIP_PLATFORM_HCC__=1)
  if (onnxruntime_USE_COMPOSABLE_KERNEL)
    target_compile_definitions(kernel_explorer PRIVATE USE_COMPOSABLE_KERNEL)
    target_link_libraries(kernel_explorer PRIVATE onnxruntime_composable_kernel_includes)
  endif()
  if (onnxruntime_USE_TRITON_KERNEL)
    target_compile_definitions(kernel_explorer PRIVATE USE_TRITON_KERNEL)
  endif()
  if (onnxruntime_USE_HIPBLASLT)
    target_compile_definitions(kernel_explorer PRIVATE USE_HIPBLASLT)
  endif()
  if (onnxruntime_USE_ROCBLAS_EXTENSION_API)
    target_compile_definitions(kernel_explorer PRIVATE USE_ROCBLAS_EXTENSION_API)
    target_compile_definitions(kernel_explorer PRIVATE ROCBLAS_NO_DEPRECATED_WARNINGS)
    target_compile_definitions(kernel_explorer PRIVATE ROCBLAS_BETA_FEATURES_API)
  endif()
endif()

add_dependencies(kernel_explorer onnxruntime_pybind11_state)

enable_testing()
find_package(Python COMPONENTS Interpreter REQUIRED)
add_test(NAME test_kernels COMMAND ${Python_EXECUTABLE} -m pytest ..)
