# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_providers_srcs
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.cc"
)

file(GLOB_RECURSE onnxruntime_contrib_ops_srcs
  "${ONNXRUNTIME_ROOT}/contrib_ops/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/*.cc"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.cc"
)

file(GLOB onnxruntime_providers_common_srcs
  "${ONNXRUNTIME_ROOT}/core/providers/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/*.cc"
)

if(onnxruntime_USE_MKLDNN)
  set(PROVIDERS_MKLDNN onnxruntime_providers_mkldnn)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES mkldnn)
endif()
if(onnxruntime_USE_CUDA)
  set(PROVIDERS_CUDA onnxruntime_providers_cuda)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES cuda)
endif()

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})
# add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_contrib_ops_srcs})
add_library(onnxruntime_providers ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs} ${onnxruntime_contrib_ops_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers onnx protobuf::libprotobuf)
target_include_directories(onnxruntime_providers PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
add_dependencies(onnxruntime_providers eigen gsl onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cpu  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
set_target_properties(onnxruntime_providers PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_providers PROPERTIES FOLDER "ONNXRuntime")

if (WIN32 AND onnxruntime_USE_OPENMP)
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
      add_definitions(/openmp)
  endif()
endif()

if (onnxruntime_USE_CUDA)
  file(GLOB_RECURSE onnxruntime_providers_cuda_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
  )
  file(GLOB_RECURSE onnxruntime_providers_cuda_cu_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cuh"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs})
  if(NOT WIN32)
    set(CUDA_CFLAGS " -std=c++14")
  endif()
  cuda_add_library(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs} OPTIONS ${CUDA_CFLAGS})
  onnxruntime_add_include_to_target(onnxruntime_providers_cuda onnx protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_cuda eigen ${onnxruntime_EXTERNAL_DEPENDENCIES} ${onnxruntime_tvm_dependencies})
  target_include_directories(onnxruntime_providers_cuda PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_CUDNN_HOME}/include ${eigen_INCLUDE_DIRS} ${TVM_INCLUDES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cuda  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_cuda PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(onnxruntime_providers_cuda PROPERTIES FOLDER "ONNXRuntime")
  if (WIN32)
    # *.cu cannot use PCH
    foreach(src_file ${onnxruntime_providers_cuda_cc_srcs})
      set_source_files_properties(${src_file}
        PROPERTIES
        COMPILE_FLAGS "/Yucuda_pch.h /FIcuda_pch.h")
    endforeach()
    set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.cc"
      PROPERTIES
      COMPILE_FLAGS "/Yccuda_pch.h"
    )
    # disable a warning from the CUDA headers about unreferenced local functions
    target_compile_options(onnxruntime_providers_cuda PRIVATE /wd4505)
    if (onnxruntime_USE_TVM)
      target_compile_options(onnxruntime_providers_cuda PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
    endif()
  endif()
endif()

if (onnxruntime_USE_MKLDNN)
  file(GLOB_RECURSE onnxruntime_providers_mkldnn_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/mkldnn/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/mkldnn/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_mkldnn_cc_srcs})
  add_library(onnxruntime_providers_mkldnn ${onnxruntime_providers_mkldnn_cc_srcs})  
  onnxruntime_add_include_to_target(onnxruntime_providers_mkldnn onnx protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_mkldnn eigen ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_mkldnn PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_mkldnn PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${MKLDNN_INCLUDE_DIR} ${MKLML_INCLUDE_DIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/mkldnn  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_mkldnn PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  include(onnxruntime_providers_internal.cmake)
endif()
