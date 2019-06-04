# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_definitions(-DNUPHAR_USE_AVX2=1)
add_definitions(-DNUPHAR_USE_MKL=1)

if (NOT onnxruntime_USE_MKLML)
  message(FATAL_ERROR "onnxruntime_USE_MKLML required for onnxruntime_USE_NUPHAR")
endif()

set(nblas_avx2_srcs
  ${ONNXRUNTIME_ROOT}/core/providers/nuphar/nblas/nblas_igemv_avx2.cc
  ${ONNXRUNTIME_ROOT}/core/providers/nuphar/nblas/nblas_igemv_avx2.h
)

set(nblas_mkl_srcs
  ${ONNXRUNTIME_ROOT}/core/providers/nuphar/nblas/nblas_igemv_mkl.cc
  ${ONNXRUNTIME_ROOT}/core/providers/nuphar/nblas/nblas_igemv_mkl.h
)

if (MSVC) 
#  string(APPEND CMAKE_CXX_FLAGS " /arch:AVX2")
  set_source_files_properties(${nblas_avx2_srcs} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
#  string(APPEND CMAKE_CXX_FLAGS " -march=broadwell")
  set_source_files_properties(${nblas_avx2_srcs} PROPERTIES COMPILE_FLAGS "-march=broadwell")
endif()

set(nuphar_blas_srcs
    ${nblas_avx2_srcs}
    ${nblas_mkl_srcs}
)

add_library(onnxruntime_nblas  ${nuphar_blas_srcs})
target_include_directories(onnxruntime_nblas PRIVATE ${ONNXRUNTIME_ROOT}/core/providers/nuphar/nblas ${MKLML_INCLUDE_DIR})
set_target_properties(onnxruntime_nblas PROPERTIES FOLDER "ONNXRuntime")
add_dependencies(onnxruntime_nblas project_mklml)

list(APPEND onnxruntime_EXTERNAL_LIBRARIES onnxruntime_nblas)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES onnxruntime_nblas)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
