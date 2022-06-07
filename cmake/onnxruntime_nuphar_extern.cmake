# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# this is for building extern functions in nuphar execution provider, using AVX2
# the separation from onnxruntime_providers.cmake is to avoid unnecessary AVX2 codegen in providers
# functions built here would be dynamically switched based on if AVX2 is available from CPUID

add_definitions(-DNUPHAR_USE_AVX2)

set(extern_avx2_srcs
  ${ONNXRUNTIME_ROOT}/core/providers/nuphar/extern/igemv_avx2.cc
  ${ONNXRUNTIME_ROOT}/core/providers/nuphar/extern/igemv_avx2.h
)

if (MSVC)
  set_source_files_properties(${extern_avx2_srcs} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
  set_source_files_properties(${extern_avx2_srcs} PROPERTIES COMPILE_FLAGS "-march=broadwell")
endif()

set(nuphar_extern_srcs
    ${extern_avx2_srcs}
)

onnxruntime_add_static_library(onnxruntime_nuphar_extern  ${nuphar_extern_srcs})

if (onnxruntime_USE_MKLML)
  add_definitions(-DNUPHAR_USE_MKL)
  target_include_directories(onnxruntime_nuphar_extern PRIVATE ${ONNXRUNTIME_ROOT}/core/providers/nuphar/extern ${MKLML_INCLUDE_DIR})
  add_dependencies(onnxruntime_nuphar_extern project_mklml)
else()
  target_include_directories(onnxruntime_nuphar_extern PRIVATE ${ONNXRUNTIME_ROOT}/core/providers/nuphar/extern)
endif()

set_target_properties(onnxruntime_nuphar_extern PROPERTIES FOLDER "ONNXRuntime")

list(APPEND onnxruntime_EXTERNAL_LIBRARIES onnxruntime_nuphar_extern)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES onnxruntime_nuphar_extern)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
