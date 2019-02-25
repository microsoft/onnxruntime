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
onnxruntime_add_include_to_target(onnxruntime_providers onnxruntime_common onnxruntime_framework gsl onnx onnx_proto protobuf::libprotobuf)
set(gemmlowp_src ${ONNXRUNTIME_ROOT}/../cmake/external/gemmlowp)
set(re2_src ${ONNXRUNTIME_ROOT}/../cmake/external/re2)
target_include_directories(onnxruntime_providers PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${gemmlowp_src} ${re2_src})
add_dependencies(onnxruntime_providers eigen gsl onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cpu  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
set_target_properties(onnxruntime_providers PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_providers PROPERTIES FOLDER "ONNXRuntime")

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
  add_library(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_cu_srcs})
  if (UNIX)
    target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-reorder>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-reorder>")
  endif()
  onnxruntime_add_include_to_target(onnxruntime_providers_cuda onnxruntime_common onnxruntime_framework gsl onnx onnx_proto protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_cuda eigen ${onnxruntime_EXTERNAL_DEPENDENCIES} ${onnxruntime_tvm_dependencies})
  target_include_directories(onnxruntime_providers_cuda PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_CUDNN_HOME}/include ${eigen_INCLUDE_DIRS} ${TVM_INCLUDES} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
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
    #target_compile_options(onnxruntime_providers_cuda PRIVATE /wd4505)
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
  onnxruntime_add_include_to_target(onnxruntime_providers_mkldnn gsl onnxruntime_common onnxruntime_framework gsl onnx onnx_proto protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_mkldnn eigen ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_mkldnn PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_mkldnn PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${MKLDNN_INCLUDE_DIR})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/mkldnn  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_mkldnn PROPERTIES LINKER_LANGUAGE CXX)
endif()

if (onnxruntime_USE_TRT)
  add_definitions(-DUSE_TRT=1)
  add_definitions("-DONNX_ML=1")
  add_definitions("-DONNX_NAMESPACE=onnx")
  include_directories(${PROJECT_SOURCE_DIR}/external/protobuf)
  set(CUDA_INCLUDE_DIRS ${CUDA_HOME}/include)
  include_directories(${ONNXRUNTIME_ROOT}/../cmake/external/onnx)
  if (WIN32)
    set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(OLD_CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4244 /wd4267 /wd4099 /wd4551 /wd4505 /wd4515 /wd4706")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -include algorithm")
    set(PROTOBUF_LIBRARY libprotobuf)
    set(DISABLED_WARNINGS_FOR_TRT /wd4267 /wd4244 /wd4996)
  endif()
  add_subdirectory(${ONNXRUNTIME_ROOT}/../cmake/external/onnx-tensorrt)
  if (WIN32)
    set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
    set(CMAKE_CUDA_FLAGS ${OLD_CMAKE_CUDA_FLAGS})
    unset(PROTOBUF_LIBRARY)
    unset(OLD_CMAKE_CXX_FLAGS)
    unset(OLD_CMAKE_CUDA_FLAGS)
    set_target_properties(nvonnxparser PROPERTIES LINK_FLAGS "/ignore:4199")
    set_target_properties(nvonnxparser_runtime PROPERTIES LINK_FLAGS "/ignore:4199")
    set_target_properties(trt_onnxify PROPERTIES LINK_FLAGS "/ignore:4199")
    target_compile_definitions(trt_onnxify PRIVATE ONNXIFI_BUILD_LIBRARY=1)
    target_sources(onnx2trt PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/getopt.cc)
    target_sources(getSupportedAPITest PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/getopt.cc)
    target_include_directories(onnx2trt PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/include)
    target_include_directories(getSupportedAPITest PRIVATE ${ONNXRUNTIME_ROOT}/test/win_getopt/mb/include)
    target_compile_options(trt_onnxify PRIVATE /FIio.h)
    target_compile_options(onnx2trt PRIVATE /FIio.h)
    target_compile_options(getSupportedAPITest PRIVATE /FIio.h)
  endif()
  include_directories(${ONNXRUNTIME_ROOT}/../cmake/external/onnx-tensorrt)
  include_directories(${TENSORRT_ROOT}/include)
  include_directories(${TENSORRT_ROOT}/lib)
  if (WIN32)
    set(trt_link_libs cudnn ${CMAKE_DL_LIBS} ${TENSORRT_ROOT}/lib/nvinfer.lib ${TENSORRT_ROOT}/lib/nvinfer_plugin.lib)
  else()
    set(trt_link_libs cudnn ${CMAKE_DL_LIBS} ${TENSORRT_ROOT}/lib/libnvinfer.so ${TENSORRT_ROOT}/lib/libnvinfer_plugin.so)
  endif()
  set(onnxparser_link_libs nvonnxparser_static nvonnxparser_plugin)

  file(GLOB_RECURSE onnxruntime_providers_trt_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/trt/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/trt/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_trt_cc_srcs})
  add_library(onnxruntime_providers_trt ${onnxruntime_providers_trt_cc_srcs})
  target_link_libraries(onnxruntime_providers_trt ${onnxparser_link_libs} ${trt_link_libs})
  onnxruntime_add_include_to_target(onnxruntime_providers_trt onnxruntime_common onnxruntime_framework gsl onnx onnx_proto protobuf::libprotobuf)
  add_dependencies(onnxruntime_providers_trt eigen ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_trt PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_CUDNN_HOME}/include ${eigen_INCLUDE_DIRS} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/trt  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  set_target_properties(onnxruntime_providers_trt PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_trt PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_trt PRIVATE ONNXIFI_BUILD_LIBRARY=1)
  target_compile_options(onnxruntime_providers_trt PRIVATE ${DISABLED_WARNINGS_FOR_TRT})
  if (WIN32)
    target_compile_options(onnxruntime_providers_trt INTERFACE /wd4996)
  endif()
  list(APPEND onnxruntime_libs onnxruntime_providers_trt)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES trt)
endif()

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  include(onnxruntime_providers_internal.cmake)
endif()

if(onnxruntime_USE_EIGEN_THREADPOOL)
    target_compile_definitions(onnxruntime_providers PUBLIC USE_EIGEN_THREADPOOL)
endif()
