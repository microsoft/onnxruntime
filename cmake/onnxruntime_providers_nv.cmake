# Copyright (c) Microsoft Corporation. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the MIT License.
  find_package(CUDAToolkit REQUIRED 12.8)
  enable_language(CUDA)
  if(onnxruntime_DISABLE_CONTRIB_OPS)
    message( FATAL_ERROR "To compile TensorRT execution provider contrib ops have to be enabled to dump an engine using com.microsoft:EPContext node." )
  endif()
  add_definitions(-DUSE_NV=1)
  if (onnxruntime_NV_PLACEHOLDER_BUILDER)
    add_definitions(-DORT_NV_PLACEHOLDER_BUILDER)
  endif()
if (NOT onnxruntime_USE_TENSORRT_BUILTIN_PARSER)
  message(FATAL_ERROR "TensorRT RTX can not be used with the open source parser.")
endif ()
  set(BUILD_LIBRARY_ONLY 1)
  add_definitions("-DONNX_ML=1")
  add_definitions("-DONNX_NAMESPACE=onnx")
  set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
  set(TENSORRT_RTX_ROOT ${onnxruntime_TENSORRT_RTX_HOME})
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(PROTOBUF_LIBRARY ${PROTOBUF_LIB})
  if (WIN32)
    set(OLD_CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4099 /wd4551 /wd4505 /wd4515 /wd4706 /wd4456 /wd4324 /wd4701 /wd4804 /wd4702 /wd4458 /wd4703")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4805")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -include algorithm")
    set(DISABLED_WARNINGS_FOR_TRT /wd4456)
  endif()
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
  endif()
  set(CXX_VERSION_DEFINED TRUE)

  find_path(TENSORRT_RTX_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_RTX_ROOT}
    PATH_SUFFIXES include)


  file(READ ${TENSORRT_RTX_INCLUDE_DIR}/NvInferVersion.h NVINFER_VER_CONTENT)
  string(REGEX MATCH "define TRT_MAJOR_RTX * +([0-9]+)" NV_TRT_MAJOR_RTX "${NVINFER_VER_CONTENT}")
  string(REGEX REPLACE "define TRT_MAJOR_RTX * +([0-9]+)" "\\1" NV_TRT_MAJOR_RTX "${NV_TRT_MAJOR_RTX}")
  string(REGEX MATCH "define TRT_MINOR_RTX * +([0-9]+)" NV_TRT_MINOR_RTX "${NVINFER_VER_CONTENT}")
  string(REGEX REPLACE "define TRT_MINOR_RTX * +([0-9]+)" "\\1" NV_TRT_MINOR_RTX "${NV_TRT_MINOR_RTX}")
  math(EXPR NV_TRT_MAJOR_RTX_INT "${NV_TRT_MAJOR_RTX}")
  math(EXPR NV_TRT_MINOR_RTX_INT "${NV_TRT_MINOR_RTX}")

  if (NV_TRT_MAJOR_RTX)
    MESSAGE(STATUS "NV_TRT_MAJOR_RTX is ${NV_TRT_MAJOR_RTX}")
  else()
    MESSAGE(STATUS "Can't find NV_TRT_MAJOR_RTX macro")
  endif()

  if (WIN32)
    set(TRT_RTX_LIB "tensorrt_rtx_${NV_TRT_MAJOR_RTX}_${NV_TRT_MINOR_RTX}")
    set(RTX_PARSER_LIB "tensorrt_onnxparser_rtx_${NV_TRT_MAJOR_RTX}_${NV_TRT_MINOR_RTX}")
  endif()

  if (NOT TRT_RTX_LIB)
     set(TRT_RTX_LIB "tensorrt_rtx")
  endif()

  if (NOT RTX_PARSER_LIB)
     set(RTX_PARSER_LIB "tensorrt_onnxparser_rtx")
  endif()

  MESSAGE(STATUS "Looking for ${TRT_RTX_LIB}")

  find_library(TENSORRT_LIBRARY_INFER ${TRT_RTX_LIB}
    HINTS ${TENSORRT_RTX_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64)

  if (NOT TENSORRT_LIBRARY_INFER)
    MESSAGE(STATUS "Can't find ${TRT_RTX_LIB}")
  endif()

  if (onnxruntime_USE_TENSORRT_BUILTIN_PARSER)
    MESSAGE(STATUS "Looking for ${RTX_PARSER_LIB}")

    find_library(TENSORRT_LIBRARY_NVONNXPARSER ${RTX_PARSER_LIB}
      HINTS  ${TENSORRT_RTX_ROOT}
      PATH_SUFFIXES lib lib64 lib/x64)

    if (NOT TENSORRT_LIBRARY_NVONNXPARSER)
      MESSAGE(STATUS "Can't find ${RTX_PARSER_LIB}")
    endif()

    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_NVONNXPARSER})
    MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
  else()
    set(ONNX_USE_LITE_PROTO ON)

    onnxruntime_fetchcontent_declare(
      onnx_tensorrt
      URL ${DEP_URL_onnx_tensorrt}
      URL_HASH SHA1=${DEP_SHA1_onnx_tensorrt}
      EXCLUDE_FROM_ALL
    )
    if (NOT CUDA_INCLUDE_DIR)
      set(CUDA_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS}) # onnx-tensorrt repo needs this variable to build
    endif()
    # The onnx_tensorrt repo contains a test program, getSupportedAPITest, which doesn't support Windows. It uses
    # unistd.h. So we must exclude it from our build. onnxruntime_fetchcontent_makeavailable is for the purpose.
    onnxruntime_fetchcontent_makeavailable(onnx_tensorrt)
    set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
    if ( CMAKE_COMPILER_IS_GNUCC )
      set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    endif()
    if (WIN32)
      set(CMAKE_CUDA_FLAGS ${OLD_CMAKE_CUDA_FLAGS})
      unset(PROTOBUF_LIBRARY)
      unset(OLD_CMAKE_CXX_FLAGS)
      unset(OLD_CMAKE_CUDA_FLAGS)
      set_target_properties(${RTX_PARSER_LIB} PROPERTIES LINK_FLAGS "/ignore:4199")
      target_compile_options(nvonnxparser_static PRIVATE /FIio.h /wd4100)
      target_compile_options(${RTX_PARSER_LIB} PRIVATE /FIio.h /wd4100)
    endif()
    # Static libraries are just nvonnxparser_static on all platforms
    set(onnxparser_link_libs nvonnxparser_static)
    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER})
    MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
  endif()

  # ${TENSORRT_LIBRARY} is empty if we link nvonnxparser_static.
  # nvonnxparser_static is linked against tensorrt libraries in onnx-tensorrt
  # See https://github.com/onnx/onnx-tensorrt/blob/8af13d1b106f58df1e98945a5e7c851ddb5f0791/CMakeLists.txt#L121
  # However, starting from TRT 10 GA, nvonnxparser_static doesn't link against tensorrt libraries.
  # Therefore, the above code finds ${TENSORRT_LIBRARY_INFER}
  set(trt_link_libs ${CMAKE_DL_LIBS} ${TENSORRT_LIBRARY})
  file(GLOB_RECURSE onnxruntime_providers_nv_tensorrt_rtx_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/nv_tensorrt_rtx/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/nv_tensorrt_rtx/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_stream_handle.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_stream_handle.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_graph.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_graph.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nv_tensorrt_rtx_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_nv_tensorrt_rtx ${onnxruntime_providers_nv_tensorrt_rtx_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_nv_tensorrt_rtx onnxruntime_common)
  target_link_libraries(onnxruntime_providers_nv_tensorrt_rtx PRIVATE Eigen3::Eigen  onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface Eigen3::Eigen)
  add_dependencies(onnxruntime_providers_nv_tensorrt_rtx onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  if (onnxruntime_USE_TENSORRT_BUILTIN_PARSER)
    target_link_libraries(onnxruntime_providers_nv_tensorrt_rtx PRIVATE ${trt_link_libs} ${ONNXRUNTIME_PROVIDERS_SHARED} ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface ${ABSEIL_LIBS} PUBLIC CUDA::cudart)
  else()
    target_link_libraries(onnxruntime_providers_nv_tensorrt_rtx PRIVATE ${onnxparser_link_libs} ${trt_link_libs} ${ONNXRUNTIME_PROVIDERS_SHARED} ${PROTOBUF_LIB} flatbuffers::flatbuffers ${ABSEIL_LIBS} PUBLIC CUDA::cudart)
  endif()
  target_include_directories(onnxruntime_providers_nv_tensorrt_rtx PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${TENSORRT_RTX_INCLUDE_DIR} ${onnx_tensorrt_SOURCE_DIR}
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

  # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
  set_target_properties(onnxruntime_providers_nv_tensorrt_rtx PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(onnxruntime_providers_nv_tensorrt_rtx PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_nv_tensorrt_rtx PRIVATE ONNXIFI_BUILD_LIBRARY=1)
  target_compile_options(onnxruntime_providers_nv_tensorrt_rtx PRIVATE ${DISABLED_WARNINGS_FOR_TRT})
  if (WIN32)
    target_compile_options(onnxruntime_providers_nv_tensorrt_rtx INTERFACE /wd4456)
  endif()
  # set CUDA_MINIMAL as default for NV provider since we do not have fallback to CUDA
  target_compile_definitions(onnxruntime_providers_nv_tensorrt_rtx PRIVATE USE_CUDA_MINIMAL=1)

  # Needed for the provider interface, as it includes training headers when training is enabled
  if (onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_providers_nv_tensorrt_rtx PRIVATE ${ORTTRAINING_ROOT})
    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      onnxruntime_add_include_to_target(onnxruntime_providers_nv_tensorrt_rtx Python::Module)
    endif()
  endif()

  if(APPLE)
    set_property(TARGET onnxruntime_providers_nv_tensorrt_rtx APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/nv_tensorrt_rtx/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_nv_tensorrt_rtx APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
    set_property(TARGET onnxruntime_providers_nv_tensorrt_rtx APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/nv_tensorrt_rtx/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_nv_tensorrt_rtx APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/nv_tensorrt_rtx/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_nv_tensorrt_rtx unknown platform, need to specify shared library exports for it")
  endif()

  install(TARGETS onnxruntime_providers_nv_tensorrt_rtx
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
