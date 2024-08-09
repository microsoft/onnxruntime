# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
  if(onnxruntime_DISABLE_CONTRIB_OPS)
    message( FATAL_ERROR "To compile TensorRT execution provider contrib ops have to be enabled to dump an engine using com.microsoft:EPContext node." )
  endif()
  add_definitions(-DUSE_TENSORRT=1)
  if (onnxruntime_TENSORRT_PLACEHOLDER_BUILDER)
    add_definitions(-DORT_TENSORRT_PLACEHOLDER_BUILDER)
  endif()
  set(BUILD_LIBRARY_ONLY 1)
  add_definitions("-DONNX_ML=1")
  add_definitions("-DONNX_NAMESPACE=onnx")
  set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
  set(TENSORRT_ROOT ${onnxruntime_TENSORRT_HOME})
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

  # There is an issue when running "Debug build" TRT EP with "Release build" TRT builtin parser on Windows.
  # We enforce following workaround for now until the real fix.
  if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(onnxruntime_USE_TENSORRT_BUILTIN_PARSER OFF)
    MESSAGE(STATUS "[Note] There is an issue when running \"Debug build\" TRT EP with \"Release build\" TRT built-in parser on Windows. This build will use tensorrt oss parser instead.")
  endif()

  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES include)


  file(READ ${TENSORRT_INCLUDE_DIR}/NvInferVersion.h NVINFER_VER_CONTENT)
  string(REGEX MATCH "define NV_TENSORRT_MAJOR * +([0-9]+)" NV_TENSORRT_MAJOR "${NVINFER_VER_CONTENT}")
  string(REGEX REPLACE "define NV_TENSORRT_MAJOR * +([0-9]+)" "\\1" NV_TENSORRT_MAJOR "${NV_TENSORRT_MAJOR}")
  string(REGEX MATCH "define NV_TENSORRT_MINOR * +([0-9]+)" NV_TENSORRT_MINOR "${NVINFER_VER_CONTENT}")
  string(REGEX REPLACE "define NV_TENSORRT_MINOR * +([0-9]+)" "\\1" NV_TENSORRT_MINOR "${NV_TENSORRT_MINOR}")
  string(REGEX MATCH "define NV_TENSORRT_PATCH * +([0-9]+)" NV_TENSORRT_PATCH "${NVINFER_VER_CONTENT}")
  string(REGEX REPLACE "define NV_TENSORRT_PATCH * +([0-9]+)" "\\1" NV_TENSORRT_PATCH "${NV_TENSORRT_PATCH}")
  math(EXPR NV_TENSORRT_MAJOR_INT "${NV_TENSORRT_MAJOR}")
  math(EXPR NV_TENSORRT_MINOR_INT "${NV_TENSORRT_MINOR}")
  math(EXPR NV_TENSORRT_PATCH_INT "${NV_TENSORRT_PATCH}")

  if (NV_TENSORRT_MAJOR)
    MESSAGE(STATUS "NV_TENSORRT_MAJOR is ${NV_TENSORRT_MAJOR}")
  else()
    MESSAGE(STATUS "Can't find NV_TENSORRT_MAJOR macro")
  endif()

  # Check TRT version >= 10.0.1.6
  if ((NV_TENSORRT_MAJOR_INT GREATER 10) OR
      (NV_TENSORRT_MAJOR_INT EQUAL 10 AND NV_TENSORRT_MINOR_INT GREATER 0) OR
      (NV_TENSORRT_MAJOR_INT EQUAL 10 AND NV_TENSORRT_PATCH_INT GREATER 0))
    set(TRT_GREATER_OR_EQUAL_TRT_10_GA ON)
  endif()

  # TensorRT 10 GA onwards, the TensorRT libraries will have major version appended to the end on Windows,
  # for example, nvinfer_10.dll, nvinfer_plugin_10.dll, nvonnxparser_10.dll ...
  if (WIN32 AND TRT_GREATER_OR_EQUAL_TRT_10_GA)
    set(NVINFER_LIB "nvinfer_${NV_TENSORRT_MAJOR}")
    set(NVINFER_PLUGIN_LIB "nvinfer_plugin_${NV_TENSORRT_MAJOR}")
    set(PARSER_LIB "nvonnxparser_${NV_TENSORRT_MAJOR}")
  endif()

  if (NOT NVINFER_LIB)
     set(NVINFER_LIB "nvinfer")
  endif()

  if (NOT NVINFER_PLUGIN_LIB)
     set(NVINFER_PLUGIN_LIB "nvinfer_plugin")
  endif()

  if (NOT PARSER_LIB)
     set(PARSER_LIB "nvonnxparser")
  endif()

  MESSAGE(STATUS "Looking for ${NVINFER_LIB} and ${NVINFER_PLUGIN_LIB}")

  find_library(TENSORRT_LIBRARY_INFER ${NVINFER_LIB}
    HINTS ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64)

  if (NOT TENSORRT_LIBRARY_INFER)
    MESSAGE(STATUS "Can't find ${NVINFER_LIB}")
  endif()

  find_library(TENSORRT_LIBRARY_INFER_PLUGIN ${NVINFER_PLUGIN_LIB}
    HINTS  ${TENSORRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x64)

  if (NOT TENSORRT_LIBRARY_INFER_PLUGIN)
    MESSAGE(STATUS "Can't find ${NVINFER_PLUGIN_LIB}")
  endif()

  if (onnxruntime_USE_TENSORRT_BUILTIN_PARSER)
    MESSAGE(STATUS "Looking for ${PARSER_LIB}")

    find_library(TENSORRT_LIBRARY_NVONNXPARSER ${PARSER_LIB}
      HINTS  ${TENSORRT_ROOT}
      PATH_SUFFIXES lib lib64 lib/x64)

    if (NOT TENSORRT_LIBRARY_NVONNXPARSER)
      MESSAGE(STATUS "Can't find ${PARSER_LIB}")
    endif()

    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_NVONNXPARSER})
    MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
  else()
    if (TRT_GREATER_OR_EQUAL_TRT_10_GA)
      set(ONNX_USE_LITE_PROTO ON)
    endif()
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
    include_directories(${onnx_tensorrt_SOURCE_DIR})
    set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
    if ( CMAKE_COMPILER_IS_GNUCC )
      set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    endif()
    if (WIN32)
      set(CMAKE_CUDA_FLAGS ${OLD_CMAKE_CUDA_FLAGS})
      unset(PROTOBUF_LIBRARY)
      unset(OLD_CMAKE_CXX_FLAGS)
      unset(OLD_CMAKE_CUDA_FLAGS)
      set_target_properties(${PARSER_LIB} PROPERTIES LINK_FLAGS "/ignore:4199")
      target_compile_options(nvonnxparser_static PRIVATE /FIio.h /wd4100)
      target_compile_options(${PARSER_LIB} PRIVATE /FIio.h /wd4100)
    endif()
    # Static libraries are just nvonnxparser_static on all platforms
    set(onnxparser_link_libs nvonnxparser_static)
    set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN})
    MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
  endif()

  include_directories(${TENSORRT_INCLUDE_DIR})
  # ${TENSORRT_LIBRARY} is empty if we link nvonnxparser_static.
  # nvonnxparser_static is linked against tensorrt libraries in onnx-tensorrt
  # See https://github.com/onnx/onnx-tensorrt/blob/8af13d1b106f58df1e98945a5e7c851ddb5f0791/CMakeLists.txt#L121
  # However, starting from TRT 10 GA, nvonnxparser_static doesn't link against tensorrt libraries.
  # Therefore, the above code finds ${TENSORRT_LIBRARY_INFER} and ${TENSORRT_LIBRARY_INFER_PLUGIN}.
  if(onnxruntime_CUDA_MINIMAL)
    set(trt_link_libs ${CMAKE_DL_LIBS} ${TENSORRT_LIBRARY})
  else()
    set(trt_link_libs CUDNN::cudnn_all cublas ${CMAKE_DL_LIBS} ${TENSORRT_LIBRARY})
  endif()
  file(GLOB_RECURSE onnxruntime_providers_tensorrt_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_stream_handle.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_stream_handle.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_graph.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_graph.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_tensorrt_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_tensorrt ${onnxruntime_providers_tensorrt_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_tensorrt onnxruntime_common onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  add_dependencies(onnxruntime_providers_tensorrt onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  if (onnxruntime_USE_TENSORRT_BUILTIN_PARSER)
    target_link_libraries(onnxruntime_providers_tensorrt PRIVATE ${trt_link_libs} ${ONNXRUNTIME_PROVIDERS_SHARED} ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface ${ABSEIL_LIBS} PUBLIC CUDA::cudart)
  else()
    target_link_libraries(onnxruntime_providers_tensorrt PRIVATE ${onnxparser_link_libs} ${trt_link_libs} ${ONNXRUNTIME_PROVIDERS_SHARED} ${PROTOBUF_LIB} flatbuffers::flatbuffers ${ABSEIL_LIBS} PUBLIC CUDA::cudart)
  endif()
  target_include_directories(onnxruntime_providers_tensorrt PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS}
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

  # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
  set_target_properties(onnxruntime_providers_tensorrt PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(onnxruntime_providers_tensorrt PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_tensorrt PRIVATE ONNXIFI_BUILD_LIBRARY=1)
  target_compile_options(onnxruntime_providers_tensorrt PRIVATE ${DISABLED_WARNINGS_FOR_TRT})
  if (WIN32)
    target_compile_options(onnxruntime_providers_tensorrt INTERFACE /wd4456)
  endif()
  if(onnxruntime_CUDA_MINIMAL)
    target_compile_definitions(onnxruntime_providers_tensorrt PRIVATE USE_CUDA_MINIMAL=1)
  endif()

  # Needed for the provider interface, as it includes training headers when training is enabled
  if (onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_providers_tensorrt PRIVATE ${ORTTRAINING_ROOT})
    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      onnxruntime_add_include_to_target(onnxruntime_providers_tensorrt Python::Module)
    endif()
  endif()

  if(APPLE)
    set_property(TARGET onnxruntime_providers_tensorrt APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/tensorrt/exported_symbols.lst")
    target_link_libraries(onnxruntime_providers_tensorrt PRIVATE nsync::nsync_cpp)
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_tensorrt APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
    set_property(TARGET onnxruntime_providers_tensorrt APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/tensorrt/version_script.lds -Xlinker --gc-sections")
    target_link_libraries(onnxruntime_providers_tensorrt PRIVATE nsync::nsync_cpp)
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_tensorrt APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/tensorrt/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_tensorrt unknown platform, need to specify shared library exports for it")
  endif()

  install(TARGETS onnxruntime_providers_tensorrt
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
