# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_definitions(-DUSE_MIGRAPHX=1)
  include_directories(${protobuf_SOURCE_DIR} ${eigen_SOURCE_DIR} ${onnx_SOURCE_DIR})
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
  endif()

  # Add search paths for default rocm installation
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm $ENV{HIP_PATH})

  if(POLICY CMP0144)
      # Suppress the warning about the small capitals of the package name
      cmake_policy(SET CMP0144 NEW)
  endif()

  if(WIN32 AND NOT HIP_PLATFORM)
    set(HIP_PLATFORM "amd")
  endif()

  find_package(hip REQUIRED)
  find_package(migraphx REQUIRED PATHS ${AMD_MIGRAPHX_HOME})

  file(GLOB_RECURSE onnxruntime_providers_migraphx_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_migraphx_cc_srcs})
  onnxruntime_add_shared_library(onnxruntime_providers_migraphx ${onnxruntime_providers_migraphx_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_migraphx onnxruntime_common onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  add_dependencies(onnxruntime_providers_migraphx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_link_libraries(onnxruntime_providers_migraphx PRIVATE migraphx::c hip::host ${ONNXRUNTIME_PROVIDERS_SHARED} onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  target_include_directories(onnxruntime_providers_migraphx PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/migraphx/onnxruntime)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_migraphx PRIVATE ONNXIFI_BUILD_LIBRARY=1 ONNX_ML=1 ONNX_NAMESPACE=onnx)
  if(MSVC)
    set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY LINK_FLAGS /DEF:${ONNXRUNTIME_ROOT}/core/providers/migraphx/symbols.def)
    target_link_libraries(onnxruntime_providers_migraphx PRIVATE ws2_32)
  else()
    target_compile_options(onnxruntime_providers_migraphx PRIVATE -Wno-error=sign-compare)
    set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
  endif()
  if(UNIX)
    set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/migraphx/version_script.lds -Xlinker --gc-sections")
    target_link_libraries(onnxruntime_providers_migraphx PRIVATE  stdc++fs)
  endif()

  set(CMAKE_REQUIRED_LIBRARIES migraphx::c)

  check_symbol_exists(migraphx_onnx_options_set_external_data_path
    "migraphx/migraphx.h" HAVE_MIGRAPHX_API_ONNX_OPTIONS_SET_EXTERNAL_DATA_PATH)

  if(HAVE_MIGRAPHX_API_ONNX_OPTIONS_SET_EXTERNAL_DATA_PATH)
    target_compile_definitions(onnxruntime_providers_migraphx PRIVATE HAVE_MIGRAPHX_API_ONNX_OPTIONS_SET_EXTERNAL_DATA_PATH=1)
  endif()

  if (onnxruntime_ENABLE_TRAINING_OPS)
    onnxruntime_add_include_to_target(onnxruntime_providers_migraphx onnxruntime_training)
    target_link_libraries(onnxruntime_providers_migraphx PRIVATE onnxruntime_training)
    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      onnxruntime_add_include_to_target(onnxruntime_providers_migraphx Python::Module)
    endif()
  endif()

  install(TARGETS onnxruntime_providers_migraphx
          EXPORT onnxruntime_providers_migraphxTargets
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
