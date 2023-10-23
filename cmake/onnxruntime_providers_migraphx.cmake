# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_definitions(-DUSE_MIGRAPHX=1)
  set(BUILD_LIBRARY_ONLY 1)
  add_definitions("-DONNX_ML=1")
  add_definitions("-DONNX_NAMESPACE=onnx")
  include_directories(${protobuf_SOURCE_DIR} ${eigen_SOURCE_DIR})
  set(MIGRAPHX_ROOT ${onnxruntime_MIGRAPHX_HOME})
  include_directories(${onnx_SOURCE_DIR})
  set(OLD_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-missing-field-initializers")
  endif()
  set(CXX_VERSION_DEFINED TRUE)
  set(CMAKE_CXX_FLAGS ${OLD_CMAKE_CXX_FLAGS})
  if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
  endif()

  # Add search paths for default rocm installation
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm)

  find_package(hip)
  find_package(migraphx PATHS ${AMD_MIGRAPHX_HOME})

  find_package(miopen)
  find_package(rocblas)

  set(migraphx_libs migraphx::c hip::host MIOpen roc::rocblas)

  file(GLOB_RECURSE onnxruntime_providers_migraphx_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/migraphx/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/rocm/rocm_stream_handle.h"
    "${ONNXRUNTIME_ROOT}/core/providers/rocm/rocm_stream_handle.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_migraphx_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_migraphx ${onnxruntime_providers_migraphx_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_migraphx onnxruntime_common onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  add_dependencies(onnxruntime_providers_migraphx onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_link_libraries(onnxruntime_providers_migraphx PRIVATE ${migraphx_libs} ${ONNXRUNTIME_PROVIDERS_SHARED} onnx flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  target_include_directories(onnxruntime_providers_migraphx PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_migraphx PROPERTIES FOLDER "ONNXRuntime")
  target_compile_definitions(onnxruntime_providers_migraphx PRIVATE ONNXIFI_BUILD_LIBRARY=1)
  target_compile_options(onnxruntime_providers_migraphx PRIVATE -Wno-error=sign-compare)
  set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-deprecated-declarations")
  set_property(TARGET onnxruntime_providers_migraphx APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/migraphx/version_script.lds -Xlinker --gc-sections")
  target_link_libraries(onnxruntime_providers_migraphx PRIVATE nsync::nsync_cpp stdc++fs)

  include(CheckLibraryExists)
  check_library_exists(migraphx::c "migraphx_program_run_async" "/opt/rocm/migraphx/lib" HAS_STREAM_SYNC)
  if(HAS_STREAM_SYNC)
      target_compile_definitions(onnxruntime_providers_migraphx PRIVATE -DMIGRAPHX_STREAM_SYNC)
      message(STATUS "MIGRAPHX GPU STREAM SYNC is ENABLED")
  else()
      message(STATUS "MIGRAPHX GPU STREAM SYNC is DISABLED")
  endif()

  if (onnxruntime_ENABLE_TRAINING_OPS)
    onnxruntime_add_include_to_target(onnxruntime_providers_migraphx onnxruntime_training)
    target_link_libraries(onnxruntime_providers_migraphx PRIVATE onnxruntime_training)
    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      onnxruntime_add_include_to_target(onnxruntime_providers_migraphx Python::Module)
    endif()
  endif()

  install(TARGETS onnxruntime_providers_migraphx
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
