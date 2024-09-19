# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_definitions(-DUSE_TVM=1)
  if (onnxruntime_TVM_USE_HASH)
    add_definitions(-DUSE_TVM_HASH=1)
  endif()

  if (onnxruntime_TVM_USE_HASH)
    file (GLOB_RECURSE onnxruntime_providers_tvm_cc_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/providers/tvm/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/tvm/*.cc"
    )
  else()
    file (GLOB onnxruntime_providers_tvm_cc_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/providers/tvm/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/tvm/*.cc"
    )
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_tvm_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_tvm ${onnxruntime_providers_tvm_cc_srcs})

  if ( CMAKE_COMPILER_IS_GNUCC )
    target_compile_options(onnxruntime_providers_tvm PRIVATE -Wno-unused-parameter -Wno-missing-field-initializers)
  endif()

  target_include_directories(onnxruntime_providers_tvm PRIVATE
          ${TVM_INCLUDES}
          ${PYTHON_INCLUDE_DIRS})
  onnxruntime_add_include_to_target(onnxruntime_providers_tvm onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface)

  add_dependencies(onnxruntime_providers_tvm ${onnxruntime_EXTERNAL_DEPENDENCIES})

  if (onnxruntime_TVM_USE_HASH)
    add_dependencies(onnxruntime_providers_tvm ippcp_s)
    target_include_directories(onnxruntime_providers_tvm PRIVATE ${IPP_CRYPTO_INCLUDE_DIR})
    target_link_libraries(onnxruntime_providers_tvm PRIVATE ippcp_s)
  endif()

  set_target_properties(onnxruntime_providers_tvm PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_tvm PROPERTIES LINKER_LANGUAGE CXX)

  if (WIN32 AND MSVC)
    # wd4100: identifier' : unreferenced formal parameter
    # wd4127: conditional expression is constant
    # wd4244: conversion from 'int' to 'char', possible loss of data
    # TODO: 4244 should not be disabled
    target_compile_options(onnxruntime_providers_tvm PRIVATE "/wd4100" "/wd4127" "/wd4244")
  else()
    target_compile_options(onnxruntime_providers_tvm PRIVATE "-Wno-error=type-limits")
  endif()
  target_compile_definitions(onnxruntime_providers_tvm PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

  install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/tvm/tvm_provider_factory.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_tvm
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()