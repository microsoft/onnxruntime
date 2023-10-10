# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_definitions(-DUSE_CANN=1)

  file(GLOB_RECURSE onnxruntime_providers_cann_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cann/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cann/*.cc"
  )

  # The shared_library files are in a separate list since they use precompiled headers, and the above files have them disabled.
  file(GLOB_RECURSE onnxruntime_providers_cann_shared_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_cann_cc_srcs} ${onnxruntime_providers_cann_shared_srcs})
  set(onnxruntime_providers_cann_src ${onnxruntime_providers_cann_cc_srcs} ${onnxruntime_providers_cann_shared_srcs})

  onnxruntime_add_shared_library_module(onnxruntime_providers_cann ${onnxruntime_providers_cann_src})
  onnxruntime_add_include_to_target(onnxruntime_providers_cann onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface)

  add_dependencies(onnxruntime_providers_cann onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_link_libraries(onnxruntime_providers_cann PRIVATE ascendcl acl_op_compiler fmk_onnx_parser nsync::nsync_cpp ${ABSEIL_LIBS} ${ONNXRUNTIME_PROVIDERS_SHARED})
  target_link_directories(onnxruntime_providers_cann PRIVATE ${onnxruntime_CANN_HOME}/lib64)
  target_include_directories(onnxruntime_providers_cann PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${eigen_INCLUDE_DIRS} ${onnxruntime_CANN_HOME} ${onnxruntime_CANN_HOME}/include)

  set_target_properties(onnxruntime_providers_cann PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_cann PROPERTIES FOLDER "ONNXRuntime")

  install(TARGETS onnxruntime_providers_cann
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})