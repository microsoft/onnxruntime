# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_definitions(-DUSE_ACL=1)
  file(GLOB_RECURSE onnxruntime_providers_acl_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/acl/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/acl/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_acl_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_acl ${onnxruntime_providers_acl_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_acl
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface
  )

  target_link_libraries(onnxruntime_providers_acl -L$ENV{LD_LIBRARY_PATH})
  add_dependencies(onnxruntime_providers_acl ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_acl PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_acl
    PRIVATE
    ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${onnxruntime_ACL_HOME} ${onnxruntime_ACL_HOME}/include
  )
  install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/acl/acl_provider_factory.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/
  )
  set_target_properties(onnxruntime_providers_acl PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_acl
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()