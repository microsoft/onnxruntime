# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  file(GLOB_RECURSE onnxruntime_providers_azure_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/azure/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/azure/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_azure_src})
  onnxruntime_add_static_library(onnxruntime_providers_azure ${onnxruntime_providers_azure_src})
  add_dependencies(onnxruntime_providers_azure ${onnxruntime_EXTERNAL_DEPENDENCIES})
  onnxruntime_add_include_to_target(onnxruntime_providers_azure onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11)
  target_link_libraries(onnxruntime_providers_azure PRIVATE onnx onnxruntime_common onnxruntime_framework)
  set_target_properties(onnxruntime_providers_azure PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_azure PROPERTIES LINKER_LANGUAGE CXX)

  install(TARGETS onnxruntime_providers_azure
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})