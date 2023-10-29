# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_QNN=1)

  # These are shared utils,
  # TODO, move this to a separated lib when used by EPs other than QNN, NNAPI and CoreML
  file(GLOB_RECURSE onnxruntime_providers_shared_utils_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/node_unit/node_unit.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/node_unit/node_unit.cc"
  )

  file(GLOB_RECURSE
    onnxruntime_providers_qnn_ep_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.cc"
  )

  file(GLOB_RECURSE
    onnxruntime_providers_qnn_builder_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/*.cc"
  )

  set(onnxruntime_providers_qnn_cc_srcs
    ${onnxruntime_providers_shared_utils_cc_srcs}
    ${onnxruntime_providers_qnn_ep_cc_srcs}
    ${onnxruntime_providers_qnn_builder_cc_srcs}
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_qnn_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_qnn ${onnxruntime_providers_qnn_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_qnn onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers::flatbuffers Boost::mp11)
  target_link_libraries(onnxruntime_providers_qnn)
  add_dependencies(onnxruntime_providers_qnn onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_qnn PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_qnn PROPERTIES FOLDER "ONNXRuntime")
  target_include_directories(onnxruntime_providers_qnn PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_QNN_HOME}/include/QNN ${onnxruntime_QNN_HOME}/include)
  set_target_properties(onnxruntime_providers_qnn PROPERTIES LINKER_LANGUAGE CXX)
  # ignore the warning unknown-pragmas on "pragma region"
  if(NOT MSVC)
    target_compile_options(onnxruntime_providers_qnn PRIVATE "-Wno-unknown-pragmas")
  endif()