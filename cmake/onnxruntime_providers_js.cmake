# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_JSEP=1)

  file(GLOB_RECURSE onnxruntime_providers_js_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/js/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/js/*.cc"
  )
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_js_contrib_ops_cc_srcs})
    list(APPEND onnxruntime_providers_js_cc_srcs ${onnxruntime_js_contrib_ops_cc_srcs})
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_providers_js_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_js ${onnxruntime_providers_js_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_js
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers Boost::mp11
  )
  target_include_directories(onnxruntime_providers_js PRIVATE  ${eigen_INCLUDE_DIRS})
  add_dependencies(onnxruntime_providers_js ${onnxruntime_EXTERNAL_DEPENDENCIES})