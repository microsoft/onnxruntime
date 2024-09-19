# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "WebNN EP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_WEBNN=1)
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    add_definitions(-DENABLE_WEBASSEMBLY_THREADS=1)
  endif()
  file(GLOB_RECURSE onnxruntime_providers_webnn_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/webnn/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/webnn/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
  )

  source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_providers_webnn_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_webnn ${onnxruntime_providers_webnn_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_webnn onnxruntime_common onnx onnx_proto flatbuffers::flatbuffers Boost::mp11 safeint_interface)

  add_dependencies(onnxruntime_providers_webnn onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_webnn PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_webnn PROPERTIES LINKER_LANGUAGE CXX)