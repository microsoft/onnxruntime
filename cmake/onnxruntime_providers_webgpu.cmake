# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "WebGPU EP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_WEBGPU=1)
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    add_definitions(-DENABLE_WEBASSEMBLY_THREADS=1)
  endif()
  file(GLOB_RECURSE onnxruntime_providers_webgpu_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/webgpu/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/webgpu/*.cc"
  )
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_webgpu_contrib_ops_cc_srcs})
    list(APPEND onnxruntime_providers_webgpu_cc_srcs ${onnxruntime_webgpu_contrib_ops_cc_srcs})
  endif()

  source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_providers_webgpu_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_webgpu ${onnxruntime_providers_webgpu_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_webgpu
    onnxruntime_common dawn::dawncpp_headers dawn::dawn_headers onnx onnx_proto flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  if (NOT onnxruntime_USE_EXTERNAL_DAWN)
    target_link_libraries(onnxruntime_providers_webgpu dawn::dawn_native)
  endif()
  target_link_libraries(onnxruntime_providers_webgpu dawn::dawn_proc)

  set_target_properties(onnxruntime_providers_webgpu PROPERTIES FOLDER "ONNXRuntime")
