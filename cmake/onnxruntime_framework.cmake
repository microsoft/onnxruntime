# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_framework_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.cc"
)

if (onnxruntime_MINIMAL_BUILD)
  file(GLOB onnxruntime_framework_src_exclude
    "${ONNXRUNTIME_ROOT}/core/framework/provider_bridge_ort.cc"
    "${ONNXRUNTIME_ROOT}/core/framework/graph_partitioner.*"
    "${ONNXRUNTIME_INCLUDE_DIR}/core/framework/customregistry.h"
    "${ONNXRUNTIME_ROOT}/core/framework/customregistry.cc"
  )

  list(REMOVE_ITEM onnxruntime_framework_srcs ${onnxruntime_framework_src_exclude})
endif()

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_framework_srcs})

add_library(onnxruntime_framework ${onnxruntime_framework_srcs})
if(onnxruntime_ENABLE_INSTRUMENT)
  target_compile_definitions(onnxruntime_framework PRIVATE ONNXRUNTIME_ENABLE_INSTRUMENT)
endif()
target_include_directories(onnxruntime_framework PRIVATE ${ONNXRUNTIME_ROOT} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
onnxruntime_add_include_to_target(onnxruntime_framework onnxruntime_common onnx onnx_proto protobuf::libprotobuf flatbuffers)
set_target_properties(onnxruntime_framework PROPERTIES FOLDER "ONNXRuntime")
# need onnx to build to create headers that this project includes
add_dependencies(onnxruntime_framework ${onnxruntime_EXTERNAL_DEPENDENCIES})

if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  target_compile_definitions(onnxruntime_framework PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
endif()


install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/framework  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(onnxruntime_framework PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()

