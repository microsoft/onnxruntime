# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_coreautoml_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/automl/cpu/*.h"
    "${ONNXRUNTIME_ROOT}/core/automl/cpu/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_automl_srcs})

add_library(onnxruntime_core_automl ${onnxruntime_automl_srcs})

target_include_directories(onnxruntime_automl PRIVATE ${ONNXRUNTIME_ROOT} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

onnxruntime_add_include_to_target(onnxruntime_automl onnxruntime_common gsl onnx onnx_proto protobuf::libprotobuf onnxruntime_automl_featurizers)

set_target_properties(onnxruntime_automl PROPERTIES FOLDER "ONNXRuntime")
# need onnx to build to create headers that this project includes
add_dependencies(onnxruntime_automl ${onnxruntime_EXTERNAL_DEPENDENCIES} onnxruntime_automl_featurizers)

if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  target_compile_definitions(onnxruntime_automl PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
endif()


install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/automl  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
if (WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(onnxruntime_automl PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/ConfigureVisualStudioCodeAnalysis.props)
endif()
