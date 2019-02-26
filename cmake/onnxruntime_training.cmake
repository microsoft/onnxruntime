# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_training_srcs
    "${ONNXRUNTIME_ROOT}/core/training/*.h"
    "${ONNXRUNTIME_ROOT}/core/training/*.cc"
)

add_library(onnxruntime_training ${onnxruntime_training_srcs})
add_dependencies(onnxruntime_training ${onnxruntime_EXTERNAL_DEPENDENCIES} onnx)
onnxruntime_add_include_to_target(onnxruntime_training  gsl onnx onnx_proto protobuf::libprotobuf)
target_include_directories(onnxruntime_training PRIVATE ${ONNXRUNTIME_ROOT})

if(WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(onnxruntime_training PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
endif()