# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_session_srcs
    "${ONNXRUNTIME_INCLUDE_DIR}/core/session/*.h"
    "${ONNXRUNTIME_ROOT}/core/session/*.h"
    "${ONNXRUNTIME_ROOT}/core/session/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_session_srcs})

add_library(onnxruntime_session ${onnxruntime_session_srcs})
onnxruntime_add_include_to_target(onnxruntime_session onnx protobuf::libprotobuf)
target_include_directories(onnxruntime_session PRIVATE ${ONNXRUNTIME_ROOT})
add_dependencies(onnxruntime_session ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_session PROPERTIES FOLDER "ONNXRuntime")


