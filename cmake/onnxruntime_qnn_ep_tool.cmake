# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(QNN_EP_TOOL_SRC_DIR ${REPO_ROOT}/samples/qnn_ep_tool)
onnxruntime_add_executable(
    qnn_ep_tool
    ${QNN_EP_TOOL_SRC_DIR}/main.cpp
    ${QNN_EP_TOOL_SRC_DIR}/utils.cpp
    ${QNN_EP_TOOL_SRC_DIR}/model_info.cpp
)
include_directories(${QNN_EP_TOOL_SRC_DIR})
target_link_libraries(qnn_ep_tool onnxruntime onnx)
