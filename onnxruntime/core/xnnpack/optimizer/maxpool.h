// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnx/onnx_pb.h>
#include "core/common/common.h"

namespace onnxruntime {
class Graph;
class Node;

bool IsMaxPoolSupportedByXNNPack(const Node& nodeRef, bool input_is_nchw);
Status ReplaceMaxPool(const Node& nodeRef, std::unique_ptr<::ONNX_NAMESPACE::GraphProto>& output_graph);

}  // namespace onnxruntime