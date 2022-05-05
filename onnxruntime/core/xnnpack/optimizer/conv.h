// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <core/graph/graph.h>

namespace onnxruntime {
Status IsConvSupportedByXNNPack(const Node& nodeRef, const std::unordered_set<const NodeArg*>& graph_const_values,
                                bool input_is_nchw);
Status ReplaceConv(
    const Node& nodeRef,
    const std::unordered_set<const NodeArg*>& graph_const_values, std::unique_ptr<::ONNX_NAMESPACE::GraphProto>& output_graph);

}  // namespace onnxruntime
