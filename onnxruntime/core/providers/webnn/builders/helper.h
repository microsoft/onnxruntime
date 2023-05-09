// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include "core/common/inlined_containers.h"
#include <core/graph/basic_types.h>
#include "core/providers/common.h"

#include <emscripten.h>
#include <emscripten/val.h>

namespace onnxruntime {

class GraphViewer;
class NodeArg;

namespace logging {
class Logger;
}

namespace webnn {

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger);

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by WebNN EP.
std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const emscripten::val& wnn_builder_,
                                                      const logging::Logger& logger);
static const InlinedHashMap<std::string, std::string> op_map = {
    {"Add", "add"},
    {"Sub", "sub"},
    {"Mul", "mul"},
    {"Div", "div"},
    {"Relu", "relu"},
    {"LeakyRelu", "leakyRelu"},
    {"Sigmoid", "sigmoid"},
    {"Clip", "clamp"},
    {"Conv", "conv2d"},
    {"ConvTranspose", "convTranspose2d"},
    {"Concat", "concat"},
    {"Gemm", "gemm"},
    {"GlobalAveragePool", "averagePool2d"},
    {"GlobalMaxPool", "maxPool2d"},
    {"AveragePool", "averagePool2d"},
    {"MaxPool", "maxPool2d"},
    {"Reshape", "reshape"},
    {"Resize", "resample2d"},
    {"Transpose", "transpose"}};

inline bool CheckSingleOp(const std::string& op_type, const emscripten::val& wnn_builder_) {
  return op_map.find(op_type) != op_map.end() && wnn_builder_[op_map.find(op_type)->second].as<bool>();
}

}  // namespace webnn
}  // namespace onnxruntime
