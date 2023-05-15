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

template <typename T>
std::string GetShapeString(std::vector<T>& shape) {
  std::stringstream shape_info;
  shape_info << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0) {
      shape_info << ", ";
    }
    shape_info << shape[i];
  }
  shape_info << "]";
  return shape_info.str();
}

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by WebNN EP.
std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const emscripten::val& wnn_builder_,
                                                      const logging::Logger& logger);
static const InlinedHashMap<std::string, std::string> op_map = {
    {"ArgMax", "argMax"},
    {"ArgMin", "argMin"},
    {"Add", "add"},
    {"Sub", "sub"},
    {"Mul", "mul"},
    {"Div", "div"},
    {"Pow", "pow"},
    {"Cos", "cos"},
    {"Erf", "erf"},
    {"Floor", "floor"},
    {"Sin", "sin"},
    {"Sqrt", "sqrt"},
    {"Relu", "relu"},
    {"LeakyRelu", "leakyRelu"},
    {"Sigmoid", "sigmoid"},
    {"Softmax", "softmax"},
    {"Cast", "cast"},
    {"Clip", "clamp"},
    {"Conv", "conv2d"},
    {"ConvTranspose", "convTranspose2d"},
    {"Concat", "concat"},
    {"Expand", "expand"},
    {"Gemm", "gemm"},
    {"MatMul", "matmul"},
    {"GlobalAveragePool", "averagePool2d"},
    {"GlobalMaxPool", "maxPool2d"},
    {"AveragePool", "averagePool2d"},
    {"MaxPool", "maxPool2d"},
    {"ReduceMax", "reduceMax"},
    {"ReduceMean", "reduceMean"},
    {"Reshape", "reshape"},
    {"Resize", "resample2d"},
    {"Transpose", "transpose"}};

inline bool CheckSingleOp(const std::string& op_type, const emscripten::val& wnn_builder_) {
  return op_map.find(op_type) != op_map.end() && wnn_builder_[op_map.find(op_type)->second].as<bool>();
}

constexpr std::array<ONNX_NAMESPACE::TensorProto_DataType, 3> supported_data_types = {
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
    ONNX_NAMESPACE::TensorProto_DataType_INT64,
};

bool IsSupportedDataType(int32_t data_type);

bool IsValidMultidirectionalBroadcast(std::vector<int64_t>& shape_a,
                                      std::vector<int64_t>& shape_b,
                                      const logging::Logger& logger);
}  // namespace webnn
}  // namespace onnxruntime
