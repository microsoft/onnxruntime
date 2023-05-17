// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include "core/common/inlined_containers.h"
#include <core/graph/basic_types.h>
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"

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

template <typename T>
bool ReadIntArrayFrom1DTensor(const onnx::TensorProto& tensor, std::vector<T>& array, const logging::Logger& logger) {
  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Error while unpacking shape: " << status.ErrorMessage();
    return false;
  }
  const auto& dims = tensor.dims();
  if (dims.size() != 1) {
    LOGS(logger, VERBOSE) << "The tensor must be 1D.";
    return false;
  }
  int64_t rank = dims[0];
  if (tensor.data_type() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    LOGS(logger, VERBOSE) << "The type of tensor's element data must be INT64.";
    return false;
  }
  const int64_t* array_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  if constexpr (std::is_same<T, int64_t>::value) {
    array.assign(array_data, array_data + rank);
  } else if constexpr (std::is_same<T, int32_t>::value) {
    std::transform(array_data, array_data + rank,
                   std::back_inserter(array),
                   [](int64_t dim) -> int32_t { return SafeInt<int32_t>(dim); });
  }
  return true;
}

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by WebNN EP.
std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const emscripten::val& wnn_builder_,
                                                      const logging::Logger& logger);
static const InlinedHashMap<std::string, std::string> op_map = {
    {"ArgMax", "argMax1"},
    {"ArgMin", "argMin1"},
    {"Add", "add1"},
    {"Sub", "sub1"},
    {"Mul", "mul1"},
    {"Div", "div1"},
    {"Pow", "pow1"},
    {"Cos", "cos1"},
    {"Erf", "erf1"},
    {"Floor", "floor1"},
    {"Sin", "sin1"},
    {"Sqrt", "sqrt1"},
    {"Relu", "relu1"},
    {"LeakyRelu", "leakyRelu1"},
    {"Sigmoid", "sigmoid1"},
    {"Softmax", "softmax1"},
    {"Cast", "cast1"},
    {"Clip", "clamp1"},
    {"Conv", "conv2d1"},
    {"ConvTranspose", "convTranspose2d1"},
    {"Concat", "concat1"},
    {"Expand", "expand"},
    {"Gemm", "gemm1"},
    {"MatMul", "matmul1"},
    {"GlobalAveragePool", "averagePool2d1"},
    {"GlobalMaxPool", "maxPool2d1"},
    {"AveragePool", "averagePool2d1"},
    {"MaxPool", "maxPool2d1"},
    {"ReduceMax", "reduceMax1"},
    {"ReduceMean", "reduceMean1"},
    {"Reshape", "reshape1"},
    {"Resize", "resample2d1"},
    {"Split", "split1"},
    {"Transpose", "transpose1"}};

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
