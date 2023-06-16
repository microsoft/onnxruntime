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

enum class WebnnDeviceType {
  CPU,
  GPU,
};

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
  switch (tensor.data_type()) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      const int64_t* array_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
      if constexpr (std::is_same<T, int64_t>::value) {
        array.assign(array_data, array_data + rank);
      } else {
        std::transform(array_data, array_data + rank,
                       std::back_inserter(array),
                       [](int64_t dim) -> T { return SafeInt<T>(dim); });
      };
      break;
    }

    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      const int32_t* array_data = reinterpret_cast<const int32_t*>(unpacked_tensor.data());
      array.assign(array_data, array_data + rank);
      break;
    }
    default:
      return false;
  }
  return true;
}

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by WebNN EP.
std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const emscripten::val& wnn_builder_,
                                                      const WebnnDeviceType device_type,
                                                      const logging::Logger& logger);
static const InlinedHashMap<std::string, std::string> op_map = {
    {"ArgMax", "argMax"},
    {"ArgMin", "argMin"},
    {"Add", "add"},
    {"Sub", "sub"},
    {"Mul", "mul"},
    {"Div", "div"},
    {"Pow", "pow"},
    {"Ceil", "ceil"},
    {"Cos", "cos"},
    {"Equal", "equal"},
    {"Erf", "erf"},
    {"Exp", "exp"},
    {"Not", "logicalNot"},
    {"Floor", "floor"},
    {"Flatten", "flattenTo2d"},
    {"Identity", "identity"},
    {"Reciprocal", "reciprocal"},
    {"Sin", "sin"},
    {"Sqrt", "sqrt"},
    {"Tan", "tan"},
    {"Relu", "relu"},
    {"LeakyRelu", "leakyRelu"},
    {"Sigmoid", "sigmoid"},
    {"Slice", "slice"},
    {"Softmax", "softmax"},
    {"Cast", "cast"},
    {"Clip", "clamp"},
    {"Conv", "conv2d"},
    {"ConvTranspose", "convTranspose2d"},
    {"Concat", "concat"},
    {"Expand", "expand"},
    {"Gather", "gather"},
    {"Gemm", "gemm"},
    {"MatMul", "matmul"},
    {"GlobalAveragePool", "averagePool2d"},
    {"GlobalMaxPool", "maxPool2d"},
    {"AveragePool", "averagePool2d"},
    {"LayerNormalization", "meanVarianceNormalization"},
    {"MaxPool", "maxPool2d"},
    {"ReduceMax", "reduceMax"},
    {"ReduceMean", "reduceMean"},
    {"Reshape", "reshape"},
    {"Resize", "resample2d"},
    {"Shape", "slice"},
    {"Split", "split"},
    {"Squeeze", "squeeze"},
    {"Transpose", "transpose"},
    {"Unsqueeze", "unsqueeze"},
};

inline bool CheckSingleOp(const std::string& op_type, const emscripten::val& wnn_builder_) {
  return op_map.find(op_type) != op_map.end() && wnn_builder_[op_map.find(op_type)->second].as<bool>();
}

constexpr std::array<ONNX_NAMESPACE::TensorProto_DataType, 1> supported_cpu_data_types = {
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
};

constexpr std::array<ONNX_NAMESPACE::TensorProto_DataType, 7> supported_gpu_data_types = {
    ONNX_NAMESPACE::TensorProto_DataType_BOOL,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
    ONNX_NAMESPACE::TensorProto_DataType_INT32,
    ONNX_NAMESPACE::TensorProto_DataType_INT64,
    ONNX_NAMESPACE::TensorProto_DataType_UINT32,
    ONNX_NAMESPACE::TensorProto_DataType_UINT64,
};

bool IsSupportedDataType(const int32_t data_type, const WebnnDeviceType device_type);

bool IsValidMultidirectionalBroadcast(std::vector<int64_t>& shape_a,
                                      std::vector<int64_t>& shape_b,
                                      const logging::Logger& logger);
}  // namespace webnn
}  // namespace onnxruntime
