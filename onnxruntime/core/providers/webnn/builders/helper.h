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

typedef struct {
  std::string opName;
  bool isCpuSupported;  // The WebNN CPU backend XNNPack supports it (not about the CPU EP).
} WebnnOpInfo;

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

inline bool ReadScalarTensorData(const onnx::TensorProto& tensor, emscripten::val& scalar, const logging::Logger& logger) {
  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS(logger, ERROR) << "Error while unpacking tensor: " << status.ErrorMessage();
    return false;
  }
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      scalar = emscripten::val{*reinterpret_cast<uint8_t*>(unpacked_tensor.data())};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      scalar = emscripten::val{MLFloat16::FromBits(*reinterpret_cast<uint16_t*>(unpacked_tensor.data())).ToFloat()};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      scalar = emscripten::val{*reinterpret_cast<float*>(unpacked_tensor.data())};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      scalar = emscripten::val{*reinterpret_cast<int32_t*>(unpacked_tensor.data())};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      scalar = emscripten::val{*reinterpret_cast<int64_t*>(unpacked_tensor.data())};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      scalar = emscripten::val{*reinterpret_cast<uint32_t*>(unpacked_tensor.data())};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      scalar = emscripten::val{*reinterpret_cast<uint64_t*>(unpacked_tensor.data())};
      break;
    default:
      LOGS(logger, ERROR) << "Unsupported data type : " << tensor.data_type();
      return false;
      break;
  }
  return true;
}

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by WebNN EP.
std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const emscripten::val& wnn_builder_,
                                                      const WebnnDeviceType device_type,
                                                      const logging::Logger& logger);
static const InlinedHashMap<std::string, WebnnOpInfo> op_map = {
    {"Abs", {"abs", true}},
    {"Add", {"add", true}},
    {"ArgMax", {"argMax", false}},
    {"ArgMin", {"argMin", false}},
    {"AveragePool", {"averagePool2d", true}},
    {"BatchNormalization", {"meanVarianceNormalization", false}},
    {"Cast", {"cast", false}},
    {"Ceil", {"ceil", true}},
    {"Clip", {"clamp", true}},
    {"Concat", {"concat", true}},
    {"Conv", {"conv2d", true}},
    {"ConvInteger", {"conv2dInteger", false}},
    {"ConvTranspose", {"convTranspose2d", true}},
    {"Cos", {"cos", false}},
    {"Div", {"div", true}},
    {"DequantizeLinear", {"dequantizeLinear", false}},
    {"DynamicQuantizeLinear", {"dynamicQuantizeLinear", false}},
    {"Elu", {"elu", true}},
    {"Equal", {"equal", false}},
    {"Erf", {"erf", false}},
    {"Exp", {"exp", false}},
    {"Expand", {"expand", false}},
    {"Flatten", {"reshape", true}},
    {"Floor", {"floor", true}},
    {"Gather", {"gather", false}},
    {"Gemm", {"gemm", true}},
    {"GlobalAveragePool", {"averagePool2d", true}},
    {"GlobalMaxPool", {"maxPool2d", true}},
    {"GlobalLpPool", {"l2Pool2d", false}},
    {"Greater", {"greater", false}},
    {"GreaterOrEqual", {"greaterOrEqual", false}},
    {"GroupNormalization", {"meanVarianceNormalization", false}},
    {"HardSigmoid", {"hardSigmoid", false}},
    {"HardSwish", {"hardSwish", true}},
    {"Identity", {"identity", false}},
    {"InstanceNormalization", {"meanVarianceNormalization", false}},
    {"LayerNormalization", {"meanVarianceNormalization", false}},
    {"LeakyRelu", {"leakyRelu", true}},
    {"Less", {"lesser", false}},
    {"LessOrEqual", {"lesserOrEqual", false}},
    {"Log", {"log", false}},
    {"LpPool", {"l2Pool2d", false}},
    {"MatMul", {"matmul", false}},
    {"MatMulInteger", {"matmulInteger", false}},
    {"Max", {"max", true}},
    {"MaxPool", {"maxPool2d", true}},
    {"Min", {"min", true}},
    {"Mul", {"mul", true}},
    {"Neg", {"neg", true}},
    {"Not", {"logicalNot", false}},
    {"Pad", {"pad", true}},
    {"Pow", {"pow", true}},
    {"PRelu", {"prelu", true}},
    {"Reciprocal", {"reciprocal", false}},
    {"ReduceL1", {"reduceL1", false}},
    {"ReduceL2", {"reduceL2", false}},
    {"ReduceLogSum", {"reduceLogSum", false}},
    {"ReduceLogSumExp", {"reduceLogSumExp", false}},
    {"ReduceMax", {"reduceMax", false}},
    {"ReduceMean", {"reduceMean", true}},
    {"ReduceMin", {"reduceMin", false}},
    {"ReduceProd", {"reduceProduct", false}},
    {"ReduceSum", {"reduceSum", true}},
    {"ReduceSumSquare", {"reduceSumSquare", false}},
    {"Relu", {"relu", true}},
    {"Reshape", {"reshape", true}},
    {"Resize", {"resample2d", true}},
    {"Shape", {"slice", true}},
    {"Sigmoid", {"sigmoid", true}},
    {"Softplus", {"softplus", false}},
    {"Softsign", {"softsign", false}},
    {"Sin", {"sin", false}},
    {"Slice", {"slice", true}},
    {"Softmax", {"softmax", true}},
    {"Split", {"split", true}},
    {"Sqrt", {"sqrt", false}},
    {"Squeeze", {"reshape", true}},
    {"Sub", {"sub", true}},
    {"Tan", {"tan", false}},
    {"Tanh", {"tanh", true}},
    {"Transpose", {"transpose", true}},
    {"Unsqueeze", {"reshape", true}},
    {"Where", {"where", false}},
};

inline bool CheckSingleOp(const std::string& op_type, const emscripten::val& wnn_builder_,
                          const WebnnDeviceType device_type) {
  // Returns false if the op_type is not listed in the op_map.
  if (op_map.find(op_type) == op_map.end()) {
    return false;
  }
  // Returns false if the WebNN op has not been implemented in MLGraphBuilder in current browser.
  if (!wnn_builder_[op_map.find(op_type)->second.opName].as<bool>()) {
    return false;
  }
  // The current WebNN CPU (XNNPack) backend supports a limited op list, and we'd rather
  // fall back early to the ORT CPU EP rather than fail in the WebNN "cpu" deviceType.
  // This is a workaround because the op may be included in MLGraphBuilder for DirectML
  // backend but without XNNPack implementation in Chromium.
  if (!op_map.find(op_type)->second.isCpuSupported && device_type == WebnnDeviceType::CPU) {
    return false;
  }

  return true;
}

constexpr std::array<ONNX_NAMESPACE::TensorProto_DataType, 1> supported_cpu_data_types = {
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
};

constexpr std::array<ONNX_NAMESPACE::TensorProto_DataType, 9> supported_gpu_data_types = {
    ONNX_NAMESPACE::TensorProto_DataType_BOOL,
    ONNX_NAMESPACE::TensorProto_DataType_INT8,
    ONNX_NAMESPACE::TensorProto_DataType_UINT8,
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

bool SetWebnnDataType(emscripten::val& desc, const int32_t data_type);

}  // namespace webnn
}  // namespace onnxruntime
