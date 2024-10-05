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
  NPU,
};

WebnnDeviceType DeviceTypeFromString(const std::string_view& device_type);

// Collects all the initializer tensors in the subGraph and its ancestor graphs.
InitializedTensorSet CollectAllInitializedTensors(const GraphViewer& graph_viewer);

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

inline std::string GetTensorName(const ConstPointerContainer<std::vector<NodeArg*>>& input_defs, const size_t index) {
  return (input_defs.size() > index) ? std::string(input_defs[index]->Name()) : "";
}

inline std::vector<uint32_t> GetVecUint32FromVecInt64(const std::vector<int64_t>& int64_vec) {
  std::vector<uint32_t> uint32_vec;
  uint32_vec.reserve(int64_vec.size());
  std::transform(int64_vec.begin(), int64_vec.end(),
                 std::back_inserter(uint32_vec),
                 [](int64_t val) -> uint32_t { return SafeInt<uint32_t>(val); });
  return uint32_vec;
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
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      scalar = emscripten::val{*reinterpret_cast<uint8_t*>(unpacked_tensor.data())};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      scalar = emscripten::val{*reinterpret_cast<int8_t*>(unpacked_tensor.data())};
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
                                                      const emscripten::val& wnn_builder,
                                                      const WebnnDeviceType device_type,
                                                      const emscripten::val& wnn_limits,
                                                      const logging::Logger& logger);
static const InlinedHashMap<std::string, std::string> op_map = {
    {"Abs", "abs"},
    {"Add", "add"},
    {"ArgMax", "argMax"},
    {"ArgMin", "argMin"},
    {"AveragePool", "averagePool2d"},
    {"BatchNormalization", "batchNormalization"},
    {"Cast", "cast"},
    {"Ceil", "ceil"},
    {"Clip", "clamp"},
    {"Concat", "concat"},
    {"Conv", "conv2d"},
    {"ConvInteger", "conv2dInteger"},
    {"ConvTranspose", "convTranspose2d"},
    {"Cos", "cos"},
    {"Div", "div"},
    {"DequantizeLinear", "dequantizeLinear"},
    {"Dropout", "identity"},
    {"DynamicQuantizeLinear", "dynamicQuantizeLinear"},
    {"Elu", "elu"},
    {"Equal", "equal"},
    {"Erf", "erf"},
    {"Exp", "exp"},
    {"Expand", "expand"},
    {"Flatten", "reshape"},
    {"Floor", "floor"},
    {"Gather", "gather"},
    {"Gelu", "gelu"},
    {"Gemm", "gemm"},
    {"GlobalAveragePool", "averagePool2d"},
    {"GlobalMaxPool", "maxPool2d"},
    {"GlobalLpPool", "l2Pool2d"},
    {"Greater", "greater"},
    {"GreaterOrEqual", "greaterOrEqual"},
    {"Gru", "gru"},
    {"HardSigmoid", "hardSigmoid"},
    {"HardSwish", "hardSwish"},
    {"Identity", "identity"},
    {"InstanceNormalization", "instanceNormalization"},
    {"LayerNormalization", "layerNormalization"},
    {"LeakyRelu", "leakyRelu"},
    {"Less", "lesser"},
    {"LessOrEqual", "lesserOrEqual"},
    {"Log", "log"},
    {"LpPool", "l2Pool2d"},
    {"LSTM", "lstm"},
    {"MatMul", "matmul"},
    {"MatMulInteger", "matmulInteger"},
    {"Max", "max"},
    {"MaxPool", "maxPool2d"},
    {"Min", "min"},
    {"Mul", "mul"},
    {"Neg", "neg"},
    {"Not", "logicalNot"},
    {"Pad", "pad"},
    {"Pow", "pow"},
    {"PRelu", "prelu"},
    {"QuantizeLinear", "quantizeLinear"},
    {"Reciprocal", "reciprocal"},
    {"ReduceL1", "reduceL1"},
    {"ReduceL2", "reduceL2"},
    {"ReduceLogSum", "reduceLogSum"},
    {"ReduceLogSumExp", "reduceLogSumExp"},
    {"ReduceMax", "reduceMax"},
    {"ReduceMean", "reduceMean"},
    {"ReduceMin", "reduceMin"},
    {"ReduceProd", "reduceProduct"},
    {"ReduceSum", "reduceSum"},
    {"ReduceSumSquare", "reduceSumSquare"},
    {"Relu", "relu"},
    {"Reshape", "reshape"},
    {"Resize", "resample2d"},
    {"Shape", "slice"},
    {"Sigmoid", "sigmoid"},
    {"Softplus", "softplus"},
    {"Softsign", "softsign"},
    {"Sin", "sin"},
    {"Slice", "slice"},
    {"Softmax", "softmax"},
    {"Split", "split"},
    {"Sqrt", "sqrt"},
    {"Squeeze", "reshape"},
    {"Sub", "sub"},
    {"Tan", "tan"},
    {"Tanh", "tanh"},
    {"Tile", "tile"},
    {"Transpose", "transpose"},
    {"Trilu", "triangular"},
    {"Unsqueeze", "reshape"},
    {"Where", "where"},
};

inline bool CheckSingleOp(const std::string& op_type, const emscripten::val& wnn_builder,
                          const WebnnDeviceType device_type) {
  auto op_map_entry = op_map.find(op_type);
  // Returns false if the op_type is not listed in the op_map or
  // if the WebNN op has not been implemented in MLGraphBuilder in current browser.
  if (op_map_entry == op_map.end() || !wnn_builder[op_map_entry->second].as<bool>()) {
    return false;
  }

  return true;
}

inline bool GetWebNNOpType(const std::string& op_type, std::string& webnn_op_type) {
  auto it = op_map.find(op_type);
  // Returns false if the op_type is not listed in the op_map.
  if (it == op_map.end()) {
    return false;
  }
  webnn_op_type = it->second;
  return true;
}

static const InlinedHashMap<ONNX_NAMESPACE::TensorProto_DataType, std::string> onnx_to_webnn_data_type_map = {
    {ONNX_NAMESPACE::TensorProto_DataType_BOOL, "uint8"},
    {ONNX_NAMESPACE::TensorProto_DataType_INT8, "int8"},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT8, "uint8"},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, "float16"},
    {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, "float32"},
    {ONNX_NAMESPACE::TensorProto_DataType_INT32, "int32"},
    {ONNX_NAMESPACE::TensorProto_DataType_INT64, "int64"},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT32, "uint32"},
    {ONNX_NAMESPACE::TensorProto_DataType_UINT64, "uint64"},
};

bool AreInputDataTypesSame(const std::string& op_type,
                           gsl::span<const int32_t> input_types,
                           const logging::Logger& logger);
bool IsSupportedDataType(const int32_t onnx_data_type, const emscripten::val& webnn_supported_data_types);
bool IsDataTypeSupportedByOp(const std::string& onnx_op_type,
                             const int32_t onnx_data_type,
                             const emscripten::val& wnn_limits,
                             const std::string& webnn_input_output_name,
                             const std::string& onnx_input_output_name,
                             const logging::Logger& logger);

bool GetBidirectionalBroadcastShape(std::vector<int64_t>& shape_a,
                                    std::vector<int64_t>& shape_b,
                                    std::vector<int64_t>& output_shape);

bool SetWebnnDataType(emscripten::val& desc, const int32_t data_type);

bool IsMLTensorSupported();

}  // namespace webnn
}  // namespace onnxruntime
