// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstring>
#include <core/common/status.h>
#include "core/common/inlined_containers.h"
#include <core/graph/basic_types.h>
#include "core/optimizer/initializer.h"
#include "core/providers/shared/utils/utils.h"
#include "map_info.h"

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

inline std::vector<int64_t> convertAxesFromNCHWtoNHWC(const std::vector<int64_t>& axes) {
  constexpr std::array<int64_t, 4> nchw_to_nhwc = {0, 3, 1, 2};
  std::vector<int64_t> new_axes;
  new_axes.reserve(axes.size());
  for (int64_t axis : axes) {
    if (axis >= nchw_to_nhwc.size()) {
      ORT_THROW("Invalid axis value: ", axis);
    }
    new_axes.push_back(nchw_to_nhwc[static_cast<size_t>(axis)]);
  }
  return new_axes;
}

inline std::vector<int64_t> HandleNegativeAxes(const std::vector<int64_t>& axes, size_t input_size) {
  std::vector<int64_t> new_axes(axes.size());
  for (size_t i = 0; i < axes.size(); ++i) {
    new_axes[i] = HandleNegativeAxis(axes[i], input_size);
  }
  return new_axes;
}

inline std::vector<int64_t> GetResolvedAxes(const NodeAttrHelper& helper, size_t input_size) {
  return HandleNegativeAxes(helper.Get("axes", std::vector<int64_t>{}), input_size);
}

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

template <typename T>
inline std::vector<T> GetNarrowedIntfromInt64(gsl::span<const int64_t> int64_vec) {
  std::vector<T> vec;
  vec.reserve(int64_vec.size());
  std::transform(int64_vec.begin(), int64_vec.end(),
                 std::back_inserter(vec),
                 [](int64_t val) -> T { return SafeInt<T>(val); });
  return vec;
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
      LOGS(logger, ERROR) << "WebNN backend does not support data type: " << tensor.data_type();
      return false;
      break;
  }
  return true;
}

inline bool IsEmptyTensor(const GraphViewer& graph_viewer, const std::string& name) {
  const auto* tensor_init = graph_viewer.GetConstantInitializer(name);
  if (name.empty() || !tensor_init) {
    return true;
  }

  const auto& tensor = *tensor_init;
  const auto dims = tensor.dims();
  // An empty tensor contains a 0 in the dimensions list.
  return std::any_of(dims.begin(), dims.end(), [](auto d) { return d == 0; });
}

inline bool IsOnnxDomain(std::string_view domain) {
  return (domain == onnxruntime::kOnnxDomain) || (domain == onnxruntime::kOnnxDomainAlias);
}

inline bool TensorExists(const ConstPointerContainer<std::vector<NodeArg*>>& defs, size_t tensor_index) noexcept {
  return tensor_index < defs.size() && defs[tensor_index]->Exists();
}

bool IsTensorShapeSupported(const NodeArg& node_arg, const std::string& parent_name,
                            const logging::Logger& logger, bool allow_empty_input = false);

bool IsInputRankSupportedByOp(const Node& node, const emscripten::val& wnn_limits, const logging::Logger& logger);

// Get a set of nodes supported by WebNN EP.
std::unordered_set<const Node*> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                  const emscripten::val& wnn_builder,
                                                  const WebnnDeviceType device_type,
                                                  const emscripten::val& wnn_limits,
                                                  const logging::Logger& logger);

// Retrieve the first input name of a WebNN op used for validating supported input data types.
// WebNN ops have various first input names such as 'a', 'input', 'inputs', etc.
// All WebNN op inputs are recorded in op_inputs_map.
inline std::string_view GetWebNNOpFirstInputName(const std::string_view webnn_op_type) {
  auto it = op_inputs_map.find(webnn_op_type);
  if (it != op_inputs_map.end()) {
    for (const auto& input : it->second.inputs) {
      if (input.index == 0) {
        return input.name;
      }
    }
  }
  return "input";
}

inline std::string_view GetWebNNOpType(const std::string_view op_type) {
  auto it = op_inputs_map.find(op_type);
  // Return an empty string if the op_type is not listed in the op_inputs_map.
  return (it != op_inputs_map.end()) ? it->second.opType : "";
}

bool AreDataTypesSame(const std::string_view op_type,
                      gsl::span<const int32_t> input_types,
                      const logging::Logger& logger);
bool IsSupportedDataType(const int32_t onnx_data_type, const emscripten::val& webnn_supported_data_types);
bool IsDataTypeSupportedByOp(const std::string_view onnx_op_type,
                             const int32_t onnx_data_type,
                             const emscripten::val& wnn_limits,
                             const std::string_view webnn_input_output_name,
                             const std::string_view onnx_input_output_name,
                             const logging::Logger& logger);
bool IsDataTypeSupportedByWebNNOp(const std::string_view onnx_op_type,
                                  const std::string_view webnn_op_type,
                                  const int32_t onnx_data_type,
                                  const emscripten::val& wnn_limits,
                                  const std::string_view webnn_input_output_name,
                                  const std::string_view onnx_input_output_name,
                                  const logging::Logger& logger);

bool GetBidirectionalBroadcastShape(std::vector<int64_t>& shape_a,
                                    std::vector<int64_t>& shape_b,
                                    std::vector<int64_t>& output_shape);

bool SetWebnnDataType(emscripten::val& desc, const int32_t data_type);

bool IsMLTensorSupported();

uint8_t PackInt8ToUint8DoubledNibbles(int8_t value, const int32_t& data_type);
uint16_t PackFloat32ToUint16AsFloat16(float value);

}  // namespace webnn
}  // namespace onnxruntime
