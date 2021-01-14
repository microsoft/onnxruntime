// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <vector>

#include <core/common/safeint.h>
#include <core/common/logging/logging.h>
#include <core/framework/tensorprotoutils.h>
#include <core/graph/graph.h>
#include <core/graph/graph_viewer.h>
#include <core/providers/common.h>

#include "helper.h"
#include "op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

using std::string;
using std::vector;

std::string GetErrorCause(int error_code) {
  switch (error_code) {
    case ANEURALNETWORKS_NO_ERROR:
      return "ANEURALNETWORKS_NO_ERROR";
    case ANEURALNETWORKS_OUT_OF_MEMORY:
      return "ANEURALNETWORKS_OUT_OF_MEMORY";
    case ANEURALNETWORKS_INCOMPLETE:
      return "ANEURALNETWORKS_INCOMPLETE";
    case ANEURALNETWORKS_UNEXPECTED_NULL:
      return "ANEURALNETWORKS_UNEXPECTED_NULL";
    case ANEURALNETWORKS_BAD_DATA:
      return "ANEURALNETWORKS_BAD_DATA";
    case ANEURALNETWORKS_OP_FAILED:
      return "ANEURALNETWORKS_OP_FAILED";
    case ANEURALNETWORKS_BAD_STATE:
      return "ANEURALNETWORKS_BAD_STATE";
    case ANEURALNETWORKS_UNMAPPABLE:
      return "ANEURALNETWORKS_UNMAPPABLE";
    case ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE:
      return "ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE";
    case ANEURALNETWORKS_UNAVAILABLE_DEVICE:
      return "ANEURALNETWORKS_UNAVAILABLE_DEVICE";

    default:
      return "Unknown error code: " + std::to_string(error_code);
  }
}

QLinearOpType GetQLinearOpType(const onnxruntime::Node& node) {
  const auto& op_type = node.OpType();
  if (op_type == "DequantizeLinear")
    return QLinearOpType::DequantizeLinear;
  else if (op_type == "QuantizeLinear")
    return QLinearOpType::QuantizeLinear;
  else if (op_type == "QLinearConv")
    return QLinearOpType::QLinearConv;
  else if (op_type == "QLinearMatMul")
    return QLinearOpType::QLinearMatMul;
  else if (op_type == "QLinearAdd")
    return QLinearOpType::QLinearAdd;

  return QLinearOpType::Unknown;
}

ConvType GetConvType(const onnxruntime::Node& node, const InitializedTensorSet& initializers) {
  const auto& op_type = node.OpType();
  bool is_qlinear_conv = (op_type == "QLinearConv");
  ORT_ENFORCE(op_type == "Conv" || is_qlinear_conv);

  NodeAttrHelper helper(node);
  const auto group = helper.Get("group", 1);

  size_t w_idx = is_qlinear_conv ? 3 : 1;
  const auto& weight = node.InputDefs()[w_idx]->Name();
  const auto& weight_tensor = *initializers.at(weight);

  // For ONNX we only have 1 conv ops
  // For NNAPI we have 3
  // Input is (N, C, H, W)
  // group == 1,                                   --> regular conv
  // group != 1 && weight is (M, 1, kH, kW),       --> depthwise conv
  // group != 1 && weight is (M, C/group, kH, kW), --> grouped conv
  if (group == 1)
    return ConvType::Regular;
  else if ((weight_tensor.dims()[1] == 1))
    return ConvType::Depthwise;
  else
    return ConvType::Grouped;
}

bool IsQLinearBinaryOp(QLinearOpType qlinear_op_type) {
  return qlinear_op_type == QLinearOpType::QLinearConv ||
         qlinear_op_type == QLinearOpType::QLinearMatMul ||
         qlinear_op_type == QLinearOpType::QLinearAdd;
}

bool HasValidBinaryOpQuantizedInputs(const Node& node) {
  auto op_type = GetQLinearOpType(node);
  int32_t a_input_type, b_input_type;
  if (!IsQLinearBinaryOp(op_type)) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType() << "] is not a binary qlinear op";
    return false;
  }

  const auto input_defs(node.InputDefs());
  if (!GetType(*input_defs[0], a_input_type))
    return false;
  if (!GetType(*input_defs[3], b_input_type))
    return false;

  // QlinearConv supports u8u8 or u8s8
  // QLinearMatMul/Add only support u8u8
  bool is_qlinear_conv = op_type == QLinearOpType::QLinearConv;
  bool has_valid_qlinear_conv_weight =
      (b_input_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
       b_input_type == ONNX_NAMESPACE::TensorProto_DataType_INT8);

  if (a_input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
      (!is_qlinear_conv && a_input_type != b_input_type) ||
      (is_qlinear_conv && !has_valid_qlinear_conv_weight)) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] A Input type: [" << a_input_type
                          << "] B Input type: [" << b_input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool HasValidQuantizationScales(const InitializedTensorSet& initializers, const Node& node,
                                const std::vector<size_t>& indices, const OpSupportCheckParams& params) {
  const auto& op_type = node.OpType();
  bool is_qlinear_conv = (op_type == "QLinearConv");
  const auto input_defs(node.InputDefs());
  for (const auto idx : indices) {
    if (idx >= input_defs.size()) {
      LOGS_DEFAULT(VERBOSE) << "HasValidQuantizationScales, Input index,  " << idx
                            << " >= input number, " << input_defs.size();
      return false;
    }
    const auto scale_name = input_defs[idx]->Name();
    if (Contains(initializers, scale_name)) {
      const auto& scale_tensor = *initializers.at(scale_name);
      int64_t scales_dim = scale_tensor.dims().empty() ? 1 : scale_tensor.dims()[0];
      bool is_conv_weight = is_qlinear_conv && idx == 4;
      bool is_conv_u8s8_weight = false;

      if (is_conv_weight) {
        const auto& weight_tensor = *initializers.at(node.InputDefs()[3]->Name());
        is_conv_u8s8_weight = weight_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8;
      }

      // We need to check the per-channel quantization scales dimensions for u8s8 QlinearConv
      // We only support per-channel quantization for u8s8
      // For all other cases, the scales should be a scalar
      if (is_conv_u8s8_weight) {
        if (params.android_sdk_ver < 29) {
          LOGS_DEFAULT(VERBOSE) << op_type << " only supports per-channel quantization on Android API 29+, "
                                << "system API level: " << params.android_sdk_ver;
          return false;
        }

        const auto& weight_tensor = *initializers.at(node.InputDefs()[3]->Name());
        if (weight_tensor.dims()[0] != scales_dim) {
          LOGS_DEFAULT(VERBOSE) << op_type << " mismatch int8 per-channel quantization weight,"
                                << " weight dimension[0] " << weight_tensor.dims()[0]
                                << " scale dimension " << scales_dim;
          return false;
        }
      } else {
        if (scales_dim != 1) {
          LOGS_DEFAULT(VERBOSE) << op_type << " does not support per-channel quantization, "
                                << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
          return false;
        }
      }
    } else {
      LOGS_DEFAULT(VERBOSE) << "The scale of " << op_type << " must be known";
      return false;
    }
  }

  return true;
}

bool HasValidQuantizationZeroPoints(const InitializedTensorSet& initializers, const Node& node,
                                    const std::vector<size_t>& indices) {
  const auto& op_type = node.OpType();
  bool is_qlinear_conv = (op_type == "QLinearConv");
  const auto input_defs(node.InputDefs());
  for (const auto idx : indices) {
    if (idx >= input_defs.size()) {
      LOGS_DEFAULT(VERBOSE) << "HasValidQuantizationZeroPoints, Input index,  " << idx
                            << " >= input number, " << input_defs.size();
      return false;
    }

    const auto zero_point_name = input_defs[idx]->Name();
    if (Contains(initializers, zero_point_name)) {
      bool is_conv_weight = is_qlinear_conv && idx == 5;
      bool is_conv_u8s8_weight = false;
      if (is_conv_weight) {
        const auto& weight_tensor = *initializers.at(node.InputDefs()[3]->Name());
        is_conv_u8s8_weight = weight_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8;
      }

      const auto& zero_tensor = *initializers.at(zero_point_name);
      int64_t zero_dim = zero_tensor.dims().empty() ? 1 : zero_tensor.dims()[0];
      if (is_conv_u8s8_weight) {
        if (zero_tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
          LOGS_DEFAULT(VERBOSE) << "u8s8 QlinearConv only supports int8 zero point for weight, "
                                << "actual zero point type: [" << zero_tensor.data_type() << "]";
          return false;
        }

        // For onnx, u8s8 QlinearConv, the weight zero point can be a scalar,
        // or a tensor with same channel as weight, for NNAPI we only support it be
        // 0 (scalar) or all 0 (tensor), NNAPI will assume the zero point for per-channel
        // quantization is 0 there is no input for it
        const auto& weight_tensor = *initializers.at(node.InputDefs()[3]->Name());
        if (weight_tensor.dims()[0] != zero_dim && zero_dim != 1) {
          LOGS_DEFAULT(VERBOSE) << op_type << " mismatch int8 per-channel quantization weight,"
                                << " weight dimension[0] " << weight_tensor.dims()[0]
                                << " zero point dimension " << zero_dim;
          return false;
        }

        std::unique_ptr<uint8_t[]> unpacked_tensor;
        size_t tensor_byte_size;
        auto status = onnxruntime::utils::UnpackInitializerData(zero_tensor, node.ModelPath(),
                                                                unpacked_tensor, tensor_byte_size);
        if (!status.IsOK()) {
          LOGS_DEFAULT(ERROR) << "QLinearConv erro when unpack zero tensor:" << status.ErrorMessage();
          return false;
        }

        // Verify all onnx weight zero point(s) are 0(s)
        const int8_t* zero_points = reinterpret_cast<const int8_t*>(unpacked_tensor.get());
        for (size_t i = 0; i < tensor_byte_size; i++) {
          if (zero_points[i] != 0) {
            LOGS_DEFAULT(VERBOSE) << "QLinearConv only support 0 as zero point, "
                                  << "zero_points[" << i << "] has value: " << zero_points[i];
            return false;
          }
        }
      } else {
        if (zero_dim != 1) {
          LOGS_DEFAULT(VERBOSE) << op_type << " does not support per-channel quantization, "
                                << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
          return false;
        }
      }
    } else {
      LOGS_DEFAULT(VERBOSE) << "The zero point of " << op_type << " must be known";
      return false;
    }
  }

  return true;
}

#define GET_TENSOR_DATA(FUNC_NAME, ELEMENT_TYPE, DATA)                                  \
  const ELEMENT_TYPE* GetTensor##FUNC_NAME(const ONNX_NAMESPACE::TensorProto& tensor) { \
    return tensor.DATA().empty()                                                        \
               ? reinterpret_cast<const ELEMENT_TYPE*>(tensor.raw_data().data())        \
               : tensor.DATA().data();                                                  \
  }

GET_TENSOR_DATA(FloatData, float, float_data)
GET_TENSOR_DATA(Int32Data, int32_t, int32_data)
GET_TENSOR_DATA(Int64Data, int64_t, int64_data)

#undef GET_TENSOR_DATA

bool GetShape(const NodeArg& node_arg, Shape& shape) {
  shape.clear();
  const auto* shape_proto = node_arg.Shape();

  if (!shape_proto) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  // NNAPI uses 0 for dynamic dimension, which is the default value for dim.dim_value()
  for (const auto& dim : shape_proto->dim())
    shape.push_back(SafeInt<uint32_t>(dim.dim_value()));

  return true;
}

bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS_DEFAULT(WARNING) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool GetClipMinMax(const InitializedTensorSet& initializers, const Node& node, float& min, float& max) {
  min = std::numeric_limits<float>::lowest();
  max = std::numeric_limits<float>::max();
  if (node.SinceVersion() < 11) {  // Clip opset 1, 6 is using attributes for min/max
    NodeAttrHelper helper(node);
    min = helper.Get("min", std::numeric_limits<float>::lowest());
    max = helper.Get("max", std::numeric_limits<float>::max());
  } else {
    if (node.InputDefs().size() > 1) {  // we have input min
      const auto& min_name = node.InputDefs()[1]->Name();
      if (!Contains(initializers, min_name)) {
        LOGS_DEFAULT(VERBOSE) << "Input min of Clip must be known";
        return false;
      }
      min = GetTensorFloatData(*initializers.at(min_name))[0];
    }

    if (node.InputDefs().size() > 2) {  // we have input max
      const auto& max_name = node.InputDefs()[2]->Name();
      if (!Contains(initializers, max_name)) {
        LOGS_DEFAULT(VERBOSE) << "Input max of Clip must be known";
        return false;
      }
      max = GetTensorFloatData(*initializers.at(max_name))[0];
    }
  }

  return true;
}

void GetFlattenOutputShape(const Node& node, const Shape& input_shape, int32_t& dim_1, int32_t& dim_2) {
  int32_t rank = static_cast<int>(input_shape.size());
  NodeAttrHelper helper(node);
  int32_t axis = helper.Get("axis", 1);
  // axis == rank is a valid input, but invalid for HandleNegativeAxis
  // Skip non-negative axis here
  if (axis < 0)
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, rank));

  dim_1 = std::accumulate(input_shape.cbegin(), input_shape.cbegin() + axis, 1, std::multiplies<int32_t>());
  dim_2 = std::accumulate(input_shape.cbegin() + axis, input_shape.cend(), 1, std::multiplies<int32_t>());
}

bool IsValidSupportedNodesVec(const std::vector<size_t>& supported_node_vec, const GraphViewer& graph_viewer) {
  if (supported_node_vec.empty())
    return false;

  if (supported_node_vec.size() == 1) {
    const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
    const auto* node(graph_viewer.GetNode(node_indices[supported_node_vec[0]]));
    const auto& op = node->OpType();
    // It is not worth it to perform a single Reshape/Flatten/Identity operator
    // which is only copying the data in NNAPI
    // If this is the case, let it fall back
    if (op == "Reshape" ||
        op == "Flatten" ||
        op == "Identity") {
      return false;
    }
  }
  return true;
}

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const OpSupportCheckParams& params) {
  const auto& op_support_checkers = GetOpSupportCheckers();
  if (Contains(op_support_checkers, node.OpType())) {
    const auto* op_support_checker = op_support_checkers.at(node.OpType());
    return op_support_checker->IsOpSupported(graph_viewer.GetAllInitializedTensors(), node, params);
  } else {
    return false;
  }
}

std::vector<std::vector<size_t>> GetSupportedNodes(const GraphViewer& graph_viewer, const OpSupportCheckParams& params) {
  std::vector<std::vector<size_t>> supported_node_vecs;
  if (params.android_sdk_ver < ORT_NNAPI_MIN_API_LEVEL) {
    LOGS_DEFAULT(WARNING) << "All ops will fallback to CPU EP, because Android API level [" << params.android_sdk_ver
                          << "] is lower than minimal supported API level [" << ORT_NNAPI_MIN_API_LEVEL
                          << "] of this build for NNAPI";
    return supported_node_vecs;
  }

  std::vector<size_t> supported_node_vec;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer.GetNode(node_indices[i]));
    bool supported = IsNodeSupported(*node, graph_viewer, params);
    LOGS_DEFAULT(VERBOSE) << "Operator type: [" << node->OpType()
                          << "] index: [" << i
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
      if (IsValidSupportedNodesVec(supported_node_vec, graph_viewer)) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }

  if (IsValidSupportedNodesVec(supported_node_vec, graph_viewer))
    supported_node_vecs.push_back(supported_node_vec);

  return supported_node_vecs;
}

std::string Shape2String(const std::vector<uint32_t>& shape) {
  std::ostringstream os;
  os << "[ ";
  for (const auto& dim : shape)
    os << dim << " ";

  os << "]";
  return os.str();
}

NodeAttrHelper::NodeAttrHelper(const onnxruntime::Node& node)
    : node_attributes_(node.GetAttributes()) {}

float NodeAttrHelper::Get(const std::string& key, float def_val) const {
  if (HasAttr(key))
    return node_attributes_.at(key).f();

  return def_val;
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  if (HasAttr(key))
    return SafeInt<int32_t>(node_attributes_.at(key).i());

  return def_val;
}

string NodeAttrHelper::Get(const std::string& key, const string& def_val) const {
  if (HasAttr(key))
    return node_attributes_.at(key).s();

  return def_val;
}

vector<int32_t> NodeAttrHelper::Get(const std::string& key, const vector<int32_t>& def_val) const {
  if (HasAttr(key)) {
    const auto& attr(node_attributes_.at(key));
    std::vector<int32_t> v;
    v.reserve(static_cast<size_t>(attr.ints_size()));
    for (int j = 0; j < attr.ints_size(); j++) {
      int64_t val = attr.ints(j);
      v.push_back(SafeInt<int32_t>(val));
    }
    return v;
  }

  return def_val;
}

vector<float> NodeAttrHelper::Get(const std::string& key, const vector<float>& def_val) const {
  if (HasAttr(key)) {
    const auto& attr(node_attributes_.at(key));
    std::vector<float> v;
    v.reserve(static_cast<size_t>(attr.ints_size()));
    for (int j = 0; j < attr.ints_size(); j++) {
      v.push_back(attr.floats(j));
    }

    return v;
  }

  return def_val;
}

bool NodeAttrHelper::HasAttr(const std::string& key) const {
  return Contains(node_attributes_, key);
}

}  // namespace nnapi
}  // namespace onnxruntime
