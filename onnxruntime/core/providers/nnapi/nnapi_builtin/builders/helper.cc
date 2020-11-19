//
// Created by daquexian on 8/3/18.
//

#include <iostream>
#include <string>
#include <vector>

#include <core/common/safeint.h>
#include <core/common/logging/logging.h>
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

bool IsQLinearBinaryOp(QLinearOpType qlinear_op_type) {
  return qlinear_op_type == QLinearOpType::QLinearConv ||
         qlinear_op_type == QLinearOpType::QLinearMatMul ||
         qlinear_op_type == QLinearOpType::QLinearAdd;
}

bool HasValidBinaryOpQuantizedInputs(const Node& node) {
  int32_t a_input_type, b_input_type;
  if (!IsQLinearBinaryOp(GetQLinearOpType(node))) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType() << "] is not a binary qlinear op";
    return false;
  }

  const auto input_defs(node.InputDefs());
  if (!GetType(*input_defs[0], a_input_type))
    return false;
  if (!GetType(*input_defs[3], b_input_type))
    return false;

  if (a_input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8 || a_input_type != b_input_type) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] A Input type: [" << a_input_type
                          << "] B Input type: [" << b_input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool HasValidQuantizationScales(const InitializedTensorSet& initializers, const Node& node,
                                const std::vector<size_t>& indices) {
  const auto& op = node.OpType();
  const auto input_defs(node.InputDefs());
  for (const auto idx : indices) {
    if (idx >= input_defs.size()) {
      LOGS_DEFAULT(VERBOSE) << "HasValidQuantizationScales, Input index,  " << idx
                            << " >= input number, " << input_defs.size();
      return false;
    }
    const auto scale_name = input_defs[idx]->Name();
    if (Contains(initializers, scale_name)) {
      const auto& tensor = *initializers.at(scale_name);
      if (!tensor.dims().empty() && tensor.dims()[0] != 1) {
        LOGS_DEFAULT(VERBOSE) << op << " does not support per-channel quantization";
        return false;
      }
    } else {
      LOGS_DEFAULT(VERBOSE) << "The scale of " << op << " must be known";
      return false;
    }
  }

  return true;
}

bool HasValidQuantizationZeroPoints(const InitializedTensorSet& initializers, const Node& node,
                                    const std::vector<size_t>& indices) {
  const auto& op = node.OpType();
  const auto input_defs(node.InputDefs());
  for (const auto idx : indices) {
    if (idx >= input_defs.size()) {
      LOGS_DEFAULT(VERBOSE) << "HasValidQuantizationZeroPoints, Input index,  " << idx
                            << " >= input number, " << input_defs.size();
      return false;
    }
    const auto zero_point_name = node.InputDefs()[idx]->Name();
    if (Contains(initializers, zero_point_name)) {
      const auto& tensor = *initializers.at(zero_point_name);
      if (!tensor.dims().empty() && tensor.dims()[0] != 1) {
        LOGS_DEFAULT(VERBOSE) << op << " does not support per-channel quantization";
        return false;
      }
      if (tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
        LOGS_DEFAULT(VERBOSE) << op << " does not support zero point data type "
                              << std::to_string(tensor.data_type());
        return false;
      }
    } else {
      LOGS_DEFAULT(VERBOSE) << "The zero point of " << op << " must be known";
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
    const auto op_support_checker = op_support_checkers.at(node.OpType());
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

  LOGS_DEFAULT(VERBOSE) << "Support vectors size is " << supported_node_vecs.size();
  for (const auto& group : supported_node_vecs)
    LOGS_DEFAULT(VERBOSE) << "Support vector size is " << group.size();

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
