// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"

#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace nnapi {

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

QuantizedOpType GetQuantizedOpType(const NodeUnit& node_unit) {
  const auto& op_type = node_unit.OpType();
  if (node_unit.UnitType() == NodeUnit::Type::SingleNode) {
    if (op_type == "DequantizeLinear")
      return QuantizedOpType::DequantizeLinear;
    else if (op_type == "QuantizeLinear")
      return QuantizedOpType::QuantizeLinear;
    else if (op_type == "QLinearConv")
      return QuantizedOpType::QLinearConv;
    else if (op_type == "QLinearMatMul")
      return QuantizedOpType::QLinearMatMul;
    else if (op_type == "QLinearAdd")
      return QuantizedOpType::QLinearAdd;
    else if (op_type == "QLinearMul")
      return QuantizedOpType::QLinearMul;
    else if (op_type == "QLinearSigmoid")
      return QuantizedOpType::QLinearSigmoid;
    else if (op_type == "QLinearAveragePool")
      return QuantizedOpType::QLinearAveragePool;
  } else if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    if (op_type == "Conv")
      return QuantizedOpType::QDQConv;
    else if (op_type == "Resize")
      return QuantizedOpType::QDQResize;
    else if (op_type == "AveragePool")
      return QuantizedOpType::QDQAveragePool;
    else if (op_type == "Add")
      return QuantizedOpType::QDQAdd;
    else if (op_type == "Mul")
      return QuantizedOpType::QDQMul;
    else if (op_type == "Transpose")
      return QuantizedOpType::QDQTranspose;
    else if (op_type == "Reshape")
      return QuantizedOpType::QDQReshape;
    else if (op_type == "Softmax")
      return QuantizedOpType::QDQSoftmax;
    else if (op_type == "Concat")
      return QuantizedOpType::QDQConcat;
    else if (op_type == "Gemm")
      return QuantizedOpType::QDQGemm;
    else if (op_type == "MatMul")
      return QuantizedOpType::QDQMatMul;
  } else {
    // throw?
  }

  return QuantizedOpType::Unknown;
}

ConvType GetConvType(const NodeUnit& node_unit, const InitializedTensorSet& initializers) {
  NodeAttrHelper helper(node_unit);
  const auto group = helper.Get("group", 1);

  const auto& weight = node_unit.Inputs()[1].node_arg.Name();
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

bool IsQuantizedConv(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearConv) ||
         (quant_op_type == QuantizedOpType::QDQConv);
}

bool IsQuantizedPool(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearAveragePool) ||
         (quant_op_type == QuantizedOpType::QDQAveragePool);
}

bool IsQuantizedGemm(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearMatMul) ||
         (quant_op_type == QuantizedOpType::QDQGemm) ||
         (quant_op_type == QuantizedOpType::QDQMatMul);
}

bool IsQuantizedBinaryOp(QuantizedOpType quant_op_type) {
  return quant_op_type == QuantizedOpType::QLinearMatMul ||
         quant_op_type == QuantizedOpType::QLinearAdd ||
         quant_op_type == QuantizedOpType::QLinearMul ||
         quant_op_type == QuantizedOpType::QDQAdd ||
         quant_op_type == QuantizedOpType::QDQMul ||
         quant_op_type == QuantizedOpType::QDQGemm ||
         quant_op_type == QuantizedOpType::QDQMatMul ||
         IsQuantizedConv(quant_op_type);
}

bool HasValidBinaryOpQuantizedInputTypes(const NodeUnit& node_unit) {
  auto quant_op_type = GetQuantizedOpType(node_unit);
  int32_t a_input_type, b_input_type;
  if (!IsQuantizedBinaryOp(quant_op_type)) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType() << "] is not a binary qlinear op";
    return false;
  }

  const auto& inputs = node_unit.Inputs();
  if (!GetType(inputs[0].node_arg, a_input_type))
    return false;
  if (!GetType(inputs[1].node_arg, b_input_type))
    return false;

  // QlinearConv/MatMul/QDQGemm/QDQMatMul supports u8u8 or u8s8
  // QLinearAdd/QLinearMul only support u8u8
  bool is_quant_conv_or_gemm = IsQuantizedConv(quant_op_type) || IsQuantizedGemm(quant_op_type);

  bool has_valid_qlinear_conv_weight =
      (b_input_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
       b_input_type == ONNX_NAMESPACE::TensorProto_DataType_INT8);

  if (a_input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
      (!is_quant_conv_or_gemm && a_input_type != b_input_type) ||
      (is_quant_conv_or_gemm && !has_valid_qlinear_conv_weight)) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType()
                          << "] A Input type: [" << a_input_type
                          << "] B Input type: [" << b_input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

common::Status GetQuantizationScaleAndZeroPoint(
    const InitializedTensorSet& initializers, const NodeUnitIODef& io_def, const Path& model_path,
    float& scale, int32_t& zero_point) {
  scale = 0.0f;
  zero_point = 0;

  if (!io_def.quant_param) {  // Not a quantized IO
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "NodeArg: ", io_def.node_arg.Name(), " is not quantized");
  }

  const auto& quant_param = *io_def.quant_param;
  {  // get the scale
    const auto& name = quant_param.scale.Name();
    Initializer unpacked_tensor(*initializers.at(name), model_path);
    // The scale should be one or more floats
    scale = unpacked_tensor.DataAsSpan<float>()[0];
  }

  if (quant_param.zero_point) {  // get the zero point if it's there
    const auto& name = quant_param.zero_point->Name();
    Initializer unpacked_tensor(*initializers.at(name), model_path);
    // Onnx quantization uses uint8 [int8 not yet supported], need to cast to int32_t used by NNAPI
    zero_point = static_cast<int32_t>(unpacked_tensor.DataAsByteSpan()[0]);
  }

  return Status::OK();
}

common::Status GetQuantizationScaleAndZeroPoint(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit, const std::string& name,
    float& scale, int32_t& zero_point, ArgType arg_type) {
  const auto& io_defs = arg_type == ArgType::kInput ? node_unit.Inputs() : node_unit.Outputs();
  for (const auto& io_def : io_defs) {
    if (io_def.node_arg.Name() == name)
      return GetQuantizationScaleAndZeroPoint(initializers, io_def, node_unit.ModelPath(),
                                              scale, zero_point);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Unknown input: ", name, ", for NodeUnit with node index: ", node_unit.Index());
}

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

Shape GetShapeInfoFromNodeArg(const GraphViewer& graph_viewer, const std::string& name) {
  // can be applied to both input and output
  Shape shape;
  const auto* node_arg = graph_viewer.GetNodeArg(name);
  const auto* shape_proto = node_arg->Shape();

  shape.reserve(shape_proto->dim_size());
  for (const auto& shape_dim : shape_proto->dim()) {
    // shape_dim here can possibly have dim_param, but as dynamic shape is not supported in NNAPI for now
    // (checked already in BaseOpSupportChecker), call dim_value here only.
    shape.push_back(SafeInt<uint32_t>(shape_dim.dim_value()));
  }
  // If we have an empty shape, (scalar input), we need to make it as {1} as
  // nnapi will treat empty shape as dynamic ranking and onnx does not support that
  if (shape_proto->dim_size() == 0) {
    shape.push_back(1);
  }
  return shape;
}

bool IsValidSupportedNodeGroup(const std::vector<const Node*>& supported_node_partition) {
  if (supported_node_partition.size() == 1) {
    const auto* node = supported_node_partition[0];
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

static bool IsInternalQuantizedNodeUnit(const NodeUnit& node_unit) {
  // First, ignore QDQ NodeUnit which is not internal quantized node
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup)
    return false;

  // These operators can use uint8 input without specific QLinear version of it
  // However, the mode has to be internal to the graph/partition (they cannot consume graph inputs)
  static const std::unordered_set<std::string> internal_quantized_op_types =
      {
          "Transpose",
          "Resize",
          "Concat",
          "MaxPool",
      };

  const auto& node = node_unit.GetNode();
  if (!Contains(internal_quantized_op_types, node.OpType()))
    return false;

  int32_t input_type;
  ORT_ENFORCE(GetType(*node.InputDefs()[0], input_type));

  return input_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8;
}

// We support some operators running using uint8 internally
// These nodes cannot use a graph input as input since onnx graph input does not carry scale/zero point info
bool IsInternalQuantizationSupported(const Node& node, const std::unordered_set<std::string>& node_outputs_in_group) {
  const auto& op_type = node.OpType();

  // The node's input(s) have to be an output of node(s) within the group
  // If not, then this node is using graph/partition input(s) as input(s)
  const auto& input_defs = node.InputDefs();

  // We only need to check input0 for all operators except "Concat"
  bool check_all_inputs = op_type == "Concat";

  for (size_t i = 0; i < (check_all_inputs ? input_defs.size() : 1); i++) {
    if (!Contains(node_outputs_in_group, input_defs[i]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Node [" << node.Name() << "] type: [" << op_type
                            << "] has input [" << input_defs[i]->Name()
                            << "] does not support using graph input(quantized) as node input";
      return false;
    }
  }

  return true;
}

bool IsNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph_viewer, const OpSupportCheckParams& params) {
  const auto& op_builders = GetOpBuilders();
  const auto op_builder_it = op_builders.find(node_unit.OpType());
  if (op_builder_it == op_builders.end()) {
    return false;
  }

  const auto* op_builder = op_builder_it->second;
  return op_builder->IsOpSupported(graph_viewer.GetAllInitializedTensors(), node_unit, params);
}

bool IsNodeSupportedInGroup(const NodeUnit& node_unit, const GraphViewer& graph_viewer,
                            const OpSupportCheckParams& params,
                            const std::unordered_set<std::string>& node_outputs_in_group) {
  if (!IsNodeSupported(node_unit, graph_viewer, params))
    return false;

  // We also want to check if the node is supported as an internal quantized node_unit
  if (IsInternalQuantizedNodeUnit(node_unit))
    return IsInternalQuantizationSupported(node_unit.GetNode(), node_outputs_in_group);

  return true;
}

std::string Shape2String(const Shape& shape) {
  std::ostringstream os;
  os << "[ ";
  for (const auto& dim : shape)
    os << dim << " ";

  os << "]";
  return os.str();
}

uint32_t ShapeSize(const Shape& shape, size_t begin_idx, size_t end_idx) {
  ORT_ENFORCE(begin_idx <= end_idx && begin_idx <= shape.size(),
              "Invalid indices: begin [", begin_idx, "], end [", end_idx, "], shape size [", shape.size(), "]");
  return std::accumulate(shape.begin() + begin_idx, shape.begin() + end_idx,
                         SafeInt<uint32_t>{1}, std::multiplies<SafeInt<uint32_t>>{});
}

bool CheckIsInitializer(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                        const std::string& input_name, const char* input_description) {
  if (!Contains(initializers, input_name)) {
    LOGS_DEFAULT(VERBOSE) << input_description << " of " << node_unit.Name() << "of type ["
                          << node_unit.OpType() << "] must be an initializer tensor";
    return false;
  }

  return true;
}

std::vector<int32_t> OnnxAxesToNnapi(gsl::span<const int64_t> onnx_axes, std::optional<size_t> input_rank) {
  std::vector<int32_t> result;
  result.reserve(onnx_axes.size());
  for (auto dim : onnx_axes) {
    if (input_rank.has_value()) {
      dim = HandleNegativeAxis(dim, *input_rank);
    }

    result.push_back(narrow<int32_t>(dim));
  }

  return result;
}

}  // namespace nnapi
}  // namespace onnxruntime
