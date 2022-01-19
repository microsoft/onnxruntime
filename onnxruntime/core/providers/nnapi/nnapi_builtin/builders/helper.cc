// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <vector>

#include "helper.h"

#include "core/common/safeint.h"
#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "op_support_checker.h"

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
  else if (op_type == "QLinearSigmoid")
    return QLinearOpType::QLinearSigmoid;
  else if (op_type == "QLinearAveragePool")
    return QLinearOpType::QLinearAveragePool;

  return QLinearOpType::Unknown;
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

bool IsQLinearBinaryOp(QLinearOpType qlinear_op_type) {
  return qlinear_op_type == QLinearOpType::QLinearConv ||
         qlinear_op_type == QLinearOpType::QLinearMatMul ||
         qlinear_op_type == QLinearOpType::QLinearAdd;
}

bool HasValidUnaryOpQuantizedInputs(const NodeUnit& node_unit) {
  int32_t input_type;
  if (!GetType(node_unit.Inputs()[0].node_arg, input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool HasValidBinaryOpQuantizedInputs(const NodeUnit& node_unit) {
  auto op_type = GetQLinearOpType(node_unit.GetNode());
  int32_t a_input_type, b_input_type;
  if (!IsQLinearBinaryOp(op_type)) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType() << "] is not a binary qlinear op";
    return false;
  }

  const auto& inputs = node_unit.Inputs();
  if (!GetType(inputs[0].node_arg, a_input_type))
    return false;
  if (!GetType(inputs[1].node_arg, b_input_type))
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
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType()
                          << "] A Input type: [" << a_input_type
                          << "] B Input type: [" << b_input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool HasValidQuantizationScales(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                const std::vector<size_t>& indices, const OpSupportCheckParams& params, bool is_input) {
  const auto& op_type = node_unit.OpType();
  auto qlinear_op_type = GetQLinearOpType(node_unit.GetNode());
  bool is_qlinear_conv = (qlinear_op_type == QLinearOpType::QLinearConv);
  bool is_qlinear_matmul = (qlinear_op_type == QLinearOpType::QLinearMatMul);
  const auto& io_defs = is_input ? node_unit.Inputs() : node_unit.Outputs();
  for (const auto idx : indices) {
    if (idx >= io_defs.size()) {
      LOGS_DEFAULT(VERBOSE) << (is_input ? "Input" : "Output") << " index,  " << idx
                            << " >= size, " << io_defs.size()
                            << " of NodeUnit: " << node_unit.Name();
      return false;
    }

    const auto& io_def = io_defs[idx];
    if (!io_def.quant_param.has_value()) {
      LOGS_DEFAULT(VERBOSE) << "HasValidQuantizationZeroPoints, Input index,  " << idx
                            << " has no quant_param";
      return false;
    }

    const auto scale_name = io_def.quant_param->scale.Name();

    if (!Contains(initializers, scale_name)) {
      LOGS_DEFAULT(VERBOSE) << "The scale of " << op_type << " must be an initializer tensor";
      return false;
    }

    // If this op is Qlinear[Conv/MatMul], we want to check u8s8 support for weight tensor (or B tensor for QlinearMatMul)
    bool is_conv_matmul_weight = is_input && (is_qlinear_conv || is_qlinear_matmul) && idx == 1;
    bool is_conv_matmul_u8s8_weight = false;

    if (is_conv_matmul_weight) {
      const auto& weight_tensor = *initializers.at(io_def.node_arg.Name());
      is_conv_matmul_u8s8_weight = weight_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8;
    }

    const auto& scale_tensor = *initializers.at(scale_name);
    int64_t scales_dim = scale_tensor.dims().empty() ? 1 : scale_tensor.dims()[0];
    if (!is_conv_matmul_u8s8_weight) {
      if (scales_dim != 1) {
        LOGS_DEFAULT(VERBOSE) << op_type << " does not support per-channel quantization, "
                              << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
        return false;
      }
    } else if (scales_dim != 1) {
      // For u8s8 Qlinear[Conv/MatMul], we support
      // 1. Per-tensor, the weight will be transformed to uint8 later
      // 2. Per-channel, only from Android API level 29
      if (is_qlinear_matmul) {
        LOGS_DEFAULT(VERBOSE) << "QLinearMatMul does not support per-channel quantization";
        return false;
      }

      if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
        LOGS_DEFAULT(VERBOSE) << op_type << " only supports per-channel quantization on Android API 29+, "
                              << "system NNAPI feature level: " << params.android_feature_level;
        return false;
      }

      const auto& weight_tensor = *initializers.at(io_def.node_arg.Name());
      if (weight_tensor.dims()[0] != scales_dim) {
        LOGS_DEFAULT(VERBOSE) << op_type << " mismatch int8 per-channel quantization weight,"
                              << " weight dimension[0] " << weight_tensor.dims()[0]
                              << " scale dimension " << scales_dim;
        return false;
      }
    }
  }

  return true;
}

bool HasValidQuantizationZeroPoints(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                    const std::vector<size_t>& indices, bool is_input) {
  const auto& op_type = node_unit.OpType();
  auto qlinear_op_type = GetQLinearOpType(node_unit.GetNode());
  bool is_qlinear_conv = (qlinear_op_type == QLinearOpType::QLinearConv);
  bool is_qlinear_matmul = (qlinear_op_type == QLinearOpType::QLinearMatMul);

  const auto& io_defs = is_input ? node_unit.Inputs() : node_unit.Outputs();
  for (const auto idx : indices) {
    if (idx >= io_defs.size()) {
      LOGS_DEFAULT(VERBOSE) << "HasValidQuantizationZeroPoints, "
                            << (is_input ? "Input" : "Output") << " index,  " << idx
                            << " >= size, " << io_defs.size();
      return false;
    }

    const auto& io_def = io_defs[idx];
    if (!io_def.quant_param.has_value()) {
      LOGS_DEFAULT(VERBOSE) << "HasValidQuantizationZeroPoints, Input index,  " << idx
                            << " has no quant_param";
      return false;
    }

    // zero point is optional here
    if (!io_def.quant_param->zero_point)
      return true;

    const auto& zero_point_name = io_def.quant_param->zero_point->Name();
    if (!Contains(initializers, zero_point_name)) {
      LOGS_DEFAULT(VERBOSE) << "The zero point of " << op_type << " must be an initializer tensor";
      return false;
    }

    bool is_conv_matmul_weight = is_input && (is_qlinear_conv || is_qlinear_matmul) && idx == 1;
    bool is_conv_matmul_u8s8_weight = false;

    if (is_conv_matmul_weight) {
      const auto& weight_tensor = *initializers.at(io_def.node_arg.Name());
      is_conv_matmul_u8s8_weight = weight_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8;
    }

    const auto& zero_tensor = *initializers.at(zero_point_name);
    int64_t zero_dim = zero_tensor.dims().empty() ? 1 : zero_tensor.dims()[0];

    if (!is_conv_matmul_u8s8_weight) {
      if (zero_dim != 1) {
        LOGS_DEFAULT(VERBOSE) << op_type << " does not support per-channel quantization, "
                              << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
        return false;
      }
    } else {
      // For u8s8 Qlinear[Conv/MatMul], we support
      // 1. Per-tensor, the weight will be transformed to uint8 later
      // 2. Per-channel, only from Android API level 29
      if (zero_tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        LOGS_DEFAULT(VERBOSE) << "u8s8 Qlinear[Conv/MatMul] only supports int8 zero point for weight, "
                              << "actual zero point type: [" << zero_tensor.data_type() << "]";
        return false;
      }

      if (zero_dim != 1) {
        if (is_qlinear_matmul) {
          LOGS_DEFAULT(VERBOSE) << "QLinearMatMul does not support per-channel quantization";
          return false;
        }
      }

      // For onnx, u8s8 QlinearConv, the weight zero point can be a scalar,
      // or a tensor with same channel as weight, for NNAPI we only support it be
      // 0 (scalar) or all 0 (tensor), NNAPI will assume the zero point for per-channel
      // quantization is 0 there is no input for it
      const auto& weight_tensor = *initializers.at(io_def.node_arg.Name());
      if (weight_tensor.dims()[0] != zero_dim && zero_dim != 1) {
        LOGS_DEFAULT(VERBOSE) << op_type << " mismatch int8 per-channel quantization weight,"
                              << " weight dimension[0] " << weight_tensor.dims()[0]
                              << " zero point dimension " << zero_dim;
        return false;
      }

      std::vector<uint8_t> unpacked_tensor;
      auto status = onnxruntime::utils::UnpackInitializerData(zero_tensor, node_unit.ModelPath(), unpacked_tensor);
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Qlinear[Conv/MatMul] error when unpack zero tensor: " << zero_point_name
                            << ", error msg: " << status.ErrorMessage();
        return false;
      }

      // Verify all onnx weight zero point(s) are 0(s)
      const int8_t* zero_points = reinterpret_cast<const int8_t*>(unpacked_tensor.data());
      for (size_t i = 0; i < unpacked_tensor.size(); i++) {
        if (zero_points[i] != 0) {
          LOGS_DEFAULT(VERBOSE) << "u8s8 Qlinear[Conv/MatMul]  only support 0 as zero point, "
                                << "zero_points[" << i << "] has value: " << zero_points[i];
          return false;
        }
      }
    }
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

  const auto unpack_tensor = [&model_path](const InitializedTensorSet& initializers,
                                           const std::string& name, std::vector<uint8_t>& unpacked_tensor) {
    const auto& tensor = *initializers.at(name);
    ORT_RETURN_IF_ERROR(
        onnxruntime::utils::UnpackInitializerData(tensor, model_path, unpacked_tensor));
    return Status::OK();
  };

  const auto& quant_param = *io_def.quant_param;
  {  // get the scale
    std::vector<uint8_t> unpacked_tensor;
    const auto& name = quant_param.scale.Name();
    ORT_RETURN_IF_ERROR(unpack_tensor(initializers, name, unpacked_tensor));
    // The scale should be one or more floats
    ORT_RETURN_IF(unpacked_tensor.size() < 4,
                  "The initializer [", name, "] should have one or more floats ",
                  "with size no less than 4, actual size: ", unpacked_tensor.size());
    scale = reinterpret_cast<const float*>(unpacked_tensor.data())[0];
  }

  if (quant_param.zero_point) {  // get the zero point if it's there
    std::vector<uint8_t> unpacked_tensor;
    const auto& name = quant_param.zero_point->Name();
    ORT_RETURN_IF_ERROR(unpack_tensor(initializers, name, unpacked_tensor));
    ORT_RETURN_IF(unpacked_tensor.empty(), "The initializer [", name, "] is empty");
    // Onnx quantization uses uint8 [int8 not yet supported], need to cast to int32_t used by NNAPI
    zero_point = static_cast<int32_t>(unpacked_tensor[0]);
  }

  return Status::OK();
}

common::Status GetQuantizationScaleAndZeroPoint(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit, const std::string& name,
    float& scale, int32_t& zero_point, bool is_input) {
  const auto& io_defs = is_input ? node_unit.Inputs() : node_unit.Outputs();
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

void GetFlattenOutputShape(const NodeUnit& node_unit, const Shape& input_shape, int32_t& dim_1, int32_t& dim_2) {
  int32_t rank = static_cast<int>(input_shape.size());
  NodeAttrHelper helper(node_unit);
  int32_t axis = helper.Get("axis", 1);
  // axis == rank is a valid input, but invalid for HandleNegativeAxis
  // Skip non-negative axis here
  if (axis < 0)
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, rank));

  dim_1 = std::accumulate(input_shape.cbegin(), input_shape.cbegin() + axis, 1, std::multiplies<int32_t>());
  dim_2 = std::accumulate(input_shape.cbegin() + axis, input_shape.cend(), 1, std::multiplies<int32_t>());
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

bool IsInternalQuantizedNode(const Node& node) {
  // These operators can use uint8 input without specific QLinear version of it
  // However, the mode has to be internal to the graph/partition (they cannot consume graph inputs)
  static const std::unordered_set<std::string> internal_quantized_op_types =
      {
          "Transpose",
          "Resize",
          "Concat",
          "MaxPool",
      };

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
  const auto& op_support_checkers = GetOpSupportCheckers();
  if (!Contains(op_support_checkers, node_unit.OpType()))
    return false;

  const auto* op_support_checker = op_support_checkers.at(node_unit.OpType());
  return op_support_checker->IsOpSupported(graph_viewer.GetAllInitializedTensors(), node_unit, params);
}

bool IsNodeSupportedInGroup(const NodeUnit& node_unit, const GraphViewer& graph_viewer,
                            const OpSupportCheckParams& params,
                            const std::unordered_set<std::string>& node_outputs_in_group) {
  if (!IsNodeSupported(node_unit, graph_viewer, params))
    return false;

  // TODO, ignore this step if the node_unit is qdq node_unit
  // We also want to check if the node is supported as an internal quantized node
  const auto& node = node_unit.GetNode();
  if (IsInternalQuantizedNode(node))
    return IsInternalQuantizationSupported(node, node_outputs_in_group);
  else  // This is not a internal quantized node, it is supported
    return true;
}

bool IsInputSupported(const NodeArg& input, const std::string& parent_name) {
  const auto& input_name = input.Name();
  const auto* shape_proto = input.Shape();
  // We do not support input with no shape
  if (!shape_proto) {
    LOGS_DEFAULT(VERBOSE) << "Input [" << input_name << "] of [" << parent_name
                          << "] has no shape";
    return false;
  }

  for (const auto& dim : shape_proto->dim()) {
    // For now we do not support dynamic shape
    if (!dim.has_dim_value()) {
      LOGS_DEFAULT(WARNING) << "Dynamic shape is not supported for now, for input:" << input_name;
      return false;
    }
  }

  return true;
}

std::string Shape2String(const std::vector<uint32_t>& shape) {
  std::ostringstream os;
  os << "[ ";
  for (const auto& dim : shape)
    os << dim << " ";

  os << "]";
  return os.str();
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

}  // namespace nnapi
}  // namespace onnxruntime
