// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

class PoolOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& params) const override {
    return params.use_nchw ? ANEURALNETWORKS_FEATURE_LEVEL_3 : ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                    const OpSupportCheckParams& params) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void PoolOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  // skip input/output scales and zeropoints
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

Status PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  NodeAttrHelper helper(node_unit);

  auto input = node_unit.Inputs()[0].node_arg.Name();
  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  const auto& op_type = node_unit.OpType();

  int32_t op_code;
  bool is_quant_pool = IsQuantizedOp(node_unit);
  bool is_average_pool = op_type == "AveragePool" || op_type == "QLinearAveragePool";
  if (is_average_pool || op_type == "GlobalAveragePool")
    op_code = ANEURALNETWORKS_AVERAGE_POOL_2D;
  else  // (op_type == "MaxPool" || op_type == "GlobalMaxPool")
    op_code = ANEURALNETWORKS_MAX_POOL_2D;

  std::vector<int32_t> onnx_pads, onnx_strides, kernel_shape;
  bool use_auto_pad = false;
  int32_t nnapi_padding_code = ANEURALNETWORKS_PADDING_VALID;
  const auto input_shape = shaper[input];
  if (is_average_pool || op_type == "MaxPool") {
    const auto auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
    kernel_shape = helper.Get("kernel_shape", std::vector<int32_t>{0, 0});
    onnx_strides = helper.Get("strides", std::vector<int>{1, 1});
    onnx_pads = helper.Get("pads", std::vector<int>{0, 0, 0, 0});
    const auto weight_size_y = static_cast<uint32_t>(kernel_shape[0]);
    const auto weight_size_x = static_cast<uint32_t>(kernel_shape[1]);
    ORT_RETURN_IF_ERROR(
        HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                      onnx_strides, {1, 1} /* onnx_dilations */,
                      auto_pad_type, use_nchw,
                      onnx_pads, nnapi_padding_code, use_auto_pad));
  } else {  // (op_type == "GlobalAveragePool" || op_type == "GlobalMaxPool")
    use_auto_pad = true;
    nnapi_padding_code = ANEURALNETWORKS_PADDING_VALID;
    onnx_strides = std::vector<int32_t>{1, 1};
    onnx_pads = std::vector<int32_t>{0, 0, 0, 0};
    if (use_nchw) {
      kernel_shape = std::vector<int32_t>{static_cast<int32_t>(input_shape[2]),
                                          static_cast<int32_t>(input_shape[3])};
    } else {
      kernel_shape = std::vector<int32_t>{static_cast<int32_t>(input_shape[1]),
                                          static_cast<int32_t>(input_shape[2])};
    }
  }

  int32_t fuse_code = model_builder.FindActivation(node_unit);

  // Get output scale and zero point if this is QLinearAveragePool
  // Otherwise we will use the scale and zero point of the input
  const OperandType& input_operand_type = operand_types.at(input);
  float y_scale = input_operand_type.operandType.scale;
  int32_t y_zero_point = input_operand_type.operandType.zeroPoint;
  if (is_quant_pool) {
    const auto& graph_viewer = model_builder.GetGraphViewer();
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        graph_viewer, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    // Verify if the scale and zero point values from onnx input and nnapi input match
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        graph_viewer, node_unit.Outputs()[0], node_unit.ModelPath(), y_scale, y_zero_point));
  }

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  if (use_auto_pad) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, nnapi_padding_code);
  } else {
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[1]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[3]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[0]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[2]);
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  if (model_builder.GetEffectiveFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {  // nchw only supported on api 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

// Operator support related

bool PoolOpBuilder::IsNodeUnitTypeSupported(const NodeUnit& node_unit) const {
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    const auto quant_type = GetQuantizedOpType(node_unit);
    return quant_type == QuantizedOpType::QDQAveragePool;
  }

  return true;
}

bool PoolOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedPool(GetQuantizedOpType(node_unit));
}

bool PoolOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                      const OpSupportCheckParams& /* params */) const {
  const auto& op_name = node_unit.Name();
  const auto& op_type = node_unit.OpType();
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE)
        << op_type << " only supports rank-4 tensor, input ["
        << inputs[0].node_arg.Name() << "] has actual dim count " << input_size;
    return false;
  }

  bool is_quant_pool = IsQuantizedOp(node_unit);
  bool is_average_pool = op_type == "AveragePool" || op_type == "QLinearAveragePool";
  if (is_average_pool || op_type == "MaxPool") {
    NodeAttrHelper helper(node_unit);

    const auto count_include_pad = helper.Get("count_include_pad", 0);
    if (count_include_pad == 1) {
      LOGS_DEFAULT(VERBOSE) << "count_include_pad == 1 is not supported";
      return false;
    }

    const auto storage_order = helper.Get("storage_order", 0);
    if (storage_order == 1) {
      LOGS_DEFAULT(VERBOSE) << "storage_order == 1 is not supported";
      return false;
    }

    if (helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "Only pooling 2d is supported";
      return false;
    }

    if (helper.Get("ceil_mode", 0) == 1) {
      LOGS_DEFAULT(VERBOSE) << "ceil_mode == 1 is not supported for pooling";
      return false;
    }

    if (helper.Get("dilations", std::vector<int32_t>{1, 1}) !=
        std::vector<int32_t>{1, 1}) {
      LOGS_DEFAULT(VERBOSE) << "Dilations of pooling is not supported";
      return false;
    }

    if (node_unit.Outputs().size() != 1) {
      LOGS_DEFAULT(VERBOSE) << "Argmax in maxpooling is not supported";
      return false;
    }
  } else if (op_type != "GlobalAveragePool" && op_type != "GlobalMaxPool") {
    LOGS_DEFAULT(VERBOSE) << "PoolOpBuilder, unknown op: " << op_type;
    return false;
  }

  // We need to check if we have valid scales and zero points for QLinearAveragePool
  if (is_average_pool && is_quant_pool) {
    // NNAPI requires Quantized Average Pool has same scale and zero point for both input and output
    float input_scale = 0.0f;
    int32_t input_zp = 0;
    auto status = GetQuantizationScaleAndZeroPoint(
        graph_viewer, node_unit.Inputs()[0], node_unit.ModelPath(), input_scale, input_zp);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                          << "] GetQuantizationScaleAndZeroPoint for input_scale/zp failed, message: "
                          << status.ErrorMessage();
      return false;
    }

    float output_scale = 0.0f;
    int32_t output_zp = 0;
    status = GetQuantizationScaleAndZeroPoint(
        graph_viewer, node_unit.Outputs()[0], node_unit.ModelPath(), output_scale, output_zp);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                          << "] GetQuantizationScaleAndZeroPoint for output_scale/zp failed, message: "
                          << status.ErrorMessage();
      return false;
    }

    if (input_scale != output_scale) {
      LOGS_DEFAULT(VERBOSE) << "Op [" << op_type << "] name [" << op_name
                            << "] has different input_scale: " << input_scale
                            << " than the output_scale: " << output_scale;
      return false;
    }

    if (input_zp != output_zp) {
      LOGS_DEFAULT(VERBOSE) << "Op [" << op_type << "] name [" << op_name
                            << "] has different input_zp: " << input_zp
                            << " than the output_zp: " << output_zp;
      return false;
    }
  }

  return true;
}

bool PoolOpBuilder::HasSupportedInputOutputsImpl(
    const GraphViewer& graph_viewer, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  const auto& op_type = node_unit.OpType();
  bool is_quant_pool = IsQuantizedOp(node_unit);
  bool is_max_pool = op_type == "MaxPool";
  bool is_average_pool = op_type == "AveragePool" || op_type == "QLinearAveragePool";
  bool is_quant_average_pool = is_quant_pool && is_average_pool;
  if (!is_max_pool && !is_quant_average_pool)
    return BaseOpBuilder::HasSupportedInputOutputsImpl(graph_viewer, node_unit, params);

  if (is_quant_average_pool) {
    if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kInput))
      return false;

    if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput))
      return false;
  }

  // is_max_pool
  // For max pool, we can support both float and uint8 input
  int32_t input_type;
  if (!GetType(node_unit.Inputs()[0].node_arg, input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<PoolOpBuilder>(
      op_type, op_registrations,
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
          "AveragePool",
          "MaxPool",
          "QLinearAveragePool",
      });
}

}  // namespace nnapi
}  // namespace onnxruntime
