// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/shared/node_unit/node_unit.h"
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
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

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

void PoolOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  // skip input/output scales and zeropoints
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

bool PoolOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedPool(GetQuantizedOpType(node_unit));
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
  const auto& input_shape = shaper[input];
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
    const auto& initializers = model_builder.GetInitializerTensors();
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    // Verify if the scale and zero point values from onnx input and nnapi input match
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Outputs()[0], node_unit.ModelPath(), y_scale, y_zero_point));
  }

  std::vector<uint32_t> input_indices;
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

  if (model_builder.GetNNAPIFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {  // nchw only supported on api 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  ORT_RETURN_IF_ERROR(shaper.Pool(input,
                                  onnx_pads, onnx_strides, kernel_shape,
                                  use_nchw,
                                  output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

}  // namespace nnapi
}  // namespace onnxruntime
