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

class ConcatOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  bool HasSupportedInputOutputsImpl(
      const GraphViewer& graph_viewer, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

// Add operator related

void ConcatOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (IsQuantizedOp(node_unit)) {
    for (size_t i = 0; i < node_unit.Inputs().size(); ++i) {
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[i].quant_param);
    }

    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }
}

Status ConcatOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();

  InlinedVector<uint32_t> input_indices;
  const auto& input0 = inputs[0].node_arg.Name();
  const auto node_input_size = inputs.size();

  bool is_quant_op = IsQuantizedOp(node_unit);

  if (!is_quant_op) {
    // If the inputs are uint8 and this is not a quantized Concat, we need to verify all the inputs have the
    // same scale and zero points.
    // [Side note: int8 input is not supported currently by the NNAPI EP (enforced in ConcatOpSupportChecker).
    // it is supported by NNAPI though and int8 input is allowed to have different scale  and zp values.]
    //
    // ONNX allows Concat (not QlinearConcat, not QDQ concat) to run directly on uint8 without scales and zps.
    // NNAPI requires all uint8 inputs to have scale values > 0. (zero point can be 0.)
    // See https://android.googlesource.com/platform/frameworks/ml/+/master/nn/common/Validation.cpp#486
    //
    // We need to use the scales and zps from the NNAPI input directly, there is no easy way to get the input
    // scales and zps in OpSupportChecker, so we need to verify here.
    // Also we have to assume the output scale and zp are the same as input 0
    if (operand_types.at(input0).type == android::nn::wrapper::Type::TENSOR_QUANT8_ASYMM) {
      auto scale = operand_types.at(input0).operandType.scale;
      auto zero_point = operand_types.at(input0).operandType.zeroPoint;

      // TODO: if we see scale == 0 in real models we could consider using 1 as a default. This is what TF does
      // https://github.com/tensorflow/tensorflow/blob/7737c518a864e54be9b676fe063436ccbbef21b9/tensorflow/lite/delegates/nnapi/nnapi_delegate.cc#L468-L471
      ORT_RETURN_IF_NOT(scale > 0, "NNAPI requires scale to be > 0.");

      // Compare scale and zp of input0 to input1~n
      for (size_t i = 1; i < node_input_size; i++) {
        const auto& type = operand_types.at(inputs[i].node_arg.Name());
        ORT_RETURN_IF_NOT(scale == type.operandType.scale,
                          "Input[", i, "]'s scale: ", type.operandType.scale,
                          " is different than input[0]'s scale: ", scale);

        ORT_RETURN_IF_NOT(zero_point == type.operandType.zeroPoint,
                          "Input[", i, "]'s zero_point: ", type.operandType.zeroPoint,
                          " is different than input[0]'s zero_point: ", zero_point);
      }
    }
  }

  std::vector<std::string> input_names;
  input_names.reserve(node_input_size);
  for (size_t i = 0; i < node_input_size; i++) {
    const auto& input = inputs[i].node_arg.Name();

    if (is_quant_op) {
      // scale and zp values consistency was checked in ConcatOpSupportChecker
      float scale = 0.0f;
      int32_t zero_point = 0;
      ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
          model_builder.GetGraphViewer(), node_unit.Inputs()[i], node_unit.ModelPath(),
          scale, zero_point));

      ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, scale, zero_point));
    }

    input_indices.push_back(operand_indices.at(input));
    input_names.push_back(input);
  }

  // Get the output scale and zp for quantized concat, default value is from input 0
  float y_scale = operand_types.at(input0).operandType.scale;
  int32_t y_zero_point = operand_types.at(input0).operandType.zeroPoint;
  if (is_quant_op) {
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        model_builder.GetGraphViewer(), node_unit.Outputs()[0], node_unit.ModelPath(),
        y_scale, y_zero_point));
  }

  int32_t rank = static_cast<int32_t>(shaper[input0].size());
  int32_t axis = static_cast<int32_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));

  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  OperandType output_operand_type(operand_types.at(input0).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CONCATENATION, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

// Operator support related

bool ConcatOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  // TODO: add support of QLinearConcat
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQConcat;
}

bool ConcatOpBuilder::IsOpSupportedImpl(const GraphViewer& /* graph_viewer */, const NodeUnit& node_unit,
                                        const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Concat only supports up to 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

bool ConcatOpBuilder::HasSupportedInputOutputsImpl(
    const GraphViewer& graph_viewer, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  const auto& op_type = node_unit.OpType();
  const auto& op_name = node_unit.Name();
  const auto input_size = node_unit.Inputs().size();
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

  if (IsQuantizedOp(node_unit)) {
    std::vector<size_t> input_indices(input_size);
    std::iota(input_indices.begin(), input_indices.end(), 0);
    if (!IsQuantizedIOSupported(graph_viewer, node_unit, input_indices, params, ArgType::kInput)) {
      return false;
    }

    if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput)) {
      return false;
    }

    // Need to verify all the input and output has the same scale and zp for API 28-
    if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
      std::vector<float> input_scales(input_size);
      std::vector<int32_t> input_zps(input_size);
      size_t input_idx = 0;

      auto status = GetQuantizationScaleAndZeroPoint(
          graph_viewer, node_unit.Inputs()[input_idx], node_unit.ModelPath(),
          input_scales[input_idx], input_zps[input_idx]);

      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                            << "] GetQuantizationScaleAndZeroPoint for input_scale/zp failed, message: "
                            << status.ErrorMessage();
        return false;
      }

      for (++input_idx; input_idx < input_size; ++input_idx) {
        if (!HasRequiredScaleAndZeroPoint(graph_viewer,
                                          MakeString("Op [", op_type, "] name [", op_name, "] input ", input_idx),
                                          node_unit.Inputs()[input_idx],
                                          node_unit.ModelPath(),
                                          input_scales[0] /* required_scale */,
                                          input_zps[0] /* required_zp */)) {
          return false;
        }
      }

      // NNAPI (28-) requires the output scale and zp be the same as the input 0
      if (!HasRequiredScaleAndZeroPoint(graph_viewer,
                                        MakeString("Op [", op_type, "] name [", op_name, "]'s output 0"),
                                        node_unit.Outputs()[0], node_unit.ModelPath(),
                                        input_scales[0] /* required_scale */,
                                        input_zps[0] /* required_zp */)) {
        return false;
      }
    }
  }

  return true;
}

void CreateConcatOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConcatOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
