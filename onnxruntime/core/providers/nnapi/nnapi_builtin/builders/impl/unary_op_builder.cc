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

class UnaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
                                           const OpSupportCheckParams& params) const override;

  bool HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                    const OpSupportCheckParams& params) const override;

  int GetMinSupportedOpSet(const NodeUnit& node_unit) const override;

  static bool IsQuantizedOpSupported(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                     const OpSupportCheckParams& params);
};

// Add operator related

bool UnaryOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  // TODO, add support for QDQ NodeUnit
  return node_unit.OpType() == "QLinearSigmoid";
}

void UnaryOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<UnaryOpBuilder>(
      op_type, op_registrations,
      {
          "Abs",
          "Exp",
          "Floor",
          "Log",
          "Sigmoid",
          "Neg",
          "Sin",
          "Sqrt",
          "Tanh",
          "QLinearSigmoid",
      });
}

Status UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& op_type(node_unit.OpType());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  bool is_qlinear_sigmoid = op_type == "QLinearSigmoid";

  int32_t op_code;
  if (op_type == "Abs")
    op_code = ANEURALNETWORKS_ABS;
  else if (op_type == "Exp")
    op_code = ANEURALNETWORKS_EXP;
  else if (op_type == "Floor")
    op_code = ANEURALNETWORKS_FLOOR;
  else if (op_type == "Log")
    op_code = ANEURALNETWORKS_LOG;
  else if (op_type == "Sigmoid" || is_qlinear_sigmoid)
    op_code = ANEURALNETWORKS_LOGISTIC;
  else if (op_type == "Neg")
    op_code = ANEURALNETWORKS_NEG;
  else if (op_type == "Sin")
    op_code = ANEURALNETWORKS_SIN;
  else if (op_type == "Sqrt")
    op_code = ANEURALNETWORKS_SQRT;
  else if (op_type == "Tanh")
    op_code = ANEURALNETWORKS_TANH;
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "UnaryOpBuilder, unknown op: ", op_type);
  }

  float y_scale = 0.0f;
  int32_t y_zero_point = 0;
  if (is_qlinear_sigmoid) {
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        model_builder.GetGraphViewer(), node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    // Verify if the scale and zero point values from onnx input and nnapi input match
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));

    // We already verified this in  UnaryOpBuilder::IsOpSupportedImpl
    y_scale = 1.f / 256;
    y_zero_point = 0;
  }

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

// Operator support related

bool UnaryOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                       const OpSupportCheckParams& params) const {
  if (node_unit.OpType() == "QLinearSigmoid") {
    return IsQuantizedOpSupported(graph_viewer, node_unit, params);
  } else if (node_unit.OpType() == "Sigmoid") {
    Shape input_shape;
    if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
      return false;

    const auto input_size = input_shape.size();
    if (input_size > 4 || input_size == 0) {
      LOGS_DEFAULT(VERBOSE) << "ANEURALNETWORKS_LOGISTIC only supports 1-4d shape, input is "
                            << input_size << "d shape";
      return false;
    }
    return true;
  }
  // Everything else are by default supported
  return true;
}

int32_t UnaryOpBuilder::GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
                                                         const OpSupportCheckParams& /* params */) const {
  const auto& op(node_unit.OpType());
  if (op == "Abs" ||
      op == "Exp" ||
      op == "Neg" ||
      op == "Sin" ||
      op == "Sqrt" ||
      op == "Log") {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  return ANEURALNETWORKS_FEATURE_LEVEL_1;
}

bool UnaryOpBuilder::HasSupportedInputOutputsImpl(
    const GraphViewer& graph_viewer, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  // We only need to override input check for QLinearSigmoid
  if (node_unit.OpType() != "QLinearSigmoid")
    return BaseOpBuilder::HasSupportedInputOutputsImpl(graph_viewer, node_unit, params);

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kInput))
    return false;

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

// All ops except "Sin" opset 5- uses consumed_inputs attribute which is not supported for now
// "Sin" op has support from opset 7, return 6 here for all ops
// "QLinearSigmoid" is a contrib op, OpSet will always be 1
int UnaryOpBuilder::GetMinSupportedOpSet(const NodeUnit& node_unit) const {
  if (node_unit.OpType() == "QLinearSigmoid")
    return 1;

  return 6;
}

/* static */ bool UnaryOpBuilder::IsQuantizedOpSupported(
    const GraphViewer& graph_viewer, const NodeUnit& node_unit, const OpSupportCheckParams& /* params */) {
  const auto& op_type = node_unit.OpType();
  ORT_ENFORCE(op_type == "QLinearSigmoid");

  // NNAPI requires the scale be 1.f/256 and zero point to be 0
  // See https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/android10-c2f2-release/nn/common/operations/Activation.cpp#180
  if (!HasRequiredScaleAndZeroPoint(graph_viewer,
                                    MakeString("Op [", op_type, "] name [", node_unit.Name(), "]'s output 0 "),
                                    node_unit.Outputs()[0], node_unit.ModelPath(),
                                    1.f / 256 /* required_scale */, 0 /* required_zp */)) {
    return false;
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
