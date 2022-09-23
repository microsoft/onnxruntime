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

class UnaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

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
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
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
    const auto& initializers = model_builder.GetInitializerTensors();
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    // Verify if the scale and zero point values from onnx input and nnapi input match
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));

    // We already verified this in  UnaryOpSupportChecker::IsOpSupportedImpl
    y_scale = 1.f / 256;
    y_zero_point = 0;
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

}  // namespace nnapi
}  // namespace onnxruntime
