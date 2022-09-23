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

class BinaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

bool BinaryOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  const auto quant_type = GetQuantizedOpType(node_unit);
  return quant_type == QuantizedOpType::QLinearAdd ||
         quant_type == QuantizedOpType::QLinearMul ||
         quant_type == QuantizedOpType::QDQAdd ||
         quant_type == QuantizedOpType::QDQMul;
}

void BinaryOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  const auto& inputs = node_unit.Inputs();
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[1].quant_param);               // b_scale, b_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& op_type(node_unit.OpType());
  const auto& inputs = node_unit.Inputs();

  int32_t op_code;
  bool add_activation = true;
  bool is_quant_op = IsQuantizedOp(node_unit);
  if (op_type == "Add" || op_type == "QLinearAdd") {  // Add/QLinearAdd/QDQAdd
    op_code = ANEURALNETWORKS_ADD;
  } else if (op_type == "Sub") {
    op_code = ANEURALNETWORKS_SUB;
  } else if (op_type == "Mul" || op_type == "QLinearMul") {  // Mul/QLinearMul/QDQMul
    op_code = ANEURALNETWORKS_MUL;
  } else if (op_type == "Div") {
    op_code = ANEURALNETWORKS_DIV;
  } else if (op_type == "Pow") {
    add_activation = false;  // ANEURALNETWORKS_POW does not have activation
    op_code = ANEURALNETWORKS_POW;
  } else if (op_type == "PRelu") {
    add_activation = false;  // ANEURALNETWORKS_PRELU does not have activation
    op_code = ANEURALNETWORKS_PRELU;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "UnaryOpBuilder, unknown op: ", op_type);
  }

  std::string input1 = inputs[0].node_arg.Name();
  std::string input2 = inputs[1].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  float a_scale = 0.0f,
        b_scale = 0.0f,
        y_scale = 0.0f;
  int32_t a_zero_point = 0,
          b_zero_point = 0,
          y_zero_point = 0;

  if (is_quant_op) {
    ORT_RETURN_IF_ERROR(GetBinaryOpQuantizationScaleAndZeroPoint(
        model_builder.GetInitializerTensors(), node_unit,
        a_scale, b_scale, y_scale,
        a_zero_point, b_zero_point, y_zero_point));
  }

  // Verify if the scale and zero point matchs from onnx input and nnapi input match
  if (is_quant_op) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
  if (add_activation) {
    fuse_code = model_builder.FindActivation(node_unit);
  }

  return AddBinaryOperator(op_code, model_builder,
                           input1, input2,
                           add_activation, fuse_code,
                           output, y_scale, y_zero_point);
}

void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<BinaryOpBuilder>(
      op_type, op_registrations,
      {
          "Add",
          "Sub",
          "Mul",
          "Div",
          "QLinearAdd",
          "QLinearMul",
          "Pow",
          "PRelu",
      });
}

}  // namespace nnapi
}  // namespace onnxruntime
