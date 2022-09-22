// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"

namespace onnxruntime {
namespace nnapi {

class BinaryOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
                                           const OpSupportCheckParams& params) const override;
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;
  int GetMinSupportedOpSet(const NodeUnit& node_unit) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateBinaryOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<BinaryOpSupportChecker>(
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

bool BinaryOpSupportChecker::IsNodeUnitTypeSupported(const NodeUnit& node_unit) const {
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    const auto quant_type = GetQuantizedOpType(node_unit);
    return quant_type == QuantizedOpType::QDQAdd ||
           quant_type == QuantizedOpType::QDQMul;
  }

  return true;
}

bool BinaryOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  const auto quant_type = GetQuantizedOpType(node_unit);
  return quant_type == QuantizedOpType::QLinearAdd ||
         quant_type == QuantizedOpType::QLinearMul ||
         quant_type == QuantizedOpType::QDQAdd ||
         quant_type == QuantizedOpType::QDQMul;
}

int32_t BinaryOpSupportChecker::GetMinSupportedNNAPIFeatureLevel(
    const NodeUnit& node_unit, const OpSupportCheckParams& /* params */) const {
  const auto& op(node_unit.OpType());
  if (op == "Sub" || op == "Div") {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  if (op == "Pow" || op == "PRelu") {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  return ANEURALNETWORKS_FEATURE_LEVEL_1;
}

int BinaryOpSupportChecker::GetMinSupportedOpSet(const NodeUnit& node_unit) const {
  const auto& op(node_unit.OpType());

  // Add/Sub/Mul/Div/Pow opset 6- has broadcast attributes we do not support now
  if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" || op == "Pow") {
    return 7;
  }

  return 1;
}

bool BinaryOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  bool is_quantized_op = IsQuantizedOp(node_unit);
  bool is_pow = node_unit.OpType() == "Pow";
  if (!is_quantized_op && !is_pow)
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);

  if (is_quantized_op) {
    // QLinearAdd/QDQAdd/QLinearMul/QDQMul
    if (!HasValidBinaryOpQuantizedInputTypes(node_unit))
      return false;

    if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0, 1}, params, ArgType::kInput))
      return false;

    if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
      return false;
  }

  // Pow we only support both input as fp32 now
  if (is_pow) {
    int32_t input_type_1;
    if (!GetType(node_unit.Inputs()[0].node_arg, input_type_1))
      return false;

    int32_t input_type_2;
    if (!GetType(node_unit.Inputs()[1].node_arg, input_type_2))
      return false;

    if (input_type_1 != ONNX_NAMESPACE::TensorProto_DataType_FLOAT || input_type_1 != input_type_2) {
      LOGS_DEFAULT(VERBOSE) << "Pow only supports fp32 inputs, actual input type"
                            << ", Input type 1: " << input_type_1
                            << ", Input type 2: " << input_type_2;
      return false;
    }
  }

  return true;
}

bool BinaryOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                               const OpSupportCheckParams& /* params */) const {
  const auto& op_type(node_unit.OpType());
  const auto& inputs = node_unit.Inputs();
  Shape input1_shape, input2_shape;
  if (!GetShape(inputs[0].node_arg, input1_shape) ||
      !GetShape(inputs[1].node_arg, input2_shape))
    return false;

  const auto input1_size = input1_shape.size();
  const auto input2_size = input2_shape.size();
  if (input1_size > 4 || input2_size > 4) {
    LOGS_DEFAULT(VERBOSE) << op_type << " only support up to 4d shape, input1 is "
                          << input1_size << "d shape, input 2 is "
                          << input2_size << "d shape";
    return false;
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
