// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

class UnaryOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& params) const override;

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
      const OpSupportCheckParams& /* params */) const override;

  int GetMinSupportedOpSet(const NodeUnit& node_unit) const override;

  static bool IsQuantizedOpSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                     const OpSupportCheckParams& params);
};

void CreateUnaryOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<UnaryOpSupportChecker>(
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

bool UnaryOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                              const OpSupportCheckParams& params) const {
  if (node_unit.OpType() == "QLinearSigmoid")
    return IsQuantizedOpSupported(initializers, node_unit, params);
  else  // Everything except "QLinearSigmoid" are by default supported
    return true;
}

int32_t UnaryOpSupportChecker::GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
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

bool UnaryOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  // We only need to override input check for QLinearSigmoid
  if (node_unit.OpType() != "QLinearSigmoid")
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput))
    return false;

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

// All ops except "Sin" opset 5- uses consumed_inputs attribute which is not supported for now
// "Sin" op has support from opset 7, return 6 here for all ops
// "QLinearSigmoid" is a contrib op, OpSet will always be 1
int UnaryOpSupportChecker::GetMinSupportedOpSet(const NodeUnit& node_unit) const {
  if (node_unit.OpType() == "QLinearSigmoid")
    return 1;

  return 6;
}

/* static */ bool UnaryOpSupportChecker::IsQuantizedOpSupported(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit, const OpSupportCheckParams& /* params */) {
  const auto& op_type = node_unit.OpType();
  ORT_ENFORCE(op_type == "QLinearSigmoid");

  // NNAPI requires the scale be 1.f/256 and zero point to be 0
  // See https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/android10-c2f2-release/nn/common/operations/Activation.cpp#180
  if (!op_support_helpers::HasRequiredScaleAndZeroPoint(initializers,
                                                        MakeString("Op [", op_type, "] name [", node_unit.Name(), "]'s output 0 "),
                                                        node_unit.Outputs()[0], node_unit.ModelPath(),
                                                        1.f / 256 /* required_scale */, 0 /* required_zp */)) {
    return false;
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
