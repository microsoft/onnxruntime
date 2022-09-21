// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"

namespace onnxruntime {
namespace nnapi {

class ConvOpSupportChecker : public BaseOpSupportChecker {
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& params) const override {
    return params.use_nchw ? ANEURALNETWORKS_FEATURE_LEVEL_3 : ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
      const OpSupportCheckParams& /* params */) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

/* static */ void ConvOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<ConvOpSupportChecker>(
      op_type, op_registrations,
      {
          "Conv",
          "QLinearConv",
      });
}

bool ConvOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedConv(GetQuantizedOpType(node_unit));
}

bool ConvOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit))
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);

  // QLinearConv only supports input of uint8 for now
  if (!HasValidBinaryOpQuantizedInputTypes(node_unit))
    return false;

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0, 1}, params, ArgType::kInput))
    return false;

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

bool ConvOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                             const OpSupportCheckParams& params) const {
  const auto& op_type = node_unit.OpType();
  bool is_quant_conv = IsQuantizedOp(node_unit);

  // We don't support nhwc com.microsoft.QLinearConv for now
  if (is_quant_conv && node_unit.Domain() == kMSDomain) {
    LOGS_DEFAULT(VERBOSE) << "com.microsoft.QLinearConv is not supported";
    return false;
  }

  const auto& inputs = node_unit.Inputs();
  NodeAttrHelper helper(node_unit);
  const auto group = helper.Get("group", 1);
  const auto weight_name = inputs[1].node_arg.Name();
  if (Contains(initializers, weight_name)) {
    const auto& tensor = *initializers.at(weight_name);
    if (tensor.dims().size() != 4) {
      LOGS_DEFAULT(VERBOSE) << "Only conv 2d is supported.";
      return false;
    }

    const auto onnx_dilations = helper.Get("dilations", std::vector<int>{1, 1});
    if (onnx_dilations != std::vector<int>{1, 1}) {
      if (group != 1 && tensor.dims()[1] != 1) {
        LOGS_DEFAULT(VERBOSE) << "dilation is not supported on grouped conv";
        return false;
      }

      if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
        LOGS_DEFAULT(VERBOSE) << op_type << " dilations is only supported on Android API level 29+, "
                              << "actual API level: " << params.android_feature_level;
        return false;
      }
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "The weight of convolution must be known";
    return false;
  }

  if (is_quant_conv) {
    if (inputs.size() > 2 && !Contains(initializers, inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Bias of QLinearConv must be known";
      return false;
    }
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
