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

class SoftMaxOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }
  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }

  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateSoftMaxOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<SoftMaxOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool SoftMaxOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQSoftmax;
}

bool SoftMaxOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                                const OpSupportCheckParams& params) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 2 && input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "SoftMax only support 2d/4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    NodeAttrHelper helper(node_unit);
    int32_t axis = helper.Get("axis", 1);
    if (axis != 1) {
      LOGS_DEFAULT(VERBOSE)
          << "SoftMax only support axis 1 on Android API level: " << params.android_feature_level
          << " input axis: " << axis;
      return false;
    }
  }

  return true;
}

bool SoftMaxOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit)) {
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);
  }

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput)) {
    return false;
  }

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput)) {
    return false;
  }

  // NNAPI requires the scale be 1.f/256 and zero point to be 0
  if (!op_support_helpers::HasRequiredScaleAndZeroPoint(initializers,
                                                        MakeString("Op [", node_unit.OpType(), "] name [", node_unit.Name(), "]'s output 0 "),
                                                        node_unit.Outputs()[0], node_unit.ModelPath(),
                                                        1.f / 256 /* required_scale */, 0 /* required_zp */)) {
    return false;
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
