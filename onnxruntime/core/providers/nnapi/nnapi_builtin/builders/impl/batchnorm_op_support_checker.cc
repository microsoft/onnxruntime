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

class BatchNormalizationOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 7; }
};

void CreateBatchNormalizationOpSupportChecker(
    const std::string& op_type,
    OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<BatchNormalizationOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool BatchNormalizationOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                                           const OpSupportCheckParams& /* params */) const {
  if (node_unit.Outputs().size() != 1) {
    LOGS_DEFAULT(VERBOSE) << "Your onnx model may be in training mode, please export "
                             "it in test mode.";
    return false;
  }

  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4) {
    LOGS_DEFAULT(VERBOSE) << "BN only support up to 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  NodeAttrHelper helper(node_unit);
  const auto spatial = helper.Get("spatial", 1);
  if (spatial != 1) {
    LOGS_DEFAULT(VERBOSE) << "Non-spatial BN is not supported";
    return false;
  }

  const auto& scale_name = inputs[1].node_arg.Name();
  const auto& b_name = inputs[2].node_arg.Name();
  const auto& mean_name = inputs[3].node_arg.Name();
  const auto& var_name = inputs[4].node_arg.Name();
  if (!Contains(initializers, scale_name)) {
    LOGS_DEFAULT(VERBOSE) << "Scale of BN must be known";
    return false;
  }
  if (!Contains(initializers, b_name)) {
    LOGS_DEFAULT(VERBOSE) << "B of BN must be known";
    return false;
  }
  if (!Contains(initializers, mean_name)) {
    LOGS_DEFAULT(VERBOSE) << "Mean of BN must be known";
    return false;
  }
  if (!Contains(initializers, var_name)) {
    LOGS_DEFAULT(VERBOSE) << "Var of BN must be known";
    return false;
  }

  return true;
}


}  // namespace nnapi
}  // namespace onnxruntime
