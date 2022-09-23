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

class UnsqueezeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateUnsqueezeOpSupportChecker(const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<UnsqueezeOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool UnsqueezeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                                  const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  // This limitation actually comes from Reshape op.
  // We are adding ANEURALNETWORKS_RESHAPE as an equivalent operation for Unsqueeze as it's not supported by nnapi
  const auto input_rank = input_shape.size();
  if (input_rank > 4 || input_rank == 0) {
    LOGS_DEFAULT(VERBOSE) << "Unsqueeze only supports 1-4d shape, input is "
                          << input_rank << "d shape";
    return false;
  }

  // Unsqueeze opset 13 uses input 1 as axes, if we have input 1 then it needs to be an initializer
  if (node_unit.SinceVersion() > 12 && inputs.size() > 1) {
    const auto& axes_name = inputs[1].node_arg.Name();
    if (!Contains(initializers, axes_name)) {
      LOGS_DEFAULT(VERBOSE) << "Input axes of Unsqueeze must be known";
      return false;
    }
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
