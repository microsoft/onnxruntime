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

class DepthToSpaceOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateDepthToSpaceOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<DepthToSpaceOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool DepthToSpaceOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                                     const OpSupportCheckParams& params) const {
  NodeAttrHelper helper(node_unit);

  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "DepthToSpace only supports 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  if (params.use_nchw && params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    LOGS_DEFAULT(VERBOSE) << "NCHW layout is not supported for android feature level: " << params.android_feature_level;
    return false;
  }

  auto since_version = node_unit.SinceVersion();
  if (since_version >= 11) {
    // For now, only DCR mode is accepted as NNAPI only supports DCR format data rearrangement
    const auto mode = helper.Get("mode", "DCR");
    if (mode != "DCR") {
      LOGS_DEFAULT(VERBOSE) << "ANEURALNETWORKS_DEPTH_TO_SPACE only supports DCR rearrangement mode. Current mode:" << mode;
      return false;
    }
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
