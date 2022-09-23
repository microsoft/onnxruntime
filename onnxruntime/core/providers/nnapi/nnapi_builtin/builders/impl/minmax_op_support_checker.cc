// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

class MinMaxOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  // Min/Max opset 5- uses consumed_inputs attribute which is not supported for now
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 6; }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

bool MinMaxOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                               const OpSupportCheckParams& /* params */) const {
  // TODO support 2+ inputs for Min/Max op
  if (node_unit.Inputs().size() != 2) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType() << "] only supports 2 inputs, "
                          << "actual input number, " << node_unit.Inputs().size();
    return false;
  }

  return true;
}

void CreateMinMaxOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<MinMaxOpSupportChecker>(
      op_type, op_registrations,
      {
          "Min",
          "Max",
      });
}

}  // namespace nnapi
}  // namespace onnxruntime
