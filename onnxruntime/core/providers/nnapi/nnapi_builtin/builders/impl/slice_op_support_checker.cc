// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

class SliceOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  // We only support slice from opset 10
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 10; }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateSliceOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<SliceOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool SliceOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                              const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4) {
    LOGS_DEFAULT(VERBOSE) << "Slice only supports 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  // TODO, replace with std::find when we switch to c++17
  if (std::any_of(input_shape.cbegin(), input_shape.cend(), [](int32_t i) { return i == 0; })) {
    LOGS_DEFAULT(VERBOSE) << "Slice doesn't support dynamic input shape";
    return false;
  }

  if (!CheckIsInitializer(initializers, node_unit, node_unit.Inputs()[1].node_arg.Name(), "starts")) {
    return false;
  }
  if (!CheckIsInitializer(initializers, node_unit, node_unit.Inputs()[2].node_arg.Name(), "ends")) {
    return false;
  }
  const auto& inputs = node_unit.Inputs();
  if (inputs.size() > 3) {
    if (!CheckIsInitializer(initializers, node_unit, node_unit.Inputs()[3].node_arg.Name(), "axes")) {
      return false;
    }
    if (inputs.size() > 4) {
      if (!CheckIsInitializer(initializers, node_unit, node_unit.Inputs()[4].node_arg.Name(), "steps")) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
