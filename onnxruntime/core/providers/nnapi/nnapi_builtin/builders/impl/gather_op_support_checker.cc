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

class GatherOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateGatherOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<GatherOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool GatherOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                               const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;

  if (!GetShape(inputs[0].node_arg, input_shape)) {
    return false;
  }

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Gather only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  if (std::any_of(input_shape.cbegin(), input_shape.cend(), [](int32_t i) { return i == 0; })) {
    LOGS_DEFAULT(VERBOSE) << "Gather doesn't support dynamic input shape";
    return false;
  }

  // Here in GatherOpSupportChecker::IsOpSupportedImpl, we removed the restriction that 2nd input "indices" must be an initializer
  // to accommodate the support for some models such as mobileBERT. It doesn't need to be an initializer for int32 as NNAPI Gather
  // uses int32 for indices so the type matches.
  // However, we still require indices of other types to be an initializer as we convert the data to int32 during model building.
  // TODO: We could potentially support non-initializer inputs for the other types if we inserted a cast.
  const auto& indices_name = inputs[1].node_arg.Name();

  int32_t indices_type;
  if (!GetType(node_unit.Inputs()[1].node_arg, indices_type))
    return false;

  if (indices_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    if (!Contains(initializers, indices_name)) {
      LOGS_DEFAULT(VERBOSE) << "Indices of Gather must be known.";
      return false;
    }
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
