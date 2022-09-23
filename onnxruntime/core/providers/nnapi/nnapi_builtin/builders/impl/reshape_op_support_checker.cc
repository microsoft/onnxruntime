// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"

namespace onnxruntime {
namespace nnapi {

class ReshapeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // Reshape opset 4- uses attributes for new shape which we do not support for now
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 5; }
  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
      const OpSupportCheckParams& /* params */) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateReshapeOpSupportChecker(
    const std::string& op_type,
    OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<ReshapeOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool ReshapeOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQReshape;
}

bool ReshapeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                                const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  const auto& perm_name = inputs[1].node_arg.Name();
  if (!Contains(initializers, perm_name)) {
    LOGS_DEFAULT(VERBOSE) << "New shape of reshape must be known";
    return false;
  }

  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Reshape only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  const auto& perm_tensor = *initializers.at(perm_name);
  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(perm_tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Error while unpacking perm_tensor: " << status.ErrorMessage();
    return false;
  }
  const int64_t* raw_perm = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  const auto perm_size = SafeInt<uint32_t>(perm_tensor.dims()[0]);

  NodeAttrHelper helper(node_unit);
  const bool allow_zero = helper.Get("allowzero", 0) == 1;
  for (uint32_t i = 0; i < perm_size; i++) {
    // NNAPI reshape does not support 0 as dimension
    if (raw_perm[i] == 0) {
      if (i < input_shape.size() && input_shape[i] == 0) {
        LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 reshape dimension on a dynamic dimension";
        return false;
      }

      if (allow_zero) {
        LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 reshape dimension when allowzero is enabled";
        return false;
      }
    }
  }

  return true;
}

bool ReshapeOpSupportChecker::HasSupportedInputOutputsImpl(
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

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
