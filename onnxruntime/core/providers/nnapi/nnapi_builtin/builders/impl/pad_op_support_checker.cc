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

class PadOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;  // for ANEURALNETWORKS_PAD_V2
  }

  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override {
    // before Pad-11, inputs `pads` and `constant_value` were attributes
    // only support inputs now
    // Note: Could add support for attributes later.
    return 11;
  }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreatePadOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<PadOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool PadOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                            const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();

  // only support 1-4d input shape
  // only support input with more than 0 elements
  {
    Shape input_shape;
    if (!GetShape(inputs[0].node_arg, input_shape)) {
      return false;
    }

    if (input_shape.size() > 4 || input_shape.empty()) {
      LOGS_DEFAULT(VERBOSE) << "Pad only supports up to 1-4d shape, input is "
                            << input_shape.size() << "d shape";
      return false;
    }

    if (std::find(input_shape.begin(), input_shape.end(), uint32_t{0}) != input_shape.end()) {
      LOGS_DEFAULT(VERBOSE) << "Pad input with zero elements is not supported";
      return false;
    }
  }

  // only support "constant" mode
  // Note: Could possibly add support for "reflect" later using ANEURALNETWORKS_MIRROR_PAD.
  {
    NodeAttrHelper helper{node_unit};
    const auto mode = helper.Get("mode", "constant");
    if (mode != "constant") {
      LOGS_DEFAULT(VERBOSE) << "Mode is not supported: " << mode;
      return false;
    }
  }

  // only support if `pads` input is known and does not contain negative values
  {
    const auto pads_initializer_it = initializers.find(inputs[1].node_arg.Name());
    if (pads_initializer_it == initializers.end()) {
      LOGS_DEFAULT(VERBOSE) << "pads must be known";
      return false;
    }

    const ONNX_NAMESPACE::TensorProto& pads_initializer = *pads_initializer_it->second;
    std::vector<uint8_t> unpacked_tensor;
    auto status = onnxruntime::utils::UnpackInitializerData(pads_initializer, unpacked_tensor);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Error while unpacking pads initializer: " << status.ErrorMessage();
      return false;
    }

    int64_t pad_value;
    ORT_ENFORCE(unpacked_tensor.size() % sizeof(pad_value) == 0);
    for (size_t i = 0; i < unpacked_tensor.size(); i += sizeof(pad_value)) {
      memcpy(&pad_value, &unpacked_tensor[i], sizeof(pad_value));
      if (pad_value < 0) {
        LOGS_DEFAULT(VERBOSE) << "Negative pad value is not supported: pads["
                              << i / sizeof(pad_value) << "] = " << pad_value;
        return false;
      }
    }
  }

  // only support if `constant_value` input is known
  // Note: Could add support for non-constant initializer later. Then we need to ensure it is a scalar (with shape []).
  if (inputs.size() > 2) {
    if (!Contains(initializers, inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "constant_value must be known";
      return false;
    }
  }

  return true;
}

}  // namespace nnapi
}  // namespace onnxruntime
