// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/fused_activation.h"

namespace onnxruntime {

common::Status GetFusedActivationAttr(const OpKernelInfo& info, MLAS_ACTIVATION& activation) {
  // Convert the activation parameters from the node into a MLAS_ACTIVATION.
  activation.ActivationKind = MlasIdentityActivation;

  std::string activation_type;
  if (info.GetAttr<std::string>("activation", &activation_type).IsOK()) {
    if (activation_type == "Relu") {
      activation.ActivationKind = MlasReluActivation;
    } else if (activation_type == "Tanh") {
      activation.ActivationKind = MlasTanhActivation;
    } else if (activation_type == "Sigmoid") {
      activation.ActivationKind = MlasLogisticActivation;
    } else {
      // The remaining activation types have additional parameters to be pulled out.
      size_t activation_params_count;
      if (activation_type == "LeakyRelu") {
        activation.ActivationKind = MlasLeakyReluActivation;
        activation_params_count = 1;
      } else if (activation_type == "Clip") {
        activation.ActivationKind = MlasClipActivation;
        activation_params_count = 2;
      } else if (activation_type == "HardSigmoid") {
        activation.ActivationKind = MlasHardSigmoidActivation;
        activation_params_count = 2;
      } else {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "unimplemented activation: " + activation_type);
      }

      std::vector<float> activation_params;
      common::Status status = info.GetAttrs<float>("activation_params", activation_params);
      if (!status.IsOK()) {
        return status;
      } else if (activation_params_count != activation_params.size()) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "activation_params count mismatch");
      }
      for (size_t i = 0; i < activation_params_count; i++) {
        activation.Parameters.Values[i] = activation_params[i];
      }
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
