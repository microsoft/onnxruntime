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
    } else if (activation_type == "LeakyRelu") {
      activation.ActivationKind = MlasLeakyReluActivation;
      activation.alpha = info.GetAttrOrDefault<float>("alpha", 0.01f);
    } else if (activation_type == "Tanh") {
      activation.ActivationKind = MlasTanhActivation;
    } else if (activation_type == "Sigmoid") {
      activation.ActivationKind = MlasLogisticActivation;
    } else {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unimplemented activation: " + activation_type);
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
