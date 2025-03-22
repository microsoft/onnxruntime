// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/fuse_utils.h"
#include <string>
namespace onnxruntime {
namespace webgpu {

Status GetFusedActivationAttr(const OpKernelInfo& info, Activation& activation) {
  // Convert the activation parameters from the node into a MLAS_ACTIVATION.
  activation.activation_kind_ = ActivationKind::None;

  std::string activation_type;
  if (info.GetAttr<std::string>("activation", &activation_type).IsOK()) {
    if (activation_type == "Relu") {
      activation.activation_kind_ = ActivationKind::Relu;
    } else if (activation_type == "Tanh") {
      activation.activation_kind_ = ActivationKind::Tanh;
    } else if (activation_type == "Sigmoid") {
      activation.activation_kind_ = ActivationKind::Sigmoid;
    } else {
      // The remaining activation types have additional parameters to be pulled out.
      size_t activation_params_count;
      if (activation_type == "LeakyRelu") {
        activation.activation_kind_ = ActivationKind::LeakyRelu;
        activation_params_count = 1;
      } else if (activation_type == "Clip") {
        activation.activation_kind_ = ActivationKind::Clip;
        activation_params_count = 2;
      } else if (activation_type == "HardSigmoid") {
        activation.activation_kind_ = ActivationKind::HardSigmoid;
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
        activation.activation_params_.values_[i] = activation_params[i];
      }
    }
  }

  return Status::OK();
}

std::string GetActivationSnippet(const Activation& activation, std::string value_type) {
  std::string snippet;
  switch (activation.activation_kind) {
    case ActivationKind::Relu:
      return "value = max(value, " + value_type + "(0.0));";
    case ActivationKind::Sigmoid:
      return "value = " + value_type + "(1.0 ) / (" + value_type + "(1.0) + exp(-value));";
    case ActivationKind::Clip:
      return "value = clamp(value, " + std::to_string(activation.activation_params_.Clip.minimum_) + ", " + std::to_string(activation.activation_params_.Clip.maximum_) + ");";
    case ActivationKind::HardSigmoid:
      return "value = clamp(" + std::to_string(activation.activation_params_.HardSigmoid.alpha_) + " * value + " + std::to_string(activation.activation_params_.HardSigmoid.beta_) + ", 0.0" + ", 1.0" + ");";
    case ActivationKind::LeakyRelu:
      return "value = value > 0.0" + std::string(" ? value : ") + std::to_string(activation.activation_params_.LeakyRelu.alpha_) + " * value;";
    case ActivationKind::Tanh:
      return "value = tanh(value);";
    default:
      return "";
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
