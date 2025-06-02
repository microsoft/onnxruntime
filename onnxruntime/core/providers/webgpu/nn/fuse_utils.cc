// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/fuse_utils.h"
#include <string>
namespace onnxruntime {
namespace webgpu {

Status GetFusedActivationAttr(const OpKernelInfo& info, Activation& activation) {
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

std::string GetActivationSnippet(const Activation& activation, std::string value_type, std::string base_type) {
  std::string snippet;
  auto base_type_cast = [base_type](float value) -> std::string {
    return base_type + "(" + std::to_string(value) + ")";
  };
  auto value_type_cast = [base_type_cast, value_type](float f) -> std::string {
    return value_type + "(" + base_type_cast(f) + ")";
  };
  switch (activation.activation_kind_) {
    case ActivationKind::Relu:
      return "value = max(value, " + value_type_cast(0.0) + ");";
    case ActivationKind::Sigmoid:
      return "value = " + value_type_cast(1.0) + " / (" + value_type_cast(1.0) + " + exp(-value));";
    case ActivationKind::Clip:
      return "value = clamp(value, " + value_type_cast(activation.activation_params_.Clip.minimum_) + ", " +
             value_type_cast(activation.activation_params_.Clip.maximum_) + ");";
    case ActivationKind::HardSigmoid:
      return "value = clamp(" + value_type_cast(activation.activation_params_.HardSigmoid.alpha_) + " * value + " +
             value_type_cast(activation.activation_params_.HardSigmoid.beta_) + ", " + value_type_cast(0.0) + ", " +
             value_type_cast(1.0) + ");";
    case ActivationKind::LeakyRelu:
      return "value = select(" + base_type_cast(activation.activation_params_.LeakyRelu.alpha_) +
             " * value, value, value >= " + value_type_cast(0.0) + ");";
    case ActivationKind::Tanh:
      return "value = tanh(value);";
    default:
      return "";
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
