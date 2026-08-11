// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/framework/op_kernel_info.h"
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
    } else if (activation_type == "HardSwish") {
      // ONNX HardSwish fixes alpha/beta at 1/6 and 1/2, so it carries no parameters.
      activation.activation_kind_ = ActivationKind::HardSwish;
    } else if (activation_type == "Softplus") {
      activation.activation_kind_ = ActivationKind::Softplus;
    } else if (activation_type == "Gelu" || activation_type == "FastGelu") {
      // Gelu's `approximate` attribute arrives as a 0/1 flag in activation_params: 0 selects the
      // exact erf form, 1 the tanh approximation. Contrib FastGelu is always the tanh form and
      // carries no parameters. Resolving the choice to a kind here, rather than keeping it as a
      // runtime parameter, leaves both forms parameterless for IsActivationSupported.
      bool tanh_approximation = activation_type == "FastGelu";
      if (!tanh_approximation) {
        std::vector<float> approximate_param;
        if (info.GetAttrs<float>("activation_params", approximate_param).IsOK() && !approximate_param.empty()) {
          tanh_approximation = approximate_param[0] != 0.0f;
        }
      }
      activation.activation_kind_ = tanh_approximation ? ActivationKind::GeluTanh : ActivationKind::Gelu;
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
      } else if (activation_type == "QuickGelu") {
        activation.activation_kind_ = ActivationKind::QuickGelu;
        activation_params_count = 1;
      } else if (activation_type == "Elu") {
        activation.activation_kind_ = ActivationKind::Elu;
        activation_params_count = 1;
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

std::string GetActivationDeclaration(const Activation& activation, std::string value_type, std::string base_type) {
  auto base_type_cast = [base_type](float value) -> std::string {
    return base_type + "(" + std::to_string(value) + ")";
  };
  auto value_type_cast = [base_type_cast, value_type](float f) -> std::string {
    return value_type + "(" + base_type_cast(f) + ")";
  };

  // Abramowitz & Stegun 7.1.26. Deliberately the same approximation the standalone Erf/Gelu
  // kernels use (math/unary_elementwise_ops.h), so fusing a Gelu does not change its numerics.
  auto erf_fn = [&]() -> std::string {
    return "fn fused_act_erf(v: " + value_type + ") -> " + value_type + " {\n" +
           "  let a = abs(v);\n" +
           "  let t = " + value_type_cast(1.0f) + " / (" + value_type_cast(1.0f) + " + " +
           value_type_cast(0.3275911f) + " * a);\n" +
           "  return sign(v) * (" + value_type_cast(1.0f) + " - ((((" + value_type_cast(1.061405429f) + " * t + " +
           value_type_cast(-1.453152027f) + ") * t + " + value_type_cast(1.421413741f) + ") * t + " +
           value_type_cast(-0.284496736f) + ") * t + " + value_type_cast(0.254829592f) + ") * t * exp(-a * a));\n" +
           "}\n";
  };

  // WGSL's built-in tanh() returns NaN once the input exceeds ~88.7 (f32) or ~11.09 (f16); see
  // https://github.com/gpuweb/gpuweb/issues/4458. The Gelu tanh approximation cubes its input, so
  // it reaches that range easily. This form is finite for every input.
  auto tanh_fn = [&]() -> std::string {
    return "fn fused_act_tanh(v: " + value_type + ") -> " + value_type + " {\n" +
           "  let e = exp(" + value_type_cast(-2.0f) + " * abs(v));\n" +
           "  return sign(v) * ((" + value_type_cast(1.0f) + " - e) / (" + value_type_cast(1.0f) + " + e));\n" +
           "}\n";
  };

  switch (activation.activation_kind_) {
    case ActivationKind::Gelu:
      return erf_fn();
    case ActivationKind::GeluTanh:
      return tanh_fn();
    default:
      return "";
  }
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
    case ActivationKind::QuickGelu:
      // QuickGelu(x, alpha) = x * sigmoid(alpha * x). alpha == 1 makes this SiLU/Swish.
      return "value = value * (" + value_type_cast(1.0) + " / (" + value_type_cast(1.0) + " + exp(-(" +
             base_type_cast(activation.activation_params_.QuickGelu.alpha_) + " * value))));";
    case ActivationKind::HardSwish:
      // HardSwish(x) = x * clamp(x/6 + 1/2, 0, 1). ONNX fixes the two constants.
      return "value = value * clamp(value * " + value_type_cast(1.0f / 6.0f) + " + " + value_type_cast(0.5) + ", " +
             value_type_cast(0.0) + ", " + value_type_cast(1.0) + ");";
    case ActivationKind::Elu:
      // Elu(x, alpha) = x for x >= 0, else alpha * (exp(x) - 1).
      return "value = select(" + base_type_cast(activation.activation_params_.Elu.alpha_) + " * (exp(value) - " +
             value_type_cast(1.0) + "), value, value >= " + value_type_cast(0.0) + ");";
    case ActivationKind::Gelu:
      // Gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))). Depends on fused_act_erf.
      return "value = " + value_type_cast(0.5) + " * value * (" + value_type_cast(1.0) + " + fused_act_erf(value * " +
             value_type_cast(0.70710678118654752f) + "));";
    case ActivationKind::GeluTanh:
      // Gelu(approximate="tanh") and contrib FastGelu:
      // x * (0.5 + 0.5 * tanh(x * (0.044715 * sqrt(2/pi) * x^2 + sqrt(2/pi)))).
      // Depends on fused_act_tanh.
      return "value = value * (" + value_type_cast(0.5) + " + " + value_type_cast(0.5) + " * fused_act_tanh(value * (" +
             value_type_cast(0.035677408136300125f) + " * value * value + " +
             value_type_cast(0.79788456080286535f) + ")));";
    case ActivationKind::Softplus:
      // softplus(x) = log(1 + exp(x)), written as max(x, 0) + log(1 + exp(-|x|)) so that the
      // exp() cannot overflow. The direct form overflows above x ~= 11 in f16.
      return "value = max(value, " + value_type_cast(0.0) + ") + log(" + value_type_cast(1.0) + " + exp(-abs(value)));";
    default:
      return "";
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
