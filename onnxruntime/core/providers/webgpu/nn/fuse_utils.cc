// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/framework/op_kernel_info.h"
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
namespace onnxruntime {
namespace webgpu {

namespace {

// Formats a float as a WGSL literal that round-trips back to the identical f32.
//
// std::to_string() must not be used here: it formats with "%f", i.e. exactly six digits after the
// decimal point, which silently truncates every math constant in this file. 1/sqrt(2) came out as
// 0.707107 (4 ULP off) and the Gelu-tanh cubic coefficient as 0.035677 (109 ULP off), so a fused
// Gelu no longer matched the standalone Erf/Gelu kernels it is supposed to be numerically
// identical to. max_digits10 (9 for float) is the shortest precision guaranteed to round-trip.
std::string FloatLiteral(float value) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}

// Number of activation uniform slots the given activation actually reads. The parameters map onto
// the slots in `activation_params_.values_` order, which is also the order
// AppendActivationUniformsData() supplies them in and the order GetActivationSnippet() reads them
// in:
//   Clip            -> activation_param_0 = min,   activation_param_1 = max
//   HardSigmoid     -> activation_param_0 = alpha, activation_param_1 = beta
//   LeakyRelu       -> activation_param_0 = alpha
//   Elu             -> activation_param_0 = alpha
//   ThresholdedRelu -> activation_param_0 = alpha
//   QuickGelu       -> activation_param_0 = alpha (none when alpha == 1; see HasUnitQuickGeluAlpha)
size_t GetActivationUsedUniformCount(const Activation& activation) {
  switch (activation.activation_kind_) {
    case ActivationKind::Clip:
    case ActivationKind::HardSigmoid:
      return 2;
    case ActivationKind::LeakyRelu:
    case ActivationKind::Elu:
    case ActivationKind::ThresholdedRelu:
      return 1;
    case ActivationKind::QuickGelu:
      return activation.HasUnitQuickGeluAlpha() ? 0 : 1;
    default:
      return 0;
  }
}

}  // namespace

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
    } else if (activation_type == "Erf") {
      activation.activation_kind_ = ActivationKind::Erf;
    } else if (activation_type == "Gelu" || activation_type == "FastGelu") {
      // Gelu's `approximate` attribute arrives as a 0/1 flag in activation_params: 0 selects the
      // exact erf form, 1 the tanh approximation. Contrib FastGelu is always the tanh form and
      // carries no parameters. The flag picks between two different expressions rather than
      // scaling one, so it is resolved to a kind here instead of being kept as a uniform.
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
      } else if (activation_type == "ThresholdedRelu") {
        activation.activation_kind_ = ActivationKind::ThresholdedRelu;
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
  auto base_type_cast = [&base_type](float value) -> std::string {
    return base_type + "(" + FloatLiteral(value) + ")";
  };
  auto value_type_cast = [&base_type_cast, &value_type](float f) -> std::string {
    return value_type + "(" + base_type_cast(f) + ")";
  };

  // Abramowitz & Stegun 7.1.26. Deliberately the same approximation, with the same constants at
  // the same precision, that the standalone Erf/Gelu kernels use (math/unary_elementwise_ops.h),
  // so fusing an Erf or Gelu does not change its numerics.
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
    case ActivationKind::Erf:
      return erf_fn();
    case ActivationKind::GeluTanh:
      return tanh_fn();
    default:
      return "";
  }
}

std::string GetActivationSnippet(const Activation& activation, std::string value_type, std::string base_type) {
  auto base_type_cast = [&base_type](float value) -> std::string {
    return base_type + "(" + FloatLiteral(value) + ")";
  };
  auto value_type_cast = [&base_type_cast, &value_type](float f) -> std::string {
    return value_type + "(" + base_type_cast(f) + ")";
  };
  // Activation parameters live in f32 uniforms, so cast them to the shader's element type. This is
  // what makes the half-precision and vectorized value types work: base_param() yields a scalar of
  // `base_type` and value_param() splats it across `value_type`.
  auto base_param = [&base_type](int index) -> std::string {
    return base_type + "(uniforms.activation_param_" + std::to_string(index) + ")";
  };
  auto value_param = [&base_param, &value_type](int index) -> std::string {
    return value_type + "(" + base_param(index) + ")";
  };
  switch (activation.activation_kind_) {
    case ActivationKind::Relu:
      return "value = max(value, " + value_type_cast(0.0) + ");";
    case ActivationKind::Sigmoid:
      return "value = " + value_type_cast(1.0) + " / (" + value_type_cast(1.0) + " + exp(-value));";
    case ActivationKind::Clip:
      return "value = clamp(value, " + value_param(0) + ", " + value_param(1) + ");";
    case ActivationKind::HardSigmoid:
      return "value = clamp(" + value_param(0) + " * value + " + value_param(1) + ", " + value_type_cast(0.0) + ", " +
             value_type_cast(1.0) + ");";
    case ActivationKind::LeakyRelu:
      return "value = select(" + base_param(0) + " * value, value, value >= " + value_type_cast(0.0) + ");";
    case ActivationKind::Tanh:
      return "value = tanh(value);";
    case ActivationKind::QuickGelu:
      // QuickGelu(x, alpha) = x * sigmoid(alpha * x). alpha == 1 makes this SiLU/Swish, where the
      // multiply folds away entirely. That is the case that follows nearly every Conv in the YOLO
      // family, so it gets its own shader variant rather than multiplying by a uniform 1.0.
      if (activation.HasUnitQuickGeluAlpha()) {
        return "value = value * (" + value_type_cast(1.0) + " / (" + value_type_cast(1.0) + " + exp(-value)));";
      }
      return "value = value * (" + value_type_cast(1.0) + " / (" + value_type_cast(1.0) + " + exp(-(" +
             base_param(0) + " * value))));";
    case ActivationKind::HardSwish:
      // HardSwish(x) = x * clamp(x/6 + 1/2, 0, 1). ONNX fixes the two constants.
      return "value = value * clamp(value * " + value_type_cast(1.0f / 6.0f) + " + " + value_type_cast(0.5) + ", " +
             value_type_cast(0.0) + ", " + value_type_cast(1.0) + ");";
    case ActivationKind::Elu:
      // Elu(x, alpha) = x for x >= 0, else alpha * (exp(x) - 1).
      return "value = select(" + base_param(0) + " * (exp(value) - " + value_type_cast(1.0) +
             "), value, value >= " + value_type_cast(0.0) + ");";
    case ActivationKind::ThresholdedRelu:
      // ThresholdedRelu(x, alpha) = x for x > alpha, else 0. alpha defaults to 1.
      return "value = select(" + value_type_cast(0.0) + ", value, value > " + value_param(0) + ");";
    case ActivationKind::Erf:
      // Depends on fused_act_erf.
      return "value = fused_act_erf(value);";
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

void AppendActivationUniformsData(const Activation& activation, std::vector<ProgramUniformVariableValue>& variables) {
  const size_t used = GetActivationUsedUniformCount(activation);
  for (size_t i = 0; i < kActivationUniformVariableCount; i++) {
    if (i < used) {
      variables.emplace_back(activation.activation_params_.values_[i]);
    } else {
      // Zero-length uniforms are dropped from both the uniform struct and the uniform buffer, so
      // an activation that reads no parameters produces exactly the shader it produced before
      // these slots existed.
      variables.emplace_back();
    }
  }
}

void AppendActivationUniformsData(const Activation& activation, ProgramBase& program) {
  std::vector<ProgramUniformVariableValue> variables;
  variables.reserve(kActivationUniformVariableCount);
  AppendActivationUniformsData(activation, variables);
  for (auto& variable : variables) {
    program.AddUniformVariable(std::move(variable));
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
