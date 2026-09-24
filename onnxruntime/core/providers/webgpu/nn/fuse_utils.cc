// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/fuse_utils.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/op_kernel_info.h"
#include "core/providers/webgpu/string_macros.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Holds the longest shortest-round-trip f32 rendering, e.g. "-1.23456792e-38".
constexpr size_t kFloatLiteralReserveSize = 32;

// OStringStream formats with std::to_chars, which ignores the global locale, so a comma decimal
// separator cannot corrupt the emitted shader source. It emits the fewest digits that round-trip
// each f32 exactly.
std::string FloatLiteral(float value) {
  SS(oss, kFloatLiteralReserveSize);
  oss << value;
  return SS_GET(oss);
}

// Parameters occupy uniform slots in activation_params_.values_ order.
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
      // ONNX fixes HardSwish alpha and beta.
      activation.activation_kind_ = ActivationKind::HardSwish;
    } else if (activation_type == "Softplus") {
      activation.activation_kind_ = ActivationKind::Softplus;
    } else if (activation_type == "Erf") {
      activation.activation_kind_ = ActivationKind::Erf;
    } else if (activation_type == "Gelu" || activation_type == "FastGelu") {
      // activation_params[0] selects erf (0) or tanh (1); FastGelu always uses tanh.
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
  auto base_type_cast = [base_type](const std::string& value) -> std::string {
    return base_type + "(" + value + ")";
  };
  auto value_type_cast = [base_type_cast, value_type](const std::string& value) -> std::string {
    return value_type + "(" + base_type_cast(value) + ")";
  };

  // Match the standalone Erf/Gelu implementation (Abramowitz & Stegun 7.1.26).
  auto erf_fn = [&]() -> std::string {
    return "fn fused_act_erf(v: " + value_type + ") -> " + value_type + " {\n" +
           "  let a = abs(v);\n" +
           "  let t = " + value_type_cast(FloatLiteral(1.0f)) + " / (" + value_type_cast(FloatLiteral(1.0f)) + " + " +
           value_type_cast(FloatLiteral(0.3275911f)) + " * a);\n" +
           "  return sign(v) * (" + value_type_cast(FloatLiteral(1.0f)) + " - ((((" +
           value_type_cast(FloatLiteral(1.061405429f)) + " * t + " + value_type_cast(FloatLiteral(-1.453152027f)) +
           ") * t + " + value_type_cast(FloatLiteral(1.421413741f)) + ") * t + " +
           value_type_cast(FloatLiteral(-0.284496736f)) + ") * t + " + value_type_cast(FloatLiteral(0.254829592f)) +
           ") * t * exp(-a * a));\n" +
           "}\n";
  };

  // WGSL tanh returns NaN for large inputs; this equivalent form stays finite.
  auto tanh_fn = [&]() -> std::string {
    return "fn fused_act_tanh(v: " + value_type + ") -> " + value_type + " {\n" +
           "  let e = exp(" + value_type_cast(FloatLiteral(-2.0f)) + " * abs(v));\n" +
           "  return sign(v) * ((" + value_type_cast(FloatLiteral(1.0f)) + " - e) / (" +
           value_type_cast(FloatLiteral(1.0f)) + " + e));\n" +
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
  auto base_type_cast = [base_type](const std::string& value) -> std::string {
    return base_type + "(" + value + ")";
  };
  auto value_type_cast = [base_type_cast, value_type](const std::string& value) -> std::string {
    return value_type + "(" + base_type_cast(value) + ")";
  };
  switch (activation.activation_kind_) {
    case ActivationKind::Relu:
      return "value = max(value, " + value_type_cast(FloatLiteral(0.0f)) + ");";
    case ActivationKind::Sigmoid:
      return "value = " + value_type_cast(FloatLiteral(1.0f)) + " / (" + value_type_cast(FloatLiteral(1.0f)) +
             " + exp(-value));";
    case ActivationKind::Clip:
      return "value = clamp(value, " + value_type_cast("uniforms.activation_param_0") + ", " +
             value_type_cast("uniforms.activation_param_1") + ");";
    case ActivationKind::HardSigmoid:
      return "value = clamp(" + value_type_cast("uniforms.activation_param_0") + " * value + " +
             value_type_cast("uniforms.activation_param_1") + ", " + value_type_cast(FloatLiteral(0.0f)) + ", " +
             value_type_cast(FloatLiteral(1.0f)) + ");";
    case ActivationKind::LeakyRelu:
      return "value = select(" + base_type_cast("uniforms.activation_param_0") + " * value, value, value >= " +
             value_type_cast(FloatLiteral(0.0f)) + ");";
    case ActivationKind::Tanh:
      return "value = tanh(value);";
    case ActivationKind::QuickGelu:
      // Alpha 1 omits the multiply and uniform read.
      if (activation.HasUnitQuickGeluAlpha()) {
        return "value = value * (" + value_type_cast(FloatLiteral(1.0f)) + " / (" +
               value_type_cast(FloatLiteral(1.0f)) + " + exp(-value)));";
      }
      return "value = value * (" + value_type_cast(FloatLiteral(1.0f)) + " / (" + value_type_cast(FloatLiteral(1.0f)) +
             " + exp(-(" + base_type_cast("uniforms.activation_param_0") + " * value))));";
    case ActivationKind::HardSwish:
      return "value = value * clamp(value * " + value_type_cast(FloatLiteral(1.0f / 6.0f)) + " + " +
             value_type_cast(FloatLiteral(0.5f)) + ", " + value_type_cast(FloatLiteral(0.0f)) + ", " +
             value_type_cast(FloatLiteral(1.0f)) + ");";
    case ActivationKind::Elu:
      return "value = select(" + base_type_cast("uniforms.activation_param_0") + " * (exp(value) - " +
             value_type_cast(FloatLiteral(1.0f)) + "), value, value >= " + value_type_cast(FloatLiteral(0.0f)) + ");";
    case ActivationKind::ThresholdedRelu:
      return "value = select(" + value_type_cast(FloatLiteral(0.0f)) + ", value, value > " +
             value_type_cast("uniforms.activation_param_0") + ");";
    case ActivationKind::Erf:
      return "value = fused_act_erf(value);";
    case ActivationKind::Gelu:
      return "value = " + value_type_cast(FloatLiteral(0.5f)) + " * value * (" + value_type_cast(FloatLiteral(1.0f)) +
             " + fused_act_erf(value * " + value_type_cast(FloatLiteral(0.70710678118654752f)) + "));";
    case ActivationKind::GeluTanh:
      return "value = value * (" + value_type_cast(FloatLiteral(0.5f)) + " + " + value_type_cast(FloatLiteral(0.5f)) +
             " * fused_act_tanh(value * (" + value_type_cast(FloatLiteral(0.035677408136300125f)) +
             " * value * value + " + value_type_cast(FloatLiteral(0.79788456080286535f)) + ")));";
    case ActivationKind::Softplus:
      // Use the overflow-safe softplus form.
      return "value = max(value, " + value_type_cast(FloatLiteral(0.0f)) + ") + log(" +
             value_type_cast(FloatLiteral(1.0f)) + " + exp(-abs(value)));";
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
      // Empty values omit unused slots from both the uniform struct and buffer.
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
