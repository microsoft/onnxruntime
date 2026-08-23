// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/framework/op_kernel_info.h"
#include <string>
#include <utility>
#include <vector>
namespace onnxruntime {
namespace webgpu {

namespace {

// Parameters occupy uniform slots in activation_params_.values_ order.
size_t GetActivationUsedUniformCount(const Activation& activation) {
  switch (activation.activation_kind_) {
    case ActivationKind::Clip:
    case ActivationKind::HardSigmoid:
      return 2;
    case ActivationKind::LeakyRelu:
      return 1;
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
  auto base_type_cast = [base_type](const std::string& value) -> std::string {
    return base_type + "(" + value + ")";
  };
  auto value_type_cast = [base_type_cast, value_type](const std::string& value) -> std::string {
    return value_type + "(" + base_type_cast(value) + ")";
  };
  switch (activation.activation_kind_) {
    case ActivationKind::Relu:
      return "value = max(value, " + value_type_cast(std::to_string(0.0f)) + ");";
    case ActivationKind::Sigmoid:
      return "value = " + value_type_cast(std::to_string(1.0f)) + " / (" + value_type_cast(std::to_string(1.0f)) +
             " + exp(-value));";
    case ActivationKind::Clip:
      return "value = clamp(value, " + value_type_cast("uniforms.activation_param_0") + ", " +
             value_type_cast("uniforms.activation_param_1") + ");";
    case ActivationKind::HardSigmoid:
      return "value = clamp(" + value_type_cast("uniforms.activation_param_0") + " * value + " +
             value_type_cast("uniforms.activation_param_1") + ", " + value_type_cast(std::to_string(0.0f)) + ", " +
             value_type_cast(std::to_string(1.0f)) + ");";
    case ActivationKind::LeakyRelu:
      return "value = select(" + base_type_cast("uniforms.activation_param_0") + " * value, value, value >= " +
             value_type_cast(std::to_string(0.0f)) + ");";
    case ActivationKind::Tanh:
      return "value = tanh(value);";
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
