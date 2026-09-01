// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <string>
#include <sstream>
#include <vector>

#include "core/common/status.h"
#include "core/providers/webgpu/program.h"

#pragma once
namespace onnxruntime {

class OpKernelInfo;

namespace webgpu {

enum class ActivationKind {
  None,
  Relu,
  Sigmoid,
  Clip,
  HardSigmoid,
  LeakyRelu,
  Tanh,
  QuickGelu,
  HardSwish,
  Elu,
  Gelu,
  GeluTanh,
  Softplus,
  ThresholdedRelu,
  Erf
};

using Activation = struct Activation {
  std::string CacheKey() const {
    std::stringstream oss;
    oss << "ActivationKind: " << static_cast<int>(activation_kind_) << ";";
    if (activation_kind_ == ActivationKind::QuickGelu) {
      oss << "QuickGeluUnitAlpha: " << (HasUnitQuickGeluAlpha() ? 1 : 0) << ";";
    }
    return oss.str();
  }
  // Alpha 1 selects a shader variant without the multiply or alpha uniform.
  bool HasUnitQuickGeluAlpha() const {
    return activation_kind_ == ActivationKind::QuickGelu && activation_params_.QuickGelu.alpha_ == 1.0f;
  }
  using ActivationParameters = union ActivationParameters {
    struct {
      float alpha_;
    } LeakyRelu;
    struct {
      float minimum_;
      float maximum_;
    } Clip;
    struct {
      float alpha_;
      float beta_;
    } HardSigmoid;
    struct {
      float alpha_;
    } QuickGelu;
    struct {
      float alpha_;
    } Elu;
    struct {
      float alpha_;
    } ThresholdedRelu;
    float values_[2];
  };
  ActivationParameters activation_params_ = {};
  ActivationKind activation_kind_ = ActivationKind::None;
};

// Fixed slots keep activation uniform definitions and values index-aligned.
constexpr size_t kActivationUniformVariableCount = 2;

// Activation uniforms must be last in each program's uniform definition list.
#define WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES                                     \
  {"activation_param_0", onnxruntime::webgpu::ProgramUniformVariableDataType::Float32}, \
  { "activation_param_1", onnxruntime::webgpu::ProgramUniformVariableDataType::Float32 }

Status GetFusedActivationAttr(const OpKernelInfo& info, Activation& activation);

std::string GetActivationSnippet(const Activation& activation, std::string value_type, std::string base_type);

// Returns module-scope WGSL required by GetActivationSnippet; emit it before the snippet's use.
std::string GetActivationDeclaration(const Activation& activation, std::string value_type, std::string base_type);

// Appends exactly kActivationUniformVariableCount values, with empty entries for unused slots.
void AppendActivationUniformsData(const Activation& activation, std::vector<ProgramUniformVariableValue>& variables);
void AppendActivationUniformsData(const Activation& activation, ProgramBase& program);

}  // namespace webgpu
}  // namespace onnxruntime
