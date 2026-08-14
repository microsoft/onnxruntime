// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>

#include "core/common/status.h"

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
  Tanh
};

using Activation = struct Activation {
  std::string ToString() const {
    std::stringstream oss;
    oss << "ActivationKind: " << static_cast<int>(activation_kind_) << ";";
    // Increase the serialization precision so that activation_params_.values_ are captured with
    // max_digits10 significant digits rather than the default 6. This ensures that WGSL shaders
    // embedding different activation constants receive distinct cache keys.
    oss << std::setprecision(std::numeric_limits<float>::max_digits10);
    oss << "ActivationParams: " << activation_params_.values_[0] << ";";
    oss << "ActivationParams: " << activation_params_.values_[1] << ";";
    return oss.str();
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
    float values_[2];
  };
  ActivationParameters activation_params_ = {};
  ActivationKind activation_kind_ = ActivationKind::None;
};

Status GetFusedActivationAttr(const OpKernelInfo& info, Activation& activation);
std::string GetActivationSnippet(const Activation& activation, std::string value_type, std::string base_type);
// Status AppendActivationUniformsData(const Activation& activation, std::vector<ProgramUniformVariableValue>& variables);
// Status AppendActivationUniforms(const Activation& activation, std::vector<float>& data);

}  // namespace webgpu
}  // namespace onnxruntime
