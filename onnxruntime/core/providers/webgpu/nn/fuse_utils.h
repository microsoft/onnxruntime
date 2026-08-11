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
    // The parameters are baked into the generated WGSL, so distinct floats must produce distinct
    // keys. max_digits10 round-trips exactly; the stream default of 6 significant digits does not
    // -- 1234567.5 and 1234567.625 both format as 1.23457e+06 while emitting different WGSL.
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
