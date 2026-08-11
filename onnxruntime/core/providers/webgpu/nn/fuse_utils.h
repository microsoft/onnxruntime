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

// The numeric values are mirrored as integer literals by the `activation_kind` parameter of
// nn/im2col_matmul.wgsl.template, so append new kinds rather than reordering existing ones.
// im2col_matmul.cc static_asserts the values that template depends on.
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
  // Gelu's `approximate` attribute selects between two different expressions rather than scaling
  // one, so it is resolved to a kind at fusion time instead of being passed as a runtime
  // parameter. Numeric parameters (alpha, beta, min, max) travel as uniforms instead.
  Gelu,
  GeluTanh,
  Softplus
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
    struct {
      float alpha_;
    } QuickGelu;
    struct {
      float alpha_;
    } Elu;
    float values_[2];
  };
  ActivationParameters activation_params_ = {};
  ActivationKind activation_kind_ = ActivationKind::None;
};

Status GetFusedActivationAttr(const OpKernelInfo& info, Activation& activation);

// Returns the statement that applies the activation to a variable named `value`.
std::string GetActivationSnippet(const Activation& activation, std::string value_type, std::string base_type);

// Returns module-scope WGSL that GetActivationSnippet's output depends on, or "" when the
// activation is expressible as a single self-contained expression.
//
// Not every activation fits in one expression: WGSL has no `erf` builtin, and its `tanh`
// overflows for large inputs, so those have to be open-coded as functions. Callers must emit
// this into shader.AdditionalImplementation() *before* any code containing the snippet, because
// WGSL has no forward declarations. Emitting it unconditionally is safe: it is empty for every
// activation that does not need it.
std::string GetActivationDeclaration(const Activation& activation, std::string value_type, std::string base_type);

}  // namespace webgpu
}  // namespace onnxruntime
