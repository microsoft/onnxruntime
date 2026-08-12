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
  Softplus,
  ThresholdedRelu,
  Erf
};

using Activation = struct Activation {
  // Cache hint contribution for the fused activation.
  //
  // Only what changes the GENERATED SHADER TEXT belongs here. Activation parameter values are
  // passed as uniforms (see AppendActivationUniformsData), so they are deliberately excluded:
  // two otherwise-identical programs that differ only in, say, LeakyRelu alpha share a single
  // compiled pipeline. The one parameter-derived shader variant is QuickGelu's alpha == 1 fast
  // path, which emits different code, so it is keyed as a boolean rather than as the raw float.
  std::string ToString() const {
    std::stringstream oss;
    oss << "ActivationKind: " << static_cast<int>(activation_kind_) << ";";
    if (activation_kind_ == ActivationKind::QuickGelu) {
      oss << "QuickGeluUnitAlpha: " << (HasUnitQuickGeluAlpha() ? 1 : 0) << ";";
    }
    return oss.str();
  }
  // QuickGelu(x, 1) is SiLU/Swish, which follows nearly every Conv in the YOLO family. At exactly
  // alpha == 1 the multiply folds away, so the shader drops it and needs no uniform for it.
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

// Number of uniform slots reserved for fused-activation parameters.
//
// The WebGPU EP matches uniform definitions to uniform values by index, and the definition list is
// a compile-time constant per program class. So every program whose shader embeds an activation
// declares a fixed number of activation slots and always supplies exactly that many values; slots
// the activation does not use get a zero-length value, which the EP drops from both the uniform
// struct and the uniform buffer (see shader_helper.cc, ShaderHelper::GetFinalSourceCode). That is
// what keeps parameterless activations emitting byte-identical shaders to before this mechanism
// existed.
constexpr size_t kActivationUniformVariableCount = 2;

// Declares the activation parameter uniforms. Append this to a program's
// WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES list, last, so the declaration order matches the order
// AppendActivationUniformsData() supplies the values in.
#define WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES                                     \
  {"activation_param_0", onnxruntime::webgpu::ProgramUniformVariableDataType::Float32}, \
  {                                                                                     \
    "activation_param_1", onnxruntime::webgpu::ProgramUniformVariableDataType::Float32  \
  }

Status GetFusedActivationAttr(const OpKernelInfo& info, Activation& activation);

// Returns the statement that applies the activation to a variable named `value`. Parameterized
// kinds read their parameters from the uniforms declared by
// WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES, so the parameter values stay out of the shader
// source and therefore out of the pipeline cache key.
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

// Supplies the values for the uniforms declared by WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES.
// Always appends exactly kActivationUniformVariableCount entries, using a zero-length value for
// any slot the activation does not read.
void AppendActivationUniformsData(const Activation& activation, std::vector<ProgramUniformVariableValue>& variables);
void AppendActivationUniformsData(const Activation& activation, ProgramBase& program);

}  // namespace webgpu
}  // namespace onnxruntime
