// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

class UnaryElementwiseProgram final : public Program<UnaryElementwiseProgram> {
 public:
  UnaryElementwiseProgram(const std::string& kernel_name, std::string_view expression, std::string_view additional_impl, ShaderUsage usage)
      : Program{kernel_name}, expression_{expression}, additional_impl_{additional_impl}, additional_usage_{usage} {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"vec_size", ProgramUniformVariableDataType::Uint32},  // output size
      {"attr", ProgramUniformVariableDataType::Float32});    // float type attribute(s)
                                                             // TODO: add u32/i32 attribute(s) if needed

 private:
  std::string_view expression_;
  std::string_view additional_impl_;
  ShaderUsage additional_usage_;
};

// TODO: after upgrading to C++20, use consteval to make a compile-time constructor so that it will be safe to switch
//       the std::string to std::string_view. This will avoid the cost of copying the string.

class UnaryElementwise : public WebGpuKernel {
 public:
  UnaryElementwise(const OpKernelInfo& info,
                   const std::string& kernel_name,
                   const std::string& expression,
                   const std::string& additional_impl = "",
                   ShaderUsage usage = ShaderUsage::None) : WebGpuKernel{info},
                                                            kernel_name_{kernel_name},
                                                            expression_{expression},
                                                            additional_impl_{additional_impl},
                                                            additional_usage_{usage} {}

 protected:
  std::string cache_hint;

  Status ComputeInternal(ComputeContext& context) const final;
  virtual Status ConfigureProgram(const ComputeContext& /*context*/, UnaryElementwiseProgram& program) const {
    program.AddUniformVariables({{}});  // empty for attribute(s)
    return Status::OK();
  }

 private:
  std::string kernel_name_;
  std::string expression_;
  std::string additional_impl_;
  ShaderUsage additional_usage_;
};

constexpr const char ErfImpl[] = R"(
const r0 = 0.3275911;
const r1 = 0.254829592;
const r2 = -0.284496736;
const r3 = 1.421413741;
const r4 = -1.453152027;
const r5 = 1.061405429;

fn erf_v(v: x_value_t) -> x_value_t {
  let absv = abs(v);
  let x = 1.0 / (1.0 + r0 * absv);
  return sign(v) * (1.0 - ((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x * exp(-absv * absv));
}
)";

constexpr const char HardSigmoidImpl[] = R"(
fn hard_sigmoid_v(v: vec4<x_element_t>) -> vec4<x_element_t> {
  let alpha = x_element_t(uniforms.attr[0]);
  let beta_v = vec4<x_element_t>(uniforms.attr[1]);
  return max(vec4<x_element_t>(0.0),
             min(vec4<x_element_t>(1.0), alpha * v + beta_v));
}
)";

// built-in function tanh() does not work with large input (f32 88.7 or f16 11.09)
// https://github.com/gpuweb/gpuweb/issues/4458
constexpr const char TanhImpl[] = R"(
fn tanh_v(a: x_value_t) -> x_value_t {
  let expr = exp(-2 * abs(a));
  return sign(a) * (1 - expr) / (1 + expr);
}
)";

constexpr const char EluImpl[] = R"(
fn elu(a: x_element_t) -> x_element_t {
  let alpha = x_element_t(uniforms.attr);
  return select((exp(a) - 1.0) * alpha, a, a >= 0.0);
}

fn elu_v(v: vec4<x_element_t>) -> vec4<x_element_t> {
  return vec4(elu(v.x), elu(v.y), elu(v.z), elu(v.w));
}
)";

// default GELU expression, depending on ErfImpl
constexpr const char GeluExpr[] = "0.5 * a * (1.0 + erf_v(a * 0.7071067811865475))";

// fast GELU expression, depending on TanhImpl
constexpr const char FastGeluExpr[] = "a * (0.5 + 0.5 * tanh_v(a * (0.035677408136300125 * a * a + 0.7978845608028654)))";

}  // namespace webgpu
}  // namespace onnxruntime
