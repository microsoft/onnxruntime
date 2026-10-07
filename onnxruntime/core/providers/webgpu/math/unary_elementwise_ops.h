// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

// Register int64 Clip kernels (dedicated ClipInt64 kernel) with conditional int64 support.
void RegisterClipInt64Kernels(KernelRegistry& kernel_registry, bool enable_int64);

#define WEBGPU_UNARY_ELEMENTWISE_PROGRAM_CONFIG(F) \
  F(std::string, program_name_)                    \
  F(ShaderLiteral, expression_)                    \
  F(ShaderLiteral, additional_impl_)               \
  F(uint32_t, additional_usage_)

struct UnaryElementwiseProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_UNARY_ELEMENTWISE_PROGRAM_CONFIG);
    Config(const std::string& kernel_name, ShaderLiteral expression, ShaderLiteral additional_impl,
           ShaderUsage usage)
        : program_name_{kernel_name},
          expression_{expression},
          additional_impl_{additional_impl},
          additional_usage_{usage.usage} {}
  };
  static std::string_view Name(const Config& config) { return config.program_name_; }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"vec_size", ProgramUniformVariableDataType::Uint32},  // output size
      {"attr", ProgramUniformVariableDataType::Float32});    // float type attribute(s)
                                                             // TODO: add u32/i32 attribute(s) if needed
};
#undef WEBGPU_UNARY_ELEMENTWISE_PROGRAM_CONFIG

using UnaryElementwiseProgram = ConfiguredProgram<UnaryElementwiseProgramShader>;

class UnaryElementwise : public WebGpuKernel {
 public:
  UnaryElementwise(const OpKernelInfo& info, const std::string& kernel_name, ShaderLiteral expression,
                   ShaderLiteral additional_impl = "", ShaderUsage usage = ShaderUsage::None)
      : WebGpuKernel{info},
        kernel_name_{kernel_name},
        expression_{expression},
        additional_impl_{additional_impl},
        additional_usage_{usage.usage} {}

 protected:
  Status ComputeInternal(ComputeContext& context) const final;
  virtual Status ConfigureProgram(const ComputeContext& /*context*/, UnaryElementwiseProgram& program) const {
    program.AddUniformVariables({{}});  // empty for attribute(s)
    return Status::OK();
  }

 private:
  std::string kernel_name_;
  ShaderLiteral expression_;
  ShaderLiteral additional_impl_;
  ShaderUsage additional_usage_;
};

class Gelu : public UnaryElementwise {
 public:
  Gelu(const OpKernelInfo& info);
};

class LinearUnit : public UnaryElementwise {
 public:
  LinearUnit(const OpKernelInfo& info,
             const std::string& kernel_name,
             ShaderLiteral expression,
             ShaderLiteral additional_impl,
             float default_alpha)
      : UnaryElementwise{info, kernel_name, expression, additional_impl, ShaderUsage::UseElementTypeAlias} {
    info.GetAttrOrDefault("alpha", &alpha_, default_alpha);
  }

  Status ConfigureProgram(const ComputeContext& /*context*/, UnaryElementwiseProgram& program) const override {
    program.AddUniformVariables({alpha_});
    return Status::OK();
  }

 protected:
  float alpha_;
};

class QuickGelu : public LinearUnit {
 public:
  QuickGelu(const OpKernelInfo& info);
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
  let beta_v = vec4<x_element_t>(x_element_t(uniforms.attr[1]));
  return max(vec4<x_element_t>(0.0),
             min(vec4<x_element_t>(1.0), alpha * v + beta_v));
}
)";

constexpr const char HardSwishImpl[] = R"(
fn hard_swish_v(v: vec4<x_element_t>) -> vec4<x_element_t> {
  let alpha = x_element_t(1.0 / 6.0);
  let beta_v = vec4<x_element_t>(x_element_t(0.5));
  return v * max(vec4<x_element_t>(0.0),
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

constexpr const char QuickGeluImpl[] = R"(
fn quick_gelu_v(a: vec4<x_element_t>) -> vec4<x_element_t> {
  let one = x_element_t(1.0);
  let zero = x_element_t(0.0);
  let alpha_vec = vec4<x_element_t>(x_element_t(uniforms.attr));
  let v = a * alpha_vec;
  var x1 : vec4<x_element_t>;
  for (var i = 0; i < 4; i = i + 1) {
    if (v[i] >= zero) {
      x1[i] = one / (one + exp(-v[i]));
    } else {
      x1[i] = one - one / (one + exp(v[i]));
    }
  }
  return a * x1;
}
)";

// default GELU expression, depending on ErfImpl
constexpr const char GeluExpr[] = "0.5 * a * (1.0 + erf_v(a * 0.7071067811865475))";

// fast GELU expression, depending on TanhImpl
constexpr const char FastGeluExpr[] = "a * (0.5 + 0.5 * tanh_v(a * (0.035677408136300125 * a * a + 0.7978845608028654)))";

}  // namespace webgpu
}  // namespace onnxruntime
