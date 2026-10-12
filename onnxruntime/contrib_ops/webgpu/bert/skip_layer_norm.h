// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

#define WEBGPU_SKIP_LAYER_NORM_PROGRAM_CONFIG(F) \
  F(bool, hasBeta_)                              \
  F(bool, hasBias_)                              \
  F(bool, has_input_skip_bias_sum_)              \
  F(bool, simplified_)                           \
  F(bool, split_hidden_dim_)

struct SkipLayerNormProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_SKIP_LAYER_NORM_PROGRAM_CONFIG);
  static Config MakeConfig(bool hasBeta, bool hasBias, bool has_input_skip_bias_sum, bool simplified, bool split_hidden_dim) {
    Config result{};

    result.hasBeta_ = hasBeta;
    result.hasBias_ = hasBias;

    result.has_input_skip_bias_sum_ = has_input_skip_bias_sum;
    result.simplified_ = simplified;
    result.split_hidden_dim_ = split_hidden_dim;

    return result;
  }
  static constexpr std::string_view name = "SkipLayerNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_SKIP_LAYER_NORM_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, components)                                   \
  U(Uint32, hidden_size)                                  \
  U(Float32, epsilon)                                     \
  U(Uint32, skip_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_SKIP_LAYER_NORM_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_SKIP_LAYER_NORM_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_SKIP_LAYER_NORM_PROGRAM_CONFIG

using SkipLayerNormProgram = ConfiguredProgram<SkipLayerNormProgramShader>;

template <bool simplified>
class SkipLayerNorm final : public WebGpuKernel {
 public:
  SkipLayerNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
    info.GetAttrOrDefault<float>("epsilon", &epsilon_, 1e-05f);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float epsilon_;
};

// Configures and dispatches a SkipLayerNormProgram. Centralizes program-setup logic
// (uniform variables, components, split_hidden_dim heuristic, workgroup sizing) so callers
// other than the SkipLayerNorm kernel (e.g. fused MatMulNBits ops) do not need to duplicate it.
// `beta`, `bias` and `input_skip_bias_sum` may be nullptr.
Status RunSkipLayerNormProgram(ComputeContext& context,
                               const Tensor* x,
                               const Tensor* skip,
                               const Tensor* gamma,
                               const Tensor* beta,
                               const Tensor* bias,
                               float epsilon,
                               bool simplified,
                               Tensor* output,
                               Tensor* input_skip_bias_sum);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
