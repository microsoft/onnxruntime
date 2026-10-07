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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_SKIP_LAYER_NORM_PROGRAM_CONFIG);
    Config(bool hasBeta, bool hasBias, bool has_input_skip_bias_sum, bool simplified, bool split_hidden_dim) {
      hasBeta_ = hasBeta;
      hasBias_ = hasBias;

      has_input_skip_bias_sum_ = has_input_skip_bias_sum;
      simplified_ = simplified;
      split_hidden_dim_ = split_hidden_dim;
    }
  };
  static constexpr std::string_view name = "SkipLayerNorm";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"components", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"skip_size", ProgramUniformVariableDataType::Uint32});
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
