// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_LAYER_NORM_CONFIG(F) \
  F(bool, has_bias)                 \
  F(bool, simplified)               \
  F(bool, has_mean_output)          \
  F(bool, has_inv_std_dev_output)   \
  F(bool, split_norm_dim)           \
  F(bool, fp32_normalization)

WEBGPU_DECLARE_CONFIG(LayerNormConfig, WEBGPU_LAYER_NORM_CONFIG);
#undef WEBGPU_LAYER_NORM_CONFIG

struct LayerNormShader {
  using Config = LayerNormConfig;
  static constexpr std::string_view name = "LayerNorm";
  static Status GenerateShaderCode(const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"components", ProgramUniformVariableDataType::Uint32},
                                          {"norm_count", ProgramUniformVariableDataType::Uint32},
                                          {"norm_size", ProgramUniformVariableDataType::Uint32},
                                          {"norm_size_vectorized", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});
};

using LayerNormProgram = ConfiguredProgram<LayerNormShader>;

template <bool simplified>
class LayerNorm final : public WebGpuKernel {
 public:
  LayerNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
    info.GetAttrOrDefault<int64_t>("axis", &axis_, -1);
    info.GetAttrOrDefault<float>("epsilon", &epsilon_, 1e-05f);
    info.GetAttrOrDefault<int64_t>("stash_type", &stash_type_, 1);
  }

  Status ComputeInternal(ComputeContext& context) const override;

 protected:
  std::string cache_hint;

 private:
  int64_t axis_;
  float epsilon_;
  int64_t stash_type_;
};

// Configures and dispatches a LayerNormProgram. Centralizes the program-setup logic
// (uniform variables, components, split_norm_dim heuristic, workgroup sizing) so callers
// other than the LayerNorm kernel (e.g. fused MatMulNBits ops) do not need to duplicate it.
// `bias`, `mean` and `inv_std_dev` may be nullptr.
// `fp32_normalization` keeps normalization and affine arithmetic in f32 until the final output cast.
Status RunLayerNormProgram(ComputeContext& context,
                           const Tensor* x,
                           const Tensor* scale,
                           const Tensor* bias,
                           float epsilon,
                           uint32_t norm_count,
                           int64_t norm_size,
                           bool simplified,
                           Tensor* y,
                           Tensor* mean,
                           Tensor* inv_std_dev,
                           bool fp32_normalization = false);

}  // namespace webgpu
}  // namespace onnxruntime
