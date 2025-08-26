// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class LayerNormProgram final : public Program<LayerNormProgram> {
 public:
  LayerNormProgram(bool has_bias, bool is_fp16, bool simplified, bool has_mean_output,
                   bool has_inv_std_dev_output, bool split_norm_dim = false)
      : Program{"LayerNorm"},
        has_bias_{has_bias},
        is_fp16_{is_fp16},
        simplified_{simplified},
        has_mean_output_{has_mean_output},
        has_inv_std_dev_output_{has_inv_std_dev_output},
        split_norm_dim_{split_norm_dim} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"components", ProgramUniformVariableDataType::Uint32},
                                          {"norm_count", ProgramUniformVariableDataType::Uint32},
                                          {"norm_size", ProgramUniformVariableDataType::Uint32},
                                          {"norm_size_vectorized", ProgramUniformVariableDataType::Uint32},
                                          {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  bool has_bias_;
  bool is_fp16_;
  bool simplified_;
  bool has_mean_output_;
  bool has_inv_std_dev_output_;
  bool split_norm_dim_;
};

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

}  // namespace webgpu
}  // namespace onnxruntime
