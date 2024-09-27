// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class LayerNormProgram final : public Program<LayerNormProgram> {
 public:
  LayerNormProgram(int64_t axis, float epsilon, int64_t stash_type, bool has_bias, size_t x_size, bool isFP16, bool simplified) : Program{"LayerNorm"} {
    axis_ = axis;
    epsilon_ = epsilon;
    stash_type_ = stash_type;
    has_bias_ = has_bias;
    x_size_ = x_size;
    isFP16_ = isFP16;
    simplified_ = simplified;
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"norm_count", ProgramUniformVariableDataType::Uint32},
      {"norm_size", ProgramUniformVariableDataType::Uint32},
      {"norm_size_vectorized", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  int64_t axis_;
  float epsilon_;
  int64_t stash_type_;
  bool has_bias_;
  int x_size_;
  bool isFP16_;
  bool simplified_;
};

template <bool simplified>
class LayerNorm final : public WebGpuKernel {
 public:
  LayerNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
    info.GetAttrOrDefault<int64_t>("axis", &axis_, -1);
    info.GetAttrOrDefault<float>("epsilon", &epsilon_, 1e-05);
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
