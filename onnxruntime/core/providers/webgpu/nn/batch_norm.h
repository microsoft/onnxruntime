// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_BATCH_NORMALIZATION_PROGRAM_CONFIG(F) \
  F(float, epsilon_)                                 \
  F(int64_t, spatial_)                               \
  F(DataLayout, format_)

struct BatchNormalizationProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_BATCH_NORMALIZATION_PROGRAM_CONFIG);
  static Config MakeConfig(float epsilon, int64_t spatial, DataLayout format) {
    Config result{};
    result.epsilon_ = float{epsilon};
    result.spatial_ = int64_t{spatial};
    result.format_ = DataLayout{format};

    return result;
  }
  static constexpr std::string_view name = "BatchNormalization";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_BATCH_NORMALIZATION_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_BATCH_NORMALIZATION_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_BATCH_NORMALIZATION_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_BATCH_NORMALIZATION_PROGRAM_CONFIG

using BatchNormalizationProgram = ConfiguredProgram<BatchNormalizationProgramShader>;

template <bool is_nhwc>
class BatchNormalization final : public WebGpuKernel {
 public:
  BatchNormalization(const OpKernelInfo& info) : WebGpuKernel(info) {
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
    momentum_ = info.GetAttrOrDefault<float>("momentum", 0.9f);
    spatial_ = info.GetAttrOrDefault<int64_t>("spatial", 1);
    training_mode_ = info.GetAttrOrDefault<int64_t>("training_mode", 0);
    // NCHW for ai.onnx domain, NHWC for com.ms.internal.nhwc domain
    format_ = is_nhwc ? DataLayout::NHWC : DataLayout::NCHW;
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  float epsilon_;
  float momentum_;
  int64_t spatial_;
  int64_t training_mode_;
  DataLayout format_;
};

}  // namespace webgpu
}  // namespace onnxruntime