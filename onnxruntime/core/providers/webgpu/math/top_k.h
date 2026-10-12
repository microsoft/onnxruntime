// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_TOP_K_PROGRAM_CONFIG(F) \
  F(uint32_t, wg_)                     \
  F(uint32_t, shared_size_)            \
  F(bool, largest_)                    \
  F(bool, is_fp16_)

struct TopKProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TOP_K_PROGRAM_CONFIG);
  static Config MakeConfig(uint32_t wg, uint32_t shared_size, bool largest, bool is_fp16) {
    Config result{};
    result.wg_ = uint32_t{wg};
    result.shared_size_ = uint32_t{shared_size};
    result.largest_ = bool{largest};
    result.is_fp16_ = bool{is_fp16};

    return result;
  }
  static constexpr std::string_view name = "TopK";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TOP_K_PROGRAM_SHADER_UNIFORMS(U) \
  U(Int32, cols)                                \
  U(Int32, k)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TOP_K_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TOP_K_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_TOP_K_PROGRAM_CONFIG

using TopKProgram = ConfiguredProgram<TopKProgramShader>;

// Global-memory bitonic sort programs for cols > 2048
#define WEBGPU_TOP_K_INIT_PROGRAM_CONFIG(F) \
  F(bool, largest_)                         \
  F(bool, is_fp16_)

struct TopKInitProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TOP_K_INIT_PROGRAM_CONFIG);
  static Config MakeConfig(bool largest, bool is_fp16) {
    Config result{};
    result.largest_ = bool{largest};
    result.is_fp16_ = bool{is_fp16};

    return result;
  }
  static constexpr std::string_view name = "TopKInit";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TOP_K_INIT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Int32, cols)                                     \
  U(Int32, padded_cols)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TOP_K_INIT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TOP_K_INIT_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_TOP_K_INIT_PROGRAM_CONFIG

using TopKInitProgram = ConfiguredProgram<TopKInitProgramShader>;

#define WEBGPU_TOP_K_SORT_STEP_PROGRAM_CONFIG(F) F(bool, largest_)

struct TopKSortStepProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TOP_K_SORT_STEP_PROGRAM_CONFIG);
  static Config MakeConfig(bool largest) {
    Config result{};
    result.largest_ = bool{largest};

    return result;
  }
  static constexpr std::string_view name = "TopKSortStep";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TOP_K_SORT_STEP_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, block_size)                                   \
  U(Uint32, gap)                                          \
  U(Uint32, padded_cols)                                  \
  U(Uint32, total_threads)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TOP_K_SORT_STEP_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TOP_K_SORT_STEP_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_TOP_K_SORT_STEP_PROGRAM_CONFIG

using TopKSortStepProgram = ConfiguredProgram<TopKSortStepProgramShader>;

#define WEBGPU_TOP_K_OUTPUT_PROGRAM_CONFIG(F)

struct TopKOutputProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TOP_K_OUTPUT_PROGRAM_CONFIG);
  static Config MakeConfig() {
    Config result{};

    return result;
  }
  static constexpr std::string_view name = "TopKOutput";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TOP_K_OUTPUT_PROGRAM_SHADER_UNIFORMS(U) \
  U(Int32, padded_cols)                                \
  U(Int32, k)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TOP_K_OUTPUT_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TOP_K_OUTPUT_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_TOP_K_OUTPUT_PROGRAM_CONFIG

using TopKOutputProgram = ConfiguredProgram<TopKOutputProgramShader>;

class TopK final : public WebGpuKernel {
 public:
  TopK(const OpKernelInfo& info) : WebGpuKernel{info} {
    opset_ = info.node().SinceVersion();
    info.GetAttrOrDefault<int64_t>("axis", &axis_, -1);
    info.GetAttrOrDefault<int64_t>("largest", &largest_, 1);
    info.GetAttrOrDefault<int64_t>("sorted", &sorted_, 1);
    if (opset_ <= 9) {
      int64_t k_temp = 0;
      ORT_ENFORCE(info.GetAttr<int64_t>("k", &k_temp).IsOK());
      attr_k_ = k_temp;
    }
  }

  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int64_t axis_ = -1;
  int64_t largest_ = 1;
  int64_t sorted_ = 1;
  int64_t attr_k_ = 0;
  int opset_;
};

}  // namespace webgpu
}  // namespace onnxruntime
