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
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_TOP_K_PROGRAM_CONFIG);
    Config(uint32_t wg, uint32_t shared_size, bool largest, bool is_fp16)
        : wg_{wg}, shared_size_{shared_size}, largest_{largest}, is_fp16_{is_fp16} {}
  };
  static constexpr std::string_view name = "TopK";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"cols", ProgramUniformVariableDataType::Int32},
      {"k", ProgramUniformVariableDataType::Int32});
};
#undef WEBGPU_TOP_K_PROGRAM_CONFIG

using TopKProgram = ConfiguredProgram<TopKProgramShader>;

// Global-memory bitonic sort programs for cols > 2048
#define WEBGPU_TOP_K_INIT_PROGRAM_CONFIG(F) \
  F(bool, largest_)                         \
  F(bool, is_fp16_)

struct TopKInitProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_TOP_K_INIT_PROGRAM_CONFIG);
    Config(bool largest, bool is_fp16) : largest_{largest}, is_fp16_{is_fp16} {}
  };
  static constexpr std::string_view name = "TopKInit";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"cols", ProgramUniformVariableDataType::Int32},
      {"padded_cols", ProgramUniformVariableDataType::Int32});
};
#undef WEBGPU_TOP_K_INIT_PROGRAM_CONFIG

using TopKInitProgram = ConfiguredProgram<TopKInitProgramShader>;

#define WEBGPU_TOP_K_SORT_STEP_PROGRAM_CONFIG(F) F(bool, largest_)

struct TopKSortStepProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_TOP_K_SORT_STEP_PROGRAM_CONFIG);
    Config(bool largest) : largest_{largest} {}
  };
  static constexpr std::string_view name = "TopKSortStep";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"gap", ProgramUniformVariableDataType::Uint32},
      {"padded_cols", ProgramUniformVariableDataType::Uint32},
      {"total_threads", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_TOP_K_SORT_STEP_PROGRAM_CONFIG

using TopKSortStepProgram = ConfiguredProgram<TopKSortStepProgramShader>;

#define WEBGPU_TOP_K_OUTPUT_PROGRAM_CONFIG(F)

struct TopKOutputProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_TOP_K_OUTPUT_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "TopKOutput";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"padded_cols", ProgramUniformVariableDataType::Int32},
      {"k", ProgramUniformVariableDataType::Int32});
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
