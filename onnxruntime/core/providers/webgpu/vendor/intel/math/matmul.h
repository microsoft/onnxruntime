// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

#define WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_CONFIG(F) \
  F(ShaderActivation, activation_)                \
  F(bool, has_bias_)                              \
  F(bool, is_vec4_)                               \
  F(bool, a_vec4_)                                \
  F(bool, b_is_fp16_)                             \
  F(bool, is_channels_last_)                      \
  F(InlinedVector<int64_t>, elements_per_thread_)

struct MatMulSubgroupProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_CONFIG);
    Config(const Activation& activation, bool bias, bool is_vec4, bool a_vec4, bool b_is_fp16, bool is_channels_last,
           const gsl::span<int64_t>& elements_per_thread)
        : activation_(activation),
          has_bias_{bias},
          is_vec4_{is_vec4},
          a_vec4_{a_vec4},
          b_is_fp16_{b_is_fp16},
          is_channels_last_{is_channels_last},
          elements_per_thread_(elements_per_thread.begin(), elements_per_thread.end()) {}
  };
  static constexpr std::string_view name = "MatMulSubgroup";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_inner", ProgramUniformVariableDataType::Uint32},
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);
};
#undef WEBGPU_MAT_MUL_SUBGROUP_PROGRAM_CONFIG

using MatMulSubgroupProgram = ConfiguredProgram<MatMulSubgroupProgramShader>;

bool CanApplyMatMulIntel(const ComputeContext& context, int64_t M, int64_t N, int64_t K);

Status ApplyMatMulIntel(ComputeContext& context,
                        const Activation& activation,
                        const std::vector<const Tensor*>& inputs,
                        Tensor* output,
                        bool is_channels_last);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
