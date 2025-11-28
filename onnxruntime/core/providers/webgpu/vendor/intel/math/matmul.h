// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

class MatMulSubgroupProgram final : public Program<MatMulSubgroupProgram> {
 public:
  MatMulSubgroupProgram(const Activation& activation,
                        bool bias,
                        bool is_vec4,
                        const gsl::span<int64_t>& elements_per_thread)
      : Program{"MatMulSubgroup"},
        activation_(activation),
        has_bias_{bias},
        is_vec4_{is_vec4},
        elements_per_thread_(elements_per_thread.begin(), elements_per_thread.end()) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"dim_a_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_b_outer", ProgramUniformVariableDataType::Uint32},
                                          {"dim_inner", ProgramUniformVariableDataType::Uint32});

 private:
  const Activation activation_;
  const bool has_bias_;
  const bool is_vec4_;
  const InlinedVector<int64_t> elements_per_thread_;
};

bool CanApplyMatMulIntel(const ComputeContext& context, int64_t M, int64_t N, int64_t K);

Status ApplyMatMulIntel(ComputeContext& context,
                        const Activation& activation,
                        std::vector<const Tensor*>& inputs,
                        Tensor* output);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
