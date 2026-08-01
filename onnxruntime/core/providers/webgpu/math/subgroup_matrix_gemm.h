// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <memory>

#include "core/providers/webgpu/math/gemm.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

// Creates a GemmOptImpl that runs the subgroup-matrix kernel on devices whose
// vendor policy supports it. The per-problem output tiling comes from a
// vendor-specific selector chosen internally from the device context (the same
// selector used by the subgroup-matrix MatMul). Returns nullptr when no vendor
// policy applies, so the caller falls back to the default Gemm path.
std::unique_ptr<Gemm::GemmOptImpl> CreateSubgroupMatrixGemmImpl(
    const Gemm& parent, const ComputeContextBase& context);

// Computes Y = alpha * op(A) @ op(B) + beta * C using subgroupMatrixMultiplyAccumulate.
// config_index selects the device subgroup-matrix config (into
// supported_subgroup_matrix_configs); sg_mat_count_m/n select how many subgroup
// matrices the tile spans along M/N; split_k is the number of subgroups that
// cooperatively reduce the K dimension. trans_a / trans_b select the A / B load
// majorness; has_c enables the beta * C epilogue (C broadcast to [M, N] via the
// c_stride_m / c_stride_n uniforms).
class SubgroupMatrixGemmProgram final : public Program<SubgroupMatrixGemmProgram> {
 public:
  SubgroupMatrixGemmProgram(bool has_c, bool trans_a, bool trans_b, int32_t config_index,
                            uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k)
      : Program{"SubgroupMatrixGemm"},
        has_c_(has_c),
        trans_a_(trans_a),
        trans_b_(trans_b),
        config_index_(config_index),
        sg_mat_count_m_(sg_mat_count_m),
        sg_mat_count_n_(sg_mat_count_n),
        split_k_(split_k) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"beta", ProgramUniformVariableDataType::Float32},
                                          {"c_stride_m", ProgramUniformVariableDataType::Uint32},
                                          {"c_stride_n", ProgramUniformVariableDataType::Uint32});

 private:
  const bool has_c_;
  const bool trans_a_;
  const bool trans_b_;
  const int32_t config_index_;
  const uint32_t sg_mat_count_m_;
  const uint32_t sg_mat_count_n_;
  const uint32_t split_k_;
};

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
