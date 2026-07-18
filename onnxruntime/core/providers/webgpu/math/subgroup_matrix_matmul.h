// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <functional>
#include <memory>
#include <optional>

#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

// Per-workgroup output tiling for one MatMul problem: the tile shape and split-K
// factor chosen by a vendor-specific policy. The subgroup-matrix shape itself is
// separate from this selection.
struct SubgroupMatrixTiling {
  uint32_t tile_m;   // output rows per workgroup
  uint32_t tile_n;   // output cols per workgroup
  uint32_t split_k;  // subgroups cooperating along K (1 = no split)
};

// Vendor-supplied callback that selects the output tiling for a given problem.
// Returning nullopt declines the problem, so MatMul falls back to another
// compute path. An empty selector likewise yields no implementation.
using SubgroupMatrixTilingSelector =
    std::function<std::optional<SubgroupMatrixTiling>(const ComputeContext& context,
                                                      uint32_t M, uint32_t N, uint32_t K)>;

// Creates a MatMulOptImpl that runs the subgroup-matrix kernel on devices whose
// vendor policy supports it. The per-problem output tiling comes from a
// vendor-specific selector chosen internally from the device context. Returns
// nullptr when no vendor policy applies, so the caller falls back to the default
// MatMul path.
std::unique_ptr<MatMul::MatMulOptImpl> CreateSubgroupMatrixMatMulImpl(
    const MatMul& parent, const ComputeContextBase& context);

// Computes Y = A @ B (+ optional bias) using subgroupMatrixMultiplyAccumulate.
// config_index selects the device subgroup-matrix config (into
// supported_subgroup_matrix_configs); sg_mat_count_m/n select how many subgroup
// matrices the tile spans along M/N; split_k is the number of subgroups that
// cooperatively reduce the K dimension.
class SubgroupMatrixMatMulProgram final : public Program<SubgroupMatrixMatMulProgram> {
 public:
  SubgroupMatrixMatMulProgram(bool has_bias, int32_t config_index,
                              uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k)
      : Program{"SubgroupMatrixMatMul"},
        has_bias_(has_bias),
        config_index_(config_index),
        sg_mat_count_m_(sg_mat_count_m),
        sg_mat_count_n_(sg_mat_count_n),
        split_k_(split_k) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"num_n_tile", ProgramUniformVariableDataType::Uint32});

 private:
  const bool has_bias_;
  const int32_t config_index_;
  const uint32_t sg_mat_count_m_;
  const uint32_t sg_mat_count_n_;
  const uint32_t split_k_;
};

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
