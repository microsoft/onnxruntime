// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <functional>
#include <memory>
#include <mutex>
#include <optional>

#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/subgroup_matrix_common.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

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
  // N is the logical output width; N_b is B's physical row stride (== N unless B
  // was column-padded to an even stride for odd N).
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"num_n_tile", ProgramUniformVariableDataType::Uint32},
                                          {"N_b", ProgramUniformVariableDataType::Uint32});

 private:
  const bool has_bias_;
  const int32_t config_index_;
  const uint32_t sg_mat_count_m_;
  const uint32_t sg_mat_count_n_;
  const uint32_t split_k_;
};

// Copies a row-major f16 weight B [K, N] into a column-padded [K, N_b] buffer
// (N_b >= N), zero-filling columns [N, N_b). Gives B an even row stride so the
// subgroup-matrix f16 load's 4-byte row-start alignment holds for odd N. Shared by
// the subgroup-matrix MatMul and Conv 1x1 paths.
class SubgroupMatrixMatMulPadBProgram final : public Program<SubgroupMatrixMatMulPadBProgram> {
 public:
  SubgroupMatrixMatMulPadBProgram() : Program{"SubgroupMatrixMatMulPadB"} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"N_b", ProgramUniformVariableDataType::Uint32});
};

// Caches an even-strided (N_b = N + 1) copy of a constant weight B with odd N so
// the column-pad pass runs once and is reused across inference steps. Shared by the
// subgroup-matrix MatMul and Conv 1x1 paths, which need the same odd-N alignment
// fix. Only valid when B is a constant initializer - a runtime B changes per run
// and must not be cached.
class SubgroupMatrixPadBCache {
 public:
  // Builds the padded copy of `b` (whose last dim must equal N) on first use and
  // returns it via `b_used` with its row stride via `n_b`.
  Status EnsurePaddedB(ComputeContext& context, const Tensor& b, uint32_t N,
                       /*out*/ const Tensor*& b_used, /*out*/ uint32_t& n_b) const;

 private:
  mutable std::once_flag pad_once_;
  mutable std::unique_ptr<Tensor> padded_b_;
  mutable uint32_t padded_b_stride_ = 0;
};

// Runs the subgroup-matrix kernel for Y = A @ B (+ optional bias) given resolved
// operands and their logical shapes. a_shape/b_shape may differ from the tensors'
// own shapes when the caller reshaped them (e.g. a 1x1 Conv folding N,H,W into M).
// Handles the shared 2D-weight and batched-B cases, odd-N even-stride padding (via
// pad_cache when b_is_constant), vendor tiling and dispatch. When `output` is
// non-null the caller's pre-allocated tensor is used as a flat buffer (its element
// count must equal batch*M*N); when null the result is allocated via
// context.Output(0, ...) shaped like a_shape with its trailing dim set to N. Sets
// handled=true on success; leaves handled=false (touching nothing) on an
// unsupported device/problem so the caller can fall back. Shared by the MatMul
// operator and the Conv 1x1 path.
Status DispatchSubgroupMatrixMatMul(ComputeContext& context,
                                    int32_t config_index,
                                    const SubgroupMatrixTilingSelector& tiling_selector,
                                    const SubgroupMatrixPadBCache& pad_cache,
                                    const Tensor* a, const Tensor* b, const Tensor* bias,
                                    Tensor* output,
                                    const TensorShape& a_shape, const TensorShape& b_shape,
                                    bool b_is_constant,
                                    /*out*/ bool& handled);

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
