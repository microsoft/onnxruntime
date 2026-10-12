// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

#define WEBGPU_MAT_MUL_N_BITS_WIDE_TILE_PROGRAM_CONFIG(F) \
  F(bool, has_zero_points_)                               \
  F(bool, has_bias_)                                      \
  F(bool, has_weight_idx_)                                \
  F(bool, has_weight_idx_indirect_)                       \
  F(uint32_t, tile_m_)                                    \
  F(uint32_t, tile_n_)                                    \
  F(uint32_t, nbits_)                                     \
  F(uint32_t, subgroup_min_size_)                         \
  F(bool, acc_f32_)

struct MatMulNBitsWideTileProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_MAT_MUL_N_BITS_WIDE_TILE_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect, uint32_t tile_m,
                           uint32_t tile_n, uint32_t nbits, uint32_t subgroup_min_size, bool acc_f32) {
    Config result{};
    result.has_zero_points_ = bool{has_zero_points};
    result.has_bias_ = bool{has_bias};
    result.has_weight_idx_ = bool{has_weight_idx};
    result.has_weight_idx_indirect_ = bool{has_weight_idx_indirect};
    result.tile_m_ = uint32_t(tile_m);
    result.tile_n_ = uint32_t(tile_n);
    result.nbits_ = uint32_t(nbits);
    result.subgroup_min_size_ = uint32_t(subgroup_min_size);
    result.acc_f32_ = bool(acc_f32);

    return result;
  }
  static constexpr std::string_view name = "MatMulNBitsWideTile";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_MAT_MUL_N_BITS_WIDE_TILE_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, Batch)                                                 \
  U(Uint32, M)                                                     \
  U(Uint32, N)                                                     \
  U(Uint32, K_of_a)                                                \
  U(Uint32, K_of_b)                                                \
  U(Uint32, n_blocks_per_col)                                      \
  U(Uint32, zero_blocks_per_col)                                   \
  U(Uint32, num_N_tile)                                            \
  U(Uint32, num_M_tile)                                            \
  U(Uint32, weight_idx)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_MAT_MUL_N_BITS_WIDE_TILE_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_MAT_MUL_N_BITS_WIDE_TILE_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_MAT_MUL_N_BITS_WIDE_TILE_PROGRAM_CONFIG

using MatMulNBitsWideTileProgram = ConfiguredProgram<MatMulNBitsWideTileProgramShader>;

#define WEBGPU_MAT_MUL_N_BITS_PROGRAM_CONFIG(F) \
  F(uint32_t, tile_size_)                       \
  F(uint32_t, nbits_)                           \
  F(bool, has_zero_points_)                     \
  F(bool, has_bias_)                            \
  F(bool, has_weight_idx_)                      \
  F(bool, has_weight_idx_indirect_)             \
  F(bool, single_scale_weights_)                \
  F(uint32_t, tile_size_k_vec_)                 \
  F(bool, broadcast_a_row_)                     \
  F(bool, acc_f32_)

struct MatMulNBitsProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_MAT_MUL_N_BITS_PROGRAM_CONFIG);
  static Config MakeConfig(uint32_t tile_size, uint32_t nbits, bool has_zero_points, bool has_bias, bool has_weight_idx,
                           bool has_weight_idx_indirect, bool single_scale_weights, uint32_t tile_size_k_vec = 16,
                           bool broadcast_a_row = false, bool acc_f32 = false) {
    Config result{};
    result.tile_size_ = uint32_t(tile_size);
    result.nbits_ = uint32_t(nbits);
    result.has_zero_points_ = bool(has_zero_points);
    result.has_bias_ = bool(has_bias);
    result.has_weight_idx_ = bool{has_weight_idx};
    result.has_weight_idx_indirect_ = bool{has_weight_idx_indirect};
    result.single_scale_weights_ = bool(single_scale_weights);
    result.tile_size_k_vec_ = uint32_t(tile_size_k_vec);
    result.broadcast_a_row_ = bool(broadcast_a_row);
    result.acc_f32_ = bool(acc_f32);

    return result;
  }
  static constexpr std::string_view name = "MatMulNBits";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);
#define WEBGPU_MAT_MUL_N_BITS_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, M)                                           \
  U(Uint32, N)                                           \
  U(Uint32, K)                                           \
  U(Uint32, K_of_a)                                      \
  U(Uint32, K_of_b)                                      \
  U(Uint32, block_size)                                  \
  U(Uint32, blocks_per_col)                              \
  U(Uint32, zero_blocks_per_col)                         \
  U(Uint32, num_N_tile)                                  \
  U(Uint32, batch_count)                                 \
  U(Uint32, weight_idx)                                  \
  U(Uint32, dispatch_M)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_MAT_MUL_N_BITS_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_MAT_MUL_N_BITS_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_MAT_MUL_N_BITS_PROGRAM_CONFIG

using MatMulNBitsProgram = ConfiguredProgram<MatMulNBitsProgramShader>;

class MatMulNBits final : public WebGpuKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : WebGpuKernel(info) {
    K_ = info.GetAttr<int64_t>("K");
    N_ = info.GetAttr<int64_t>("N");
    block_size_ = info.GetAttr<int64_t>("block_size");
    bits_ = info.GetAttr<int64_t>("bits");
    accuracy_level_ = info.GetAttrOrDefault<int64_t>("accuracy_level", 4);
    ORT_ENFORCE(bits_ == 4 || bits_ == 8 || bits_ == 2,
                "Only 4b/8b/2b quantization is supported for MatMulNBits op, additional bits support is planned.");
  }

  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t accuracy_level_;
  int64_t bits_;
};

Status ApplyMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op, int64_t N_op, int64_t block_size_op, int64_t accuracy_level, int64_t bits_op,
                        onnxruntime::webgpu::ComputeContext& context, Tensor* y, const uint32_t weight_index = 0,
                        const Tensor* weight_index_indirect = nullptr,
                        uint32_t override_M = 0);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
