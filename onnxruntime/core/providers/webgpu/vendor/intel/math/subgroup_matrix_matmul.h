// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <memory>

#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// Subgroup matrix configuration used by this implementation (Intel Xe2/Xe3, F16).
constexpr uint32_t kSubgroupMatrixM = 8;
constexpr uint32_t kSubgroupMatrixN = 16;
constexpr uint32_t kSubgroupMatrixK = 16;
// One or more 32-lane subgroups compute one output tile. The tile shape is chosen
// adaptively per call from these candidate sizes (must be multiples of the
// subgroup-matrix M/N): TileM in {8,16,32,64}, TileN in {16,32,64}. When the
// output has few tiles, the K dimension is split across up to kSubgroupMatrixMaxSplitK
// subgroups that each accumulate part of K and are reduced in shared memory; the
// workgroup then has kSubgroupMatrixSubgroupSize * split_k threads.
constexpr uint32_t kSubgroupMatrixSubgroupSize = 32;
constexpr uint32_t kSubgroupMatrixMaxTileM = 64;
constexpr uint32_t kSubgroupMatrixMaxTileN = 64;
constexpr uint32_t kSubgroupMatrixMaxSplitK = 8;

// Computes Y = A @ B (+ optional bias) using subgroupMatrixMultiplyAccumulate.
// sg_mat_count_m/n select how many subgroup matrices the tile spans along M/N.
// split_k is the number of subgroups that cooperatively reduce the K dimension.
class SubgroupMatrixMatMulProgram final : public Program<SubgroupMatrixMatMulProgram> {
 public:
  SubgroupMatrixMatMulProgram(bool has_bias, uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k)
      : Program{"SubgroupMatrixMatMul"},
        has_bias_(has_bias),
        sg_mat_count_m_(sg_mat_count_m),
        sg_mat_count_n_(sg_mat_count_n),
        split_k_(split_k) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32});

 private:
  const bool has_bias_;
  const uint32_t sg_mat_count_m_;
  const uint32_t sg_mat_count_n_;
  const uint32_t split_k_;
};

// Creates the Intel subgroup-matrix MatMul implementation if this device supports
// the required 8x16x16 F16 subgroup matrix configuration. Returns nullptr otherwise,
// in which case the caller falls back to the default MatMul path.
std::unique_ptr<MatMul::MatMulOptImpl> CreateSubgroupMatrixMatMulImpl(const ComputeContextBase& context,
                                                                      const MatMul& parent);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
