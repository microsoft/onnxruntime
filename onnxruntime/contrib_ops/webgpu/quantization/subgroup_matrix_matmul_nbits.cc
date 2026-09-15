// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// The subgroup matrix config table, support check, and component-type validation live in the
// shared core header (core/providers/webgpu/math/subgroup_matrix_config.h) so both this contrib
// kernel and the core subgroup-matrix MatMul share them.
using onnxruntime::webgpu::SelectSubgroupMatrixConfig;
using onnxruntime::webgpu::supported_subgroup_matrix_configs;

struct SubgroupMatrixMatMulNBitsTiling {
  uint32_t tile_m;
  uint32_t tile_n;
  uint32_t workgroup_size;
  uint32_t output_m_multiple;  // required M % this == 0 (1 means unconstrained)
  uint32_t output_n_multiple;  // required N % this == 0 (1 means unconstrained)
  uint32_t chunk_size_k;       // required K % this == 0

  constexpr bool SupportsShape(uint32_t M, uint32_t N, uint32_t K) const {
    if (M < tile_m) {
      return false;
    }

    // TODO: Support arbitrary M/N/K shapes via input pre-packing or padded output buffers.
    return M % output_m_multiple == 0 &&
           N % output_n_multiple == 0 &&
           K % chunk_size_k == 0;
  }
};

constexpr SubgroupMatrixMatMulNBitsTiling GetSubgroupMatrixMatMulNBitsTiling(
    const onnxruntime::webgpu::SupportedSubgroupMatrixConfig& config, bool has_bias, uint32_t M, uint32_t N) {
  if (config.Is(8, 16, 16)) {
    // Cap tile at 64x64 to stay within workgroup memory limits
    if (has_bias) {
      return {64, 64, 256, 1, 1, 32};
    }
    // Optimized tile configuration: for large shape: 128x256 (512 threads).
    if (M >= 128 && N % 256 == 0) {
      return {128, 256, 512, 1, 256, 32};
    }
    // Default: 64x64 (256 threads).
    return {64, 64, 256, 1, 64, 32};
  }
  if (config.Is(16, 16, 16)) {
    return {128, 128, 128, 1, 1, 32};
  }
  return {32, 64, 128, 1, 64, 32};
}

// This program optimizes the layout of input matrix A(MxK) for SubgroupMatrixLoad, so that all elements of each
// subgroup matrix(mxk) are arranged continuously in memory.
// Take "M = 4, K = 4, m = 2, k = 2" as an example, the input matrix A is arranged in row-major order as follows:
// d00, d01, | d02, d03,
// d10, d11, | d12, d13,
// ---------------------
// d20, d21, | d22, d23,
// d30, d31, | d32, d33,
//
// The prepack program rearranges the input matrix A to be in the following order:
// d00, d01,
// d10, d11,
// ---------
// d02, d03,
// d12, d13,
// ---------
// d20, d21,
// d30, d31,
// ---------
// d22, d23,
// d32, d33,
class PrepackProgram final : public Program<PrepackProgram> {
 public:
  PrepackProgram(uint32_t m, uint32_t k) : Program{"SubgroupMatrixMatMulLayout"},
                                           m_(m),
                                           k_(k) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"M", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t m_;
  uint32_t k_;
};

Status PrepackProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddOutput("output_a", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_prepack.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, k_),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, m_));
}

Status GenerateShaderCode16x16x16(ShaderHelper& shader,
                                  const ShaderVariableHelper& b,
                                  const ShaderVariableHelper& scales_b,
                                  const ShaderVariableHelper& output,
                                  uint32_t nbits, int32_t config_index, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect) {
  const auto& config = supported_subgroup_matrix_configs[config_index];
  // Use 128x128 tile shader for 16x16x16 config (index 0)
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_16x16x16_128.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, config.K),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, config.M),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, config.N),
                             WGSL_TEMPLATE_VARIABLE(input_b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

Status GenerateShaderCode8x16x16(ShaderHelper& shader,
                                 const ShaderVariableHelper& b,
                                 const ShaderVariableHelper& scales_b,
                                 const ShaderVariableHelper& output,
                                 uint32_t nbits, int32_t config_index, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect,
                                 uint32_t tile_m, uint32_t tile_n, bool has_tail_buffer) {
  const auto& config = supported_subgroup_matrix_configs[config_index];
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_8x16x16.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias),
                             WGSL_TEMPLATE_PARAMETER(has_tail_buffer, has_tail_buffer),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, config.K),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, config.M),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, config.N),
                             WGSL_TEMPLATE_PARAMETER(tile_m, tile_m),
                             WGSL_TEMPLATE_PARAMETER(tile_n, tile_n),
                             WGSL_TEMPLATE_VARIABLE(input_b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

Status GenerateShaderCode8x8x8(ShaderHelper& shader, const ShaderVariableHelper& a, const ShaderVariableHelper& b,
                               const ShaderVariableHelper& scales_b,
                               const ShaderVariableHelper& output, uint32_t nbits, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect) {
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_8x8x8.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

Status SubgroupMatrixMatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseUniform);
  const auto& scales_b = shader.AddInput("scales_b", ShaderUsage::UseUniform);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  if (has_tail_buffer_) {
    shader.AddOutput("tail_output", ShaderUsage::None);
  }

  const auto& config = supported_subgroup_matrix_configs[config_index_];
  if (config.Is(8, 8, 8)) {
    return GenerateShaderCode8x8x8(shader, a, b, scales_b, output, nbits_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_);
  } else if (config.Is(8, 16, 16)) {
    return GenerateShaderCode8x16x16(shader, b, scales_b, output, nbits_, config_index_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_, tile_m_, tile_n_, has_tail_buffer_);
  } else if (config.Is(16, 16, 16)) {
    return GenerateShaderCode16x16x16(shader, b, scales_b, output, nbits_, config_index_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_);
  } else {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                  "Unsupported subgroup matrix config dimensions.");
  }
}

// Crops the valid rows out of the padded tail-tile buffer written by the
// SubgroupMatrixMatMulNBitsProgram's no-bias edge-tile fast path (see
// `has_tail_buffer` in subgroup_matrix_matmul_nbits_8x16x16.wgsl.template) and
// copies them into the real output tensor at the tail tile's row offset.
class SubgroupMatrixMatMulNBitsTailCopyProgram final : public Program<SubgroupMatrixMatMulNBitsTailCopyProgram> {
 public:
  SubgroupMatrixMatMulNBitsTailCopyProgram() : Program{"SubgroupMatrixMatMulNBitsTailCopy"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& tail_input = shader.AddInput("tail_input", ShaderUsage::UseValueTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_tail_copy.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(tail_input, tail_input));
  }
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"output_offset", ProgramUniformVariableDataType::Uint32});
};

Status ApplySubgroupMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                      const Tensor* zero_points, const Tensor* bias,
                                      uint32_t M,
                                      uint32_t N,
                                      uint32_t K,
                                      uint32_t nbits,
                                      uint32_t zero_blocks_per_col,
                                      int32_t config_index,
                                      onnxruntime::webgpu::ComputeContext& context,
                                      Tensor* y,
                                      const uint32_t weight_index,
                                      const Tensor* weight_index_indirect) {
  const bool has_zero_points = zero_points != nullptr;
  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;

  // Determine tile sizes first (needed for prepack padding).
  const auto& config = supported_subgroup_matrix_configs[config_index];
  const auto tiling = GetSubgroupMatrixMatMulNBitsTiling(config, has_bias, M, N);

  // If applicable, layout optimization of input matrix A(MxK) can be used for SubgroupMatrixLoad.
  Tensor a_prepack;
  if (config.needsPrepack) {
    const auto m = config.M;
    const auto k = config.K;

    // Optimize the layout of input matrix A(MxK) for SubgroupMatrixLoad.
    PrepackProgram prepack_program{m, k};
    prepack_program.SetWorkgroupSize(config.subgroupSize);
    if (context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
      prepack_program.SetSubgroupSize(config.subgroupSize);
    }

    // Pad M to workgroup tile size so all subgroups read valid prepacked data.
    const uint32_t padded_M = ((M + tiling.tile_m - 1) / tiling.tile_m) * tiling.tile_m;
    const auto dispatch_group_size_x = padded_M / m;
    ORT_ENFORCE(K % k == 0, "K must be a multiple of ", k);
    const auto dispatch_group_size_y = K / k;
    // Each workgroup will process one subgroup matrix of size m x k.
    prepack_program.SetDispatchGroupSize(dispatch_group_size_x, dispatch_group_size_y, 1);

    TensorShape a_prepack_shape{padded_M, K};
    a_prepack = context.CreateGPUTensor(a->DataType(), a_prepack_shape);
    prepack_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutputs({{&a_prepack, ProgramTensorMetadataDependency::Rank, a_prepack.Shape(), 1}})
        .AddUniformVariables({{M}, {K}})
        .CacheHint(m, k);
    ORT_RETURN_IF_ERROR(context.RunProgram(prepack_program));
    a = &a_prepack;
  }

  constexpr uint32_t kU32Components = 4;
  TensorShape y_shape{1, M, N};
  // For the no-bias 8x16x16 path, a trailing partial M-tile is written via the same
  // branch-free fast path as an in-bounds tile, but into a small [tile_m, N]
  // padded buffer instead of `y` (which is too short by this point); the crop-copy
  // dispatch below moves the valid rows into `y`. This keeps the no-bias write-out
  // free of any bounds-checked workgroup-scratch store.
  const bool has_tail_buffer = !has_bias && config.Is(8, 16, 16) && (M % tiling.tile_m != 0);
  SubgroupMatrixMatMulNBitsProgram mul_program{nbits, config_index, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, tiling.tile_m, tiling.tile_n, has_tail_buffer};
  mul_program.SetWorkgroupSize(tiling.workgroup_size);

  // Pin kernels running on variable-size adapters to the subgroup size they were written for.
  if (context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
    mul_program.SetSubgroupSize(config.subgroupSize);
  }

  uint32_t num_N_tile = CeilDiv(N, tiling.tile_n);
  uint32_t num_M_tile = CeilDiv(M, tiling.tile_m);
  mul_program.SetDispatchGroupSize(num_N_tile, num_M_tile, 1);

  // The 8x16x16 shader always reads input_b as Uint8x16 (vec4<u32>) chunks, regardless of nbits;
  // TODO: Apply vec4 to all configs.
  const int input_b_components = static_cast<int>(
      config.Is(8, 16, 16)
          ? kU32Components * 4
          : (nbits == 4 ? kU32Components : 2 * kU32Components));
  mul_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1},
                         {b, ProgramTensorMetadataDependency::TypeAndRank, input_b_components},
                         {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
      .AddUniformVariables({{M}, {N}, {K}, {zero_blocks_per_col}, {num_N_tile}, {num_M_tile}, {weight_index}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, y_shape, 1})
      .CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, tiling.tile_m, tiling.tile_n, has_tail_buffer);
  if (has_zero_points) {
    mul_program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (bias) {
    mul_program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    mul_program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }

  Tensor tail_buffer;
  if (has_tail_buffer) {
    tail_buffer = context.CreateGPUTensor(y->DataType(), TensorShape{tiling.tile_m, N});
    mul_program.AddOutput({&tail_buffer, ProgramTensorMetadataDependency::None});
  }
  ORT_RETURN_IF_ERROR(context.RunProgram(mul_program));

  if (has_tail_buffer) {
    // Only the rows below `tail_rows` in `tail_buffer` were ever written by the
    // main kernel (the tail tile is always the last M-tile, so its row offset is
    // fixed); crop-copy just those into the real output.
    constexpr uint32_t kTailCopyComponents = 4;
    ORT_ENFORCE(N % kTailCopyComponents == 0, "N must be a multiple of ", kTailCopyComponents);
    const uint32_t tail_rows = M - (num_M_tile - 1) * tiling.tile_m;
    const uint32_t row_offset = (num_M_tile - 1) * tiling.tile_m;
    const uint32_t n_vec4 = N / kTailCopyComponents;
    const uint32_t output_size = tail_rows * n_vec4;
    const uint32_t output_offset = row_offset * n_vec4;
    constexpr uint32_t kCopyWorkgroupSize = 256;
    SubgroupMatrixMatMulNBitsTailCopyProgram copy_program;
    copy_program.SetWorkgroupSize(kCopyWorkgroupSize);
    copy_program.SetDispatchGroupSize(CeilDiv(output_size, kCopyWorkgroupSize));

    copy_program.AddInput({&tail_buffer, ProgramTensorMetadataDependency::Type, kTailCopyComponents})
        .AddOutput({y, ProgramTensorMetadataDependency::Type, kTailCopyComponents})
        .AddUniformVariables({{output_size}, {output_offset}});
    ORT_RETURN_IF_ERROR(context.RunProgram(copy_program));
  }

  return Status::OK();
}

bool CanApplySubgroupMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                       uint64_t accuracy_level,
                                       uint32_t block_size,
                                       uint32_t batch_count,
                                       uint32_t N,
                                       uint32_t K,
                                       uint32_t nbits,
                                       bool is_fp16,
                                       int32_t& config_index,
                                       uint32_t M,
                                       bool has_weight_idx_indirect,
                                       bool has_bias) {
  // Subgroup matrix kernels only support 4-bit/8-bit quantization with block_size 32.
  if (!((nbits == 4 || nbits == 8) && block_size == 32)) {
    return false;
  }

  // TODO: Support batch.
  if (batch_count != 1) {
    return false;
  }

  // TODO: Clean up the weight_idx_indirect in shaders.
  if (has_weight_idx_indirect) {
    return false;
  }

  // Every fp16 config in supported_subgroup_matrix_configs has resultComponentType == F16, and
  // the kernels declare subgroup_matrix_result<f16, ...> to match, so the accumulation inside
  // subgroupMatrixMultiplyAccumulate is f16 and there is no variant of this kernel that can
  // honour an f32 accumulator request. Decline the path instead of ignoring the option: the
  // caller then falls through to a kernel that does honour it. Only fp16 outputs are affected;
  // the fp32 config accumulates in f32 already.
  if (is_fp16 && context.EnableMatmulFp32Accumulation()) {
    return false;
  }

  // On Apple, this kernel is only validated for accuracy level 4. Higher accuracy requires
  // a slower f32-compute variant.
  if (context.AdapterInfo().vendor == std::string_view{"apple"} && accuracy_level != 4) {
    return false;
  }

  const auto selected_config = SelectSubgroupMatrixConfig(
      context, is_fp16, {{16, 16, 16, 32}, {8, 16, 16, 32}, {8, 8, 8, 32}});
  if (!selected_config) {
    return false;
  }

  config_index = *selected_config;

  const auto& config = supported_subgroup_matrix_configs[config_index];
  const auto tiling = GetSubgroupMatrixMatMulNBitsTiling(config, has_bias, M, N);
  return tiling.SupportsShape(M, N, K);
}
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
