// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status DP4AMatMulQuantizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("scales", ShaderUsage::UseUniform);
  return WGSL_TEMPLATE_APPLY(shader, "quantization/dp4a_quantize.wgsl.template");
}

Status DP4AMatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("scales_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddInput("scales_b", ShaderUsage::UseUniform);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  return WGSL_TEMPLATE_APPLY(shader, "quantization/dp4a_matmul.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(block_size, block_size_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, true));
}

// scale_A components = 1, b components = 4, output components = 1
Status DP4AMatMulNBitsSmallMProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddInput("scales_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddInput("scales_b", ShaderUsage::UseUniform);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  ORT_ENFORCE(WorkgroupSizeX() % tile_size_k_vec_ == 0 && tile_size_k_vec_ % 4 == 0, "tile_size_k_vec_ must evenly divide workgroup size X and be divisible by 4");
  const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec_;
  ORT_ENFORCE(tile_size_ % sub_tile_count == 0, "tile_size_ must be divisible by sub_tile_count");

  return WGSL_TEMPLATE_APPLY(shader, "quantization/dp4a_matmul_small_m.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, true),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec_));
}

Status ApplyDP4AMatrixMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                  const Tensor* zero_points,
                                  uint32_t M,
                                  uint32_t N,
                                  uint32_t K,
                                  uint32_t block_size,
                                  uint32_t zero_blocks_per_col,
                                  uint32_t min_M_for_tile_optimization,
                                  uint32_t nbits,
                                  onnxruntime::webgpu::ComputeContext& context,
                                  Tensor* y) {
  constexpr uint32_t kVec4Components = 4;
  constexpr uint32_t kVec2Components = 2;
  constexpr uint32_t kU32Components = 4;

  constexpr uint32_t kBlockSizeA = 128;
  DP4AMatMulQuantizeProgram quantize_program;
  quantize_program.SetWorkgroupSize(64);
  uint32_t tile_size = 64 * kVec4Components;
  quantize_program.SetDispatchGroupSize((M * K + tile_size - 1) / tile_size, 1, 1);
  TensorShape a_quant_shape{1, M, K / kU32Components};
  Tensor a_quant = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), a_quant_shape);
  TensorShapeVector a_scales_dims({1, 1, M, K / kBlockSizeA});
  Tensor a_scale = context.CreateGPUTensor(a->DataType(), a_scales_dims);
  quantize_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)}})
      .AddOutputs({{&a_quant, ProgramTensorMetadataDependency::Rank, a_quant.Shape(), 1},
                   {&a_scale, ProgramTensorMetadataDependency::Rank, 1}})
      .AddUniformVariable({M * K / kU32Components});
  ORT_RETURN_IF_ERROR(context.RunProgram(quantize_program));
  const bool has_zero_points = zero_points != nullptr;
  if (M < min_M_for_tile_optimization) {
    uint32_t tile_size_k_vec = 16;
    uint32_t tile_size_n = 32;

    if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
      tile_size_k_vec = 32;
      tile_size_n = 4;
    }

    DP4AMatMulNBitsSmallMProgram mul_program{tile_size_k_vec, tile_size_n, nbits, has_zero_points};
    uint32_t num_N_tile = (N + tile_size_n - 1) / tile_size_n;
    mul_program.SetWorkgroupSize(128);
    mul_program.SetDispatchGroupSize(M * num_N_tile);
    mul_program.AddInputs({{&a_quant, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)},
                           {&a_scale, ProgramTensorMetadataDependency::TypeAndRank, 1},
                           {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components * kU32Components)},
                           {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddUniformVariables({M, N, K, K / 16, K / 32, block_size, num_N_tile, zero_blocks_per_col})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, 1})
        .CacheHint(nbits, tile_size_k_vec, tile_size_n, has_zero_points);
    if (has_zero_points) {
      mul_program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
    }
    return context.RunProgram(mul_program);
  }

  constexpr uint32_t kTileSize = 64;
  TensorShape reshaped_y_shape{1, M, N / kVec4Components};
  uint32_t num_M_tile = (M + kTileSize - 1) / kTileSize;
  uint32_t num_N_tile = (N + kTileSize - 1) / kTileSize;
  DP4AMatMulNBitsProgram mul_program{block_size, nbits, has_zero_points};
  mul_program.SetWorkgroupSize(256);
  mul_program.SetDispatchGroupSize(num_M_tile * num_N_tile);
  mul_program.AddInputs({{&a_quant, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)},
                         {&a_scale, ProgramTensorMetadataDependency::TypeAndRank, 1},
                         {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(nbits == 4 ? kVec2Components * kU32Components : kVec4Components * kU32Components)},
                         {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
      .AddUniformVariables({{static_cast<uint32_t>(M)},
                            {static_cast<uint32_t>(N)},
                            {static_cast<uint32_t>(K)},
                            {static_cast<uint32_t>(K / 8)},
                            {static_cast<uint32_t>(K / 16)},
                            {num_N_tile},
                            {zero_blocks_per_col}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, static_cast<int>(kVec4Components)})
      .CacheHint("Block" + std::to_string(block_size), nbits, has_zero_points);
  if (has_zero_points) {
    mul_program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  return context.RunProgram(mul_program);
}

bool CanApplyDP4AMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                   uint64_t accuracy_level,
                                   uint32_t block_size,
                                   uint32_t batch_count,
                                   uint32_t N,
                                   uint32_t K,
                                   uint32_t components_k) {
  // macOS - Avoid using dp4a on Metal, as it does not appear to have native dp4a support.
  // https://github.com/gpuweb/gpuweb/issues/2677#issuecomment-1713292226
  // Use 'vendor' to check for metal; 'backend' is always WEBGPU when running under wasm.
  bool use_dp4a = context.HasFeature(wgpu::FeatureName::Subgroups) &&
                  context.AdapterInfo().vendor != std::string_view{"apple"};
  return (accuracy_level == 4 && block_size % 32 == 0 &&
          batch_count == 1 && components_k == 4 && K % 128 == 0 && N % 16 == 0 &&
          use_dp4a);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
