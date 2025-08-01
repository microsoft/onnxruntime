// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/bitlinear.h"

#include <string>
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    BitLinear,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    BitLinear);

Status BitLinearQuantizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("output_5th", ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("scales", ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "quantization/bitlinear_quantize.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(K4, K_ / 4),
                             WGSL_TEMPLATE_PARAMETER(K_PADDED_4, K_PADDED_ / 4));
}

Status BitLinearMultiplyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("input_a5", ShaderUsage::UseValueTypeAlias);
  shader.AddInput("scales_a", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("input_b", ShaderUsage::UseValueTypeAlias);
  shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "quantization/bitlinear_multiply.wgsl.template");
}

Status BitLinearMultiplySmallMProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddInput("input_a5", ShaderUsage::UseUniform);
  shader.AddInput("scales_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  ORT_ENFORCE(WorkgroupSizeX() % tile_size_k_vec_ == 0 && tile_size_k_vec_ % 4 == 0,
              "tile_size_k_vec_ must evenly divide workgroup size X and be divisible by 4");
  const uint32_t kSubtileCount = WorkgroupSizeX() / tile_size_k_vec_;
  ORT_ENFORCE(tile_size_ % kSubtileCount == 0, "tile_size_ must be divisible by sub_tile_count");

  // This algorithm processes K in chunks for efficient computation with BitLinear's ternary quantization
  // Each workgroup handles one row of matrix A and tile_size rows of matrix B
  // Uses the BitLinear-specific packing where 5 ternary weights are packed per uint8
  return WGSL_TEMPLATE_APPLY(shader, "quantization/bitlinear_multiply_small_m.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, kSubtileCount));
}

Status BitLinear::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* a = context.Input(0);
  const Tensor* b = context.Input(1);

  // Validate input shapes
  TensorShape b_shape({N_, K_});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  auto* y = context.Output(0, helper.OutputShape());

  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());

  // Validate input B shape more specifically
  const uint32_t kQuantizationBlockSize = 20;
  const uint32_t kWeightsPerByte = 5;
  // When K is not divisible by kQuantizationBlockSize, weights are padded to fit kQuantizationBlockSize.
  // During quantization of A we also pad the resulting output to K_PADDED to match the weights.
  const uint32_t K_PADDED = ((K + (kQuantizationBlockSize - 1)) / kQuantizationBlockSize) * kQuantizationBlockSize;
  TensorShape expected_b_shape({N, K_PADDED / kWeightsPerByte});
  ORT_ENFORCE(b->Shape() == expected_b_shape, "Unexpected input B shape", b->Shape().ToString());

  // Step 1: Quantize input A using BitLinearQuantizeProgram
  const uint32_t quantize_output_size = (M * (K_PADDED - (K_PADDED / kWeightsPerByte)) / 4);  // skipping every 5th, packed into u32
  const uint32_t quantize_5th_output_size = M * K_PADDED / kQuantizationBlockSize;   // every 5th element packed int u32

  TensorShape quantize_output_shape({quantize_output_size});
  TensorShape quantize_5th_output_shape({quantize_5th_output_size});
  TensorShape scales_output_shape({M});

  auto quantized_a = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), quantize_output_shape);
  auto quantized_a5 = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), quantize_5th_output_shape);
  auto scales_a = context.CreateGPUTensor(a->DataType(), scales_output_shape);
  constexpr uint32_t kVec4Components = 4;
  constexpr uint32_t kU32Components = 4;

  {
    BitLinearQuantizeProgram quantize_program(K, K_PADDED);
    quantize_program
        .SetWorkgroupSize(128)
        .SetDispatchGroupSize(M)
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)}})
        .AddOutputs({{&quantized_a, ProgramTensorMetadataDependency::None},
                     {&quantized_a5, ProgramTensorMetadataDependency::None},
                     {&scales_a, ProgramTensorMetadataDependency::None}})
        .CacheHint(K);
    ORT_RETURN_IF_ERROR(context.RunProgram(quantize_program));
  }

  // Step 2: Matrix multiplication using appropriate program based on M size
  const uint32_t min_M_for_tile_optimization = 32;  // Similar to DP4A implementation

  if (M < min_M_for_tile_optimization) {
    // Use small M optimized program for generation mode (small batch sizes)
    uint32_t tile_size_k_vec = 16;
    uint32_t tile_size = 32;

    if (context.AdapterInfo().vendor == std::string_view{"intel"}) {
      tile_size_k_vec = 32;
      tile_size = 4;
    }

    BitLinearMultiplySmallMProgram multiply_program(tile_size_k_vec, tile_size);
    uint32_t num_N_tile = (N + tile_size - 1) / tile_size;
    const uint32_t input_a_stride = (K_PADDED - (K_PADDED / 5)) / 16;

    multiply_program
        .SetWorkgroupSize(128)
        .SetDispatchGroupSize(M * num_N_tile)
        .AddInputs({{&quantized_a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)},
                    {&quantized_a5, ProgramTensorMetadataDependency::TypeAndRank},
                    {&scales_a, ProgramTensorMetadataDependency::TypeAndRank},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kU32Components)}})
        .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddUniformVariables({{M}, {N}, {K_PADDED}, {K_PADDED / 20}, {input_a_stride}, {scale_b_}, {num_N_tile}})
        .CacheHint(tile_size_k_vec, tile_size);
    ORT_RETURN_IF_ERROR(context.RunProgram(multiply_program));
  } else {
    // Use original tiled program for larger batch sizes
    BitLinearMultiplyProgram multiply_program;
    // input_a is vectorized as vec4<u32> which gives a packing of 16 elements per value.
    // Support for cases where (K_PADDED - (K_PADDED / 5)) is not divisible by 16, is not implemented.
    ORT_ENFORCE((K_PADDED - (K_PADDED / 5)) % 16 == 0, "K_PADDED must be divisible by 16 after skipping every 5th element. K_PADDED: ", K_PADDED);
    const uint32_t input_a_stride = (K_PADDED - (K_PADDED / 5)) / 16;
    constexpr uint32_t kTileSize = 64;
    TensorShape reshaped_y_shape{1, M, N / kVec4Components};
    uint32_t num_M_tile = (M + kTileSize - 1) / kTileSize;
    uint32_t num_N_tile = (N + kTileSize - 1) / kTileSize;
    multiply_program
        .SetWorkgroupSize(256)
        .SetDispatchGroupSize(num_M_tile * num_N_tile)
        .AddInputs({{&quantized_a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kVec4Components)},
                    {&quantized_a5, ProgramTensorMetadataDependency::TypeAndRank},
                    {&scales_a, ProgramTensorMetadataDependency::TypeAndRank},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(kU32Components)}})
        .AddOutputs({{y, ProgramTensorMetadataDependency::TypeAndRank, reshaped_y_shape, static_cast<int>(kVec4Components)}})
        .AddUniformVariables({{M}, {N}, {K_PADDED}, {input_a_stride}, {scale_b_}, {num_N_tile}});
    ORT_RETURN_IF_ERROR(context.RunProgram(multiply_program));
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
