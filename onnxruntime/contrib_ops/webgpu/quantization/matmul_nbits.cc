// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string_view>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_qlora.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(MatMulNBits, kMSDomain, 1, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T1", WebGpuSupportedFloatTypes())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
                            .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
                            .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
                        MatMulNBits);

Status MatMulNBitsWideTileProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  if (has_qlora_up_) {
    shader.AddInput("lora_t", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("lora_b", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("lora_b_scales");
    GenerateQLoRAUp(shader, acc_f32_);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  const uint32_t workgroup_size = WorkgroupSizeX() * WorkgroupSizeY() * WorkgroupSizeZ();
  ORT_ENFORCE(tile_n_ == workgroup_size, "tile_n must be workgroup_size.");
  ORT_ENFORCE(tile_m_ % (workgroup_size / 8) == 0, "tile_m must be a multiple of workgroup_size / 8.");
  ORT_ENFORCE(nbits_ == 4 || nbits_ == 8, "Only 4/8 bits are supported for webgpu matmulnbits.");

  return WGSL_TEMPLATE_APPLY(
      shader, "quantization/matmul_nbits_wide_tile.wgsl.template", WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32_),
      WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_), WGSL_TEMPLATE_PARAMETER(has_qlora_up, has_qlora_up_),
      WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
      WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
      WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_), WGSL_TEMPLATE_PARAMETER(nbits, nbits_),
      WGSL_TEMPLATE_PARAMETER(subgroup_min_size, subgroup_min_size_), WGSL_TEMPLATE_PARAMETER(tile_m, tile_m_),
      WGSL_TEMPLATE_PARAMETER(tile_n, tile_n_), WGSL_TEMPLATE_VARIABLE(a, a), WGSL_TEMPLATE_VARIABLE(b, b),
      WGSL_TEMPLATE_VARIABLE(output, output), WGSL_TEMPLATE_VARIABLE(scales, scales));
}

// Apply similar idea with DP4AMatMulNBitsKParallelProgram algorithm.
Status MatMulNBitsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b");
  const auto& scales_b = shader.AddInput("scales_b");
  if (has_zero_points_) {
    shader.AddInput("zero_points", ShaderUsage::UseUniform);
  }
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  if (has_qlora_up_) {
    shader.AddInput("lora_t", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("lora_b", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("lora_b_scales");
    GenerateQLoRAUp(shader, acc_f32_);
  }
  if (has_qlora_down_) {
    shader.AddInput("lora_at");
    shader.AddInput("lora_a_scales");
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
  if (has_qlora_down_) {
    shader.AddOutput("lora_down", ShaderUsage::UseElementTypeAlias);
    GenerateQLoRADown(shader, "uniforms.num_N_tile", a.GetByOffset("k4"), acc_f32_, qlora_down_subgroups_);
  }

  const uint32_t components_a = a.NumComponents();
  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includes 4 uint8.
  const uint32_t tile_size_k_vec = tile_size_k_vec_;
  const uint32_t elements_in_value_b = components_b * (32 / nbits_);
  const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
  const uint32_t a_length_per_tile = tile_size_k / components_a;
  uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

  return WGSL_TEMPLATE_APPLY(
      shader, "quantization/matmul_nbits.wgsl.template", WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
      WGSL_TEMPLATE_PARAMETER(acc_f32, acc_f32_), WGSL_TEMPLATE_PARAMETER(broadcast_a_row, broadcast_a_row_),
      WGSL_TEMPLATE_PARAMETER(component_a, components_a), WGSL_TEMPLATE_PARAMETER(component_b, components_b),
      WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b), WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
      WGSL_TEMPLATE_PARAMETER(has_qlora_up, has_qlora_up_), WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
      WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
      WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_), WGSL_TEMPLATE_PARAMETER(n_bits, nbits_),
      WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
      WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
      WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count), WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
      WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k), WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
      WGSL_TEMPLATE_VARIABLE(a, a), WGSL_TEMPLATE_VARIABLE(b, b), WGSL_TEMPLATE_VARIABLE(output, output),
      WGSL_TEMPLATE_VARIABLE(scales_b, scales_b));
}

Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* a = context.Input(0);
  const Tensor* b = context.Input(1);
  const Tensor* scales = context.Input(2);
  const Tensor* zero_points = context.Input(3);
  const Tensor* g_idx = context.Input(4);
  const Tensor* bias = context.Input(5);

  ORT_ENFORCE(g_idx == nullptr, "group_idx as input is not supported yet.");

  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(),
                "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  auto output_shape = helper.OutputShape();
  Tensor* y = context.Output(0, output_shape);
  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  return ApplyMatMulNBits(a, b, scales, zero_points, bias, K_, N_, block_size_, accuracy_level_, bits_, context, y, 0);
}

/**
 * @brief Applies a quantized matrix multiplication using N-bit precision.
 *
 * This function computes the matrix multiplication of the quantized tensor inputs with multiple
 * optional optimizations tailored to the GPU backend. Depending on the provided parameters and GPU
 * capabilities, it selects one of several optimized kernels (such as subgroup matrix multiplication,
 * DP4A, wide tile programs, or the default matmul program) to perform the computation.
 * It can be called by the MatMulNBits operator or directly for custom scenarios like QMoe.
 *
 * @param a              Pointer to the left-hand side (activation) tensor.
 * @param b              Pointer to the quantized weight tensor.
 *                       b has the shape (N, k_blocks, blob_size) or (weight_batch, N, k_blocks, blob_size)
 * @param scales         Pointer to the tensor containing scaling factors for quantization.
 *                       scales has the shape (N) or (weight_batch, N)
 * @param zero_points    Pointer to the zero-point tensor for quantization; must be of type uint8 if provided.
 *                       weight_index > 0 is only supported when zero_points is nullptr.
 * @param bias           Pointer to the bias tensor; optional.
 * @param K_op           The K dimension of the operation (number of columns in 'a' and rows in 'b' before
 * quantization).
 * @param N_op           The N dimension of the operation (number of columns in 'b').
 * @param block_size_op  The block size used for quantization partitioning.
 * @param accuracy_level Accuracy level influencing the choice of optimized kernel.
 * @param nbits          Number of bits used for quantization.
 * @param weight_index   Index of the weight matrix in case of stacked weights; defaults to 0.
 * @param context        Compute context for WebGPU, providing device-specific information and execution facilities.
 * @param y              Pointer to the output tensor that will hold the result.
 *
 * @return Status indicating whether the operation was successful or if an error occurred.
 *
 * @note Special optimizations are considered:
 *       - Subgroup matrix multiplication for GPUs with supported configs.
 *       - DP4A-based multiplication on FP32-only GPUs for specific dimensions and conditions.
 *       - A wide tile program is used when block size, component count, and other criteria are met.
 *       - Otherwise, a default matmul program is used.
 */
Status ApplyMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points,
                        const Tensor* bias, int64_t K_op, int64_t N_op, int64_t block_size_op, int64_t accuracy_level,
                        int64_t nbits, onnxruntime::webgpu::ComputeContext& context, Tensor* y,
                        const uint32_t weight_index, const Tensor* weight_index_indirect, uint32_t override_M,
                        const Tensor* lora_t, const Tensor* lora_b, const Tensor* lora_a_transposed,
                        Tensor* lora_a_output, const Tensor* lora_b_scales, const Tensor* lora_a_scales,
                        DP4AQuantizedInput* quantized_input) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const bool has_qlora_up = lora_t != nullptr;
  const bool has_qlora_down = lora_a_transposed != nullptr;
  ORT_RETURN_IF_NOT(!has_qlora_down || (lora_a_output != nullptr && lora_a_scales != nullptr && !has_qlora_up &&
                                        a->Shape().Size() == K_op && nbits == 4 && block_size_op == 32),
                    "Unsupported joint QLoRA projection.");
  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;
  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(),
                "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t dispatch_M = (override_M > 0) ? override_M : M;
  const bool broadcast_a = dispatch_M > M;
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);

  // Special case matrix used by bitnets where there is a single scale for the entire
  const bool single_scale_weights = (block_size == K * N);
  const uint32_t block_size_per_col = single_scale_weights ? K : block_size;
  const uint32_t n_blocks_per_col = (K + block_size_per_col - 1) / block_size_per_col;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  uint32_t components = GetMaxComponents(N);
  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)].
  // The shader uses a flat linear index to address individual n-bit zero point values.
  // Since each column's zero points are byte-aligned in the packed buffer, we must round
  // n_blocks_per_col up to the next multiple of (8/nbits) — the number of zero point
  // values per byte — so that the linear stride correctly skips byte-boundary padding.
  const uint32_t zp_elements_per_byte = 8 / static_cast<uint32_t>(nbits);
  uint32_t zero_blocks_per_col =
      (n_blocks_per_col + zp_elements_per_byte - 1) / zp_elements_per_byte * zp_elements_per_byte;

  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  std::optional<SubgroupMatrixConfig> subgroup_matrix_config;
  if (!has_qlora_down && CanApplySubgroupMatrixMatMulNBits(context, accuracy_level, block_size, batch_count, N, K,
                                                           static_cast<uint32_t>(nbits),
                                                           y->DataType() == DataTypeImpl::GetType<MLFloat16>(),
                                                           subgroup_matrix_config, M, has_weight_idx_indirect)) {
    if (has_qlora_up) {
      auto base_output = context.CreateGPUTensor(y->DataType(), y->Shape());
      ORT_RETURN_IF_ERROR(ApplySubgroupMatrixMatMulNBits(
          a, b, scales, zero_points, bias, M, N, K, static_cast<uint32_t>(nbits), zero_blocks_per_col,
          *subgroup_matrix_config, context, &base_output, weight_index, weight_index_indirect));
      return ApplyQLoRAOutput(context, &base_output, lora_t, lora_b, lora_b_scales, y);
    }
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K, static_cast<uint32_t>(nbits),
                                          zero_blocks_per_col, *subgroup_matrix_config, context, y, weight_index,
                                          weight_index_indirect);
  }

  // On FP32 only GPUs and Qualcomm GPUs, integer math is faster than FP32 therefore always use DP4A independent of
  // length of M. DP4A Q2 path now supports custom zero points via a 1024-entry LUT (4 zero-point sections × 256 byte
  // values).
  if (!has_qlora_down && CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a, M,
                                                       has_weight_idx_indirect, y)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, zero_points, bias, batch_count, M, dispatch_M, N, K, block_size,
                                      zero_blocks_per_col, kMinMForTileOptimization, static_cast<uint32_t>(nbits),
                                      context, y, weight_index, weight_index_indirect, lora_t, lora_b, nullptr, nullptr,
                                      lora_b_scales, quantized_input);
  }

  // WideTileProgram
  // This program is optimized for Block32 prefill.
  const bool use_wide_tile_program =
      !has_qlora_down &&
      CanApplyWideTileMatMulNBits(M, K, block_size, nbits, has_weight_idx_indirect, components_a, components_b);

  const bool acc_f32 = context.EnableMatmulFp32Accumulation();

  if (use_wide_tile_program) {
    // Enforce output components to 1.
    components = 1;

    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_n = workgroup_size;
    const bool is_f16 = a->DataType() == DataTypeImpl::GetType<MLFloat16>();
    const uint32_t tile_m = is_f16 ? 2 * workgroup_size / 8 : workgroup_size / 8;
    const uint32_t num_N_tile = CeilDiv(N, tile_n);
    const uint32_t num_M_tile = CeilDiv(dispatch_M, tile_m);

    bool is_nvidia = context.AdapterInfo().vendor == std::string_view{"nvidia"};

    // The template shuffles tile_m lanes in bands of subgroup_min_size when that's a
    // width it supports; otherwise (including when Subgroups isn't supported) it falls
    // back to a direct reduction.
    // NOTE: NVIDIA GPUs prefer direct reduction.
    const uint32_t subgroup_min_size =
        (context.HasFeature(wgpu::FeatureName::Subgroups) && !is_nvidia) ? context.AdapterInfo().subgroupMinSize : 0u;

    MatMulNBitsWideTileProgram program{has_zero_points,
                                       has_bias,
                                       has_weight_idx,
                                       has_weight_idx_indirect,
                                       tile_m,
                                       tile_n,
                                       static_cast<uint32_t>(nbits),
                                       subgroup_min_size,
                                       acc_f32,
                                       has_qlora_up};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, num_M_tile, batch_count);

    constexpr uint32_t kU32Components = 4;
    const uint32_t components_b_with_u32 = components_b * kU32Components;
    const uint32_t K_of_b = n_blocks_per_col * blob_size / components_b_with_u32;
    const uint32_t K_of_a = K / components_a;

    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, onnxruntime::narrow<int>(components_a)});
    program.AddInput(
        {b, ProgramTensorMetadataDependency::TypeAndRank, onnxruntime::narrow<int>(components_b_with_u32)});
    program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_zero_points) {
      program.AddInput({zero_points,
                        ProgramTensorMetadataDependency::TypeAndRank,
                        {CeilDiv(zero_points->Shape().Size(), static_cast<int64_t>(4))},
                        4});
    }
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    if (has_weight_idx_indirect) {
      program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
    }
    program.AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, onnxruntime::narrow<int>(components)});
    program.AddUniformVariables({{batch_count},
                                 {dispatch_M},
                                 {N},
                                 {K_of_a},
                                 {K_of_b},
                                 {n_blocks_per_col},
                                 {zero_blocks_per_col},
                                 {num_N_tile},
                                 {num_M_tile},
                                 {weight_index}});
    program.CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, subgroup_min_size,
                      acc_f32, has_qlora_up);

    if (has_qlora_up) {
      program.AddInput({lora_t, ProgramTensorMetadataDependency::Type, 4});
      program.AddInput({lora_b, ProgramTensorMetadataDependency::Type, 16});
      program.AddInput({lora_b_scales, ProgramTensorMetadataDependency::Type});
    }
    return context.RunProgram(program);
  }

  // Use tile_size_k_vec=32 by default for better K-dimension parallelism.
  // Intel devices use 16 as they have different subgroup/cache characteristics.
  const uint32_t tile_size_k_vec = (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;

  constexpr uint32_t workgroup_size = 128;
  constexpr uint32_t tile_size = 8;
  constexpr uint32_t kU32Components = 4;
  uint32_t components_b_with_u32 = components_b * kU32Components;
  uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;
  // The decode reduction assigns two ranks to each 32-lane subgroup. Devices
  // with a variable subgroup range must explicitly request that width.
  const auto& adapter = context.AdapterInfo();
  const bool fixed_subgroup_32 = adapter.subgroupMinSize == 32 && adapter.subgroupMaxSize == 32;
  const bool request_subgroup_32 = has_qlora_down && !fixed_subgroup_32 &&
                                   context.HasFeature(wgpu::FeatureName::SubgroupSizeControl) &&
                                   adapter.subgroupMinSize <= 32 && adapter.subgroupMaxSize >= 32;
  const bool qlora_down_subgroups = has_qlora_down && context.HasFeature(wgpu::FeatureName::Subgroups) &&
                                    context.HasWGSLLanguageFeature(wgpu::WGSLLanguageFeatureName::SubgroupId) &&
                                    (fixed_subgroup_32 || request_subgroup_32);
  MatMulNBitsProgram program{tile_size,
                             static_cast<uint32_t>(nbits),
                             has_zero_points,
                             has_bias,
                             has_weight_idx,
                             has_weight_idx_indirect,
                             single_scale_weights,
                             tile_size_k_vec,
                             broadcast_a,
                             acc_f32,
                             has_qlora_up,
                             has_qlora_down,
                             qlora_down_subgroups};
  program.SetUseSubgroupId(qlora_down_subgroups);
  program.SetWorkgroupSize(workgroup_size);
  if (qlora_down_subgroups && request_subgroup_32) {
    program.SetSubgroupSize(32);
  }
  uint32_t num_N_tile = (N + tile_size - 1) / tile_size;
  program.SetDispatchGroupSize(num_N_tile + (has_qlora_down ? 4 : 0), dispatch_M, batch_count);
  program
      .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)},
                  {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({{M},
                            {N},
                            {K},
                            {K / components_a},
                            {K_of_b},
                            {block_size},
                            {n_blocks_per_col},
                            {zero_blocks_per_col},
                            {num_N_tile},
                            {batch_count},
                            {weight_index},
                            {dispatch_M}})
      .CacheHint(nbits, has_zero_points, single_scale_weights, has_bias, has_weight_idx, has_weight_idx_indirect,
                 tile_size_k_vec, broadcast_a, acc_f32, has_qlora_up, has_qlora_down, qlora_down_subgroups);
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }
  if (has_qlora_up) {
    program.AddInput({lora_t, ProgramTensorMetadataDependency::Type, 4});
    program.AddInput({lora_b, ProgramTensorMetadataDependency::Type, 16});
    program.AddInput({lora_b_scales, ProgramTensorMetadataDependency::Type});
  }
  if (has_qlora_down) {
    ORT_RETURN_IF_NOT(components_a == 4, "Joint QLoRA projection requires vec4 X.");
    program.AddInput({lora_a_transposed, ProgramTensorMetadataDependency::Type, 16});
    program.AddInput({lora_a_scales, ProgramTensorMetadataDependency::Type});
    program.AddOutput({lora_a_output, ProgramTensorMetadataDependency::Type});
  }
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
