// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string_view>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
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

namespace {

constexpr unsigned int kMinMForTileOptimization = 4;

template <typename T>
inline T ceil_div(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
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
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  const uint32_t workgroup_size = WorkgroupSizeX() * WorkgroupSizeY();
  ORT_ENFORCE(tile_m_ == workgroup_size / 8, "tile_m must be workgroup_size / 8.");
  ORT_ENFORCE(tile_n_ == workgroup_size, "tile_n must be workgroup_size.");
  ORT_ENFORCE(nbits_ == 4 || nbits_ == 8, "Only 4/8 bits are supported for webgpu matmulnbits.");

  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_wide_tile.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(nbits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(tile_m, tile_m_),
                             WGSL_TEMPLATE_PARAMETER(tile_n, tile_n_),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(scales, scales));
}

// Apply similar idea with DP4AMatMulNBitsSmallMProgram algorithm.
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
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

  const uint32_t components_a = a.NumComponents();
  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includes 4 uint8.
  constexpr uint32_t tile_size_k_vec = 16;
  const uint32_t elements_in_value_b = components_b * (32 / nbits_);
  const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
  const uint32_t a_length_per_tile = tile_size_k / components_a;
  uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                             WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                             WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                             WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_zero_points, has_zero_points_),
                             WGSL_TEMPLATE_PARAMETER(n_bits, nbits_),
                             WGSL_TEMPLATE_PARAMETER(output_type_i32, false),
                             WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                             WGSL_TEMPLATE_VARIABLE(a, a),
                             WGSL_TEMPLATE_VARIABLE(b, b),
                             WGSL_TEMPLATE_VARIABLE(output, output),
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
  const uint32_t nbits = onnxruntime::narrow<uint32_t>(bits_);
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
    ORT_ENFORCE(nbits != 2, "Currently, zero points are not supported for Q2 quantization.");
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
 * @param scales         Pointer to the tensor containing scaling factors for quantization.
 * @param zero_points    Pointer to the zero-point tensor for quantization; must be of type uint8 if provided.
 * @param bias           Pointer to the bias tensor; optional.
 * @param K_op           The K dimension of the operation (number of columns in 'a' and rows in 'b' before quantization).
 * @param N_op           The N dimension of the operation (number of columns in 'b').
 * @param block_size_op  The block size used for quantization partitioning.
 * @param accuracy_level Accuracy level influencing the choice of optimized kernel.
 * @param nbits          Number of bits used for quantization.
 * @param context        Compute context for WebGPU, providing device-specific information and execution facilities.
 * @param y              Pointer to the output tensor that will hold the result.
 * @param offsets        Pointer to an optional offsets tensor; may be nullptr if not used.
 *
 * @return Status indicating whether the operation was successful or if an error occurred.
 *
 * @note Use of offsets: this allows you to use stacked weights, input and outputs. If provided, offset is a
 *       on device tensor of shape [k, 4]. It contains k entries of vec4<uint32>. Entry.x is the offset of input,
 *       entry.y is the offset of weights and entry.z is the offset of output.
 *       For example: you want to multiply the same input a with 2 different weights (that are stacked) you'd pass
 *        [vec4<uint32>(0, 0, 0), vec4<uint32>(0, 1, 1)] and the output will contain 2 stacked results.
 *
 * @note Special optimizations are considered:
 *       - Subgroup matrix multiplication for eligible Apple/Intel GPUs without offsets or bias.
 *       - DP4A-based multiplication on FP32-only GPUs for specific dimensions and conditions.
 *       - A wide tile program is used when block size, component count, and other criteria are met.
 *       - Otherwise, a default matmul program is used.
 */
Status ApplyMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales, const Tensor* zero_points, const Tensor* bias,
                        int64_t K_op,
                        int64_t N_op,
                        int64_t block_size_op,
                        int64_t accuracy_level,
                        int64_t nbits,
                        onnxruntime::webgpu::ComputeContext& context,
                        Tensor* y,
                        const uint32_t weigth_offset) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const bool has_bias = bias != nullptr;
  const bool has_zero_points = zero_points != nullptr;
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
  }

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
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
  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)]. So here we need to check whether n_blocks_per_col is divisible by 8/nbits.
  // For bits==4, this is counted by elements of uint4. Need add 1 if not divisible by 2.
  uint32_t zero_blocks_per_col = n_blocks_per_col % (8 / nbits) == 0 ? n_blocks_per_col : n_blocks_per_col + 1;

#if !defined(__wasm__)
  int32_t subgroup_matrix_config_index = -1;
  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  if (M >= kMinMForTileOptimization && (context.AdapterInfo().vendor == std::string_view{"apple"} || context.AdapterInfo().vendor == std::string_view{"intel"}) &&
      CanApplySubgroupMatrixMatMulNBits(context, accuracy_level, block_size, batch_count, N, K, subgroup_matrix_config_index)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K, static_cast<uint32_t>(nbits), zero_blocks_per_col, subgroup_matrix_config_index, context, y, weigth_offset);
  }
#endif

  // On FP32 only GPUs, integer math is faster than FP32 therefore always use DP4A independent of length of M.
  if ((M >= kMinMForTileOptimization || y->DataType() == DataTypeImpl::GetType<float>() || context.AdapterInfo().vendor == std::string_view{"qualcomm"}) &&
      CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, batch_count, N, K, components_a)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K, block_size, zero_blocks_per_col, kMinMForTileOptimization, static_cast<uint32_t>(nbits), context, y, weigth_offset);
  }

  // WideTileProgram
  // This program is optimized for Block32 prefill using Tile16x128.
  const bool use_wide_tile_program = block_size == 32 &&
                                     components_a == 4 &&
                                     components_b == 4 &&
                                     nbits != 2 &&
                                     M >= kMinMForTileOptimization;

  if (use_wide_tile_program) {
    // Enforce output components to 1.
    components = 1;

    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_m = workgroup_size / 8;
    constexpr uint32_t tile_n = workgroup_size;
    const uint32_t num_N_tile = ceil_div(N, tile_n);
    const uint32_t num_M_tile = ceil_div(M, tile_m);

    MatMulNBitsWideTileProgram program{has_zero_points, has_bias, tile_m, tile_n, static_cast<uint32_t>(nbits)};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, num_M_tile, batch_count);

    constexpr uint32_t kU32Components = 4;
    const uint32_t components_b_with_u32 = components_b * kU32Components;
    const uint32_t K_of_b = n_blocks_per_col * blob_size / components_b_with_u32;
    const uint32_t K_of_a = K / components_a;

    program.AddInput({a,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_a)});
    program.AddInput({b,
                      ProgramTensorMetadataDependency::TypeAndRank,
                      onnxruntime::narrow<int>(components_b_with_u32)});
    program.AddInput({scales, ProgramTensorMetadataDependency::TypeAndRank});
    if (has_zero_points) {
      program.AddInput({zero_points,
                        ProgramTensorMetadataDependency::TypeAndRank,
                        {ceil_div(zero_points->Shape().Size(), static_cast<int64_t>(4))},
                        4});
    }
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
    }
    program.AddOutput({y,
                       ProgramTensorMetadataDependency::TypeAndRank,
                       onnxruntime::narrow<int>(components)});
    program.AddUniformVariables({{batch_count},
                                 {M},
                                 {N},
                                 {K_of_a},
                                 {K_of_b},
                                 {n_blocks_per_col},
                                 {zero_blocks_per_col},
                                 {num_N_tile},
                                 {num_M_tile},
                                 {weigth_offset}});
    program.CacheHint(nbits, has_zero_points, has_bias);

    return context.RunProgram(program);
  }

  constexpr uint32_t workgroup_size = 128;
  constexpr uint32_t tile_size = 8;
  constexpr uint32_t kU32Components = 4;
  uint32_t components_b_with_u32 = components_b * kU32Components;
  uint32_t num_N_tile = (N + tile_size - 1) / tile_size;
  uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;
  MatMulNBitsProgram program{tile_size, static_cast<uint32_t>(nbits), has_zero_points, has_bias, single_scale_weights};
  program.SetWorkgroupSize(workgroup_size);
  program.SetDispatchGroupSize((N + tile_size - 1) / tile_size, M, batch_count);
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
                            {weigth_offset}})
      .CacheHint(nbits, has_zero_points, single_scale_weights, has_bias);
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
