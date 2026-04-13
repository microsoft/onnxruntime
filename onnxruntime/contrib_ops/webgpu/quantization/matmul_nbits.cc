// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <string_view>
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <mutex>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/macros.h"
#include "core/platform/env.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {
constexpr unsigned int kMinMForTileOptimization = 4;
constexpr uint32_t kMatMulNBitsMinNForAutoTuning = 65536;
constexpr const char* kMatMulNBitsAutoTunerEnvVar = "ORT_WEBGPU_MATMUL_NBITS_ENABLE_AUTO_TUNER";
constexpr double kMatMulNBitsTuneMinImprovementRatio = 0.10;

struct MatMulNBitsTuneParams {
  uint32_t tile_size_k_vec;
  uint32_t workgroup_size;
  uint32_t tile_size;
};

std::mutex g_matmul_nbits_tune_mutex;
InlinedHashMap<std::string, MatMulNBitsTuneParams> g_matmul_nbits_tuned_params;

constexpr std::array<MatMulNBitsTuneParams, 12> kMatMulNBitsTuneCandidates{{
    {8u, 64u, 8u},
    {8u, 128u, 8u},
    {8u, 64u, 16u},
    {8u, 128u, 16u},
    {16u, 64u, 8u},
    {16u, 128u, 8u},
    {16u, 64u, 16u},
    {16u, 128u, 16u},
    {32u, 64u, 8u},
    {32u, 128u, 8u},
    {32u, 64u, 16u},
    {32u, 128u, 16u},
}};

constexpr int kMatMulNBitsTuneWarmupRuns = 1;
constexpr int kMatMulNBitsTuneMeasuredRuns = 2;

std::string_view ToStdStringView(wgpu::StringView value) {
  return std::string_view{value.data ? value.data : "", value.length};
}

std::string MakeMatMulNBitsTuneKey(const onnxruntime::webgpu::ComputeContext& context,
                                   uint32_t M,
                                   uint32_t N,
                                   uint32_t K,
                                   uint32_t block_size,
                                   uint32_t nbits,
                                   bool single_scale_weights,
                                   bool is_fp16) {
  return MakeStringWithClassicLocale(ToStdStringView(context.AdapterInfo().vendor),
                                     "|",
                                     ToStdStringView(context.AdapterInfo().architecture),
                                     "|",
                                     ToStdStringView(context.AdapterInfo().device),
                                     "|M=",
                                     M,
                                     "|N=",
                                     N,
                                     "|K=",
                                     K,
                                     "|block=",
                                     block_size,
                                     "|bits=",
                                     nbits,
                                     "|single_scale=",
                                     single_scale_weights ? 1 : 0,
                                     "|fp16=",
                                     is_fp16 ? 1 : 0);
}

bool IsValidTuneCandidate(const onnxruntime::webgpu::ComputeContext& context,
                          const MatMulNBitsTuneParams& candidate) {
  if (candidate.workgroup_size % candidate.tile_size_k_vec != 0 ||
      candidate.workgroup_size > context.DeviceLimits().maxComputeInvocationsPerWorkgroup) {
    return false;
  }

  const uint32_t sub_tile_count = candidate.workgroup_size / candidate.tile_size_k_vec;
  return sub_tile_count > 0 && candidate.tile_size % sub_tile_count == 0;
}

MatMulNBitsTuneParams GetDefaultMatMulNBitsTuneParams(const onnxruntime::webgpu::ComputeContext& context) {
  return MatMulNBitsTuneParams{
      (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u,
      128u,
      8u,
  };
}

bool ShouldTuneDefaultMatMulNBitsProgram(const onnxruntime::webgpu::ComputeContext& context,
                                         uint32_t batch_count,
                                         uint32_t M,
                                         uint32_t N,
                                         bool has_zero_points,
                                         bool has_bias,
                                         bool has_weight_idx,
                                         bool has_weight_idx_indirect) {
  std::string auto_tuner_env = Env::Default().GetEnvironmentVar(kMatMulNBitsAutoTunerEnvVar);
  if (auto_tuner_env.empty()) {
    return false;
  }

  std::transform(auto_tuner_env.begin(), auto_tuner_env.end(), auto_tuner_env.begin(),
                 [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
  if (auto_tuner_env == "0" || auto_tuner_env == "false" || auto_tuner_env == "off") {
    return false;
  }

  return !context.IsGraphCaptureEnabled() &&
         batch_count == 1 &&
         M == 1 &&
         N >= kMatMulNBitsMinNForAutoTuning &&
         !has_zero_points &&
         !has_bias &&
         !has_weight_idx &&
         !has_weight_idx_indirect;
}

Status RunDefaultMatMulNBitsProgram(const Tensor* a,
                                    const Tensor* b,
                                    const Tensor* scales,
                                    const Tensor* zero_points,
                                    const Tensor* bias,
                                    uint32_t batch_count,
                                    uint32_t M,
                                    uint32_t N,
                                    uint32_t K,
                                    uint32_t block_size,
                                    uint32_t n_blocks_per_col,
                                    uint32_t zero_blocks_per_col,
                                    uint32_t blob_size,
                                    uint32_t nbits,
                                    bool has_zero_points,
                                    bool has_bias,
                                    bool has_weight_idx,
                                    bool has_weight_idx_indirect,
                                    bool single_scale_weights,
                                    onnxruntime::webgpu::ComputeContext& context,
                                    Tensor* y,
                                    uint32_t weight_index,
                                    const Tensor* weight_index_indirect,
                                    const MatMulNBitsTuneParams& params,
                                    bool wait_for_completion = false) {
  constexpr uint32_t kU32Components = 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  const uint32_t components_b_with_u32 = components_b * kU32Components;
  const uint32_t num_N_tile = CeilDiv(N, params.tile_size);
  const uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;

  MatMulNBitsProgram program{params.tile_size,
                             nbits,
                             has_zero_points,
                             has_bias,
                             has_weight_idx,
                             has_weight_idx_indirect,
                             single_scale_weights,
                             params.tile_size_k_vec};
  program.SetWorkgroupSize(params.workgroup_size);
  program.SetDispatchGroupSize(num_N_tile, M, batch_count);
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
                            {weight_index}})
      .CacheHint(nbits, has_zero_points, single_scale_weights, has_bias, has_weight_idx, has_weight_idx_indirect, params.tile_size_k_vec);
  if (has_zero_points) {
    program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }

  ORT_RETURN_IF_ERROR(context.RunProgram(program));
  if (wait_for_completion) {
    return context.FlushAndWait();
  }
  return Status::OK();
}

MatMulNBitsTuneParams GetTunedMatMulNBitsParams(const Tensor* a,
                                                const Tensor* b,
                                                const Tensor* scales,
                                                const Tensor* zero_points,
                                                const Tensor* bias,
                                                uint32_t batch_count,
                                                uint32_t M,
                                                uint32_t N,
                                                uint32_t K,
                                                uint32_t block_size,
                                                uint32_t n_blocks_per_col,
                                                uint32_t zero_blocks_per_col,
                                                uint32_t blob_size,
                                                uint32_t nbits,
                                                bool has_zero_points,
                                                bool has_bias,
                                                bool has_weight_idx,
                                                bool has_weight_idx_indirect,
                                                bool single_scale_weights,
                                                onnxruntime::webgpu::ComputeContext& context,
                                                Tensor* y,
                                                uint32_t weight_index,
                                                const Tensor* weight_index_indirect) {
  const bool is_fp16 = y->DataType() == DataTypeImpl::GetType<MLFloat16>();
  const std::string tune_key = MakeMatMulNBitsTuneKey(context, M, N, K, block_size, nbits, single_scale_weights, is_fp16);

  {
    std::lock_guard<std::mutex> lock(g_matmul_nbits_tune_mutex);
    const auto it = g_matmul_nbits_tuned_params.find(tune_key);
    if (it != g_matmul_nbits_tuned_params.end()) {
      return it->second;
    }
  }

  const MatMulNBitsTuneParams default_params = GetDefaultMatMulNBitsTuneParams(context);
  MatMulNBitsTuneParams best_params = default_params;
  double default_seconds = 0.0;
  double best_seconds = 0.0;

  bool default_failed = false;
  for (int i = 0; i < kMatMulNBitsTuneWarmupRuns; ++i) {
    const Status status = RunDefaultMatMulNBitsProgram(a,
                                                       b,
                                                       scales,
                                                       zero_points,
                                                       bias,
                                                       batch_count,
                                                       M,
                                                       N,
                                                       K,
                                                       block_size,
                                                       n_blocks_per_col,
                                                       zero_blocks_per_col,
                                                       blob_size,
                                                       nbits,
                                                       has_zero_points,
                                                       has_bias,
                                                       has_weight_idx,
                                                       has_weight_idx_indirect,
                                                       single_scale_weights,
                                                       context,
                                                       y,
                                                       weight_index,
                                                       weight_index_indirect,
                                                       default_params,
                                                       true);
    if (!status.IsOK()) {
      default_failed = true;
      break;
    }
  }

  if (!default_failed) {
    for (int i = 0; i < kMatMulNBitsTuneMeasuredRuns; ++i) {
      const auto start = std::chrono::steady_clock::now();
      const Status status = RunDefaultMatMulNBitsProgram(a,
                                                         b,
                                                         scales,
                                                         zero_points,
                                                         bias,
                                                         batch_count,
                                                         M,
                                                         N,
                                                         K,
                                                         block_size,
                                                         n_blocks_per_col,
                                                         zero_blocks_per_col,
                                                         blob_size,
                                                         nbits,
                                                         has_zero_points,
                                                         has_bias,
                                                         has_weight_idx,
                                                         has_weight_idx_indirect,
                                                         single_scale_weights,
                                                         context,
                                                         y,
                                                         weight_index,
                                                         weight_index_indirect,
                                                         default_params,
                                                         true);
      const auto end = std::chrono::steady_clock::now();
      if (!status.IsOK()) {
        default_failed = true;
        break;
      }
      default_seconds += std::chrono::duration<double>(end - start).count();
    }
  }

  if (default_failed) {
    LOGS(context.Logger(), WARNING) << "MatMulNBits tuner kept default params for " << tune_key
                                    << " because the baseline measurement failed"
                                    << ": tile_size_k_vec=" << default_params.tile_size_k_vec
                                    << ", workgroup_size=" << default_params.workgroup_size
                                    << ", tile_size=" << default_params.tile_size;

    std::lock_guard<std::mutex> lock(g_matmul_nbits_tune_mutex);
    g_matmul_nbits_tuned_params.insert_or_assign(tune_key, default_params);
    return default_params;
  }

  best_seconds = default_seconds;

  for (const auto& candidate : kMatMulNBitsTuneCandidates) {
    if (!IsValidTuneCandidate(context, candidate)) {
      continue;
    }

    if (candidate.tile_size_k_vec == default_params.tile_size_k_vec &&
        candidate.workgroup_size == default_params.workgroup_size &&
        candidate.tile_size == default_params.tile_size) {
      continue;
    }

    bool candidate_failed = false;
    for (int i = 0; i < kMatMulNBitsTuneWarmupRuns; ++i) {
      const Status status = RunDefaultMatMulNBitsProgram(a,
                                                         b,
                                                         scales,
                                                         zero_points,
                                                         bias,
                                                         batch_count,
                                                         M,
                                                         N,
                                                         K,
                                                         block_size,
                                                         n_blocks_per_col,
                                                         zero_blocks_per_col,
                                                         blob_size,
                                                         nbits,
                                                         has_zero_points,
                                                         has_bias,
                                                         has_weight_idx,
                                                         has_weight_idx_indirect,
                                                         single_scale_weights,
                                                         context,
                                                         y,
                                                         weight_index,
                                                         weight_index_indirect,
                                                         candidate,
                                                         true);
      if (!status.IsOK()) {
        candidate_failed = true;
        break;
      }
    }
    if (candidate_failed) {
      continue;
    }

    double candidate_seconds = 0.0;
    for (int i = 0; i < kMatMulNBitsTuneMeasuredRuns; ++i) {
      const auto start = std::chrono::steady_clock::now();
      const Status status = RunDefaultMatMulNBitsProgram(a,
                                                         b,
                                                         scales,
                                                         zero_points,
                                                         bias,
                                                         batch_count,
                                                         M,
                                                         N,
                                                         K,
                                                         block_size,
                                                         n_blocks_per_col,
                                                         zero_blocks_per_col,
                                                         blob_size,
                                                         nbits,
                                                         has_zero_points,
                                                         has_bias,
                                                         has_weight_idx,
                                                         has_weight_idx_indirect,
                                                         single_scale_weights,
                                                         context,
                                                         y,
                                                         weight_index,
                                                         weight_index_indirect,
                                                         candidate,
                                                         true);
      const auto end = std::chrono::steady_clock::now();
      if (!status.IsOK()) {
        candidate_failed = true;
        break;
      }
      candidate_seconds += std::chrono::duration<double>(end - start).count();
    }

    if (!candidate_failed && candidate_seconds < best_seconds) {
      best_seconds = candidate_seconds;
      best_params = candidate;
    }
  }

  const double improvement_ratio = default_seconds > 0.0
                                       ? (default_seconds - best_seconds) / default_seconds
                                       : 0.0;
  if (improvement_ratio <= kMatMulNBitsTuneMinImprovementRatio) {
    best_params = default_params;
    best_seconds = default_seconds;
  }

  LOGS(context.Logger(), WARNING) << "MatMulNBits tuner selected params for " << tune_key
                                  << ": tile_size_k_vec=" << best_params.tile_size_k_vec
                                  << ", workgroup_size=" << best_params.workgroup_size
                                  << ", tile_size=" << best_params.tile_size
                                  << ", default_measured_seconds=" << default_seconds
                                  << ", selected_measured_seconds=" << best_seconds
                                  << ", improvement_ratio=" << improvement_ratio;

  std::lock_guard<std::mutex> lock(g_matmul_nbits_tune_mutex);
  g_matmul_nbits_tuned_params.insert_or_assign(tune_key, best_params);
  return best_params;
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
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  const uint32_t workgroup_size = WorkgroupSizeX() * WorkgroupSizeY();
  ORT_ENFORCE(tile_m_ == workgroup_size / 8, "tile_m must be workgroup_size / 8.");
  ORT_ENFORCE(tile_n_ == workgroup_size, "tile_n must be workgroup_size.");
  ORT_ENFORCE(nbits_ == 4 || nbits_ == 8, "Only 4/8 bits are supported for webgpu matmulnbits.");

  return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_wide_tile.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
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
  if (has_weight_idx_indirect_) {
    shader.AddInput("weight_index_indirect", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

  const uint32_t components_a = a.NumComponents();
  const uint32_t components_b = b.NumComponents() / 4;  // b is stored as uint32 which includes 4 uint8.
  const uint32_t tile_size_k_vec = tile_size_k_vec_;
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
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx, has_weight_idx_),
                             WGSL_TEMPLATE_PARAMETER(has_weight_idx_indirect, has_weight_idx_indirect_),
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
  if (has_zero_points) {
    ORT_ENFORCE(zero_points->DataType() == DataTypeImpl::GetType<uint8_t>(), "Currently, only uint8 is supported for zero points, but got ", zero_points->DataType());
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
 * @param K_op           The K dimension of the operation (number of columns in 'a' and rows in 'b' before quantization).
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
 *       - Subgroup matrix multiplication for eligible Apple/Intel GPUs.
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
                        const uint32_t weight_index,
                        const Tensor* weight_index_indirect) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;
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
  // zero_points has shape[N * CeilDiv(n_blocks_per_col * bits, 8)].
  // The shader uses a flat linear index to address individual n-bit zero point values.
  // Since each column's zero points are byte-aligned in the packed buffer, we must round
  // n_blocks_per_col up to the next multiple of (8/nbits) — the number of zero point
  // values per byte — so that the linear stride correctly skips byte-boundary padding.
  const uint32_t zp_elements_per_byte = 8 / static_cast<uint32_t>(nbits);
  uint32_t zero_blocks_per_col = (n_blocks_per_col + zp_elements_per_byte - 1) / zp_elements_per_byte * zp_elements_per_byte;

#if !defined(__wasm__)
  int32_t subgroup_matrix_config_index = -1;
  // apple|intel - Experimental dawn support for subgroup matrix matmul.
  if (M >= kMinMForTileOptimization && (context.AdapterInfo().vendor == std::string_view{"apple"} || context.AdapterInfo().vendor == std::string_view{"intel"}) &&
      CanApplySubgroupMatrixMatMulNBits(context, accuracy_level, block_size, batch_count, N, K, static_cast<uint32_t>(nbits), y->DataType() == DataTypeImpl::GetType<MLFloat16>(), subgroup_matrix_config_index)) {
    return ApplySubgroupMatrixMatMulNBits(a, b, scales, zero_points, bias, M, N, K, static_cast<uint32_t>(nbits), zero_blocks_per_col, subgroup_matrix_config_index, context, y, weight_index, weight_index_indirect);
  }
#endif

  // On FP32 only GPUs and Qualcomm GPUs, integer math is faster than FP32 therefore always use DP4A independent of length of M.
  // DP4A Q2 path now supports custom zero points via a 1024-entry LUT (4 zero-point sections × 256 byte values).
  if ((M >= kMinMForTileOptimization || y->DataType() == DataTypeImpl::GetType<float>() || context.AdapterInfo().vendor == std::string_view{"qualcomm"}) &&
      CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a)) {
    return ApplyDP4AMatrixMatMulNBits(a, b, scales, zero_points, bias, batch_count, M, N, K, block_size, zero_blocks_per_col, kMinMForTileOptimization, static_cast<uint32_t>(nbits), context, y, weight_index, weight_index_indirect);
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
    const uint32_t num_N_tile = CeilDiv(N, tile_n);
    const uint32_t num_M_tile = CeilDiv(M, tile_m);

    MatMulNBitsWideTileProgram program{has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect, tile_m, tile_n, static_cast<uint32_t>(nbits)};
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
                        {CeilDiv(zero_points->Shape().Size(), static_cast<int64_t>(4))},
                        4});
    }
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    if (has_weight_idx_indirect) {
      program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
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
                                 {weight_index}});
    program.CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect);

    return context.RunProgram(program);
  }

  MatMulNBitsTuneParams params{};
  if (ShouldTuneDefaultMatMulNBitsProgram(context, batch_count, M, N, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect)) {
    params = GetTunedMatMulNBitsParams(a,
                                       b,
                                       scales,
                                       zero_points,
                                       bias,
                                       batch_count,
                                       M,
                                       N,
                                       K,
                                       block_size,
                                       n_blocks_per_col,
                                       zero_blocks_per_col,
                                       blob_size,
                                       static_cast<uint32_t>(nbits),
                                       has_zero_points,
                                       has_bias,
                                       has_weight_idx,
                                       has_weight_idx_indirect,
                                       single_scale_weights,
                                       context,
                                       y,
                                       weight_index,
                                       weight_index_indirect);
  } else {
    params = GetDefaultMatMulNBitsTuneParams(context);
  }

  return RunDefaultMatMulNBitsProgram(a,
                                      b,
                                      scales,
                                      zero_points,
                                      bias,
                                      batch_count,
                                      M,
                                      N,
                                      K,
                                      block_size,
                                      n_blocks_per_col,
                                      zero_blocks_per_col,
                                      blob_size,
                                      static_cast<uint32_t>(nbits),
                                      has_zero_points,
                                      has_bias,
                                      has_weight_idx,
                                      has_weight_idx_indirect,
                                      single_scale_weights,
                                      context,
                                      y,
                                      weight_index,
                                      weight_index_indirect,
                                      params);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
