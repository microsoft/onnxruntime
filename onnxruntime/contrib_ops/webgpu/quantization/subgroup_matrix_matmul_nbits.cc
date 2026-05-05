// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

constexpr std::string_view ComponentTypeName[] = {"unknown", "f32", "f16", "u32", "i32"};
template <std::size_t N>
constexpr bool ValidateComponentTypeName(const std::array<wgpu::SubgroupMatrixComponentType, N>& component_type) {
  bool matched = true;
  for (auto type : component_type) {
    switch (type) {
      case wgpu::SubgroupMatrixComponentType::F32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::F32)] == "f32";
        break;
      case wgpu::SubgroupMatrixComponentType::F16:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::F16)] == "f16";
        break;
      case wgpu::SubgroupMatrixComponentType::U32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::U32)] == "u32";
        break;
      case wgpu::SubgroupMatrixComponentType::I32:
        matched = ComponentTypeName[static_cast<uint32_t>(wgpu::SubgroupMatrixComponentType::I32)] == "i32";
        break;
      default:
        return false;
    }

    if (!matched) {
      return matched;
    }
  }

  return matched;
}
static_assert(ValidateComponentTypeName<4>({wgpu::SubgroupMatrixComponentType::F32,
                                            wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::U32,
                                            wgpu::SubgroupMatrixComponentType::I32}),
              "The elements' sequence of ComponentTypeName array do not match wgpu::SubgroupMatrixComponentType");

// Vendor-agnostic subgroup matrix config: {componentType, resultComponentType, M, N, K, subgroupMinSize, subgroupMaxSize, needsPrepack}
// Any GPU reporting a matching config from wgpu::AdapterPropertiesSubgroupMatrixConfigs is supported.
struct SupportedSubgroupMatrixConfig {
  wgpu::SubgroupMatrixComponentType componentType;
  wgpu::SubgroupMatrixComponentType resultComponentType;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t subgroupMinSize;
  uint32_t subgroupMaxSize;
  bool needsPrepack;  // Whether input A needs layout optimization for subgroupMatrixLoad
};

static const SupportedSubgroupMatrixConfig supported_subgroup_matrix_configs[] = {
    // 16x16x16 config with 128x128 tiles (NVIDIA Blackwell, subgroup size 32)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 16, 16, 16, 32, 32, true},
    // 8x16x16 config (Intel Xe2/Xe3, subgroup size 16-32)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 8, 16, 16, 16, 32, true},
    // 8x8x8 config (Apple M-series, etc.)
    {wgpu::SubgroupMatrixComponentType::F16, wgpu::SubgroupMatrixComponentType::F16, 8, 8, 8, 32, 32, false},
    {wgpu::SubgroupMatrixComponentType::F32, wgpu::SubgroupMatrixComponentType::F32, 8, 8, 8, 32, 32, false},
};

bool IsSubgroupMatrixConfigSupported(onnxruntime::webgpu::ComputeContext& context, bool is_fp16, int32_t& config_index) {
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();
  const wgpu::AdapterPropertiesSubgroupMatrixConfigs& subgroup_matrix_configs = context.SubgroupMatrixConfigs();
  int32_t index = 0;
  for (const auto& supported_config : supported_subgroup_matrix_configs) {
    // F16 configs require FP16 output; skip them when output is F32.
    // F32 configs require FP32 output; skip them when output is FP16.
    if ((supported_config.componentType == wgpu::SubgroupMatrixComponentType::F16 && !is_fp16) ||
        (supported_config.componentType == wgpu::SubgroupMatrixComponentType::F32 && is_fp16)) {
      index++;
      continue;
    }
    for (size_t i = 0; i < subgroup_matrix_configs.configCount; i++) {
      const auto& device_config = subgroup_matrix_configs.configs[i];
      if (device_config.componentType == supported_config.componentType &&
          device_config.resultComponentType == supported_config.resultComponentType &&
          device_config.M == supported_config.M &&
          device_config.N == supported_config.N &&
          device_config.K == supported_config.K &&
          adapter_info.subgroupMinSize == supported_config.subgroupMinSize &&
          adapter_info.subgroupMaxSize == supported_config.subgroupMaxSize) {
        config_index = index;
        return true;
      }
    }
    index++;
  }
  return false;
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
                                 uint32_t nbits, int32_t config_index, bool has_zero_points, bool has_bias, bool has_weight_idx, bool has_weight_idx_indirect) {
  const auto& config = supported_subgroup_matrix_configs[config_index];
  return WGSL_TEMPLATE_APPLY(shader, "quantization/subgroup_matrix_matmul_nbits_8x16x16.wgsl.template",
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

  const auto& config = supported_subgroup_matrix_configs[config_index_];
  if (config.M == 8 && config.N == 8 && config.K == 8) {
    return GenerateShaderCode8x8x8(shader, a, b, scales_b, output, nbits_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_);
  } else if (config.M == 8 && config.N == 16 && config.K == 16) {
    return GenerateShaderCode8x16x16(shader, b, scales_b, output, nbits_, config_index_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_);
  } else if (config.M == 16 && config.N == 16 && config.K == 16) {
    return GenerateShaderCode16x16x16(shader, b, scales_b, output, nbits_, config_index_, has_zero_points_, has_bias_, has_weight_idx_, has_weight_idx_indirect_);
  } else {
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                  "Unsupported subgroup matrix config dimensions.");
  }
}

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
  // Determine tile sizes first (needed for prepack padding).
  const auto& config = supported_subgroup_matrix_configs[config_index];
  uint32_t tile_size_a = 32;
  uint32_t tile_size_b = 64;
  uint32_t work_group_size = 128;
  if (config.M == 8 && config.N == 16 && config.K == 16) {
    // 8x16x16 config: 8 subgroups, 256 threads, 64x64 tiles
    tile_size_a = 64;
    work_group_size = 256;
  } else if (config.M == 16 && config.N == 16 && config.K == 16) {
    // 16x16x16 config: 4 subgroups, 128 threads, 128x128 tiles
    tile_size_a = 128;
    tile_size_b = 128;
    work_group_size = 128;
  }

  // If applicable, layout optimization of input matrix A(MxK) can be used for SubgroupMatrixLoad.
  Tensor a_prepack;
  if (config.needsPrepack) {
    const auto m = config.M;
    const auto k = config.K;

    // Optimize the layout of input matrix A(MxK) for SubgroupMatrixLoad.
    PrepackProgram prepack_program{m, k};
    constexpr uint32_t kSubgroupSize = 32;
    prepack_program.SetWorkgroupSize(kSubgroupSize);

    // Pad M to workgroup tile size so all subgroups read valid prepacked data.
    const uint32_t padded_M = ((M + tile_size_a - 1) / tile_size_a) * tile_size_a;
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
  const bool has_zero_points = zero_points != nullptr;
  const bool has_bias = bias != nullptr;
  const bool has_weight_idx_indirect = weight_index_indirect != nullptr;
  const bool has_weight_idx = weight_index > 0 || has_weight_idx_indirect;
  SubgroupMatrixMatMulNBitsProgram mul_program{nbits, config_index, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect};
  mul_program.SetWorkgroupSize(work_group_size);
  mul_program.SetDispatchGroupSize(
      (N + tile_size_b - 1) / tile_size_b,
      (M + tile_size_a - 1) / tile_size_a, 1);
  mul_program.AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1},
                         {b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(nbits == 4 ? kU32Components : 2 * kU32Components)},
                         {scales, ProgramTensorMetadataDependency::TypeAndRank, 1}})
      .AddUniformVariables({{M}, {N}, {K}, {zero_blocks_per_col}, {weight_index}})
      .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank, y_shape, 1})
      .CacheHint(nbits, has_zero_points, has_bias, has_weight_idx, has_weight_idx_indirect);
  if (has_zero_points) {
    mul_program.AddInput({zero_points, ProgramTensorMetadataDependency::None, {(zero_points->Shape().Size() + 3) / 4}, 4});
  }
  if (bias) {
    mul_program.AddInput({bias, ProgramTensorMetadataDependency::None});
  }
  if (has_weight_idx_indirect) {
    mul_program.AddInput({weight_index_indirect, ProgramTensorMetadataDependency::None});
  }
  return context.RunProgram(mul_program);
}

bool CanApplySubgroupMatrixMatMulNBits(onnxruntime::webgpu::ComputeContext& context,
                                       uint64_t accuracy_level,
                                       uint32_t block_size,
                                       uint32_t batch_count,
                                       uint32_t N,
                                       uint32_t K,
                                       uint32_t nbits,
                                       bool is_fp16,
                                       int32_t& config_index) {
  // Subgroup matrix kernels only support 4-bit/8-bit quantization.
  if (nbits != 4 && nbits != 8) {
    return false;
  }

  bool has_subgroup_matrix = context.HasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
  if (has_subgroup_matrix) {
    // Check if the adapter reports a subgroup matrix config we support.
    has_subgroup_matrix = IsSubgroupMatrixConfigSupported(context, is_fp16, config_index);
    if (has_subgroup_matrix) {
      if (context.AdapterInfo().vendor == std::string_view{"apple"}) {
        // For now SubgroupMatrixMatMulNBits is only supported for accuracy level 4, because with Fp16 there are
        // some precision issues with subgroupMatrixMultiplyAccumulate. It is possible to support higher accuracy
        // by setting compute_precision to Fp32, but that will be slower. For 1K token prefill FP16 Phi 3.5 is around 5s,
        // FP32 is around 7s.
        has_subgroup_matrix = accuracy_level == 4;
      }
    }
  }

  return has_subgroup_matrix &&
         block_size == 32 &&
         batch_count == 1 &&
         K % 32 == 0 &&
         N % 64 == 0;
}
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime

#endif
