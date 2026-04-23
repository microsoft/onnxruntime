// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_silu.h"

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/bert/skip_layer_norm.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/nn/layer_norm.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {

constexpr unsigned int kMinMForTileOptimization = 4;

constexpr uint32_t kFusedDecodeFastPathBits = 4u;
constexpr uint32_t kFusedDecodeFastPathBlockSize = 32u;
constexpr float kSkipSimplifiedLayerNormEpsilon = 1e-05f;

bool WouldApplySubgroupMatrixMatMulNBitsInCurrentDispatch(const Tensor* a,
                                                          int64_t K_op,
                                                          int64_t N_op,
                                                          int64_t block_size_op,
                                                          int64_t accuracy_level,
                                                          int64_t nbits,
                                                          onnxruntime::webgpu::ComputeContext& context,
                                                          Tensor* y) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  if (!helper.Compute(a->Shape(), b_shape, false, true).IsOK()) {
    return false;
  }

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);

#if !defined(__wasm__)
  int32_t subgroup_matrix_config_index = -1;
  return (M >= kMinMForTileOptimization) &&
         (context.AdapterInfo().vendor == std::string_view{"apple"} ||
          context.AdapterInfo().vendor == std::string_view{"intel"}) &&
         CanApplySubgroupMatrixMatMulNBits(context,
                                           accuracy_level,
                                           block_size,
                                           batch_count,
                                           N,
                                           K,
                                           static_cast<uint32_t>(nbits),
                                           y->DataType() == DataTypeImpl::GetType<MLFloat16>(),
                                           subgroup_matrix_config_index);
#endif

  return false;
}

bool WouldApplyWideTileMatMulNBitsInCurrentDispatch(const Tensor* a,
                                                    int64_t K_op,
                                                    int64_t N_op,
                                                    int64_t block_size_op,
                                                    int64_t nbits) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  if (!helper.Compute(a->Shape(), b_shape, false, true).IsOK()) {
    return false;
  }

  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);

  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t block_size_per_col = block_size;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);

  return block_size == 32 &&
         components_a == 4 &&
         components_b == 4 &&
         nbits != 2 &&
         M >= kMinMForTileOptimization;
}

bool WouldApplyDP4AMatMulNBitsInCurrentDispatch(const Tensor* a,
                                                int64_t K_op,
                                                int64_t N_op,
                                                int64_t block_size_op,
                                                int64_t accuracy_level,
                                                onnxruntime::webgpu::ComputeContext& context,
                                                Tensor* y) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  if (!helper.Compute(a->Shape(), b_shape, false, true).IsOK()) {
    return false;
  }

  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);
  const uint32_t components_a = GetMaxComponents(K);

  return ((M >= kMinMForTileOptimization) ||
          y->DataType() == DataTypeImpl::GetType<float>() ||
          context.AdapterInfo().vendor == std::string_view{"qualcomm"}) &&
         CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a);
}

TensorShape GetOverrideShape(const TensorShape& shape, int components) {
  return TensorShape{shape.Size() / components};
}

Status ApplySimplifiedLayerNorm(const Tensor* x,
                                const Tensor* scale,
                                float epsilon,
                                onnxruntime::webgpu::ComputeContext& context,
                                Tensor* y) {
  const auto& x_shape = x->Shape();
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  const int64_t norm_size = x_shape[x_shape.NumDimensions() - 1];
  const uint32_t norm_count = onnxruntime::narrow<uint32_t>(x_shape.Size() / norm_size);
  const int components = GetMaxComponents(norm_size);
  const uint32_t norm_size_vectorized = onnxruntime::narrow<uint32_t>((norm_size + components - 1) / components);
  const bool split_norm_dim = norm_size % 512 == 0 && norm_count == 1;

  onnxruntime::webgpu::LayerNormProgram program{/*has_bias=*/false,
                                                /*simplified=*/true,
                                                /*has_mean_output=*/false,
                                                /*has_inv_std_dev_output=*/false,
                                                split_norm_dim};

  program.CacheHint(components, true, split_norm_dim)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, GetOverrideShape(x_shape, components), components},
                  {scale, ProgramTensorMetadataDependency::Type, GetOverrideShape(scale->Shape(), components), components}})
      .AddOutputs({{y, ProgramTensorMetadataDependency::None, GetOverrideShape(y->Shape(), components), components}})
      .AddUniformVariables({{static_cast<uint32_t>(components)},
                            {norm_count},
                            {static_cast<uint32_t>(norm_size)},
                            {norm_size_vectorized},
                            {epsilon}});

  if (split_norm_dim) {
    const uint32_t workgroup_size_x = 128;
    const uint32_t dispatch_size_x = onnxruntime::narrow<uint32_t>(norm_size / (workgroup_size_x * components));
    program.SetDispatchGroupSize(dispatch_size_x, 1, 1)
        .SetWorkgroupSize(workgroup_size_x);
  } else {
    program.SetDispatchGroupSize(norm_count);
  }

  return context.RunProgram(program);
}

Status ApplySkipSimplifiedLayerNorm(const Tensor* x,
                                    const Tensor* skip,
                                    const Tensor* scale,
                                    float epsilon,
                                    onnxruntime::webgpu::ComputeContext& context,
                                    Tensor* y,
                                    Tensor* input_skip_bias_sum) {
  const auto& x_shape = x->Shape();
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  const uint32_t hidden_size = onnxruntime::narrow<uint32_t>(x_shape[x_shape.NumDimensions() - 1]);
  const int components = GetMaxComponents(hidden_size);
  const uint32_t norm_count = onnxruntime::narrow<uint32_t>(x_shape.SizeToDimension(x_shape.NumDimensions() - 1));
  const bool split_hidden_dim = hidden_size % 512 == 0 && norm_count == 1;
  const uint32_t skip_size = onnxruntime::narrow<uint32_t>(skip->Shape().Size());

  SkipLayerNormProgram program{/*hasBeta=*/false,
                               /*hasBias=*/false,
                               epsilon,
                               hidden_size,
                               input_skip_bias_sum != nullptr,
                               /*simplified=*/true,
                               split_hidden_dim};
  program
      .CacheHint(/*simplified=*/true, input_skip_bias_sum != nullptr, split_hidden_dim)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{skip, ProgramTensorMetadataDependency::Type, components}})
      .AddInputs({{scale, ProgramTensorMetadataDependency::Type, components}})
      .AddOutputs({{y, ProgramTensorMetadataDependency::None, components}})
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(ceil(1.0 * x_shape.Size() / hidden_size)))
      .AddUniformVariables({{static_cast<uint32_t>(components)}})
      .AddUniformVariables({{hidden_size}})
      .AddUniformVariables({{epsilon}})
      .AddUniformVariables({{skip_size}});

  if (split_hidden_dim) {
    const uint32_t workgroup_size_x = 128;
    const uint32_t dispatch_size_x = (input_skip_bias_sum != nullptr ? 2u : 1u) * hidden_size /
                                     (workgroup_size_x * components);
    program.SetDispatchGroupSize(dispatch_size_x, 1, 1)
        .SetWorkgroupSize(workgroup_size_x);
  }

  if (input_skip_bias_sum != nullptr) {
    program.AddOutputs({{input_skip_bias_sum, ProgramTensorMetadataDependency::None, components}});
  }

  return context.RunProgram(program);
}

Status ApplyUnfusedSiluMul(const Tensor* a,
                           const Tensor* gate_b,
                           const Tensor* gate_scales,
                           const Tensor* gate_bias,
                           const Tensor* up_b,
                           const Tensor* up_scales,
                           const Tensor* up_bias,
                           int64_t K,
                           int64_t N,
                           int64_t block_size,
                           int64_t accuracy_level,
                           int64_t bits,
                           onnxruntime::webgpu::ComputeContext& context,
                           Tensor* y);

class MatMulNBitsSiluMulDecodeProgram final : public Program<MatMulNBitsSiluMulDecodeProgram> {
 public:
  MatMulNBitsSiluMulDecodeProgram(uint32_t tile_size,
                                  bool has_gate_bias,
                                  bool has_up_bias,
                                  bool has_norm_input,
                                  bool has_skip_input,
                                  bool has_skip_output,
                                  bool single_scale_weights,
                                  uint32_t tile_size_k_vec,
                                  uint32_t k_unroll_tiles,
                                  bool has_full_n_tiles,
                                  bool has_full_k_tiles)
      : Program{"MatMulNBitsSiluMulDecode"},
        tile_size_(tile_size),
        has_gate_bias_(has_gate_bias),
        has_up_bias_(has_up_bias),
        has_norm_input_(has_norm_input),
        has_skip_input_(has_skip_input),
        has_skip_output_(has_skip_output),
        single_scale_weights_(single_scale_weights),
        tile_size_k_vec_(tile_size_k_vec),
        k_unroll_tiles_(k_unroll_tiles),
        has_full_n_tiles_(has_full_n_tiles),
        has_full_k_tiles_(has_full_k_tiles) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto* skip = has_skip_input_ ? &shader.AddInput("skip", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias) : nullptr;
    const auto* norm_scale = has_norm_input_ ? &shader.AddInput("norm_scale", ShaderUsage::UseValueTypeAlias) : nullptr;
    const auto& gate_b = shader.AddInput("gate_b");
    const auto& gate_scales_b = shader.AddInput("gate_scales_b");
    const auto& up_b = shader.AddInput("up_b");
    const auto& up_scales_b = shader.AddInput("up_scales_b");
    if (has_gate_bias_) {
      shader.AddInput("gate_bias", ShaderUsage::UseUniform);
    }
    if (has_up_bias_) {
      shader.AddInput("up_bias", ShaderUsage::UseUniform);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    const auto* input_skip_bias_sum = has_skip_output_
                                          ? &shader.AddOutput("input_skip_bias_sum",
                                                              ShaderUsage::UseValueTypeAlias |
                                                                  ShaderUsage::UseElementTypeAlias)
                                          : nullptr;

    const uint32_t components_a = a.NumComponents();
    const uint32_t components_b = gate_b.NumComponents() / 4;
    const uint32_t tile_size_k_vec = tile_size_k_vec_;
    const uint32_t elements_in_value_b = components_b * 8u;
    const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
    const uint32_t a_length_per_tile = tile_size_k / components_a;
    const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

    if (has_norm_input_ && has_skip_output_) {
      return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_silu_mul.wgsl.template",
                                 WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                                 WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                                 WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                                 WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                                 WGSL_TEMPLATE_PARAMETER(has_full_k_tiles, has_full_k_tiles_),
                                 WGSL_TEMPLATE_PARAMETER(has_full_n_tiles, has_full_n_tiles_),
                                 WGSL_TEMPLATE_PARAMETER(has_gate_bias, has_gate_bias_),
                                 WGSL_TEMPLATE_PARAMETER(has_norm_input, has_norm_input_),
                                 WGSL_TEMPLATE_PARAMETER(has_skip_input, has_skip_input_),
                                 WGSL_TEMPLATE_PARAMETER(has_skip_output, has_skip_output_),
                                 WGSL_TEMPLATE_PARAMETER(has_up_bias, has_up_bias_),
                                 WGSL_TEMPLATE_PARAMETER(k_unroll_tiles, k_unroll_tiles_),
                                 WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                                 WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                                 WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                                 WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                                 WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                                 WGSL_TEMPLATE_VARIABLE(a, a),
                                 WGSL_TEMPLATE_VARIABLE(gate_b, gate_b),
                                 WGSL_TEMPLATE_VARIABLE(gate_scales_b, gate_scales_b),
                                 WGSL_TEMPLATE_VARIABLE(input_skip_bias_sum, *input_skip_bias_sum),
                                 WGSL_TEMPLATE_VARIABLE(norm_scale, *norm_scale),
                                 WGSL_TEMPLATE_VARIABLE(output, output),
                                 WGSL_TEMPLATE_VARIABLE(skip, *skip),
                                 WGSL_TEMPLATE_VARIABLE(up_b, up_b),
                                 WGSL_TEMPLATE_VARIABLE(up_scales_b, up_scales_b));
    }

    if (has_norm_input_) {
      return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_silu_mul.wgsl.template",
                                 WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                                 WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                                 WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                                 WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                                 WGSL_TEMPLATE_PARAMETER(has_full_k_tiles, has_full_k_tiles_),
                                 WGSL_TEMPLATE_PARAMETER(has_full_n_tiles, has_full_n_tiles_),
                                 WGSL_TEMPLATE_PARAMETER(has_gate_bias, has_gate_bias_),
                                 WGSL_TEMPLATE_PARAMETER(has_norm_input, has_norm_input_),
                                 WGSL_TEMPLATE_PARAMETER(has_skip_input, has_skip_input_),
                                 WGSL_TEMPLATE_PARAMETER(has_skip_output, has_skip_output_),
                                 WGSL_TEMPLATE_PARAMETER(has_up_bias, has_up_bias_),
                                 WGSL_TEMPLATE_PARAMETER(k_unroll_tiles, k_unroll_tiles_),
                                 WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                                 WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                                 WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                                 WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                                 WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                                 WGSL_TEMPLATE_VARIABLE(a, a),
                                 WGSL_TEMPLATE_VARIABLE(gate_b, gate_b),
                                 WGSL_TEMPLATE_VARIABLE(gate_scales_b, gate_scales_b),
                                 WGSL_TEMPLATE_VARIABLE(norm_scale, *norm_scale),
                                 WGSL_TEMPLATE_VARIABLE(output, output),
                                 WGSL_TEMPLATE_VARIABLE(skip, *skip),
                                 WGSL_TEMPLATE_VARIABLE(up_b, up_b),
                                 WGSL_TEMPLATE_VARIABLE(up_scales_b, up_scales_b));
    }

    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_silu_mul.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                               WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                               WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                               WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                               WGSL_TEMPLATE_PARAMETER(has_full_k_tiles, has_full_k_tiles_),
                               WGSL_TEMPLATE_PARAMETER(has_full_n_tiles, has_full_n_tiles_),
                               WGSL_TEMPLATE_PARAMETER(has_gate_bias, has_gate_bias_),
                               WGSL_TEMPLATE_PARAMETER(has_norm_input, has_norm_input_),
                               WGSL_TEMPLATE_PARAMETER(has_skip_input, has_skip_input_),
                               WGSL_TEMPLATE_PARAMETER(has_skip_output, has_skip_output_),
                               WGSL_TEMPLATE_PARAMETER(has_up_bias, has_up_bias_),
                               WGSL_TEMPLATE_PARAMETER(k_unroll_tiles, k_unroll_tiles_),
                               WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                               WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                               WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                               WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                               WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                               WGSL_TEMPLATE_VARIABLE(a, a),
                               WGSL_TEMPLATE_VARIABLE(gate_b, gate_b),
                               WGSL_TEMPLATE_VARIABLE(gate_scales_b, gate_scales_b),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(up_b, up_b),
                               WGSL_TEMPLATE_VARIABLE(up_scales_b, up_scales_b));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K", ProgramUniformVariableDataType::Uint32},
      {"K_of_a", ProgramUniformVariableDataType::Uint32},
      {"K_of_b", ProgramUniformVariableDataType::Uint32},
      {"block_size", ProgramUniformVariableDataType::Uint32},
      {"blocks_per_col", ProgramUniformVariableDataType::Uint32},
      {"num_N_tile", ProgramUniformVariableDataType::Uint32},
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"skip_size", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32});

 private:
  uint32_t tile_size_;
  bool has_gate_bias_;
  bool has_up_bias_;
  bool has_norm_input_;
  bool has_skip_input_;
  bool has_skip_output_;
  bool single_scale_weights_;
  uint32_t tile_size_k_vec_;
  uint32_t k_unroll_tiles_;
  bool has_full_n_tiles_;
  bool has_full_k_tiles_;
};

class MatMulNBitsSiluMulWideTileM1Program final : public Program<MatMulNBitsSiluMulWideTileM1Program> {
 public:
  MatMulNBitsSiluMulWideTileM1Program(bool has_gate_bias,
                                      bool has_up_bias,
                                      uint32_t outputs_per_thread)
      : Program{"MatMulNBitsSiluMulWideTileM1Decode"},
        has_gate_bias_(has_gate_bias),
        has_up_bias_(has_up_bias),
        outputs_per_thread_(outputs_per_thread) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& gate_b = shader.AddInput("gate_b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& gate_scales_b = shader.AddInput("gate_scales_b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& up_b = shader.AddInput("up_b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& up_scales_b = shader.AddInput("up_scales_b", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    if (has_gate_bias_) {
      shader.AddInput("gate_bias", ShaderUsage::UseUniform);
    }
    if (has_up_bias_) {
      shader.AddInput("up_bias", ShaderUsage::UseUniform);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_silu_mul_wide_tile_m1.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_gate_bias, has_gate_bias_),
                               WGSL_TEMPLATE_PARAMETER(has_up_bias, has_up_bias_),
                               WGSL_TEMPLATE_PARAMETER(outputs_per_thread, outputs_per_thread_),
                               WGSL_TEMPLATE_VARIABLE(a, a),
                               WGSL_TEMPLATE_VARIABLE(gate_b, gate_b),
                               WGSL_TEMPLATE_VARIABLE(gate_scales_b, gate_scales_b),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(up_b, up_b),
                               WGSL_TEMPLATE_VARIABLE(up_scales_b, up_scales_b));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_count", ProgramUniformVariableDataType::Uint32},
      {"N", ProgramUniformVariableDataType::Uint32},
      {"K_of_a", ProgramUniformVariableDataType::Uint32},
      {"K_of_b", ProgramUniformVariableDataType::Uint32},
      {"n_blocks_per_col", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_gate_bias_;
  bool has_up_bias_;
  uint32_t outputs_per_thread_;
};

class MatMulNBitsSiluMulProgram final : public Program<MatMulNBitsSiluMulProgram> {
 public:
  MatMulNBitsSiluMulProgram() : Program{"MatMulNBitsSiluMul"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& gate = shader.AddInput("gate", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& up = shader.AddInput("up", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

    shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")
                              << "let gate_value = " << gate.GetByOffset("global_idx") << ";\n"
                              << "let up_value = " << up.GetByOffset("global_idx") << ";\n"
                              << "let one = output_value_t(1.0);\n"
                              << "let silu_value = gate_value * (one / (one + exp(-gate_value)));\n"
                              << output.SetByOffset("global_idx", "silu_value * up_value");

    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32});
};

Status ApplyUnfusedSiluMul(const Tensor* a,
                           const Tensor* gate_b,
                           const Tensor* gate_scales,
                           const Tensor* gate_bias,
                           const Tensor* up_b,
                           const Tensor* up_scales,
                           const Tensor* up_bias,
                           int64_t K,
                           int64_t N,
                           int64_t block_size,
                           int64_t accuracy_level,
                           int64_t bits,
                           onnxruntime::webgpu::ComputeContext& context,
                           Tensor* y) {
  MatMulComputeHelper helper;
  TensorShape b_shape({N, K});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  const auto output_shape = helper.OutputShape();

  Tensor gate_output = context.CreateGPUTensor(a->DataType(), output_shape);
  Tensor up_output = context.CreateGPUTensor(a->DataType(), output_shape);

  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(a, gate_b, gate_scales, nullptr, gate_bias, K, N, block_size, accuracy_level, bits, context, &gate_output));
  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(a, up_b, up_scales, nullptr, up_bias, K, N, block_size, accuracy_level, bits, context, &up_output));

  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  const uint32_t vec_size = (data_size + 3u) / 4u;
  MatMulNBitsSiluMulProgram program;
  program
      .AddInputs({{&gate_output, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4},
                  {&up_output, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4}})
      .AddOutput({y, ProgramTensorMetadataDependency::Type, {vec_size}, 4})
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({vec_size});

  return context.RunProgram(program);
}

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    MatMulNBitsSiluMul,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBitsSiluMul);

Status MatMulNBitsSiluMul::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* a = context.Input<Tensor>(0);
  const Tensor* skip = context.Input<Tensor>(1);
  const Tensor* norm_scale = context.Input<Tensor>(2);
  const Tensor* gate_b = context.Input<Tensor>(3);
  const Tensor* gate_scales = context.Input<Tensor>(4);
  const Tensor* gate_bias = context.Input<Tensor>(5);
  const Tensor* up_b = context.Input<Tensor>(6);
  const Tensor* up_scales = context.Input<Tensor>(7);
  const Tensor* up_bias = context.Input<Tensor>(8);

  ORT_ENFORCE(skip == nullptr || norm_scale != nullptr,
              "MatMulNBitsSiluMul requires norm_scale when skip is present.");

  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  const auto output_shape = helper.OutputShape();
  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_);
  const uint32_t components_a = GetMaxComponents(K);
  const bool single_scale_weights = (block_size == K * N);
  const uint32_t block_size_per_col = single_scale_weights ? K : block_size;
  const uint32_t n_blocks_per_col = (K + block_size_per_col - 1) / block_size_per_col;
  const uint32_t blob_size = (block_size_per_col / 8) * onnxruntime::narrow<uint32_t>(bits_);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  constexpr uint32_t kU32Components = 4;
  const uint32_t components_b_with_u32 = components_b * kU32Components;
  const uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;

  Tensor* y = context.Output(0, output_shape);
  Tensor* input_skip_bias_sum = skip != nullptr ? context.Output(1, a->Shape()) : nullptr;
  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  if (norm_scale != nullptr) {
    ORT_ENFORCE(norm_scale->Shape().Size() == K_, "norm_scale must have shape [K].");
  }

  const bool is_decode_fast_path_candidate =
      M == 1 &&
      bits_ == kFusedDecodeFastPathBits &&
      block_size == kFusedDecodeFastPathBlockSize;
  const bool has_norm_input = norm_scale != nullptr;

  const bool would_use_subgroup_unfused =
      WouldApplySubgroupMatrixMatMulNBitsInCurrentDispatch(a,
                                                           K_,
                                                           N_,
                                                           block_size_,
                                                           accuracy_level_,
                                                           bits_,
                                                           context,
                                                           y);
  const bool would_use_dp4a_unfused =
      WouldApplyDP4AMatMulNBitsInCurrentDispatch(a,
                                                 K_,
                                                 N_,
                                                 block_size_,
                                                 accuracy_level_,
                                                 context,
                                                 y);
  const bool would_use_wide_tile_unfused =
      WouldApplyWideTileMatMulNBitsInCurrentDispatch(a,
                             K_,
                             N_,
                             block_size_,
                             bits_);

  // The experimental wide M==1 fused path regressed badly on NVIDIA decode shapes.
  // Keep the implementation around for future work, but do not dispatch to it.

  const bool can_use_decode_fast_path =
      is_decode_fast_path_candidate &&
      !would_use_subgroup_unfused &&
      !would_use_dp4a_unfused &&
      !would_use_wide_tile_unfused;

  if (can_use_decode_fast_path) {
    //ORT_ENFORCE(false, "The experimental wide M==1 fused path regressed badly on NVIDIA decode shapes. Keep the implementation around for future work, but do not dispatch to it.");
    ORT_ENFORCE(bits_ == kFusedDecodeFastPathBits,
                "MatMulNBitsSiluMulDecodeProgram is specialized for 4-bit weights only.");
    ORT_ENFORCE(block_size == kFusedDecodeFastPathBlockSize,
                "MatMulNBitsSiluMulDecodeProgram is specialized for block_size=32 only.");

    const bool has_gate_bias = gate_bias != nullptr;
    const bool has_up_bias = up_bias != nullptr;
    const bool has_skip_input = skip != nullptr;
    const bool has_skip_output = input_skip_bias_sum != nullptr;
    uint32_t workgroup_size = 128;
    uint32_t tile_size = 8;
    uint32_t tile_size_k_vec =
      (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;

    // For the smallest decode-like M==1 case, reduce K-split width and workgroup size
    // so the generic fused kernel spends less time on reduction and barriers.
    if (context.AdapterInfo().vendor != std::string_view{"intel"} && N <= 2048) {
      workgroup_size = 64;
      tile_size = 4;
      tile_size_k_vec = 16;
    }

    const uint32_t elements_in_value_b = components_b * (32u / onnxruntime::narrow<uint32_t>(bits_));
    const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
    const bool has_full_n_tiles = (N % tile_size) == 0;
    const bool has_full_k_tiles = (K % tile_size_k) == 0;
    const uint32_t k_tile_iterations = K / tile_size_k;

    uint32_t k_unroll_tiles = 1;
    if (has_full_k_tiles) {
      if (k_tile_iterations >= 8 && N <= 2048 && context.AdapterInfo().vendor != std::string_view{"intel"}) {
        k_unroll_tiles = 4;
      } else if (k_tile_iterations >= 4) {
        k_unroll_tiles = 2;
      }
    }

    const uint32_t num_N_tile = CeilDiv(N, tile_size);

    MatMulNBitsSiluMulDecodeProgram program{tile_size,
                                            has_gate_bias,
                                            has_up_bias,
                                            has_norm_input,
                                            has_skip_input,
                                            has_skip_output,
                                            single_scale_weights,
                                            tile_size_k_vec,
                                            k_unroll_tiles,
                                            has_full_n_tiles,
                                            has_full_k_tiles};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, 1, batch_count);
    program.AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)});
    if (has_skip_input) {
      program.AddInput({skip, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)});
    }
    if (has_norm_input) {
      program.AddInput({norm_scale, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)});
    }
    program
        .AddInputs({{gate_b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                    {gate_scales, ProgramTensorMetadataDependency::TypeAndRank},
                    {up_b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                    {up_scales, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddOutput({y, ProgramTensorMetadataDependency::TypeAndRank})
        .AddUniformVariables({{N},
                              {K},
                              {K / components_a},
                              {K_of_b},
                              {block_size},
                              {n_blocks_per_col},
                              {num_N_tile},
                              {batch_count},
                              {has_skip_input ? onnxruntime::narrow<uint32_t>(skip->Shape().Size()) : 0u},
                              {kSkipSimplifiedLayerNormEpsilon}})
        .CacheHint(single_scale_weights,
                   has_gate_bias,
                   has_up_bias,
                   has_norm_input,
                   has_skip_input,
                   has_skip_output,
                   tile_size_k_vec,
                   k_unroll_tiles,
                   has_full_n_tiles,
                   has_full_k_tiles,
                   "decode_4bit");
    if (has_skip_output) {
      program.AddOutput({input_skip_bias_sum,
                         ProgramTensorMetadataDependency::TypeAndRank,
                         static_cast<int>(components_a)});
    }
    if (has_gate_bias) {
      program.AddInput({gate_bias, ProgramTensorMetadataDependency::None});
    }
    if (has_up_bias) {
      program.AddInput({up_bias, ProgramTensorMetadataDependency::None});
    }

    return context.RunProgram(program);
  }

  if (skip != nullptr) {
    Tensor normalized_a = context.CreateGPUTensor(a->DataType(), a->Shape());
    ORT_RETURN_IF_ERROR(ApplySkipSimplifiedLayerNorm(a, skip, norm_scale, kSkipSimplifiedLayerNormEpsilon,
                                                     context, &normalized_a, input_skip_bias_sum));
    return ApplyUnfusedSiluMul(&normalized_a,
                               gate_b,
                               gate_scales,
                               gate_bias,
                               up_b,
                               up_scales,
                               up_bias,
                               K_,
                               N_,
                               block_size_,
                               accuracy_level_,
                               bits_,
                               context,
                               y);
  }

  if (norm_scale != nullptr) {
    Tensor normalized_a = context.CreateGPUTensor(a->DataType(), a->Shape());
    ORT_RETURN_IF_ERROR(ApplySimplifiedLayerNorm(a, norm_scale, kSkipSimplifiedLayerNormEpsilon, context, &normalized_a));
    return ApplyUnfusedSiluMul(&normalized_a,
                               gate_b,
                               gate_scales,
                               gate_bias,
                               up_b,
                               up_scales,
                               up_bias,
                               K_,
                               N_,
                               block_size_,
                               accuracy_level_,
                               bits_,
                               context,
                               y);
  }

  return ApplyUnfusedSiluMul(a,
                             gate_b,
                             gate_scales,
                             gate_bias,
                             up_b,
                             up_scales,
                             up_bias,
                             K_,
                             N_,
                             block_size_,
                             accuracy_level_,
                             bits_,
                             context,
                             y);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
