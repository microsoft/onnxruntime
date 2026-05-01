// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_mlp.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/bert/skip_layer_norm.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/platform/env.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/nn/layer_norm.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace {

constexpr uint32_t kFusedDecodeFastPathBits = 4u;
constexpr uint32_t kFusedDecodeFastPathBlockSize = 32u;

// Override env var for the fused MLP decode kernel's dp4a inner-loop path:
//   "1" / "true" / "on"  -> force dp4a when the eligibility checks pass (default behavior).
//   "0" / "false" / "off" -> never use dp4a; fall back to the fp16 inner loop.
// Anything else (or unset) leaves the default policy in place. Intended for A/B perf and
// correctness comparisons without a rebuild.
constexpr const char* kFusedMlpDp4aEnvVar = "ORT_WEBGPU_MATMUL_NBITS_MLP_FUSED_DP4A";

// Override env var for the fused MLP decode kernel's external-quantize-A dp4a path:
// when enabled (and the variant has no fused norm/skip and dp4a is otherwise eligible),
// a small pre-pass quantizes A to int8 + fp16 scales and writes them to GPU memory; the
// fused matmul then reads pre-quantized A directly, skipping the in-kernel quantize step
// (and its workgroup barriers). Defaults to OFF until perf is validated.
//   "1" / "true" / "on"  -> force on when eligible.
//   "0" / "false" / "off" -> never use external-quantize path.
constexpr const char* kFusedMlpDp4aExternalEnvVar = "ORT_WEBGPU_MATMUL_NBITS_MLP_FUSED_DP4A_EXTERNAL";

// Tri-state env override: returns std::nullopt if unset/unrecognized, otherwise the parsed bool.
std::optional<bool> ParseBoolEnv(const char* name) {
  std::string raw = onnxruntime::Env::Default().GetEnvironmentVar(name);
  if (raw.empty()) {
    return std::nullopt;
  }
  std::transform(raw.begin(), raw.end(), raw.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (raw == "1" || raw == "true" || raw == "on" || raw == "yes") {
    return true;
  }
  if (raw == "0" || raw == "false" || raw == "off" || raw == "no") {
    return false;
  }
  return std::nullopt;
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

// Block-32 quantize-A pre-pass for the fused MLP decode kernel.
// Outputs:
//   - q_a:  array<u32>, K/4 entries per row (each u32 = pack4xI8 of 4 i8s)
//   - a_scales: array<fp16>, K/32 entries per row
// Layout matches the in-kernel `quantize_tile_A_dp4a` path so the matmul shader
// can consume these buffers without other changes. Requires subgroup_size >= 32.
class MatMulNBitsMlpQuantizeAProgram final : public Program<MatMulNBitsMlpQuantizeAProgram> {
 public:
  MatMulNBitsMlpQuantizeAProgram() : Program{"MatMulNBitsMlpQuantizeA"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input_a = shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
    const auto& output_q = shader.AddOutput("output_q", ShaderUsage::UseUniform);
    const auto& output_scales = shader.AddOutput("output_scales", ShaderUsage::UseUniform);
    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_mlp_quantize_a.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(input_a, input_a),
                               WGSL_TEMPLATE_VARIABLE(output_q, output_q),
                               WGSL_TEMPLATE_VARIABLE(output_scales, output_scales));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_blocks", ProgramUniformVariableDataType::Uint32});
};

class MatMulNBitsMlpDecodeProgram final : public Program<MatMulNBitsMlpDecodeProgram> {
 public:
  MatMulNBitsMlpDecodeProgram(uint32_t tile_size,
                              bool has_gate_bias,
                              bool has_up_bias,
                              bool has_norm_input,
                              bool has_skip_input,
                              bool has_skip_output,
                              bool single_scale_weights,
                              uint32_t tile_size_k_vec,
                              uint32_t k_unroll_tiles,
                              bool use_dp4a,
                              bool use_dp4a_external)
      : Program{"MatMulNBitsMlpDecode"},
        tile_size_(tile_size),
        has_gate_bias_(has_gate_bias),
        has_up_bias_(has_up_bias),
        has_norm_input_(has_norm_input),
        has_skip_input_(has_skip_input),
        has_skip_output_(has_skip_output),
        single_scale_weights_(single_scale_weights),
        tile_size_k_vec_(tile_size_k_vec),
        k_unroll_tiles_(k_unroll_tiles),
        use_dp4a_(use_dp4a),
        use_dp4a_external_(use_dp4a_external) {}

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
    // Pre-quantized A inputs (only present when use_dp4a_external_ is enabled).
    const auto* input_a_q = use_dp4a_external_ ? &shader.AddInput("input_a_q", ShaderUsage::UseUniform) : nullptr;
    const auto* input_a_scales = use_dp4a_external_ ? &shader.AddInput("input_a_scales", ShaderUsage::UseUniform) : nullptr;
    const auto& output = shader.AddOutput("output",
                                          ShaderUsage::UseElementTypeAlias);
    const auto* input_skip_bias_sum = has_skip_output_
                                          ? &shader.AddOutput("input_skip_bias_sum",
                                                              ShaderUsage::UseValueTypeAlias |
                                                                  ShaderUsage::UseElementTypeAlias)
                                          : nullptr;
    const auto& skip_var = skip != nullptr ? *skip : a;
    const auto& norm_scale_var = norm_scale != nullptr ? *norm_scale : a;
    const auto& input_skip_bias_sum_var = input_skip_bias_sum != nullptr ? *input_skip_bias_sum : output;
    // Placeholder bindings for the optional pre-quantized A inputs. Their template
    // variable references are only emitted under `#if use_dp4a_external`, so when the
    // path is disabled the placeholder is never referenced in generated WGSL.
    const auto& input_a_q_var = input_a_q != nullptr ? *input_a_q : a;
    const auto& input_a_scales_var = input_a_scales != nullptr ? *input_a_scales : a;

    const uint32_t components_a = a.NumComponents();
    const uint32_t components_b = gate_b.NumComponents() / 4;
    const uint32_t tile_size_k_vec = tile_size_k_vec_;
    const uint32_t elements_in_value_b = components_b * 8u;
    const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
    const uint32_t a_length_per_tile = tile_size_k / components_a;
    const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

    if (has_skip_input_) {
      if (has_norm_input_ && has_skip_output_) {
        return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_mlp.wgsl.template",
                                   WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                                   WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                                   WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                                   WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
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
                                   WGSL_TEMPLATE_PARAMETER(use_dp4a, use_dp4a_),
                                   WGSL_TEMPLATE_PARAMETER(use_dp4a_external, false),
                                   WGSL_TEMPLATE_VARIABLE(a, a),
                                   WGSL_TEMPLATE_VARIABLE(gate_b, gate_b),
                                   WGSL_TEMPLATE_VARIABLE(gate_scales_b, gate_scales_b),
                                   WGSL_TEMPLATE_VARIABLE(input_skip_bias_sum, input_skip_bias_sum_var),
                                   WGSL_TEMPLATE_VARIABLE(norm_scale, norm_scale_var),
                                   WGSL_TEMPLATE_VARIABLE(output, output),
                                   WGSL_TEMPLATE_VARIABLE(skip, skip_var),
                                   WGSL_TEMPLATE_VARIABLE(up_b, up_b),
                                   WGSL_TEMPLATE_VARIABLE(up_scales_b, up_scales_b));
      }

      if (has_norm_input_) {
        return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_mlp.wgsl.template",
                                   WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                                   WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                                   WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                                   WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
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
                                   WGSL_TEMPLATE_PARAMETER(use_dp4a, use_dp4a_),
                                   WGSL_TEMPLATE_PARAMETER(use_dp4a_external, false),
                                   WGSL_TEMPLATE_VARIABLE(a, a),
                                   WGSL_TEMPLATE_VARIABLE(gate_b, gate_b),
                                   WGSL_TEMPLATE_VARIABLE(gate_scales_b, gate_scales_b),
                                   WGSL_TEMPLATE_VARIABLE(input_skip_bias_sum, input_skip_bias_sum_var),
                                   WGSL_TEMPLATE_VARIABLE(norm_scale, norm_scale_var),
                                   WGSL_TEMPLATE_VARIABLE(output, output),
                                   WGSL_TEMPLATE_VARIABLE(skip, skip_var),
                                   WGSL_TEMPLATE_VARIABLE(up_b, up_b),
                                   WGSL_TEMPLATE_VARIABLE(up_scales_b, up_scales_b));
      }
    }

    if (has_norm_input_) {
      return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_mlp.wgsl.template",
                                 WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                                 WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                                 WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                                 WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
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
                                 WGSL_TEMPLATE_PARAMETER(use_dp4a, use_dp4a_),
                                 WGSL_TEMPLATE_PARAMETER(use_dp4a_external, false),
                                 WGSL_TEMPLATE_VARIABLE(a, a),
                                 WGSL_TEMPLATE_VARIABLE(gate_b, gate_b),
                                 WGSL_TEMPLATE_VARIABLE(gate_scales_b, gate_scales_b),
                                 WGSL_TEMPLATE_VARIABLE(input_skip_bias_sum, input_skip_bias_sum_var),
                                 WGSL_TEMPLATE_VARIABLE(norm_scale, norm_scale_var),
                                 WGSL_TEMPLATE_VARIABLE(output, output),
                                 WGSL_TEMPLATE_VARIABLE(skip, skip_var),
                                 WGSL_TEMPLATE_VARIABLE(up_b, up_b),
                                 WGSL_TEMPLATE_VARIABLE(up_scales_b, up_scales_b));
    }

    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_mlp.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                               WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                               WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                               WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
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
                               WGSL_TEMPLATE_PARAMETER(use_dp4a, use_dp4a_),
                               WGSL_TEMPLATE_PARAMETER(use_dp4a_external, use_dp4a_external_),
                               WGSL_TEMPLATE_VARIABLE(a, a),
                               WGSL_TEMPLATE_VARIABLE(gate_b, gate_b),
                               WGSL_TEMPLATE_VARIABLE(gate_scales_b, gate_scales_b),
                               WGSL_TEMPLATE_VARIABLE(input_a_q, input_a_q_var),
                               WGSL_TEMPLATE_VARIABLE(input_a_scales, input_a_scales_var),
                               WGSL_TEMPLATE_VARIABLE(input_skip_bias_sum, input_skip_bias_sum_var),
                               WGSL_TEMPLATE_VARIABLE(norm_scale, norm_scale_var),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(skip, skip_var),
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
  bool use_dp4a_;
  bool use_dp4a_external_;
};

class MatMulNBitsMlpProgram final : public Program<MatMulNBitsMlpProgram> {
 public:
  MatMulNBitsMlpProgram() : Program{"MatMulNBitsMlp"} {}

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

Status ApplyUnfusedMlp(const Tensor* a,
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
  MatMulNBitsMlpProgram program;
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
    MatMulNBitsMlp,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBitsMlp);

Status MatMulNBitsMlp::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
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
              "MatMulNBitsMlp requires norm_scale when skip is present.");

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
  Tensor* input_skip_bias_sum = (skip != nullptr && context.OutputCount() > 1)
                                    ? context.Output(1, a->Shape())
                                    : nullptr;
  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  if (norm_scale != nullptr) {
    ORT_ENFORCE(norm_scale->Shape().Size() == K_, "norm_scale must have shape [K].");
  }

  const bool has_skip_input = skip != nullptr;
  const bool has_skip_output = input_skip_bias_sum != nullptr;

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

  const bool can_use_decode_fast_path =
      is_decode_fast_path_candidate &&
      !would_use_subgroup_unfused &&
      !would_use_dp4a_unfused &&
      !would_use_wide_tile_unfused;

  if (can_use_decode_fast_path) {
    ORT_ENFORCE(bits_ == kFusedDecodeFastPathBits,
                "MatMulNBitsMlpDecodeProgram is specialized for 4-bit weights only.");
    ORT_ENFORCE(block_size == kFusedDecodeFastPathBlockSize,
                "MatMulNBitsMlpDecodeProgram is specialized for block_size=32 only.");

    const bool has_gate_bias = gate_bias != nullptr;
    const bool has_up_bias = up_bias != nullptr;
    uint32_t workgroup_size = 128;
    uint32_t tile_size = 8;
    uint32_t tile_size_k_vec =
        (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;

    const uint32_t elements_in_value_b = components_b * (32u / onnxruntime::narrow<uint32_t>(bits_));
    const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
    const uint32_t k_tile_iterations = K / tile_size_k;

    uint32_t k_unroll_tiles = 1;
    if ((K % tile_size_k) == 0) {
      if (k_tile_iterations >= 8 && N <= 2048 && context.AdapterInfo().vendor != std::string_view{"intel"}) {
        k_unroll_tiles = 4;
      } else if (k_tile_iterations >= 4) {
        k_unroll_tiles = 2;
      }
    }

    const uint32_t num_N_tile = CeilDiv(N, tile_size);

    // The dp4a fast path quantizes tile_A on-the-fly to int8 and uses dot4I8Packed in the
    // inner loop. It requires:
    //   - components_a == 4 (the per-block-32 absmax + pack4xI8 path assumes vec4 lanes);
    //   - components_b == 4 (each weight value covers exactly one block_size=32, so there
    //     is one weight scale and one A-block scale per inner-loop call);
    //   - tile_size_k % 32 == 0 (block-32 alignment for the absmax reduction);
    //   - the device feature gate already proven by the existing DP4A path.
    const bool dp4a_eligible =
        components_a == 4 &&
        components_b == 4 &&
        (tile_size_k % 32u) == 0u &&
        CanApplyDP4AMatrixMatMulNBits(context, accuracy_level_, block_size, N, K, components_a);
    // Allow A/B testing at runtime via env var. Default policy is "on when eligible".
    const std::optional<bool> dp4a_env = ParseBoolEnv(kFusedMlpDp4aEnvVar);
    const bool use_dp4a = dp4a_eligible && dp4a_env.value_or(true);

    // External-quantize path: pre-quantize A in a separate pass, then feed q_a + a_scales
    // into the matmul. Only enabled for the Simplified variant (no fused norm/skip) so
    // we can quantize the activation directly without un-fusing the norm. Defaults OFF.
    const std::optional<bool> dp4a_external_env = ParseBoolEnv(kFusedMlpDp4aExternalEnvVar);
    const bool dp4a_external_eligible =
        dp4a_eligible &&
        !has_norm_input &&
        !has_skip_input &&
        !has_skip_output &&
        (K % 32u) == 0u &&
        batch_count == 1u &&  // single token decode; pre-quantize sized for batch=1
        M == 1u;
    const bool use_dp4a_external = dp4a_external_eligible && dp4a_external_env.value_or(false);
    // When use_dp4a_external is on, the in-kernel dp4a path is not needed.
    const bool effective_use_dp4a = use_dp4a && !use_dp4a_external;

    // Pre-quantize A if requested.
    Tensor a_quant;
    Tensor a_scale;
    if (use_dp4a_external) {
      const uint32_t total_blocks = K / 32u;
      TensorShape a_quant_shape{static_cast<int64_t>(K / 4u)};
      a_quant = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), a_quant_shape);
      TensorShape a_scale_shape{static_cast<int64_t>(total_blocks)};
      a_scale = context.CreateGPUTensor(a->DataType(), a_scale_shape);
      MatMulNBitsMlpQuantizeAProgram quantize_program;
      quantize_program.SetWorkgroupSize(32);
      quantize_program.SetDispatchGroupSize(total_blocks, 1, 1);
      quantize_program
          .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, 1})
          .AddOutputs({{&a_quant, ProgramTensorMetadataDependency::Rank, 1},
                       {&a_scale, ProgramTensorMetadataDependency::Rank, 1}})
          .AddUniformVariable({total_blocks});
      ORT_RETURN_IF_ERROR(context.RunProgram(quantize_program));
    }

    MatMulNBitsMlpDecodeProgram program{tile_size,
                                        has_gate_bias,
                                        has_up_bias,
                                        has_norm_input,
                                        has_skip_input,
                                        has_skip_output,
                                        single_scale_weights,
                                        tile_size_k_vec,
                                        k_unroll_tiles,
                                        effective_use_dp4a,
                                        use_dp4a_external};
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
                              {epsilon_}})
        .CacheHint(single_scale_weights,
                   has_gate_bias,
                   has_up_bias,
                   has_norm_input,
                   has_skip_input,
                   has_skip_output,
                   tile_size_k_vec,
                   k_unroll_tiles,
                   effective_use_dp4a,
                   use_dp4a_external,
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
    if (use_dp4a_external) {
      program.AddInput({&a_quant, ProgramTensorMetadataDependency::Rank, 1});
      program.AddInput({&a_scale, ProgramTensorMetadataDependency::Rank, 1});
    }

    return context.RunProgram(program);
  }

  if (skip != nullptr) {
    Tensor normalized_a = context.CreateGPUTensor(a->DataType(), a->Shape());
    ORT_RETURN_IF_ERROR(ApplySkipSimplifiedLayerNorm(a, skip, norm_scale, epsilon_,
                                                     context, &normalized_a, input_skip_bias_sum));
    return ApplyUnfusedMlp(&normalized_a,
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
    ORT_RETURN_IF_ERROR(ApplySimplifiedLayerNorm(a, norm_scale, epsilon_, context, &normalized_a));
    return ApplyUnfusedMlp(&normalized_a,
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

  return ApplyUnfusedMlp(a,
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
