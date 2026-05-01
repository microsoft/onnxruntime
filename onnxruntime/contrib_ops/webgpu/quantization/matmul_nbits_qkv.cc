// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_qkv.h"

#include <optional>

#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
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
    const uint32_t dispatch_size_x = (input_skip_bias_sum != nullptr ? 2u : 1u) * hidden_size / (workgroup_size_x * components);
    program.SetDispatchGroupSize(dispatch_size_x, 1, 1)
        .SetWorkgroupSize(workgroup_size_x);
  }

  if (input_skip_bias_sum != nullptr) {
    program.AddOutputs({{input_skip_bias_sum, ProgramTensorMetadataDependency::None, components}});
  }

  return context.RunProgram(program);
}

Status ApplyUnfusedQKVSimplifiedLayerNorm(const Tensor* a,
                                          const Tensor* norm_scale,
                                          const Tensor* q_b,
                                          const Tensor* q_scales,
                                          const Tensor* k_b,
                                          const Tensor* k_scales,
                                          const Tensor* v_b,
                                          const Tensor* v_scales,
                                          int64_t K,
                                          int64_t Nq,
                                          int64_t Nkv,
                                          int64_t block_size,
                                          int64_t accuracy_level,
                                          int64_t bits,
                                          float epsilon,
                                          onnxruntime::webgpu::ComputeContext& context,
                                          Tensor* q_output,
                                          Tensor* k_output,
                                          Tensor* v_output) {
  Tensor normalized_a = context.CreateGPUTensor(a->DataType(), a->Shape());
  ORT_RETURN_IF_ERROR(ApplySimplifiedLayerNorm(a, norm_scale, epsilon, context, &normalized_a));
  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(&normalized_a, q_b, q_scales, nullptr, nullptr,
                                       K, Nq, block_size, accuracy_level, bits, context, q_output));
  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(&normalized_a, k_b, k_scales, nullptr, nullptr,
                                       K, Nkv, block_size, accuracy_level, bits, context, k_output));
  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(&normalized_a, v_b, v_scales, nullptr, nullptr,
                                       K, Nkv, block_size, accuracy_level, bits, context, v_output));
  return Status::OK();
}

Status ApplyUnfusedQKVSkipSimplifiedLayerNorm(const Tensor* a,
                                              const Tensor* skip,
                                              const Tensor* norm_scale,
                                              const Tensor* q_b,
                                              const Tensor* q_scales,
                                              const Tensor* k_b,
                                              const Tensor* k_scales,
                                              const Tensor* v_b,
                                              const Tensor* v_scales,
                                              int64_t K,
                                              int64_t Nq,
                                              int64_t Nkv,
                                              int64_t block_size,
                                              int64_t accuracy_level,
                                              int64_t bits,
                                              float epsilon,
                                              onnxruntime::webgpu::ComputeContext& context,
                                              Tensor* q_output,
                                              Tensor* k_output,
                                              Tensor* v_output,
                                              Tensor* input_skip_bias_sum) {
  Tensor normalized_a = context.CreateGPUTensor(a->DataType(), a->Shape());
  ORT_RETURN_IF_ERROR(ApplySkipSimplifiedLayerNorm(a, skip, norm_scale, epsilon, context, &normalized_a, input_skip_bias_sum));
  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(&normalized_a, q_b, q_scales, nullptr, nullptr,
                                       K, Nq, block_size, accuracy_level, bits, context, q_output));
  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(&normalized_a, k_b, k_scales, nullptr, nullptr,
                                       K, Nkv, block_size, accuracy_level, bits, context, k_output));
  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(&normalized_a, v_b, v_scales, nullptr, nullptr,
                                       K, Nkv, block_size, accuracy_level, bits, context, v_output));
  return Status::OK();
}

class MatMulNBitsQkvDecodeProgram final
    : public Program<MatMulNBitsQkvDecodeProgram> {
 public:
  MatMulNBitsQkvDecodeProgram(uint32_t tile_size,
                              bool single_scale_weights,
                              uint32_t tile_size_k_vec,
                              uint32_t k_unroll_tiles,
                              bool has_skip_input,
                              bool has_skip_output)
      : Program{"MatMulNBitsQkvDecode"},
        tile_size_(tile_size),
        single_scale_weights_(single_scale_weights),
        tile_size_k_vec_(tile_size_k_vec),
        k_unroll_tiles_(k_unroll_tiles),
        has_skip_input_(has_skip_input),
        has_skip_output_(has_skip_output) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
    const auto* skip = has_skip_input_ ? &shader.AddInput("skip", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias) : nullptr;
    const auto& norm_scale = shader.AddInput("norm_scale", ShaderUsage::UseValueTypeAlias);
    const auto& q_b = shader.AddInput("q_b", ShaderUsage::UseValueTypeAlias);
    const auto& q_scales_b = shader.AddInput("q_scales_b");
    const auto& k_b = shader.AddInput("k_b");
    const auto& k_scales_b = shader.AddInput("k_scales_b");
    const auto& v_b = shader.AddInput("v_b");
    const auto& v_scales_b = shader.AddInput("v_scales_b");
    const auto& q_output = shader.AddOutput("q_output",
                                            ShaderUsage::UseValueTypeAlias |
                                                ShaderUsage::UseElementTypeAlias);
    const auto& k_output = shader.AddOutput("k_output",
                                            ShaderUsage::UseValueTypeAlias |
                                                ShaderUsage::UseElementTypeAlias);
    const auto& v_output = shader.AddOutput("v_output",
                                            ShaderUsage::UseValueTypeAlias |
                                                ShaderUsage::UseElementTypeAlias);
    const auto* input_skip_bias_sum = has_skip_output_ ? &shader.AddOutput("input_skip_bias_sum", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias) : nullptr;
    const auto& skip_var = skip != nullptr ? *skip : a;
    const auto& input_skip_bias_sum_var = input_skip_bias_sum != nullptr ? *input_skip_bias_sum : q_output;

    const uint32_t components_a = a.NumComponents();
    const uint32_t components_b = q_b.NumComponents() / 4;
    const uint32_t tile_size_k_vec = tile_size_k_vec_;
    const uint32_t elements_in_value_b = components_b * 8u;
    const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
    const uint32_t a_length_per_tile = tile_size_k / components_a;
    const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

    if (skip != nullptr) {
      if (input_skip_bias_sum != nullptr) {
        return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_qkv.wgsl.template",
                                   WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                                   WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                                   WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                                   WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                                   WGSL_TEMPLATE_PARAMETER(has_skip_input, has_skip_input_),
                                   WGSL_TEMPLATE_PARAMETER(has_skip_output, has_skip_output_),
                                   WGSL_TEMPLATE_PARAMETER(k_unroll_tiles, k_unroll_tiles_),
                                   WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                                   WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                                   WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                                   WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                                   WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                                   WGSL_TEMPLATE_VARIABLE(a, a),
                                   WGSL_TEMPLATE_VARIABLE(input_skip_bias_sum, input_skip_bias_sum_var),
                                   WGSL_TEMPLATE_VARIABLE(k_b, k_b),
                                   WGSL_TEMPLATE_VARIABLE(k_output, k_output),
                                   WGSL_TEMPLATE_VARIABLE(k_scales_b, k_scales_b),
                                   WGSL_TEMPLATE_VARIABLE(norm_scale, norm_scale),
                                   WGSL_TEMPLATE_VARIABLE(q_b, q_b),
                                   WGSL_TEMPLATE_VARIABLE(q_output, q_output),
                                   WGSL_TEMPLATE_VARIABLE(q_scales_b, q_scales_b),
                                   WGSL_TEMPLATE_VARIABLE(skip, skip_var),
                                   WGSL_TEMPLATE_VARIABLE(v_b, v_b),
                                   WGSL_TEMPLATE_VARIABLE(v_output, v_output),
                                   WGSL_TEMPLATE_VARIABLE(v_scales_b, v_scales_b));
      }

      return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_qkv.wgsl.template",
                                 WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                                 WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                                 WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                                 WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                                 WGSL_TEMPLATE_PARAMETER(has_skip_input, has_skip_input_),
                                 WGSL_TEMPLATE_PARAMETER(has_skip_output, has_skip_output_),
                                 WGSL_TEMPLATE_PARAMETER(k_unroll_tiles, k_unroll_tiles_),
                                 WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                                 WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                                 WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                                 WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                                 WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                                 WGSL_TEMPLATE_VARIABLE(a, a),
                                 WGSL_TEMPLATE_VARIABLE(input_skip_bias_sum, input_skip_bias_sum_var),
                                 WGSL_TEMPLATE_VARIABLE(k_b, k_b),
                                 WGSL_TEMPLATE_VARIABLE(k_output, k_output),
                                 WGSL_TEMPLATE_VARIABLE(k_scales_b, k_scales_b),
                                 WGSL_TEMPLATE_VARIABLE(norm_scale, norm_scale),
                                 WGSL_TEMPLATE_VARIABLE(q_b, q_b),
                                 WGSL_TEMPLATE_VARIABLE(q_output, q_output),
                                 WGSL_TEMPLATE_VARIABLE(q_scales_b, q_scales_b),
                                 WGSL_TEMPLATE_VARIABLE(skip, skip_var),
                                 WGSL_TEMPLATE_VARIABLE(v_b, v_b),
                                 WGSL_TEMPLATE_VARIABLE(v_output, v_output),
                                 WGSL_TEMPLATE_VARIABLE(v_scales_b, v_scales_b));
    }

    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_qkv.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                               WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                               WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                               WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                               WGSL_TEMPLATE_PARAMETER(has_skip_input, has_skip_input_),
                               WGSL_TEMPLATE_PARAMETER(has_skip_output, has_skip_output_),
                               WGSL_TEMPLATE_PARAMETER(k_unroll_tiles, k_unroll_tiles_),
                               WGSL_TEMPLATE_PARAMETER(single_scale_weights, single_scale_weights_),
                               WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                               WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                               WGSL_TEMPLATE_PARAMETER(tile_size_k, tile_size_k),
                               WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                               WGSL_TEMPLATE_VARIABLE(a, a),
                               WGSL_TEMPLATE_VARIABLE(input_skip_bias_sum, input_skip_bias_sum_var),
                               WGSL_TEMPLATE_VARIABLE(k_b, k_b),
                               WGSL_TEMPLATE_VARIABLE(k_output, k_output),
                               WGSL_TEMPLATE_VARIABLE(k_scales_b, k_scales_b),
                               WGSL_TEMPLATE_VARIABLE(norm_scale, norm_scale),
                               WGSL_TEMPLATE_VARIABLE(q_b, q_b),
                               WGSL_TEMPLATE_VARIABLE(q_output, q_output),
                               WGSL_TEMPLATE_VARIABLE(q_scales_b, q_scales_b),
                               WGSL_TEMPLATE_VARIABLE(skip, skip_var),
                               WGSL_TEMPLATE_VARIABLE(v_b, v_b),
                               WGSL_TEMPLATE_VARIABLE(v_output, v_output),
                               WGSL_TEMPLATE_VARIABLE(v_scales_b, v_scales_b));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"Nq", ProgramUniformVariableDataType::Uint32},
      {"Nkv", ProgramUniformVariableDataType::Uint32},
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
  bool single_scale_weights_;
  uint32_t tile_size_k_vec_;
  uint32_t k_unroll_tiles_;
  bool has_skip_input_;
  bool has_skip_output_;
};

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    MatMulNBitsQkv,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBitsQkv);

Status MatMulNBitsQkv::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* a = context.Input(0);
  const Tensor* skip = context.Input(1);
  const Tensor* norm_scale = context.Input(2);
  const Tensor* q_b = context.Input(3);
  const Tensor* q_scales = context.Input(4);
  const Tensor* k_b = context.Input(5);
  const Tensor* k_scales = context.Input(6);
  const Tensor* v_b = context.Input(7);
  const Tensor* v_scales = context.Input(8);

  ORT_ENFORCE(bits_ == 4, "MatMulNBitsQkv currently supports 4-bit weights only.");
  ORT_ENFORCE(block_size_ == 32, "MatMulNBitsQkv currently supports block_size=32 only.");

  TensorShape q_b_shape({Nq_, K_});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), q_b_shape, false, true));

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t Nq = onnxruntime::narrow<uint32_t>(Nq_);
  const uint32_t Nkv = onnxruntime::narrow<uint32_t>(Nkv_);

  auto q_shape = helper.OutputShape();
  TensorShapeVector kv_dims(q_shape.GetDims().begin(), q_shape.GetDims().end());
  kv_dims.back() = Nkv_;
  TensorShape kv_shape(kv_dims);
  Tensor* q_output = context.Output(0, q_shape);
  Tensor* k_output = context.Output(1, kv_shape);
  Tensor* v_output = context.Output(2, kv_shape);
  Tensor* input_skip_bias_sum = (skip != nullptr && context.OutputCount() > 3)
                                    ? context.Output(3, a->Shape())
                                    : nullptr;
  if (q_output->Shape().Size() == 0) {
    return Status::OK();
  }

  ORT_ENFORCE(norm_scale->Shape().Size() == K_, "norm_scale must have shape [K].");

  const bool would_use_subgroup_unfused =
      WouldApplySubgroupMatrixMatMulNBitsInCurrentDispatch(a,
                                                           K_,
                                                           Nq_,
                                                           block_size_,
                                                           accuracy_level_,
                                                           bits_,
                                                           context,
                                                           q_output);
  const bool would_use_dp4a_unfused =
      !would_use_subgroup_unfused &&
      WouldApplyDP4AMatMulNBitsInCurrentDispatch(a,
                                                 K_,
                                                 Nq_,
                                                 block_size_,
                                                 accuracy_level_,
                                                 context,
                                                 q_output);
  const bool would_use_wide_tile_unfused =
      !would_use_subgroup_unfused &&
      !would_use_dp4a_unfused &&
      WouldApplyWideTileMatMulNBitsInCurrentDispatch(a,
                                                     K_,
                                                     Nq_,
                                                     block_size_,
                                                     bits_);

  if (would_use_subgroup_unfused || would_use_dp4a_unfused || would_use_wide_tile_unfused || M != 1) {
    if (skip != nullptr) {
      return ApplyUnfusedQKVSkipSimplifiedLayerNorm(a,
                                                    skip,
                                                    norm_scale,
                                                    q_b,
                                                    q_scales,
                                                    k_b,
                                                    k_scales,
                                                    v_b,
                                                    v_scales,
                                                    K_,
                                                    Nq_,
                                                    Nkv_,
                                                    block_size_,
                                                    accuracy_level_,
                                                    bits_,
                                                    epsilon_,
                                                    context,
                                                    q_output,
                                                    k_output,
                                                    v_output,
                                                    input_skip_bias_sum);
    }
    return ApplyUnfusedQKVSimplifiedLayerNorm(a,
                                              norm_scale,
                                              q_b,
                                              q_scales,
                                              k_b,
                                              k_scales,
                                              v_b,
                                              v_scales,
                                              K_,
                                              Nq_,
                                              Nkv_,
                                              block_size_,
                                              accuracy_level_,
                                              bits_,
                                              epsilon_,
                                              context,
                                              q_output,
                                              k_output,
                                              v_output);
  }

  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_);
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t block_size_per_col = block_size;
  const uint32_t n_blocks_per_col = (K + block_size_per_col - 1) / block_size_per_col;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(bits_);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);
  constexpr uint32_t kU32Components = 4;
  const uint32_t components_b_with_u32 = components_b * kU32Components;
  const uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;
  const bool single_scale_weights =
      q_scales->Shape().Size() == 1 && k_scales->Shape().Size() == 1 && v_scales->Shape().Size() == 1;

  uint32_t workgroup_size = 128;
  uint32_t tile_size = 8;
  uint32_t tile_size_k_vec = (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;

  const uint32_t elements_in_value_b = components_b * (32u / onnxruntime::narrow<uint32_t>(bits_));
  const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
  const uint32_t k_tile_iterations = K / tile_size_k;

  std::optional<Tensor> input_skip_bias_sum_scratch;
  Tensor* decode_input_skip_bias_sum = input_skip_bias_sum;
  if (skip != nullptr && decode_input_skip_bias_sum == nullptr) {
    input_skip_bias_sum_scratch.emplace(context.CreateGPUTensor(a->DataType(), a->Shape()));
    decode_input_skip_bias_sum = &*input_skip_bias_sum_scratch;
  }

  uint32_t k_unroll_tiles = 1;
  if ((K % tile_size_k) == 0) {
    if (k_tile_iterations >= 8 && std::max(Nq, Nkv) <= 2048 &&
        context.AdapterInfo().vendor != std::string_view{"intel"}) {
      k_unroll_tiles = 4;
    } else if (k_tile_iterations >= 4) {
      k_unroll_tiles = 2;
    }
  }

  const uint32_t num_N_tile = CeilDiv(std::max(Nq, Nkv), tile_size);
  MatMulNBitsQkvDecodeProgram program{tile_size,
                                      single_scale_weights,
                                      tile_size_k_vec,
                                      k_unroll_tiles,
                                      skip != nullptr,
                                      decode_input_skip_bias_sum != nullptr};
  program.SetWorkgroupSize(workgroup_size);
  program.SetDispatchGroupSize(num_N_tile, 1, batch_count);
  program
      .AddInput({a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)});
  if (skip != nullptr) {
    program.AddInput({skip, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)});
  }
  program
      .AddInputs({{norm_scale, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)},
                  {q_b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {q_scales, ProgramTensorMetadataDependency::TypeAndRank},
                  {k_b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {k_scales, ProgramTensorMetadataDependency::TypeAndRank},
                  {v_b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
                  {v_scales, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({{q_output, ProgramTensorMetadataDependency::TypeAndRank},
                   {k_output, ProgramTensorMetadataDependency::TypeAndRank},
                   {v_output, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddUniformVariables({{Nq},
                            {Nkv},
                            {K},
                            {K / components_a},
                            {K_of_b},
                            {block_size},
                            {n_blocks_per_col},
                            {num_N_tile},
                            {batch_count},
                            {skip != nullptr ? onnxruntime::narrow<uint32_t>(skip->Shape().Size()) : 0u},
                            {epsilon_}})
      .CacheHint(Nq,
                 Nkv,
                 K,
                 tile_size,
                 tile_size_k_vec,
                 k_unroll_tiles,
                 single_scale_weights,
                 skip != nullptr,
                 decode_input_skip_bias_sum != nullptr,
                 "decode_qkv_sln");
  if (decode_input_skip_bias_sum != nullptr) {
    program.AddOutput({decode_input_skip_bias_sum,
                       ProgramTensorMetadataDependency::TypeAndRank,
                       static_cast<int>(components_a)});
  }

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
