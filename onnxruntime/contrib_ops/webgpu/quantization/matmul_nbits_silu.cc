// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_silu.h"

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

constexpr uint32_t kFusedDecodeFastPathBits = 4u;
constexpr uint32_t kFusedDecodeFastPathBlockSize = 32u;

bool IsFusedDecodeFastPathEnabled() {
  return true;
}

Status WouldApplyGenericMatMulNBitsInCurrentDispatch(const Tensor* a,
                                                     int64_t K_op,
                                                     int64_t N_op,
                                                     int64_t block_size_op,
                                                     int64_t accuracy_level,
                                                     int64_t nbits,
                                                     onnxruntime::webgpu::ComputeContext& context,
                                                     Tensor* y,
                                                     bool& would_apply_generic) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));

  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_op);

  const bool single_scale_weights = (block_size == K * N);
  const uint32_t block_size_per_col = single_scale_weights ? K : block_size;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);

#if !defined(__wasm__)
  int32_t subgroup_matrix_config_index = -1;
  const bool would_apply_subgroup =
      (M >= kMinMForTileOptimization) &&
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
  if (would_apply_subgroup) {
    would_apply_generic = false;
    return Status::OK();
  }
#endif

  const bool would_apply_dp4a =
      ((M >= kMinMForTileOptimization ||
        y->DataType() == DataTypeImpl::GetType<float>() ||
        context.AdapterInfo().vendor == std::string_view{"qualcomm"}) &&
       CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a));
  if (would_apply_dp4a) {
    would_apply_generic = false;
    return Status::OK();
  }

  const bool would_apply_wide_tile = block_size == 32 &&
                                     components_a == 4 &&
                                     components_b == 4 &&
                                     nbits != 2 &&
                                     M >= kMinMForTileOptimization;
  would_apply_generic = !would_apply_wide_tile;
  return Status::OK();
}

class MatMulNBitsSiluMulDecodeProgram final : public Program<MatMulNBitsSiluMulDecodeProgram> {
 public:
  MatMulNBitsSiluMulDecodeProgram(uint32_t tile_size,
                                  bool has_gate_bias,
                                  bool has_up_bias,
                                  bool single_scale_weights,
                                  uint32_t tile_size_k_vec)
      : Program{"MatMulNBitsSiluMulDecode"},
        tile_size_(tile_size),
        has_gate_bias_(has_gate_bias),
        has_up_bias_(has_up_bias),
        single_scale_weights_(single_scale_weights),
        tile_size_k_vec_(tile_size_k_vec) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
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

    const uint32_t components_a = a.NumComponents();
    const uint32_t components_b = gate_b.NumComponents() / 4;
    const uint32_t tile_size_k_vec = tile_size_k_vec_;
    const uint32_t elements_in_value_b = components_b * 8u;
    const uint32_t tile_size_k = tile_size_k_vec * elements_in_value_b;
    const uint32_t a_length_per_tile = tile_size_k / components_a;
    const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;

    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_nbits_silu_mul.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(a_length_per_tile, a_length_per_tile),
                               WGSL_TEMPLATE_PARAMETER(component_a, components_a),
                               WGSL_TEMPLATE_PARAMETER(component_b, components_b),
                               WGSL_TEMPLATE_PARAMETER(elements_in_value_b, elements_in_value_b),
                               WGSL_TEMPLATE_PARAMETER(has_gate_bias, has_gate_bias_),
                               WGSL_TEMPLATE_PARAMETER(has_up_bias, has_up_bias_),
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
      {"batch_count", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_;
  bool has_gate_bias_;
  bool has_up_bias_;
  bool single_scale_weights_;
  uint32_t tile_size_k_vec_;
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
  const Tensor* gate_b = context.Input<Tensor>(1);
  const Tensor* gate_scales = context.Input<Tensor>(2);
  const Tensor* gate_bias = context.Input<Tensor>(3);
  const Tensor* up_b = context.Input<Tensor>(4);
  const Tensor* up_scales = context.Input<Tensor>(5);
  const Tensor* up_bias = context.Input<Tensor>(6);

  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  const auto output_shape = helper.OutputShape();
  const uint32_t batch_count = onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size());
  const uint32_t M = onnxruntime::narrow<uint32_t>(helper.M());
  const uint32_t N = onnxruntime::narrow<uint32_t>(helper.N());
  const uint32_t K = onnxruntime::narrow<uint32_t>(helper.K());
  const uint32_t block_size = onnxruntime::narrow<uint32_t>(block_size_);

  Tensor* y = context.Output(0, output_shape);
  const uint32_t data_size = onnxruntime::narrow<uint32_t>(y->Shape().Size());
  if (data_size == 0) {
    return Status::OK();
  }

  bool gate_would_use_generic_matmul = false;
  bool up_would_use_generic_matmul = false;
  ORT_RETURN_IF_ERROR(WouldApplyGenericMatMulNBitsInCurrentDispatch(a,
                                                                    K_,
                                                                    N_,
                                                                    block_size_,
                                                                    accuracy_level_,
                                                                    bits_,
                                                                    context,
                                                                    y,
                                                                    gate_would_use_generic_matmul));
  ORT_RETURN_IF_ERROR(WouldApplyGenericMatMulNBitsInCurrentDispatch(a,
                                                                    K_,
                                                                    N_,
                                                                    block_size_,
                                                                    accuracy_level_,
                                                                    bits_,
                                                                    context,
                                                                    y,
                                                                    up_would_use_generic_matmul));

  if (IsFusedDecodeFastPathEnabled() && M == 1 && bits_ == kFusedDecodeFastPathBits &&
      block_size == kFusedDecodeFastPathBlockSize && gate_would_use_generic_matmul &&
      up_would_use_generic_matmul) {
    ORT_ENFORCE(bits_ == kFusedDecodeFastPathBits,
                "MatMulNBitsSiluMulDecodeProgram is specialized for 4-bit weights only.");
    ORT_ENFORCE(block_size == kFusedDecodeFastPathBlockSize,
                "MatMulNBitsSiluMulDecodeProgram is specialized for block_size=32 only.");

    const bool has_gate_bias = gate_bias != nullptr;
    const bool has_up_bias = up_bias != nullptr;
    const bool single_scale_weights = (block_size == K * N);
    const uint32_t block_size_per_col = single_scale_weights ? K : block_size;
    const uint32_t n_blocks_per_col = (K + block_size_per_col - 1) / block_size_per_col;
    const uint32_t blob_size = (block_size_per_col / 8) * onnxruntime::narrow<uint32_t>(bits_);
    const uint32_t blob_size_in_words = blob_size / 4;
    const uint32_t components_a = GetMaxComponents(K);
    const uint32_t components_b = GetMaxComponents(blob_size_in_words);
    constexpr uint32_t kU32Components = 4;
    const uint32_t components_b_with_u32 = components_b * kU32Components;
    const uint32_t K_of_b = (n_blocks_per_col * blob_size) / components_b_with_u32;
    constexpr uint32_t workgroup_size = 128;
    constexpr uint32_t tile_size = 8;
    const uint32_t tile_size_k_vec =
        (context.AdapterInfo().vendor == std::string_view{"intel"}) ? 16u : 32u;
    const uint32_t num_N_tile = CeilDiv(N, tile_size);

    MatMulNBitsSiluMulDecodeProgram program{tile_size,
                                            has_gate_bias,
                                            has_up_bias,
                                            single_scale_weights,
                                            tile_size_k_vec};
    program.SetWorkgroupSize(workgroup_size);
    program.SetDispatchGroupSize(num_N_tile, 1, batch_count);
    program
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_a)},
                    {gate_b, ProgramTensorMetadataDependency::TypeAndRank, static_cast<int>(components_b_with_u32)},
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
                              {batch_count}})
        .CacheHint(single_scale_weights, has_gate_bias, has_up_bias, tile_size_k_vec, "decode_4bit");
    if (has_gate_bias) {
      program.AddInput({gate_bias, ProgramTensorMetadataDependency::None});
    }
    if (has_up_bias) {
      program.AddInput({up_bias, ProgramTensorMetadataDependency::None});
    }

    return context.RunProgram(program);
  }

  Tensor gate_output = context.CreateGPUTensor(a->DataType(), output_shape);
  Tensor up_output = context.CreateGPUTensor(a->DataType(), output_shape);

  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(a, gate_b, gate_scales, nullptr, gate_bias, K_, N_, block_size_, accuracy_level_, bits_, context, &gate_output));
  ORT_RETURN_IF_ERROR(ApplyMatMulNBits(a, up_b, up_scales, nullptr, up_bias, K_, N_, block_size_, accuracy_level_, bits_, context, &up_output));

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

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
