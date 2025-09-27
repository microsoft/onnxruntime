// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/moe/qmoe.h"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "core/providers/webgpu/math/gemm_packed.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

class GateProgram final : public Program<GateProgram> {
 public:
  GateProgram(int k, bool is_fp16) : Program<GateProgram>{"Gate"}, k_{k}, is_fp16_{is_fp16} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("topk_values");
    shader.AddOutput("topk_indices_fc1");
    shader.AddOutput("topk_indices_fc2");

    return WGSL_TEMPLATE_APPLY(shader, "moe/qmoe_gate.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                             WGSL_TEMPLATE_PARAMETER(k, k_));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32});

 private:
  int k_;
  bool is_fp16_;
};

class SwigLuProgram final : public Program<SwigLuProgram> {
 public:
  SwigLuProgram() : Program<SwigLuProgram>{"SwigLu"} {
  };

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

    return WGSL_TEMPLATE_APPLY(shader, "moe/swiglu.wgsl.template");
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"swiglu_limit", ProgramUniformVariableDataType::Float32});

 private:
};

class QMoEFinalMixProgram final : public Program<QMoEFinalMixProgram> {
 public:
  QMoEFinalMixProgram() : Program<QMoEFinalMixProgram>{"QMoEFinalMix"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("fc2_outputs", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("router_values", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

    return WGSL_TEMPLATE_APPLY(shader, "moe/qmoe_final_mix.wgsl.template");
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"output_size", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"k", ProgramUniformVariableDataType::Uint32});

 private:
};

// #define GSDEBUG 1

Status QMoE::ComputeInternal(ComputeContext& context) const {
  const Tensor* input = context.Input<Tensor>(0);
  const Tensor* router_logits = context.Input<Tensor>(1);
  // fc1 is up
  const Tensor* fc1_experts_weights = context.Input<Tensor>(2);
  const Tensor* fc1_scales = context.Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context.Input<Tensor>(4);
  // fc2 is down
  const Tensor* fc2_experts_weights = context.Input<Tensor>(5);
  const Tensor* fc2_scales = context.Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context.Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context.Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context.Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context.Input<Tensor>(10);

  MoEParameters moe_params;

  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_logits,
      fc1_experts_weights, fc1_experts_bias_optional, fc1_scales,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional,
      expert_weight_bits_ == 4 ? 2 : 1,
      activation_type_ == MoEActivationType::SwiGLU, block_size_));

  const auto& input_shape = input->Shape();

  // SwiGLU validation - FC3 not supported (match CUDA FasterTransformer)
  bool is_swiglu = (activation_type_ == MoEActivationType::SwiGLU);
  if (is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SwiGLU activation is not supported with fc3. Gate weights should be concatenated with FC1 weights.");
  }
  if (!is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 gating is not yet implemented for non-SwiGLU activations on CPU.");
  }

  const int64_t fc1_output_size = is_swiglu && swiglu_fusion_ > 0 ? 2 * moe_params.inter_size : moe_params.inter_size;
  const int64_t total_output_size = moe_params.num_rows * moe_params.hidden_size;
  const bool is_fp16 = input->DataType() == DataTypeImpl::GetType<MLFloat16>();
  const auto dtype = is_fp16 ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>();

  //
  // Step 1: run the gate to get router indices and values
  //
  TensorShape gate_idx_shape({moe_params.num_rows, k_, 4});
  TensorShape gate_value_shape({moe_params.num_rows, k_});
  Tensor router_idx_fc1 = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), gate_idx_shape);
  Tensor router_idx_fc2 = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), gate_idx_shape);
  Tensor router_values = context.CreateGPUTensor(dtype, gate_value_shape);

  GateProgram gate{k_, is_fp16};
  gate
      .AddInputs({{router_logits, ProgramTensorMetadataDependency::Type}})
      .AddOutput({&router_values, ProgramTensorMetadataDependency::None})
      .AddOutput({&router_idx_fc1, ProgramTensorMetadataDependency::None, 4})
      .AddOutput({&router_idx_fc2, ProgramTensorMetadataDependency::None, 4})
      .SetWorkgroupSize(static_cast<uint32_t>(moe_params.num_experts))
      .SetDispatchGroupSize(static_cast<uint32_t>(moe_params.num_rows))
      .AddUniformVariables({static_cast<uint32_t>(moe_params.num_rows),
                            static_cast<uint32_t>(moe_params.num_experts)})
      .CacheHint(k_, is_fp16 ? "fp16" : "fp32");

  ORT_RETURN_IF_ERROR(context.RunProgram(gate));

  const int64_t K_fc1 = moe_params.hidden_size; // left_shape[left_num_dims - 1]
  const int64_t N_fc1 = fc1_output_size;        // right_shape[right_num_dims - 1]
  const int64_t K_fc2 = moe_params.inter_size;  // left_shape[left_num_dims - 1]
  const int64_t N_fc2 = moe_params.inter_size;  // right_shape[right_num_dims - 1]
  const int64_t accuracy_level = 4;
  const int64_t block_size = (block_size_ != 0) ? block_size_ : fc1_experts_weights->Shape()[2];
  Status status;

  //
  // Step 2: matmul the input with fc1 (gate_up) of the selected experts
  //
  TensorShape fc1_output_shape({k_, moe_params.num_rows, fc1_output_size});
  Tensor fc1_outputs = context.CreateGPUTensor(dtype, fc1_output_shape);

  status = ApplyMatMulNBits(input, fc1_experts_weights, fc1_scales, nullptr, fc1_experts_bias_optional,
                            K_fc1, N_fc1, block_size, accuracy_level, expert_weight_bits_, context,
                            &fc1_outputs, &router_idx_fc1);
  ORT_RETURN_IF_ERROR(status);

#ifdef GSDEBUG
  DumpTensor<MLFloat16>(input, "input", context);
  DumpTensor<MLFloat16>(router_logits, "router_logits", context);
  DumpTensor<MLFloat16>(&router_values, "router_values", context);
  DumpTensor<MLFloat16>(&fc1_outputs, "fc1_outputs", context);
  DumpTensor<uint32_t>(&router_idx_fc1, "router_idx_fc1", context);
  DumpTensor<uint32_t>(&router_idx_fc2, "router_idx_fc2", context);
#endif

  //
  // Step 3: apply swigly
  //
  TensorShape fc1_activated_shape({k_, moe_params.num_rows, moe_params.inter_size});
  Tensor fc1_activated = context.CreateGPUTensor(dtype, fc1_activated_shape);
  if (is_swiglu) {
    SwigLuProgram swiglu;
    swiglu
        .AddInputs({{&fc1_outputs, ProgramTensorMetadataDependency::Type, 2}})
        .AddOutput({&fc1_activated, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize(((k_ * moe_params.num_rows * moe_params.inter_size) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({static_cast<uint32_t>(k_ * moe_params.num_rows), static_cast<uint32_t>(moe_params.inter_size), activation_alpha_, activation_beta_, swiglu_limit_});
    ORT_RETURN_IF_ERROR(context.RunProgram(swiglu));
  } else {
    ORT_THROW("only swiglu is supported for now.");
  }

#ifdef GSDEBUG
  DumpTensor<MLFloat16>(&fc1_activated, "fc1_activated", context);
#endif

  //
  // Step 4: multiply fc1_activated with fc2 (gate_down) of the selected experts
  //
  TensorShape fc2_output_shape({k_, moe_params.num_rows, moe_params.inter_size});
  Tensor fc2_outputs = context.CreateGPUTensor(dtype, fc2_output_shape);

  status = ApplyMatMulNBits(&fc1_activated, fc2_experts_weights, fc2_scales, nullptr, fc2_experts_bias_optional,
                            K_fc2, N_fc2, block_size, accuracy_level, expert_weight_bits_, context,
                            &fc2_outputs, &router_idx_fc2);
  ORT_RETURN_IF_ERROR(status);

#ifdef GSDEBUG
  DumpTensor<MLFloat16>(&fc2_outputs, "fc2_outputs", context);
#endif

  //
  // Step 5: multiply fc2_outputs with router_values and accumulate
  //
  Tensor* output_tensor = context.Output(0, input_shape);

  QMoEFinalMixProgram final_mix;
  final_mix
      .AddInputs({{&fc2_outputs, ProgramTensorMetadataDependency::Type}})
      .AddInputs({{&router_values, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((total_output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({static_cast<uint32_t>(total_output_size),
                            static_cast<uint32_t>(moe_params.hidden_size),
                            static_cast<uint32_t>(k_)});

  ORT_RETURN_IF_ERROR(context.RunProgram(final_mix));

#ifdef GSDEBUG
  DumpTensor<MLFloat16>(output_tensor, "output_tensor", context);
#endif

  return Status::OK();
}

namespace {
const std::vector<MLDataType>& QMoET1Constraint() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<uint8_t>()};
  return types;
}
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T1", QMoET1Constraint())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    QMoE);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
