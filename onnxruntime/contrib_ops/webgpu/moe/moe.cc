// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "contrib_ops/webgpu/moe/moe.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include <algorithm>
#include <optional>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

namespace {

class MoEGateProgram final : public Program<MoEGateProgram> {
 public:
  MoEGateProgram(int k, bool is_fp16, bool normalize_routing_weights)
      : Program<MoEGateProgram>{"MoeGate"},
        k_{k},
        is_fp16_{is_fp16},
        normalize_routing_weights_{normalize_routing_weights} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("router_logits", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("topk_values");
    shader.AddOutput("hiddenstate_for_expert");
    shader.AddOutput("tokencount_for_expert");
    return WGSL_TEMPLATE_APPLY(shader, "moe/gate.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                               WGSL_TEMPLATE_PARAMETER(k, k_),
                               WGSL_TEMPLATE_PARAMETER(normalize_routing_weights, normalize_routing_weights_));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"token_offset", ProgramUniformVariableDataType::Uint32});

 private:
  int k_;
  bool is_fp16_;
  bool normalize_routing_weights_;
};

class MoEHiddenStateGatherProgram final : public Program<MoEHiddenStateGatherProgram> {
 public:
  MoEHiddenStateGatherProgram() : Program<MoEHiddenStateGatherProgram>{"MoeHiddenStateGather"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("hiddenstate_for_expert", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("hidden_state", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("new_hidden_state");
    shader.AddOutput("tokens");
    return WGSL_TEMPLATE_APPLY(shader, "moe/hidden_state_gather.wgsl.template");
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"num_tokens", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32});
};

class MoEZeroTensorProgram final : public Program<MoEZeroTensorProgram> {
 public:
  MoEZeroTensorProgram() : Program<MoEZeroTensorProgram>{"MoeZeroTensor"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddOutput("tensor", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_tensor.wgsl.template");
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
};

class MoEZeroU32Program final : public Program<MoEZeroU32Program> {
 public:
  MoEZeroU32Program() : Program<MoEZeroU32Program>{"MoeZeroU32"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddOutput("output");
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_u32.wgsl.template");
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
};

class MoEExpertMatMulProgram final : public Program<MoEExpertMatMulProgram> {
 public:
  explicit MoEExpertMatMulProgram(bool has_bias)
      : Program<MoEExpertMatMulProgram>{"MoeExpertMatMul"}, has_bias_{has_bias} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("weights", ShaderUsage::UseElementTypeAlias);
    if (has_bias_) {
      shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    }
    shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/expert_matmul.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"inner", ProgramUniformVariableDataType::Uint32},
      {"expert_idx", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
};

class MoEActivationProgram final : public Program<MoEActivationProgram> {
 public:
  MoEActivationProgram(MoEActivationType activation_type, int swiglu_fusion, bool has_fc3)
      : Program<MoEActivationProgram>{"MoeActivation"},
        activation_type_{activation_type},
        swiglu_fusion_{swiglu_fusion},
        has_fc3_{has_fc3} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    if (has_fc3_) {
      shader.AddInput("fc3_input", ShaderUsage::UseElementTypeAlias);
    }
    shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/activation.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(activation, static_cast<int>(activation_type_)),
                               WGSL_TEMPLATE_PARAMETER(has_fc3, has_fc3_),
                               WGSL_TEMPLATE_PARAMETER(swiglu_fusion, swiglu_fusion_));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"swiglu_limit", ProgramUniformVariableDataType::Float32});

 private:
  MoEActivationType activation_type_;
  int swiglu_fusion_;
  bool has_fc3_;
};

class MoEFinalMixProgram final : public Program<MoEFinalMixProgram> {
 public:
  MoEFinalMixProgram() : Program<MoEFinalMixProgram>{"MoeFinalMix"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("fc2_outputs", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("router_values", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("expert_tokens", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/final_mix.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_router_weights, false));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"token_offset", ProgramUniformVariableDataType::Uint32});
};

Status RunExpertMatMul(ComputeContext& context, const Tensor* input, const Tensor* weights,
                       const Tensor* bias, Tensor* output, uint32_t rows, uint32_t cols,
                       uint32_t inner, uint32_t expert_idx) {
  MoEExpertMatMulProgram matmul{bias != nullptr};
  matmul.AddInputs({{input, ProgramTensorMetadataDependency::Type}})
      .AddInputs({{weights, ProgramTensorMetadataDependency::Type}});
  if (bias) {
    matmul.AddInputs({{bias, ProgramTensorMetadataDependency::Type}});
  }
  matmul.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(64)
      .SetDispatchGroupSize((rows * cols + 63) / 64)
      .AddUniformVariables({rows, cols, inner, expert_idx})
      .CacheHint(bias != nullptr);
  return context.RunProgram(matmul);
}

}  // namespace

Status MoE::ComputeInternal(ComputeContext& context) const {
  const Tensor* hidden_state = context.Input<Tensor>(0);
  const Tensor* router_logits = context.Input<Tensor>(1);
  const Tensor* fc1_weights = context.Input<Tensor>(2);
  const Tensor* fc1_bias = context.Input<Tensor>(3);
  const Tensor* fc2_weights = context.Input<Tensor>(4);
  const Tensor* fc2_bias = context.Input<Tensor>(5);
  const Tensor* fc3_weights = context.Input<Tensor>(6);
  const Tensor* fc3_bias = context.Input<Tensor>(7);

  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == MoEActivationType::SwiGLU && swiglu_fusion == 0 && fc3_weights == nullptr) {
    swiglu_fusion = 1;
  }
  const bool is_fused_swiglu = activation_type_ == MoEActivationType::SwiGLU &&
                               swiglu_fusion != 0 && fc3_weights == nullptr;

  MoEParameters params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      params, hidden_state, router_logits, fc1_weights, fc1_bias, nullptr, nullptr,
      fc2_weights, fc2_bias, nullptr, nullptr, fc3_weights, fc3_bias, nullptr, nullptr,
      1, is_fused_swiglu));
  ORT_RETURN_IF_NOT(k_ > 0 && k_ <= params.num_experts,
                    "MoE requires 0 < k <= num_experts, got k=", k_,
                    " and num_experts=", params.num_experts);

  const auto dtype = hidden_state->DataType();
  const auto dtype_uint32 = DataTypeImpl::GetType<uint32_t>();
  const bool is_fp16 = dtype == DataTypeImpl::GetType<MLFloat16>();
  const uint32_t num_experts = static_cast<uint32_t>(params.num_experts);
  const uint32_t hidden_size = static_cast<uint32_t>(params.hidden_size);
  const uint32_t inter_size = static_cast<uint32_t>(params.inter_size);
  const uint32_t fc1_cols = is_fused_swiglu ? 2 * inter_size : inter_size;
  constexpr int max_tokens = 2 * 1024;

  Tensor* output = context.Output(0, hidden_state->Shape());
  if (params.num_rows == 0) {
    return Status::OK();
  }
  const uint32_t output_vec4_size = static_cast<uint32_t>((hidden_state->Shape().Size() + 3) / 4);
  MoEZeroTensorProgram zero;
  zero.AddOutput({output, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten, 4})
      .SetDispatchGroupSize((output_vec4_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({output_vec4_size});
  ORT_RETURN_IF_ERROR(context.RunProgram(zero));

  for (int token_offset = 0; token_offset < params.num_rows; token_offset += max_tokens) {
    const uint32_t num_tokens = static_cast<uint32_t>(
        std::min<int64_t>(max_tokens, params.num_rows - token_offset));
    Tensor router_values = context.CreateGPUTensor(dtype, TensorShape({num_tokens, num_experts}));
    Tensor gate_counts = context.CreateGPUTensor(dtype_uint32, TensorShape({num_experts}));
    Tensor gate_hidden = context.CreateGPUTensor(dtype_uint32, TensorShape({num_experts, num_tokens}));

    MoEZeroU32Program zero_counts;
    zero_counts.AddOutput({&gate_counts, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((num_experts + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({num_experts});
    ORT_RETURN_IF_ERROR(context.RunProgram(zero_counts));

    MoEGateProgram gate{k_, is_fp16, normalize_routing_weights_};
    gate.AddInputs({{router_logits, ProgramTensorMetadataDependency::Type}})
        .AddOutput({&router_values, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_hidden, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_counts, ProgramTensorMetadataDependency::None, ProgramOutput::Atomic})
        .SetWorkgroupSize(num_experts)
        .SetDispatchGroupSize(num_tokens)
        .AddUniformVariables({num_tokens, num_experts, static_cast<uint32_t>(token_offset)})
        .CacheHint(k_, is_fp16 ? "fp16" : "fp32", normalize_routing_weights_);
    ORT_RETURN_IF_ERROR(context.RunProgram(gate));

    Tensor gate_counts_cpu = context.CreateCPUTensor(dtype_uint32, TensorShape({num_experts}));
    ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(gate_counts, gate_counts_cpu));
    for (uint32_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      const uint32_t used_by = gate_counts_cpu.Data<uint32_t>()[expert_idx];
      if (used_by == 0) {
        continue;
      }

      Tensor expert_hidden = context.CreateGPUTensor(dtype, TensorShape({used_by, hidden_size}));
      Tensor expert_tokens = context.CreateGPUTensor(dtype_uint32, TensorShape({used_by}));
      MoEHiddenStateGatherProgram gather;
      gather.AddInputs({{&gate_hidden, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{hidden_state, ProgramTensorMetadataDependency::Type, 1}})
          .AddOutput({&expert_hidden, ProgramTensorMetadataDependency::None, 1})
          .AddOutput({&expert_tokens, ProgramTensorMetadataDependency::None})
          .SetDispatchGroupSize(used_by)
          .AddUniformVariables({expert_idx, num_experts, num_tokens, hidden_size});
      ORT_RETURN_IF_ERROR(context.RunProgram(gather));

      Tensor fc1_output = context.CreateGPUTensor(dtype, TensorShape({used_by, fc1_cols}));
      ORT_RETURN_IF_ERROR(RunExpertMatMul(context, &expert_hidden, fc1_weights, fc1_bias,
                                          &fc1_output, used_by, fc1_cols, hidden_size, expert_idx));

      std::optional<Tensor> fc3_output;
      if (fc3_weights) {
        fc3_output.emplace(context.CreateGPUTensor(dtype, TensorShape({used_by, inter_size})));
        ORT_RETURN_IF_ERROR(RunExpertMatMul(context, &expert_hidden, fc3_weights, fc3_bias,
                                            &*fc3_output, used_by, inter_size, hidden_size, expert_idx));
      }

      Tensor activated = context.CreateGPUTensor(dtype, TensorShape({used_by, inter_size}));
      MoEActivationProgram activation{activation_type_, swiglu_fusion, fc3_output.has_value()};
      activation.AddInputs({{&fc1_output, ProgramTensorMetadataDependency::Type}});
      if (fc3_output) {
        activation.AddInputs({{&*fc3_output, ProgramTensorMetadataDependency::Type}});
      }
      activation.AddOutput({&activated, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(128)
          .SetDispatchGroupSize((used_by * inter_size + 127) / 128)
          .AddUniformVariables({used_by, inter_size, activation_alpha_, activation_beta_, swiglu_limit_})
          .CacheHint(static_cast<int>(activation_type_), swiglu_fusion, fc3_output.has_value());
      ORT_RETURN_IF_ERROR(context.RunProgram(activation));

      Tensor fc2_output = context.CreateGPUTensor(dtype, TensorShape({used_by, hidden_size}));
      ORT_RETURN_IF_ERROR(RunExpertMatMul(context, &activated, fc2_weights, fc2_bias,
                                          &fc2_output, used_by, hidden_size, inter_size, expert_idx));

      MoEFinalMixProgram mix;
      mix.AddInputs({{&fc2_output, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&router_values, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&expert_tokens, ProgramTensorMetadataDependency::Type}})
          .AddOutput({output, ProgramTensorMetadataDependency::None})
          .SetDispatchGroupSize(used_by)
          .AddUniformVariables({hidden_size, num_experts, expert_idx, static_cast<uint32_t>(token_offset)});
      ORT_RETURN_IF_ERROR(context.RunProgram(mix));
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    MoE,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    MoE);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
