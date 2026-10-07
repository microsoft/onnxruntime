// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/configured_program.h"
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

#define WEBGPU_MO_E_GATE_PROGRAM_CONFIG(F) \
  F(int, k_)                               \
  F(bool, is_fp16_)                        \
  F(bool, normalize_routing_weights_)

struct MoEGateProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MO_E_GATE_PROGRAM_CONFIG);
    Config(int k, bool is_fp16, bool normalize_routing_weights)
        : k_{k}, is_fp16_{is_fp16}, normalize_routing_weights_{normalize_routing_weights} {}
  };
  static constexpr std::string_view name = "MoeGate";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader) {
    const auto& router_logits = shader.AddInput("router_logits", ShaderUsage::UseElementTypeAlias);
    const auto& topk_values = shader.AddOutput("topk_values");
    const auto& hiddenstate_for_expert = shader.AddOutput("hiddenstate_for_expert");
    shader.AddOutput("tokencount_for_expert");
    return WGSL_TEMPLATE_APPLY(shader, "moe/gate.wgsl.template", WGSL_TEMPLATE_PARAMETER(has_router_weights, false),
                               WGSL_TEMPLATE_PARAMETER(is_fp16, config.is_fp16_), WGSL_TEMPLATE_PARAMETER(k, config.k_),
                               WGSL_TEMPLATE_PARAMETER(normalize_routing_weights, config.normalize_routing_weights_),
                               WGSL_TEMPLATE_VARIABLE(hiddenstate_for_expert, hiddenstate_for_expert),
                               WGSL_TEMPLATE_VARIABLE(router_logits, router_logits),
                               WGSL_TEMPLATE_VARIABLE(router_weights, router_logits),
                               WGSL_TEMPLATE_VARIABLE(topk_values, topk_values));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"token_offset", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_MO_E_GATE_PROGRAM_CONFIG

using MoEGateProgram = ConfiguredProgram<MoEGateProgramShader>;

#define WEBGPU_MO_E_HIDDEN_STATE_GATHER_PROGRAM_CONFIG(F)

struct MoEHiddenStateGatherProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MO_E_HIDDEN_STATE_GATHER_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "MoeHiddenStateGather";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader) {
    const auto& hiddenstate_for_expert = shader.AddInput("hiddenstate_for_expert", ShaderUsage::UseElementTypeAlias);
    const auto& hidden_state = shader.AddInput("hidden_state", ShaderUsage::UseElementTypeAlias);
    const auto& new_hidden_state = shader.AddOutput("new_hidden_state");
    const auto& tokens = shader.AddOutput("tokens");
    return WGSL_TEMPLATE_APPLY(shader, "moe/hidden_state_gather.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(hidden_state, hidden_state),
                               WGSL_TEMPLATE_VARIABLE(hiddenstate_for_expert, hiddenstate_for_expert),
                               WGSL_TEMPLATE_VARIABLE(new_hidden_state, new_hidden_state),
                               WGSL_TEMPLATE_VARIABLE(tokens, tokens));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"num_tokens", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_MO_E_HIDDEN_STATE_GATHER_PROGRAM_CONFIG

using MoEHiddenStateGatherProgram = ConfiguredProgram<MoEHiddenStateGatherProgramShader>;

#define WEBGPU_MO_E_ZERO_TENSOR_PROGRAM_CONFIG(F)

struct MoEZeroTensorProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MO_E_ZERO_TENSOR_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "MoeZeroTensor";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader) {
    const auto& tensor = shader.AddOutput("tensor", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_tensor.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(tensor, tensor));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_MO_E_ZERO_TENSOR_PROGRAM_CONFIG

using MoEZeroTensorProgram = ConfiguredProgram<MoEZeroTensorProgramShader>;

#define WEBGPU_MO_E_ZERO_U32_PROGRAM_CONFIG(F)

struct MoEZeroU32ProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MO_E_ZERO_U32_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "MoeZeroU32";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader) {
    const auto& output = shader.AddOutput("output");
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_u32.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(output, output));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_MO_E_ZERO_U32_PROGRAM_CONFIG

using MoEZeroU32Program = ConfiguredProgram<MoEZeroU32ProgramShader>;

#define WEBGPU_MO_E_EXPERT_MAT_MUL_PROGRAM_CONFIG(F) F(bool, has_bias_)

struct MoEExpertMatMulProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MO_E_EXPERT_MAT_MUL_PROGRAM_CONFIG);
    Config(bool has_bias) : has_bias_{has_bias} {}
  };
  static constexpr std::string_view name = "MoeExpertMatMul";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader) {
    const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    const auto& weights = shader.AddInput("weights", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* bias = &weights;
    if (config.has_bias_) {
      bias = &shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/expert_matmul.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_bias, config.has_bias_), WGSL_TEMPLATE_VARIABLE(bias, *bias),
                               WGSL_TEMPLATE_VARIABLE(input, input), WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(weights, weights));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"inner", ProgramUniformVariableDataType::Uint32},
      {"expert_idx", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_MO_E_EXPERT_MAT_MUL_PROGRAM_CONFIG

using MoEExpertMatMulProgram = ConfiguredProgram<MoEExpertMatMulProgramShader>;

#define WEBGPU_MO_E_ACTIVATION_PROGRAM_CONFIG(F) \
  F(MoEActivationType, activation_type_)         \
  F(int, swiglu_fusion_)                         \
  F(bool, has_fc3_)

struct MoEActivationProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MO_E_ACTIVATION_PROGRAM_CONFIG);
    Config(MoEActivationType activation_type, int swiglu_fusion, bool has_fc3)
        : activation_type_{activation_type}, swiglu_fusion_{swiglu_fusion}, has_fc3_{has_fc3} {}
  };
  static constexpr std::string_view name = "MoeActivation";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader) {
    const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* fc3_input = &input;
    if (config.has_fc3_) {
      fc3_input = &shader.AddInput("fc3_input", ShaderUsage::UseElementTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/activation.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(activation, static_cast<int>(config.activation_type_)),
                               WGSL_TEMPLATE_PARAMETER(has_fc3, config.has_fc3_),
                               WGSL_TEMPLATE_PARAMETER(swiglu_fusion, config.swiglu_fusion_),
                               WGSL_TEMPLATE_VARIABLE(fc3_input, *fc3_input), WGSL_TEMPLATE_VARIABLE(input, input),
                               WGSL_TEMPLATE_VARIABLE(output, output));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"swiglu_limit", ProgramUniformVariableDataType::Float32});
};
#undef WEBGPU_MO_E_ACTIVATION_PROGRAM_CONFIG

using MoEActivationProgram = ConfiguredProgram<MoEActivationProgramShader>;

#define WEBGPU_MO_E_FINAL_MIX_PROGRAM_CONFIG(F)

struct MoEFinalMixProgramShader {
  struct Config final {
    WEBGPU_CONFIG_MEMBERS(WEBGPU_MO_E_FINAL_MIX_PROGRAM_CONFIG);
    Config() {}
  };
  static constexpr std::string_view name = "MoeFinalMix";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& shader) {
    const auto& fc2_outputs = shader.AddInput("fc2_outputs", ShaderUsage::UseElementTypeAlias);
    const auto& router_values = shader.AddInput("router_values", ShaderUsage::UseElementTypeAlias);
    const auto& expert_tokens = shader.AddInput("expert_tokens", ShaderUsage::UseElementTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/final_mix.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(expert_tokens, expert_tokens),
                               WGSL_TEMPLATE_VARIABLE(fc2_outputs, fc2_outputs),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(router_values, router_values));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"token_offset", ProgramUniformVariableDataType::Uint32});
};
#undef WEBGPU_MO_E_FINAL_MIX_PROGRAM_CONFIG

using MoEFinalMixProgram = ConfiguredProgram<MoEFinalMixProgramShader>;

Status RunExpertMatMul(ComputeContext& context, const Tensor* input, const Tensor* weights,
                       const Tensor* bias, Tensor* output, uint32_t rows, uint32_t cols,
                       uint32_t inner, uint32_t expert_idx) {
  MoEExpertMatMulProgram matmul{bias != nullptr};
  matmul.AddInputs({{input, ProgramTensorMetadataDependency::None}})
      .AddInputs({{weights, ProgramTensorMetadataDependency::None}});
  if (bias) {
    matmul.AddInputs({{bias, ProgramTensorMetadataDependency::None}});
  }
  matmul.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(64)
      .SetDispatchGroupSize((rows * cols + 63) / 64)
      .AddUniformVariables({rows, cols, inner, expert_idx});
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
  Tensor* output = context.Output(0, hidden_state->Shape());
  if (params.num_rows == 0) {
    return Status::OK();
  }
  const auto& device_limits = context.DeviceLimits();
  ORT_RETURN_IF_NOT(num_experts <= device_limits.maxComputeWorkgroupSizeX &&
                        num_experts <= device_limits.maxComputeInvocationsPerWorkgroup,
                    "WebGPU MoE requires num_experts to fit in one workgroup; got ", num_experts,
                    ", maxComputeWorkgroupSizeX=", device_limits.maxComputeWorkgroupSizeX,
                    ", maxComputeInvocationsPerWorkgroup=", device_limits.maxComputeInvocationsPerWorkgroup, ".");
  const uint32_t hidden_size = static_cast<uint32_t>(params.hidden_size);
  const uint32_t inter_size = static_cast<uint32_t>(params.inter_size);
  const uint32_t fc1_cols = is_fused_swiglu ? 2 * inter_size : inter_size;
  constexpr int max_tokens = 2 * 1024;

  const uint32_t output_vec4_size = static_cast<uint32_t>((hidden_state->Shape().Size() + 3) / 4);
  MoEZeroTensorProgram zero;
  zero.AddOutput({output, ProgramTensorMetadataDependency::None, ProgramOutput::Flatten, 4})
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
    gate.AddInputs({{router_logits, ProgramTensorMetadataDependency::None}})
        .AddOutput({&router_values, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_hidden, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_counts, ProgramTensorMetadataDependency::None, ProgramOutput::Atomic})
        .SetWorkgroupSize(num_experts)
        .SetDispatchGroupSize(num_tokens)
        .AddUniformVariables({num_tokens, num_experts, static_cast<uint32_t>(token_offset)});
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
      gather.AddInputs({{&gate_hidden, ProgramTensorMetadataDependency::None}})
          .AddInputs({{hidden_state, ProgramTensorMetadataDependency::None, 1}})
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
      activation.AddInputs({{&fc1_output, ProgramTensorMetadataDependency::None}});
      if (fc3_output) {
        activation.AddInputs({{&*fc3_output, ProgramTensorMetadataDependency::None}});
      }
      activation.AddOutput({&activated, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(128)
          .SetDispatchGroupSize((used_by * inter_size + 127) / 128)
          .AddUniformVariables({used_by, inter_size, activation_alpha_, activation_beta_, swiglu_limit_});
      ORT_RETURN_IF_ERROR(context.RunProgram(activation));

      Tensor fc2_output = context.CreateGPUTensor(dtype, TensorShape({used_by, hidden_size}));
      ORT_RETURN_IF_ERROR(RunExpertMatMul(context, &activated, fc2_weights, fc2_bias,
                                          &fc2_output, used_by, hidden_size, inter_size, expert_idx));

      MoEFinalMixProgram mix;
      mix.AddInputs({{&fc2_output, ProgramTensorMetadataDependency::None}})
          .AddInputs({{&router_values, ProgramTensorMetadataDependency::None}})
          .AddInputs({{&expert_tokens, ProgramTensorMetadataDependency::None}})
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
