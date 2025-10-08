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
  GateProgram(int k, bool is_fp16) : Program<GateProgram>{"QmoeGate"}, k_{k}, is_fp16_{is_fp16} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("hidden_state", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("topk_values");
    shader.AddOutput("hiddenstate_for_expert");
    shader.AddOutput("tokencount_for_expert");

    return WGSL_TEMPLATE_APPLY(shader, "moe/gate.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                             WGSL_TEMPLATE_PARAMETER(k, k_)
                            );
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"token_offset", ProgramUniformVariableDataType::Uint32}
    );

 private:
  int k_;
  bool is_fp16_;
};

class HiddenStateGatherProgram final : public Program<HiddenStateGatherProgram> {
 public:
  HiddenStateGatherProgram() : Program<HiddenStateGatherProgram>{"QmoeHiddenStateGather"} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddInput("hiddenstate_for_expert", ShaderUsage::UseElementTypeAlias);
    shader.AddInput("hidden_state", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("new_hidden_state");
    shader.AddOutput("tokens");

    return WGSL_TEMPLATE_APPLY(shader, "moe/hidden_state_gather.wgsl.template");
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"num_tokens", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32}
    );

 private:
};

class ZeroTensorProgram final : public Program<ZeroTensorProgram> {
 public:
  ZeroTensorProgram() : Program<ZeroTensorProgram>{"QmoeZeroTensor"} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    shader.AddOutput("tensor", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_tensor.wgsl.template");
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32}
    );

 private:
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
    shader.AddInput("expert_tokens", ShaderUsage::UseElementTypeAlias);
    shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

    return WGSL_TEMPLATE_APPLY(shader, "moe/final_mix.wgsl.template");
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"used_by", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"expert_idx", ProgramUniformVariableDataType::Uint32});
 private:
};

#define GSDEBUG 1
#define DISPATCH(x, y) ((x + y - 1) / y)

Status QMoE::ComputeInternal(ComputeContext& context) const {
  const Tensor* hidden_state = context.Input<Tensor>(0);
  const Tensor* router_logits = context.Input<Tensor>(1);
  // fc1 is gate_up_proj
  const Tensor* fc1_experts_weights = context.Input<Tensor>(2);
  const Tensor* fc1_scales = context.Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context.Input<Tensor>(4);
  // fc2 is gate_down_proj
  const Tensor* fc2_experts_weights = context.Input<Tensor>(5);
  const Tensor* fc2_scales = context.Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context.Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context.Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context.Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context.Input<Tensor>(10);

  MoEParameters moe_params;

  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, hidden_state, router_logits,
      fc1_experts_weights, fc1_experts_bias_optional, fc1_scales,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional,
      expert_weight_bits_ == 4 ? 2 : 1,
      activation_type_ == MoEActivationType::SwiGLU, block_size_));

  const auto& input_shape = hidden_state->Shape();

  // SwiGLU validation
  bool is_swiglu = (activation_type_ == MoEActivationType::SwiGLU);
  if (is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SwiGLU activation is not supported with fc3. Gate weights should be concatenated with FC1 weights.");
  }
  if (!is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 gating is not yet implemented for non-SwiGLU activations on CPU.");
  }

  const int max_tokens = 128; // TODO: 64 for testing, target = 256
  const uint32_t num_experts = static_cast<uint32_t>(moe_params.num_experts);
  const uint32_t hidden_size = static_cast<uint32_t>(moe_params.hidden_size);
  const int64_t fc1_output_size = is_swiglu && swiglu_fusion_ > 0 ? 2 * moe_params.inter_size : moe_params.inter_size;
  const bool is_fp16 = hidden_state->DataType() == DataTypeImpl::GetType<MLFloat16>();
  const auto dtype = is_fp16 ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>();
  const auto dtype_uint32 = DataTypeImpl::GetType<uint32_t>();

  const int64_t K_fc1 = moe_params.hidden_size; // left_shape[left_num_dims - 1]
  const int64_t N_fc1 = fc1_output_size;        // right_shape[right_num_dims - 1]
  const int64_t K_fc2 = moe_params.inter_size;  // left_shape[left_num_dims - 1]
  const int64_t N_fc2 = moe_params.inter_size;  // right_shape[right_num_dims - 1]
  const int64_t accuracy_level = 4;
  const int64_t block_size = (block_size_ != 0) ? block_size_ : fc1_experts_weights->Shape()[2];
  Status status;

  Tensor* output_tensor = context.Output(0, input_shape);
  const int total_output_size = static_cast<int>(input_shape.Size()) / 4;
  // we are accumulating expert results into output_tensor, need to initialize to zero
  ZeroTensorProgram zero;
  zero
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None, 4})
      .SetDispatchGroupSize(DISPATCH(total_output_size, WORKGROUP_SIZE))
      .AddUniformVariables({static_cast<uint32_t>(total_output_size)});
  ORT_RETURN_IF_ERROR(context.RunProgram(zero));

#ifdef GSDEBUG
    NpyTensor<MLFloat16>(hidden_state, "/tmp/hidden_state.npy", context);
    NpyTensor<MLFloat16>(router_logits, "/tmp/router_logits.npy", context);
#endif

  for (int token_offset = 0; token_offset < moe_params.num_rows; token_offset += max_tokens) {
    //
    // Step 1: run the gate to get router indices and values
    //
    int num_tokens = static_cast<int>(moe_params.num_rows) - token_offset;
    if (num_tokens > max_tokens) {
      num_tokens = max_tokens;
    }
    TensorShape gate_idx_shape({num_experts, 4});
    TensorShape gate_value_shape({num_tokens, num_experts});  // use max_tokens ?
    TensorShape gate_hidden_shape({num_experts, num_tokens}); // use max_tokens ?
    TensorShape gate_count_shape({num_experts});

    // router_values: [num_tokens, num_experts], per expert float we multiply final results with
    Tensor router_values = context.CreateGPUTensor(dtype, gate_value_shape);
    // gate_counts: [num_experts], number of tokens assigned to each expert
    Tensor gate_counts = context.CreateGPUTensor(dtype_uint32, gate_count_shape);
    // gate_hidden: [num_experts, num_tokens], token_idx assigned to each expert
    Tensor gate_hidden = context.CreateGPUTensor(dtype_uint32, gate_hidden_shape);

    GateProgram gate{k_, is_fp16};
    gate
        .AddInputs({{router_logits, ProgramTensorMetadataDependency::Type}})
        .AddOutput({&router_values, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_hidden, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_counts, ProgramTensorMetadataDependency::None, ProgramOutput::Atomic})
        .SetWorkgroupSize(num_experts)
        .SetDispatchGroupSize(static_cast<uint32_t>(num_tokens))
        .AddUniformVariables({static_cast<uint32_t>(num_tokens),
                              num_experts,
                              static_cast<uint32_t>(token_offset)
                            })
        .CacheHint(k_, is_fp16 ? "fp16" : "fp32");

    ORT_RETURN_IF_ERROR(context.RunProgram(gate));
#ifdef GSDEBUG
    DumpTensor<MLFloat16>(&router_values, "router_values", context);
    NpyTensor<MLFloat16>(&router_values, "/tmp/router_values.npy", context);
    NpyTensor<uint32_t>(&gate_counts, "/tmp/gate_counts.npy", context);
    NpyTensor<uint32_t>(&gate_hidden, "/tmp/gate_hidden.npy", context);
#endif

    Tensor gate_counts_cpu = context.CreateCPUTensor(dtype_uint32, gate_count_shape);
    Info().GetDataTransferManager().CopyTensor(gate_counts, gate_counts_cpu);

    for (uint32_t expert_idx=0; expert_idx < num_experts; expert_idx++) {

      uint32_t used_by = *(gate_counts_cpu.Data<uint32_t>() + expert_idx);
      if (used_by <= 0) {
        continue;
      }

      //
      // Step 2: for each expert, gather the hidden_state rows assigned to it
      // FIXME: use vec4
      //
      TensorShape expert_hidden_shape({used_by, moe_params.hidden_size});
      // expert_hidden: [used_by, hidden_size], hidden states assigned to this expert
      Tensor expert_hidden = context.CreateGPUTensor(dtype, expert_hidden_shape);
      TensorShape expert_tokens_shape({used_by});
      // expert_tokens: [used_by], token_idx that match expert_hidden rows
      Tensor expert_tokens = context.CreateGPUTensor(dtype_uint32, expert_tokens_shape);
      HiddenStateGatherProgram gather;
      gather
        .AddInputs({{&gate_hidden, ProgramTensorMetadataDependency::Type}})
        .AddInputs({{hidden_state, ProgramTensorMetadataDependency::Type, 1}})
        .AddOutput({&expert_hidden, ProgramTensorMetadataDependency::None, 1})
        .AddOutput({&expert_tokens, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize(used_by)
        .AddUniformVariables({expert_idx,
                              num_experts,
                              static_cast<uint32_t>(num_tokens),
                              hidden_size
                            });
      ORT_RETURN_IF_ERROR(context.RunProgram(gather));

#ifdef GSDEBUG
      if (expert_idx == 0) {
        DumpTensor<MLFloat16>(&expert_hidden, "expert_hidden", context);
        NpyTensor<MLFloat16>(&expert_hidden, "/tmp/expert_hidden.npy", context);
        NpyTensor<uint32_t>(&expert_tokens, "/tmp/expert_tokens.npy", context);
        DumpTensor<uint32_t>(&expert_tokens, "expert_tokens", context);
      }
#endif

      TensorShape fc1_output_shape({used_by, fc1_output_size});
      Tensor fc1_outputs = context.CreateGPUTensor(dtype, fc1_output_shape);
      TensorShape fc1_activated_shape({used_by, moe_params.inter_size});
      Tensor fc1_activated = context.CreateGPUTensor(dtype, fc1_activated_shape);
      TensorShape fc2_output_shape({used_by, moe_params.inter_size});
      Tensor fc2_outputs = context.CreateGPUTensor(dtype, fc2_output_shape);

      //
      // Step 3: matmul the hidden_state with fc1 (gate_up) of the selected experts
      //
      status = ApplyMatMulNBits(&expert_hidden, fc1_experts_weights, fc1_scales, nullptr, fc1_experts_bias_optional,
                                K_fc1, N_fc1, block_size, accuracy_level, expert_weight_bits_, context,
                                &fc1_outputs, expert_idx);
      ORT_RETURN_IF_ERROR(status);

  #ifdef GSDEBUG
      if (expert_idx == 0) {
        DumpTensor<MLFloat16>(&fc1_outputs, "fc1_outputs", context);
        NpyTensor<MLFloat16>(&fc1_outputs, "/tmp/fc1_outputs.npy", context);
      }
  #endif

      //
      // Step 4: apply swigly
      //
      if (is_swiglu) {
        SwigLuProgram swiglu;
        swiglu
            .AddInputs({{&fc1_outputs, ProgramTensorMetadataDependency::Type, 2}})
            .AddOutput({&fc1_activated, ProgramTensorMetadataDependency::None})
            .SetWorkgroupSize(128)
            .SetDispatchGroupSize(((used_by * moe_params.inter_size) + 127) / 128)
            .AddUniformVariables({
              static_cast<uint32_t>(used_by),
              static_cast<uint32_t>(moe_params.inter_size),
              activation_alpha_,
              activation_beta_,
              swiglu_limit_});
        ORT_RETURN_IF_ERROR(context.RunProgram(swiglu));
      } else {
        ORT_THROW("only swiglu is supported for now.");
      }
  #ifdef GSDEBUG
      if (expert_idx == 0) {
        DumpTensor<MLFloat16>(&fc1_activated, "fc1_activated", context);
        NpyTensor<MLFloat16>(&fc1_activated, "/tmp/fc1_activated.npy", context);
      }
  #endif

      //
      // Step 5: multiply fc1_activated with fc2 (gate_down) of the selected experts
      //
      status = ApplyMatMulNBits(&fc1_activated, fc2_experts_weights, fc2_scales, nullptr, fc2_experts_bias_optional,
                                K_fc2, N_fc2, block_size, accuracy_level, expert_weight_bits_, context,
                                &fc2_outputs, expert_idx);
      ORT_RETURN_IF_ERROR(status);
#ifdef GSDEBUG
      if (expert_idx == 0) {
        DumpTensor<MLFloat16>(&fc2_outputs, "fc2_outputs", context);
        NpyTensor<MLFloat16>(&fc2_outputs, "/tmp/fc2_outputs.npy", context);
      }
#endif

      //
      // Step 5: multiply fc2_outputs with router_values and accumulate
      //
      QMoEFinalMixProgram final_mix;
      final_mix
          .AddInputs({{&fc2_outputs, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&router_values, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&expert_tokens, ProgramTensorMetadataDependency::Type}})
          .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
          .SetDispatchGroupSize(used_by)
          .AddUniformVariables({used_by,
                                hidden_size,
                                num_experts,
                                expert_idx});

      ORT_RETURN_IF_ERROR(context.RunProgram(final_mix));

#ifdef GSDEBUG
    NpyTensor<MLFloat16>(output_tensor, "/tmp/output.npy", context);
    DumpTensor<MLFloat16>(output_tensor, "output_tensor", context);
#endif
    }
  }

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
