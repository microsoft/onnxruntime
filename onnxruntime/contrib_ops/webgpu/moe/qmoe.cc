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
#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"
#endif

#include <cstring>
#include <optional>
#include <sstream>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

namespace {

std::string BuildFp8E4M3DequantLutWgsl() {
  std::ostringstream oss;
  oss << "const kFp8DequantLutBits = array<u32, 256>(";
#if !defined(DISABLE_FLOAT8_TYPES)
  for (int i = 0; i < 256; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    const float value = Float8E4M3FN(static_cast<uint8_t>(i), Float8E4M3FN::FromBits()).ToFloat();
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    oss << bits << "u";
  }
#endif
  oss << ");\n";
  return oss.str();
}

class BlockFp8ExpertMatMulProgram final : public Program<BlockFp8ExpertMatMulProgram> {
 public:
  BlockFp8ExpertMatMulProgram(bool has_bias, bool has_indirect_experts, bool broadcast_input)
      : Program<BlockFp8ExpertMatMulProgram>{"QMoEBlockFp8ExpertMatMul"},
        has_bias_{has_bias},
        has_indirect_experts_{has_indirect_experts},
        broadcast_input_{broadcast_input} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    if (!has_indirect_experts_) {
      shader.AddInput("weights_backing", ShaderUsage::UseElementTypeAlias);
    }
    const auto& weights = shader.AddInput("weights", ShaderUsage::UseElementTypeAlias);
    if (!has_indirect_experts_) {
      shader.AddInput("scales_backing", ShaderUsage::UseElementTypeAlias);
    }
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* bias = &input;
    if (has_bias_) {
      if (!has_indirect_experts_) {
        shader.AddInput("bias_backing", ShaderUsage::UseElementTypeAlias);
      }
      bias = &shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    }
    const ShaderVariableHelper* indirect_experts = &weights;
    if (has_indirect_experts_) {
      indirect_experts = &shader.AddInput("indirect_experts", ShaderUsage::UseElementTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << BuildFp8E4M3DequantLutWgsl();
    return WGSL_TEMPLATE_APPLY(shader, "moe/block_fp8_expert_matmul.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(broadcast_input, broadcast_input_),
                               WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                               WGSL_TEMPLATE_PARAMETER(has_indirect_experts, has_indirect_experts_),
                               WGSL_TEMPLATE_VARIABLE(bias, *bias),
                               WGSL_TEMPLATE_VARIABLE(indirect_experts, *indirect_experts),
                               WGSL_TEMPLATE_VARIABLE(input, input),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(scales, scales),
                               WGSL_TEMPLATE_VARIABLE(weights, weights));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"inner", ProgramUniformVariableDataType::Uint32},
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"scale_n_blocks", ProgramUniformVariableDataType::Uint32},
      {"scale_k_blocks", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  bool has_indirect_experts_;
  bool broadcast_input_;
};

Status ApplyBlockFp8ExpertMatMul(ComputeContext& context,
                                 const Tensor* input,
                                 const Tensor* weights,
                                 const Tensor* scales,
                                 const Tensor* bias,
                                 Tensor* output,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t inner,
                                 uint32_t expert_idx,
                                 const Tensor* indirect_experts = nullptr,
                                 bool broadcast_input = false) {
  auto memory_info = OrtMemoryInfo{
      WEBGPU_BUFFER,
      OrtDeviceAllocator,
      OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0}};
  Tensor raw_weights(DataTypeImpl::GetType<uint8_t>(), weights->Shape(),
                     const_cast<void*>(weights->DataRaw()), memory_info);

  const uint32_t scale_n_blocks = (cols + 127) / 128;
  const uint32_t scale_k_blocks = (inner + 127) / 128;
  BlockFp8ExpertMatMulProgram program{bias != nullptr, indirect_experts != nullptr, broadcast_input};
  program.AddInputs({{input, ProgramTensorMetadataDependency::Type}});
  if (indirect_experts) {
    program.AddInputs({{&raw_weights, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4}})
        .AddInputs({{scales, ProgramTensorMetadataDependency::Type}});
    if (bias) {
      program.AddInputs({{bias, ProgramTensorMetadataDependency::Type}});
    }
    program.AddInputs({{indirect_experts, ProgramTensorMetadataDependency::Type}});
  } else {
    const uint32_t weight_elements = cols * inner;
    ORT_RETURN_IF_NOT(weight_elements % 4 == 0,
                      "Block-scaled FP8 expert weight slices must be divisible by four elements.");
    program.AddInputs({{&raw_weights, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4}})
        .AddInputs({ProgramInput::BufferView(&raw_weights,
                                             ProgramTensorMetadataDependency::Type,
                                             TensorShape({weight_elements / 4}),
                                             expert_idx * weight_elements / 4,
                                             4)})
        .AddInputs({{scales, ProgramTensorMetadataDependency::Type}})
        .AddInputs({ProgramInput::BufferView(scales,
                                             ProgramTensorMetadataDependency::Type,
                                             TensorShape({scale_n_blocks * scale_k_blocks}),
                                             expert_idx * scale_n_blocks * scale_k_blocks)});
    if (bias) {
      program.AddInputs({{bias, ProgramTensorMetadataDependency::Type}})
          .AddInputs({ProgramInput::BufferView(bias,
                                               ProgramTensorMetadataDependency::Type,
                                               TensorShape({cols}),
                                               expert_idx * cols)});
    }
  }
  constexpr uint32_t workgroup_size = 64;
  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(workgroup_size)
      .SetDispatchGroupSize((cols + workgroup_size - 1) / workgroup_size, rows)
      .AddUniformVariables({rows, cols, inner, indirect_experts ? expert_idx : 0,
                            scale_n_blocks, scale_k_blocks})
      .CacheHint(bias != nullptr, indirect_experts != nullptr, broadcast_input);
  return context.RunProgram(program);
}

Status ValidateBlockFp8Scales(const Tensor* scales, const char* name,
                              int64_t experts, int64_t output_size, int64_t input_size) {
  ORT_RETURN_IF_NOT(scales != nullptr, name, " is required for block-scaled FP8 QMoE.");
  ORT_RETURN_IF_NOT(scales->DataType() == DataTypeImpl::GetType<float>(),
                    name, " must have float32 elements for block-scaled FP8 QMoE.");
  const TensorShape expected({experts, (output_size + 127) / 128, (input_size + 127) / 128});
  ORT_RETURN_IF_NOT(scales->Shape() == expected, name, " must have shape ", expected,
                    " for block-scaled FP8 QMoE, got ", scales->Shape(), ".");
  return Status::OK();
}

}  // namespace

class GateProgram final : public Program<GateProgram> {
 public:
  GateProgram(int k, bool is_fp16, bool has_router_weights, bool normalize_routing_weights)
      : Program<GateProgram>{"QmoeGate"},
        k_{k},
        is_fp16_{is_fp16},
        has_router_weights_{has_router_weights},
        normalize_routing_weights_{normalize_routing_weights} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& router_logits = shader.AddInput("router_logits", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* router_weights = &router_logits;
    if (has_router_weights_) {
      router_weights = &shader.AddInput("router_weights", ShaderUsage::UseElementTypeAlias);
    }
    const auto& topk_values = shader.AddOutput("topk_values");
    const auto& hiddenstate_for_expert = shader.AddOutput("hiddenstate_for_expert");
    shader.AddOutput("tokencount_for_expert");

    return WGSL_TEMPLATE_APPLY(shader, "moe/gate.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_router_weights, has_router_weights_),
                               WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                               WGSL_TEMPLATE_PARAMETER(k, k_),
                               WGSL_TEMPLATE_PARAMETER(normalize_routing_weights, normalize_routing_weights_),
                               WGSL_TEMPLATE_VARIABLE(hiddenstate_for_expert, hiddenstate_for_expert),
                               WGSL_TEMPLATE_VARIABLE(router_logits, router_logits),
                               WGSL_TEMPLATE_VARIABLE(router_weights, *router_weights),
                               WGSL_TEMPLATE_VARIABLE(topk_values, topk_values));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"token_offset", ProgramUniformVariableDataType::Uint32});

 private:
  int k_;
  bool is_fp16_;
  bool has_router_weights_;
  bool normalize_routing_weights_;
};

class Gate1TokenProgram final : public Program<Gate1TokenProgram> {
 public:
  Gate1TokenProgram(int k, bool is_fp16, bool has_router_weights, bool normalize_routing_weights)
      : Program<Gate1TokenProgram>{"QmoeGate1Token"},
        k_{k},
        is_fp16_{is_fp16},
        has_router_weights_{has_router_weights},
        normalize_routing_weights_{normalize_routing_weights} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& router_logits = shader.AddInput("router_logits", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* router_weights = &router_logits;
    if (has_router_weights_) {
      router_weights = &shader.AddInput("router_weights", ShaderUsage::UseElementTypeAlias);
    }
    const auto& topk_values = shader.AddOutput("topk_values");
    const auto& indirect_experts = shader.AddOutput("indirect_experts");

    return WGSL_TEMPLATE_APPLY(shader, "moe/gate_1token.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_router_weights, has_router_weights_),
                               WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                               WGSL_TEMPLATE_PARAMETER(k, k_),
                               WGSL_TEMPLATE_PARAMETER(normalize_routing_weights, normalize_routing_weights_),
                               WGSL_TEMPLATE_VARIABLE(indirect_experts, indirect_experts),
                               WGSL_TEMPLATE_VARIABLE(router_logits, router_logits),
                               WGSL_TEMPLATE_VARIABLE(router_weights, *router_weights),
                               WGSL_TEMPLATE_VARIABLE(topk_values, topk_values));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32});

 private:
  int k_;
  bool is_fp16_;
  bool has_router_weights_;
  bool normalize_routing_weights_;
};

class HiddenStateGatherProgram final : public Program<HiddenStateGatherProgram> {
 public:
  HiddenStateGatherProgram() : Program<HiddenStateGatherProgram>{"QmoeHiddenStateGather"} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& hiddenstate_for_expert = shader.AddInput("hiddenstate_for_expert", ShaderUsage::UseElementTypeAlias);
    const auto& hidden_state = shader.AddInput("hidden_state", ShaderUsage::UseElementTypeAlias);
    const auto& new_hidden_state = shader.AddOutput("new_hidden_state");
    const auto& tokens = shader.AddOutput("tokens");

    return WGSL_TEMPLATE_APPLY(shader, "moe/hidden_state_gather.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(hidden_state, hidden_state),
                               WGSL_TEMPLATE_VARIABLE(hiddenstate_for_expert, hiddenstate_for_expert),
                               WGSL_TEMPLATE_VARIABLE(new_hidden_state, new_hidden_state),
                               WGSL_TEMPLATE_VARIABLE(tokens, tokens));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"num_tokens", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32});

 private:
};

class ZeroTensorProgram final : public Program<ZeroTensorProgram> {
 public:
  ZeroTensorProgram() : Program<ZeroTensorProgram>{"QmoeZeroTensor"} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& tensor = shader.AddOutput("tensor", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_tensor.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(tensor, tensor));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32});

 private:
};

class ZeroU32Program final : public Program<ZeroU32Program> {
 public:
  ZeroU32Program() : Program<ZeroU32Program>{"QmoeZeroU32"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& output = shader.AddOutput("output");
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_u32.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(output, output));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
};

class MoEActivationProgram final : public Program<MoEActivationProgram> {
 public:
  MoEActivationProgram(MoEActivationType activation_type, int swiglu_fusion, bool has_fc3)
      : Program<MoEActivationProgram>{"MoEActivation"},
        activation_type_{activation_type},
        swiglu_fusion_{swiglu_fusion},
        has_fc3_{has_fc3} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* fc3_input = &input;
    if (has_fc3_) {
      fc3_input = &shader.AddInput("fc3_input", ShaderUsage::UseElementTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

    return WGSL_TEMPLATE_APPLY(shader, "moe/activation.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(activation, static_cast<int>(activation_type_)),
                               WGSL_TEMPLATE_PARAMETER(has_fc3, has_fc3_),
                               WGSL_TEMPLATE_PARAMETER(swiglu_fusion, swiglu_fusion_),
                               WGSL_TEMPLATE_VARIABLE(fc3_input, *fc3_input),
                               WGSL_TEMPLATE_VARIABLE(input, input),
                               WGSL_TEMPLATE_VARIABLE(output, output));
  };

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

class FusedFinalMix1TokenProgram final : public Program<FusedFinalMix1TokenProgram> {
 public:
  FusedFinalMix1TokenProgram() : Program<FusedFinalMix1TokenProgram>{"QmoeFusedFinalMix1Token"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& fc2_outputs = shader.AddInput("fc2_outputs", ShaderUsage::UseElementTypeAlias);
    const auto& router_values = shader.AddInput("router_values", ShaderUsage::UseElementTypeAlias);
    const auto& indirect_experts = shader.AddInput("indirect_experts", ShaderUsage::UseElementTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/fused_final_mix_1token.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(fc2_outputs, fc2_outputs),
                               WGSL_TEMPLATE_VARIABLE(indirect_experts, indirect_experts),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(router_values, router_values));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"k", ProgramUniformVariableDataType::Uint32});
};

class QMoEFinalMixProgram final : public Program<QMoEFinalMixProgram> {
 public:
  QMoEFinalMixProgram() : Program<QMoEFinalMixProgram>{"QMoEFinalMix"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
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
  const Tensor* fc1_zero_points = context.Input<Tensor>(11);
  const Tensor* fc2_zero_points = context.Input<Tensor>(12);
  const Tensor* fc3_zero_points = context.Input<Tensor>(13);
  const Tensor* router_weights = context.Input<Tensor>(14);
  const Tensor* fc1_global_scale = context.Input<Tensor>(15);
  const Tensor* fc2_global_scale = context.Input<Tensor>(16);

  MoEParameters moe_params;

  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == MoEActivationType::SwiGLU && swiglu_fusion == 0 &&
      fc3_experts_weights_optional == nullptr) {
    swiglu_fusion = 1;
  }
  const bool is_swiglu = activation_type_ == MoEActivationType::SwiGLU;
  const bool is_fused_swiglu = is_swiglu && swiglu_fusion != 0 && fc3_experts_weights_optional == nullptr;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, hidden_state, router_logits,
      fc1_experts_weights, fc1_experts_bias_optional, is_block_fp8_ ? nullptr : fc1_scales, fc1_zero_points,
      fc2_experts_weights, fc2_experts_bias_optional, is_block_fp8_ ? nullptr : fc2_scales, fc2_zero_points,
      fc3_experts_weights_optional, fc3_experts_bias_optional, is_block_fp8_ ? nullptr : fc3_scales_optional, fc3_zero_points,
      moe_helper::MoEWeightBits{fc1_expert_weight_bits_,
                                fc2_expert_weight_bits_,
                                fc3_expert_weight_bits_},
      is_fused_swiglu, block_size_));
  ORT_RETURN_IF(router_weights && router_weights->Shape() != router_logits->Shape(),
                "router_weights must have the same shape as router_probs; got ",
                router_weights->Shape(), " and ", router_logits->Shape());
  ORT_RETURN_IF_NOT(k_ > 0 && k_ <= moe_params.num_experts,
                    "QMoE requires 0 < k <= num_experts, got k=", k_,
                    " and num_experts=", moe_params.num_experts);
  ORT_RETURN_IF_NOT(block_size_ > 0 ||
                        (fc1_zero_points == nullptr && fc2_zero_points == nullptr && fc3_zero_points == nullptr),
                    "WebGPU QMoE row-wise quantization does not support explicit zero points. "
                    "Use block-wise quantization or omit the zero-point inputs.");
  ORT_RETURN_IF_NOT(moe_params.num_rows != 1 ||
                        (fc1_zero_points == nullptr && fc2_zero_points == nullptr && fc3_zero_points == nullptr),
                    "WebGPU QMoE does not support explicit zero points on the optimized single-token path.");

  if (is_block_fp8_) {
    ORT_RETURN_IF_NOT(fc1_zero_points == nullptr && fc2_zero_points == nullptr && fc3_zero_points == nullptr,
                      "Block-scaled FP8 QMoE does not support zero points.");
    ORT_RETURN_IF_NOT(fc1_global_scale == nullptr && fc2_global_scale == nullptr,
                      "Block-scaled FP8 QMoE uses fc1_scales/fc2_scales directly; global scales must be omitted.");
    ORT_RETURN_IF_NOT(fc1_experts_weights->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
                          fc2_experts_weights->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
                          (fc3_experts_weights_optional == nullptr ||
                           fc3_experts_weights_optional->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN),
                      "Block-scaled FP8 QMoE weights must use float8e4m3fn elements.");
    const int64_t fc1_size = is_fused_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
    ORT_RETURN_IF_ERROR(ValidateBlockFp8Scales(fc1_scales, "fc1_scales", moe_params.num_experts,
                                               fc1_size, moe_params.hidden_size));
    ORT_RETURN_IF_ERROR(ValidateBlockFp8Scales(fc2_scales, "fc2_scales", moe_params.num_experts,
                                               moe_params.hidden_size, moe_params.inter_size));
    if (fc3_experts_weights_optional) {
      ORT_RETURN_IF_ERROR(ValidateBlockFp8Scales(fc3_scales_optional, "fc3_scales", moe_params.num_experts,
                                                 moe_params.inter_size, moe_params.hidden_size));
    }
  }

  if (fc1_expert_weight_bits_ != expert_weight_bits_ ||
      fc2_expert_weight_bits_ != expert_weight_bits_ ||
      fc3_expert_weight_bits_ != expert_weight_bits_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Mixed-width QMoE execution is not yet implemented on WebGPU.");
  }

  const auto& input_shape = hidden_state->Shape();

  // process tokens in chunks of max_tokens to put some cap on memory usage
  const int max_tokens = 2 * 1024;

  const uint32_t num_experts = static_cast<uint32_t>(moe_params.num_experts);
  Tensor* output_tensor = context.Output(0, input_shape);
  if (moe_params.num_rows == 0) {
    return Status::OK();
  }
  const auto& device_limits = context.DeviceLimits();
  ORT_RETURN_IF_NOT(num_experts <= device_limits.maxComputeWorkgroupSizeX &&
                        num_experts <= device_limits.maxComputeInvocationsPerWorkgroup,
                    "WebGPU QMoE requires num_experts to fit in one workgroup; got ", num_experts,
                    ", maxComputeWorkgroupSizeX=", device_limits.maxComputeWorkgroupSizeX,
                    ", maxComputeInvocationsPerWorkgroup=", device_limits.maxComputeInvocationsPerWorkgroup, ".");
  const uint32_t hidden_size = static_cast<uint32_t>(moe_params.hidden_size);
  const int64_t fc1_output_size = is_fused_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const bool is_fp16 = hidden_state->DataType() == DataTypeImpl::GetType<MLFloat16>();
  const auto dtype = is_fp16 ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>();
  const auto dtype_uint32 = DataTypeImpl::GetType<uint32_t>();

  const int64_t K_fc1 = moe_params.hidden_size;
  const int64_t N_fc1 = fc1_output_size;
  const int64_t K_fc2 = moe_params.inter_size;
  const int64_t N_fc2 = moe_params.hidden_size;
  const int64_t accuracy_level = 4;
  const int64_t block_size_fc1 = (block_size_ != 0) ? block_size_ : K_fc1;
  const int64_t block_size_fc2 = (block_size_ != 0) ? block_size_ : K_fc2;
  const int64_t block_size_fc3 = block_size_fc1;
  Status status;

  if (moe_params.num_rows == 1) {
    // Fused MoE path for 1 token: instead of looping k times with separate dispatches,
    // run a single batched MatMulNBits with M=k where each row uses a different expert's
    // weights via weight_index_indirect. A's single row is broadcast to all k rows.
    // This reduces dispatches from 1 + k*4 = 17 to 5 (gate + fc1 + swiglu + fc2 + mix).

    const uint32_t k = static_cast<uint32_t>(k_);
    const uint32_t num_tokens = 1;
    TensorShape gate_value_shape({num_tokens, num_experts});
    TensorShape indirect_experts_shape({k});

    Tensor router_values = context.CreateGPUTensor(dtype, gate_value_shape);
    Tensor indirect_experts = context.CreateGPUTensor(dtype_uint32, indirect_experts_shape);

    // Step 1: Gate — select top-k experts
    Gate1TokenProgram gate{k_, is_fp16, router_weights != nullptr, normalize_routing_weights_};
    gate
        .AddInputs({{router_logits, ProgramTensorMetadataDependency::Type}});
    if (router_weights) {
      gate.AddInputs({{router_weights, ProgramTensorMetadataDependency::Type}});
    }
    gate.AddOutput({&router_values, ProgramTensorMetadataDependency::None})
        .AddOutput({&indirect_experts, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(num_experts)
        .SetDispatchGroupSize(num_tokens)
        .AddUniformVariables({num_tokens, num_experts})
        .CacheHint(k_, is_fp16 ? "fp16" : "fp32", router_weights != nullptr, normalize_routing_weights_);
    ORT_RETURN_IF_ERROR(context.RunProgram(gate));

    // Step 2: Batched fc1 MatMulNBits with M=k, per-row expert selection.
    // A is (1, hidden_size) but dispatched with override_M=k; shader broadcasts A row 0.
    TensorShape fc1_output_shape({static_cast<int64_t>(k), fc1_output_size});
    Tensor fc1_outputs = context.CreateGPUTensor(dtype, fc1_output_shape);
    status = is_block_fp8_
                 ? ApplyBlockFp8ExpertMatMul(context, hidden_state, fc1_experts_weights, fc1_scales,
                                             fc1_experts_bias_optional, &fc1_outputs, k,
                                             static_cast<uint32_t>(N_fc1), static_cast<uint32_t>(K_fc1), 0,
                                             &indirect_experts, true)
                 : ApplyMatMulNBits(hidden_state, fc1_experts_weights, fc1_scales, fc1_zero_points,
                                    fc1_experts_bias_optional, K_fc1, N_fc1, block_size_fc1, accuracy_level,
                                    expert_weight_bits_, context, &fc1_outputs, 0, &indirect_experts,
                                    /*override_M=*/k);
    ORT_RETURN_IF_ERROR(status);

    // Step 3: Apply the activation, optionally combining FC1 with a separate FC3 projection.
    TensorShape fc1_activated_shape({static_cast<int64_t>(k), moe_params.inter_size});
    Tensor fc1_activated = context.CreateGPUTensor(dtype, fc1_activated_shape);
    std::optional<Tensor> fc3_outputs;
    if (fc3_experts_weights_optional) {
      fc3_outputs.emplace(context.CreateGPUTensor(dtype, fc1_activated_shape));
      ORT_RETURN_IF_ERROR(is_block_fp8_
                              ? ApplyBlockFp8ExpertMatMul(context, hidden_state, fc3_experts_weights_optional,
                                                          fc3_scales_optional, fc3_experts_bias_optional,
                                                          &*fc3_outputs, k, static_cast<uint32_t>(moe_params.inter_size),
                                                          static_cast<uint32_t>(K_fc1), 0, &indirect_experts, true)
                              : ApplyMatMulNBits(hidden_state, fc3_experts_weights_optional, fc3_scales_optional,
                                                 fc3_zero_points, fc3_experts_bias_optional,
                                                 K_fc1, moe_params.inter_size, block_size_fc3, accuracy_level,
                                                 expert_weight_bits_, context, &*fc3_outputs, 0, &indirect_experts,
                                                 /*override_M=*/k));
    }
    MoEActivationProgram activation{activation_type_, swiglu_fusion, fc3_outputs.has_value()};
    activation.AddInputs({{&fc1_outputs, ProgramTensorMetadataDependency::Type}});
    if (fc3_outputs) {
      activation.AddInputs({{&*fc3_outputs, ProgramTensorMetadataDependency::Type}});
    }
    activation
        .AddOutput({&fc1_activated, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(128)
        .SetDispatchGroupSize(((k * static_cast<uint32_t>(moe_params.inter_size)) + 127) / 128)
        .AddUniformVariables({k, static_cast<uint32_t>(moe_params.inter_size), activation_alpha_,
                              activation_beta_, swiglu_limit_})
        .CacheHint(static_cast<int>(activation_type_), swiglu_fusion, fc3_outputs.has_value());
    ORT_RETURN_IF_ERROR(context.RunProgram(activation));

    // Step 4: Batched fc2 MatMulNBits with M=k, per-row expert selection
    // fc1_activated already has k rows (one per expert), no override_M needed.
    TensorShape fc2_output_shape({static_cast<int64_t>(k), N_fc2});
    Tensor fc2_outputs = context.CreateGPUTensor(dtype, fc2_output_shape);
    status = is_block_fp8_
                 ? ApplyBlockFp8ExpertMatMul(context, &fc1_activated, fc2_experts_weights, fc2_scales,
                                             fc2_experts_bias_optional, &fc2_outputs, k,
                                             static_cast<uint32_t>(N_fc2), static_cast<uint32_t>(K_fc2), 0,
                                             &indirect_experts)
                 : ApplyMatMulNBits(&fc1_activated, fc2_experts_weights, fc2_scales, fc2_zero_points,
                                    fc2_experts_bias_optional, K_fc2, N_fc2, block_size_fc2, accuracy_level,
                                    expert_weight_bits_, context, &fc2_outputs, 0, &indirect_experts,
                                    /*override_M=*/0);
    ORT_RETURN_IF_ERROR(status);

    // Step 5: Fused FinalMix — accumulate all k expert results weighted by router_values
    // Dispatch across hidden_size (not k) to avoid race: each thread accumulates all k experts.
    const uint32_t mix_wg_size = 256;
    FusedFinalMix1TokenProgram final_mix;
    final_mix
        .AddInputs({{&fc2_outputs, ProgramTensorMetadataDependency::Type}})
        .AddInputs({{&router_values, ProgramTensorMetadataDependency::Type}})
        .AddInputs({{&indirect_experts, ProgramTensorMetadataDependency::Type}})
        .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(mix_wg_size)
        .SetDispatchGroupSize((hidden_size + mix_wg_size - 1) / mix_wg_size)
        .AddUniformVariables({hidden_size, k});
    ORT_RETURN_IF_ERROR(context.RunProgram(final_mix));

    return Status::OK();
  }

  // Multi-token path: accumulates into output_tensor, need to initialize to zero.
  const int total_output_size = (static_cast<int>(input_shape.Size()) + 3) / 4;
  ZeroTensorProgram zero;
  zero
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten, 4})
      .SetDispatchGroupSize((total_output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({static_cast<uint32_t>(total_output_size)});
  ORT_RETURN_IF_ERROR(context.RunProgram(zero));

  // path for num_tokens > 1
  // process tokens in chunks of max_tokens to put some cap on memory usage
  for (int token_offset = 0; token_offset < moe_params.num_rows; token_offset += max_tokens) {
    //
    // Step 1: run the gate to get router indices and values
    //
    int num_tokens = static_cast<int>(moe_params.num_rows) - token_offset;
    if (num_tokens > max_tokens) {
      num_tokens = max_tokens;
    }
    TensorShape gate_value_shape({num_tokens, num_experts});   // use max_tokens ?
    TensorShape gate_hidden_shape({num_experts, num_tokens});  // use max_tokens ?
    TensorShape gate_count_shape({num_experts});

    // router_values: per expert float we multiply final results with
    Tensor router_values = context.CreateGPUTensor(dtype, gate_value_shape);
    // gate_counts: number of tokens assigned to each expert
    Tensor gate_counts = context.CreateGPUTensor(dtype_uint32, gate_count_shape);
    // gate_hidden: token_idx assigned to each expert
    //  token_idx is the global index into hidden_state
    Tensor gate_hidden = context.CreateGPUTensor(dtype_uint32, gate_hidden_shape);

    ZeroU32Program zero_counts;
    zero_counts.AddOutput({&gate_counts, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((num_experts + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({num_experts});
    ORT_RETURN_IF_ERROR(context.RunProgram(zero_counts));

    GateProgram gate{k_, is_fp16, router_weights != nullptr, normalize_routing_weights_};
    gate
        .AddInputs({{router_logits, ProgramTensorMetadataDependency::Type}});
    if (router_weights) {
      gate.AddInputs({{router_weights, ProgramTensorMetadataDependency::Type}});
    }
    gate.AddOutput({&router_values, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_hidden, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_counts, ProgramTensorMetadataDependency::None, ProgramOutput::Atomic})
        .SetWorkgroupSize(num_experts)
        .SetDispatchGroupSize(static_cast<uint32_t>(num_tokens))
        .AddUniformVariables({static_cast<uint32_t>(num_tokens), num_experts, static_cast<uint32_t>(token_offset)})
        .CacheHint(k_, is_fp16 ? "fp16" : "fp32", router_weights != nullptr, normalize_routing_weights_);

    ORT_RETURN_IF_ERROR(context.RunProgram(gate));

    Tensor gate_counts_cpu = context.CreateCPUTensor(dtype_uint32, gate_count_shape);
    ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(gate_counts, gate_counts_cpu));

    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      uint32_t used_by = *(gate_counts_cpu.Data<uint32_t>() + expert_idx);
      if (used_by <= 0) {
        continue;
      }

      //
      // Step 2: for each expert, gather the hidden_state rows assigned to it
      // FIXME: use vec4
      //
      TensorShape expert_hidden_shape({used_by, moe_params.hidden_size});
      // expert_hidden: hidden states assigned to this expert
      Tensor expert_hidden = context.CreateGPUTensor(dtype, expert_hidden_shape);
      TensorShape expert_tokens_shape({used_by});
      // expert_tokens: token_idx that match expert_hidden rows
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
                                hidden_size});
      ORT_RETURN_IF_ERROR(context.RunProgram(gather));

      TensorShape fc1_output_shape({used_by, fc1_output_size});
      Tensor fc1_outputs = context.CreateGPUTensor(dtype, fc1_output_shape);
      TensorShape fc1_activated_shape({used_by, moe_params.inter_size});
      Tensor fc1_activated = context.CreateGPUTensor(dtype, fc1_activated_shape);
      TensorShape fc2_output_shape({used_by, N_fc2});
      Tensor fc2_outputs = context.CreateGPUTensor(dtype, fc2_output_shape);

      //
      // Step 3: matmul the hidden_state with fc1 (gate_up) of the selected experts
      //
      status = is_block_fp8_
                   ? ApplyBlockFp8ExpertMatMul(context, &expert_hidden, fc1_experts_weights, fc1_scales,
                                               fc1_experts_bias_optional, &fc1_outputs, used_by,
                                               static_cast<uint32_t>(N_fc1), static_cast<uint32_t>(K_fc1), expert_idx)
                   : ApplyMatMulNBits(&expert_hidden, fc1_experts_weights, fc1_scales, fc1_zero_points,
                                      fc1_experts_bias_optional, K_fc1, N_fc1, block_size_fc1, accuracy_level,
                                      expert_weight_bits_, context, &fc1_outputs, expert_idx);
      ORT_RETURN_IF_ERROR(status);

      //
      // Step 4: apply the activation and optional FC3 gate.
      //
      std::optional<Tensor> fc3_outputs;
      if (fc3_experts_weights_optional) {
        fc3_outputs.emplace(context.CreateGPUTensor(dtype, fc1_activated_shape));
        ORT_RETURN_IF_ERROR(is_block_fp8_
                                ? ApplyBlockFp8ExpertMatMul(context, &expert_hidden, fc3_experts_weights_optional,
                                                            fc3_scales_optional, fc3_experts_bias_optional,
                                                            &*fc3_outputs, used_by,
                                                            static_cast<uint32_t>(moe_params.inter_size),
                                                            static_cast<uint32_t>(K_fc1), expert_idx)
                                : ApplyMatMulNBits(&expert_hidden, fc3_experts_weights_optional, fc3_scales_optional,
                                                   fc3_zero_points, fc3_experts_bias_optional,
                                                   K_fc1, moe_params.inter_size, block_size_fc3, accuracy_level,
                                                   expert_weight_bits_, context, &*fc3_outputs, expert_idx));
      }
      MoEActivationProgram activation{activation_type_, swiglu_fusion, fc3_outputs.has_value()};
      activation.AddInputs({{&fc1_outputs, ProgramTensorMetadataDependency::Type}});
      if (fc3_outputs) {
        activation.AddInputs({{&*fc3_outputs, ProgramTensorMetadataDependency::Type}});
      }
      activation
          .AddOutput({&fc1_activated, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(128)
          .SetDispatchGroupSize((static_cast<uint32_t>(moe_params.inter_size) + 127) / 128, used_by)
          .AddUniformVariables({used_by, static_cast<uint32_t>(moe_params.inter_size), activation_alpha_,
                                activation_beta_, swiglu_limit_})
          .CacheHint(static_cast<int>(activation_type_), swiglu_fusion, fc3_outputs.has_value());
      ORT_RETURN_IF_ERROR(context.RunProgram(activation));

      //
      // Step 5: multiply fc1_activated with fc2 (gate_down) of the selected experts
      //
      status = is_block_fp8_
                   ? ApplyBlockFp8ExpertMatMul(context, &fc1_activated, fc2_experts_weights, fc2_scales,
                                               fc2_experts_bias_optional, &fc2_outputs, used_by,
                                               static_cast<uint32_t>(N_fc2), static_cast<uint32_t>(K_fc2), expert_idx)
                   : ApplyMatMulNBits(&fc1_activated, fc2_experts_weights, fc2_scales, fc2_zero_points,
                                      fc2_experts_bias_optional, K_fc2, N_fc2, block_size_fc2, accuracy_level,
                                      expert_weight_bits_, context, &fc2_outputs, expert_idx);
      ORT_RETURN_IF_ERROR(status);

      //
      // Step 6: multiply fc2_outputs with router_values and accumulate
      //
      QMoEFinalMixProgram final_mix;
      final_mix
          .AddInputs({{&fc2_outputs, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&router_values, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&expert_tokens, ProgramTensorMetadataDependency::Type}})
          .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
          .SetDispatchGroupSize(used_by)
          .AddUniformVariables({hidden_size,
                                num_experts,
                                expert_idx,
                                static_cast<uint32_t>(token_offset)})
          .CacheHint(router_weights != nullptr, normalize_routing_weights_);

      ORT_RETURN_IF_ERROR(context.RunProgram(final_mix));
    }
  }

  return Status::OK();
}

namespace {
const std::vector<MLDataType>& QMoET1Constraint() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<uint8_t>(),
#if !defined(DISABLE_FLOAT8_TYPES)
      DataTypeImpl::GetTensorType<Float8E4M3FN>(),
#endif
  };
  return types;
}
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T1", QMoET1Constraint())
        .TypeConstraint("T2", WebGpuSupportedFloatTypes()),
    QMoE);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
