// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/gated_delta_net.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime::contrib::webgpu {

namespace {

GatedDeltaNetUpdateRule ParseUpdateRule(const std::string& rule) {
  if (rule == "linear") return GatedDeltaNetUpdateRule::Linear;
  if (rule == "gated") return GatedDeltaNetUpdateRule::Gated;
  if (rule == "delta") return GatedDeltaNetUpdateRule::Delta;
  if (rule == "gated_delta") return GatedDeltaNetUpdateRule::GatedDelta;
  return GatedDeltaNetUpdateRule::Invalid;
}

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    GatedDeltaNet,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("TS", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>())
        .MayInplace(6, 1),
    GatedDeltaNet);

GatedDeltaNet::GatedDeltaNet(const OpKernelInfo& info) : WebGpuKernel(info) {
  update_rule_ = ParseUpdateRule(info.GetAttrOrDefault<std::string>("update_rule", "gated_delta"));
  ORT_ENFORCE(update_rule_ != GatedDeltaNetUpdateRule::Invalid,
              "update_rule must be one of: linear, gated, delta, gated_delta");
  const auto gate_activation = info.GetAttrOrDefault<std::string>("gate_activation", "none");
  ORT_ENFORCE(gate_activation == "none" || gate_activation == "qwen",
              "gate_activation must be one of: none, qwen");
  qwen_gate_ = gate_activation == "qwen";
  const auto beta_activation = info.GetAttrOrDefault<std::string>("beta_activation", "none");
  ORT_ENFORCE(beta_activation == "none" || beta_activation == "sigmoid",
              "beta_activation must be one of: none, sigmoid");
  sigmoid_beta_ = beta_activation == "sigmoid";
  qk_l2_norm_ = info.GetAttrOrDefault<int64_t>("qk_l2_norm", 0) != 0;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  const auto state_update_capacity = info.GetAttrOrDefault<int64_t>("state_update_capacity", 0);
  ORT_ENFORCE(state_update_capacity == 0,
              "WebGPU GatedDeltaNet does not support state_update_capacity > 0");
}

Status GatedDeltaNetProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("query", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("key", ShaderUsage::UseElementTypeAlias);
  shader.AddInput("value", ShaderUsage::UseElementTypeAlias);
  if (has_cu_seqlens_) shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  if (update_rule_ == GatedDeltaNetUpdateRule::Gated || update_rule_ == GatedDeltaNetUpdateRule::GatedDelta) {
    shader.AddInput("decay", ShaderUsage::UseUniform);
  }
  if (update_rule_ == GatedDeltaNetUpdateRule::Delta || update_rule_ == GatedDeltaNetUpdateRule::GatedDelta) {
    shader.AddInput("beta", ShaderUsage::UseUniform);
  }
  if (has_initial_state_ && !initial_state_in_final_state_) {
    shader.AddInput("initial_state", ShaderUsage::UseUniform);
  }
  if (qwen_gate_) {
    shader.AddInput("a_log", ShaderUsage::UseUniform);
    shader.AddInput("dt_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("final_state", ShaderUsage::UseUniform);

  int update_rule = 0;
  if (update_rule_ == GatedDeltaNetUpdateRule::Gated) update_rule = 1;
  if (update_rule_ == GatedDeltaNetUpdateRule::Delta) update_rule = 2;
  if (update_rule_ == GatedDeltaNetUpdateRule::GatedDelta) update_rule = 3;
  return WGSL_TEMPLATE_APPLY(shader, "bert/gated_delta_net.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(update_rule, update_rule),
                             WGSL_TEMPLATE_PARAMETER(has_cu_seqlens, has_cu_seqlens_),
                             WGSL_TEMPLATE_PARAMETER(has_initial_state, has_initial_state_),
                             WGSL_TEMPLATE_PARAMETER(initial_state_in_final_state, initial_state_in_final_state_),
                             WGSL_TEMPLATE_PARAMETER(qwen_gate, qwen_gate_),
                             WGSL_TEMPLATE_PARAMETER(sigmoid_beta, sigmoid_beta_),
                             WGSL_TEMPLATE_PARAMETER(qk_l2_norm, qk_l2_norm_));
}

Status GatedDeltaNet::ComputeInternal(ComputeContext& context) const {
  const auto* query = context.Input(0);
  const auto* key = context.Input(1);
  const auto* value = context.Input(2);
  const auto* cu_seqlens = context.Input(3);
  const auto* decay = context.Input(4);
  const auto* beta = context.Input(5);
  const auto* initial_state = context.Input(6);
  const auto* a_log = context.Input(7);
  const auto* dt_bias = context.Input(8);

  const bool needs_decay = update_rule_ == GatedDeltaNetUpdateRule::Gated ||
                           update_rule_ == GatedDeltaNetUpdateRule::GatedDelta;
  const bool needs_beta = update_rule_ == GatedDeltaNetUpdateRule::Delta ||
                          update_rule_ == GatedDeltaNetUpdateRule::GatedDelta;
  ORT_RETURN_IF_NOT(query && key && value, "query, key and value are required");
  ORT_RETURN_IF_NOT(needs_decay == (decay != nullptr), "decay input presence must match update_rule");
  ORT_RETURN_IF_NOT(needs_beta == (beta != nullptr), "beta input presence must match update_rule");

  const auto& q_shape = query->Shape();
  const auto& k_shape = key->Shape();
  const auto& v_shape = value->Shape();
  const size_t rank = q_shape.NumDimensions();
  ORT_RETURN_IF_NOT(rank == 3 || rank == 4, "query, key and value must be rank 3 or 4");
  ORT_RETURN_IF_NOT(k_shape.NumDimensions() == rank && v_shape.NumDimensions() == rank,
                    "query, key and value must have the same rank");
  const size_t token_dims = rank - 2;
  const int64_t total_tokens = q_shape.SizeToDimension(token_dims);
  ORT_RETURN_IF_NOT(total_tokens > 0 && k_shape.SizeToDimension(token_dims) == total_tokens &&
                        v_shape.SizeToDimension(token_dims) == total_tokens,
                    "query, key and value must agree on a positive total_tokens");
  const int64_t hq = q_shape[token_dims], hk = k_shape[token_dims], hv = v_shape[token_dims];
  const int64_t dk = q_shape[token_dims + 1], dv = v_shape[token_dims + 1];
  constexpr int64_t kMaxInt32 = std::numeric_limits<int32_t>::max();
  constexpr int64_t kMaxUint32 = std::numeric_limits<uint32_t>::max();
  ORT_RETURN_IF_NOT(hq > 0 && hq == hk && hv > 0 && hv % hq == 0 && dk > 0 && dv > 0 &&
                        k_shape[token_dims + 1] == dk,
                    "query/key heads must match and value heads must be a positive multiple of query heads");
  ORT_RETURN_IF_NOT(total_tokens <= kMaxInt32 && hq <= kMaxInt32 && hv <= kMaxInt32 &&
                        dk <= kMaxInt32 && dv <= kMaxInt32,
                    "GatedDeltaNet dimensions must fit in int32");
  ORT_RETURN_IF_NOT(total_tokens <= kMaxUint32 / hq / dk &&
                        total_tokens <= kMaxUint32 / hv / dv,
                    "GatedDeltaNet input sizes must fit in uint32");
  ORT_RETURN_IF_NOT(dk <= 256, "WebGPU GatedDeltaNet requires head_size_qk <= 256, got ", dk);

  int64_t batch = 1;
  if (cu_seqlens != nullptr) {
    ORT_RETURN_IF_NOT(rank == 3 && cu_seqlens->Shape().NumDimensions() == 1 &&
                          cu_seqlens->Shape()[0] >= 2,
                      "cu_seqlens requires rank-3 inputs and must have at least two elements");
    batch = cu_seqlens->Shape()[0] - 1;
  } else if (rank == 4) {
    batch = q_shape[0];
  } else {
    ORT_RETURN_IF_NOT(initial_state != nullptr,
                      "rank-3 uniform packing requires initial_state to determine batch size");
    batch = initial_state->Shape()[0];
    ORT_RETURN_IF_NOT(total_tokens % batch == 0, "total_tokens must be divisible by batch");
  }
  ORT_RETURN_IF_NOT(batch > 0, "batch size must be positive");
  ORT_RETURN_IF_NOT(batch <= kMaxInt32 && batch <= kMaxUint32 / hv / dv &&
                        batch <= kMaxUint32 / hv / dv / dk,
                    "GatedDeltaNet dispatch size must fit in uint32");
  const TensorShape state_shape{batch, hv, dv, dk};
  if (initial_state != nullptr) {
    ORT_RETURN_IF_NOT(initial_state->Shape() == state_shape,
                      "initial_state must be [batch, num_heads_v, head_size_v, head_size_qk] (V-major)");
  }
  if (decay != nullptr) {
    ORT_RETURN_IF_NOT(decay->Shape().NumDimensions() == token_dims + 1 &&
                          decay->Shape().SizeToDimension(token_dims) == total_tokens &&
                          decay->Shape()[token_dims] == hv,
                      "WebGPU GatedDeltaNet supports scalar decay with shape [...tokens, num_heads_v]");
  }
  if (beta != nullptr) {
    ORT_RETURN_IF_NOT(beta->Shape().NumDimensions() == token_dims + 1 &&
                          beta->Shape().SizeToDimension(token_dims) == total_tokens &&
                          beta->Shape()[token_dims] == hv,
                      "beta must have shape [...tokens, num_heads_v]");
  }
  if (qwen_gate_) {
    ORT_RETURN_IF_NOT(a_log != nullptr && dt_bias != nullptr && a_log->Shape().Size() == hv &&
                          dt_bias->Shape().Size() == hv,
                      "gate_activation=qwen requires a_log and dt_bias with num_heads_v elements");
  } else {
    ORT_RETURN_IF_NOT(a_log == nullptr && dt_bias == nullptr, "a_log and dt_bias require gate_activation=qwen");
  }

  TensorShapeVector output_dims(q_shape.GetDims().begin(), q_shape.GetDims().begin() + token_dims);
  output_dims.push_back(std::max(hq, hv));
  output_dims.push_back(dv);
  auto* output = context.Output(0, TensorShape(output_dims));
  auto* final_state = context.Output(1, state_shape);
  context.Output(2, TensorShape{batch, 0});
  ORT_RETURN_IF_NOT(output != nullptr && final_state != nullptr, "output and final_state are required");

  const bool state_alias = initial_state != nullptr && initial_state->DataRaw() == final_state->DataRaw();
  GatedDeltaNetProgram program{update_rule_, cu_seqlens != nullptr, initial_state != nullptr, state_alias,
                               qwen_gate_, sigmoid_beta_, qk_l2_norm_};
  program.AddInputs({{query, ProgramTensorMetadataDependency::Type},
                     {key, ProgramTensorMetadataDependency::Type},
                     {value, ProgramTensorMetadataDependency::Type}});
  if (cu_seqlens != nullptr) program.AddInput({cu_seqlens, ProgramTensorMetadataDependency::None});
  if (decay != nullptr) program.AddInput({decay, ProgramTensorMetadataDependency::None});
  if (beta != nullptr) program.AddInput({beta, ProgramTensorMetadataDependency::None});
  if (initial_state != nullptr && !state_alias) program.AddInput({initial_state, ProgramTensorMetadataDependency::None});
  if (qwen_gate_) program.AddInputs({{a_log, ProgramTensorMetadataDependency::None}, {dt_bias, ProgramTensorMetadataDependency::None}});
  program.AddOutputs({{output, ProgramTensorMetadataDependency::Type}, {final_state, ProgramTensorMetadataDependency::None}})
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(batch * hv * dv))
      .SetWorkgroupSize(256)
      .CacheHint(static_cast<int>(update_rule_), cu_seqlens != nullptr, initial_state != nullptr, state_alias,
                 qwen_gate_, sigmoid_beta_, qk_l2_norm_)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_tokens)},
                            {onnxruntime::narrow<uint32_t>(batch)},
                            {onnxruntime::narrow<uint32_t>(hq)},
                            {onnxruntime::narrow<uint32_t>(hv)},
                            {onnxruntime::narrow<uint32_t>(dk)},
                            {onnxruntime::narrow<uint32_t>(dv)},
                            {scale_ != 0.0f ? scale_ : 1.0f / std::sqrt(static_cast<float>(dk))}});
  return context.RunProgram(program);
}

}  // namespace onnxruntime::contrib::webgpu
