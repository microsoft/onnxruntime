// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/linear_attention.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

using namespace onnxruntime::webgpu;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

LinearAttentionUpdateRule ParseUpdateRule(const std::string& rule_str) {
  if (rule_str == "linear") {
    return LinearAttentionUpdateRule::Linear;
  } else if (rule_str == "gated") {
    return LinearAttentionUpdateRule::Gated;
  } else if (rule_str == "delta") {
    return LinearAttentionUpdateRule::Delta;
  } else if (rule_str == "gated_delta") {
    return LinearAttentionUpdateRule::GatedDelta;
  }
  return LinearAttentionUpdateRule::Invalid;
}

// =============================================================================
// LinearAttention Shader Implementation
// =============================================================================
//
// Design overview:
// - Each workgroup handles one (batch, head, dv_tile) combination
// - Workgroup size = head_dim_k (dk): one thread per state row
// - Each thread maintains TILE_V columns of its state row in private memory
// - Tokens are processed sequentially; matrix ops are parallelized across threads
// - Reductions across dk (for S^T @ k and S^T @ q) use shared memory
//

Status LinearAttentionProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const bool use_vec4 = (components_ == 4);

  // Map update rule to integer for template conditionals
  int update_rule_int = 0;
  switch (update_rule_) {
    case LinearAttentionUpdateRule::Linear:
      update_rule_int = 0;
      break;
    case LinearAttentionUpdateRule::Gated:
      update_rule_int = 1;
      break;
    case LinearAttentionUpdateRule::Delta:
      update_rule_int = 2;
      break;
    case LinearAttentionUpdateRule::GatedDelta:
      update_rule_int = 3;
      break;
    case LinearAttentionUpdateRule::Invalid:
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid update rule");
  }

  // Add inputs
  shader.AddInput("query", ShaderUsage::UseUniform);
  shader.AddInput("key", ShaderUsage::UseUniform);
  shader.AddInput("value", ShaderUsage::UseUniform);
  if (has_initial_state_) {
    shader.AddInput("initial_state", ShaderUsage::UseUniform);
  }
  if (has_decay_) {
    shader.AddInput("decay", ShaderUsage::UseUniform);
  }
  if (has_beta_) {
    shader.AddInput("beta", ShaderUsage::UseUniform);
  }

  // Add outputs - UseValueTypeAlias for vec4 writes, UseElementTypeAlias for scalar writes
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("present_state", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(shader, "bert/linear_attention.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(decay_broadcast_dk, decay_broadcast_dk_),
                             WGSL_TEMPLATE_PARAMETER(has_initial_state, has_initial_state_),
                             WGSL_TEMPLATE_PARAMETER(tile_v, tile_v_),
                             WGSL_TEMPLATE_PARAMETER(update_rule, update_rule_int),
                             WGSL_TEMPLATE_PARAMETER(use_vec4, use_vec4));
}

// =============================================================================
// LinearAttention Kernel Registration and Computation
// =============================================================================

ONNX_OPERATOR_KERNEL_EX(
    LinearAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    LinearAttention);

LinearAttention::LinearAttention(const OpKernelInfo& info)
    : WebGpuKernel(info) {
  std::string update_rule_str = info.GetAttr<std::string>("update_rule");
  update_rule_ = ParseUpdateRule(update_rule_str);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  q_num_heads_ = static_cast<int>(info.GetAttr<int64_t>("q_num_heads"));
  kv_num_heads_ = static_cast<int>(info.GetAttr<int64_t>("kv_num_heads"));
}

/*
  3D packed inputs:
      query:         (B, T, H_q * d_k)   — packed query
      key:           (B, T, H_kv * d_k)  — packed key
      value:         (B, T, H_kv * d_v)  — packed value
      past_state:    (B, H_kv, d_k, d_v) — recurrent state (4D)
      decay:         (B, T, H_kv * d_k) or (B, T, H_kv) — decay gate (3D)
      beta:          (B, T, H_kv) or (B, T, 1)           — update rate (3D)

  Outputs:
      output:        (B, T, H_q * d_v)   — packed attention output
      present_state: (B, H_kv, d_k, d_v) — updated recurrent state (4D)
*/
Status LinearAttention::ComputeInternal(ComputeContext& context) const {
  const Tensor* query = context.Input(0);
  const Tensor* key = context.Input(1);
  const Tensor* value = context.Input(2);
  const Tensor* past_state = context.Input(3);  // optional
  const Tensor* decay = context.Input(4);       // optional
  const Tensor* beta = context.Input(5);        // optional

  // Validate 3D packed inputs
  const auto& q_shape = query->Shape();
  ORT_RETURN_IF(q_shape.NumDimensions() != 3, "query must be 3D (B, T, H_q*d_k)");
  const auto& k_shape = key->Shape();
  ORT_RETURN_IF(k_shape.NumDimensions() != 3, "key must be 3D (B, T, H_k*d_k)");
  const auto& v_shape = value->Shape();
  ORT_RETURN_IF(v_shape.NumDimensions() != 3, "value must be 3D (B, T, H_v*d_v)");

  const int64_t batch_size = q_shape[0];
  const int64_t seq_length = q_shape[1];
  ORT_RETURN_IF(k_shape[0] != batch_size || k_shape[1] != seq_length,
                "key batch/sequence dimensions must match query");
  ORT_RETURN_IF(v_shape[0] != batch_size || v_shape[1] != seq_length,
                "value batch/sequence dimensions must match query");

  const int64_t q_packed_dim = q_shape[2];
  ORT_RETURN_IF(q_num_heads_ <= 0 || q_packed_dim % q_num_heads_ != 0,
                "query packed dim must be divisible by q_num_heads");
  const int64_t head_dim_k = q_packed_dim / q_num_heads_;
  const int64_t k_packed_dim = k_shape[2];
  ORT_RETURN_IF(k_packed_dim % head_dim_k != 0,
                "key packed dim must be divisible by query head dimension");
  const int64_t n_k_heads = k_packed_dim / head_dim_k;
  const int64_t v_packed_dim = v_shape[2];
  const int64_t head_dim_v = v_packed_dim / kv_num_heads_;
  ORT_RETURN_IF(v_packed_dim != head_dim_v * kv_num_heads_,
                "value packed dim must be divisible by kv_num_heads");

  // ==== GQA head mapping ====
  // Standard GQA: q_num_heads >= kv_num_heads, multiple Q heads per KV group.
  // Inverse GQA: q_num_heads < kv_num_heads (e.g., Qwen3.5 9B: n_q=16, n_kv=32).
  // Also n_k_heads may differ from both (K has its own head count).
  int64_t heads_per_group;  // Q heads per KV group (0 if inverse GQA)
  if (q_num_heads_ >= kv_num_heads_) {
    ORT_RETURN_IF_NOT(q_num_heads_ % kv_num_heads_ == 0,
                      "q_num_heads must be divisible by kv_num_heads");
    heads_per_group = q_num_heads_ / kv_num_heads_;
  } else {
    ORT_RETURN_IF_NOT(kv_num_heads_ % q_num_heads_ == 0,
                      "kv_num_heads must be divisible by q_num_heads (inverse GQA)");
    heads_per_group = 0;  // signals inverse GQA
  }

  // K-to-KV head mapping: when n_k < kv_num_heads, multiple KV heads share one K head
  ORT_RETURN_IF_NOT(kv_num_heads_ % n_k_heads == 0,
                    "kv_num_heads must be divisible by n_k_heads");
  int64_t kv_per_k_head = kv_num_heads_ / n_k_heads;

  // Validate update rule has required inputs
  bool needs_decay = (update_rule_ == LinearAttentionUpdateRule::Gated ||
                      update_rule_ == LinearAttentionUpdateRule::GatedDelta);
  bool needs_beta = (update_rule_ == LinearAttentionUpdateRule::Delta ||
                     update_rule_ == LinearAttentionUpdateRule::GatedDelta);
  ORT_RETURN_IF(needs_decay && decay == nullptr, "decay input required for gated/gated_delta update rules");
  ORT_RETURN_IF(needs_beta && beta == nullptr, "beta input required for delta/gated_delta update rules");

  // Compute scale: 0.0 means derive from d_k
  float scale = scale_;
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));
  }

  // Allocate outputs — output is 3D packed, state is 4D
  // Output uses kv_num_heads (matches schema inference: output_dim == V_dim).
  // For inverse GQA (q < kv): each KV head writes its own output slot.
  // For standard/MHA (q >= kv): q == kv with this schema, so equivalent.
  TensorShapeVector output_shape({batch_size, seq_length, kv_num_heads_ * head_dim_v});
  Tensor* output = context.Output(0, output_shape);

  TensorShapeVector state_shape({batch_size, kv_num_heads_, head_dim_k, head_dim_v});
  Tensor* present_state = context.Output(1, state_shape);

  // Vectorization: when head_dim_v is divisible by 4, use vec4 to pack 4 dv values
  // per element. This replaces scalar TILE_V loops with native vec4 SIMD operations,
  // reduces shared memory access overhead, and enables coalesced memory reads/writes.
  const int components = (head_dim_v % 4 == 0 && head_dim_v >= 4) ? 4 : 1;
  int tile_v = (components == 4) ? 1 : 4;
  if (components == 1 && head_dim_v <= 4) {
    tile_v = onnxruntime::narrow<int>(head_dim_v);
  }
  const int head_dim_v_vectorized = onnxruntime::narrow<int>(head_dim_v) / components;

  constexpr uint32_t kMaxSupportedWorkgroupSize = 256;
  ORT_RETURN_IF_NOT(head_dim_k <= static_cast<int64_t>(kMaxSupportedWorkgroupSize),
                    "LinearAttention WebGPU kernel requires head_dim_k <= ",
                    kMaxSupportedWorkgroupSize,
                    ", got ",
                    head_dim_k);
  uint32_t workgroup_size = 1;
  while (workgroup_size < static_cast<uint32_t>(head_dim_k)) {
    workgroup_size *= 2;
  }
  // Cap at GPU limits
  workgroup_size = std::min(workgroup_size, kMaxSupportedWorkgroupSize);

  const int num_dv_tiles = (head_dim_v_vectorized + tile_v - 1) / tile_v;
  const uint32_t num_workgroups = onnxruntime::narrow<uint32_t>(batch_size * kv_num_heads_ * num_dv_tiles);

  bool has_initial_state = past_state != nullptr;
  bool has_decay = decay != nullptr;
  bool has_beta = beta != nullptr;

  // Detect whether decay is (B,T,H_kv) or (B,T,H_kv*dk)
  bool decay_broadcast_dk = false;
  if (has_decay) {
    const auto& decay_shape = decay->Shape();
    // (B, T, H_kv) = 3D with last dim == num_heads
    int decay_last_dim = static_cast<int>(decay_shape[decay_shape.NumDimensions() - 1]);
    if (decay_last_dim == kv_num_heads_) {
      decay_broadcast_dk = true;
    }
  }

  LinearAttentionProgram program{update_rule_, has_initial_state, has_decay, has_beta, decay_broadcast_dk, tile_v, components};

  program.AddInputs({{query, ProgramTensorMetadataDependency::TypeAndRank},
                     {key, ProgramTensorMetadataDependency::TypeAndRank},
                     {value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (has_initial_state) {
    program.AddInput({past_state, ProgramTensorMetadataDependency::TypeAndRank, components});
  }
  if (has_decay) {
    program.AddInput({decay, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (has_beta) {
    program.AddInput({beta, ProgramTensorMetadataDependency::TypeAndRank});
  }

  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components},
                      {present_state, ProgramTensorMetadataDependency::TypeAndRank, components}});

  program.SetDispatchGroupSize(num_workgroups)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(std::to_string(static_cast<int>(update_rule_)),
                 has_initial_state, has_decay, has_beta, decay_broadcast_dk, tile_v, components)
      .AddUniformVariables({{static_cast<uint32_t>(batch_size)},
                            {static_cast<uint32_t>(kv_num_heads_)},
                            {static_cast<uint32_t>(seq_length)},
                            {static_cast<uint32_t>(head_dim_k)},
                            {static_cast<uint32_t>(head_dim_v_vectorized)},
                            {scale},
                            {static_cast<uint32_t>(num_dv_tiles)},
                            {static_cast<uint32_t>(heads_per_group)},
                            {static_cast<uint32_t>(kv_per_k_head)},
                            {static_cast<uint32_t>(q_num_heads_)},
                            {static_cast<uint32_t>(n_k_heads)}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
