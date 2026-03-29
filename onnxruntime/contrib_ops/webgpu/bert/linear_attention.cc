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
  ORT_THROW("Unknown update rule: ", rule_str);
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

  // Shared memory for parallel reduction across dk threads
  // and for broadcasting delta values.
  // When use_vec4, each reduction_buf entry is a vec4<f32> (4 dv values packed),
  // eliminating the inner TILE_V loop and enabling native SIMD operations.
  if (use_vec4) {
    shader.AdditionalImplementation()
        << "var<workgroup> reduction_buf: array<vec4<f32>, workgroup_size_x>;\n"
        << "var<workgroup> broadcast_val: vec4<f32>;\n";
  } else {
    // TILE_V is emitted as a compile-time constant (not overridable) because
    // private address space arrays require fixed sizes in WGSL.
    shader.AdditionalImplementation()
        << "const TILE_V: u32 = " << tile_v_ << "u;\n"
        << "var<workgroup> reduction_buf: array<f32, workgroup_size_x * TILE_V>;\n"
        << "var<workgroup> broadcast_buf: array<f32, TILE_V>;\n";
  }

  // Identify which (batch, head, dv_tile) this workgroup handles
  shader.MainFunctionBody()
      << "let bh = workgroup_idx / uniforms.num_dv_tiles;\n"
      << "let dv_tile_idx = workgroup_idx % uniforms.num_dv_tiles;\n"
      << "let batch_idx = bh / uniforms.num_heads;\n"
      << "let head_idx = bh % uniforms.num_heads;\n"
      << "let dk_idx = local_idx;  // thread index = row in state matrix\n";
  if (!use_vec4) {
    shader.MainFunctionBody() << "let dv_start = dv_tile_idx * TILE_V;\n";
  }
  // Precompute packed strides for 3D packed inputs (B, T, H*D)
  // When use_vec4, head_dim_v is already divided by 4 (vectorized), so
  // packed_dv = num_heads * (head_dim_v/4) and dv_tile_idx indexes vec4 elements.
  shader.MainFunctionBody()
      << "\n"
      << "let packed_dk = uniforms.num_heads * uniforms.head_dim_k;\n"
      << "let packed_dv = uniforms.num_heads * uniforms.head_dim_v;\n"
      << "\n";

  // Initialize state tile in private memory
  if (use_vec4) {
    shader.MainFunctionBody() << "var state = vec4<f32>(0.0);\n";
  } else {
    shader.MainFunctionBody()
        << "var state: array<f32, TILE_V>;\n"
        << "for (var j = 0u; j < TILE_V; j++) {\n"
        << "  state[j] = 0.0;\n"
        << "}\n";
  }

  // Load initial state if provided
  if (has_initial_state_) {
    shader.MainFunctionBody() << "if (dk_idx < uniforms.head_dim_k) {\n";
    if (use_vec4) {
      shader.MainFunctionBody()
          << "  let state_offset = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.head_dim_k + dk_idx) * uniforms.head_dim_v + dv_tile_idx;\n"
          << "  state = vec4<f32>(initial_state[state_offset]);\n";
    } else {
      shader.MainFunctionBody()
          << "  let state_base = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.head_dim_k + dk_idx) * uniforms.head_dim_v + dv_start;\n"
          << "  for (var j = 0u; j < TILE_V; j++) {\n"
          << "    if (dv_start + j < uniforms.head_dim_v) {\n"
          << "      state[j] = f32(initial_state[state_base + j]);\n"
          << "    }\n"
          << "  }\n";
    }
    shader.MainFunctionBody() << "}\n";
  }

  // Main token processing loop
  shader.MainFunctionBody()
      << "\n// Process each token sequentially\n"
      << "for (var t = 0u; t < uniforms.seq_length; t++) {\n"
      << "  let bt_offset = batch_idx * uniforms.seq_length + t;\n"
      << "  var k_val: f32 = 0.0;\n"
      << "  var q_val: f32 = 0.0;\n"
      << "  if (dk_idx < uniforms.head_dim_k) {\n"
      << "    let qk_idx = bt_offset * packed_dk + head_idx * uniforms.head_dim_k + dk_idx;\n"
      << "    k_val = f32(key[qk_idx]);\n"
      << "    q_val = f32(query[qk_idx]);\n"
      << "  }\n";

  // Step 1: Apply decay (for gated and gated_delta modes)
  if (update_rule_ == LinearAttentionUpdateRule::Gated || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    shader.MainFunctionBody()
        << "\n  // Apply exponential decay: S *= exp(decay)\n";
    if (decay_broadcast_dk_) {
      shader.MainFunctionBody()
          << "  let exp_g = exp(f32(decay[bt_offset * uniforms.num_heads + head_idx]));\n";
    } else {
      shader.MainFunctionBody()
          << "  var exp_g: f32 = 1.0;\n"
          << "  if (dk_idx < uniforms.head_dim_k) {\n"
          << "    exp_g = exp(f32(decay[bt_offset * packed_dk + head_idx * uniforms.head_dim_k + dk_idx]));\n"
          << "  }\n";
    }
    if (use_vec4) {
      shader.MainFunctionBody() << "  state *= exp_g;\n";
    } else {
      shader.MainFunctionBody()
          << "  for (var j = 0u; j < TILE_V; j++) {\n"
          << "    state[j] *= exp_g;\n"
          << "  }\n";
    }
  }

  // Step 2: For delta/gated_delta rules, compute retrieved = S^T @ k (reduction across dk)
  if (update_rule_ == LinearAttentionUpdateRule::Delta || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    if (use_vec4) {
      shader.MainFunctionBody()
          << "\n  // Compute retrieved = S^T @ k (parallel reduction over dk)\n"
          << "  reduction_buf[dk_idx] = state * k_val;\n"
          << "  workgroupBarrier();\n"
          << "  for (var stride = workgroup_size_x >> 1u; stride > 0u; stride = stride >> 1u) {\n"
          << "    if (dk_idx < stride) {\n"
          << "      reduction_buf[dk_idx] += reduction_buf[dk_idx + stride];\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "  }\n"
          << "  // Compute delta = beta * (v - retrieved) and broadcast\n"
          << "  let v_idx = bt_offset * packed_dv + head_idx * uniforms.head_dim_v + dv_tile_idx;\n"
          << "  let beta_base = bt_offset * uniforms.num_heads + head_idx;\n"
          << "  if (dk_idx == 0u) {\n"
          << "    let beta_val = f32(beta[beta_base]);\n"
          << "    broadcast_val = beta_val * (vec4<f32>(value[v_idx]) - reduction_buf[0]);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  state += k_val * broadcast_val;\n"
          << "  workgroupBarrier();\n";
    } else {
      shader.MainFunctionBody()
          << "\n  // Compute retrieved = S^T @ k (parallel reduction over dk)\n"
          << "  for (var j = 0u; j < TILE_V; j++) {\n"
          << "    reduction_buf[j * workgroup_size_x + dk_idx] = state[j] * k_val;\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  // Tree reduction\n"
          << "  for (var stride = workgroup_size_x >> 1u; stride > 0u; stride = stride >> 1u) {\n"
          << "    if (dk_idx < stride) {\n"
          << "      for (var j = 0u; j < TILE_V; j++) {\n"
          << "        reduction_buf[j * workgroup_size_x + dk_idx] += reduction_buf[j * workgroup_size_x + dk_idx + stride];\n"
          << "      }\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "  }\n"
          << "  // Compute delta = beta * (v - retrieved) and broadcast\n"
          << "  let v_base = bt_offset * packed_dv + head_idx * uniforms.head_dim_v + dv_start;\n"
          << "  let beta_base = bt_offset * uniforms.num_heads + head_idx;\n"
          << "  if (dk_idx == 0u) {\n"
          << "    let beta_val = f32(beta[beta_base]);\n"
          << "    for (var j = 0u; j < TILE_V; j++) {\n"
          << "      if (dv_start + j < uniforms.head_dim_v) {\n"
          << "        let retrieved_j = reduction_buf[j * workgroup_size_x];\n"
          << "        let v_val = f32(value[v_base + j]);\n"
          << "        broadcast_buf[j] = beta_val * (v_val - retrieved_j);\n"
          << "      }\n"
          << "    }\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  // Update state: S += k ⊗ delta\n"
          << "  for (var j = 0u; j < TILE_V; j++) {\n"
          << "    state[j] += k_val * broadcast_buf[j];\n"
          << "  }\n"
          << "  workgroupBarrier();\n";
    }
  } else {
    // For linear and gated modes: S += k ⊗ v (no delta rule)
    if (use_vec4) {
      shader.MainFunctionBody()
          << "\n  // Update state: S += k ⊗ v\n"
          << "  let v_idx = bt_offset * packed_dv + head_idx * uniforms.head_dim_v + dv_tile_idx;\n"
          << "  state += k_val * vec4<f32>(value[v_idx]);\n";
    } else {
      shader.MainFunctionBody()
          << "\n  // Update state: S += k ⊗ v\n"
          << "  let v_base = bt_offset * packed_dv + head_idx * uniforms.head_dim_v + dv_start;\n"
          << "  for (var j = 0u; j < TILE_V; j++) {\n"
          << "    if (dv_start + j < uniforms.head_dim_v) {\n"
          << "      let v_val = f32(value[v_base + j]);\n"
          << "      state[j] += k_val * v_val;\n"
          << "    }\n"
          << "  }\n";
    }
  }

  // Step 3: Compute output = scale * S^T @ q (reduction across dk)
  if (use_vec4) {
    shader.MainFunctionBody()
        << "\n  // Compute output = scale * S^T @ q (parallel reduction over dk)\n"
        << "  reduction_buf[dk_idx] = state * q_val;\n"
        << "  workgroupBarrier();\n"
        << "  for (var stride = workgroup_size_x >> 1u; stride > 0u; stride = stride >> 1u) {\n"
        << "    if (dk_idx < stride) {\n"
        << "      reduction_buf[dk_idx] += reduction_buf[dk_idx + stride];\n"
        << "    }\n"
        << "    workgroupBarrier();\n"
        << "  }\n"
        << "  if (dk_idx == 0u) {\n"
        << "    let out_idx = bt_offset * packed_dv + head_idx * uniforms.head_dim_v + dv_tile_idx;\n"
        << "    output[out_idx] = output_value_t(reduction_buf[0] * uniforms.scale);\n"
        << "  }\n"
        << "  workgroupBarrier();\n"
        << "}\n";  // end token loop
  } else {
    shader.MainFunctionBody()
        << "\n  // Compute output = scale * S^T @ q (parallel reduction over dk)\n"
        << "  for (var j = 0u; j < TILE_V; j++) {\n"
        << "    reduction_buf[j * workgroup_size_x + dk_idx] = state[j] * q_val;\n"
        << "  }\n"
        << "  workgroupBarrier();\n"
        << "  for (var stride = workgroup_size_x >> 1u; stride > 0u; stride = stride >> 1u) {\n"
        << "    if (dk_idx < stride) {\n"
        << "      for (var j = 0u; j < TILE_V; j++) {\n"
        << "        reduction_buf[j * workgroup_size_x + dk_idx] += reduction_buf[j * workgroup_size_x + dk_idx + stride];\n"
        << "      }\n"
        << "    }\n"
        << "    workgroupBarrier();\n"
        << "  }\n"
        << "  if (dk_idx == 0u) {\n"
        << "    let out_base = bt_offset * packed_dv + head_idx * uniforms.head_dim_v + dv_start;\n"
        << "    for (var j = 0u; j < TILE_V; j++) {\n"
        << "      if (dv_start + j < uniforms.head_dim_v) {\n"
        << "        output[out_base + j] = output_element_t(reduction_buf[j * workgroup_size_x] * uniforms.scale);\n"
        << "      }\n"
        << "    }\n"
        << "  }\n"
        << "  workgroupBarrier();\n"
        << "}\n";  // end token loop
  }

  // Write final state (4D: B, H_kv, dk, dv)
  shader.MainFunctionBody()
      << "\n// Write present_state\n"
      << "if (dk_idx < uniforms.head_dim_k) {\n";
  if (use_vec4) {
    shader.MainFunctionBody()
        << "  let state_offset = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.head_dim_k + dk_idx) * uniforms.head_dim_v + dv_tile_idx;\n"
        << "  present_state[state_offset] = output_value_t(state);\n";
  } else {
    shader.MainFunctionBody()
        << "  let state_base = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.head_dim_k + dk_idx) * uniforms.head_dim_v + dv_start;\n"
        << "  for (var j = 0u; j < TILE_V; j++) {\n"
        << "    if (dv_start + j < uniforms.head_dim_v) {\n"
        << "      present_state[state_base + j] = output_element_t(state[j]);\n"
        << "    }\n"
        << "  }\n";
  }
  shader.MainFunctionBody() << "}\n";

  return Status::OK();
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
  std::string update_rule_str = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
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
  const Tensor* past_state = context.Input(3);   // optional
  const Tensor* decay = context.Input(4);         // optional
  const Tensor* beta = context.Input(5);          // optional

  // Validate 3D packed inputs
  const auto& q_shape = query->Shape();
  ORT_RETURN_IF(q_shape.NumDimensions() != 3, "query must be 3D (B, T, H_q*d_k)");

  const int batch_size = static_cast<int>(q_shape[0]);
  const int seq_length = static_cast<int>(q_shape[1]);
  const int q_packed_dim = static_cast<int>(q_shape[2]);
  const int num_heads = kv_num_heads_;

  ORT_RETURN_IF(q_num_heads_ != kv_num_heads_,
                "GQA (q_num_heads != kv_num_heads) is not yet supported");

  const int head_dim_k = q_packed_dim / q_num_heads_;
  ORT_RETURN_IF(q_packed_dim != head_dim_k * q_num_heads_,
                "query packed dim must be divisible by q_num_heads");

  const int v_packed_dim = static_cast<int>(value->Shape()[2]);
  const int head_dim_v = v_packed_dim / kv_num_heads_;
  ORT_RETURN_IF(v_packed_dim != head_dim_v * kv_num_heads_,
                "value packed dim must be divisible by kv_num_heads");

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
  TensorShapeVector output_shape({batch_size, seq_length, q_num_heads_ * head_dim_v});
  Tensor* output = context.Output(0, output_shape);

  TensorShapeVector state_shape({batch_size, num_heads, head_dim_k, head_dim_v});
  Tensor* present_state = context.Output(1, state_shape);

  // Vectorization: when head_dim_v is divisible by 4, use vec4 to pack 4 dv values
  // per element. This replaces scalar TILE_V loops with native vec4 SIMD operations,
  // reduces shared memory access overhead, and enables coalesced memory reads/writes.
  const int components = (head_dim_v % 4 == 0 && head_dim_v >= 4) ? 4 : 1;
  int tile_v = (components == 4) ? 1 : 4;
  if (components == 1 && head_dim_v <= 4) {
    tile_v = head_dim_v;
  }
  const int head_dim_v_vectorized = head_dim_v / components;
  const int num_dv_tiles = (head_dim_v_vectorized + tile_v - 1) / tile_v;

  // Workgroup size = head_dim_k (one thread per dk row)
  // Ensure it's a power of 2 for tree reduction (round up)
  uint32_t workgroup_size = 1;
  while (workgroup_size < static_cast<uint32_t>(head_dim_k)) {
    workgroup_size *= 2;
  }
  // Cap at GPU limits
  workgroup_size = std::min(workgroup_size, static_cast<uint32_t>(256));

  const uint32_t num_workgroups = batch_size * num_heads * num_dv_tiles;

  bool has_initial_state = past_state != nullptr;
  bool has_decay = decay != nullptr;
  bool has_beta = beta != nullptr;

  // Detect whether decay is (B,T,H_kv) or (B,T,H_kv*dk)
  bool decay_broadcast_dk = false;
  if (has_decay) {
    const auto& decay_shape = decay->Shape();
    // (B, T, H_kv) = 3D with last dim == num_heads
    int decay_last_dim = static_cast<int>(decay_shape[decay_shape.NumDimensions() - 1]);
    if (decay_last_dim == num_heads) {
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
                            {static_cast<uint32_t>(num_heads)},
                            {static_cast<uint32_t>(seq_length)},
                            {static_cast<uint32_t>(head_dim_k)},
                            {static_cast<uint32_t>(head_dim_v_vectorized)},
                            {scale},
                            {static_cast<uint32_t>(num_dv_tiles)}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
