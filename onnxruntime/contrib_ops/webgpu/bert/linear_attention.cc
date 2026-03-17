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

  // Add outputs
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("final_state", ShaderUsage::UseUniform);

  // Shared memory for parallel reduction across dk threads
  // and for broadcasting delta values
  // TILE_V is emitted as a compile-time constant (not overridable) because
  // private address space arrays require fixed sizes in WGSL.
  shader.AdditionalImplementation()
      << "const TILE_V: u32 = " << tile_v_ << "u;\n"
      << "var<workgroup> reduction_buf: array<f32, workgroup_size_x * TILE_V>;\n"
      << "var<workgroup> broadcast_buf: array<f32, TILE_V>;\n";

  shader.MainFunctionBody()
      // Identify which (batch, head, dv_tile) this workgroup handles
      // workgroup_idx is already defined by the framework
      << "let bh = workgroup_idx / uniforms.num_dv_tiles;\n"
      << "let dv_tile_idx = workgroup_idx % uniforms.num_dv_tiles;\n"
      << "let batch_idx = bh / uniforms.num_heads;\n"
      << "let head_idx = bh % uniforms.num_heads;\n"
      << "let dk_idx = local_idx;  // thread index = row in state matrix\n"
      << "let dv_start = dv_tile_idx * TILE_V;\n"
      << "\n"
      // Initialize state tile in private memory
      << "var state: array<f32, TILE_V>;\n"
      << "for (var j = 0u; j < TILE_V; j++) {\n"
      << "  state[j] = 0.0;\n"
      << "}\n";

  // Load initial state if provided
  if (has_initial_state_) {
    shader.MainFunctionBody()
        << "// Load initial state: initial_state[batch, head, dk_idx, dv_start..dv_start+TILE_V]\n"
        << "let state_base = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.head_dim_k + dk_idx) * uniforms.head_dim_v + dv_start;\n"
        << "for (var j = 0u; j < TILE_V; j++) {\n"
        << "  if (dv_start + j < uniforms.head_dim_v) {\n"
        << "    state[j] = f32(initial_state[state_base + j]);\n"
        << "  }\n"
        << "}\n";
  }

  // Main token processing loop
  shader.MainFunctionBody()
      << "\n// Process each token sequentially\n"
      << "for (var t = 0u; t < uniforms.seq_length; t++) {\n"
      // Load k and q for this thread's dk row
      << "  let qkv_bh_offset = (batch_idx * uniforms.num_heads + head_idx) * uniforms.seq_length;\n"
      << "  let k_base = (qkv_bh_offset + t) * uniforms.head_dim_k + dk_idx;\n"
      << "  let k_val = f32(key[k_base]);\n"
      << "  let q_val = f32(query[k_base]);\n";

  // Step 1: Apply decay (for gated and gated_delta modes)
  if (update_rule_ == LinearAttentionUpdateRule::Gated || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    shader.MainFunctionBody()
        << "\n  // Apply exponential decay: S *= exp(decay)\n"
        << "  let decay_base = (qkv_bh_offset + t) * uniforms.head_dim_k + dk_idx;\n"
        << "  let exp_g = exp(f32(decay[decay_base]));\n"
        << "  for (var j = 0u; j < TILE_V; j++) {\n"
        << "    state[j] *= exp_g;\n"
        << "  }\n";
  }

  // Step 2: For delta/gated_delta rules, compute retrieved = S^T @ k (reduction across dk)
  if (update_rule_ == LinearAttentionUpdateRule::Delta || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
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
        // Thread 0 computes delta and broadcasts via shared memory
        << "  // Compute delta = beta * (v - retrieved) and broadcast\n"
        << "  let v_base = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.seq_length + t) * uniforms.head_dim_v + dv_start;\n"
        << "  let beta_base = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.seq_length + t);\n"
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
        // All threads update their state row using the broadcast delta
        << "  // Update state: S += k ⊗ delta\n"
        << "  for (var j = 0u; j < TILE_V; j++) {\n"
        << "    state[j] += k_val * broadcast_buf[j];\n"
        << "  }\n"
        << "  workgroupBarrier();\n";
  } else {
    // For linear and gated modes: S += k ⊗ v (no delta rule)
    shader.MainFunctionBody()
        << "\n  // Update state: S += k ⊗ v\n"
        << "  let v_base = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.seq_length + t) * uniforms.head_dim_v + dv_start;\n"
        << "  for (var j = 0u; j < TILE_V; j++) {\n"
        << "    if (dv_start + j < uniforms.head_dim_v) {\n"
        << "      let v_val = f32(value[v_base + j]);\n"
        << "      state[j] += k_val * v_val;\n"
        << "    }\n"
        << "  }\n";
  }

  // Step 3: Compute output = scale * S^T @ q (reduction across dk)
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
      // Thread 0 writes the output for this token and dv_tile
      << "  if (dk_idx == 0u) {\n"
      << "    let out_base = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.seq_length + t) * uniforms.head_dim_v + dv_start;\n"
      << "    for (var j = 0u; j < TILE_V; j++) {\n"
      << "      if (dv_start + j < uniforms.head_dim_v) {\n"
      << "        output[out_base + j] = output_element_t(reduction_buf[j * workgroup_size_x] * uniforms.scale);\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << "}\n";  // end token loop

  // Write final state
  shader.MainFunctionBody()
      << "\n// Write final state\n"
      << "let final_state_base = ((batch_idx * uniforms.num_heads + head_idx) * uniforms.head_dim_k + dk_idx) * uniforms.head_dim_v + dv_start;\n"
      << "for (var j = 0u; j < TILE_V; j++) {\n"
      << "  if (dv_start + j < uniforms.head_dim_v) {\n"
      << "    final_state[final_state_base + j] = output_element_t(state[j]);\n"
      << "  }\n"
      << "}\n";

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
  chunk_size_ = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
}

Status LinearAttention::ComputeInternal(ComputeContext& context) const {
  const Tensor* query = context.Input(0);
  const Tensor* key = context.Input(1);
  const Tensor* value = context.Input(2);
  const Tensor* initial_state = context.Input(3);  // optional
  const Tensor* decay = context.Input(4);           // optional
  const Tensor* beta = context.Input(5);            // optional

  // Validate inputs
  const auto& q_shape = query->Shape();
  ORT_RETURN_IF(q_shape.NumDimensions() != 4, "query must be 4D (B, H, T, dk)");

  const int batch_size = static_cast<int>(q_shape[0]);
  const int num_heads = static_cast<int>(q_shape[1]);
  const int seq_length = static_cast<int>(q_shape[2]);
  const int head_dim_k = static_cast<int>(q_shape[3]);
  const int head_dim_v = static_cast<int>(value->Shape()[3]);

  // Validate update rule has required inputs
  bool needs_decay = (update_rule_ == LinearAttentionUpdateRule::Gated ||
                      update_rule_ == LinearAttentionUpdateRule::GatedDelta);
  bool needs_beta = (update_rule_ == LinearAttentionUpdateRule::Delta ||
                     update_rule_ == LinearAttentionUpdateRule::GatedDelta);
  ORT_RETURN_IF(needs_decay && decay == nullptr, "decay input required for gated/gated_delta update rules");
  ORT_RETURN_IF(needs_beta && beta == nullptr, "beta input required for delta/gated_delta update rules");

  // Compute scale
  float scale = scale_;
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));
  }

  // Allocate outputs
  TensorShapeVector output_shape({batch_size, num_heads, seq_length, head_dim_v});
  Tensor* output = context.Output(0, output_shape);

  TensorShapeVector state_shape({batch_size, num_heads, head_dim_k, head_dim_v});
  Tensor* final_state = context.Output(1, state_shape);

  // Choose tile size: balance parallelism vs shared memory
  // TILE_V * WORKGROUP_SIZE * 4 bytes must fit in shared memory (typically 16KB limit)
  // E.g., TILE_V=4, WORKGROUP_SIZE=128: 4*128*4 = 2048 bytes
  int tile_v = 4;
  if (head_dim_v <= 4) {
    tile_v = head_dim_v;
  }
  const int num_dv_tiles = (head_dim_v + tile_v - 1) / tile_v;

  // Workgroup size = head_dim_k (one thread per dk row)
  // Ensure it's a power of 2 for tree reduction (round up)
  uint32_t workgroup_size = 1;
  while (workgroup_size < static_cast<uint32_t>(head_dim_k)) {
    workgroup_size *= 2;
  }
  // Cap at GPU limits
  workgroup_size = std::min(workgroup_size, static_cast<uint32_t>(256));

  const uint32_t num_workgroups = batch_size * num_heads * num_dv_tiles;

  bool has_initial_state = initial_state != nullptr;
  bool has_decay = decay != nullptr;
  bool has_beta = beta != nullptr;

  LinearAttentionProgram program{update_rule_, has_initial_state, has_decay, has_beta, tile_v};

  program.AddInputs({{query, ProgramTensorMetadataDependency::TypeAndRank},
                     {key, ProgramTensorMetadataDependency::TypeAndRank},
                     {value, ProgramTensorMetadataDependency::TypeAndRank}});
  if (has_initial_state) {
    program.AddInput({initial_state, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (has_decay) {
    program.AddInput({decay, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (has_beta) {
    program.AddInput({beta, ProgramTensorMetadataDependency::TypeAndRank});
  }

  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank},
                      {final_state, ProgramTensorMetadataDependency::TypeAndRank}});

  program.SetDispatchGroupSize(num_workgroups)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(std::to_string(static_cast<int>(update_rule_)),
                 has_initial_state, has_decay, has_beta, tile_v)
      .AddUniformVariables({{static_cast<uint32_t>(batch_size)},
                            {static_cast<uint32_t>(num_heads)},
                            {static_cast<uint32_t>(seq_length)},
                            {static_cast<uint32_t>(head_dim_k)},
                            {static_cast<uint32_t>(head_dim_v)},
                            {scale},
                            {static_cast<uint32_t>(num_dv_tiles)}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
