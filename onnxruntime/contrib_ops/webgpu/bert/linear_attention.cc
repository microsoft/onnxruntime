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
// LinearAttentionRecurrent Implementation
// =============================================================================

ONNX_OPERATOR_KERNEL_EX(
    LinearAttentionRecurrent,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    LinearAttentionRecurrent);

LinearAttentionRecurrent::LinearAttentionRecurrent(const OpKernelInfo& info)
    : WebGpuKernel(info) {
  std::string update_rule_str = info.GetAttrOrDefault<std::string>("update_rule", "gated_delta");
  update_rule_ = ParseUpdateRule(update_rule_str);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

Status LinearAttentionRecurrentProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Input tensors - with proper accessor methods and element type alias for scaling
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
  const auto& value = shader.AddInput("value", ShaderUsage::UseUniform);
  const auto& past_state = shader.AddInput("past_state", ShaderUsage::UseUniform);

  // Optional inputs based on update rule
  const ShaderVariableHelper* decay_ptr = nullptr;
  const ShaderVariableHelper* beta_ptr = nullptr;
  if (has_decay_) {
    decay_ptr = &shader.AddInput("decay", ShaderUsage::UseUniform);
  }
  if (has_beta_) {
    beta_ptr = &shader.AddInput("beta", ShaderUsage::UseUniform);
  }

  // Output tensors
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  const auto& present_state = shader.AddOutput("present_state", ShaderUsage::UseUniform);

  // Each workgroup handles one (batch, head) pair
  // Within the workgroup, we compute the state update and output
  shader.MainFunctionBody() << R"SHADER(
  let batch_idx = workgroup_id.x;
  let head_idx = workgroup_id.y;
  let local_k = local_id.x;
  let local_v = local_id.y;

  // Bounds check
  if (batch_idx >= uniforms.batch_size || head_idx >= uniforms.num_heads) {
    return;
  }

  let head_dim_k = uniforms.head_dim_k;
  let head_dim_v = uniforms.head_dim_v;
  // Cast scale factor to element type to match tensor data type
  let scale_factor = query_element_t(select(1.0 / sqrt(f32(head_dim_k)), uniforms.scale, uniforms.scale != 0.0));

  // Compute base offsets
  let qkv_base = (batch_idx * uniforms.num_heads + head_idx) * head_dim_k;
  let v_base = (batch_idx * uniforms.num_heads + head_idx) * head_dim_v;
  let state_base = (batch_idx * uniforms.num_heads + head_idx) * head_dim_k * head_dim_v;

  // Process state update for this (k, v) element
  if (local_k < head_dim_k && local_v < head_dim_v) {
    let state_idx = state_base + local_k * head_dim_v + local_v;

    // Load current state value
)SHADER";

  shader.MainFunctionBody() << "    var state_val = " << past_state.GetByOffset("state_idx") << ";\n";

  // Load k and v values
  shader.MainFunctionBody() << "    let k_val = " << key.GetByOffset("qkv_base + local_k") << ";\n";
  shader.MainFunctionBody() << "    let v_val = " << value.GetByOffset("v_base + local_v") << ";\n";

  // Apply decay if needed (gated or gated_delta)
  if (update_rule_ == LinearAttentionUpdateRule::Gated || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    shader.MainFunctionBody() << "    // Load decay and compute exp(decay) - decay is in log space\n";
    shader.MainFunctionBody() << "    let decay_val = " << decay_ptr->GetByOffset("qkv_base + local_k") << ";\n";
    shader.MainFunctionBody() << "    let exp_decay = exp(decay_val);\n";
    shader.MainFunctionBody() << "    state_val = state_val * exp_decay;\n";
  }

  // Compute the update delta based on update rule
  if (update_rule_ == LinearAttentionUpdateRule::Linear) {
    shader.MainFunctionBody() << R"SHADER(
    // Linear update: S += k ⊗ v
    let update = k_val * v_val;
    state_val = state_val + update;
)SHADER";
  } else if (update_rule_ == LinearAttentionUpdateRule::Gated) {
    shader.MainFunctionBody() << R"SHADER(
    // Gated update: S = exp(g) * S + k ⊗ v (decay already applied above)
    let update = k_val * v_val;
    state_val = state_val + update;
)SHADER";
  } else if (update_rule_ == LinearAttentionUpdateRule::Delta) {
    // Delta update requires computing retrieved = S^T @ k
    shader.MainFunctionBody() << "    // Delta update: S += β * k ⊗ (v - S^T k)\n";
    shader.MainFunctionBody() << "    var retrieved = " << past_state.GetByOffset("state_base + 0u * head_dim_v + local_v")
                              << " * " << key.GetByOffset("qkv_base + 0u") << ";\n";
    shader.MainFunctionBody() << "    for (var k_i: u32 = 1u; k_i < head_dim_k; k_i = k_i + 1u) {\n";
    shader.MainFunctionBody() << "      let s_idx = state_base + k_i * head_dim_v + local_v;\n";
    shader.MainFunctionBody() << "      retrieved = retrieved + " << past_state.GetByOffset("s_idx")
                              << " * " << key.GetByOffset("qkv_base + k_i") << ";\n";
    shader.MainFunctionBody() << "    }\n";
    shader.MainFunctionBody() << "    let beta_val = " << beta_ptr->GetByOffset("(batch_idx * uniforms.num_heads + head_idx)") << ";\n";
    shader.MainFunctionBody() << "    let delta = beta_val * (v_val - retrieved);\n";
    shader.MainFunctionBody() << "    let update = k_val * delta;\n";
    shader.MainFunctionBody() << "    state_val = state_val + update;\n";
  } else {  // GatedDelta
    // Gated Delta update
    shader.MainFunctionBody() << "    // Gated Delta update: S = exp(g) * S + β * k ⊗ (v - exp(g) * S^T k)\n";
    shader.MainFunctionBody() << "    var retrieved = " << past_state.GetByOffset("state_base + 0u * head_dim_v + local_v")
                              << " * exp(" << decay_ptr->GetByOffset("qkv_base + 0u") << ")"
                              << " * " << key.GetByOffset("qkv_base + 0u") << ";\n";
    shader.MainFunctionBody() << "    for (var k_i: u32 = 1u; k_i < head_dim_k; k_i = k_i + 1u) {\n";
    shader.MainFunctionBody() << "      let s_idx = state_base + k_i * head_dim_v + local_v;\n";
    shader.MainFunctionBody() << "      let decay_k = " << decay_ptr->GetByOffset("qkv_base + k_i") << ";\n";
    shader.MainFunctionBody() << "      retrieved = retrieved + " << past_state.GetByOffset("s_idx")
                              << " * exp(decay_k) * " << key.GetByOffset("qkv_base + k_i") << ";\n";
    shader.MainFunctionBody() << "    }\n";
    shader.MainFunctionBody() << "    let beta_val = " << beta_ptr->GetByOffset("(batch_idx * uniforms.num_heads + head_idx)") << ";\n";
    shader.MainFunctionBody() << "    let delta = beta_val * (v_val - retrieved);\n";
    shader.MainFunctionBody() << "    let update = k_val * delta;\n";
    shader.MainFunctionBody() << "    state_val = state_val + update;\n";
  }

  // Write updated state and compute output
  shader.MainFunctionBody() << "    // Write updated state\n";
  shader.MainFunctionBody() << "    " << present_state.SetByOffset("state_idx", "state_val") << "\n";
  shader.MainFunctionBody() << "  }\n";

  shader.MainFunctionBody() << R"SHADER(
  // Synchronize before computing output
  workgroupBarrier();

  // Compute output: o = scale * q^T @ S
  // Each thread computes one element of the output
  if (local_k == 0u && local_v < head_dim_v) {
)SHADER";

  shader.MainFunctionBody() << "    var out_val = " << query.GetByOffset("qkv_base + 0u")
                            << " * " << present_state.GetByOffset("state_base + 0u * head_dim_v + local_v") << ";\n";
  shader.MainFunctionBody() << "    for (var k_i: u32 = 1u; k_i < head_dim_k; k_i = k_i + 1u) {\n";
  shader.MainFunctionBody() << "      let q_val = " << query.GetByOffset("qkv_base + k_i") << ";\n";
  shader.MainFunctionBody() << "      let s_idx = state_base + k_i * head_dim_v + local_v;\n";
  shader.MainFunctionBody() << "      out_val = out_val + q_val * " << present_state.GetByOffset("s_idx") << ";\n";
  shader.MainFunctionBody() << "    }\n";
  shader.MainFunctionBody() << "    " << output.SetByOffset("v_base + local_v", "out_val * scale_factor") << "\n";
  shader.MainFunctionBody() << "  }\n";

  return Status::OK();
}

Status LinearAttentionRecurrent::ComputeInternal(ComputeContext& context) const {
  const auto* query = context.Input<Tensor>(0);
  const auto* key = context.Input<Tensor>(1);
  const auto* value = context.Input<Tensor>(2);
  const auto* past_state = context.Input<Tensor>(3);
  const auto* decay = context.Input<Tensor>(4);  // Optional
  const auto* beta = context.Input<Tensor>(5);   // Optional

  const auto& query_shape = query->Shape();
  ORT_ENFORCE(query_shape.NumDimensions() == 4, "Query must be 4D: (B, H, 1, d_k)");

  const auto batch_size = static_cast<uint32_t>(query_shape[0]);
  const auto num_heads = static_cast<uint32_t>(query_shape[1]);
  const auto head_dim_k = static_cast<uint32_t>(query_shape[3]);
  const auto head_dim_v = static_cast<uint32_t>(value->Shape()[3]);

  // Validate decay and beta based on update rule
  bool has_decay = (decay != nullptr);
  bool has_beta = (beta != nullptr);

  if (update_rule_ == LinearAttentionUpdateRule::Gated || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    ORT_ENFORCE(has_decay, "Decay input is required for gated and gated_delta update rules");
  }
  if (update_rule_ == LinearAttentionUpdateRule::Delta || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    ORT_ENFORCE(has_beta, "Beta input is required for delta and gated_delta update rules");
  }

  // Create output tensors
  TensorShape output_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads), 1, static_cast<int64_t>(head_dim_v)});
  auto* output = context.Output(0, output_shape);
  auto* present_state = context.Output(1, past_state->Shape());

  // Setup and run the program
  LinearAttentionRecurrentProgram program{update_rule_, has_decay, has_beta};

  program.AddInputs({{query, ProgramTensorMetadataDependency::TypeAndRank},
                     {key, ProgramTensorMetadataDependency::TypeAndRank},
                     {value, ProgramTensorMetadataDependency::TypeAndRank},
                     {past_state, ProgramTensorMetadataDependency::TypeAndRank}});

  if (has_decay) {
    program.AddInput({decay, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (has_beta) {
    program.AddInput({beta, ProgramTensorMetadataDependency::TypeAndRank});
  }

  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank},
                      {present_state, ProgramTensorMetadataDependency::TypeAndRank}});

  // Dispatch: one workgroup per (batch, head), with threads for (k, v) elements
  // Use a fixed workgroup size that can cover typical head dimensions
  const uint32_t workgroup_size_k = std::min(head_dim_k, 16u);
  const uint32_t workgroup_size_v = std::min(head_dim_v, 16u);

  program.SetDispatchGroupSize(batch_size, num_heads, 1)
      .SetWorkgroupSize(workgroup_size_k, workgroup_size_v, 1)
      .AddUniformVariables({{batch_size},
                            {num_heads},
                            {head_dim_k},
                            {head_dim_v},
                            {scale_}});

  return context.RunProgram(program);
}

// =============================================================================
// LinearAttentionChunkParallel Implementation
// =============================================================================

ONNX_OPERATOR_KERNEL_EX(
    LinearAttentionChunkParallel,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    LinearAttentionChunkParallel);

LinearAttentionChunkParallel::LinearAttentionChunkParallel(const OpKernelInfo& info)
    : LinearAttentionRecurrent(info) {
  chunk_size_ = info.GetAttrOrDefault<int64_t>("chunk_size", 64);
}

Status LinearAttentionChunkIntraProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Inputs - referenced by name in WGSL shader
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
  const auto& value = shader.AddInput("value", ShaderUsage::UseUniform);

  std::string decay_name;
  if (has_decay_) {
    shader.AddInput("decay", ShaderUsage::UseUniform);
    decay_name = "decay";
  }
  if (has_beta_) {
    shader.AddInput("beta", ShaderUsage::UseUniform);
  }

  // Outputs
  const auto& intra_output = shader.AddOutput("intra_output", ShaderUsage::UseUniform);
  const auto& chunk_states = shader.AddOutput("chunk_states", ShaderUsage::UseUniform);

  // Compute intra-chunk causal attention
  // For each position i in chunk, compute output using positions 0..i
  shader.MainFunctionBody() << R"SHADER(
  let batch_idx = workgroup_id.x;
  let head_idx = workgroup_id.y;
  let chunk_idx = workgroup_id.z;

  if (batch_idx >= uniforms.batch_size || head_idx >= uniforms.num_heads || chunk_idx >= uniforms.num_chunks) {
    return;
  }

  let head_dim_k = uniforms.head_dim_k;
  let head_dim_v = uniforms.head_dim_v;
  let chunk_size = uniforms.chunk_size;
  let seq_len = uniforms.sequence_length;
  let scale_factor = query_element_t(select(1.0 / sqrt(f32(head_dim_k)), uniforms.scale, uniforms.scale != 0.0));

  // Chunk boundaries
  let chunk_start = chunk_idx * chunk_size;
  let chunk_end = min(chunk_start + chunk_size, seq_len);
  let actual_chunk_size = chunk_end - chunk_start;

  // Base offsets
  let bh_offset = batch_idx * uniforms.num_heads + head_idx;

  // Local thread handles one position in the chunk
  let local_pos = local_id.x;

  if (local_pos < actual_chunk_size) {
    let global_pos = chunk_start + local_pos;

    // Initialize local state for causal computation within chunk
    // We need to accumulate state from positions 0..local_pos
    let q_base = (bh_offset * seq_len + global_pos) * head_dim_k;
    let out_base = (bh_offset * seq_len + global_pos) * head_dim_v;

    // Compute output for this position using causal mask within chunk
    for (var v_i: u32 = 0u; v_i < head_dim_v; v_i = v_i + 1u) {
      var out_val: query_element_t = query_element_t(0.0);

      // Accumulate contributions from positions 0 to local_pos (inclusive)
      for (var src_pos: u32 = 0u; src_pos <= local_pos; src_pos = src_pos + 1u) {
        let src_global = chunk_start + src_pos;
        let k_base = (bh_offset * seq_len + src_global) * head_dim_k;
        let v_base = (bh_offset * seq_len + src_global) * head_dim_v;

        // Compute q @ k^T for this position pair
        var qk_dot: query_element_t = query_element_t(0.0);
        for (var k_i: u32 = 0u; k_i < head_dim_k; k_i = k_i + 1u) {
          qk_dot = qk_dot + )SHADER" << query.GetByOffset("q_base + k_i") << " * " << key.GetByOffset("k_base + k_i") << R"SHADER(;
        }

        // For linear attention variants, we need to apply the appropriate weighting
        let v_val = )SHADER" << value.GetByOffset("v_base + v_i") << R"SHADER(;
)SHADER";

  // Apply decay-based weighting if needed
  if (has_decay_) {
    shader.MainFunctionBody() << R"SHADER(
        // Compute cumulative decay from src_pos to local_pos
        var cum_decay: query_element_t = query_element_t(0.0);
        for (var d_pos: u32 = src_pos + 1u; d_pos <= local_pos; d_pos = d_pos + 1u) {
          let d_global = chunk_start + d_pos;
          // Average decay across k dimensions for simplicity
          var avg_decay: query_element_t = query_element_t(0.0);
          for (var k_i: u32 = 0u; k_i < head_dim_k; k_i = k_i + 1u) {
            avg_decay = avg_decay + decay[(bh_offset * seq_len + d_global) * head_dim_k + k_i];
          }
          cum_decay = cum_decay + avg_decay / query_element_t(head_dim_k);
        }
        let decay_weight = exp(cum_decay);
        out_val = out_val + qk_dot * v_val * decay_weight;
)SHADER";
  } else {
    shader.MainFunctionBody() << R"SHADER(
        out_val = out_val + qk_dot * v_val;
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
      }

      )SHADER" << intra_output.SetByOffset("out_base + v_i", "out_val * scale_factor") << R"SHADER(;
    }
  }

  // Compute accumulated state at the end of this chunk
  // Each thread contributes to building the chunk-end state
  workgroupBarrier();

  // Compute chunk-end state: accumulate k ⊗ v for all positions in chunk
  let state_base = (bh_offset * uniforms.num_chunks + chunk_idx) * head_dim_k * head_dim_v;

  for (var k_i: u32 = local_id.x; k_i < head_dim_k; k_i = k_i + 64u) {
    for (var v_i: u32 = 0u; v_i < head_dim_v; v_i = v_i + 1u) {
      var state_val: query_element_t = query_element_t(0.0);

      for (var pos: u32 = 0u; pos < actual_chunk_size; pos = pos + 1u) {
        let global_pos = chunk_start + pos;
        let k_base = (bh_offset * seq_len + global_pos) * head_dim_k;
        let v_base = (bh_offset * seq_len + global_pos) * head_dim_v;

        let k_val = )SHADER" << key.GetByOffset("k_base + k_i") << R"SHADER(;
        let v_val = )SHADER" << value.GetByOffset("v_base + v_i") << R"SHADER(;
)SHADER";

  if (has_decay_) {
    shader.MainFunctionBody() << R"SHADER(
        // Decay from this position to chunk end
        var decay_to_end: query_element_t = query_element_t(0.0);
        for (var d_pos: u32 = pos + 1u; d_pos < actual_chunk_size; d_pos = d_pos + 1u) {
          let d_global = chunk_start + d_pos;
          decay_to_end = decay_to_end + decay[(bh_offset * seq_len + d_global) * head_dim_k + k_i];
        }
        state_val = state_val + k_val * v_val * exp(decay_to_end);
)SHADER";
  } else {
    shader.MainFunctionBody() << R"SHADER(
        state_val = state_val + k_val * v_val;
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
      }

      let state_idx = state_base + k_i * head_dim_v + v_i;
      )SHADER" << chunk_states.SetByOffset("state_idx", "state_val") << R"SHADER(;
    }
  }
)SHADER";

  return Status::OK();
}

Status LinearAttentionChunkInterProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Inputs - referenced by name in WGSL shader
  shader.AddInput("intra_output", ShaderUsage::UseUniform);
  shader.AddInput("chunk_states", ShaderUsage::UseUniform);
  shader.AddInput("query", ShaderUsage::UseUniform);

  if (has_initial_state_) {
    shader.AddInput("initial_state", ShaderUsage::UseUniform);
  }
  if (has_decay_) {
    shader.AddInput("decay", ShaderUsage::UseUniform);
  }

  // Outputs - referenced by name in WGSL shader
  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("final_state", ShaderUsage::UseUniform);

  // Propagate state between chunks and compute final output
  shader.MainFunctionBody() << R"SHADER(
  let batch_idx = workgroup_id.x;
  let head_idx = workgroup_id.y;

  if (batch_idx >= uniforms.batch_size || head_idx >= uniforms.num_heads) {
    return;
  }

  let head_dim_k = uniforms.head_dim_k;
  let head_dim_v = uniforms.head_dim_v;
  let chunk_size = uniforms.chunk_size;
  let num_chunks = uniforms.num_chunks;
  let seq_len = uniforms.sequence_length;
  let scale = select(1.0 / sqrt(f32(head_dim_k)), uniforms.scale, uniforms.scale != 0.0);

  let bh_offset = batch_idx * uniforms.num_heads + head_idx;

  // Process each sequence position
  let pos = local_id.x;
  if (pos < seq_len) {
    let chunk_idx = pos / chunk_size;
    let q_base = (bh_offset * seq_len + pos) * head_dim_k;
    let out_base = (bh_offset * seq_len + pos) * head_dim_v;

    // Start with intra-chunk output
    for (var v_i: u32 = 0u; v_i < head_dim_v; v_i = v_i + 1u) {
      var out_val = intra_output[out_base + v_i];

      // Add contribution from previous chunks' accumulated state
      // This is q^T @ (sum of states from chunks 0 to chunk_idx-1)
      for (var prev_chunk: u32 = 0u; prev_chunk < chunk_idx; prev_chunk = prev_chunk + 1u) {
        let state_base = (bh_offset * num_chunks + prev_chunk) * head_dim_k * head_dim_v;

        for (var k_i: u32 = 0u; k_i < head_dim_k; k_i = k_i + 1u) {
          let q_val = query[q_base + k_i];
          let state_val = chunk_states[state_base + k_i * head_dim_v + v_i];
)SHADER";

  if (has_decay_) {
    shader.MainFunctionBody() << R"SHADER(
          // Compute cumulative decay from end of prev_chunk to current position
          var cum_decay: f32 = 0.0;
          let prev_chunk_end = (prev_chunk + 1u) * chunk_size;
          for (var d_pos: u32 = prev_chunk_end; d_pos <= pos; d_pos = d_pos + 1u) {
            cum_decay = cum_decay + decay[(bh_offset * seq_len + d_pos) * head_dim_k + k_i];
          }
          out_val = out_val + q_val * state_val * exp(cum_decay) * scale;
)SHADER";
  } else {
    shader.MainFunctionBody() << R"SHADER(
          out_val = out_val + q_val * state_val * scale;
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
        }
      }
)SHADER";

  if (has_initial_state_) {
    shader.MainFunctionBody() << R"SHADER(
      // Add contribution from initial state
      let init_state_base = bh_offset * head_dim_k * head_dim_v;
      for (var k_i: u32 = 0u; k_i < head_dim_k; k_i = k_i + 1u) {
        let q_val = query[q_base + k_i];
        let state_val = initial_state[init_state_base + k_i * head_dim_v + v_i];
)SHADER";
    if (has_decay_) {
      shader.MainFunctionBody() << R"SHADER(
        // Decay from start to current position
        var cum_decay: f32 = 0.0;
        for (var d_pos: u32 = 0u; d_pos <= pos; d_pos = d_pos + 1u) {
          cum_decay = cum_decay + decay[(bh_offset * seq_len + d_pos) * head_dim_k + k_i];
        }
        out_val = out_val + q_val * state_val * exp(cum_decay) * scale;
)SHADER";
    } else {
      shader.MainFunctionBody() << R"SHADER(
        out_val = out_val + q_val * state_val * scale;
)SHADER";
    }
    shader.MainFunctionBody() << R"SHADER(
      }
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
      output[out_base + v_i] = out_val;
    }
  }

  // Compute final state: sum all chunk states with appropriate decay
  workgroupBarrier();

  let final_state_base = bh_offset * head_dim_k * head_dim_v;
  for (var idx: u32 = local_id.x; idx < head_dim_k * head_dim_v; idx = idx + 256u) {
    let k_i = idx / head_dim_v;
    let v_i = idx % head_dim_v;

    var state_val: f32 = 0.0;
)SHADER";

  if (has_initial_state_) {
    shader.MainFunctionBody() << R"SHADER(
    // Start with initial state
    let init_state_base = bh_offset * head_dim_k * head_dim_v;
    state_val = initial_state[init_state_base + idx];
)SHADER";
    if (has_decay_) {
      shader.MainFunctionBody() << R"SHADER(
    // Decay initial state through entire sequence
    var total_decay: f32 = 0.0;
    for (var d_pos: u32 = 0u; d_pos < seq_len; d_pos = d_pos + 1u) {
      total_decay = total_decay + decay[(bh_offset * seq_len + d_pos) * head_dim_k + k_i];
    }
    state_val = state_val * exp(total_decay);
)SHADER";
    }
  }

  shader.MainFunctionBody() << R"SHADER(
    // Accumulate all chunk states
    for (var c: u32 = 0u; c < num_chunks; c = c + 1u) {
      let chunk_state_base = (bh_offset * num_chunks + c) * head_dim_k * head_dim_v;
      var chunk_val = chunk_states[chunk_state_base + idx];
)SHADER";

  if (has_decay_) {
    shader.MainFunctionBody() << R"SHADER(
      // Decay this chunk's state to end of sequence
      let chunk_end = min((c + 1u) * chunk_size, seq_len);
      var decay_to_end: f32 = 0.0;
      for (var d_pos: u32 = chunk_end; d_pos < seq_len; d_pos = d_pos + 1u) {
        decay_to_end = decay_to_end + decay[(bh_offset * seq_len + d_pos) * head_dim_k + k_i];
      }
      chunk_val = chunk_val * exp(decay_to_end);
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
      state_val = state_val + chunk_val;
    }

    final_state[final_state_base + idx] = state_val;
  }
)SHADER";

  return Status::OK();
}

Status LinearAttentionFullSequentialProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Full sequential computation for delta/gated_delta update rules.
  // These rules have state updates that depend on the current state (S^T k term),
  // making chunk-parallel decomposition incorrect.
  shader.AddInput("query", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
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

  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("final_state", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << R"SHADER(
  let batch_idx = workgroup_id.x;
  let head_idx = workgroup_id.y;

  if (batch_idx >= uniforms.batch_size || head_idx >= uniforms.num_heads) {
    return;
  }

  let dk = uniforms.head_dim_k;
  let dv = uniforms.head_dim_v;
  let seq_len = uniforms.sequence_length;
  let scale_val = query_element_t(select(1.0 / sqrt(f32(dk)), uniforms.scale, uniforms.scale != 0.0));
  let bh = batch_idx * uniforms.num_heads + head_idx;
  let state_size = dk * dv;

  // Initialize state array (supports up to 32x32 head dimensions)
  var state: array<query_element_t, 1024>;
  for (var i = 0u; i < state_size; i = i + 1u) {
    state[i] = query_element_t(0.0);
  }
)SHADER";

  if (has_initial_state_) {
    shader.MainFunctionBody() << R"SHADER(
  // Load initial state
  let init_base = bh * state_size;
  for (var i = 0u; i < state_size; i = i + 1u) {
    state[i] = query_element_t(initial_state[init_base + i]);
  }
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
  // Process each timestep sequentially
  for (var t = 0u; t < seq_len; t = t + 1u) {
    let qk_base = (bh * seq_len + t) * dk;
    let v_base = (bh * seq_len + t) * dv;
)SHADER";

  if (has_decay_) {
    shader.MainFunctionBody() << R"SHADER(
    // Apply decay: state *= exp(decay)
    for (var ki = 0u; ki < dk; ki = ki + 1u) {
      let exp_g = query_element_t(exp(decay[qk_base + ki]));
      for (var vi = 0u; vi < dv; vi = vi + 1u) {
        state[ki * dv + vi] = state[ki * dv + vi] * exp_g;
      }
    }
)SHADER";
  }

  shader.MainFunctionBody() << R"SHADER(
    // Delta update: S += beta * k \u2297 (v - S^T k)
    let beta_val = query_element_t(beta[bh * seq_len + t]);
    for (var vi = 0u; vi < dv; vi = vi + 1u) {
      // Compute retrieved = S^T @ k for this v dimension
      var retrieved = query_element_t(0.0);
      for (var ki = 0u; ki < dk; ki = ki + 1u) {
        retrieved = retrieved + state[ki * dv + vi] * query_element_t(key[qk_base + ki]);
      }
      let v_val = query_element_t(value[v_base + vi]);
      let delta_val = beta_val * (v_val - retrieved);

      for (var ki = 0u; ki < dk; ki = ki + 1u) {
        state[ki * dv + vi] = state[ki * dv + vi] + query_element_t(key[qk_base + ki]) * delta_val;
      }
    }

    // Compute output: o = scale * q^T @ state
    let out_base = (bh * seq_len + t) * dv;
    for (var vi = 0u; vi < dv; vi = vi + 1u) {
      var out_val = query_element_t(0.0);
      for (var ki = 0u; ki < dk; ki = ki + 1u) {
        out_val = out_val + query_element_t(query[qk_base + ki]) * state[ki * dv + vi];
      }
      output[out_base + vi] = out_val * scale_val;
    }
  }

  // Write final state
  let final_base = bh * state_size;
  for (var i = 0u; i < state_size; i = i + 1u) {
    final_state[final_base + i] = state[i];
  }
)SHADER";

  return Status::OK();
}

Status LinearAttentionChunkParallel::ComputeInternal(ComputeContext& context) const {
  const auto* query = context.Input<Tensor>(0);
  const auto* key = context.Input<Tensor>(1);
  const auto* value = context.Input<Tensor>(2);
  const auto* initial_state = context.Input<Tensor>(3);  // Optional
  const auto* decay = context.Input<Tensor>(4);           // Optional
  const auto* beta = context.Input<Tensor>(5);            // Optional

  const auto& query_shape = query->Shape();
  ORT_ENFORCE(query_shape.NumDimensions() == 4, "Query must be 4D: (B, H, L, d_k)");

  const auto batch_size = static_cast<uint32_t>(query_shape[0]);
  const auto num_heads = static_cast<uint32_t>(query_shape[1]);
  const auto seq_length = static_cast<uint32_t>(query_shape[2]);
  const auto head_dim_k = static_cast<uint32_t>(query_shape[3]);
  const auto head_dim_v = static_cast<uint32_t>(value->Shape()[3]);

  bool has_initial_state = (initial_state != nullptr);
  bool has_decay = (decay != nullptr);
  bool has_beta = (beta != nullptr);

  // Validate inputs based on update rule
  if (update_rule_ == LinearAttentionUpdateRule::Gated || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    ORT_ENFORCE(has_decay, "Decay input is required for gated and gated_delta update rules");
  }
  if (update_rule_ == LinearAttentionUpdateRule::Delta || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    ORT_ENFORCE(has_beta, "Beta input is required for delta and gated_delta update rules");
  }

  const uint32_t chunk_size = static_cast<uint32_t>(chunk_size_);
  const uint32_t num_chunks = (seq_length + chunk_size - 1) / chunk_size;

  // Create output tensors
  TensorShape output_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads),
                            static_cast<int64_t>(seq_length), static_cast<int64_t>(head_dim_v)});
  TensorShape state_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads),
                           static_cast<int64_t>(head_dim_k), static_cast<int64_t>(head_dim_v)});

  auto* output = context.Output(0, output_shape);
  auto* final_state = context.Output(1, state_shape);

  // For delta/gated_delta rules, use sequential computation.
  // Chunk-parallel decomposition doesn't work because state updates depend on the
  // running state through the S^T k term, making chunks non-independent.
  if (update_rule_ == LinearAttentionUpdateRule::Delta || update_rule_ == LinearAttentionUpdateRule::GatedDelta) {
    LinearAttentionFullSequentialProgram program{update_rule_, has_decay, has_beta, has_initial_state};

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

    program.SetDispatchGroupSize(batch_size, num_heads, 1)
        .SetWorkgroupSize(1, 1, 1)
        .AddUniformVariables({{batch_size},
                              {num_heads},
                              {seq_length},
                              {head_dim_k},
                              {head_dim_v},
                              {scale_}});

    return context.RunProgram(program);
  }

  // Linear/Gated rules: Use two-phase chunk-parallel approach
  // Allocate intermediate tensors for chunk computation
  TensorShape chunk_states_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads),
                                  static_cast<int64_t>(num_chunks), static_cast<int64_t>(head_dim_k),
                                  static_cast<int64_t>(head_dim_v)});

  // Allocate temporary tensors - need separate intra_output to avoid aliasing
  Tensor intra_output_tensor = context.CreateGPUTensor(query->DataType(), output_shape);
  Tensor chunk_states_tensor = context.CreateGPUTensor(query->DataType(), chunk_states_shape);

  // Step 1: Compute intra-chunk attention and per-chunk states
  {
    LinearAttentionChunkIntraProgram intra_program{update_rule_, has_decay, has_beta};

    intra_program.AddInputs({{query, ProgramTensorMetadataDependency::TypeAndRank},
                             {key, ProgramTensorMetadataDependency::TypeAndRank},
                             {value, ProgramTensorMetadataDependency::TypeAndRank}});

    if (has_decay) {
      intra_program.AddInput({decay, ProgramTensorMetadataDependency::TypeAndRank});
    }
    if (has_beta) {
      intra_program.AddInput({beta, ProgramTensorMetadataDependency::TypeAndRank});
    }

    intra_program.AddOutputs({{&intra_output_tensor, ProgramTensorMetadataDependency::TypeAndRank},
                              {&chunk_states_tensor, ProgramTensorMetadataDependency::TypeAndRank}});

    intra_program.SetDispatchGroupSize(batch_size, num_heads, num_chunks)
        .SetWorkgroupSize(64, 1, 1)
        .AddUniformVariables({{batch_size},
                              {num_heads},
                              {seq_length},
                              {head_dim_k},
                              {head_dim_v},
                              {chunk_size},
                              {num_chunks},
                              {scale_}});

    ORT_RETURN_IF_ERROR(context.RunProgram(intra_program));
  }

  // Step 2: Inter-chunk state propagation and final output computation
  {
    LinearAttentionChunkInterProgram inter_program{update_rule_, has_decay, has_beta, has_initial_state};

    // Use separate intra_output_tensor as input (read-only) and output (write-only) to avoid aliasing
    inter_program.AddInputs({{&intra_output_tensor, ProgramTensorMetadataDependency::TypeAndRank},  // intra_output
                             {&chunk_states_tensor, ProgramTensorMetadataDependency::TypeAndRank},  // chunk_states
                             {query, ProgramTensorMetadataDependency::TypeAndRank}});

    if (has_initial_state) {
      inter_program.AddInput({initial_state, ProgramTensorMetadataDependency::TypeAndRank});
    }
    if (has_decay) {
      inter_program.AddInput({decay, ProgramTensorMetadataDependency::TypeAndRank});
    }

    inter_program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank},
                              {final_state, ProgramTensorMetadataDependency::TypeAndRank}});

    inter_program.SetDispatchGroupSize(batch_size, num_heads, 1)
        .SetWorkgroupSize(256, 1, 1)
        .AddUniformVariables({{batch_size},
                              {num_heads},
                              {seq_length},
                              {head_dim_k},
                              {head_dim_v},
                              {chunk_size},
                              {num_chunks},
                              {scale_}});

    ORT_RETURN_IF_ERROR(context.RunProgram(inter_program));
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
