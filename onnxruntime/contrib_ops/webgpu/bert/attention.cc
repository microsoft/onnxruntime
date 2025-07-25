// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/attention.h"

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/multihead_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status TransferBSDToBNSHProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("qkv_input", ShaderUsage::UseUniform);
  const auto& qkv_output = shader.AddOutput("qkv_output", ShaderUsage::UseUniform | ShaderUsage::UseOffsetToIndices);

  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size")
                            << "let output_indices = " << qkv_output.OffsetToIndices("global_idx") << ";\n"
                            << "let input_offset_idx = output_indices[0] * uniforms.batch_offset + output_indices[1] *"
                            << " uniforms.head_offset + output_indices[2] * uniforms.sequence_offset + output_indices[3];\n";
  if (has_bias_) {
    shader.MainFunctionBody() << "let bias_offset_idx = (input_offset_idx % uniforms.sequence_offset) + uniforms.bias_offset;\n";
  }
  shader.MainFunctionBody() << "qkv_output[global_idx] = qkv_input[input_offset_idx]";
  if (has_bias_) {
    shader.MainFunctionBody() << " + bias[bias_offset_idx];\n";
  } else {
    shader.MainFunctionBody() << ";\n";
  }

  return Status::OK();
}

Status TransferBSDToBNSH(onnxruntime::webgpu::ComputeContext& context, int num_heads, int sequence_length,
                         int head_size, const Tensor* input_tensor, const Tensor* bias, int bias_offset, Tensor* output_tensor) {
  ORT_ENFORCE(input_tensor->Shape().GetDims().size() == 3);
  ORT_ENFORCE(output_tensor->Shape().GetDims().size() == 4);

  uint32_t data_size = SafeInt<uint32_t>(output_tensor->Shape().Size());
  const int batch_offset = num_heads * sequence_length * head_size;
  const int sequence_offset = num_heads * head_size;
  const int head_offset = head_size;
  bool has_bias = bias != nullptr;

  TransferBSDToBNSHProgram program{has_bias};
  program.AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{data_size},
                            {static_cast<uint32_t>(batch_offset)},
                            {static_cast<uint32_t>(sequence_offset)},
                            {static_cast<uint32_t>(head_offset)},
                            {static_cast<uint32_t>(bias_offset)}});

  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
  }

  return context.RunProgram(program);
};

void InitVarStub(std::ostringstream& ss, bool has_seqlen_k) {
  if (has_seqlen_k) {
    ss << "total_sequence_length = u32(seqlen_k[batch_idx]) + 1;\n";
    ss << "var past_sequence_length: u32 = select(total_sequence_length - sequence_length, 0u, uniforms.is_first_prompt > 0);\n";
  } else {
    ss << "let past_sequence_length = uniforms.past_sequence_length;\n";
  }
}

Status AttentionProbsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (feed_past_key_) {
    shader.AddInput("past_key", ShaderUsage::UseUniform);
  }
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  if (has_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (has_present_key_) {
    shader.AddOutput("present_key", ShaderUsage::UseUniform);
  }

  shader.AdditionalImplementation() << "var<workgroup> tileQ: array<q_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "var<workgroup> tileK: array<key_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n";

  shader.MainFunctionBody() << "// x holds the N and y holds the M\n"
                            << "let m = u32(workgroup_idx / uniforms.num_total_seq_length_tile) % uniforms.num_seq_length_tile  * TILE_SIZE;\n"
                            << "let n = (workgroup_idx % uniforms.num_total_seq_length_tile) * TILE_SIZE;\n"
                            << "let batch_head_idx = u32(workgroup_idx / (uniforms.num_total_seq_length_tile * uniforms.num_seq_length_tile));\n"
                            << "let batch_idx = batch_head_idx / uniforms.num_heads;\n"
                            << "let qOffset = batch_head_idx * uniforms.M * uniforms.K + m * uniforms.K;\n"
                            << "let sequence_length = uniforms.M;\n"
                            << "var total_sequence_length = uniforms.N;\n";
  std::ostringstream oss;
  InitVarStub(oss, has_seqlen_k_);
  shader.MainFunctionBody() << oss.str();
  shader.MainFunctionBody() << "let kOffset = (batch_head_idx / uniforms.n_reps) * uniforms.kv_sequence_length * uniforms.K;\n";
  if (has_present_key_) {
    shader.MainFunctionBody() << "let presentKeyOffset = (batch_head_idx / uniforms.n_reps) * uniforms.present_sequence_length * uniforms.K;\n";
  }

  shader.MainFunctionBody() << "var value = f32_val_t(0);\n"
                               "for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {\n"
                               "  if (m + local_id.y < uniforms.M && w + local_id.x < uniforms.K) {\n"
                               "    tileQ[TILE_SIZE * local_id.y + local_id.x] = q[qOffset + local_id.y * uniforms.K + w + local_id.x];\n"
                               "  }\n"
                               "  if (n + local_id.y < uniforms.N && w + local_id.x < uniforms.K) {\n"
                               "    var idx = TILE_SIZE * local_id.y + local_id.x;\n";

  if ((feed_past_key_ && has_present_key_) || (past_present_share_buffer_ && !is_first_prompt_)) {
    shader.MainFunctionBody() << "    if (n + local_id.y < past_sequence_length) {\n"
                              << "      let pastKeyOffset = (batch_head_idx / uniforms.n_reps) * uniforms.past_sequence_length * uniforms.K;\n"
                              << "      tileK[idx] = " << (past_present_share_buffer_ ? "present_key" : "past_key") << "[pastKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x];\n"
                              << "    } else  if (n + local_id.y - past_sequence_length < uniforms.kv_sequence_length) {\n"
                              << "      tileK[idx] = key[kOffset + (n + local_id.y - past_sequence_length) * uniforms.K + w + local_id.x];\n"
                              << "    }\n";
  } else {
    shader.MainFunctionBody() << "    if (n + local_id.y < uniforms.kv_sequence_length) {\n"
                                 "      tileK[idx] = key[kOffset + (n + local_id.y) * uniforms.K + w + local_id.x];\n"
                                 "    }\n";
  }

  if (has_present_key_) {
    if (past_present_share_buffer_) {
      shader.MainFunctionBody() << "    if (n + local_id.y >= past_sequence_length && n + local_id.y < uniforms.kv_sequence_length + past_sequence_length) {\n";
    } else {
      shader.MainFunctionBody() << "    if (n + local_id.y < uniforms.kv_sequence_length + past_sequence_length) {\n";
    }
    shader.MainFunctionBody() << "      present_key[presentKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x] = tileK[idx];\n"
                              << "    }\n";
  }

  shader.MainFunctionBody() << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "  for (var k: u32 = 0u; k < TILE_SIZE && w+k < uniforms.K; k++) {\n"
                            << "    value += f32_val_t(tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * local_id.x + k]);\n"
                            << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "}\n";

  shader.MainFunctionBody() << "if (m + local_id.y < uniforms.M && n + local_id.x < total_sequence_length) {\n"
                            << "  let headOffset = batch_head_idx * uniforms.M * uniforms.N;\n"
                            << "  let outputIdx = headOffset + (m + local_id.y) * uniforms.N + n + local_id.x;\n"
                            << "  var sum: f32 = " << (components_ == 4 ? "value.x + value.y + value.z + value.w" : (components_ == 2 ? "value.x + value.y" : "value")) << ";\n";

  shader.MainFunctionBody() << "  output[outputIdx] = output_value_t(sum * uniforms.alpha)";
  if (has_attention_bias_) {
    shader.MainFunctionBody() << " + attention_bias[outputIdx]";
  }
  shader.MainFunctionBody() << ";\n"
                            << "}\n";

  return Status::OK();
}

Status ComputeAttentionProbs(onnxruntime::webgpu::ComputeContext& context, int output_count, const Tensor* Q,
                             const Tensor* K, const Tensor* past_key, const Tensor* attention_bias, Tensor* probs, Tensor* present_key,
                             WebgpuAttentionParameters& parameters, int past_sequence_length, int total_sequence_length,
                             const Tensor* seqlen_k) {
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  const bool feed_past_key = present_key != nullptr && past_key != nullptr && past_key->SizeInBytes() > 0 && !parameters.past_present_share_buffer_;
  const bool has_present_key = output_count > 1 && past_key;
  const bool has_attention_bias = attention_bias != nullptr;
  constexpr int tile_size = 12;
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);

  AttentionProbsProgram program{"AttentionProbs", feed_past_key, has_present_key, has_attention_bias, tile_size,
                                components, parameters.is_first_prompt_, seqlen_k != nullptr, parameters.past_present_share_buffer_};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {K, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (feed_past_key) {
    program.AddInput({past_key, ProgramTensorMetadataDependency::TypeAndRank, components});
  }
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (seqlen_k != nullptr) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{probs, ProgramTensorMetadataDependency::Rank}});
  if (has_present_key) {
    program.AddOutput({present_key, ProgramTensorMetadataDependency::Rank, components});
  }

  const uint32_t vectorized_head_size = (parameters.head_size_ + components - 1) / components;
  const uint32_t num_total_seq_length_tile = (total_sequence_length + tile_size - 1) / tile_size;
  const uint32_t num_seq_length_tile = (parameters.sequence_length_ + tile_size - 1) / tile_size;
  program.SetDispatchGroupSize(parameters.batch_size_ * parameters.num_heads_ * num_seq_length_tile * num_total_seq_length_tile)
      .SetWorkgroupSize(tile_size, tile_size)
      .CacheHint(std::to_string(tile_size), parameters.past_present_share_buffer_, feed_past_key, has_present_key, has_attention_bias, seqlen_k != nullptr, components, parameters.is_first_prompt_)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<uint32_t>(total_sequence_length)},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.head_size_)},
                            {static_cast<float>(alpha)},
                            {static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(seqlen_k == nullptr ? total_sequence_length : parameters.seqlen_present_kv_cache_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {static_cast<uint32_t>(parameters.is_first_prompt_ ? 1 : 0)},
                            {num_total_seq_length_tile},
                            {num_seq_length_tile}})
      .SetOverridableConstants({{static_cast<uint32_t>(tile_size)}});

  return context.RunProgram(program);
}

Status InPlaceSoftmaxProgram::GenerateShaderCode(ShaderHelper& shader) const {
  bool has_sliding_window = local_window_size_ != -1;

  if (has_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::UseUniform);
  }
  if (has_head_sink_) {
    shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  shader.AddOutput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation() << "var<workgroup> thread_max: array<f32, " << work_group_size_ << ">;\n"
                                    << "var<workgroup> thread_sum: array<f32, " << work_group_size_ << ">;\n"
                                    << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n";
  shader.MainFunctionBody() << "let sequence_length = uniforms.sequence_length;\n"
                            << "let batch_idx = u32(workgroup_idx / sequence_length) / uniforms.num_heads;\n"
                            << "let head_idx = u32(workgroup_idx / sequence_length) % uniforms.num_heads;\n"
                            << "var total_sequence_length = uniforms.total_sequence_length_comp * " << components_ << ";\n";
  std::ostringstream oss;
  InitVarStub(oss, has_seqlen_k_);
  shader.MainFunctionBody() << oss.str()
                            << "let seq_causal_length = " << (has_seqlen_k_ ? "past_sequence_length + workgroup_idx % sequence_length + 1" : "uniforms.total_sequence_length_comp") << ";\n"
                            << "let local_offset = local_idx * uniforms.elements_per_thread;\n"
                            << "let offset = workgroup_idx * uniforms.total_sequence_length_comp + local_offset;\n";
  if (has_sliding_window) {
    // Sliding window
    shader.MainFunctionBody()
        << "let should_apply_local_window = uniforms.local_window_size >= 0 && seq_causal_length > uniforms.local_window_size + 1;\n"
        << "let start_offset = select(0, seq_causal_length - uniforms.local_window_size, should_apply_local_window);\n"
        << "let effective_seq_length = select(seq_causal_length, uniforms.local_window_size, should_apply_local_window);\n";
  } else {
    // No sliding window: we keep the code for sliding window in the shader but
    // using const for start_offset and should_apply_local_window will make the compiler optimize it out.
    shader.MainFunctionBody()
        << "const start_offset = 0;\n"
        << "const should_apply_local_window = false;\n"
        << "let effective_seq_length = seq_causal_length;\n";
  }
  shader.MainFunctionBody()
      << "var thread_max_vector = f32_val_t(-3.402823e+38f);\n"
      << "for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < effective_seq_length; i++) {\n"
      << "  let actual_pos = local_offset + i + start_offset;\n"
      << "  if (!should_apply_local_window || actual_pos < seq_causal_length) {\n"
      << "      thread_max_vector = max(f32_val_t(x[offset + i + start_offset]), thread_max_vector);\n"
      << "  }\n"
      << "}\n"
      << "thread_max[local_idx] = " << (components_ == 4 ? "max(max(thread_max_vector.x, thread_max_vector.y), max(thread_max_vector.z, thread_max_vector.w))" : (components_ == 2 ? "max(thread_max_vector.x, thread_max_vector.y)" : "thread_max_vector")) << ";\n"
      << "workgroupBarrier();\n";

  if (has_head_sink_) {
    // Handle head sink
    shader.MainFunctionBody() << "let sink_value: f32 = head_sink[head_idx];\n"
                              << "var max_value = sink_value;\n";
  } else if (use_smooth_softmax_) {
    shader.MainFunctionBody() << "var max_value: f32 = 0.0;\n";
  } else {
    shader.MainFunctionBody() << "var max_value = f32(-3.402823e+38f);\n";
  }

  shader.MainFunctionBody() << "for (var i = 0u; i < " << work_group_size_ << "; i++) {\n"
                            << "  max_value = max(thread_max[i], max_value);\n"
                            << "}\n"
                            << "var sum_vector = f32_val_t(0);\n"
                            << "for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < effective_seq_length; i++) {\n"
                            << "  let actual_pos = local_offset + i + start_offset;\n"
                            << "  if (!should_apply_local_window || actual_pos < seq_causal_length) {\n"
                            << "     sum_vector += exp(f32_val_t(x[offset + i + start_offset]) - max_value);\n"
                            << "  }\n"
                            << "}\n"
                            << "thread_sum[local_idx] = " << (components_ == 4 ? "sum_vector.x + sum_vector.y + sum_vector.z + sum_vector.w" : (components_ == 2 ? "sum_vector.x + sum_vector.y" : "sum_vector")) << ";\n"
                            << "workgroupBarrier();\n"
                            << "var sum: f32 = 0;\n"
                            << "for (var i = 0u; i < " << work_group_size_ << "; i++) {\n"
                            << "  sum += thread_sum[i]\n;"
                            << "}\n";

  if (has_head_sink_) {
    shader.MainFunctionBody() << "sum += exp(sink_value - max_value);\n";
  } else if (use_smooth_softmax_) {
    shader.MainFunctionBody() << "sum += exp(-max_value);\n";
  }

  shader.MainFunctionBody() << "if (sum == 0) {\n"
                            << "  for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < effective_seq_length; i++) {\n"
                            << "  let actual_pos = local_offset + i + start_offset;\n"
                            << "    if (actual_pos < seq_causal_length) {\n"
                            << "      x[offset + i + start_offset] = x_value_t(x_element_t(1.0)/x_element_t(effective_seq_length));\n"
                            << "    }\n"
                            << "  }\n"
                            << "} else {\n"
                            << "  for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < effective_seq_length; i++) {\n"
                            << "    let actual_pos = local_offset + i + start_offset;\n"
                            << "    let pos = offset + i + start_offset;\n"
                            << "    if (!should_apply_local_window || actual_pos < seq_causal_length) {\n"
                            << "       var f32input = f32_val_t(x[pos]);\n"
                            << "       x[pos] = x_value_t(exp(f32input - max_value) / sum);\n"
                            << "    }\n"
                            << "  }\n"
                            << "}\n";

  // zero out elements outsize the sliding window
  shader.MainFunctionBody() << "if (should_apply_local_window) {\n"
                            << "  for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {\n"
                            << "    let global_pos = i + local_offset;\n"
                            << "    if (global_pos < start_offset) {\n"
                            << "      x[offset + i] = x_value_t(x_element_t(0));\n"
                            << "    }\n"
                            << "  }\n"
                            << "}\n";

  if (has_seqlen_k_) {
    shader.MainFunctionBody() << "for (var total_seq_id: u32 = seq_causal_length; total_seq_id + local_offset < uniforms.total_sequence_length_comp; total_seq_id++) {\n"
                              << "   x[offset + total_seq_id] = x_value_t(x_element_t(0));\n"
                              << "}\n";
  }

  return Status::OK();
}

Status ComputeInPlaceSoftmax(onnxruntime::webgpu::ComputeContext& context, Tensor* probs, int32_t batch_size, int32_t num_heads, int32_t past_sequence_length, int32_t sequence_length, int32_t total_sequence_length,
                             const Tensor* seqlen_k, bool is_first_prompt, bool use_smooth_softmax, const Tensor* head_sink, int local_window_size) {
  const int components = seqlen_k != nullptr ? 1 : (total_sequence_length % 4 == 0 ? 4 : (total_sequence_length % 2 == 0 ? 2 : 1));
  int work_group_size = 64;
  const int total_sequence_length_comp = (total_sequence_length + components - 1) / components;
  if (total_sequence_length_comp < work_group_size) {
    work_group_size = 32;
  }
  const int elementsPerThread = (total_sequence_length_comp + work_group_size - 1) / work_group_size;

  InPlaceSoftmaxProgram program{work_group_size, components, use_smooth_softmax, seqlen_k != nullptr, head_sink != nullptr, local_window_size};
  if (seqlen_k != nullptr) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (head_sink != nullptr) {
    program.AddInput({head_sink, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutputs({{probs, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .CacheHint(work_group_size, use_smooth_softmax, local_window_size != -1)
      .SetDispatchGroupSize(batch_size * num_heads * sequence_length)
      .SetWorkgroupSize(work_group_size)
      .AddUniformVariables({{static_cast<uint32_t>(batch_size)},
                            {static_cast<uint32_t>(num_heads)},
                            {static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(sequence_length)},
                            {static_cast<uint32_t>(total_sequence_length_comp)},
                            {static_cast<uint32_t>(elementsPerThread)},
                            {static_cast<uint32_t>(is_first_prompt ? 1 : 0)},
                            {static_cast<uint32_t>(local_window_size)}});

  return context.RunProgram(program);
}

Status VxAttentionScoreProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("probs", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("v", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (feed_past_value_) {
    shader.AddInput("past_value", ShaderUsage::UseUniform);
  }
  if (seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (has_present_value_) {
    shader.AddOutput("present_value", ShaderUsage::UseUniform);
  }

  shader.AdditionalImplementation() << "var<workgroup> tileQ: array<probs_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "var<workgroup> tileK: array<v_value_t, " << tile_size_ * tile_size_ << ">;\n";
  shader.MainFunctionBody() << "let batch_head_idx = u32(workgroup_idx / (uniforms.num_head_size_tile * uniforms.num_seq_length_tile));\n"
                            << "let head_idx = batch_head_idx % uniforms.num_heads;\n"
                            << "let batch_idx = batch_head_idx / uniforms.num_heads;\n"
                            << "let m = (u32(workgroup_idx / uniforms.num_head_size_tile) % uniforms.num_seq_length_tile) * TILE_SIZE + local_id.y;\n"
                            << "let n = (workgroup_idx % uniforms.num_head_size_tile) * TILE_SIZE + local_id.x;\n"
                            << "let offsetA = batch_head_idx * (uniforms.M * uniforms.K) + m * uniforms.K;\n"
                            << "let sequence_length = uniforms.M;\n"
                            << "var total_sequence_length = uniforms.K;\n";
  std::ostringstream oss;
  InitVarStub(oss, seqlen_k_);
  shader.MainFunctionBody() << oss.str();
  shader.MainFunctionBody() << "let vOffset = (batch_head_idx / uniforms.n_reps) * uniforms.N * uniforms.kv_sequence_length + n;\n";
  if (has_present_value_) {
    shader.MainFunctionBody() << "let presentValueOffset = (batch_head_idx / uniforms.n_reps) * uniforms.N * uniforms.present_sequence_length + n;\n";
  }

  shader.MainFunctionBody() << "var value = output_value_t(0);\n"
                            << "for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {\n"
                            << "  if (m < uniforms.M && w + local_id.x < uniforms.K) {\n"
                            << "    tileQ[TILE_SIZE * local_id.y + local_id.x] = probs[offsetA + w + local_id.x];\n"
                            << "  }\n"
                            << "  if (n < uniforms.N && w + local_id.y < uniforms.K) {\n"
                            << "    var idx = TILE_SIZE * local_id.y + local_id.x;\n";

  if ((feed_past_value_ && has_present_value_) || (past_present_share_buffer_ && !is_first_prompt_)) {
    shader.MainFunctionBody() << "    if (w + local_id.y < past_sequence_length) {\n"
                              << "      let pastValueOffset = (batch_head_idx / uniforms.n_reps) * uniforms.N * uniforms.past_sequence_length + n;\n"
                              << "      tileK[idx] = " << (past_present_share_buffer_ ? "present_value" : "past_value") << "[pastValueOffset + (w + local_id.y) * uniforms.N];\n"
                              << "    } else if (w + local_id.y - past_sequence_length < uniforms.kv_sequence_length) {\n"
                              << "      tileK[idx] = v[vOffset + (w + local_id.y - past_sequence_length) * uniforms.N];\n"
                              << "    }\n";
  } else {
    shader.MainFunctionBody() << "    if (w + local_id.y < uniforms.kv_sequence_length) {\n"
                              << "      tileK[idx] = v[vOffset + (w + local_id.y) * uniforms.N];\n"
                              << "    }\n";
  }

  if (has_present_value_) {
    if (past_present_share_buffer_) {
      shader.MainFunctionBody() << "    if (w + local_id.y >= past_sequence_length && w + local_id.y < uniforms.kv_sequence_length + past_sequence_length) {\n";
    } else {
      shader.MainFunctionBody() << "    if (w + local_id.y < uniforms.kv_sequence_length + past_sequence_length) {\n";
    }
    shader.MainFunctionBody() << "      present_value[presentValueOffset + (w + local_id.y) * uniforms.N] = tileK[idx];\n"
                              << "    }\n";
  }

  shader.MainFunctionBody() << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "  for (var k: u32 = 0u; k < TILE_SIZE && w+k < total_sequence_length; k++) {\n"
                            << "    value += tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * k + local_id.x];\n"
                            << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "}\n";

  shader.MainFunctionBody() << "// we need to transpose output from BNSH_v to BSND_v\n"
                            << "if (m < uniforms.M && n < uniforms.N) {\n"
                            << "  let outputIdx = batch_idx * uniforms.M * uniforms.v_hidden_size + "
                            << "  m * uniforms.v_hidden_size + head_idx * uniforms.N + n;\n"
                            << "  output[outputIdx] = value;\n"
                            << "}\n";

  return Status::OK();
}

Status ComputeVxAttentionScore(onnxruntime::webgpu::ComputeContext& context, int output_count,
                               const Tensor* probs,
                               const Tensor* V,
                               const Tensor* past_value,
                               Tensor* output,
                               Tensor* present_value,
                               WebgpuAttentionParameters& parameters,
                               int past_sequence_length,
                               int total_sequence_length,
                               const Tensor* seqlen_k) {
  const bool feed_past_value = present_value != nullptr && past_value != nullptr && past_value->SizeInBytes() > 0 && !parameters.past_present_share_buffer_;
  const bool has_present_value = output_count > 1 && past_value != nullptr;
  const int components = parameters.v_head_size_ % 4 == 0 ? 4 : (parameters.v_head_size_ % 2 == 0 ? 2 : 1);
  constexpr int tile_size = 12;
  int tile_n_size = tile_size * components;
  VxAttentionScoreProgram program{"VxAttentionScore", feed_past_value, has_present_value, tile_size, parameters.is_first_prompt_, seqlen_k, parameters.past_present_share_buffer_};
  program.AddInputs({{probs, ProgramTensorMetadataDependency::TypeAndRank},
                     {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (feed_past_value) {
    program.AddInput({past_value, ProgramTensorMetadataDependency::TypeAndRank, components});
  }
  if (seqlen_k != nullptr) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (has_present_value) {
    program.AddOutput({present_value, ProgramTensorMetadataDependency::TypeAndRank, components});
  }

  const uint32_t num_head_size_tile = (parameters.v_head_size_ + tile_n_size - 1) / tile_n_size;
  const uint32_t num_seq_length_tile = (parameters.sequence_length_ + tile_size - 1) / tile_size;
  program.SetDispatchGroupSize(parameters.batch_size_ * parameters.num_heads_ * num_head_size_tile * num_seq_length_tile)
      .CacheHint(std::to_string(tile_size), parameters.past_present_share_buffer_, feed_past_value, has_present_value, seqlen_k != nullptr, parameters.is_first_prompt_)
      .SetWorkgroupSize(tile_size, tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(total_sequence_length)},
                            {static_cast<uint32_t>(parameters.v_head_size_ / components)},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.head_size_)},
                            {static_cast<uint32_t>(parameters.v_hidden_size_ * parameters.n_reps / components)},
                            {static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(seqlen_k == nullptr ? total_sequence_length : parameters.seqlen_present_kv_cache_)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {static_cast<uint32_t>(parameters.is_first_prompt_)},
                            {num_head_size_tile},
                            {num_seq_length_tile}})
      .SetOverridableConstants({{static_cast<uint32_t>(tile_size)}});

  return context.RunProgram(program);
}

Status ApplyAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                      const Tensor* past_key, const Tensor* past_value, Tensor* output, Tensor* present_key, Tensor* present_value,
                      WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context, const Tensor* head_sink,
                      const Tensor* seqlen_k, int local_window_size) {
  const int output_count = std::min({context.OutputCount(), 1 + (past_key != nullptr ? 1 : 0) + (past_value != nullptr ? 1 : 0)});
  const int past_sequence_length = output_count > 1 ? parameters.past_sequence_length_ : 0;
  const int total_sequence_length =
      parameters.is_gqa_ ? parameters.total_sequence_length_ : past_sequence_length + parameters.kv_sequence_length_;

  const TensorShapeVector probs_dims({parameters.batch_size_, parameters.num_heads_,
                                      parameters.sequence_length_, total_sequence_length});
  const TensorShape probs_shape(probs_dims);
  Tensor probs = context.CreateGPUTensor(Q->DataType(), probs_shape);
  ORT_RETURN_IF_ERROR(ComputeAttentionProbs(context, output_count, Q, K, past_key, attention_bias, &probs, present_key,
                                            parameters, past_sequence_length, total_sequence_length, seqlen_k));

  ORT_RETURN_IF_ERROR(ComputeInPlaceSoftmax(context, &probs,
                                            parameters.batch_size_, parameters.num_heads_, parameters.past_sequence_length_, parameters.sequence_length_, total_sequence_length, seqlen_k, parameters.is_first_prompt_, parameters.use_smooth_softmax_, head_sink, local_window_size));

  ORT_RETURN_IF_ERROR(ComputeVxAttentionScore(context, output_count, &probs, V, past_value, output, present_value,
                                              parameters, past_sequence_length, total_sequence_length, seqlen_k));

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
