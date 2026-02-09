// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/attention.h"

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/webgpu/bert/multihead_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/math/matmul.h"
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

  return WGSL_TEMPLATE_APPLY(shader, "bert/transfer_bsd_to_bnsh.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_VARIABLE(qkv_output, qkv_output));
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

Status SplitPackedQKVProgram::GenerateShaderCode(ShaderHelper& sh) const {
  // Inputs: packed_qkv [B, S, D], outputs: Q, K, V [B, S, D]
  const auto& packed_qkv = sh.AddInput("packed_qkv", ShaderUsage::UseOffsetToIndices | ShaderUsage::UseUniform);
  const auto& query = sh.AddOutput("query", ShaderUsage::UseSetByIndices | ShaderUsage::UseUniform);
  const auto& key = sh.AddOutput("key", ShaderUsage::UseSetByIndices | ShaderUsage::UseUniform);
  const auto& val = sh.AddOutput("val", ShaderUsage::UseSetByIndices | ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(sh, "bert/split_packed_qkv.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(packed_qkv, packed_qkv),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(val, val));
}

Status SplitPackedQKV(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& params,
                      const Tensor* packedQKV, Tensor* query, Tensor* key, Tensor* val, int kv_hidden_size) {
  // Output Q, K, V in BSD format
  const int components = std::min({GetMaxComponents(params.hidden_size_), GetMaxComponents(kv_hidden_size), GetMaxComponents(params.v_hidden_size_)});
  SplitPackedQKVProgram program;
  auto input_size = packedQKV->Shape().Size();
  const uint32_t vectorized_input_size = static_cast<uint32_t>(input_size / components);
  program
      .AddInput({packedQKV, ProgramTensorMetadataDependency::TypeAndRank, components})
      .AddOutputs({{query, ProgramTensorMetadataDependency::TypeAndRank, components}, {key, ProgramTensorMetadataDependency::TypeAndRank, components}, {val, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .AddUniformVariables({
          {vectorized_input_size},
          {static_cast<uint32_t>(params.hidden_size_ / components)},
          {static_cast<uint32_t>(kv_hidden_size / components)},
      })
      .SetDispatchGroupSize((vectorized_input_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
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

  return WGSL_TEMPLATE_APPLY(shader, "bert/attention_probs.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(components, components_),
                             WGSL_TEMPLATE_PARAMETER(feed_past_key, feed_past_key_),
                             WGSL_TEMPLATE_PARAMETER(has_present_key, has_present_key_),
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_seqlen_k, has_seqlen_k_),
                             WGSL_TEMPLATE_PARAMETER(past_present_share_buffer, past_present_share_buffer_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_param, tile_size_));
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
                                components, seqlen_k != nullptr, parameters.past_present_share_buffer_, parameters.is_unidirectional_};
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

  // Get attention bias dimensions for broadcasting
  uint32_t attn_bias_dim0 = 1;
  uint32_t attn_bias_dim1 = 1;
  if (has_attention_bias) {
    const auto& bias_shape = attention_bias->Shape();
    attn_bias_dim0 = static_cast<uint32_t>(bias_shape[0]);
    attn_bias_dim1 = static_cast<uint32_t>(bias_shape[1]);
  }

  program.SetDispatchGroupSize(parameters.batch_size_ * parameters.num_heads_ * num_seq_length_tile * num_total_seq_length_tile)
      .SetWorkgroupSize(tile_size, tile_size)
      .CacheHint(std::to_string(tile_size), parameters.past_present_share_buffer_, feed_past_key, has_present_key, has_attention_bias, seqlen_k != nullptr, components, parameters.is_first_prompt_, parameters.is_unidirectional_)
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
                            {num_seq_length_tile},
                            {attn_bias_dim0},
                            {attn_bias_dim1}})
      .SetOverridableConstants({{static_cast<uint32_t>(tile_size)}});

  return context.RunProgram(program);
}

Status InPlaceSoftmaxProgram::GenerateShaderCode(ShaderHelper& shader) const {
  if (has_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::UseUniform);
  }
  if (has_head_sink_) {
    shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  shader.AddOutput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "bert/inplace_softmax.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(components, components_),
                             WGSL_TEMPLATE_PARAMETER(work_group_size, work_group_size_),
                             WGSL_TEMPLATE_PARAMETER(use_smooth_softmax, use_smooth_softmax_),
                             WGSL_TEMPLATE_PARAMETER(has_seqlen_k, has_seqlen_k_),
                             WGSL_TEMPLATE_PARAMETER(has_head_sink, has_head_sink_),
                             WGSL_TEMPLATE_PARAMETER(has_sliding_window, local_window_size_ != -1));
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

  return WGSL_TEMPLATE_APPLY(shader, "bert/vx_attention_score.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(feed_past_value, feed_past_value_),
                             WGSL_TEMPLATE_PARAMETER(has_present_value, has_present_value_),
                             WGSL_TEMPLATE_PARAMETER(has_seqlen_k, seqlen_k_ != nullptr),
                             WGSL_TEMPLATE_PARAMETER(past_present_share_buffer, past_present_share_buffer_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_param, tile_size_));
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
  VxAttentionScoreProgram program{"VxAttentionScore", feed_past_value, has_present_value, tile_size, seqlen_k, parameters.past_present_share_buffer_};
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
                      Tensor* output_qk, WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context,
                      const Tensor* head_sink, const Tensor* seqlen_k, int local_window_size) {
  if (context.IsGraphCaptureEnabled()) {
    ORT_NOT_IMPLEMENTED("Graph capture not implemented for non flash attention path");
  }
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

  if (output_qk != nullptr) {
    // Copy the attention scores (scaled Q*K^T) to output_qk
    ORT_RETURN_IF_ERROR(context.CopyTensor(probs, *output_qk));
  }

  ORT_RETURN_IF_ERROR(ComputeInPlaceSoftmax(context, &probs,
                                            parameters.batch_size_, parameters.num_heads_, parameters.past_sequence_length_, parameters.sequence_length_, total_sequence_length, seqlen_k, parameters.is_first_prompt_, parameters.use_smooth_softmax_, head_sink, local_window_size));

  ORT_RETURN_IF_ERROR(ComputeVxAttentionScore(context, output_count, &probs, V, past_value, output, present_value,
                                              parameters, past_sequence_length, total_sequence_length, seqlen_k));

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    Attention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    Attention);

Attention::Attention(const OpKernelInfo& info)
    : WebGpuKernel(info),
      onnxruntime::contrib::AttentionBase(info, false) {
}

Status PrepareQKV(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                  const Tensor* input, const Tensor* weights, const Tensor* bias,
                  Tensor* q, Tensor* k, Tensor* v) {
  // Use MatMul to compute packed QKV output: input * weights + bias
  // Then use SplitPackedQKV to split into Q, K, V in BSD format
  // Returns Q, K, V in BSD format

  // Create packed QKV tensor with shape [batch_size, sequence_length, hidden_size + hidden_size + v_hidden_size]
  const int64_t packed_qkv_size = parameters.hidden_size_ + parameters.hidden_size_ + parameters.v_hidden_size_;
  TensorShapeVector packed_qkv_shape({parameters.batch_size_, parameters.sequence_length_, packed_qkv_size});
  Tensor packed_qkv = context.CreateGPUTensor(input->DataType(), TensorShape(packed_qkv_shape));

  // Prepare inputs for MatMul
  std::vector<const Tensor*> matmul_inputs = {input, weights, bias};

  // Call MatMul: packed_qkv = input * weights + bias
  ORT_RETURN_IF_ERROR(onnxruntime::webgpu::ComputeMatMul(&context, Activation(), matmul_inputs, &packed_qkv, true));

  // Output Q, K, V in BSD format
  return SplitPackedQKV(context, parameters, &packed_qkv, q, k, v, parameters.hidden_size_);
}

Status Attention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* input = context.Input(0);
  const Tensor* weights = context.Input(1);
  const Tensor* bias = context.Input(2);
  const Tensor* mask_index = context.Input(3);
  const Tensor* past = context.Input(4);
  const Tensor* attention_bias = context.Input(5);
  const Tensor* past_seq_len = context.Input(6);

  if (past) {
    ORT_NOT_IMPLEMENTED("past tensor not implemented for webgpu Attention");
  }
  if (mask_index) {
    ORT_NOT_IMPLEMENTED("mask_index tensor not implemented for webgpu Attention");
  }

  AttentionParameters params;
  // Use the second dimension from weight for bias to get q_hidden_size when bias is nullptr
  std::vector<int64_t> bias_dims{weights->Shape().GetDims()[1]};
  const TensorShape bias_shape{bias_dims};
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights->Shape(),
                                  bias != nullptr ? bias->Shape() : bias_shape,
                                  mask_index,
                                  past,
                                  attention_bias,
                                  &params,
                                  context.DeviceLimits().maxComputeInvocationsPerWorkgroup,
                                  past_seq_len));

  WebgpuAttentionParameters parameters(params);

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size_);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length_);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size_);
  Tensor* output = context.Output(0, output_shape);

  // If optional outputs aren't needed, present_key and present_value will be null
  std::vector<int64_t> present_dims{
      2,
      parameters.batch_size_,
      parameters.num_heads_,
      parameters.total_sequence_length_,
      parameters.head_size_,
  };
  TensorShape present_shape(present_dims);
  Tensor* present = context.Output(1, present_shape);
  if (present) {
    ORT_NOT_IMPLEMENTED("present tensor not implemented for webgpu Attention");
  }

  // Create Q, K, V tensors in BSD format from input * weights + bias
  TensorShapeVector qkv_bsd_shape({parameters.batch_size_, parameters.sequence_length_, parameters.hidden_size_});
  TensorShapeVector v_bsd_shape({parameters.batch_size_, parameters.sequence_length_, parameters.v_hidden_size_});
  Tensor Q_bsd = context.CreateGPUTensor(input->DataType(), TensorShape(qkv_bsd_shape));
  Tensor K_bsd = context.CreateGPUTensor(input->DataType(), TensorShape(qkv_bsd_shape));
  Tensor V_bsd = context.CreateGPUTensor(input->DataType(), TensorShape(v_bsd_shape));

  // Compute Q, K, V from input, weights, and bias (returns BSD format)
  ORT_RETURN_IF_ERROR(PrepareQKV(context, parameters, input, weights, bias, &Q_bsd, &K_bsd, &V_bsd));
  parameters.qkv_format_ = Q_K_V_BSNH;

  // Check if we can use flash attention
  if (CanApplyFlashAttention(nullptr, parameters, context)) {
    // FlashAttention supports Q_K_V_BSNH format directly
    return ApplyFlashAttention(&Q_bsd, &K_bsd, &V_bsd, attention_bias, output, nullptr, nullptr, nullptr, nullptr,
                               parameters, context, nullptr);
  }

  // For non-flash attention path, convert BSD to BNSH format
  TensorShapeVector qkv_bnsh_shape({parameters.batch_size_, parameters.num_heads_, parameters.sequence_length_, parameters.head_size_});
  TensorShapeVector v_bnsh_shape({parameters.batch_size_, parameters.num_heads_, parameters.sequence_length_, parameters.v_head_size_});
  Tensor Q = context.CreateGPUTensor(input->DataType(), TensorShape(qkv_bnsh_shape));
  Tensor K = context.CreateGPUTensor(input->DataType(), TensorShape(qkv_bnsh_shape));
  Tensor V = context.CreateGPUTensor(input->DataType(), TensorShape(v_bnsh_shape));

  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads_, parameters.sequence_length_,
                                        parameters.head_size_, &Q_bsd, nullptr, 0, &Q));
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads_, parameters.sequence_length_,
                                        parameters.head_size_, &K_bsd, nullptr, 0, &K));
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads_, parameters.sequence_length_,
                                        parameters.v_head_size_, &V_bsd, nullptr, 0, &V));

  // Apply the actual attention computation
  return ApplyAttention(&Q, &K, &V, attention_bias, nullptr, nullptr, output, /* present_key */ nullptr,
                        /* present_value */ nullptr, /* output_qk */ nullptr, parameters, context, nullptr, nullptr, -1);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
