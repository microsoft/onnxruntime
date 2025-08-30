// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status CopyKVCacheProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(total_sequence_length)
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  shader.AddInput("value", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddOutput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& present_value = shader.AddOutput("present_value", ShaderUsage::UseUniform);
  const auto& copy_kv_shape = shader.AddIndices("copy_kv_shape");
  shader.AddInput("seqlen_k");
  // If prepare_indirect_dispatch is enabled, add seqlen_k input and indirect_buffer output
  if (prepare_indirect_dispatch_) {
    shader.AddOutput("indirect_buffer", ShaderUsage::UseUniform);
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.copy_size")
                            << "  let output_indices = " << copy_kv_shape.OffsetToIndices("global_idx") << ";\n"
                            << "  let head_size_id = output_indices[3];\n"
                               "  let sequence_id = output_indices[2];\n"
                               "  let num_head_id = output_indices[1];\n"
                               "  let batch = output_indices[0];\n"
                               "  let total_seq_length = u32(seqlen_k[0u]) + 1u;\n";

  // Add indirect dispatch logic for thread 0
  if (prepare_indirect_dispatch_) {
    shader.MainFunctionBody() << "  // Prepare indirect dispatch buffer for thread 0\n"
                              << "  if (global_idx == 0u) {\n"
                              << "    let num_total_seq_length_tile = (total_seq_length + uniforms.tile_size - 1u) / uniforms.tile_size;\n"
                              << "    indirect_buffer[0] = num_total_seq_length_tile;\n"
                              << "    indirect_buffer[1] = uniforms.num_heads;\n"
                              << "    indirect_buffer[2] = 1u;\n"
                              << "  }\n\n";
  }

  if (has_past_) {
    shader.MainFunctionBody() << "let past_sequence_length = total_seq_length - uniforms.kv_sequence_length;\n";
    if (past_present_share_buffer_) {
      shader.MainFunctionBody() << "  let present_offset = " << present_key.IndicesToOffset("present_key_indices_t(batch, num_head_id, past_sequence_length + sequence_id, head_size_id)") << ";\n"
                                << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id, head_size_id)" : "key_indices_t(batch, sequence_id, num_head_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n";
    } else {
      const auto& past_key = shader.AddInput("past_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
      shader.AddInput("past_value", ShaderUsage::UseUniform);
      shader.MainFunctionBody() << "let present_offset = global_idx;"
                                << "if (sequence_id < past_sequence_length) {\n"
                                << "  let pastOffset = " << past_key.IndicesToOffset("past_key_indices_t(batch, num_head_id, sequence_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "past_key[pastOffset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "past_value[pastOffset]") << ";\n"
                                << "} else {\n"
                                << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id - past_sequence_length, head_size_id)" : "key_indices_t(batch, sequence_id - past_sequence_length, num_head_id, head_size_id)") << ";\n"
                                << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                                << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n"
                                << "}";
    }
  } else {
    shader.MainFunctionBody() << "  let present_offset = " << (past_present_share_buffer_ ? present_key.IndicesToOffset("output_indices") : "global_idx") << ";\n"
                              << "let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id, head_size_id)" : "key_indices_t(batch, sequence_id, num_head_id, head_size_id)") << ";\n"
                              << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                              << present_value.SetByOffset("present_offset", "value[offset]") << ";\n";
  }
  return Status::OK();
}

Status CopyKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                   const Tensor* V, const Tensor* past_value, Tensor* present_value,
                   const Tensor* seqlen_k, Tensor* indirect_buffer) {
  // CopyKVCache takes past key/value and current key/value and copies them to present key and value.
  // This makes it so that FlashAttention only needs to look at present key and value, and saves
  // number of input buffers in the shader, which we run out of (<=8) without this optimization.
  // If indirect_buffer is provided, also prepare indirect dispatch buffer for flash attention.
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);
  bool has_past = (parameters.total_sequence_length_ - parameters.kv_sequence_length_) > 0;
  // parameters.total_sequence_length_ is past_sequence_length + kv_sequence_length.
  // parameters.kv_num_heads_ may be smaller than parameters.num_heads_ when parameters.is_gqa_ is true.
  int num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  // Only copy the new kv data for static kv cache
  int copy_sequence_length = has_past && parameters.past_present_share_buffer_ ? parameters.kv_sequence_length_ : parameters.total_sequence_length_;
  TensorShape copy_kv_shape{parameters.batch_size_, num_heads, copy_sequence_length, parameters.head_size_ / components};
  int64_t copy_size = copy_kv_shape.Size();

  // Determine if we need to prepare indirect dispatch
  bool prepare_indirect_dispatch = (indirect_buffer != nullptr);
  constexpr uint32_t tile_size = 64;

  CopyKVCacheProgram program{"CopyKVCache", has_past, parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH, parameters.past_present_share_buffer_,
                             prepare_indirect_dispatch, tile_size, static_cast<uint32_t>(parameters.num_heads_)};
  if (parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    ORT_ENFORCE(parameters.qkv_format_ == Q_K_V_BSNH, "qkv format ", parameters.qkv_format_, " is not supported yet in CopyKVCache.");
    // Reshape (batch_size, kv_sequence_length, kv_hidden_size) to (batch_size, kv_sequence_length, num_head, head_size)
    TensorShape reshaped_KV_shape{parameters.batch_size_, parameters.kv_sequence_length_, num_heads, parameters.head_size_ / components};
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components}});
  }

  if (seqlen_k != nullptr) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }

  if (has_past && !parameters.past_present_share_buffer_) {
    program.AddInputs({{past_key, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }
  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank, components},
                      {present_value, ProgramTensorMetadataDependency::Rank, components}});

  // Add indirect_buffer output if preparing indirect dispatch
  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  program.AddIndices(std::move(copy_kv_shape));
  program.SetDispatchGroupSize(static_cast<uint32_t>((copy_size + 63) / 64))
      .SetWorkgroupSize(64)
      .CacheHint(has_past, parameters.qkv_format_, parameters.past_present_share_buffer_, prepare_indirect_dispatch)
      .AddUniformVariables({{static_cast<uint32_t>(copy_size)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {tile_size},
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status FlashAttentionProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(new_sequence_length)(total_sequence_length)
  //
  //  Expectation is that present_key, and present_value contain past key and values since
  //  we are out of storage buffers a shader can have and both past/present cant be passed.
  // The hidden size of each q head should be a multiple of 4 because shader uses vectorized loads.
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform);
  shader.AddInput("present_value", ShaderUsage::UseUniform);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                             WGSL_TEMPLATE_PARAMETER(is_qualcomm, is_qualcomm_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(prefer_subgroupshuffle, !is_nvidia_),
                             WGSL_TEMPLATE_PARAMETER(qkv_head_size, qkv_head_size_),
                             WGSL_TEMPLATE_PARAMETER(qkv_num_heads, qkv_num_heads_));
}

Status FlashAttentionDecodeQKTProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("seqlens_k");
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);
  shader.AddOutput("metadata", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  const uint32_t tile_size_k_vec = 8;
  const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;
  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_decode_qkt.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec));
}

Status ComputeFlashAttentionDecodeQKT(onnxruntime::webgpu::ComputeContext& context, const Tensor* Q,
                                      const Tensor* attention_bias, Tensor* output, Tensor* present_key, Tensor* metadata, const Tensor* seqlen_k,
                                      const WebgpuAttentionParameters& parameters, const Tensor* indirect_buffer, uint32_t num_total_seq_length_tile, uint32_t num_present_sequence_length_tile, uint32_t tile_size, bool use_indirect_dispatch, uint32_t present_sequence_length) {
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  const bool has_attention_bias = attention_bias != nullptr;
  const int components = 4;

  FlashAttentionDecodeQKTProgram program{"FlashAttentionDecodeQKT", has_attention_bias, tile_size};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {seqlen_k, ProgramTensorMetadataDependency::None}});
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::Rank},
                      {metadata, ProgramTensorMetadataDependency::Rank, 2}});

  const uint32_t vectorized_head_size = parameters.head_size_ / components;
  if (use_indirect_dispatch) {
    program.SetIndirectDispatchTensor(indirect_buffer);
  } else {
    program.SetDispatchGroupSize(parameters.num_heads_ * num_total_seq_length_tile);
  }
  program.SetWorkgroupSize(64)
      .CacheHint(tile_size, has_attention_bias)
      .AddUniformVariables({{static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<float>(alpha)},
                            {static_cast<uint32_t>(present_sequence_length)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {num_present_sequence_length_tile},
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status FlashAttentionDecodeSplitVxProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("metadata", ShaderUsage::UseUniform);
  shader.AddInput("qk", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("present_value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("seqlens_k");
  shader.AddOutput("out_split_vx", ShaderUsage::UseUniform);

  const uint32_t tile_size_k_vec = 8u;

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_decode_split_vx.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(head_size_vec, head_size_vec_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, WorkgroupSizeX() / tile_size_k_vec),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec));
}

Status ComputeFlashAttentionDecodeSplitVxScore(onnxruntime::webgpu::ComputeContext& context,
                                               const Tensor* metadata,
                                               const Tensor* qk,
                                               Tensor* out_split_vx,
                                               Tensor* present_value,
                                               const Tensor* seqlen_k,
                                               const WebgpuAttentionParameters& parameters,
                                               const Tensor* indirect_buffer,
                                               uint32_t num_total_seq_length_tile,
                                               uint32_t num_present_sequence_length_tile,
                                               uint32_t tile_size,
                                               bool use_indirect_dispatch,
                                               uint32_t present_sequence_length) {
  const int components = 4;
  int head_size_vec = parameters.v_head_size_ / components;
  FlashAttentionDecodeSplitVxProgram program{"FlashAttentionDecodeSplitVx", tile_size, head_size_vec};
  program.AddInputs({{metadata, ProgramTensorMetadataDependency::TypeAndRank, 2},
                     {qk, ProgramTensorMetadataDependency::TypeAndRank},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {seqlen_k, ProgramTensorMetadataDependency::None}});
  program.AddOutputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components}});  // [B, N, split_k, head_size]
  if (use_indirect_dispatch) {
    program.SetIndirectDispatchTensor(indirect_buffer);
  } else {
    program.SetDispatchGroupSize(parameters.num_heads_ * num_total_seq_length_tile);
  }
  program.CacheHint(tile_size, head_size_vec)
      .SetWorkgroupSize(64)
      .AddUniformVariables({{static_cast<uint32_t>(head_size_vec)},
                            {static_cast<uint32_t>(present_sequence_length)},
                            {static_cast<uint32_t>(parameters.n_reps)},
                            num_present_sequence_length_tile,
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status FlashAttentionDecodeVxReduceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input", ShaderUsage::UseUniform);
  shader.AddInput("seqlens_k");
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_decode_vx_reduce.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_));
}

Status ComputeFlashAttentionDecodeVxReduce(onnxruntime::webgpu::ComputeContext& context,
                                           const Tensor* out_split_vx,
                                           Tensor* output,
                                           const Tensor* seqlen_k,
                                           const WebgpuAttentionParameters& parameters,
                                           uint32_t num_present_sequence_length_tile) {
  const int components = 4;
  constexpr int tile_size = 8;
  int tile_head_size = tile_size * components;
  FlashAttentionDecodeVxReduceProgram program{"FlashAttentionDecodeVxReduce", tile_size};
  program.AddInputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {seqlen_k, ProgramTensorMetadataDependency::None}});
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components}});
  const uint32_t num_head_size_tile = static_cast<uint32_t>((parameters.v_head_size_ + tile_head_size - 1) / tile_head_size);
  program.SetDispatchGroupSize(parameters.num_heads_ * num_head_size_tile)
      .CacheHint(tile_size)
      .SetWorkgroupSize(tile_size * tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.v_head_size_ / components)},
                            num_present_sequence_length_tile,
                            {num_head_size_tile},
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context, const Tensor* seqlen_k) {
  // Extract present_sequence_length directly from present_key tensor shape:
  // (batch_size, num_heads, total_sequence_length/max_sequence_length, head_size)
  const int present_sequence_length = static_cast<int>(present_key->Shape()[2]);

  if (parameters.sequence_length_ > 1) {
    // For encode path, use the original CopyKVCache without indirect dispatch preparation
    ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value, seqlen_k, nullptr));

    const uint32_t tile_size = 64;
    bool has_attention_bias = attention_bias != nullptr;
    bool is_qualcomm = context.AdapterInfo().vendor == std::string_view{"qualcomm"};
    bool is_nvidia = context.AdapterInfo().vendor == std::string_view{"nvidia"};
    bool is_fp16 = (Q->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    FlashAttentionProgram program{"FlashAttention",
                                  has_attention_bias,
                                  is_qualcomm,
                                  is_fp16,
                                  parameters.head_size_,
                                  parameters.num_heads_,
                                  parameters.is_unidirectional_,
                                  is_nvidia};
    program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {present_key, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {present_value, ProgramTensorMetadataDependency::TypeAndRank, 4}});
    if (has_attention_bias) {
      program.AddInputs({{attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
    }
    program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
    const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                  : parameters.scale_;
    const uint32_t num_seq_tile = (parameters.sequence_length_ + tile_size - 1) / tile_size;
    program.SetDispatchGroupSize(parameters.num_heads_ * num_seq_tile)
        .SetWorkgroupSize(tile_size)
        .CacheHint(has_attention_bias, parameters.head_size_, parameters.num_heads_, parameters.is_unidirectional_, is_qualcomm, is_nvidia)
        .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                              {static_cast<uint32_t>(parameters.total_sequence_length_)},
                              {static_cast<uint32_t>(present_sequence_length)},
                              {static_cast<uint32_t>(parameters.total_sequence_length_ - parameters.kv_sequence_length_)},
                              {static_cast<uint32_t>(parameters.n_reps)},
                              {alpha},
                              {num_seq_tile}});

    return context.RunProgram(program);
  }

  // For decode path (sequence_length == 1)
  const TensorShapeVector qk_dims({parameters.batch_size_, parameters.num_heads_,
                                   parameters.sequence_length_, present_sequence_length});
  const TensorShape qk_shape(qk_dims);
  Tensor qk = context.CreateGPUTensor(Q->DataType(), qk_shape);
  constexpr uint32_t tile_size = 64;
  const uint32_t num_total_seq_length_tile = (parameters.total_sequence_length_ + tile_size - 1) / tile_size;
  const uint32_t num_present_sequence_length_tile = (present_sequence_length + tile_size - 1) / tile_size;

  // Determine if we should use indirect dispatch
  const bool use_indirect_dispatch = parameters.past_present_share_buffer_ &&
                                     seqlen_k != nullptr &&
                                     context.IsGraphCaptureEnabled();

  // Create indirect dispatch buffer if using indirect dispatch
  Tensor* indirect_buffer_ptr = nullptr;
  Tensor indirect_buffer;
  if (use_indirect_dispatch) {
    const TensorShape indirect_buffer_shape{3};  // 3 uint32 values for dispatch dimensions
    indirect_buffer = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), indirect_buffer_shape);
    indirect_buffer_ptr = &indirect_buffer;
    // Use the fused CopyKVCache that also prepares the indirect dispatch buffer
    ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value, seqlen_k, indirect_buffer_ptr));
  } else {
    // Use the original CopyKVCache without indirect dispatch preparation
    ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value, seqlen_k, nullptr));
  }

  // The metadata is used to store the max and sum of each tile.
  const TensorShapeVector metadata_dims({parameters.batch_size_, parameters.num_heads_,
                                         num_present_sequence_length_tile, 2});
  const TensorShape metadata_shape(metadata_dims);
  Tensor metadata = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), metadata_shape);
  ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeQKT(context, Q, attention_bias, &qk, present_key, &metadata, seqlen_k,
                                                     parameters, indirect_buffer_ptr, num_total_seq_length_tile,
                                                     num_present_sequence_length_tile, tile_size, use_indirect_dispatch,
                                                     present_sequence_length));

  const TensorShapeVector out_split_vx_dims({parameters.batch_size_, parameters.num_heads_,
                                             num_present_sequence_length_tile, parameters.head_size_});
  const TensorShape out_split_vx_shape(out_split_vx_dims);
  Tensor out_split_vx = context.CreateGPUTensor(Q->DataType(), out_split_vx_shape);
  ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeSplitVxScore(context, &metadata, &qk, &out_split_vx, present_value,
                                                              seqlen_k, parameters, indirect_buffer_ptr,
                                                              num_total_seq_length_tile,
                                                              num_present_sequence_length_tile, tile_size,
                                                              use_indirect_dispatch, present_sequence_length));
  ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeVxReduce(context, &out_split_vx, output, seqlen_k, parameters,
                                                          num_present_sequence_length_tile));

  return Status::OK();
}

bool CanApplyFlashAttention(const Tensor* bias, const Tensor* present_key, const Tensor* present_value,
                            const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  return parameters.batch_size_ == 1 &&
         !parameters.is_packed_qkv_ &&
         parameters.head_size_ == parameters.v_head_size_ &&
         bias == nullptr &&
         context.HasFeature(wgpu::FeatureName::Subgroups) &&
         present_key != nullptr && present_value != nullptr && present_key->SizeInBytes() > 0 &&
         present_value->SizeInBytes() > 0 &&
         ((context.AdapterInfo().vendor == std::string_view{"qualcomm"} && parameters.head_size_ % 8 == 0) || parameters.head_size_ % 4 == 0);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
