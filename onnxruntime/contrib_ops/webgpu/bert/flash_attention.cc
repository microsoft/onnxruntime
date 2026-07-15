// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/flash_attention.h"
#include "contrib_ops/webgpu/bert/hadamard_transform.h"
#include "contrib_ops/webgpu/bert/turbo_quant_hadamard.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// WGSL helper function for normalizing on-device indirect dispatch dims.
// Shared by CopyKVCacheProgram and SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram.
// Mirrors ProgramManager::NormalizeDispatchGroupSize three tiers:
//   1) direct (x, y, z) write when every dim is within the spec limit (65535);
//   2) 2D sqrt collapse when the product fits a square layout;
//   3) 3D cbrt collapse otherwise.
// Consumers are unaffected by the chosen layout: ShaderHelper flattens
// workgroup_id (x, y, z) into a single linear workgroup_idx.
// Caller contract: must register a storage output named exactly
// `indirect_buffer` of array<u32> with at least 3 elements.
constexpr const char kPopulateIndirectDispatchBufferFn[] = R"(
fn populate_indirect_dispatch_buffer(x: u32, y: u32, z: u32) {
  let limit = 65535u;  // WebGPU spec maxComputeWorkgroupsPerDimension
  if (x <= limit && y <= limit && z <= limit) {
    indirect_buffer[0] = x;
    indirect_buffer[1] = y;
    indirect_buffer[2] = z;
    return;
  }
  let size = f32(x) * f32(y) * f32(z);
  let dispatch_avg_2d = u32(ceil(sqrt(size)));
  if (dispatch_avg_2d <= limit) {
    indirect_buffer[0] = dispatch_avg_2d;
    indirect_buffer[1] = dispatch_avg_2d;
    indirect_buffer[2] = 1u;
    return;
  }
  let dispatch_avg_3d = u32(ceil(pow(size, 1.0 / 3.0)));
  indirect_buffer[0] = dispatch_avg_3d;
  indirect_buffer[1] = dispatch_avg_3d;
  indirect_buffer[2] = dispatch_avg_3d;
}
)";

Status SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& packed_qkv = sh.AddInput("packed_qkv", ShaderUsage::UseUniform);
  const auto& seqlens = sh.AddInput("seqlens", ShaderUsage::UseUniform);
  const auto& cos_cache = sh.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = sh.AddInput("sin_cache", ShaderUsage::UseUniform);
  if (prepare_indirect_dispatch_) {
    sh.AddInput("total_sequence_length_input", ShaderUsage::None);
  }

  const auto& query = sh.AddOutput("query", ShaderUsage::UseUniform);
  const auto& present_key = sh.AddOutput("present_key", ShaderUsage::UseUniform);
  const auto& present_value = sh.AddOutput("present_value", ShaderUsage::UseUniform);

  if (prepare_indirect_dispatch_) {
    sh.AddOutput("indirect_buffer", ShaderUsage::None);
  }

  return WGSL_TEMPLATE_APPLY(sh, "bert/split_packed_qkv_with_rotary_embedding_and_copykv.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(interleaved, interleaved_),
                             WGSL_TEMPLATE_PARAMETER(multi_rotary_cache_concat_offset, multi_rotary_cache_concat_offset_),
                             WGSL_TEMPLATE_PARAMETER(prepare_indirect_dispatch, prepare_indirect_dispatch_),
                             WGSL_TEMPLATE_PARAMETER(use_multi_rotary_cache_concat, multi_rotary_cache_concat_offset_ > 0),
                             WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                             WGSL_TEMPLATE_VARIABLE(packed_qkv, packed_qkv),
                             WGSL_TEMPLATE_VARIABLE(present_key, present_key),
                             WGSL_TEMPLATE_VARIABLE(present_value, present_value),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(seqlens, seqlens),
                             WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache));
}

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
  if (use_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::None);
  }
  // If prepare_indirect_dispatch is enabled, add total_sequence_length_input
  // and indirect_buffer output. total_sequence_length_input is the global max
  // total sequence length across the batch (from GQA input #6); using it for
  // dispatch sizing covers right-padded batches where batch 0 is not the max.
  if (prepare_indirect_dispatch_) {
    shader.AddInput("total_sequence_length_input", ShaderUsage::None);
    shader.AddOutput("indirect_buffer", ShaderUsage::None);
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.copy_size")
                            << "  let output_indices = " << copy_kv_shape.OffsetToIndices("global_idx") << ";\n"
                            << "  let head_size_id = output_indices[3];\n"
                               "  let sequence_id = output_indices[2];\n"
                               "  let num_head_id = output_indices[1];\n"
                               "  let batch = output_indices[0];\n";
  if (use_seqlen_k_) {
    shader.MainFunctionBody() << "  let total_seq_length = u32(seqlen_k[batch]) + 1u;\n";
  } else {
    shader.MainFunctionBody() << "  let total_seq_length = uniforms.total_sequence_length;\n";
  }
  // Right-padded batches with prompt shorter than kv_sequence_length would underflow u32; clamp to 0.
  shader.MainFunctionBody() << "  let past_sequence_length = select(total_seq_length - uniforms.kv_sequence_length, 0u, total_seq_length <= uniforms.kv_sequence_length);\n";
  if (past_present_share_buffer_) {
    shader.MainFunctionBody() << "  let present_offset = " << present_key.IndicesToOffset("present_key_indices_t(batch, num_head_id, past_sequence_length + sequence_id, head_size_id)") << ";\n";
  } else {
    shader.MainFunctionBody() << "  let present_offset = " << present_key.IndicesToOffset("present_key_indices_t(batch, num_head_id, sequence_id, head_size_id)") << ";\n";
  }

  // Add indirect dispatch logic for thread 0
  if (prepare_indirect_dispatch_) {
    shader.AdditionalImplementation() << kPopulateIndirectDispatchBufferFn;
    shader.MainFunctionBody() << "  if (global_idx == 0u) {\n"
                              << "    let global_total_seq_length = u32(total_sequence_length_input[0]);\n"
                              << "    let num_total_seq_length_tile = (global_total_seq_length + uniforms.tile_size - 1u) / uniforms.tile_size;\n"
                              << "    populate_indirect_dispatch_buffer(num_total_seq_length_tile, uniforms.num_heads * uniforms.num_q_tiles, uniforms.batch_size);\n"
                              << "  }\n\n";
  }

  if (has_past_) {
    const auto& past_key = shader.AddInput("past_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
    shader.AddInput("past_value", ShaderUsage::UseUniform);
    shader.MainFunctionBody() << "if (sequence_id < past_sequence_length) {\n"
                              << "  let pastOffset = " << past_key.IndicesToOffset("past_key_indices_t(batch, num_head_id, sequence_id, head_size_id)") << ";\n"
                              << "  " << present_key.SetByOffset("present_offset", "past_key[pastOffset]") << ";\n"
                              << "  " << present_value.SetByOffset("present_offset", "past_value[pastOffset]") << ";\n"
                              << "} else {\n"
                              << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id - past_sequence_length, head_size_id)" : "key_indices_t(batch, sequence_id - past_sequence_length, num_head_id, head_size_id)") << ";\n"
                              << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                              << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n"
                              << "}";
  } else {
    shader.MainFunctionBody() << "  let offset = " << key.IndicesToOffset(kv_BNSH_ ? "key_indices_t(batch, num_head_id, sequence_id, head_size_id)" : "key_indices_t(batch, sequence_id, num_head_id, head_size_id)") << ";\n"
                              << "  " << present_key.SetByOffset("present_offset", "key[offset]") << ";\n"
                              << "  " << present_value.SetByOffset("present_offset", "value[offset]") << ";\n";
  }
  return Status::OK();
}

Status PrepareIndirectDispatchProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("total_sequence_length_input", ShaderUsage::None);
  shader.AddOutput("indirect_buffer", ShaderUsage::None);
  shader.AdditionalImplementation() << kPopulateIndirectDispatchBufferFn;
  shader.MainFunctionBody()
      << "  let global_total_seq_length = u32(total_sequence_length_input[0]);\n"
      << "  let num_total_seq_length_tile = (global_total_seq_length + uniforms.tile_size - 1u) / uniforms.tile_size;\n"
      << "  populate_indirect_dispatch_buffer(num_total_seq_length_tile, uniforms.num_heads * uniforms.num_q_tiles, uniforms.batch_size);\n";
  return Status::OK();
}

Status CopyKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                   const Tensor* V, const Tensor* past_value, Tensor* present_value,
                   uint32_t tile_size, const Tensor* seqlen_k, Tensor* indirect_buffer, uint32_t num_q_tiles,
                   const Tensor* total_seqlen) {
  // CopyKVCache takes past key/value and current key/value and copies them to present key and value.
  // This makes it so that FlashAttention only needs to look at present key and value, and saves
  // number of input buffers in the shader, which we run out of (<=8) without this optimization.
  // If indirect_buffer is provided, also prepare indirect dispatch buffer for flash attention.
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);
  // has_past means non-static kv cache with valid past data
  bool has_past = !parameters.past_present_share_buffer_ && past_key != nullptr && past_value != nullptr && past_key->SizeInBytes() > 0;
  // parameters.total_sequence_length_ is past_sequence_length + kv_sequence_length.
  // parameters.kv_num_heads_ may be smaller than parameters.num_heads_ when parameters.is_gqa_ is true.
  int num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  // Only copy the new kv data for static kv cache
  int copy_sequence_length = parameters.past_present_share_buffer_ ? parameters.kv_sequence_length_ : parameters.total_sequence_length_;
  TensorShape copy_kv_shape{parameters.batch_size_, num_heads, copy_sequence_length, parameters.head_size_ / components};
  int64_t copy_size = copy_kv_shape.Size();

  // Determine if we need to prepare indirect dispatch
  bool prepare_indirect_dispatch = (indirect_buffer != nullptr);
  bool use_seqlen_k = (seqlen_k != nullptr);
  bool kv_BNSH = parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH || parameters.qkv_format_ == Q_K_V_BNSH;
  CopyKVCacheProgram program{"CopyKVCache", has_past, kv_BNSH, parameters.past_present_share_buffer_,
                             prepare_indirect_dispatch, use_seqlen_k};
  if (kv_BNSH) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    ORT_ENFORCE(parameters.qkv_format_ == Q_K_V_BSNH, "qkv format ", parameters.qkv_format_, " is not supported yet in CopyKVCache.");
    // Reshape (batch_size, kv_sequence_length, kv_hidden_size) to (batch_size, kv_sequence_length, num_head, head_size)
    TensorShape reshaped_KV_shape{parameters.batch_size_, parameters.kv_sequence_length_, num_heads, parameters.head_size_ / components};
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components}});
  }

  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (prepare_indirect_dispatch) {
    program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
  }

  if (has_past) {
    program.AddInputs({{past_key, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }
  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank, components},
                      {present_value, ProgramTensorMetadataDependency::Rank, components}});

  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  program.AddIndices(std::move(copy_kv_shape));
  program.SetDispatchGroupSize(static_cast<uint32_t>((copy_size + 63) / 64))
      .SetWorkgroupSize(64)
      .CacheHint(has_past, parameters.qkv_format_, parameters.past_present_share_buffer_, prepare_indirect_dispatch, use_seqlen_k)
      .AddUniformVariables({{static_cast<uint32_t>(copy_size)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {tile_size},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.batch_size_)},
                            {num_q_tiles}});

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
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (has_head_sink_) {
    shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(has_head_sink, has_head_sink_),
                             WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                             WGSL_TEMPLATE_PARAMETER(is_qualcomm, is_qualcomm_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(max_k_step_param, max_k_step_),
                             WGSL_TEMPLATE_PARAMETER(prefer_subgroupshuffle, !is_nvidia_),
                             WGSL_TEMPLATE_PARAMETER(q_BNSH, q_BNSH_),
                             WGSL_TEMPLATE_PARAMETER(qkv_head_size, qkv_head_size_),
                             WGSL_TEMPLATE_PARAMETER(qkv_num_heads, qkv_num_heads_),
                             WGSL_TEMPLATE_PARAMETER(turbo_quant, turbo_quant_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_PARAMETER(use_shm_path, use_shm_path_));
}

Status FlashAttentionDecodeQKVProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& q = shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& present_key = shader.AddInput("present_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& present_value = shader.AddInput("present_value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (use_indirect_dispatch_) {
    // Global max total sequence length across batches (from GQA input #6).
    // Used in indirect-dispatch mode for the workgroup_idx slicing so that
    // batch 0's per-batch length cannot undersize the dispatch grid.
    shader.AddInput("total_sequence_length_input", ShaderUsage::None);
  }
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  const auto& out_split_vx = shader.AddOutput("out_split_vx", ShaderUsage::UseUniform);
  const auto& metadata = shader.AddOutput("metadata", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  // Wider K tiling (32 vec4) with a 128-thread workgroup is used for decode (m_tile == 1) to
  // mirror MatMulNBits and improve GPU time. For prefill (m_tile > 1) the shared-memory
  // arrays that scale with tile_size_k_vec and m_tile would exceed the 32 KB workgroup
  // storage limit on some adapters, so keep the original 8 vec4 / 64-thread shape there.
  const uint32_t tile_size_k_vec = (m_tile_ == 1u) ? 32u : 8u;
  const uint32_t sub_tile_count = WorkgroupSizeX() / tile_size_k_vec;
  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_decode_qkv.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(has_attention_bias, has_attention_bias_),
                             WGSL_TEMPLATE_PARAMETER(is_unidirectional, is_unidirectional_),
                             WGSL_TEMPLATE_PARAMETER(m_tile, m_tile_),
                             WGSL_TEMPLATE_PARAMETER(q_BNSH, q_BNSH_),
                             WGSL_TEMPLATE_PARAMETER(sub_tile_count, sub_tile_count),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size_k_vec, tile_size_k_vec),
                             WGSL_TEMPLATE_PARAMETER(turbo_quant, turbo_quant_),
                             WGSL_TEMPLATE_PARAMETER(use_indirect_dispatch, use_indirect_dispatch_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_PARAMETER(v_head_size_vec, head_size_vec_),
                             WGSL_TEMPLATE_VARIABLE(metadata, metadata),
                             WGSL_TEMPLATE_VARIABLE(out_split_vx, out_split_vx),
                             WGSL_TEMPLATE_VARIABLE(present_key, present_key),
                             WGSL_TEMPLATE_VARIABLE(present_value, present_value),
                             WGSL_TEMPLATE_VARIABLE(q, q));
}

Status ComputeFlashAttentionDecodeQKV(onnxruntime::webgpu::ComputeContext& context, const Tensor* Q,
                                      const Tensor* attention_bias, Tensor* out_split_vx, Tensor* present_key, Tensor* present_value,
                                      Tensor* metadata, const Tensor* seqlen_k,
                                      const WebgpuAttentionParameters& parameters, const Tensor* indirect_buffer, uint32_t num_total_seq_length_tile, uint32_t num_present_sequence_length_tile, uint32_t tile_size, bool use_indirect_dispatch, uint32_t present_sequence_length, uint32_t m_tile, bool use_seqlen_k, const Tensor* total_seqlen,
                                      bool turbo_quant, int compressed_head_size_u32) {
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  const bool has_attention_bias = attention_bias != nullptr;
  const int components = 4;
  // TurboQuant changes view of kv cache from fp16/fp32 to packed u32.
  // It already packs 4 float values into a single u32, so KV cache tensors use 1 component.
  const int kv_cache_components = turbo_quant ? 1 : components;
  const int head_size_vec = parameters.v_head_size_ / components;

  bool q_BNSH = parameters.qkv_format_ == Q_K_V_BNSH;
  bool is_unidirectional = parameters.is_unidirectional_;
  FlashAttentionDecodeQKVProgram program{"FlashAttentionDecodeQKV", has_attention_bias, tile_size, head_size_vec, use_indirect_dispatch, q_BNSH, is_unidirectional, m_tile, use_seqlen_k, turbo_quant, compressed_head_size_u32};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, kv_cache_components},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, kv_cache_components}});
  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (use_indirect_dispatch) {
    program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
  }
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components},
                      {metadata, ProgramTensorMetadataDependency::Rank, 2}});

  const uint32_t vectorized_head_size = parameters.head_size_ / components;

  uint32_t attn_bias_dim0 = 1;
  uint32_t attn_bias_dim1 = 1;
  uint32_t attn_bias_dim3 = 0;
  if (has_attention_bias) {
    const auto& bias_shape = attention_bias->Shape();
    attn_bias_dim0 = static_cast<uint32_t>(bias_shape[0]);
    attn_bias_dim1 = static_cast<uint32_t>(bias_shape[1]);
    attn_bias_dim3 = static_cast<uint32_t>(bias_shape[3]);
  }

  if (use_indirect_dispatch) {
    program.SetIndirectDispatchTensor(indirect_buffer);
  } else {
    program.SetDispatchGroupSize(parameters.batch_size_ * parameters.num_heads_ * ((parameters.sequence_length_ + m_tile - 1) / m_tile) * num_total_seq_length_tile);
  }
  // Workgroup size mirrors the tile_size_k_vec choice inside the program's shader (see
  // FlashAttentionDecodeQKVProgram::GenerateShaderCode): 128 threads with 32 vec4 K tiles
  // for decode, 64 threads with 8 vec4 K tiles for prefill.
  const uint32_t workgroup_size = (m_tile == 1u) ? 128u : 64u;
  program.SetWorkgroupSize(workgroup_size)
      .CacheHint(tile_size, head_size_vec, has_attention_bias, use_indirect_dispatch, q_BNSH, is_unidirectional, m_tile, use_seqlen_k, turbo_quant, compressed_head_size_u32)
      .AddUniformVariables({{static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)},
                            {static_cast<float>(alpha)},
                            present_sequence_length,
                            {static_cast<uint32_t>(parameters.n_reps)},
                            {num_present_sequence_length_tile},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.batch_size_)},
                            {attn_bias_dim0},
                            {attn_bias_dim1},
                            {attn_bias_dim3},
                            {static_cast<uint32_t>(parameters.sequence_length_)}});

  return context.RunProgram(program);
}

Status FlashAttentionDecodeVxReduceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const auto& metadata = shader.AddInput("metadata", ShaderUsage::UseUniform);
  if (use_seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::None);
  }
  if (has_head_sink_) {
    shader.AddInput("head_sink", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "bert/flash_attention_decode_vx_reduce.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_head_sink, has_head_sink_),
                             WGSL_TEMPLATE_PARAMETER(m_tile, m_tile_),
                             WGSL_TEMPLATE_PARAMETER(seq_tile_size, seq_tile_size_),
                             WGSL_TEMPLATE_PARAMETER(tile_size, tile_size_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(metadata, metadata),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

Status ComputeFlashAttentionDecodeVxReduce(onnxruntime::webgpu::ComputeContext& context,
                                           const Tensor* out_split_vx,
                                           const Tensor* metadata,
                                           Tensor* output,
                                           const Tensor* seqlen_k,
                                           const WebgpuAttentionParameters& parameters,
                                           uint32_t num_total_seq_length_tile,
                                           uint32_t num_present_sequence_length_tile,
                                           uint32_t seq_tile_size,
                                           const Tensor* head_sink,
                                           uint32_t m_tile,
                                           bool use_seqlen_k) {
  const int components = 4;
  constexpr int tile_size = 8;
  int tile_head_size = tile_size * components;
  bool has_head_sink = head_sink != nullptr;
  FlashAttentionDecodeVxReduceProgram program{"FlashAttentionDecodeVxReduce", tile_size, seq_tile_size, has_head_sink, m_tile, use_seqlen_k};
  program.AddInputs({{out_split_vx, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {metadata, ProgramTensorMetadataDependency::TypeAndRank, 2}});
  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (has_head_sink) {
    program.AddInput({head_sink, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, components}});
  const uint32_t num_head_size_tile = static_cast<uint32_t>((parameters.v_head_size_ + tile_head_size - 1) / tile_head_size);
  const uint32_t batch_heads = static_cast<uint32_t>(parameters.batch_size_ * parameters.num_heads_);
  program.SetDispatchGroupSize(batch_heads * ((parameters.sequence_length_ + m_tile - 1) / m_tile) * num_head_size_tile)
      .CacheHint(tile_size, seq_tile_size, has_head_sink, m_tile, use_seqlen_k)
      .SetWorkgroupSize(tile_size * tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.v_head_size_ / components)},
                            num_total_seq_length_tile,
                            num_present_sequence_length_tile,
                            {num_head_size_tile},
                            {batch_heads},
                            {static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(parameters.num_heads_)}});

  return context.RunProgram(program);
}

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context, const Tensor* seqlen_k,
                           const Tensor* cos_cache, const Tensor* sin_cache, const Tensor* head_sink,
                           const Tensor* total_seqlen) {
  constexpr uint32_t tile_size = 64;

  const bool turbo_quant_enabled = context.KvCacheQuantizationEnabled();
  if (turbo_quant_enabled && (parameters.head_size_ < 8 || (parameters.head_size_ & (parameters.head_size_ - 1)) != 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "KV cache quantization requires head_size >= 8 and a power of 2. Got head_size=",
                           parameters.head_size_);
  }

  // Compressed head dimension, expressed in two units:
  //   compressed_head_size_u32 — u32 words per head (1 scale + head_size/8 packed 4-bit indices),
  //                              passed to the shaders as the packed KV dimension.
  //   present_last_dim         — the same span counted in Q elements (fp16/fp32), used to size an
  //                              internally-allocated present buffer so its u32 view lines up
  //                              (compressed_head_size_u32 * 4 bytes == present_last_dim * sizeof(Q elem)).
  const int compressed_head_size_u32 = turbo_quant_enabled ? (parameters.head_size_ / 8 + 1) : 0;
  const int64_t present_last_dim =
      turbo_quant_enabled
          ? static_cast<int64_t>(compressed_head_size_u32) * 4 / static_cast<int64_t>(Q->DataType()->Size())
          : parameters.head_size_;
  // Create present_key and present_value tensors if they are nullptr.
  // Skip allocation for kv_empty — present will be aliased to past below.
  Tensor internal_present_key;
  Tensor internal_present_value;
  const int present_kv_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  const bool kv_empty = (parameters.kv_sequence_length_ == 0);
  if (!kv_empty && present_key == nullptr) {
    TensorShapeVector present_kv_shape({parameters.batch_size_, present_kv_heads,
                                        parameters.total_sequence_length_, present_last_dim});
    internal_present_key = context.CreateGPUTensor(Q->DataType(), TensorShape(present_kv_shape));
    present_key = &internal_present_key;
  }
  if (!kv_empty && present_value == nullptr) {
    TensorShapeVector present_kv_shape({parameters.batch_size_, present_kv_heads,
                                        parameters.total_sequence_length_, present_last_dim});
    internal_present_value = context.CreateGPUTensor(Q->DataType(), TensorShape(present_kv_shape));
    present_value = &internal_present_value;
  }

  // Read seqlens_k per batch_idx in the shader whenever seqlens_k is supplied.
  // This covers both graph-capture (total_sequence_length_ is 0 on the host) and
  // right-padded batches (batch_size > 1 with distinct per-batch totals), and lets
  // batch=1 share the same path. When seqlens_k is null, kernels fall back to
  // uniforms.total_sequence_length.
  const bool use_seqlen_k = seqlen_k != nullptr;

  // Declare query_output at function scope to ensure it persists throughout the function
  Tensor query_output;
  // Declare rotated_q at function scope so the pointer remains valid
  Tensor rotated_q;

  // Compute m_tile early so it can be passed to CopyKVCache for indirect dispatch.
  const uint32_t m_tile = parameters.sequence_length_ >= 4 ? 4u : (parameters.sequence_length_ >= 2 ? 2u : 1u);
  const uint32_t num_q_tiles = (static_cast<uint32_t>(parameters.sequence_length_) + m_tile - 1u) / m_tile;

  // Create indirect dispatch buffer if using indirect dispatch
  Tensor* indirect_buffer_ptr = nullptr;
  Tensor indirect_buffer;

  // Prepare indirect dispatch buffer for split-reduce path with static KV cache.
  // When graph capture is enabled, total_sequence_length_ may be 0 (GPU-based
  // seqlen_k), so the indirect buffer computes dispatch sizes on GPU.
  // Static KV cache (past_present_share_buffer_) is guaranteed by GQA's
  // ORT_ENFORCE when graph capture is enabled.
  const bool use_indirect_dispatch = seqlen_k != nullptr &&
                                     total_seqlen != nullptr &&
                                     context.IsGraphCaptureEnabled();
  if (use_indirect_dispatch) {
    const TensorShape indirect_buffer_shape{3};  // 3 uint32 values for dispatch dimensions
    indirect_buffer = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), indirect_buffer_shape);
    indirect_buffer_ptr = &indirect_buffer;
  }

  const bool do_rotary = (cos_cache != nullptr && sin_cache != nullptr);

  // kv_empty (kv_sequence_length_ == 0) occurs in KV-shared / cross-layer KV reuse layers: the
  // layer computes its own Q but borrows another layer's already-populated KV cache instead of
  // producing new K/V. There is nothing to copy, so CopyKVCache is skipped and attention reads
  // the past buffers directly. Because no new KV is written, present buffers are intentionally
  // not allocated above and some call sites pass nullptr present outputs — so we alias past as
  // present here.
  if (kv_empty) {
    // do_rotary must be false here: GQA passes cos_cache=nullptr, sin_cache=nullptr for kv_empty
    // layers (rotary is applied to Q separately in GQA before calling ApplyFlashAttention).
    ORT_ENFORCE(!do_rotary, "kv_empty (kv_sequence_length==0) is incompatible with fused rotary+copyKV.");
    ORT_ENFORCE(past_key != nullptr && past_value != nullptr,
                "kv_empty path requires past KV context (KV-shared layers reuse another layer's cache).");
    // When past_present_share_buffer_ is true (MayInplace optimization), present already
    // shares the past buffer. No aliasing needed — the data is already in place.
    if (!parameters.past_present_share_buffer_) {
      // Alias past as present — flash attention only reads present_key/present_value,
      // and CopyKVCache is skipped when kv_empty, so no writes occur through these pointers.
      present_key = const_cast<Tensor*>(past_key);
      present_value = const_cast<Tensor*>(past_value);
    }

    // CopyKVCache normally prepares the indirect dispatch buffer. For kv_empty layers
    // CopyKVCache is skipped, so we prepare it here. Only needed under graph capture
    // because that is when total_seqlen is GPU-resident and CPU-side dispatch sizing
    // is unavailable.
    if (use_indirect_dispatch) {
      PrepareIndirectDispatchProgram program;
      program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
      program.AddOutput({indirect_buffer_ptr, ProgramTensorMetadataDependency::None});
      program.SetDispatchGroupSize(1)
          .SetWorkgroupSize(1)
          .AddUniformVariables({{tile_size},
                                {static_cast<uint32_t>(parameters.num_heads_)},
                                {num_q_tiles},
                                {static_cast<uint32_t>(parameters.batch_size_)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(program));
    }
  }

  // When TurboQuant is active, create u32 tensor views over present/past KV cache buffers.
  Tensor present_key_u32, present_value_u32;
  Tensor past_key_u32, past_value_u32;
  Tensor* tq_present_key = present_key;
  Tensor* tq_present_value = present_value;
  const Tensor* tq_past_key = past_key;
  const Tensor* tq_past_value = past_value;
  if (turbo_quant_enabled) {
    const int64_t bytes_per_elem = static_cast<int64_t>(present_key->DataType()->Size());
    const int64_t expected_last_dim_bytes = static_cast<int64_t>(compressed_head_size_u32) * 4;
    ORT_RETURN_IF_ERROR(
        (present_key->Shape().NumDimensions() == 4 && present_value->Shape().NumDimensions() == 4)
            ? Status::OK()
            : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "TurboQuant expects present_key/present_value to be 4-D tensors."));
    ORT_RETURN_IF_ERROR(
        (present_key->Shape()[3] * bytes_per_elem == expected_last_dim_bytes)
            ? Status::OK()
            : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "TurboQuant KV cache shape mismatch for present_key. Expected last_dim_bytes==",
                              expected_last_dim_bytes, ", got shape=", present_key->Shape().ToString()));
    ORT_RETURN_IF_ERROR(
        (present_value->Shape()[3] * bytes_per_elem == expected_last_dim_bytes)
            ? Status::OK()
            : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "TurboQuant KV cache shape mismatch for present_value. Expected last_dim_bytes==",
                              expected_last_dim_bytes, ", got shape=", present_value->Shape().ToString()));

    TensorShapeVector u32_present_shape({present_key->Shape()[0], present_key->Shape()[1],
                                         present_key->Shape()[2],
                                         static_cast<int64_t>(compressed_head_size_u32)});
    present_key_u32 = Tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape(u32_present_shape),
                             present_key->MutableDataRaw(), present_key->Location());
    present_value_u32 = Tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape(u32_present_shape),
                               present_value->MutableDataRaw(), present_value->Location());
    tq_present_key = &present_key_u32;
    tq_present_value = &present_value_u32;

    if (past_key != nullptr && past_key->SizeInBytes() > 0) {
      TensorShapeVector u32_past_shape({past_key->Shape()[0], past_key->Shape()[1],
                                        past_key->Shape()[2],
                                        static_cast<int64_t>(compressed_head_size_u32)});
      // past_key_u32 / past_value_u32 are read-only aliases over the past KV cache buffers.
      // The Tensor ctor takes a non-const data pointer, so const_cast is required here, but the
      // flash attention kernels only read through tq_past_key / tq_past_value — never write.
      past_key_u32 = Tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape(u32_past_shape),
                            const_cast<void*>(past_key->DataRaw()), past_key->Location());
      past_value_u32 = Tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape(u32_past_shape),
                              const_cast<void*>(past_value->DataRaw()), past_value->Location());
      tq_past_key = &past_key_u32;
      tq_past_value = &past_value_u32;
    }
  }

  // K/V copy is skipped for kv_empty (see the aliasing block above for why).
  if (!kv_empty) {
    if (do_rotary) {
      ORT_ENFORCE(parameters.is_packed_qkv_, "Fused SplitPackedQKVWithRotaryEmbeddingAndCopyKV requires packed QKV input.");
      ORT_ENFORCE(parameters.past_present_share_buffer_, "Fused SplitPackedQKVWithRotaryEmbeddingAndCopyKV requires static KV cache.");

      // Q points to the packed QKV tensor in this case, create query output tensor
      query_output = context.CreateGPUTensor(Q->DataType(), TensorShape({parameters.batch_size_, parameters.sequence_length_, parameters.hidden_size_}));

      if (turbo_quant_enabled) {
        ORT_RETURN_IF_ERROR(TurboQuantApplyRotaryAndCopyToQuantizedKVCache(context, parameters,
                                                                           Q, seqlen_k,
                                                                           cos_cache, sin_cache,
                                                                           &query_output, tq_present_key, tq_present_value,
                                                                           indirect_buffer_ptr, tile_size, num_q_tiles));
      } else {
        ORT_RETURN_IF_ERROR(RunSplitPackedQKVWithRotaryEmbeddingAndCopyKV(context, parameters,
                                                                          Q, seqlen_k,
                                                                          cos_cache, sin_cache,
                                                                          &query_output, present_key, present_value,
                                                                          indirect_buffer_ptr, tile_size, num_q_tiles,
                                                                          total_seqlen));
      }
      Q = &query_output;
    } else if (turbo_quant_enabled) {
      // TurboQuant without rotary: K/V must be non-null (kv_empty already handled above).
      ORT_ENFORCE(K != nullptr && V != nullptr,
                  "TurboQuant requires non-null K/V inputs when kv_sequence_length > 0.");
      ORT_RETURN_IF_ERROR(TurboQuantCopyToQuantizedKVCache(context, parameters, K, tq_past_key, tq_present_key, V, tq_past_value, tq_present_value,
                                                           tile_size, use_seqlen_k ? seqlen_k : nullptr, indirect_buffer_ptr, num_q_tiles));
    } else {
      ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value, tile_size, use_seqlen_k ? seqlen_k : nullptr, indirect_buffer_ptr, num_q_tiles, total_seqlen));
    }
  }

  // Extract present_sequence_length directly from present_key tensor shape
  // after kv_empty aliasing ensures present_key is valid:
  // (batch_size, num_heads, total_sequence_length/max_sequence_length, head_size)
  const uint32_t present_sequence_length = static_cast<uint32_t>(present_key->Shape()[2]);

  // Rotate Q before attention (Hadamard transform for TurboQuant).
  if (turbo_quant_enabled) {
    rotated_q = context.CreateGPUTensor(Q->DataType(), Q->Shape());
    ORT_RETURN_IF_ERROR(ApplyHadamardTransform(context, Q, &rotated_q, parameters.head_size_));
    Q = &rotated_q;
  }

  // When TurboQuant is active, write attention output to a temp buffer, then
  // inverse-Hadamard from temp -> final output.
  Tensor attn_output_temp;
  Tensor* attn_output = output;
  if (turbo_quant_enabled) {
    attn_output_temp = context.CreateGPUTensor(output->DataType(), output->Shape());
    attn_output = &attn_output_temp;
  }

  // Route between prefill path (FlashAttentionProgram, single kernel)
  // and split-reduce decode path (QKV + VxReduce, 2 kernels).
  // Split-reduce wins for short Q (sequence_length < 32) across all KV
  // cache lengths measured: 1.13x-2.07x faster at total_sequence_length
  // 128 / 500 / 2000 on a representative LLM (32 heads, head_size 96).
  const bool use_split_reduce = (parameters.sequence_length_ < 32);

  if (!use_split_reduce) {
    // Prefill path: FlashAttentionProgram (single kernel with subgroup shuffles)
    bool has_attention_bias = attention_bias != nullptr;
    bool is_qualcomm = context.AdapterInfo().vendor == std::string_view{"qualcomm"};
    bool is_nvidia = context.AdapterInfo().vendor == std::string_view{"nvidia"};
    bool is_apple = context.AdapterInfo().vendor == std::string_view{"apple"};
    bool has_subgroups = context.HasFeature(wgpu::FeatureName::Subgroups);
    bool is_fp16 = (Q->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    bool q_BNSH = parameters.qkv_format_ == Q_K_V_BNSH;
    bool has_head_sink = head_sink != nullptr;
    FlashAttentionProgram program{"FlashAttention",
                                  has_attention_bias,
                                  is_qualcomm,
                                  is_fp16,
                                  parameters.head_size_,
                                  parameters.num_heads_,
                                  parameters.is_unidirectional_,
                                  is_nvidia,
                                  is_apple,
                                  has_subgroups,
                                  q_BNSH,
                                  use_seqlen_k,
                                  has_head_sink,
                                  turbo_quant_enabled,
                                  compressed_head_size_u32};
    // When TQ is active, KV cache is u32-packed — use u32 tensor views for present_key/present_value.
    const Tensor* fa_present_key = turbo_quant_enabled ? tq_present_key : present_key;
    const Tensor* fa_present_value = turbo_quant_enabled ? tq_present_value : present_value;
    program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {fa_present_key, ProgramTensorMetadataDependency::TypeAndRank, turbo_quant_enabled ? 1 : 4},
                       {fa_present_value, ProgramTensorMetadataDependency::TypeAndRank, turbo_quant_enabled ? 1 : 4}});
    if (has_attention_bias) {
      program.AddInputs({{attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
    }
    if (use_seqlen_k) {
      program.AddInputs({{seqlen_k, ProgramTensorMetadataDependency::None}});
    }
    if (has_head_sink) {
      program.AddInputs({{head_sink, ProgramTensorMetadataDependency::Type}});
    }
    program.AddOutputs({{attn_output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
    const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                  : parameters.scale_;

    // On Apple GPUs, use a larger workgroup size to reduce barrier overhead.
    const uint32_t prefill_tile_size = is_apple ? 128 : tile_size;
    const uint32_t num_seq_tile = (parameters.sequence_length_ + prefill_tile_size - 1) / prefill_tile_size;

    uint32_t attn_bias_dim0 = 1;
    uint32_t attn_bias_dim1 = 1;
    uint32_t attn_bias_dim3 = 0;
    if (has_attention_bias) {
      const auto& bias_shape = attention_bias->Shape();
      attn_bias_dim0 = static_cast<uint32_t>(bias_shape[0]);
      attn_bias_dim1 = static_cast<uint32_t>(bias_shape[1]);
      attn_bias_dim3 = static_cast<uint32_t>(bias_shape[3]);
    }

    program.SetDispatchGroupSize(parameters.batch_size_ * parameters.num_heads_ * num_seq_tile)
        .SetWorkgroupSize(prefill_tile_size)
        .CacheHint(has_attention_bias, parameters.head_size_, parameters.num_heads_, parameters.is_unidirectional_, is_qualcomm, is_nvidia, is_apple, has_subgroups, q_BNSH, use_seqlen_k, has_head_sink, turbo_quant_enabled, compressed_head_size_u32, program.max_k_step())
        .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                              {static_cast<uint32_t>(parameters.total_sequence_length_)},
                              {static_cast<uint32_t>(present_sequence_length)},
                              {static_cast<uint32_t>(parameters.batch_size_)},
                              {static_cast<uint32_t>(parameters.n_reps)},
                              {alpha},
                              {num_seq_tile},
                              {attn_bias_dim0},
                              {attn_bias_dim1},
                              {attn_bias_dim3}});

    ORT_RETURN_IF_ERROR(context.RunProgram(program));
  } else {
    // Split-reduce path (fused QKV + VxReduce). Handles both TQ and non-TQ.
    const uint32_t num_total_seq_length_tile = (parameters.total_sequence_length_ + tile_size - 1) / tile_size;
    const uint32_t num_present_sequence_length_tile = (present_sequence_length + tile_size - 1) / tile_size;

    const TensorShapeVector metadata_dims({parameters.batch_size_, parameters.num_heads_,
                                           parameters.sequence_length_, num_present_sequence_length_tile, 2});
    const TensorShape metadata_shape(metadata_dims);
    Tensor metadata = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), metadata_shape);

    const TensorShapeVector out_split_vx_dims({parameters.batch_size_, parameters.num_heads_,
                                               parameters.sequence_length_, num_present_sequence_length_tile, parameters.head_size_});
    const TensorShape out_split_vx_shape(out_split_vx_dims);
    Tensor out_split_vx = context.CreateGPUTensor(Q->DataType(), out_split_vx_shape);

    Tensor* qkv_present_key = turbo_quant_enabled ? tq_present_key : present_key;
    Tensor* qkv_present_value = turbo_quant_enabled ? tq_present_value : present_value;

    ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeQKV(context, Q, attention_bias, &out_split_vx, qkv_present_key, qkv_present_value,
                                                       &metadata, seqlen_k,
                                                       parameters, indirect_buffer_ptr, num_total_seq_length_tile,
                                                       num_present_sequence_length_tile, tile_size, use_indirect_dispatch,
                                                       present_sequence_length, m_tile, use_seqlen_k, total_seqlen,
                                                       turbo_quant_enabled, compressed_head_size_u32));

    ORT_RETURN_IF_ERROR(ComputeFlashAttentionDecodeVxReduce(context, &out_split_vx, &metadata, attn_output, seqlen_k, parameters,
                                                            num_total_seq_length_tile,
                                                            num_present_sequence_length_tile, tile_size,
                                                            head_sink, m_tile, use_seqlen_k));
  }

  // Apply inverse Hadamard transform: attn_output_temp -> output.
  if (turbo_quant_enabled) {
    ORT_RETURN_IF_ERROR(ApplyHadamardTransform(context, attn_output, output, parameters.head_size_));
  }

  return Status::OK();
}

bool CanApplyFlashAttention(const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  return !parameters.is_packed_qkv_ &&
         parameters.head_size_ == parameters.v_head_size_ &&
         ((context.AdapterInfo().vendor == std::string_view{"qualcomm"} && parameters.head_size_ % 8 == 0) || parameters.head_size_ % 4 == 0);
}

Status RunSplitPackedQKVWithRotaryEmbeddingAndCopyKV(onnxruntime::webgpu::ComputeContext& context,
                                                     const WebgpuAttentionParameters& params,
                                                     const Tensor* packedQKV,
                                                     const Tensor* seqlen_k,
                                                     const Tensor* cos_cache,
                                                     const Tensor* sin_cache,
                                                     Tensor* query,
                                                     Tensor* present_key,
                                                     Tensor* present_value,
                                                     Tensor* indirect_buffer,
                                                     uint32_t tile_size, uint32_t num_q_tiles,
                                                     const Tensor* total_seqlen) {
  const auto half_rotary_embedding_dim = gsl::narrow_cast<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = params.head_size_;

  int components = 1;
  // Currently we only support vectorization when RoPE is not interleaved
  if (!params.rotary_interleaved_) {
    if ((params.head_size_ % 4 == 0) && (half_rotary_embedding_dim % 4 == 0)) {
      components = 4;
    } else if ((params.head_size_ % 2 == 0) && (half_rotary_embedding_dim % 2 == 0)) {
      components = 2;
    }
  }
  // Adjust dimensions for vectorization
  const auto half_rotary_embedding_dim_vec = half_rotary_embedding_dim / components;
  const auto head_size_vec = head_size / components;

  // Dispatch: batch_size * sequence_length * num_heads * (half_rotary_dim + need_copy_dim)
  // work_per_head = half_rotary_dim + (head_size - 2 * half_rotary_dim)
  //               = head_size - half_rotary_dim
  const auto work_per_head = head_size_vec - half_rotary_embedding_dim_vec;
  auto dispatch_size = static_cast<uint32_t>(params.batch_size_ * params.sequence_length_ * params.num_heads_ * work_per_head);

  // Extract present_sequence_length from present_key tensor shape
  const uint32_t present_sequence_length = gsl::narrow_cast<uint32_t>(present_key->Shape()[2]);

  const bool prepare_indirect_dispatch = (indirect_buffer != nullptr);
  const uint32_t multi_rotary_cache_concat_offset = context.MultiRotaryCacheConcatOffset();

  SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram program(params.rotary_interleaved_, prepare_indirect_dispatch, multi_rotary_cache_concat_offset);
  program
      .CacheHint(params.rotary_interleaved_, prepare_indirect_dispatch, multi_rotary_cache_concat_offset)
      .AddInput({packedQKV, ProgramTensorMetadataDependency::TypeAndRank, components})
      .AddInputs({
          {seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
          {cos_cache, ProgramTensorMetadataDependency::Rank, components},
          {sin_cache, ProgramTensorMetadataDependency::Rank, components},
      });
  if (prepare_indirect_dispatch) {
    program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
  }
  program.AddOutputs({{query, ProgramTensorMetadataDependency::None, components},
                      {present_key, ProgramTensorMetadataDependency::None, components},
                      {present_value, ProgramTensorMetadataDependency::None, components}});

  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  program.AddUniformVariables({
      {static_cast<uint32_t>(params.sequence_length_)},
      {static_cast<uint32_t>(params.hidden_size_ / components)},
      {static_cast<uint32_t>(params.kv_hidden_size_ / components)},
      {static_cast<uint32_t>(params.num_heads_)},
      {static_cast<uint32_t>(params.kv_num_heads_)},
      {static_cast<uint32_t>(head_size_vec)},
      {static_cast<uint32_t>(half_rotary_embedding_dim_vec)},
      {present_sequence_length},
      {tile_size},
      {static_cast<uint32_t>(dispatch_size)},
      {static_cast<uint32_t>(params.batch_size_)},
      {num_q_tiles},
  });

  program.SetDispatchGroupSize((dispatch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
