// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/kv_cache_block_quant_int8.h"
#include "contrib_ops/webgpu/bert/kv_cache_quantization.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status KvCacheBlockQuantInt8Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias |
                                               ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  const auto& value = shader.AddInput("value", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddOutput("present_key", ShaderUsage::UseUniform);
  const auto& present_value = shader.AddOutput("present_value", ShaderUsage::UseUniform);

  if (use_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::None);
  }
  if (prepare_indirect_dispatch_) {
    shader.AddInput("total_sequence_length_input", ShaderUsage::None);
    shader.AddOutput("indirect_buffer", ShaderUsage::None);
  }

  const ShaderVariableHelper* past_key = &key;
  const ShaderVariableHelper* past_value = &value;
  if (has_past_) {
    past_key = &shader.AddInput("past_key", ShaderUsage::UseUniform);
    past_value = &shader.AddInput("past_value", ShaderUsage::UseUniform);
  }

  return WGSL_TEMPLATE_APPLY(shader, "bert/kv_cache_block_quant_int8.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(components, components_),
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(has_past, has_past_),
                             WGSL_TEMPLATE_PARAMETER(head_size, head_size_),
                             WGSL_TEMPLATE_PARAMETER(kv_BNSH, kv_BNSH_),
                             WGSL_TEMPLATE_PARAMETER(past_present_share_buffer, past_present_share_buffer_),
                             WGSL_TEMPLATE_PARAMETER(prepare_indirect_dispatch, prepare_indirect_dispatch_),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_VARIABLE(key, key),
                             WGSL_TEMPLATE_VARIABLE(past_key, *past_key),
                             WGSL_TEMPLATE_VARIABLE(past_value, *past_value),
                             WGSL_TEMPLATE_VARIABLE(present_key, present_key),
                             WGSL_TEMPLATE_VARIABLE(present_value, present_value),
                             WGSL_TEMPLATE_VARIABLE(value, value));
}

Status BlockQuantInt8CopyToKvCache(onnxruntime::webgpu::ComputeContext& context,
                                   const WebgpuAttentionParameters& parameters,
                                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                                   const Tensor* V, const Tensor* past_value, Tensor* present_value,
                                   uint32_t tile_size, const Tensor* seqlen_k, Tensor* indirect_buffer,
                                   uint32_t num_q_tiles, const Tensor* total_seqlen) {
  constexpr uint32_t bit_width = 8;
  const int head_size = parameters.head_size_;
  ORT_ENFORCE(head_size >= 4 && head_size % 4 == 0,
              "Q8 block KV cache quantization requires head_size to be divisible by 4, got ", head_size);
  ORT_ENFORCE(context.KvCacheQuantizationBits() == bit_width,
              "Q8 block quantization requires an 8-bit KV cache.");

  constexpr int components = 4;
  const int compressed_head_size_u32 = KvCacheQuantizedHeadSizeU32(head_size, bit_width);
  const bool has_past = !parameters.past_present_share_buffer_ &&
                        past_key != nullptr && past_value != nullptr && past_key->SizeInBytes() > 0;
  const int kv_num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  const int copy_sequence_length =
      parameters.past_present_share_buffer_ ? parameters.kv_sequence_length_ : parameters.total_sequence_length_;
  const uint32_t num_slices_per_kv =
      static_cast<uint32_t>(parameters.batch_size_ * kv_num_heads * copy_sequence_length);
  const uint32_t total_workgroups = 2 * num_slices_per_kv;
  constexpr uint32_t workgroup_size = 64;

  const bool prepare_indirect_dispatch = indirect_buffer != nullptr;
  const bool use_seqlen_k = seqlen_k != nullptr;
  const bool kv_BNSH =
      parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH || parameters.qkv_format_ == Q_K_V_BNSH;

  KvCacheBlockQuantInt8Program program{has_past, kv_BNSH, parameters.past_present_share_buffer_,
                                       head_size, components, compressed_head_size_u32,
                                       prepare_indirect_dispatch, use_seqlen_k};
  if (kv_BNSH) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    ORT_RETURN_IF_ERROR(
        (parameters.qkv_format_ == Q_K_V_BSNH)
            ? Status::OK()
            : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "qkv format ", parameters.qkv_format_, " is not supported yet."));
    TensorShape reshaped_KV_shape{
        parameters.batch_size_, parameters.kv_sequence_length_, kv_num_heads, head_size / components};
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
    program.AddInputs({{past_key, ProgramTensorMetadataDependency::TypeAndRank},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank},
                      {present_value, ProgramTensorMetadataDependency::Rank}});
  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  const uint32_t past_input_seq_length =
      has_past ? static_cast<uint32_t>(past_key->Shape()[2]) : 0u;
  const uint32_t present_seq_length = static_cast<uint32_t>(present_key->Shape()[2]);

  program.SetDispatchGroupSize(total_workgroups)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(has_past, parameters.qkv_format_, parameters.past_present_share_buffer_,
                 prepare_indirect_dispatch, use_seqlen_k, head_size, components,
                 compressed_head_size_u32)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.batch_size_)},
                            {static_cast<uint32_t>(compressed_head_size_u32)},
                            {static_cast<uint32_t>(copy_sequence_length)},
                            {static_cast<uint32_t>(kv_num_heads)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {num_q_tiles},
                            {num_slices_per_kv},
                            {past_input_seq_length},
                            {present_seq_length},
                            {tile_size},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)}});

  return context.RunProgram(program);
}

Status KvCacheBlockQuantInt8FusedRotaryProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& packed_qkv = shader.AddInput("packed_qkv", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);

  if (use_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::None);
  }
  if (prepare_indirect_dispatch_) {
    shader.AddInput("total_sequence_length_input", ShaderUsage::None);
  }

  const auto& query = shader.AddOutput("query", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddOutput("present_key", ShaderUsage::UseUniform);
  const auto& present_value = shader.AddOutput("present_value", ShaderUsage::UseUniform);
  if (prepare_indirect_dispatch_) {
    shader.AddOutput("indirect_buffer", ShaderUsage::None);
  }

  return WGSL_TEMPLATE_APPLY(shader, "bert/kv_cache_block_quant_int8_fused_rotary.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(half_rotary_dim, half_rotary_dim_),
                             WGSL_TEMPLATE_PARAMETER(head_size, head_size_),
                             WGSL_TEMPLATE_PARAMETER(multi_rotary_cache_concat_offset,
                                                     multi_rotary_cache_concat_offset_),
                             WGSL_TEMPLATE_PARAMETER(past_present_share_buffer,
                                                     past_present_share_buffer_),
                             WGSL_TEMPLATE_PARAMETER(prepare_indirect_dispatch,
                                                     prepare_indirect_dispatch_),
                             WGSL_TEMPLATE_PARAMETER(use_multi_rotary_cache_concat,
                                                     multi_rotary_cache_concat_offset_ > 0),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                             WGSL_TEMPLATE_VARIABLE(packed_qkv, packed_qkv),
                             WGSL_TEMPLATE_VARIABLE(present_key, present_key),
                             WGSL_TEMPLATE_VARIABLE(present_value, present_value),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache));
}

Status BlockQuantInt8ApplyRotaryAndCopyToKvCache(
    onnxruntime::webgpu::ComputeContext& context,
    const WebgpuAttentionParameters& parameters,
    const Tensor* packedQKV,
    const Tensor* seqlen_k,
    const Tensor* cos_cache,
    const Tensor* sin_cache,
    Tensor* query,
    Tensor* present_key,
    Tensor* present_value,
    Tensor* indirect_buffer,
    uint32_t tile_size,
    uint32_t num_q_tiles,
    const Tensor* total_seqlen) {
  constexpr uint32_t bit_width = 8;
  const int head_size = parameters.head_size_;
  ORT_ENFORCE(head_size >= 4 && head_size % 4 == 0,
              "Q8 block KV cache quantization requires head_size to be divisible by 4, got ", head_size);
  ORT_ENFORCE(context.KvCacheQuantizationBits() == bit_width,
              "Q8 block quantization requires an 8-bit KV cache.");

  const int compressed_head_size_u32 = KvCacheQuantizedHeadSizeU32(head_size, bit_width);
  const int kv_num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  const int half_rotary_dim = static_cast<int>(cos_cache->Shape()[1]);
  const uint32_t num_kv_slices =
      static_cast<uint32_t>(parameters.batch_size_ * kv_num_heads * parameters.kv_sequence_length_);
  const uint32_t num_q_slices =
      static_cast<uint32_t>(parameters.batch_size_ * parameters.num_heads_ * parameters.kv_sequence_length_);
  const uint32_t total_workgroups = 2 * num_kv_slices + num_q_slices;
  constexpr uint32_t workgroup_size = 64;

  const bool prepare_indirect_dispatch = indirect_buffer != nullptr;
  const bool use_seqlen_k = seqlen_k != nullptr;
  const uint32_t multi_rotary_cache_concat_offset = context.MultiRotaryCacheConcatOffset();

  KvCacheBlockQuantInt8FusedRotaryProgram program{
      head_size, half_rotary_dim, compressed_head_size_u32,
      parameters.past_present_share_buffer_, prepare_indirect_dispatch, use_seqlen_k,
      multi_rotary_cache_concat_offset};
  program.AddInput({packedQKV, ProgramTensorMetadataDependency::TypeAndRank});
  program.AddInputs({
      {cos_cache, ProgramTensorMetadataDependency::Rank},
      {sin_cache, ProgramTensorMetadataDependency::Rank},
  });
  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }
  if (prepare_indirect_dispatch) {
    program.AddInput({total_seqlen, ProgramTensorMetadataDependency::None});
  }
  program.AddOutputs({{query, ProgramTensorMetadataDependency::None},
                      {present_key, ProgramTensorMetadataDependency::Rank},
                      {present_value, ProgramTensorMetadataDependency::Rank}});
  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  const uint32_t present_seq_length = static_cast<uint32_t>(present_key->Shape()[2]);
  program.SetDispatchGroupSize(total_workgroups)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(parameters.past_present_share_buffer_, prepare_indirect_dispatch,
                 use_seqlen_k, head_size, half_rotary_dim, compressed_head_size_u32,
                 multi_rotary_cache_concat_offset)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.batch_size_)},
                            {static_cast<uint32_t>(compressed_head_size_u32)},
                            {static_cast<uint32_t>(parameters.hidden_size_)},
                            {static_cast<uint32_t>(parameters.kv_hidden_size_)},
                            {static_cast<uint32_t>(kv_num_heads)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {num_kv_slices},
                            {num_q_slices},
                            {num_q_tiles},
                            {present_seq_length},
                            {tile_size},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
