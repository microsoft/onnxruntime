// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/turbo_quant_hadamard.h"
#include "contrib_ops/webgpu/bert/hadamard_transform.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/common/logging/logging.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status TurboQuantHadamardProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias |
                                               ShaderUsage::UseElementTypeAlias | ShaderUsage::UseIndicesTypeAlias);
  const auto& value = shader.AddInput("value", ShaderUsage::UseUniform);
  // present_key/present_value are u32 arrays (packed 4-bit quantized data)
  const auto& present_key = shader.AddOutput("present_key", ShaderUsage::UseUniform);
  const auto& present_value = shader.AddOutput("present_value", ShaderUsage::UseUniform);

  if (use_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::None);
  }
  if (prepare_indirect_dispatch_) {
    shader.AddOutput("indirect_buffer", ShaderUsage::None);
  }

  // Past KV cache is already u32-packed — add as uniform only (no type aliases needed).
  // The variable bindings are always passed to the template; when has_past_ is false the
  // template never references them (guarded by #if has_past), so binding them to placeholder
  // variables (key/value) is harmless and avoids passing a null variable pointer.
  const ShaderVariableHelper* past_key = &key;
  const ShaderVariableHelper* past_value = &value;
  if (has_past_) {
    past_key = &shader.AddInput("past_key", ShaderUsage::UseUniform);
    past_value = &shader.AddInput("past_value", ShaderUsage::UseUniform);
  }

  return WGSL_TEMPLATE_APPLY(shader, "bert/turbo_quant_hadamard.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(components, components_),
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(hadamard_size_log2, head_size_log2_),
                             WGSL_TEMPLATE_PARAMETER(has_past, has_past_),
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

Status TurboQuantCopyToQuantizedKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                                        const Tensor* K, const Tensor* past_key, Tensor* present_key,
                                        const Tensor* V, const Tensor* past_value, Tensor* present_value,
                                        uint32_t tile_size, const Tensor* seqlen_k, Tensor* indirect_buffer,
                                        uint32_t num_q_tiles) {
  const int head_size = parameters.head_size_;
  const int components = head_size % 4 == 0 ? 4 : (head_size % 2 == 0 ? 2 : 1);
  ORT_ENFORCE((head_size & (head_size - 1)) == 0 && head_size >= 8,
              "head_size must be a power of 2 >= 8 for Hadamard transform, got ", head_size);

  const int head_size_log2 = Log2OfPowerOfTwo(head_size);

  // Compressed KV cache: 1 u32 for norm + head_size/8 u32s for packed 4-bit indices.
  const int compressed_head_size_u32 = head_size / 8 + 1;

  bool has_past = !parameters.past_present_share_buffer_ && past_key != nullptr && past_value != nullptr && past_key->SizeInBytes() > 0;
  int kv_num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  int copy_sequence_length = parameters.past_present_share_buffer_ ? parameters.kv_sequence_length_ : parameters.total_sequence_length_;
  uint32_t num_slices_per_kv = static_cast<uint32_t>(parameters.batch_size_ * kv_num_heads * copy_sequence_length);
  uint32_t total_workgroups = 2 * num_slices_per_kv;  // K + V

  const uint32_t workgroup_size = std::min(static_cast<uint32_t>(head_size / 2), 64u);

  bool prepare_indirect_dispatch = (indirect_buffer != nullptr);
  bool use_seqlen_k = (seqlen_k != nullptr);
  ORT_RETURN_IF_ERROR(
      (!use_seqlen_k || parameters.batch_size_ == 1)
          ? Status::OK()
          : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "TurboQuant graph-capture decode path reads seqlen_k[0] for all batches and "
                            "currently supports batch_size == 1 only; got batch_size = ",
                            parameters.batch_size_));
  bool kv_BNSH = parameters.qkv_format_ == Q_K_V_BSNH_BNSH_BNSH || parameters.qkv_format_ == Q_K_V_BNSH;

  TurboQuantHadamardProgram program{"TurboQuantCopyToQuantizedKVCache", has_past, kv_BNSH,
                                    parameters.past_present_share_buffer_,
                                    head_size_log2, components,
                                    compressed_head_size_u32,
                                    prepare_indirect_dispatch, use_seqlen_k};
  // Inputs: K and V in their original format (fp16/fp32, vectorized).
  if (kv_BNSH) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    ORT_RETURN_IF_ERROR(
        (parameters.qkv_format_ == Q_K_V_BSNH)
            ? Status::OK()
            : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "qkv format ", parameters.qkv_format_, " is not supported yet."));
    TensorShape reshaped_KV_shape{parameters.batch_size_, parameters.kv_sequence_length_, kv_num_heads, head_size / components};
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, reshaped_KV_shape, components}});
  }

  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }

  // Past KV cache is already u32-packed (no vectorization).
  if (has_past) {
    program.AddInputs({{past_key, ProgramTensorMetadataDependency::TypeAndRank},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank}});
  }

  // Output: present KV cache as u32 (packed 4-bit quantized).
  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank},
                      {present_value, ProgramTensorMetadataDependency::Rank}});

  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  // present_key has shape (batch, kv_num_heads, present_seq_length, compressed_head_size_u32)
  uint32_t present_seq_length = static_cast<uint32_t>(present_key->Shape()[2]);

  program.SetDispatchGroupSize(total_workgroups)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(has_past, parameters.qkv_format_, parameters.past_present_share_buffer_,
                 prepare_indirect_dispatch, use_seqlen_k, head_size_log2, components, compressed_head_size_u32)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.batch_size_)},
                            {static_cast<uint32_t>(compressed_head_size_u32)},
                            {static_cast<uint32_t>(kv_num_heads)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {num_q_tiles},
                            {num_slices_per_kv},
                            {present_seq_length},
                            {tile_size},
                            {static_cast<uint32_t>(parameters.total_sequence_length_)}});

  return context.RunProgram(program);
}

Status TurboQuantFusedRotaryProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& packed_qkv = shader.AddInput("packed_qkv", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);

  if (use_seqlen_k_) {
    shader.AddInput("seqlen_k", ShaderUsage::None);
  }

  const auto& query = shader.AddOutput("query", ShaderUsage::UseUniform);
  // present_key/present_value are u32 arrays (packed 4-bit quantized data)
  const auto& present_key = shader.AddOutput("present_key", ShaderUsage::UseUniform);
  const auto& present_value = shader.AddOutput("present_value", ShaderUsage::UseUniform);

  if (prepare_indirect_dispatch_) {
    shader.AddOutput("indirect_buffer", ShaderUsage::None);
  }

  return WGSL_TEMPLATE_APPLY(shader, "bert/turbo_quant_fused_rotary_hadamard.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(compressed_head_size_u32, compressed_head_size_u32_),
                             WGSL_TEMPLATE_PARAMETER(hadamard_size_log2, head_size_log2_),
                             WGSL_TEMPLATE_PARAMETER(half_rotary_dim, half_rotary_dim_),
                             WGSL_TEMPLATE_PARAMETER(multi_rotary_cache_concat_offset, multi_rotary_cache_concat_offset_),
                             WGSL_TEMPLATE_PARAMETER(past_present_share_buffer, past_present_share_buffer_),
                             WGSL_TEMPLATE_PARAMETER(prepare_indirect_dispatch, prepare_indirect_dispatch_),
                             WGSL_TEMPLATE_PARAMETER(use_multi_rotary_cache_concat, multi_rotary_cache_concat_offset_ > 0),
                             WGSL_TEMPLATE_PARAMETER(use_seqlen_k, use_seqlen_k_),
                             WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                             WGSL_TEMPLATE_VARIABLE(packed_qkv, packed_qkv),
                             WGSL_TEMPLATE_VARIABLE(present_key, present_key),
                             WGSL_TEMPLATE_VARIABLE(present_value, present_value),
                             WGSL_TEMPLATE_VARIABLE(query, query),
                             WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache));
}

Status TurboQuantApplyRotaryAndCopyToQuantizedKVCache(onnxruntime::webgpu::ComputeContext& context,
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
                                                      uint32_t num_q_tiles) {
  const int head_size = parameters.head_size_;
  ORT_ENFORCE((head_size & (head_size - 1)) == 0 && head_size >= 8,
              "head_size must be a power of 2 >= 8 for TurboQuant fused rotary, got ", head_size);

  const int head_size_log2 = Log2OfPowerOfTwo(head_size);

  const int compressed_head_size_u32 = head_size / 8 + 1;
  const int kv_num_heads = parameters.is_gqa_ ? parameters.kv_num_heads_ : parameters.num_heads_;
  const int half_rotary_dim = static_cast<int>(cos_cache->Shape()[1]);

  // Dispatch: K slices + V slices + Q slices
  uint32_t num_kv_slices = static_cast<uint32_t>(parameters.batch_size_ * kv_num_heads * parameters.kv_sequence_length_);
  uint32_t num_q_slices = static_cast<uint32_t>(parameters.batch_size_ * parameters.num_heads_ * parameters.kv_sequence_length_);
  uint32_t total_workgroups = 2 * num_kv_slices + num_q_slices;

  const uint32_t workgroup_size = std::min(static_cast<uint32_t>(head_size / 2), 64u);

  bool prepare_indirect_dispatch = (indirect_buffer != nullptr);
  bool use_seqlen_k = (seqlen_k != nullptr);
  ORT_RETURN_IF_ERROR(
      (!use_seqlen_k || parameters.batch_size_ == 1)
          ? Status::OK()
          : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "TurboQuant graph-capture decode path reads seqlen_k[0] for all batches and "
                            "currently supports batch_size == 1 only; got batch_size = ",
                            parameters.batch_size_));
  const uint32_t multi_rotary_cache_concat_offset = context.MultiRotaryCacheConcatOffset();

  TurboQuantFusedRotaryProgram program{"TurboQuantFusedRotary", head_size_log2,
                                       half_rotary_dim,
                                       compressed_head_size_u32,
                                       parameters.past_present_share_buffer_,
                                       prepare_indirect_dispatch, use_seqlen_k,
                                       multi_rotary_cache_concat_offset};

  program.AddInput({packedQKV, ProgramTensorMetadataDependency::TypeAndRank});
  program.AddInputs({
      {cos_cache, ProgramTensorMetadataDependency::Rank},
      {sin_cache, ProgramTensorMetadataDependency::Rank},
  });

  if (use_seqlen_k) {
    program.AddInput({seqlen_k, ProgramTensorMetadataDependency::None});
  }

  program.AddOutputs({{query, ProgramTensorMetadataDependency::None},
                      {present_key, ProgramTensorMetadataDependency::Rank},
                      {present_value, ProgramTensorMetadataDependency::Rank}});

  if (prepare_indirect_dispatch) {
    program.AddOutput({indirect_buffer, ProgramTensorMetadataDependency::None});
  }

  uint32_t present_seq_length = static_cast<uint32_t>(present_key->Shape()[2]);

  program.SetDispatchGroupSize(total_workgroups)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(parameters.past_present_share_buffer_,
                 prepare_indirect_dispatch, use_seqlen_k, head_size_log2,
                 half_rotary_dim, compressed_head_size_u32, multi_rotary_cache_concat_offset)
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
