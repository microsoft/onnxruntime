// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/webgpu/bert/attention_common.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using onnxruntime::webgpu::Program;
using onnxruntime::webgpu::ProgramUniformVariableDataType;
using onnxruntime::webgpu::ShaderHelper;

class KvCacheBlockQuantInt8Program final : public Program<KvCacheBlockQuantInt8Program> {
 public:
  KvCacheBlockQuantInt8Program(bool has_past, bool kv_BNSH, bool past_present_share_buffer,
                               int head_size, int components, int compressed_head_size_u32,
                               bool prepare_indirect_dispatch, bool use_seqlen_k)
      : Program{"KvCacheBlockQuantInt8Copy"},
        has_past_(has_past),
        kv_BNSH_(kv_BNSH),
        past_present_share_buffer_(past_present_share_buffer),
        head_size_(head_size),
        components_(components),
        compressed_head_size_u32_(compressed_head_size_u32),
        prepare_indirect_dispatch_(prepare_indirect_dispatch),
        use_seqlen_k_(use_seqlen_k) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"compressed_head_size_u32", ProgramUniformVariableDataType::Uint32},
                                          {"copy_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"num_q_tiles", ProgramUniformVariableDataType::Uint32},
                                          {"num_slices_per_kv", ProgramUniformVariableDataType::Uint32},
                                          {"past_input_seq_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_seq_length", ProgramUniformVariableDataType::Uint32},
                                          {"tile_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_;
  bool kv_BNSH_;
  bool past_present_share_buffer_;
  int head_size_;
  int components_;
  int compressed_head_size_u32_;
  bool prepare_indirect_dispatch_;
  bool use_seqlen_k_;
};

Status BlockQuantInt8CopyToKvCache(onnxruntime::webgpu::ComputeContext& context,
                                   const WebgpuAttentionParameters& parameters,
                                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                                   const Tensor* V, const Tensor* past_value, Tensor* present_value,
                                   uint32_t tile_size, const Tensor* seqlen_k, Tensor* indirect_buffer,
                                   uint32_t num_q_tiles, const Tensor* total_seqlen);

class KvCacheBlockQuantInt8FusedRotaryProgram final
    : public Program<KvCacheBlockQuantInt8FusedRotaryProgram> {
 public:
  KvCacheBlockQuantInt8FusedRotaryProgram(int head_size, int half_rotary_dim,
                                          int compressed_head_size_u32,
                                          bool past_present_share_buffer,
                                          bool prepare_indirect_dispatch, bool use_seqlen_k,
                                          uint32_t multi_rotary_cache_concat_offset)
      : Program{"KvCacheBlockQuantInt8FusedRotary"},
        head_size_(head_size),
        half_rotary_dim_(half_rotary_dim),
        compressed_head_size_u32_(compressed_head_size_u32),
        past_present_share_buffer_(past_present_share_buffer),
        prepare_indirect_dispatch_(prepare_indirect_dispatch),
        use_seqlen_k_(use_seqlen_k),
        multi_rotary_cache_concat_offset_(multi_rotary_cache_concat_offset) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"compressed_head_size_u32", ProgramUniformVariableDataType::Uint32},
                                          {"hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"kv_hidden_size", ProgramUniformVariableDataType::Uint32},
                                          {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"num_kv_slices", ProgramUniformVariableDataType::Uint32},
                                          {"num_q_slices", ProgramUniformVariableDataType::Uint32},
                                          {"num_q_tiles", ProgramUniformVariableDataType::Uint32},
                                          {"present_seq_length", ProgramUniformVariableDataType::Uint32},
                                          {"tile_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32});

 private:
  int head_size_;
  int half_rotary_dim_;
  int compressed_head_size_u32_;
  bool past_present_share_buffer_;
  bool prepare_indirect_dispatch_;
  bool use_seqlen_k_;
  uint32_t multi_rotary_cache_concat_offset_;
};

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
    const Tensor* total_seqlen);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
