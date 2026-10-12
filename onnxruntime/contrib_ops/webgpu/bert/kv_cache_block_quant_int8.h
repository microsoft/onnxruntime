// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/webgpu/bert/attention_common.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using onnxruntime::webgpu::ConfiguredProgram;
using onnxruntime::webgpu::ConfiguredShaderHelper;
using onnxruntime::webgpu::ProgramUniformVariableDataType;

#define WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_PROGRAM_CONFIG(F) \
  F(bool, has_past_)                                       \
  F(bool, kv_BNSH_)                                        \
  F(bool, past_present_share_buffer_)                      \
  F(int, head_size_)                                       \
  F(int, components_)                                      \
  F(int, compressed_head_size_u32_)                        \
  F(bool, prepare_indirect_dispatch_)                      \
  F(bool, use_seqlen_k_)

struct KvCacheBlockQuantInt8ProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_PROGRAM_CONFIG);
  static Config MakeConfig(bool has_past, bool kv_BNSH, bool past_present_share_buffer, int head_size, int components,
                           int compressed_head_size_u32, bool prepare_indirect_dispatch, bool use_seqlen_k) {
    Config result{};
    result.has_past_ = bool(has_past);
    result.kv_BNSH_ = bool(kv_BNSH);
    result.past_present_share_buffer_ = bool(past_present_share_buffer);
    result.head_size_ = int(head_size);
    result.components_ = int(components);
    result.compressed_head_size_u32_ = int(compressed_head_size_u32);
    result.prepare_indirect_dispatch_ = bool(prepare_indirect_dispatch);
    result.use_seqlen_k_ = bool(use_seqlen_k);

    return result;
  }
  static constexpr std::string_view name = "KvCacheBlockQuantInt8Copy";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                             \
  U(Uint32, compressed_head_size_u32)                               \
  U(Uint32, copy_sequence_length)                                   \
  U(Uint32, kv_num_heads)                                           \
  U(Uint32, kv_sequence_length)                                     \
  U(Uint32, num_heads)                                              \
  U(Uint32, num_q_tiles)                                            \
  U(Uint32, num_slices_per_kv)                                      \
  U(Uint32, past_input_seq_length)                                  \
  U(Uint32, present_seq_length)                                     \
  U(Uint32, tile_size)                                              \
  U(Uint32, total_sequence_length)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_PROGRAM_CONFIG

using KvCacheBlockQuantInt8Program = ConfiguredProgram<KvCacheBlockQuantInt8ProgramShader>;

Status BlockQuantInt8CopyToKvCache(onnxruntime::webgpu::ComputeContext& context,
                                   const WebgpuAttentionParameters& parameters,
                                   const Tensor* K, const Tensor* past_key, Tensor* present_key,
                                   const Tensor* V, const Tensor* past_value, Tensor* present_value,
                                   uint32_t tile_size, const Tensor* seqlen_k, Tensor* indirect_buffer,
                                   uint32_t num_q_tiles, const Tensor* total_seqlen);

#define WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_FUSED_ROTARY_PROGRAM_CONFIG(F) \
  F(int, head_size_)                                                    \
  F(int, half_rotary_dim_)                                              \
  F(int, compressed_head_size_u32_)                                     \
  F(bool, past_present_share_buffer_)                                   \
  F(bool, prepare_indirect_dispatch_)                                   \
  F(bool, use_seqlen_k_)                                                \
  F(uint32_t, multi_rotary_cache_concat_offset_)

struct KvCacheBlockQuantInt8FusedRotaryProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_FUSED_ROTARY_PROGRAM_CONFIG);
  static Config MakeConfig(int head_size, int half_rotary_dim, int compressed_head_size_u32, bool past_present_share_buffer,
                           bool prepare_indirect_dispatch, bool use_seqlen_k, uint32_t multi_rotary_cache_concat_offset) {
    Config result{};
    result.head_size_ = int(head_size);
    result.half_rotary_dim_ = int(half_rotary_dim);
    result.compressed_head_size_u32_ = int(compressed_head_size_u32);
    result.past_present_share_buffer_ = bool(past_present_share_buffer);
    result.prepare_indirect_dispatch_ = bool(prepare_indirect_dispatch);
    result.use_seqlen_k_ = bool(use_seqlen_k);
    result.multi_rotary_cache_concat_offset_ = uint32_t(multi_rotary_cache_concat_offset);

    return result;
  }
  static constexpr std::string_view name = "KvCacheBlockQuantInt8FusedRotary";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_FUSED_ROTARY_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                                          \
  U(Uint32, compressed_head_size_u32)                                            \
  U(Uint32, hidden_size)                                                         \
  U(Uint32, kv_hidden_size)                                                      \
  U(Uint32, kv_num_heads)                                                        \
  U(Uint32, kv_sequence_length)                                                  \
  U(Uint32, num_heads)                                                           \
  U(Uint32, num_kv_slices)                                                       \
  U(Uint32, num_q_slices)                                                        \
  U(Uint32, num_q_tiles)                                                         \
  U(Uint32, present_seq_length)                                                  \
  U(Uint32, tile_size)                                                           \
  U(Uint32, total_sequence_length)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_FUSED_ROTARY_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_FUSED_ROTARY_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_KV_CACHE_BLOCK_QUANT_INT8_FUSED_ROTARY_PROGRAM_CONFIG

using KvCacheBlockQuantInt8FusedRotaryProgram = ConfiguredProgram<KvCacheBlockQuantInt8FusedRotaryProgramShader>;

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
