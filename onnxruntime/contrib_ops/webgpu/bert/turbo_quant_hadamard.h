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

// Avoid `using namespace` in headers. Pull in only what we need.
using onnxruntime::webgpu::Program;
using onnxruntime::webgpu::ProgramUniformVariableDataType;
using onnxruntime::webgpu::ShaderHelper;

// Fused TurboQuant copy-to-KV-cache with Hadamard rotation and 4-bit quantization.
// Applies the Walsh-Hadamard transform to new K/V tokens, quantizes to 4-bit
// centroid indices packed into u32 words with fp32 L2 norm, then writes into
// the present KV cache (stored as u32).
// Each workgroup handles one (batch, head, seq) slice for either K or V.
class TurboQuantHadamardProgram final : public Program<TurboQuantHadamardProgram> {
 public:
  TurboQuantHadamardProgram(const std::string& kernel_name, bool has_past, bool kv_BNSH,
                            bool past_present_share_buffer, int head_size_log2, int components,
                            int compressed_head_size_u32,
                            bool prepare_indirect_dispatch = false, bool use_seqlen_k = false)
      : Program{kernel_name},
        has_past_(has_past),
        kv_BNSH_(kv_BNSH),
        past_present_share_buffer_(past_present_share_buffer),
        head_size_log2_(head_size_log2),
        components_(components),
        compressed_head_size_u32_(compressed_head_size_u32),
        prepare_indirect_dispatch_(prepare_indirect_dispatch),
        use_seqlen_k_(use_seqlen_k) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"compressed_head_size_u32", ProgramUniformVariableDataType::Uint32},
                                          {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"num_q_tiles", ProgramUniformVariableDataType::Uint32},
                                          {"num_slices_per_kv", ProgramUniformVariableDataType::Uint32},
                                          {"present_seq_length", ProgramUniformVariableDataType::Uint32},
                                          {"tile_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_;
  bool kv_BNSH_;
  bool past_present_share_buffer_;
  int head_size_log2_;
  int components_;
  int compressed_head_size_u32_;
  bool prepare_indirect_dispatch_;
  bool use_seqlen_k_;
};

// ---------------------------------------------------------------------------
// TurboQuant present-KV allocator contract (IMPORTANT for pre-allocated outputs)
//
// When TurboQuant is active the present_key/present_value buffers do NOT store
// fp16/fp32 head vectors. Each head is compressed to:
//     compressed_u32_words = head_size / 8 + 1
// u32 words (one fp32 L2-norm scale + head_size 4-bit indices packed 8-per-u32),
// i.e. (head_size * 4 + 32) bits per head. Expressed in the tensor's element
// type the last dimension is compressed_u32_words * (4 / sizeof(element)).
//
// ONNX shape inference is provider-agnostic and still reports the uncompressed
// head_size for these outputs, because TurboQuant is a WebGPU provider option
// resolved at runtime — there is no static graph metadata channel to advertise
// the compressed layout. Therefore any caller that PRE-ALLOCATES the present
// buffers (IO-binding, graph capture, ORT GenAI) MUST size the last dimension
// to the compressed length above, keyed off the same `turboQuant` provider
// option, NOT off the model-reported head_size.
//
// This formula is duplicated in the GenAI allocator
// (onnxruntime-genai/src/models/kv_cache.cpp, ComputeTurboQuantHeadSize) and
// must be kept in sync. As a safety net, ApplyFlashAttention validates the
// supplied present buffer's last-dim byte size and fails with INVALID_ARGUMENT
// on a mismatch rather than corrupting memory.
// ---------------------------------------------------------------------------
Status TurboQuantCopyToQuantizedKVCache(onnxruntime::webgpu::ComputeContext& context, const WebgpuAttentionParameters& parameters,
                                        const Tensor* K, const Tensor* past_key, Tensor* present_key,
                                        const Tensor* V, const Tensor* past_value, Tensor* present_value,
                                        uint32_t tile_size, const Tensor* seqlen_k, Tensor* indirect_buffer,
                                        uint32_t num_q_tiles);

// Fused TurboQuant: Split packed QKV + Rotary K + Hadamard + Quantize K/V + Rotary Q.
// Single dispatch handles all Q/K/V processing from packed QKV input.
class TurboQuantFusedRotaryProgram final : public Program<TurboQuantFusedRotaryProgram> {
 public:
  TurboQuantFusedRotaryProgram(const std::string& kernel_name, int head_size_log2,
                               int half_rotary_dim,
                               int compressed_head_size_u32,
                               bool past_present_share_buffer,
                               bool prepare_indirect_dispatch, bool use_seqlen_k,
                               uint32_t multi_rotary_cache_concat_offset)
      : Program{kernel_name},
        head_size_log2_(head_size_log2),
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
  int head_size_log2_;
  int half_rotary_dim_;
  int compressed_head_size_u32_;
  bool past_present_share_buffer_;
  bool prepare_indirect_dispatch_;
  bool use_seqlen_k_;
  uint32_t multi_rotary_cache_concat_offset_;
};

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
                                                      uint32_t num_q_tiles);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
