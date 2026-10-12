// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/webgpu/bert/attention_common.h"
#include "contrib_ops/webgpu/bert/kv_cache_quantization.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// Avoid `using namespace` in headers. Pull in only what we need.
using onnxruntime::webgpu::ConfiguredProgram;
using onnxruntime::webgpu::ConfiguredShaderHelper;
using onnxruntime::webgpu::ProgramUniformVariableDataType;

// Fused Q4 TurboQuant copy-to-KV-cache using Walsh-Hadamard, L2 normalization,
// and centroid quantization.
// Each workgroup handles one (batch, head, seq) slice for either K or V.
#define WEBGPU_TURBO_QUANT_HADAMARD_PROGRAM_CONFIG(F) \
  F(bool, has_past_)                                  \
  F(bool, kv_BNSH_)                                   \
  F(bool, past_present_share_buffer_)                 \
  F(int, head_size_log2_)                             \
  F(int, components_)                                 \
  F(int, compressed_head_size_u32_)                   \
  F(bool, prepare_indirect_dispatch_)                 \
  F(bool, use_seqlen_k_)

struct TurboQuantHadamardProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TURBO_QUANT_HADAMARD_PROGRAM_CONFIG);
  static onnxruntime::webgpu::ShaderProgramSetup<Config> MakeConfig(const std::string& kernel_name, bool has_past, bool kv_BNSH, bool past_present_share_buffer,
                                                                    int head_size_log2, int components, int compressed_head_size_u32, bool prepare_indirect_dispatch = false,
                                                                    bool use_seqlen_k = false) {
    Config result{};

    result.has_past_ = bool(has_past);
    result.kv_BNSH_ = bool(kv_BNSH);
    result.past_present_share_buffer_ = bool(past_present_share_buffer);
    result.head_size_log2_ = int(head_size_log2);
    result.components_ = int(components);
    result.compressed_head_size_u32_ = int(compressed_head_size_u32);
    result.prepare_indirect_dispatch_ = bool(prepare_indirect_dispatch);
    result.use_seqlen_k_ = bool(use_seqlen_k);

    return {std::move(result), kernel_name};
  }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TURBO_QUANT_HADAMARD_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                        \
  U(Uint32, compressed_head_size_u32)                          \
  U(Uint32, copy_sequence_length)                              \
  U(Uint32, kv_num_heads)                                      \
  U(Uint32, kv_sequence_length)                                \
  U(Uint32, num_heads)                                         \
  U(Uint32, num_q_tiles)                                       \
  U(Uint32, num_slices_per_kv)                                 \
  U(Uint32, past_input_seq_length)                             \
  U(Uint32, present_seq_length)                                \
  U(Uint32, tile_size)                                         \
  U(Uint32, total_sequence_length)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TURBO_QUANT_HADAMARD_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TURBO_QUANT_HADAMARD_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_TURBO_QUANT_HADAMARD_PROGRAM_CONFIG

using TurboQuantHadamardProgram = ConfiguredProgram<TurboQuantHadamardProgramShader>;

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
                                        uint32_t num_q_tiles, const Tensor* total_seqlen);

// Fused Q4 TurboQuant cache: split packed QKV, apply rotary to Q/K, then
// apply Walsh-Hadamard and centroid quantization to K/V.
// Single dispatch handles all Q/K/V processing from packed QKV input.
#define WEBGPU_TURBO_QUANT_FUSED_ROTARY_PROGRAM_CONFIG(F) \
  F(int, head_size_log2_)                                 \
  F(int, half_rotary_dim_)                                \
  F(int, compressed_head_size_u32_)                       \
  F(bool, past_present_share_buffer_)                     \
  F(bool, prepare_indirect_dispatch_)                     \
  F(bool, use_seqlen_k_)                                  \
  F(uint32_t, multi_rotary_cache_concat_offset_)

struct TurboQuantFusedRotaryProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_TURBO_QUANT_FUSED_ROTARY_PROGRAM_CONFIG);
  static onnxruntime::webgpu::ShaderProgramSetup<Config> MakeConfig(const std::string& kernel_name, int head_size_log2, int half_rotary_dim, int compressed_head_size_u32,
                                                                    bool past_present_share_buffer, bool prepare_indirect_dispatch, bool use_seqlen_k,
                                                                    uint32_t multi_rotary_cache_concat_offset) {
    Config result{};

    result.head_size_log2_ = int(head_size_log2);
    result.half_rotary_dim_ = int(half_rotary_dim);
    result.compressed_head_size_u32_ = int(compressed_head_size_u32);
    result.past_present_share_buffer_ = bool(past_present_share_buffer);
    result.prepare_indirect_dispatch_ = bool(prepare_indirect_dispatch);
    result.use_seqlen_k_ = bool(use_seqlen_k);
    result.multi_rotary_cache_concat_offset_ = uint32_t(multi_rotary_cache_concat_offset);

    return {std::move(result), kernel_name};
  }
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_TURBO_QUANT_FUSED_ROTARY_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, batch_size)                                            \
  U(Uint32, compressed_head_size_u32)                              \
  U(Uint32, hidden_size)                                           \
  U(Uint32, kv_hidden_size)                                        \
  U(Uint32, kv_num_heads)                                          \
  U(Uint32, kv_sequence_length)                                    \
  U(Uint32, num_heads)                                             \
  U(Uint32, num_kv_slices)                                         \
  U(Uint32, num_q_slices)                                          \
  U(Uint32, num_q_tiles)                                           \
  U(Uint32, present_seq_length)                                    \
  U(Uint32, tile_size)                                             \
  U(Uint32, total_sequence_length)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_TURBO_QUANT_FUSED_ROTARY_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_TURBO_QUANT_FUSED_ROTARY_PROGRAM_SHADER_UNIFORMS
};
#undef WEBGPU_TURBO_QUANT_FUSED_ROTARY_PROGRAM_CONFIG

using TurboQuantFusedRotaryProgram = ConfiguredProgram<TurboQuantFusedRotaryProgramShader>;

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
                                                      uint32_t num_q_tiles,
                                                      const Tensor* total_seqlen);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
