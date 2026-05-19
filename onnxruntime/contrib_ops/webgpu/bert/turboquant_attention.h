// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "contrib_ops/webgpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

// Encode kernel — packs fresh fp16 K/V into the uint8 cache.  One workgroup
// per (batch, kv_head, new_token) slot.
class TurboQuantEncodeProgram final : public Program<TurboQuantEncodeProgram> {
 public:
  TurboQuantEncodeProgram(uint32_t head_dim, uint32_t key_bits, uint32_t value_bits,
                          bool norm_correction)
      : Program{"TurboQuantEncode"},
        head_dim_(head_dim),
        key_bits_(key_bits),
        value_bits_(value_bits),
        norm_correction_(norm_correction) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"new_seq_len", ProgramUniformVariableDataType::Uint32},
      {"n_kv_heads", ProgramUniformVariableDataType::Uint32},
      {"max_seq_len", ProgramUniformVariableDataType::Uint32},
      {"past_seq_len", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t head_dim_;
  uint32_t key_bits_;
  uint32_t value_bits_;
  [[maybe_unused]] bool norm_correction_;  // unused in encode template; kept for symmetry with decode
};

// Decode kernel — reads packed cache, writes fp16 K/V scratch in BNSH layout
// for downstream attention.  One workgroup per (batch, kv_head, slot).
class TurboQuantDecodeProgram final : public Program<TurboQuantDecodeProgram> {
 public:
  TurboQuantDecodeProgram(uint32_t head_dim, uint32_t key_bits, uint32_t value_bits,
                          bool norm_correction)
      : Program{"TurboQuantDecode"},
        head_dim_(head_dim),
        key_bits_(key_bits),
        value_bits_(value_bits),
        norm_correction_(norm_correction) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"batch_size", ProgramUniformVariableDataType::Uint32},
      {"total_seq_len", ProgramUniformVariableDataType::Uint32},
      {"n_kv_heads", ProgramUniformVariableDataType::Uint32},
      {"max_seq_len", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t head_dim_;
  uint32_t key_bits_;
  uint32_t value_bits_;
  bool norm_correction_;
};

// Top-level orchestrator: runs encode + decode + standard FlashAttention.
// `key`, `value` are the fresh fp16 K/V from the model (BSNH layout, before
// rotary).  `past_key`, `past_value` are the packed uint8 cache (BNSH layout).
// `present_key`, `present_value` are the packed uint8 cache outputs.
//
// Behaviour mirrors the CUDA orchestrator with Option ε:
//   - For prompt step (past_seq == 0): encode → ApplyFlashAttention on fresh
//     fp16 K/V (skip decode entirely; bit-equivalent to fp16).
//   - For decode step (past_seq > 0): encode → decode (dequant cache) →
//     ApplyFlashAttention on the fp16 scratch.
Status RunTurboQuantAttention(onnxruntime::webgpu::ComputeContext& context,
                              const WebgpuAttentionParameters& params,
                              const Tensor* query,
                              const Tensor* key,
                              const Tensor* value,
                              const Tensor* past_key,
                              const Tensor* past_value,
                              const Tensor* k_codebook,
                              const Tensor* hadamard,
                              const Tensor* attention_bias,
                              const Tensor* head_sink,
                              const Tensor* seqlen_k,
                              const Tensor* cos_cache,
                              const Tensor* sin_cache,
                              Tensor* present_key,
                              Tensor* present_value,
                              Tensor* output,
                              uint32_t key_bits,
                              uint32_t value_bits,
                              bool norm_correction,
                              int local_window_size);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
