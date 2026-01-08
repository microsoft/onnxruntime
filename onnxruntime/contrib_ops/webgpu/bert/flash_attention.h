// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/webgpu/bert/attention_common.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram final : public Program<SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram> {
 public:
  SplitPackedQKVWithRotaryEmbeddingAndCopyKVProgram(bool interleaved, bool prepare_indirect_dispatch)
      : Program{"SplitPackedQKVWithRotaryEmbeddingAndCopyKV"},
        interleaved_(interleaved),
        prepare_indirect_dispatch_(prepare_indirect_dispatch) {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"kv_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"half_rotary_dim", ProgramUniformVariableDataType::Uint32},
      {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"tile_size", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});

 private:
  const bool interleaved_;
  const bool prepare_indirect_dispatch_;
};

class CopyKVCacheProgram final : public Program<CopyKVCacheProgram> {
 public:
  CopyKVCacheProgram(const std::string& kernel_name, bool has_past, bool kv_BNSH, bool past_present_share_buffer,
                     bool prepare_indirect_dispatch = false, bool use_seqlen_k = false)
      : Program{kernel_name}, has_past_(has_past), kv_BNSH_(kv_BNSH), past_present_share_buffer_(past_present_share_buffer), prepare_indirect_dispatch_(prepare_indirect_dispatch), use_seqlen_k_(use_seqlen_k) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"copy_size", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"kv_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"tile_size", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_;
  bool kv_BNSH_;
  bool past_present_share_buffer_;
  bool prepare_indirect_dispatch_;
  bool use_seqlen_k_;
};

class FlashAttentionProgram final : public Program<FlashAttentionProgram> {
 public:
  FlashAttentionProgram(const std::string& kernel_name,
                        bool has_attention_bias,
                        bool is_qualcomm,
                        bool is_fp16,
                        int qkv_head_size,
                        int qkv_num_heads,
                        bool is_unidirectional,
                        bool is_nvidia,
                        bool q_BNSH,
                        bool use_seqlen_k = false)
      : Program{kernel_name},
        has_attention_bias_(has_attention_bias),
        is_qualcomm_(is_qualcomm),
        is_fp16_(is_fp16),
        qkv_head_size_(qkv_head_size),
        qkv_num_heads_(qkv_num_heads),
        is_unidirectional_(is_unidirectional),
        is_nvidia_(is_nvidia),
        q_BNSH_(q_BNSH),
        use_seqlen_k_(use_seqlen_k) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"num_seq_tile", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim0", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim1", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_attention_bias_;
  bool is_qualcomm_;
  bool is_fp16_;
  int qkv_head_size_;
  int qkv_num_heads_;
  bool is_unidirectional_;
  bool is_nvidia_;
  bool q_BNSH_;
  bool use_seqlen_k_;
};

class FlashAttentionDecodeQKTProgram final : public Program<FlashAttentionDecodeQKTProgram> {
 public:
  FlashAttentionDecodeQKTProgram(const std::string& kernel_name,
                                 bool has_attention_bias, uint32_t tile_size, bool use_indirect_dispatch)
      : Program{kernel_name}, has_attention_bias_(has_attention_bias), tile_size_(tile_size), use_indirect_dispatch_(use_indirect_dispatch) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"head_size_vec", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"num_present_sequence_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_heads", ProgramUniformVariableDataType::Uint32},
                                          {"batch_size", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim0", ProgramUniformVariableDataType::Uint32},
                                          {"attn_bias_dim1", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_attention_bias_;
  uint32_t tile_size_;
  bool use_indirect_dispatch_;
};

class FlashAttentionDecodeSplitVxProgram final : public Program<FlashAttentionDecodeSplitVxProgram> {
 public:
  FlashAttentionDecodeSplitVxProgram(const std::string& kernel_name, uint32_t tile_size, int head_size_vec, bool use_indirect_dispatch)
      : Program{kernel_name}, tile_size_(tile_size), head_size_vec_(head_size_vec), use_indirect_dispatch_(use_indirect_dispatch) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"head_size_vec", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"num_present_sequence_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"batch_heads", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_;
  int head_size_vec_;
  bool use_indirect_dispatch_;
};

class FlashAttentionDecodeVxReduceProgram final : public Program<FlashAttentionDecodeVxReduceProgram> {
 public:
  FlashAttentionDecodeVxReduceProgram(const std::string& kernel_name, uint32_t tile_size, uint32_t seq_tile_size, bool use_indirect_dispatch)
      : Program{kernel_name}, tile_size_(tile_size), seq_tile_size_(seq_tile_size), use_indirect_dispatch_(use_indirect_dispatch) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"head_size_vec", ProgramUniformVariableDataType::Uint32},
                                          {"num_total_seq_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_present_sequence_length_tile", ProgramUniformVariableDataType::Uint32},
                                          {"num_head_size_tile", ProgramUniformVariableDataType::Uint32},
                                          {"batch_heads", ProgramUniformVariableDataType::Uint32});

 private:
  uint32_t tile_size_;
  uint32_t seq_tile_size_;
  bool use_indirect_dispatch_;
};

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context, const Tensor* seqlen_k = nullptr,
                           const Tensor* cos_cache = nullptr, const Tensor* sin_cache = nullptr);

bool CanApplyFlashAttention(const Tensor* bias,
                            const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context);

// Split packed QKV with Q/K rotary embedding and copy KV cache fusion
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
                                                     uint32_t tile_size);
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
