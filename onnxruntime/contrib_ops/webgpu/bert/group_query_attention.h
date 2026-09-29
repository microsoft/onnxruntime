// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class SplitPackedQKVWithRotaryEmbeddingProgram final : public Program<SplitPackedQKVWithRotaryEmbeddingProgram> {
 public:
  SplitPackedQKVWithRotaryEmbeddingProgram(bool interleaved,
                                           uint32_t multi_rotary_cache_concat_offset,
                                           bool use_total_sequence_length_input)
      : Program{"SplitPackedQKVWithRotaryEmbedding"},
        interleaved_{interleaved},
        multi_rotary_cache_concat_offset_{multi_rotary_cache_concat_offset},
        use_total_sequence_length_input_{use_total_sequence_length_input} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"kv_hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"kv_num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"half_rotary_dim", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"dispatch_size", ProgramUniformVariableDataType::Uint32});

 private:
  const bool interleaved_;
  const uint32_t multi_rotary_cache_concat_offset_;
  const bool use_total_sequence_length_input_;
};

class GroupQueryAttention final : public WebGpuKernel {
 public:
  GroupQueryAttention(const OpKernelInfo& info) : WebGpuKernel(info) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    int64_t kv_num_heads = 0;
    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0);
    kv_num_heads_ = static_cast<int>(kv_num_heads);

    scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
    softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);

    do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
    rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;

    use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;

    local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));

    const int64_t causal = info.GetAttrOrDefault<int64_t>("causal", 1);
    ORT_ENFORCE(causal == 0 || causal == 1, "causal must be 0 or 1.");
    if (causal == 0) {
      ORT_NOT_IMPLEMENTED("GroupQueryAttention (WebGPU): causal=0 is not implemented.");
    }

    // The windowed KV cache (cache-relative indexing + shift compaction) is implemented only by the
    // CUDA kernel. Fail loudly rather than treating the window-sized buffer as a full-length cache.
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("sliding_window_cache", 0) == 0,
                "GroupQueryAttention (WebGPU): sliding_window_cache=1 is not implemented.");

    qk_norm_epsilon_ = info.GetAttrOrDefault<float>("qk_norm_epsilon", 1e-6f);
  }

  int num_heads_;     // number of attention heads of Q
  int kv_num_heads_;  // number of attention heads of K or V
  float scale_;       // the scaling factor applied before softmax
  float softcap_;
  bool do_rotary_;  // whether or not to use rotary embeddings
  bool rotary_interleaved_;
  int local_window_size_;

  bool use_smooth_softmax_;
  // Epsilon used by per-head RMSNorm when q_norm_weight / k_norm_weight (inputs 14 / 15) are
  // provided. Consumed whenever those optional norm inputs are used (decode fast path or
  // prefill fallback), and ignored otherwise.
  float qk_norm_epsilon_;
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;
};

KernelCreateInfo CreateGroupQueryAttentionKernelInfo(bool enable_graph_capture);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
