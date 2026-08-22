// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <cstdint>
#include <string>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class VarlenLinearAttention final : public onnxruntime::cuda::CudaKernel {
 public:
  VarlenLinearAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string update_rule_;
  std::string decay_activation_;
  std::string beta_activation_;
  float scale_;
  int max_checkpoints_;
};

enum class VarlenDecayActivation : int {
  kNone,
  kSoftplusDecay,
};

enum class VarlenBetaActivation : int {
  kNone,
  kSigmoid,
  kTwiceSigmoid,
};

// Packed THD recurrence. State is always FP32 and V-major:
// initial_state/final_state [B, Hv, V, K], checkpoints [W, B, Hv, V, K].
// Every kernel block validates the global and local device offsets before touching token data.
template <typename T, typename G>
Status LaunchVarlenLinearAttentionKernel(
    cudaStream_t stream,
    const T* query,
    const T* key,
    const T* value,
    const G* decay,
    const G* beta,
    const float* a_log,
    const float* dt_bias,
    T* output,
    const float* initial_state,
    float* final_state,
    float* checkpoints,
    const int32_t* cu_seqlens,
    int total_tokens,
    int batch_size,
    int q_num_heads,
    int k_num_heads,
    int v_num_heads,
    int d_k,
    int d_v,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    VarlenDecayActivation decay_activation,
    bool decay_params_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    VarlenBetaActivation beta_activation,
    bool needs_retrieval,
    int max_checkpoints,
    int max_threads_per_block,
    size_t max_shared_memory_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
