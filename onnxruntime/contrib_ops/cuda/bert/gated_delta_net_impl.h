// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "core/common/status.h"
#include "contrib_ops/cuda/bert/gated_delta_net_plan.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace gated_delta_net {

// Device pointers bound at execute time. The analog of a cuDNN variant pack.
template <typename T>
struct VariantPack {
  const T* query = nullptr;                // [total_tokens, Hq, K]
  const T* key = nullptr;                  // [total_tokens, Hk, K]
  const T* value = nullptr;                // [total_tokens, Hv, V]
  const int32_t* cu_seqlens = nullptr;     // [B+1] or null for uniform packing
  const int32_t* capture_count = nullptr;  // [B] or null when transition capture is disabled
  const float* decay = nullptr;            // [total_tokens, Hv] or [total_tokens, Hv, K]
  const float* beta = nullptr;             // [total_tokens, Hv]
  const float* initial_state = nullptr;    // [B, Hv, V, K] V-major, may alias final_state
  const float* a_log = nullptr;            // [Hv]
  const float* dt_bias = nullptr;          // [Hv]
  T* output = nullptr;                     // [total_tokens, max(Hq,Hv), V]
  float* final_state = nullptr;            // [B, Hv, V, K]
  float* state_update = nullptr;           // [B, C * (Hv + Hk*K + Hv*V)]
  void* workspace = nullptr;
};

template <typename T>
Status LaunchGatedDeltaNet(const Descriptor& desc, const Plan& plan, const VariantPack<T>& pack,
                           float scale, int max_threads_per_block,
                           size_t max_shared_memory_per_block, cudaStream_t stream);

// Engine::kChunkedSplit. Defined in gated_delta_net_split_impl.cu; float16 only.
template <typename T>
Status LaunchGatedDeltaNetSplit(const Descriptor& desc, const Plan& plan,
                                const VariantPack<T>& pack, float scale, cudaStream_t stream);

}  // namespace gated_delta_net
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
