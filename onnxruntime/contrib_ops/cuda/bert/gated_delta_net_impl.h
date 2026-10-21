// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
  const void* initial_state = nullptr;     // [B, Hv, V, K] V-major, may alias final_state
  const float* a_log = nullptr;            // [Hv]
  const float* dt_bias = nullptr;          // [Hv]
  T* output = nullptr;                     // [total_tokens, max(Hq,Hv), V]
  void* final_state = nullptr;             // [B, Hv, V, K]
  bool state_is_fp16 = false;
  float* state_update = nullptr;  // [B, C * (Hv + Hk*K + Hv*V)]
  void* workspace = nullptr;

#ifdef __CUDACC__
  __device__ float LoadState(int64_t index) const {
    if (initial_state == nullptr) return 0.0f;
    return state_is_fp16 ? __half2float(static_cast<const half*>(initial_state)[index])
                         : static_cast<const float*>(initial_state)[index];
  }

  __device__ void StoreState(int64_t index, float value) const {
    if (state_is_fp16) {
      static_cast<half*>(final_state)[index] = __float2half_rn(value);
    } else {
      static_cast<float*>(final_state)[index] = value;
    }
  }

  __device__ float RoundState(float value) const {
    return state_is_fp16 ? __half2float(__float2half_rn(value)) : value;
  }
#endif
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
