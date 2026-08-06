// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dsv4_moe_router_impl.h"

#include <cfloat>

#include "contrib_ops/cuda/math/dsv4_common.cuh"
#include "core/platform/env_var_utils.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// The graph fills the unselected experts with this, not with -inf, so that a Cast to fp16
// still lands on a finite value on the way into QMoE. -1e30 does not survive that cast --
// fp16 saturates just past 65504 -- and an -inf mask is not harmless here: a token whose
// top-k experts all live on other ranks leaves the whole row masked, and softmax then
// evaluates -inf - (-inf). Any magnitude far below a real log weight, which is bounded by
// log(1) = 0 from above and stays within a few tens of zero in practice, masks identically,
// so fp16 takes one that fits. bf16 keeps the original value and is bit-for-bit unchanged.
template <typename CudaT>
struct MaskedLogWeight {
  static constexpr float kValue = -1e30f;
};

template <>
struct MaskedLogWeight<half> {
  static constexpr float kValue = -1e4f;
};

// ORT's Softplus keeps the exponent non-positive on both branches; log1p would not match it.
__device__ __forceinline__ float Softplus(float a) {
  return a > 0.0f ? a + logf(expf(-a) + 1.0f) : logf(expf(a) + 1.0f);
}

// Largest value wins, lowest expert index breaks a tie -- the order TopK(sorted=1) emits.
__device__ __forceinline__ void WarpArgMax(float& value, int& index) {
  for (int stride = 16; stride > 0; stride >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu, value, stride);
    const int other_index = __shfl_down_sync(0xffffffffu, index, stride);
    if (other > value || (other == value && other_index < index)) {
      value = other;
      index = other_index;
    }
  }
}

// Same pairwise shape as ORT's ReduceSum, which is what produces the row sums being replaced.
__device__ __forceinline__ float WarpTreeSum(float value) {
  for (int stride = 16; stride > 0; stride >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, stride);
  }
  return value;
}

// One block per token. The whole routing decision for a token reads its own `num_experts`
// scores and nothing else, so this is a couple of KB of shared memory and no global traffic
// beyond the row -- the nineteen nodes it replaces were each paying a kernel launch for it.
template <typename CudaT>
__global__ void DSV4MoERouterKernel(const DSV4MoERouterParams p,
                                    const float* __restrict__ scores,
                                    const float* __restrict__ bias,
                                    const int64_t* __restrict__ expert_ids,
                                    CudaT* __restrict__ router_probs,
                                    float* __restrict__ weight_scale,
                                    bool log_routed_experts) {
  extern __shared__ float smem[];
  const int num_experts = p.num_experts;
  const int topk = p.topk;

  float* s_orig = smem;                                    // [num_experts]
  float* s_sel = s_orig + num_experts;                     // [num_experts]
  float* s_weight = s_sel + num_experts;                   // [topk]
  float* s_warp = s_weight + topk;                         // [kWarps]
  int* s_index = reinterpret_cast<int*>(s_warp + kWarps);  // [topk]
  int* s_warp_index = s_index + topk;                      // [kWarps]

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const float* score_row = scores + static_cast<int64_t>(token) * num_experts;
  for (int e = tid; e < num_experts; e += kThreads) {
    const float orig = sqrtf(Softplus(score_row[e]));
    s_orig[e] = orig;
    s_sel[e] = bias == nullptr ? orig : orig + bias[e];
  }
  __syncthreads();

  if (expert_ids != nullptr) {
    // Hash routing: the experts are looked up by token id, so there is nothing to select.
    const int64_t* id_row = expert_ids + static_cast<int64_t>(token) * topk;
    for (int j = tid; j < topk; j += kThreads) {
      s_index[j] = static_cast<int>(id_row[j]);
    }
    __syncthreads();
  } else {
    for (int j = 0; j < topk; ++j) {
      float best = -FLT_MAX;
      int best_index = num_experts;
      for (int e = tid; e < num_experts; e += kThreads) {
        const float value = s_sel[e];
        if (value > best || (value == best && e < best_index)) {
          best = value;
          best_index = e;
        }
      }
      WarpArgMax(best, best_index);
      if (lane == 0) {
        s_warp[warp] = best;
        s_warp_index[warp] = best_index;
      }
      __syncthreads();
      if (warp == 0) {
        best = lane < kWarps ? s_warp[lane] : -FLT_MAX;
        best_index = lane < kWarps ? s_warp_index[lane] : num_experts;
        WarpArgMax(best, best_index);
        if (lane == 0) {
          s_index[j] = best_index;
          s_sel[best_index] = -FLT_MAX;
        }
      }
      __syncthreads();
    }
  }

  if (log_routed_experts && tid == 0 && topk == 6) {
    printf("ORT_DSV4_ROUTED_EXPERTS tokens=%d token=%d ids=%d,%d,%d,%d,%d,%d\n",
           p.num_tokens, token, s_index[0], s_index[1], s_index[2], s_index[3], s_index[4], s_index[5]);
  }

  if (warp == 0) {
    const float weight = lane < topk ? s_orig[s_index[lane]] : 0.0f;
    const float total = WarpTreeSum(weight);
    if (lane == 0) s_warp[0] = total;
  }
  __syncthreads();

  const float weight_sum = s_warp[0];
  for (int j = tid; j < topk; j += kThreads) {
    s_weight[j] = s_orig[s_index[j]] / weight_sum;
  }
  __syncthreads();

  // QMoE takes its router input in the log domain and softmaxes it, which returns exactly
  // these weights. This rank sees only its own expert columns, so the softmax hands back
  // w_e / W_local and `weight_scale` carries W_local back out to undo that after the GEMM.
  const int local_count = p.local_expert_count;
  CudaT* out_row = router_probs + static_cast<int64_t>(token) * local_count;
  float local_sum = 0.0f;
  for (int c = tid; c < local_count; c += kThreads) {
    const int expert = p.local_expert_start + c;
    float log_weight = MaskedLogWeight<CudaT>::kValue;
    float weight = 0.0f;
    for (int j = 0; j < topk; ++j) {
      if (s_index[j] == expert) {
        weight = s_weight[j];
        log_weight = logf(weight);
        break;
      }
    }
    out_row[c] = DSV4Conv<CudaT>::FromFloat(log_weight);
    local_sum += weight;
  }

  local_sum = WarpTreeSum(local_sum);
  if (lane == 0) s_warp[warp] = local_sum;
  __syncthreads();
  if (warp == 0) {
    float total = lane < kWarps ? s_warp[lane] : 0.0f;
    total = WarpTreeSum(total);
    if (lane == 0) weight_scale[token] = total * p.route_scale;
  }
}

}  // namespace

template <typename T>
Status LaunchDSV4MoERouter(cudaStream_t stream, const DSV4MoERouterParams& p, const float* scores,
                           const float* bias, const int64_t* expert_ids, T* router_probs,
                           float* weight_scale) {
  typedef typename ToCudaType<T>::MappedType CudaT;
    const static bool log_routed_experts =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DSV4_LOG_ROUTED_EXPERTS", 0) == 1;
  const size_t shared = (2 * static_cast<size_t>(p.num_experts) + p.topk + kWarps) * sizeof(float) +
                        (static_cast<size_t>(p.topk) + kWarps) * sizeof(int);
  DSV4MoERouterKernel<CudaT><<<p.num_tokens, kThreads, shared, stream>>>(
      p, scores, bias, expert_ids, reinterpret_cast<CudaT*>(router_probs), weight_scale, log_routed_experts);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

#define INSTANTIATE(T)                                                                   \
  template Status LaunchDSV4MoERouter<T>(cudaStream_t, const DSV4MoERouterParams&,       \
                                         const float*, const float*, const int64_t*, T*, \
                                         float*)

INSTANTIATE(float);
INSTANTIATE(MLFloat16);
INSTANTIATE(BFloat16);

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
