// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "contrib_ops/cuda/bert/varlen_linear_attention.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kColsPerBlock = 32;

__device__ __forceinline__ float Sigmoid(float x) {
  return x > 0.0f ? 1.0f / (1.0f + expf(-x)) : 1.0f - 1.0f / (1.0f + expf(x));
}

__device__ __forceinline__ float Softplus(float x) {
  return x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x));
}

template <typename G>
__device__ __forceinline__ float LoadDecay(
    const G* decay,
    const float* a_log,
    const float* dt_bias,
    int64_t token,
    int h,
    int i,
    int v_num_heads,
    int d_k,
    bool decay_per_key_dim,
    VarlenDecayActivation activation,
    bool params_per_key_dim) {
  const int64_t gate_index = decay_per_key_dim
                                 ? (token * v_num_heads + h) * d_k + i
                                 : token * v_num_heads + h;
  const float raw = to_float(decay[gate_index]);
  if (activation == VarlenDecayActivation::kNone) {
    return raw;
  }
  const int64_t param_index = params_per_key_dim ? static_cast<int64_t>(h) * d_k + i : h;
  return -expf(a_log[param_index]) * Softplus(raw + dt_bias[param_index]);
}

template <typename G>
__device__ __forceinline__ float LoadBeta(
    const G* beta,
    int64_t token,
    int h,
    int v_num_heads,
    bool beta_per_head,
    VarlenBetaActivation activation) {
  const float raw = to_float(beta[beta_per_head ? token * v_num_heads + h : token]);
  if (activation == VarlenBetaActivation::kSigmoid) {
    return Sigmoid(raw);
  }
  if (activation == VarlenBetaActivation::kTwiceSigmoid) {
    return 2.0f * Sigmoid(raw);
  }
  return raw;
}

__device__ __forceinline__ bool ValidateOffsets(
    const int32_t* cu_seqlens,
    int b,
    int batch_size,
    int total_tokens,
    bool require_one,
    int& start,
    int& end) {
  // This is containment rather than synchronous host validation. Every block checks the global
  // endpoints and its own row before any token/state access. A malformed row simply writes
  // nothing; all outputs are unspecified for a malformed offsets tensor.
  const int first = cu_seqlens[0];
  const int last = cu_seqlens[batch_size];
  start = cu_seqlens[b];
  end = cu_seqlens[b + 1];
  return first == 0 && last == total_tokens &&
         start >= 0 && start < end && end <= total_tokens &&
         (!require_one || end - start == 1);
}

struct ReadoutHeads {
  int query_head;
  int output_head;
};

__device__ __forceinline__ int ReadoutHeadCount(int q_num_heads, int v_num_heads) {
  return q_num_heads >= v_num_heads ? q_num_heads / v_num_heads : 1;
}

__device__ __forceinline__ ReadoutHeads GetReadoutHeads(
    int v_head, int q_num_heads, int v_num_heads, int group) {
  if (q_num_heads >= v_num_heads) {
    const int q_head = v_head * (q_num_heads / v_num_heads) + group;
    return {q_head, q_head};
  }
  // Inverse mapping. In particular, when Hq == Hk and Hv % Hq == 0 (Qwen), Q and K
  // use the same directly corresponding head group for each V head.
  return {v_head * q_num_heads / v_num_heads, v_head};
}

template <typename T, typename G, int DK, bool kAllOnes>
__global__ void VarlenLinearAttentionColKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const G* __restrict__ decay,
    const G* __restrict__ beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    T* __restrict__ output,
    const float* initial_state,
    float* final_state,
    float* checkpoints,
    const int32_t* __restrict__ cu_seqlens,
    int total_tokens,
    int batch_size,
    int q_num_heads,
    int k_num_heads,
    int v_num_heads,
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
    int max_checkpoints) {
  const int b = blockIdx.x;
  const int h_v = blockIdx.y;
  const int col0 = blockIdx.z * kColsPerBlock;
  const int tid = threadIdx.x;
  const int col = col0 + tid;

  int start;
  int end;
  if (!ValidateOffsets(cu_seqlens, b, batch_size, total_tokens, kAllOnes, start, end)) {
    return;
  }
  const int local_len = end - start;
  const int h_k = h_v / (v_num_heads / k_num_heads);

  // The production state is V-major. Cooperatively staging the contiguous [32,K] tile makes
  // both the prologue and epilogue warp-coalesced. Padding removes shared-memory bank conflicts
  // when a thread subsequently walks its K values.
  __shared__ float state_tile[kColsPerBlock][DK + 1];
  __shared__ float k_buf[DK];
  __shared__ float q_buf[DK];
  const int64_t state_head = (static_cast<int64_t>(b) * v_num_heads + h_v) * d_v * DK;
  for (int linear = tid; linear < kColsPerBlock * DK; linear += kColsPerBlock) {
    const int tile_v = linear / DK;
    const int i = linear % DK;
    state_tile[tile_v][i] = initial_state[state_head + static_cast<int64_t>(col0 + tile_v) * DK + i];
  }
  __syncthreads();

  float s_col[DK];
#pragma unroll
  for (int i = 0; i < DK; ++i) {
    s_col[i] = state_tile[tid][i];
  }

  const int64_t k_token_stride = static_cast<int64_t>(k_num_heads) * DK;
  const int64_t q_token_stride = static_cast<int64_t>(q_num_heads) * DK;
  const int64_t v_token_stride = static_cast<int64_t>(v_num_heads) * d_v;
  const int out_heads = max(q_num_heads, v_num_heads);

  for (int t = 0; t < local_len; ++t) {
    const int64_t token = static_cast<int64_t>(start) + t;
    for (int i = tid; i < DK; i += kColsPerBlock) {
      k_buf[i] = to_float(key[token * k_token_stride + h_k * DK + i]);
    }
    __syncthreads();

    if (needs_decay) {
      const bool scalar_decay =
          !decay_per_key_dim &&
          (decay_activation == VarlenDecayActivation::kNone || !decay_params_per_key_dim);
      const float scalar_decay_factor =
          scalar_decay
              ? expf(LoadDecay(decay, a_log, dt_bias, token, h_v, 0, v_num_heads, DK,
                               false, decay_activation, false))
              : 0.0f;
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        const float decay_factor =
            scalar_decay
                ? scalar_decay_factor
                : expf(LoadDecay(decay, a_log, dt_bias, token, h_v, i, v_num_heads, DK,
                                 decay_per_key_dim, decay_activation, decay_params_per_key_dim));
        s_col[i] *= decay_factor;
      }
    }

    float retrieval = 0.0f;
    if (needs_retrieval) {
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        retrieval += s_col[i] * k_buf[i];
      }
    }

    float delta = to_float(value[token * v_token_stride + h_v * d_v + col]);
    if (needs_beta) {
      delta = LoadBeta(beta, token, h_v, v_num_heads, beta_per_head, beta_activation) *
              (delta - retrieval);
    }
#pragma unroll
    for (int i = 0; i < DK; ++i) {
      s_col[i] += k_buf[i] * delta;
    }

    if (checkpoints != nullptr && t < max_checkpoints) {
      // Checkpoint copies transpose thread-owned columns back into coalesced V-major tiles.
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        state_tile[tid][i] = s_col[i];
      }
      __syncthreads();
      const int64_t checkpoint_head =
          ((static_cast<int64_t>(t) * batch_size + b) * v_num_heads + h_v) * d_v * DK;
      for (int linear = tid; linear < kColsPerBlock * DK; linear += kColsPerBlock) {
        const int tile_v = linear / DK;
        const int i = linear % DK;
        checkpoints[checkpoint_head + static_cast<int64_t>(col0 + tile_v) * DK + i] =
            state_tile[tile_v][i];
      }
    }

    const int head_count = ReadoutHeadCount(q_num_heads, v_num_heads);
    for (int group = 0; group < head_count; ++group) {
      const ReadoutHeads heads = GetReadoutHeads(h_v, q_num_heads, v_num_heads, group);
      __syncthreads();
      for (int i = tid; i < DK; i += kColsPerBlock) {
        q_buf[i] = to_float(query[token * q_token_stride + heads.query_head * DK + i]);
      }
      __syncthreads();
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        acc += s_col[i] * q_buf[i];
      }
      output[(token * out_heads + heads.output_head) * d_v + col] = from_float<T>(scale * acc);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < DK; ++i) {
    state_tile[tid][i] = s_col[i];
  }
  __syncthreads();
  for (int linear = tid; linear < kColsPerBlock * DK; linear += kColsPerBlock) {
    const int tile_v = linear / DK;
    const int i = linear % DK;
    final_state[state_head + static_cast<int64_t>(col0 + tile_v) * DK + i] =
        state_tile[tile_v][i];
  }
}

template <typename T, typename G>
__global__ void VarlenLinearAttentionRecurrentKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const G* __restrict__ decay,
    const G* __restrict__ beta,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    T* __restrict__ output,
    const float* initial_state,
    float* final_state,
    float* checkpoints,
    const int32_t* __restrict__ cu_seqlens,
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
    int max_checkpoints) {
  const int b = blockIdx.x;
  const int h_v = blockIdx.y;
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  int start;
  int end;
  if (!ValidateOffsets(cu_seqlens, b, batch_size, total_tokens, false, start, end)) {
    return;
  }
  const int h_k = h_v / (v_num_heads / k_num_heads);
  const int state_stride = d_k + 1;  // padding avoids K-stride bank conflicts
  const int64_t state_head = (static_cast<int64_t>(b) * v_num_heads + h_v) * d_v * d_k;

  extern __shared__ float smem[];
  float* state = smem;                              // [V,K+1], V-major
  float* k_buf = state + d_v * state_stride;        // [K]
  float* scratch = k_buf + d_k;                     // [max(K,V)]
  for (int linear = tid; linear < d_v * d_k; linear += threads) {
    const int v = linear / d_k;
    const int i = linear % d_k;
    state[v * state_stride + i] = initial_state[state_head + linear];
  }
  __syncthreads();

  const int64_t k_token_stride = static_cast<int64_t>(k_num_heads) * d_k;
  const int64_t q_token_stride = static_cast<int64_t>(q_num_heads) * d_k;
  const int64_t v_token_stride = static_cast<int64_t>(v_num_heads) * d_v;
  const int out_heads = max(q_num_heads, v_num_heads);

  for (int t = 0; t < end - start; ++t) {
    const int64_t token = static_cast<int64_t>(start) + t;
    if (tid < d_k) {
      k_buf[tid] = to_float(key[token * k_token_stride + h_k * d_k + tid]);
    }
    __syncthreads();

    if (needs_decay && tid < d_v) {
      const bool scalar_decay =
          !decay_per_key_dim &&
          (decay_activation == VarlenDecayActivation::kNone || !decay_params_per_key_dim);
      const float scalar_decay_factor =
          scalar_decay
              ? expf(LoadDecay(decay, a_log, dt_bias, token, h_v, 0, v_num_heads, d_k,
                               false, decay_activation, false))
              : 0.0f;
      for (int i = 0; i < d_k; ++i) {
        const float decay_factor =
            scalar_decay
                ? scalar_decay_factor
                : expf(LoadDecay(decay, a_log, dt_bias, token, h_v, i, v_num_heads, d_k,
                                 decay_per_key_dim, decay_activation, decay_params_per_key_dim));
        state[tid * state_stride + i] *= decay_factor;
      }
    }
    __syncthreads();

    if (needs_retrieval && tid < d_v) {
      float acc = 0.0f;
      for (int i = 0; i < d_k; ++i) {
        acc += state[tid * state_stride + i] * k_buf[i];
      }
      scratch[tid] = acc;
    }
    __syncthreads();

    if (tid < d_v) {
      float delta = to_float(value[token * v_token_stride + h_v * d_v + tid]);
      if (needs_beta) {
        delta = LoadBeta(beta, token, h_v, v_num_heads, beta_per_head, beta_activation) *
                (delta - scratch[tid]);
      }
      for (int i = 0; i < d_k; ++i) {
        state[tid * state_stride + i] += k_buf[i] * delta;
      }
    }
    __syncthreads();

    if (checkpoints != nullptr && t < max_checkpoints) {
      const int64_t checkpoint_head =
          ((static_cast<int64_t>(t) * batch_size + b) * v_num_heads + h_v) * d_v * d_k;
      for (int linear = tid; linear < d_v * d_k; linear += threads) {
        const int v = linear / d_k;
        const int i = linear % d_k;
        checkpoints[checkpoint_head + linear] = state[v * state_stride + i];
      }
    }

    const int head_count = ReadoutHeadCount(q_num_heads, v_num_heads);
    for (int group = 0; group < head_count; ++group) {
      const ReadoutHeads heads = GetReadoutHeads(h_v, q_num_heads, v_num_heads, group);
      __syncthreads();
      if (tid < d_k) {
        scratch[tid] = to_float(query[token * q_token_stride + heads.query_head * d_k + tid]);
      }
      __syncthreads();
      if (tid < d_v) {
        float acc = 0.0f;
        for (int i = 0; i < d_k; ++i) {
          acc += state[tid * state_stride + i] * scratch[i];
        }
        output[(token * out_heads + heads.output_head) * d_v + tid] = from_float<T>(scale * acc);
      }
    }
    __syncthreads();
  }

  for (int linear = tid; linear < d_v * d_k; linear += threads) {
    const int v = linear / d_k;
    const int i = linear % d_k;
    final_state[state_head + linear] = state[v * state_stride + i];
  }
}

}  // namespace

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
    size_t max_shared_memory_per_block) {
  const bool column_eligible = (d_k == 64 || d_k == 128) && d_v % kColsPerBlock == 0;
  const bool all_ones_candidate = total_tokens == batch_size;
  if (column_eligible) {
    const dim3 grid(batch_size, v_num_heads, d_v / kColsPerBlock);
    const dim3 block(kColsPerBlock);
    auto launch = [&](auto dk_tag, auto all_ones_tag) -> Status {
      constexpr int DK = decltype(dk_tag)::value;
      constexpr bool kAllOnes = decltype(all_ones_tag)::value;
      VarlenLinearAttentionColKernel<T, G, DK, kAllOnes><<<grid, block, 0, stream>>>(
          query, key, value, decay, beta, a_log, dt_bias, output, initial_state, final_state,
          checkpoints, cu_seqlens, total_tokens, batch_size, q_num_heads, k_num_heads,
          v_num_heads, d_v, scale, needs_decay, decay_per_key_dim, decay_activation,
          decay_params_per_key_dim, needs_beta, beta_per_head, beta_activation,
          needs_retrieval, max_checkpoints);
      return CUDA_CALL(cudaGetLastError());
    };
    if (d_k == 64) {
      return all_ones_candidate
                 ? launch(std::integral_constant<int, 64>{}, std::true_type{})
                 : launch(std::integral_constant<int, 64>{}, std::false_type{});
    }
    return all_ones_candidate
               ? launch(std::integral_constant<int, 128>{}, std::true_type{})
               : launch(std::integral_constant<int, 128>{}, std::false_type{});
  }

  const size_t shared_bytes =
      (static_cast<size_t>(d_v) * (d_k + 1) + d_k + std::max(d_k, d_v)) * sizeof(float);
  ORT_RETURN_IF_NOT(shared_bytes <= max_shared_memory_per_block,
                    "VarlenLinearAttention generic kernel requires ", shared_bytes,
                    " shared-memory bytes, but the device supports ", max_shared_memory_per_block);
  const int threads = ((std::max(d_k, d_v) + 31) / 32) * 32;
  ORT_RETURN_IF_NOT(threads <= max_threads_per_block,
                    "VarlenLinearAttention requires more threads than the device supports");

  // Set a device-constant value, not the shape-dependent launch size. Concurrent calls therefore
  // never race to mutate the function attribute to different values. The launch still requests
  // exactly shared_bytes.
  if (shared_bytes > 48 * 1024) {
    CUDA_RETURN_IF_ERROR(cudaFuncSetAttribute(
        VarlenLinearAttentionRecurrentKernel<T, G>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(max_shared_memory_per_block)));
  }

  VarlenLinearAttentionRecurrentKernel<T, G>
      <<<dim3(batch_size, v_num_heads), threads, shared_bytes, stream>>>(
          query, key, value, decay, beta, a_log, dt_bias, output, initial_state, final_state,
          checkpoints, cu_seqlens, total_tokens, batch_size, q_num_heads, k_num_heads,
          v_num_heads, d_k, d_v, scale, needs_decay, decay_per_key_dim, decay_activation,
          decay_params_per_key_dim, needs_beta, beta_per_head, beta_activation,
          needs_retrieval, max_checkpoints);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T, G)                                                                            \
  template Status LaunchVarlenLinearAttentionKernel<T, G>(                                           \
      cudaStream_t, const T*, const T*, const T*, const G*, const G*, const float*, const float*,     \
      T*, const float*, float*, float*, const int32_t*, int, int, int, int, int, int, int, float,     \
      bool, bool, VarlenDecayActivation, bool, bool, bool, VarlenBetaActivation, bool, int, int,       \
      size_t);

INSTANTIATE(float, float)
INSTANTIATE(half, half)
INSTANTIATE(half, float)
INSTANTIATE(__nv_bfloat16, __nv_bfloat16)
INSTANTIATE(__nv_bfloat16, float)

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
