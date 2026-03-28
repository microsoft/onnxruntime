// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Fused recurrent linear attention CUDA kernel for gated_delta / delta / gated / linear update rules.
//
// Design: One thread block per (batch, kv_head). The state matrix [d_k, d_v] is loaded into
// shared memory at the start and kept there for the entire token loop. Each token's
// decay → retrieval → delta → update → readout sequence runs without global memory
// round-trips for the state. This matches the FLA (flash-linear-attention) kernel design.
//
// State tiles: For d_k=128, d_v=128, fp32 state = 64 KB shared memory. On SM80+ GPUs with
// 164 KB shared memory per SM, this fits with room for scratch. Requires
// cudaFuncSetAttribute to opt into extended shared memory (>48 KB).
//
// Thread mapping: num_threads = max(d_k, d_v) rounded to warp boundary. Each thread
// participates in both row operations (decay/update: tid < d_k handles row tid) and
// column operations (retrieval/readout: tid < d_v computes column tid's dot product).
//
// Reductions: Matrix-vector products (S^T @ k, S^T @ q) use column-per-thread dot products
// instead of atomicAdd, eliminating contention. Each thread tid computes
// sum_i(S[i, tid] * scalar[i]) by reading shared memory column-wise (bank-conflict-free
// when d_v is a multiple of 32).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "contrib_ops/cuda/bert/linear_attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Convert half/bfloat16 to float
template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float(float val) { return val; }

template <>
__device__ __forceinline__ float to_float(half val) { return __half2float(val); }

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }
#endif

// Convert float to half/bfloat16/float
template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ float from_float(float val) { return val; }

template <>
__device__ __forceinline__ half from_float(float val) { return __float2half(val); }

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }
#endif

// =============================================================================
// Fused recurrent linear attention kernel
//
// Grid:  (batch_size, kv_num_heads, 1)
// Block: (max(d_k, d_v) rounded to warp, 1, 1)
//
// Shared memory layout (dynamic):
//   float S_smem[d_k * d_v]              — recurrent state matrix
//   float s_scratch[max(d_k, d_v)]       — broadcast/retrieval/delta buffer
// =============================================================================
template <typename T>
__global__ void LinearAttentionRecurrentKernel(
    const T* __restrict__ query,    // [B, T, H_q * d_k]
    const T* __restrict__ key,      // [B, T, H_kv * d_k]
    const T* __restrict__ value,    // [B, T, H_kv * d_v]
    float* __restrict__ state,      // [B, H_kv, d_k, d_v] — in-place updated
    const T* __restrict__ decay,    // [B, T, H_kv] or [B, T, H_kv*d_k] or nullptr
    const T* __restrict__ beta_in,  // [B, T, H_kv] or [B, T, 1] or nullptr
    T* __restrict__ output,         // [B, T, H_q * d_v]
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int d_k,
    int d_v,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval) {
  const int b = blockIdx.x;     // batch index
  const int h_kv = blockIdx.y;  // kv head index
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;
  const int heads_per_group = q_num_heads / kv_num_heads;

  // Global state pointer for this (batch, head): [d_k, d_v]
  float* S_global = state + ((int64_t)b * kv_num_heads + h_kv) * d_k * d_v;

  // Shared memory layout
  extern __shared__ float smem[];
  float* S_smem = smem;  // [d_k * d_v]
  int scratch_size = (d_k > d_v) ? d_k : d_v;
  float* s_scratch = smem + d_k * d_v;  // [max(d_k, d_v)]

  // ---- Load state from global memory into shared memory ----
  for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
    S_smem[idx] = S_global[idx];
  }
  __syncthreads();

  // ---- Token loop ----
  for (int t = 0; t < seq_len; ++t) {
    // Load k_t[tid] into register (each thread loads one element)
    float kt_val = 0.0f;
    if (tid < d_k) {
      int k_offset = ((int64_t)b * seq_len + t) * (kv_num_heads * d_k) + h_kv * d_k + tid;
      kt_val = to_float(key[k_offset]);
    }

    // ---- Step 1: Decay — row-per-thread on shared memory ----
    if (needs_decay) {
      if (tid < d_k) {
        float exp_g;
        if (decay_per_key_dim) {
          int g_offset = ((int64_t)b * seq_len + t) * (kv_num_heads * d_k) + h_kv * d_k + tid;
          exp_g = expf(to_float(decay[g_offset]));
        } else {
          int g_offset = ((int64_t)b * seq_len + t) * kv_num_heads + h_kv;
          exp_g = expf(to_float(decay[g_offset]));
        }
        // Decay row tid of state in shared memory
        for (int j = 0; j < d_v; ++j) {
          S_smem[tid * d_v + j] *= exp_g;
        }
      }
      __syncthreads();
    }

    // ---- Step 2: Retrieval = S^T @ k_t — column-per-thread dot product ----
    if (needs_retrieval) {
      // Broadcast k_t values to shared memory for cross-thread access
      if (tid < d_k) {
        s_scratch[tid] = kt_val;
      }
      __syncthreads();

      // Each thread tid computes one column: retrieved[tid] = sum_i(S[i, tid] * k[i])
      // Column access S[i * d_v + tid] is bank-conflict-free within a warp when d_v % 32 == 0
      if (tid < d_v) {
        float acc = 0.0f;
        for (int i = 0; i < d_k; ++i) {
          acc += S_smem[i * d_v + tid] * s_scratch[i];
        }
        s_scratch[tid] = acc;
      }
      __syncthreads();
    }

    // ---- Step 3: State update — row-per-thread on shared memory ----
    if (needs_beta) {
      // Load beta_t (all threads can read — L1 broadcast)
      float bt;
      if (beta_per_head) {
        bt = to_float(beta_in[((int64_t)b * seq_len + t) * kv_num_heads + h_kv]);
      } else {
        bt = to_float(beta_in[((int64_t)b * seq_len + t)]);
      }

      // Pre-compute delta = beta * (v - retrieved) into scratch, one element per thread
      if (tid < d_v) {
        int v_base = ((int64_t)b * seq_len + t) * (kv_num_heads * d_v) + h_kv * d_v;
        float vj = to_float(value[v_base + tid]);
        s_scratch[tid] = bt * (vj - s_scratch[tid]);
      }
      __syncthreads();

      // Update state rows: S[tid, j] += k_t[tid] * delta[j]
      if (tid < d_k) {
        for (int j = 0; j < d_v; ++j) {
          S_smem[tid * d_v + j] += kt_val * s_scratch[j];
        }
      }
    } else {
      // linear/gated: Pre-load v_t into scratch
      if (tid < d_v) {
        int v_base = ((int64_t)b * seq_len + t) * (kv_num_heads * d_v) + h_kv * d_v;
        s_scratch[tid] = to_float(value[v_base + tid]);
      }
      __syncthreads();

      // S[tid, j] += k_t[tid] * v[j]
      if (tid < d_k) {
        for (int j = 0; j < d_v; ++j) {
          S_smem[tid * d_v + j] += kt_val * s_scratch[j];
        }
      }
    }
    __syncthreads();

    // ---- Step 4: Query readout — column-per-thread dot product ----
    for (int g = 0; g < heads_per_group; ++g) {
      if (g > 0) {
        // Ensure prior group finished reading s_scratch before it is overwritten.
        __syncthreads();
      }

      int h_q = h_kv * heads_per_group + g;

      // Broadcast q_t to shared memory for cross-thread access
      if (tid < d_k) {
        int q_offset = ((int64_t)b * seq_len + t) * (q_num_heads * d_k) + h_q * d_k + tid;
        s_scratch[tid] = to_float(query[q_offset]);
      }
      __syncthreads();

      // Each thread tid computes one output: scale * sum_i(S[i, tid] * q[i])
      if (tid < d_v) {
        float acc = 0.0f;
        for (int i = 0; i < d_k; ++i) {
          acc += S_smem[i * d_v + tid] * s_scratch[i];
        }
        int out_offset = ((int64_t)b * seq_len + t) * (q_num_heads * d_v) + h_q * d_v + tid;
        output[out_offset] = from_float<T>(scale * acc);
      }
    }

    // Keep ordering explicit before the next token iteration reuses shared scratch.
    __syncthreads();
  }

  // ---- Write state back from shared memory to global memory ----
  for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
    S_global[idx] = S_smem[idx];
  }
}

template <typename T, int DK, int DV>
__global__ void LinearAttentionRecurrentKernelFixedShape(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    float* __restrict__ state,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ output,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval) {
  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int tid = threadIdx.x;
  const int heads_per_group = q_num_heads / kv_num_heads;

  float* S_global = state + ((int64_t)b * kv_num_heads + h_kv) * DK * DV;

  extern __shared__ float smem[];
  float* S_smem = smem;               // [DK * DV]
  float* s_scratch = smem + DK * DV;  // [max(DK, DV)]

  for (int idx = tid; idx < DK * DV; idx += blockDim.x) {
    S_smem[idx] = S_global[idx];
  }
  __syncthreads();

  for (int t = 0; t < seq_len; ++t) {
    float kt_val = 0.0f;
    if (tid < DK) {
      int k_offset = ((int64_t)b * seq_len + t) * (kv_num_heads * DK) + h_kv * DK + tid;
      kt_val = to_float(key[k_offset]);
    }

    if (needs_decay) {
      if (tid < DK) {
        float exp_g;
        if (decay_per_key_dim) {
          int g_offset = ((int64_t)b * seq_len + t) * (kv_num_heads * DK) + h_kv * DK + tid;
          exp_g = expf(to_float(decay[g_offset]));
        } else {
          int g_offset = ((int64_t)b * seq_len + t) * kv_num_heads + h_kv;
          exp_g = expf(to_float(decay[g_offset]));
        }
#pragma unroll
        for (int j = 0; j < DV; ++j) {
          S_smem[tid * DV + j] *= exp_g;
        }
      }
      __syncthreads();
    }

    if (needs_retrieval) {
      if (tid < DK) {
        s_scratch[tid] = kt_val;
      }
      __syncthreads();

      if (tid < DV) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < DK; ++i) {
          acc += S_smem[i * DV + tid] * s_scratch[i];
        }
        s_scratch[tid] = acc;
      }
      __syncthreads();
    }

    if (needs_beta) {
      float bt;
      if (beta_per_head) {
        bt = to_float(beta_in[((int64_t)b * seq_len + t) * kv_num_heads + h_kv]);
      } else {
        bt = to_float(beta_in[((int64_t)b * seq_len + t)]);
      }

      if (tid < DV) {
        int v_base = ((int64_t)b * seq_len + t) * (kv_num_heads * DV) + h_kv * DV;
        float vj = to_float(value[v_base + tid]);
        s_scratch[tid] = bt * (vj - s_scratch[tid]);
      }
      __syncthreads();

      if (tid < DK) {
#pragma unroll
        for (int j = 0; j < DV; ++j) {
          S_smem[tid * DV + j] += kt_val * s_scratch[j];
        }
      }
    } else {
      if (tid < DV) {
        int v_base = ((int64_t)b * seq_len + t) * (kv_num_heads * DV) + h_kv * DV;
        s_scratch[tid] = to_float(value[v_base + tid]);
      }
      __syncthreads();

      if (tid < DK) {
#pragma unroll
        for (int j = 0; j < DV; ++j) {
          S_smem[tid * DV + j] += kt_val * s_scratch[j];
        }
      }
    }
    __syncthreads();

    for (int g = 0; g < heads_per_group; ++g) {
      if (g > 0) {
        __syncthreads();
      }

      int h_q = h_kv * heads_per_group + g;
      if (tid < DK) {
        int q_offset = ((int64_t)b * seq_len + t) * (q_num_heads * DK) + h_q * DK + tid;
        s_scratch[tid] = to_float(query[q_offset]);
      }
      __syncthreads();

      if (tid < DV) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < DK; ++i) {
          acc += S_smem[i * DV + tid] * s_scratch[i];
        }
        int out_offset = ((int64_t)b * seq_len + t) * (q_num_heads * DV) + h_q * DV + tid;
        output[out_offset] = from_float<T>(scale * acc);
      }
    }

    __syncthreads();
  }

  for (int idx = tid; idx < DK * DV; idx += blockDim.x) {
    S_global[idx] = S_smem[idx];
  }
}

}  // anonymous namespace

template <typename T>
Status LaunchLinearAttentionKernel(
    cudaStream_t stream,
    const T* query,
    const T* key,
    const T* value,
    const T* decay,
    const T* beta,
    T* output,
    float* present_state,
    int batch_size,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int d_k,
    int d_v,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval,
    int max_threads_per_block) {
  // Grid: one block per (batch, kv_head)
  const dim3 grid(batch_size, kv_num_heads, 1);

  auto launch_fixed_shape = [&](int dim) -> Status {
    const int scratch_elems = dim;
    const size_t fixed_smem_size = (static_cast<size_t>(dim) * dim + scratch_elems) * sizeof(float);
    const dim3 fixed_block(dim, 1, 1);

    if (fixed_smem_size > 48 * 1024) {
      cudaError_t attr_err;
      if (dim == 64) {
        attr_err = cudaFuncSetAttribute(
            LinearAttentionRecurrentKernelFixedShape<T, 64, 64>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(fixed_smem_size));
      } else {
        attr_err = cudaFuncSetAttribute(
            LinearAttentionRecurrentKernelFixedShape<T, 128, 128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(fixed_smem_size));
      }
      if (attr_err != cudaSuccess) {
        return CUDA_CALL(attr_err);
      }
    }

    if (dim == 64) {
      LinearAttentionRecurrentKernelFixedShape<T, 64, 64><<<grid, fixed_block, fixed_smem_size, stream>>>(
          query, key, value, present_state, decay, beta, output,
          seq_len, q_num_heads, kv_num_heads, scale,
          needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval);
    } else {
      LinearAttentionRecurrentKernelFixedShape<T, 128, 128><<<grid, fixed_block, fixed_smem_size, stream>>>(
          query, key, value, present_state, decay, beta, output,
          seq_len, q_num_heads, kv_num_heads, scale,
          needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval);
    }

    return CUDA_CALL(cudaGetLastError());
  };

  // Fast paths for common square dims in decoder workloads.
  if (d_k == d_v && d_k == 64 && max_threads_per_block >= 64) {
    return launch_fixed_shape(64);
  }
  if (d_k == d_v && d_k == 128 && max_threads_per_block >= 128) {
    return launch_fixed_shape(128);
  }

  // Block: max(d_k, d_v) threads, rounded up to warp boundary
  int threads = (d_k > d_v) ? d_k : d_v;
  threads = ((threads + 31) / 32) * 32;
  if (threads > max_threads_per_block) {
    threads = max_threads_per_block;
  }
  const dim3 block(threads, 1, 1);

  // Shared memory: state[d_k*d_v] + scratch[max(d_k,d_v)]
  int scratch_elems = (d_k > d_v) ? d_k : d_v;
  size_t smem_size = (static_cast<size_t>(d_k) * d_v + scratch_elems) * sizeof(float);

  // Request extended shared memory if needed (default limit is 48 KB)
  if (smem_size > 48 * 1024) {
    cudaError_t attr_err = cudaFuncSetAttribute(
        LinearAttentionRecurrentKernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size));
    if (attr_err != cudaSuccess) {
      return CUDA_CALL(attr_err);
    }
  }

  LinearAttentionRecurrentKernel<T><<<grid, block, smem_size, stream>>>(
      query, key, value, present_state, decay, beta, output,
      seq_len, q_num_heads, kv_num_heads, d_k, d_v, scale,
      needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchLinearAttentionKernel<float>(
    cudaStream_t, const float*, const float*, const float*,
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);

template Status LaunchLinearAttentionKernel<half>(
    cudaStream_t, const half*, const half*, const half*,
    const half*, const half*, half*, float*,
    int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchLinearAttentionKernel<__nv_bfloat16>(
    cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*,
    int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
