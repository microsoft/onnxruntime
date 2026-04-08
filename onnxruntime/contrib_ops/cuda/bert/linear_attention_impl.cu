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
#include <type_traits>
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
//   float S_smem[d_k * d_v]              — recurrent state matrix (fp32)
//   float s_scratch[max(d_k, d_v)]       — broadcast/retrieval/delta buffer
//
// State is stored as type T in global memory but computed in fp32 in shared
// memory for numerical stability.
// =============================================================================
template <typename T>
__global__ void LinearAttentionRecurrentKernel(
    const T* __restrict__ query,    // [B, T, H_q * d_k]
    const T* __restrict__ key,      // [B, T, n_k * d_k]
    const T* __restrict__ value,    // [B, T, H_kv * d_v]
    T* __restrict__ state,          // [B, H_kv, d_k, d_v] — in-place updated
    const T* __restrict__ decay,    // [B, T, H_kv] or [B, T, H_kv*d_k] or nullptr
    const T* __restrict__ beta_in,  // [B, T, H_kv] or [B, T, 1] or nullptr
    T* __restrict__ output,         // [B, T, max(H_q, H_kv) * d_v]
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int n_k_heads,
    int d_k,
    int d_v,
    int output_hidden,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval) {
  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;
  const int kv_per_k = kv_num_heads / n_k_heads;
  const int h_k = h_kv / kv_per_k;

  // Global state pointer for this (batch, head): [d_k, d_v]
  T* S_global = state + ((int64_t)b * kv_num_heads + h_kv) * d_k * d_v;

  // Shared memory layout
  extern __shared__ float smem[];
  float* S_smem = smem;                       // [d_k * d_v]
  float* k_buf = smem + d_k * d_v;            // [d_k]
  float* s_scratch = smem + d_k * d_v + d_k;  // [max(d_k, d_v)]

  // Load state from global memory (type T) into shared memory (fp32)
  for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
    S_smem[idx] = to_float(S_global[idx]);
  }
  __syncthreads();

  // ---- Token loop ----
  for (int t = 0; t < seq_len; ++t) {
    // Load k_t[tid] into register (each thread loads one element)
    float kt_val = 0.0f;
    if (tid < d_k) {
      int k_offset = ((int64_t)b * seq_len + t) * (n_k_heads * d_k) + h_k * d_k + tid;
      kt_val = to_float(key[k_offset]);
    }

    // Steps 1+2: Decay + Retrieval (fused for scalar per-head decay)
    bool fused_decay_update = false;
    float fused_exp_g = 1.0f;

    if (needs_decay && needs_retrieval && !decay_per_key_dim) {
      if (tid < d_k) {
        k_buf[tid] = kt_val;
      }
      if (tid == 0) {
        int g_offset = ((int64_t)b * seq_len + t) * kv_num_heads + h_kv;
        s_scratch[0] = expf(to_float(decay[g_offset]));
      }
      __syncthreads();

      fused_exp_g = s_scratch[0];

      if (tid < d_v) {
        float acc = 0.0f;
        for (int i = 0; i < d_k; ++i) {
          acc += S_smem[i * d_v + tid] * k_buf[i];
        }
        s_scratch[tid] = fused_exp_g * acc;
      }
      __syncthreads();

      fused_decay_update = true;

    } else {
      // Non-fused path: separate decay and retrieval steps
      if (needs_decay) {
        if (!decay_per_key_dim) {
          if (tid == 0) {
            int g_offset = ((int64_t)b * seq_len + t) * kv_num_heads + h_kv;
            s_scratch[0] = expf(to_float(decay[g_offset]));
          }
          __syncthreads();
        }
        if (tid < d_k) {
          float exp_g;
          if (decay_per_key_dim) {
            int g_offset = ((int64_t)b * seq_len + t) * (kv_num_heads * d_k) + h_kv * d_k + tid;
            exp_g = expf(to_float(decay[g_offset]));
          } else {
            exp_g = s_scratch[0];
          }
          for (int j = 0; j < d_v; ++j) {
            S_smem[tid * d_v + j] *= exp_g;
          }
        }
        __syncthreads();
      }

      if (needs_retrieval) {
        // Store k in k_buf (not s_scratch) to avoid inter-warp race when
        // d_k > 32: retrieval overwrites s_scratch[tid] while other warps
        // may still be reading s_scratch[i] in the dot product loop.
        if (tid < d_k) {
          k_buf[tid] = kt_val;
        }
        __syncthreads();

        if (tid < d_v) {
          float acc = 0.0f;
          for (int i = 0; i < d_k; ++i) {
            acc += S_smem[i * d_v + tid] * k_buf[i];
          }
          s_scratch[tid] = acc;
        }
        __syncthreads();
      }
    }

    // Step 3: State update — S += k_t ⊗ delta (or k_t ⊗ v_t for linear)
    // When fused_decay_update, applies: S = exp_g * S + k * delta
    if (needs_beta) {
      float bt;
      if (beta_per_head) {
        bt = to_float(beta_in[((int64_t)b * seq_len + t) * kv_num_heads + h_kv]);
      } else {
        bt = to_float(beta_in[((int64_t)b * seq_len + t)]);
      }

      if (tid < d_v) {
        int v_base = ((int64_t)b * seq_len + t) * (kv_num_heads * d_v) + h_kv * d_v;
        float vj = to_float(value[v_base + tid]);
        s_scratch[tid] = bt * (vj - s_scratch[tid]);
      }
      __syncthreads();

      if (tid < d_k) {
        if (fused_decay_update) {
          for (int j = 0; j < d_v; ++j) {
            S_smem[tid * d_v + j] = fused_exp_g * S_smem[tid * d_v + j] + kt_val * s_scratch[j];
          }
        } else {
          for (int j = 0; j < d_v; ++j) {
            S_smem[tid * d_v + j] += kt_val * s_scratch[j];
          }
        }
      }
    } else {
      if (tid < d_v) {
        int v_base = ((int64_t)b * seq_len + t) * (kv_num_heads * d_v) + h_kv * d_v;
        s_scratch[tid] = to_float(value[v_base + tid]);
      }
      __syncthreads();

      if (tid < d_k) {
        if (fused_decay_update) {
          for (int j = 0; j < d_v; ++j) {
            S_smem[tid * d_v + j] = fused_exp_g * S_smem[tid * d_v + j] + kt_val * s_scratch[j];
          }
        } else {
          for (int j = 0; j < d_v; ++j) {
            S_smem[tid * d_v + j] += kt_val * s_scratch[j];
          }
        }
      }
    }
    __syncthreads();

    // Step 4: Query readout — output = S^T @ q_t (standard GQA or inverse GQA)
    if (q_num_heads >= kv_num_heads) {
      int heads_per_group = q_num_heads / kv_num_heads;
      for (int g = 0; g < heads_per_group; ++g) {
        if (g > 0) {
          __syncthreads();
        }

        int h_q = h_kv * heads_per_group + g;
        if (tid < d_k) {
          int q_offset = ((int64_t)b * seq_len + t) * (q_num_heads * d_k) + h_q * d_k + tid;
          s_scratch[tid] = to_float(query[q_offset]);
        }
        __syncthreads();

        if (tid < d_v) {
          float acc = 0.0f;
          for (int i = 0; i < d_k; ++i) {
            acc += S_smem[i * d_v + tid] * s_scratch[i];
          }
          int out_offset = ((int64_t)b * seq_len + t) * output_hidden + h_q * d_v + tid;
          output[out_offset] = from_float<T>(scale * acc);
        }
      }
    } else {
      int h_q = h_kv * q_num_heads / kv_num_heads;
      if (tid < d_k) {
        int q_offset = ((int64_t)b * seq_len + t) * (q_num_heads * d_k) + h_q * d_k + tid;
        s_scratch[tid] = to_float(query[q_offset]);
      }
      __syncthreads();

      if (tid < d_v) {
        float acc = 0.0f;
        for (int i = 0; i < d_k; ++i) {
          acc += S_smem[i * d_v + tid] * s_scratch[i];
        }
        int out_offset = ((int64_t)b * seq_len + t) * output_hidden + h_kv * d_v + tid;
        output[out_offset] = from_float<T>(scale * acc);
      }
    }

    __syncthreads();
  }

  // Write back state from shared memory (fp32) to global memory (type T)
  for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
    S_global[idx] = from_float<T>(S_smem[idx]);
  }
}

// Compile-time specialized variant for common (d_k, d_v) pairs.
// Optimizations over the generic kernel:
//   1. #pragma unroll on all inner loops for better ILP
//   2. float4 vectorized row operations (decay, state update) — 4x fewer shared memory transactions
//   3. Fused decay+retrieval for scalar per-head decay — eliminates one state pass and one __syncthreads()
//   4. Dedicated k_buf in shared memory avoids scratch aliasing during fused path
template <typename T, int DK, int DV>
__global__ void LinearAttentionRecurrentKernelFixedShape(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ state,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ output,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int n_k_heads,
    int output_hidden,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval) {
  static_assert(DV % 4 == 0 && DK % 4 == 0, "DK and DV must be multiples of 4 for float4 optimization");
  constexpr int DV4 = DV / 4;

  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int tid = threadIdx.x;
  const int kv_per_k = kv_num_heads / n_k_heads;
  const int h_k = h_kv / kv_per_k;

  T* S_global = state + ((int64_t)b * kv_num_heads + h_kv) * DK * DV;

  // Shared memory layout:
  //   S_smem[DK * DV]             — recurrent state matrix (fp32)
  //   k_buf[DK]                   — persistent key broadcast buffer
  //   s_scratch[max(DK, DV)]      — general scratch (retrieval, delta, query broadcast)
  extern __shared__ float smem[];
  float* S_smem = smem;                    // [DK * DV]
  float* k_buf = smem + DK * DV;           // [DK]
  float* s_scratch = smem + DK * DV + DK;  // [max(DK, DV)]

  // Load state from global memory (type T) into shared memory (fp32) — vectorized
  if constexpr (sizeof(T) == 2 && DV % 2 == 0) {
    // half/bf16: load 2 elements at a time via uint32
    const uint32_t* S_global_u32 = reinterpret_cast<const uint32_t*>(S_global);
    int half_pairs = (DK * DV) / 2;
    for (int idx = tid; idx < half_pairs; idx += blockDim.x) {
      uint32_t packed = S_global_u32[idx];
      T lo, hi;
      memcpy(&lo, &packed, sizeof(T));
      memcpy(&hi, reinterpret_cast<const char*>(&packed) + sizeof(T), sizeof(T));
      S_smem[idx * 2] = to_float(lo);
      S_smem[idx * 2 + 1] = to_float(hi);
    }
  } else {
    for (int idx = tid; idx < DK * DV; idx += blockDim.x) {
      S_smem[idx] = to_float(S_global[idx]);
    }
  }
  __syncthreads();

  // Precompute per-batch strides to avoid repeated int64 multiplications in the token loop
  const int64_t b_seq = (int64_t)b * seq_len;
  const int k_hidden = n_k_heads * DK;
  const int kv_v_hidden = kv_num_heads * DV;
  const int q_hidden = q_num_heads * DK;
  const int kv_dk_hidden = kv_num_heads * DK;

  for (int t = 0; t < seq_len; ++t) {
    const int64_t bt = b_seq + t;

    float kt_val = 0.0f;
    if (tid < DK) {
      kt_val = to_float(key[bt * k_hidden + h_k * DK + tid]);
    }

    // ==================================================================
    // Steps 1+2: Decay + Retrieval
    // ==================================================================
    // For the fused scalar-decay + gated_delta path, we also fuse the
    // state update (step 3) to avoid a separate decay pass entirely:
    //   retrieval = exp_g * (S^T @ k)       [on old S]
    //   delta = beta * (v - retrieval)
    //   S = exp_g * S + k ⊗ delta           [single fused pass]
    // This reduces 3 state passes (decay, retrieval, update) to 2 (retrieval, fused update).
    bool fused_decay_update = false;
    float fused_exp_g = 1.0f;

    if (needs_decay && needs_retrieval && !decay_per_key_dim) {
      // --- FUSED path: scalar per-head decay + retrieval ---
      if (tid < DK) {
        k_buf[tid] = kt_val;
      }
      if (tid == 0) {
        s_scratch[0] = expf(to_float(decay[bt * kv_num_heads + h_kv]));
      }
      __syncthreads();

      fused_exp_g = s_scratch[0];

      // Retrieval on old state, pre-scaled by exp_g
      if (tid < DV) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < DK; ++i) {
          acc += S_smem[i * DV + tid] * k_buf[i];
        }
        s_scratch[tid] = fused_exp_g * acc;
      }
      __syncthreads();

      // Decay is deferred to the update step (fused_decay_update = true)
      fused_decay_update = true;

    } else if (needs_decay && needs_retrieval) {
      // --- Per-key-dim decay then retrieval (cannot fuse — exp_g differs per row) ---
      if (tid < DK) {
        k_buf[tid] = kt_val;
        float exp_g = expf(to_float(decay[bt * kv_dk_hidden + h_kv * DK + tid]));
        float4* row = reinterpret_cast<float4*>(S_smem + tid * DV);
#pragma unroll
        for (int j = 0; j < DV4; ++j) {
          float4 v = row[j];
          v.x *= exp_g;
          v.y *= exp_g;
          v.z *= exp_g;
          v.w *= exp_g;
          row[j] = v;
        }
      }
      __syncthreads();  // decay done, k_buf visible

      if (tid < DV) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < DK; ++i) {
          acc += S_smem[i * DV + tid] * k_buf[i];
        }
        s_scratch[tid] = acc;
      }
      __syncthreads();  // retrieval done

    } else {
      // --- Decay only, retrieval only, or neither ---
      if (needs_decay) {
        if (!decay_per_key_dim) {
          if (tid == 0) {
            s_scratch[0] = expf(to_float(decay[bt * kv_num_heads + h_kv]));
          }
          __syncthreads();
        }
        if (tid < DK) {
          float exp_g;
          if (decay_per_key_dim) {
            exp_g = expf(to_float(decay[bt * kv_dk_hidden + h_kv * DK + tid]));
          } else {
            exp_g = s_scratch[0];
          }
          float4* row = reinterpret_cast<float4*>(S_smem + tid * DV);
#pragma unroll
          for (int j = 0; j < DV4; ++j) {
            float4 v = row[j];
            v.x *= exp_g;
            v.y *= exp_g;
            v.z *= exp_g;
            v.w *= exp_g;
            row[j] = v;
          }
        }
        __syncthreads();
      }

      if (needs_retrieval) {
        if (tid < DK) {
          k_buf[tid] = kt_val;
        }
        __syncthreads();  // k_buf visible

        if (tid < DV) {
          float acc = 0.0f;
#pragma unroll
          for (int i = 0; i < DK; ++i) {
            acc += S_smem[i * DV + tid] * k_buf[i];
          }
          s_scratch[tid] = acc;
        }
        __syncthreads();  // retrieval done
      }
    }

    // ==================================================================
    // Step 3: State update with float4 vectorization
    // When fused_decay_update is true, decay is applied here:
    //   S[i,j] = exp_g * S[i,j] + k[i] * delta[j]
    // ==================================================================
    if (needs_beta) {
      float beta_t;
      if (beta_per_head) {
        beta_t = to_float(beta_in[bt * kv_num_heads + h_kv]);
      } else {
        beta_t = to_float(beta_in[bt]);
      }

      if (tid < DV) {
        float vj = to_float(value[bt * kv_v_hidden + h_kv * DV + tid]);
        s_scratch[tid] = beta_t * (vj - s_scratch[tid]);
      }
      __syncthreads();

      if (tid < DK) {
        float4* row = reinterpret_cast<float4*>(S_smem + tid * DV);
        const float4* delta4 = reinterpret_cast<const float4*>(s_scratch);
        if (fused_decay_update) {
          // Fused: S = exp_g * S + k * delta (single pass, no separate decay)
#pragma unroll
          for (int j = 0; j < DV4; ++j) {
            float4 s = row[j];
            float4 d = delta4[j];
            s.x = fused_exp_g * s.x + kt_val * d.x;
            s.y = fused_exp_g * s.y + kt_val * d.y;
            s.z = fused_exp_g * s.z + kt_val * d.z;
            s.w = fused_exp_g * s.w + kt_val * d.w;
            row[j] = s;
          }
        } else {
#pragma unroll
          for (int j = 0; j < DV4; ++j) {
            float4 s = row[j];
            float4 d = delta4[j];
            s.x += kt_val * d.x;
            s.y += kt_val * d.y;
            s.z += kt_val * d.z;
            s.w += kt_val * d.w;
            row[j] = s;
          }
        }
      }
    } else {
      if (tid < DV) {
        s_scratch[tid] = to_float(value[bt * kv_v_hidden + h_kv * DV + tid]);
      }
      __syncthreads();

      if (tid < DK) {
        float4* row = reinterpret_cast<float4*>(S_smem + tid * DV);
        const float4* v4 = reinterpret_cast<const float4*>(s_scratch);
        if (fused_decay_update) {
#pragma unroll
          for (int j = 0; j < DV4; ++j) {
            float4 s = row[j];
            float4 v = v4[j];
            s.x = fused_exp_g * s.x + kt_val * v.x;
            s.y = fused_exp_g * s.y + kt_val * v.y;
            s.z = fused_exp_g * s.z + kt_val * v.z;
            s.w = fused_exp_g * s.w + kt_val * v.w;
            row[j] = s;
          }
        } else {
#pragma unroll
          for (int j = 0; j < DV4; ++j) {
            float4 s = row[j];
            float4 v = v4[j];
            s.x += kt_val * v.x;
            s.y += kt_val * v.y;
            s.z += kt_val * v.z;
            s.w += kt_val * v.w;
            row[j] = s;
          }
        }
      }
    }
    __syncthreads();

    // ==================================================================
    // Step 4: Query readout (column dot products — not float4-vectorizable)
    // ==================================================================
    if (q_num_heads >= kv_num_heads) {
      int heads_per_group = q_num_heads / kv_num_heads;
      for (int g = 0; g < heads_per_group; ++g) {
        if (g > 0) {
          __syncthreads();
        }

        int h_q = h_kv * heads_per_group + g;
        if (tid < DK) {
          s_scratch[tid] = to_float(query[bt * q_hidden + h_q * DK + tid]);
        }
        __syncthreads();

        if (tid < DV) {
          float acc = 0.0f;
#pragma unroll
          for (int i = 0; i < DK; ++i) {
            acc += S_smem[i * DV + tid] * s_scratch[i];
          }
          output[bt * output_hidden + h_q * DV + tid] = from_float<T>(scale * acc);
        }
      }
    } else {
      int h_q = h_kv * q_num_heads / kv_num_heads;
      if (tid < DK) {
        s_scratch[tid] = to_float(query[bt * q_hidden + h_q * DK + tid]);
      }
      __syncthreads();

      if (tid < DV) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < DK; ++i) {
          acc += S_smem[i * DV + tid] * s_scratch[i];
        }
        output[bt * output_hidden + h_kv * DV + tid] = from_float<T>(scale * acc);
      }
    }

    __syncthreads();
  }

  // Write back state from shared memory (fp32) to global memory (type T) — vectorized
  if constexpr (sizeof(T) == 2 && DV % 2 == 0) {
    uint32_t* S_global_u32 = reinterpret_cast<uint32_t*>(S_global);
    int half_pairs = (DK * DV) / 2;
    for (int idx = tid; idx < half_pairs; idx += blockDim.x) {
      T lo = from_float<T>(S_smem[idx * 2]);
      T hi = from_float<T>(S_smem[idx * 2 + 1]);
      uint32_t packed;
      memcpy(&packed, &lo, sizeof(T));
      memcpy(reinterpret_cast<char*>(&packed) + sizeof(T), &hi, sizeof(T));
      S_global_u32[idx] = packed;
    }
  } else if constexpr (sizeof(T) == 4 && DV % 4 == 0) {
    float4* S_global_f4 = reinterpret_cast<float4*>(S_global);
    int quads = (DK * DV) / 4;
    for (int idx = tid; idx < quads; idx += blockDim.x) {
      float4 v;
      v.x = S_smem[idx * 4];
      v.y = S_smem[idx * 4 + 1];
      v.z = S_smem[idx * 4 + 2];
      v.w = S_smem[idx * 4 + 3];
      S_global_f4[idx] = v;
    }
  } else {
    for (int idx = tid; idx < DK * DV; idx += blockDim.x) {
      S_global[idx] = from_float<T>(S_smem[idx]);
    }
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
    T* present_state,
    int batch_size,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int n_k_heads,
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

  int output_hidden = std::max(q_num_heads, kv_num_heads) * d_v;

  auto launch_fixed = [&](auto dk_tag, auto dv_tag) -> Status {
    constexpr int DK = decltype(dk_tag)::value;
    constexpr int DV = decltype(dv_tag)::value;
    constexpr int max_dim = (DK > DV) ? DK : DV;
    // Layout: S_smem[DK*DV] + k_buf[DK] + s_scratch[max(DK,DV)]
    const size_t fixed_smem_size = (static_cast<size_t>(DK) * DV + DK + max_dim) * sizeof(float);
    const dim3 fixed_block(max_dim, 1, 1);

    if (fixed_smem_size > 48 * 1024) {
      cudaError_t attr_err = cudaFuncSetAttribute(
          LinearAttentionRecurrentKernelFixedShape<T, DK, DV>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(fixed_smem_size));
      if (attr_err != cudaSuccess) {
        return CUDA_CALL(attr_err);
      }
    }

    LinearAttentionRecurrentKernelFixedShape<T, DK, DV><<<grid, fixed_block, fixed_smem_size, stream>>>(
        query, key, value, present_state, decay, beta, output,
        seq_len, q_num_heads, kv_num_heads, n_k_heads, output_hidden, scale,
        needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval);

    return CUDA_CALL(cudaGetLastError());
  };

  // Fast paths for common (d_k, d_v) pairs
  if (d_k == 64 && d_v == 64 && max_threads_per_block >= 64) {
    return launch_fixed(std::integral_constant<int, 64>{}, std::integral_constant<int, 64>{});
  }
  if (d_k == 128 && d_v == 128 && max_threads_per_block >= 128) {
    return launch_fixed(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{});
  }
  if (d_k == 128 && d_v == 64 && max_threads_per_block >= 128) {
    return launch_fixed(std::integral_constant<int, 128>{}, std::integral_constant<int, 64>{});
  }
  if (d_k == 64 && d_v == 128 && max_threads_per_block >= 128) {
    return launch_fixed(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{});
  }

  // Generic fallback
  // Block: max(d_k, d_v) threads, rounded up to warp boundary
  int threads = ((std::max(d_k, d_v) + 31) / 32) * 32;
  if (threads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "LinearAttention: max(d_k=", d_k, ", d_v=", d_v,
                           ") exceeds max threads per block (", max_threads_per_block,
                           "). Use a model with smaller head dimensions.");
  }
  const dim3 block(threads, 1, 1);

  // Shared memory: state[d_k*d_v] + k_buf[d_k] + scratch[max(d_k,d_v)]
  size_t smem_size = (static_cast<size_t>(d_k) * d_v + d_k + std::max(d_k, d_v)) * sizeof(float);

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
      seq_len, q_num_heads, kv_num_heads, n_k_heads, d_k, d_v, output_hidden, scale,
      needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchLinearAttentionKernel<float>(
    cudaStream_t, const float*, const float*, const float*,
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);

template Status LaunchLinearAttentionKernel<half>(
    cudaStream_t, const half*, const half*, const half*,
    const half*, const half*, half*, half*,
    int, int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchLinearAttentionKernel<__nv_bfloat16>(
    cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
