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
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
// Full-warp sum reduction (result broadcast to all lanes).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += onnxruntime::cuda::WARP_SHFL_XOR(v, offset);
  }
  return v;
}

template <typename T>
__device__ __forceinline__ float ComputeLinearAttentionDeltaColumn(
    const T* value,
    const T* beta_in,
    int64_t batch_token_offset,
    int value_hidden,
    int kv_head,
    int d_v,
    int col,
    int kv_num_heads,
    bool needs_beta,
    bool beta_per_head,
    float retrieval_col) {
  const float value_col = to_float(value[batch_token_offset * value_hidden + kv_head * d_v + col]);
  if (!needs_beta) {
    return value_col;
  }

  const float beta_value = beta_per_head ? to_float(beta_in[batch_token_offset * kv_num_heads + kv_head])
                                         : to_float(beta_in[batch_token_offset]);
  return beta_value * (value_col - retrieval_col);
}

struct LinearAttentionReadoutHeads {
  int query_head;
  int output_head;
};

__device__ __forceinline__ int GetLinearAttentionReadoutHeadCount(int q_num_heads, int kv_num_heads) {
  return q_num_heads >= kv_num_heads ? q_num_heads / kv_num_heads : 1;
}

__device__ __forceinline__ LinearAttentionReadoutHeads GetLinearAttentionReadoutHeads(
    int kv_head,
    int q_num_heads,
    int kv_num_heads,
    int group_index) {
  if (q_num_heads >= kv_num_heads) {
    const int heads_per_group = q_num_heads / kv_num_heads;
    const int query_head = kv_head * heads_per_group + group_index;
    return {query_head, query_head};
  }

  return {kv_head * q_num_heads / kv_num_heads, kv_head};
}

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
    const T* past_state,            // [W, B, H_kv, d_k, d_v] -- may alias present_state
    T* present_state,               // [W, B, H_kv, d_k, d_v]
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
    bool needs_retrieval,
    int batch_size,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;
  const int kv_per_k = kv_num_heads / n_k_heads;
  const int h_k = h_kv / kv_per_k;

  // Global state pointers for this (batch, head): [d_k, d_v] within window slot W-1, the state
  // after the last token. They may alias exactly. Window-major layout means one slot spans the
  // whole batch, so slot W-1 is a single contiguous [B, H_kv, d_k, d_v] block.
  const int64_t slot_stride = (int64_t)kv_num_heads * d_k * d_v;
  const int64_t state_offset = ((int64_t)(state_window - 1) * batch_size + b) * slot_stride +
                               (int64_t)h_kv * d_k * d_v;
  const T* S_past = past_state + state_offset;
  T* S_present = present_state + state_offset;

  // Shared memory layout
  extern __shared__ float smem[];
  float* S_smem = smem;                       // [d_k * d_v]
  float* k_buf = smem + d_k * d_v;            // [d_k]
  float* s_scratch = smem + d_k * d_v + d_k;  // [max(d_k, d_v)]

  // Load state from global memory (type T) into shared memory (fp32)
  for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
    S_smem[idx] = to_float(S_past[idx]);
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

    // Emit the recurrent state AFTER processing token t into the right-aligned window slot
    // t + W - seq_len (negative => this position falls outside the window and is dropped).
    // The last token's slot is always W-1 and is written by the (vectorized) epilogue below.
    // Layout [W, B, H_kv, d_k, d_v] row-major; element (i, j) at base_t + i*d_v + j.
    const int state_slot = t + state_window - seq_len;
    if (state_slot >= 0 && t + 1 < seq_len) {
      const int64_t base_t = ((int64_t)state_slot * batch_size + b) * slot_stride +
                             (int64_t)h_kv * d_k * d_v;
      for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
        present_state[base_t + idx] = from_float<T>(S_smem[idx]);
      }
    }

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
    S_present[idx] = from_float<T>(S_smem[idx]);
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
    const T* past_state,
    T* present_state,
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
    bool needs_retrieval,
    int batch_size,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  static_assert(DV % 4 == 0 && DK % 4 == 0, "DK and DV must be multiples of 4 for float4 optimization");
  constexpr int DV4 = DV / 4;

  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int tid = threadIdx.x;
  const int kv_per_k = kv_num_heads / n_k_heads;
  const int h_k = h_kv / kv_per_k;

  // Window slot W-1 holds the state after the last token; that is what past_state is read from
  // and what the epilogue writes. Window-major, so a slot spans the whole batch.
  const int64_t slot_stride = (int64_t)kv_num_heads * DK * DV;
  const int64_t state_offset = ((int64_t)(state_window - 1) * batch_size + b) * slot_stride +
                               (int64_t)h_kv * DK * DV;
  const T* S_past = past_state + state_offset;
  T* S_present = present_state + state_offset;

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
    const uint32_t* S_global_u32 = reinterpret_cast<const uint32_t*>(S_past);
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
      S_smem[idx] = to_float(S_past[idx]);
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

    // Emit the recurrent state AFTER processing token t into the right-aligned window slot
    // t + W - seq_len (negative => outside the window, dropped). The last token's slot is always
    // W-1 and is written by the vectorized epilogue below.
    // Layout [W, B, H_kv, DK, DV] row-major; element (i, j) at base_t + i*DV + j.
    const int state_slot = t + state_window - seq_len;
    if (state_slot >= 0 && t + 1 < seq_len) {
      const int64_t base_t = ((int64_t)state_slot * batch_size + b) * slot_stride +
                             (int64_t)h_kv * DK * DV;
      for (int idx = tid; idx < DK * DV; idx += blockDim.x) {
        present_state[base_t + idx] = from_float<T>(S_smem[idx]);
      }
    }

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
    uint32_t* S_global_u32 = reinterpret_cast<uint32_t*>(S_present);
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
    float4* S_global_f4 = reinterpret_cast<float4*>(S_present);
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
      S_present[idx] = from_float<T>(S_smem[idx]);
    }
  }
}

// =============================================================================
// Decode-optimized kernel (small seq_len).
//
// Unlike the recurrent kernels above (one block per head, state cached in shared
// memory across the token loop, block-wide __syncthreads barriers), this kernel
// parallelizes the d_v output columns across many warps/blocks so the whole GPU
// is saturated at decode (batch=1). This v1 warp-per-column kernel has no
// block-wide barriers, only warp shuffles. It mirrors the flash-linear-attention
// / llama.cpp decode design and is far better for the latency-bound
// seq_len<=few case where the shared-memory state caching of the recurrent
// kernels yields no amortization.
//
// Grid:  (batch_size, kv_num_heads, ceil(d_v / kWarpsPerBlock))
// Block: (32, kWarpsPerBlock)
// Each warp owns exactly one output column `col`. The recurrent state column
// S[:, col] is sharded into registers across the warp's 32 lanes (DK/32 rows per
// lane). Matrix-vector reductions (S^T@k, S^T@q) use warp_reduce_sum.
//
// Requires DK % 32 == 0. State global layout is unchanged (row-major [DK, d_v]),
// so the present_state is bit-compatible with the recurrent kernels above and
// can be freely interchanged between prefill and decode steps.
// =============================================================================
constexpr int kWarpsPerBlock = 4;

template <typename T, int DK>
__global__ void LinearAttentionDecodeKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* past_state,
    T* present_state,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ output,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int n_k_heads,
    int d_v,
    int output_hidden,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval,
    int batch_size,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  static_assert(DK % 32 == 0, "DK must be a multiple of warp size (32)");
  constexpr int ROWS = DK / 32;

  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int lane = threadIdx.x;
  const int col = blockIdx.z * kWarpsPerBlock + threadIdx.y;
  if (col >= d_v) {
    return;
  }

  const int kv_per_k = kv_num_heads / n_k_heads;
  const int h_k = h_kv / kv_per_k;

  // State column S[:, col] sharded into registers: lane holds rows {r*32 + lane}.
  // Window slot W-1 holds the state after the last token; window-major, so a slot spans the batch.
  const int64_t slot_stride = (int64_t)kv_num_heads * DK * d_v;
  const int64_t state_offset = ((int64_t)(state_window - 1) * batch_size + b) * slot_stride +
                               (int64_t)h_kv * DK * d_v + col;
  const T* S_past_col = past_state + state_offset;
  T* S_present_col = present_state + state_offset;
  float s_shard[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    s_shard[r] = to_float(S_past_col[(int64_t)(r * 32 + lane) * d_v]);
  }

  const int k_hidden = n_k_heads * DK;
  const int q_hidden = q_num_heads * DK;
  const int v_hidden = kv_num_heads * d_v;

  for (int t = 0; t < seq_len; ++t) {
    const int64_t bt = (int64_t)b * seq_len + t;

    // Load this lane's key shard.
    float k_reg[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      k_reg[r] = to_float(key[bt * k_hidden + h_k * DK + (r * 32 + lane)]);
    }

    // Per-row decay multipliers (scalar broadcast or per-key-dim).
    float exp_g[ROWS];
    if (needs_decay) {
      if (decay_per_key_dim) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          exp_g[r] = expf(to_float(decay[bt * (kv_num_heads * DK) + h_kv * DK + (r * 32 + lane)]));
        }
      } else {
        float g = expf(to_float(decay[bt * kv_num_heads + h_kv]));
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          exp_g[r] = g;
        }
      }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        s_shard[r] *= exp_g[r];
      }
    }

    // Retrieval: r_col = sum_i (decayed S[i][col]) * k[i].
    float r_col = 0.0f;
    if (needs_retrieval) {
      float partial = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        partial += s_shard[r] * k_reg[r];
      }
      r_col = warp_reduce_sum(partial);
    }

    const float delta_col = ComputeLinearAttentionDeltaColumn(
        value, beta_in, bt, v_hidden, h_kv, d_v, col, kv_num_heads, needs_beta, beta_per_head, r_col);

    // State update: S[i][col] = decayed S[i][col] + k[i] * delta_col.
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      s_shard[r] += k_reg[r] * delta_col;
    }

    // Emit the recurrent state AFTER processing token t into the right-aligned window slot
    // t + W - seq_len (negative => outside the window, dropped). The last token's slot is always
    // W-1 and is written by the epilogue below. This lane owns rows {r*32 + lane} of column
    // `col`. Layout [W, B, H_kv, DK, d_v] row-major; element (row, col) at base_t + row*d_v + col.
    const int state_slot = t + state_window - seq_len;
    if (state_slot >= 0 && t + 1 < seq_len) {
      const int64_t base_t = ((int64_t)state_slot * batch_size + b) * slot_stride +
                             (int64_t)h_kv * DK * d_v;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        present_state[base_t + (int64_t)(r * 32 + lane) * d_v + col] = from_float<T>(s_shard[r]);
      }
    }

    // Readout: output = scale * sum_i S[i][col] * q[i].
    const int head_count = GetLinearAttentionReadoutHeadCount(q_num_heads, kv_num_heads);
    for (int group_index = 0; group_index < head_count; ++group_index) {
      const LinearAttentionReadoutHeads readout_heads =
          GetLinearAttentionReadoutHeads(h_kv, q_num_heads, kv_num_heads, group_index);
      float partial = 0.0f;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        float q_reg = to_float(query[bt * q_hidden + readout_heads.query_head * DK + (r * 32 + lane)]);
        partial += s_shard[r] * q_reg;
      }
      float acc = warp_reduce_sum(partial);
      if (lane == 0) {
        output[bt * output_hidden + readout_heads.output_head * d_v + col] = from_float<T>(scale * acc);
      }
    }
  }

  // Write the updated state column back (row-major layout, strided).
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    S_present_col[(int64_t)(r * 32 + lane) * d_v] = from_float<T>(s_shard[r]);
  }
}

// =============================================================================
// Decode-optimized kernel v2 — column-per-thread, coalesced row-major state.
//
// The v1 warp-per-column kernel above shards the state column across a warp's
// lanes (DK/32 rows each). In the row-major [DK, d_v] state layout a column is
// strided by d_v, so the per-token state load/store is fully uncoalesced (32
// lanes hit 32 separate sectors). llama.cpp avoids this by storing the state
// transposed, but that would change this op's present_state output layout.
//
// This v2 keeps the state layout unchanged (row-major [DK, d_v], contract and
// parity preserved) and instead maps ONE THREAD per output column. Thread `col`
// owns the whole column S[:, col] in registers (DK values). For a fixed row i,
// consecutive threads read consecutive addresses (i*d_v + col), so the state
// load/store is fully COALESCED — no transpose required. Per-column reductions
// (S^T@k, S^T@q) are sequential within the thread, so there are no cross-thread
// reductions; the only shared data are the per-token k/q/decay broadcasts, with
// block-wide barriers around those cooperative loads.
//
// Grid:  (batch_size, kv_num_heads, ceil(d_v / kColsPerBlock))
// Block: (kColsPerBlock)
// Requires d_v % kColsPerBlock == 0 (so every thread maps to a valid column and
// all threads participate in the cooperative broadcast loads / __syncthreads).
// =============================================================================
constexpr int kColsPerBlock = 32;

template <typename T, int DK>
__global__ void LinearAttentionDecodeColKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* past_state,
    T* present_state,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ output,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int n_k_heads,
    int d_v,
    int output_hidden,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval,
    bool force_sequential_state_roundtrip,
    int batch_size,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int tid = threadIdx.x;
  const int col = blockIdx.z * kColsPerBlock + tid;
  // d_v is required to be a multiple of kColsPerBlock by the dispatcher, so all
  // threads have a valid column and may participate in the barriers below.

  const int kv_per_k = kv_num_heads / n_k_heads;
  const int h_k = h_kv / kv_per_k;

  // This thread owns column `col`: S[i][col] lives at i*d_v + col (row-major) within window
  // slot W-1, the state after the last token. Window-major, so a slot spans the whole batch.
  const int64_t slot_stride = (int64_t)kv_num_heads * DK * d_v;
  const int64_t state_offset = ((int64_t)(state_window - 1) * batch_size + b) * slot_stride +
                               (int64_t)h_kv * DK * d_v + col;
  const T* S_past_head = past_state + state_offset;
  T* S_present_head = present_state + state_offset;
  float s_col[DK];
#pragma unroll
  for (int i = 0; i < DK; ++i) {
    s_col[i] = to_float(S_past_head[(int64_t)i * d_v]);
  }

  // Per-token broadcasts shared across all columns of this (b, h_kv).
  __shared__ float k_sh[DK];
  __shared__ float q_sh[DK];
  __shared__ float g_sh[DK];
  __shared__ float scalar_g;

  const int k_hidden = n_k_heads * DK;
  const int q_hidden = q_num_heads * DK;
  const int v_hidden = kv_num_heads * d_v;

  for (int t = 0; t < seq_len; ++t) {
    const int64_t bt = (int64_t)b * seq_len + t;

    for (int i = tid; i < DK; i += kColsPerBlock) {
      k_sh[i] = to_float(key[bt * k_hidden + h_k * DK + i]);
    }
    if (needs_decay) {
      if (decay_per_key_dim) {
        for (int i = tid; i < DK; i += kColsPerBlock) {
          g_sh[i] = expf(to_float(decay[bt * (kv_num_heads * DK) + h_kv * DK + i]));
        }
      } else if (tid == 0) {
        scalar_g = expf(to_float(decay[bt * kv_num_heads + h_kv]));
      }
    }
    __syncthreads();

    // Decay.
    if (needs_decay) {
      if (decay_per_key_dim) {
#pragma unroll
        for (int i = 0; i < DK; ++i) {
          s_col[i] *= g_sh[i];
        }
      } else {
        const float g = scalar_g;
#pragma unroll
        for (int i = 0; i < DK; ++i) {
          s_col[i] *= g;
        }
      }
    }

    // Retrieval: r_col = sum_i (decayed S[i][col]) * k[i].
    float r_col = 0.0f;
    if (needs_retrieval) {
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        r_col += s_col[i] * k_sh[i];
      }
    }

    const float delta_col = ComputeLinearAttentionDeltaColumn(
        value, beta_in, bt, v_hidden, h_kv, d_v, col, kv_num_heads, needs_beta, beta_per_head, r_col);

    // State update: S[i][col] = decayed S[i][col] + k[i] * delta_col.
#pragma unroll
    for (int i = 0; i < DK; ++i) {
      s_col[i] += k_sh[i] * delta_col;
    }

    // Emit the recurrent state AFTER processing token t into the right-aligned window slot
    // t + W - seq_len (negative => outside the window, dropped). The last token's slot is always
    // W-1 and is written by the coalesced epilogue below. This thread owns column `col`.
    // Layout [W, B, H_kv, DK, d_v] row-major; element (i, col) at base_t + i*d_v + col.
    const int state_slot = t + state_window - seq_len;
    if (state_slot >= 0 && t + 1 < seq_len) {
      const int64_t base_t = ((int64_t)state_slot * batch_size + b) * slot_stride +
                             (int64_t)h_kv * DK * d_v;
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        present_state[base_t + (int64_t)i * d_v + col] = from_float<T>(s_col[i]);
      }
    }

    // Readout: output = scale * sum_i S[i][col] * q[i].
    const int head_count = GetLinearAttentionReadoutHeadCount(q_num_heads, kv_num_heads);
    for (int group_index = 0; group_index < head_count; ++group_index) {
      const LinearAttentionReadoutHeads readout_heads =
          GetLinearAttentionReadoutHeads(h_kv, q_num_heads, kv_num_heads, group_index);
      __syncthreads();
      for (int i = tid; i < DK; i += kColsPerBlock) {
        q_sh[i] = to_float(query[bt * q_hidden + readout_heads.query_head * DK + i]);
      }
      __syncthreads();
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        acc += s_col[i] * q_sh[i];
      }
      output[bt * output_hidden + readout_heads.output_head * d_v + col] = from_float<T>(scale * acc);
    }
    if (force_sequential_state_roundtrip && t + 1 < seq_len) {
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        s_col[i] = to_float(from_float<T>(s_col[i]));
      }
    }
    __syncthreads();  // before next token overwrites k_sh/g_sh
  }

  // Store the updated column back (coalesced).
#pragma unroll
  for (int i = 0; i < DK; ++i) {
    S_present_head[(int64_t)i * d_v] = from_float<T>(s_col[i]);
  }
}

// =============================================================================
// v3 = v2 with the DK axis additionally split across RS threads.
//
// v2 gives one thread a whole state column, so the launch is
// batch * kv_num_heads * (d_v / kColsPerBlock) blocks of a *single* warp. On a
// 132-SM H200 with 32 KV heads that is one warp per SM, which cannot keep
// enough state loads/stores in flight to hide HBM latency. Splitting DK across
// RS threads multiplies resident warps by RS and cuts the per-thread register
// footprint by RS; the price is one block-wide reduction per token for each of
// the two DK-length dot products (retrieval and readout).
//
// Grid:  (batch_size, kv_num_heads, d_v / kColsPerBlock)
// Block: (kColsPerBlock, RS)
// =============================================================================
template <typename T, int DK, int RS>
__global__ void LinearAttentionDecodeColSplitKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* past_state,
    T* present_state,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ output,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int n_k_heads,
    int d_v,
    int output_hidden,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval,
    bool force_sequential_state_roundtrip,
    int batch_size,
    int state_window) {
  constexpr int DKP = DK / RS;
  constexpr int kBlockThreads = kColsPerBlock * RS;

  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int tx = static_cast<int>(threadIdx.x);
  const int part = static_cast<int>(threadIdx.y);
  const int tid = part * kColsPerBlock + tx;
  const int col = blockIdx.z * kColsPerBlock + tx;
  const int i0 = part * DKP;

  const int kv_per_k = kv_num_heads / n_k_heads;
  const int h_k = h_kv / kv_per_k;

  const int64_t slot_stride = (int64_t)kv_num_heads * DK * d_v;
  const int64_t state_offset = ((int64_t)(state_window - 1) * batch_size + b) * slot_stride +
                               (int64_t)h_kv * DK * d_v + col;
  const T* S_past_head = past_state + state_offset;
  T* S_present_head = present_state + state_offset;

  float s_col[DKP];
#pragma unroll
  for (int r = 0; r < DKP; ++r) {
    s_col[r] = to_float(S_past_head[(int64_t)(i0 + r) * d_v]);
  }

  __shared__ float k_sh[DK];
  __shared__ float q_sh[DK];
  __shared__ float g_sh[DK];
  __shared__ float scalar_g;
  __shared__ float red[RS][kColsPerBlock];

  const int k_hidden = n_k_heads * DK;
  const int q_hidden = q_num_heads * DK;
  const int v_hidden = kv_num_heads * d_v;

  for (int t = 0; t < seq_len; ++t) {
    const int64_t bt = (int64_t)b * seq_len + t;

    for (int i = tid; i < DK; i += kBlockThreads) {
      k_sh[i] = to_float(key[bt * k_hidden + h_k * DK + i]);
    }
    if (needs_decay) {
      if (decay_per_key_dim) {
        for (int i = tid; i < DK; i += kBlockThreads) {
          g_sh[i] = expf(to_float(decay[bt * (kv_num_heads * DK) + h_kv * DK + i]));
        }
      } else if (tid == 0) {
        scalar_g = expf(to_float(decay[bt * kv_num_heads + h_kv]));
      }
    }
    __syncthreads();

    if (needs_decay) {
      if (decay_per_key_dim) {
#pragma unroll
        for (int r = 0; r < DKP; ++r) {
          s_col[r] *= g_sh[i0 + r];
        }
      } else {
        const float g = scalar_g;
#pragma unroll
        for (int r = 0; r < DKP; ++r) {
          s_col[r] *= g;
        }
      }
    }

    float r_col = 0.0f;
    if (needs_retrieval) {
      float partial = 0.0f;
#pragma unroll
      for (int r = 0; r < DKP; ++r) {
        partial += s_col[r] * k_sh[i0 + r];
      }
      red[part][tx] = partial;
      __syncthreads();
#pragma unroll
      for (int p = 0; p < RS; ++p) {
        r_col += red[p][tx];
      }
    }

    const float delta_col = ComputeLinearAttentionDeltaColumn(
        value, beta_in, bt, v_hidden, h_kv, d_v, col, kv_num_heads, needs_beta, beta_per_head, r_col);

#pragma unroll
    for (int r = 0; r < DKP; ++r) {
      s_col[r] += k_sh[i0 + r] * delta_col;
    }

    const int state_slot = t + state_window - seq_len;
    if (state_slot >= 0 && t + 1 < seq_len) {
      const int64_t base_t = ((int64_t)state_slot * batch_size + b) * slot_stride +
                             (int64_t)h_kv * DK * d_v;
#pragma unroll
      for (int r = 0; r < DKP; ++r) {
        present_state[base_t + (int64_t)(i0 + r) * d_v + col] = from_float<T>(s_col[r]);
      }
    }

    const int head_count = GetLinearAttentionReadoutHeadCount(q_num_heads, kv_num_heads);
    for (int group_index = 0; group_index < head_count; ++group_index) {
      const LinearAttentionReadoutHeads readout_heads =
          GetLinearAttentionReadoutHeads(h_kv, q_num_heads, kv_num_heads, group_index);
      __syncthreads();
      for (int i = tid; i < DK; i += kBlockThreads) {
        q_sh[i] = to_float(query[bt * q_hidden + readout_heads.query_head * DK + i]);
      }
      __syncthreads();
      float partial = 0.0f;
#pragma unroll
      for (int r = 0; r < DKP; ++r) {
        partial += s_col[r] * q_sh[i0 + r];
      }
      red[part][tx] = partial;
      __syncthreads();
      float acc = 0.0f;
#pragma unroll
      for (int p = 0; p < RS; ++p) {
        acc += red[p][tx];
      }
      if (part == 0) {
        output[bt * output_hidden + readout_heads.output_head * d_v + col] = from_float<T>(scale * acc);
      }
    }

    if (force_sequential_state_roundtrip && t + 1 < seq_len) {
#pragma unroll
      for (int r = 0; r < DKP; ++r) {
        s_col[r] = to_float(from_float<T>(s_col[r]));
      }
    }
    __syncthreads();  // before next token overwrites k_sh/g_sh
  }

#pragma unroll
  for (int r = 0; r < DKP; ++r) {
    S_present_head[(int64_t)(i0 + r) * d_v] = from_float<T>(s_col[r]);
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
    const T* past_state,
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
    int decode_seq_threshold,
    int row_split,
    int multiprocessor_count,
    int max_threads_per_block,
    size_t max_shared_memory_per_block,
    int state_window) {
  // Grid: one block per (batch, kv_head)
  const dim3 grid(batch_size, kv_num_heads, 1);

  int output_hidden = std::max(q_num_heads, kv_num_heads) * d_v;

  // ---------------------------------------------------------------------------
  // Decode fast path: small seq_len is latency-bound. The recurrent kernels keep
  // state in shared memory across the token loop (great for prefill) but at
  // decode that caching yields no amortization while only using kv_num_heads
  // blocks. The column-parallel decode kernel increases occupancy with
  // finer-grained work distribution. Engaged for the common decode shapes
  // (DK in {64,128,256});
  // everything else falls through to the recurrent kernels below.
  // ---------------------------------------------------------------------------
  // Prefill is *also* routed to the column-parallel kernels whenever the recurrent grid would
  // leave the GPU idle. The recurrent kernels launch exactly batch_size * kv_num_heads blocks, so
  // a single-sequence hybrid model with 32 KV heads occupies 32 of 132 SMs on H200 while walking
  // the token loop with ~5 __syncthreads() apiece. Splitting d_v across kColsPerBlock-wide blocks
  // multiplies the block count by d_v / kColsPerBlock at identical total thread count, and the
  // column recurrence is mathematically independent per column so the result is unchanged up to
  // reduction order. Measured on Qwen3.6-35B-A3B (batch 1, 32 KV heads, d_k = d_v = 128, 30
  // linear-attention layers): 1024-token prefill 215 -> 97 us/token.
  //
  // Once batch_size * kv_num_heads already fills the machine the recurrent kernels keep their
  // shared-memory state amortization, so the split is only applied below that point.
  const bool recurrent_grid_underfills_gpu =
      static_cast<int64_t>(batch_size) * kv_num_heads < multiprocessor_count;
  const size_t recurrent_smem_size =
      (static_cast<size_t>(d_k) * d_v + d_k + std::max(d_k, d_v)) * sizeof(float);
  const bool recurrent_smem_exceeds_device = recurrent_smem_size > max_shared_memory_per_block;
  // Only the v2 (column-per-thread) kernel below has been validated at prefill lengths, so the
  // occupancy and shared-memory overrides are limited to the shapes it accepts; everything else
  // keeps the original seq_len cutoff.
  const bool prefill_column_split =
      (recurrent_grid_underfills_gpu || recurrent_smem_exceeds_device) &&
      d_k <= 128 && (d_v % kColsPerBlock) == 0;

  if ((seq_len <= decode_seq_threshold || prefill_column_split) &&
      (d_k == 64 || d_k == 128 || d_k == 256)) {
    // v2 (column-per-thread, coalesced row-major state) is the default for
    // DK <= 128. It requires d_v % kColsPerBlock == 0; otherwise fall back
    // to the v1 warp-per-column kernel (which handles any d_v). DK=256 also
    // uses v1 to avoid the high per-thread register footprint of s_col[256].
    if (d_k <= 128 && d_v % kColsPerBlock == 0) {
      const bool force_sequential_state_roundtrip =
          ParseEnvironmentVariableWithDefault<bool>("ORT_LINEAR_ATTENTION_FORCE_SEQUENTIAL_STATE_ROUNDTRIP", false);
      const dim3 decode_grid(batch_size, kv_num_heads,
                             (d_v + kColsPerBlock - 1) / kColsPerBlock);
      const dim3 decode_block(kColsPerBlock, 1, 1);

      // v3 splits the DK axis across `row_split` threads so the launch has
      // row_split warps per block instead of one; see the kernel comment.
      auto launch_col_split = [&](auto dk_tag, auto rs_tag) -> Status {
        constexpr int DK = decltype(dk_tag)::value;
        constexpr int RS = decltype(rs_tag)::value;
        const dim3 split_block(kColsPerBlock, RS, 1);
        LinearAttentionDecodeColSplitKernel<T, DK, RS><<<decode_grid, split_block, 0, stream>>>(
            query, key, value, past_state, present_state, decay, beta, output,
            seq_len, q_num_heads, kv_num_heads, n_k_heads, d_v, output_hidden, scale,
            needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval,
            force_sequential_state_roundtrip, batch_size, state_window);
        return CUDA_CALL(cudaGetLastError());
      };

      auto launch_col = [&](auto dk_tag) -> Status {
        constexpr int DK = decltype(dk_tag)::value;
        if (row_split > 1 && (DK % row_split) == 0 &&
            kColsPerBlock * row_split <= max_threads_per_block) {
          switch (row_split) {
            case 2:
              return launch_col_split(dk_tag, std::integral_constant<int, 2>{});
            case 4:
              return launch_col_split(dk_tag, std::integral_constant<int, 4>{});
            case 8:
              return launch_col_split(dk_tag, std::integral_constant<int, 8>{});
            case 16:
              return launch_col_split(dk_tag, std::integral_constant<int, 16>{});
            case 32:
              return launch_col_split(dk_tag, std::integral_constant<int, 32>{});
            default:
              break;
          }
        }
        LinearAttentionDecodeColKernel<T, DK><<<decode_grid, decode_block, 0, stream>>>(
            query, key, value, past_state, present_state, decay, beta, output,
            seq_len, q_num_heads, kv_num_heads, n_k_heads, d_v, output_hidden, scale,
            needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval,
            force_sequential_state_roundtrip, batch_size, state_window);
        return CUDA_CALL(cudaGetLastError());
      };

      if (d_k == 64) {
        return launch_col(std::integral_constant<int, 64>{});
      } else if (d_k == 128) {
        return launch_col(std::integral_constant<int, 128>{});
      }
    }

    const dim3 decode_grid(batch_size, kv_num_heads,
                           (d_v + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const dim3 decode_block(32, kWarpsPerBlock, 1);

    auto launch_decode = [&](auto dk_tag) -> Status {
      constexpr int DK = decltype(dk_tag)::value;
      LinearAttentionDecodeKernel<T, DK><<<decode_grid, decode_block, 0, stream>>>(
          query, key, value, past_state, present_state, decay, beta, output,
          seq_len, q_num_heads, kv_num_heads, n_k_heads, d_v, output_hidden, scale,
          needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval, batch_size, state_window);
      return CUDA_CALL(cudaGetLastError());
    };

    if (d_k == 64) {
      return launch_decode(std::integral_constant<int, 64>{});
    } else if (d_k == 128) {
      return launch_decode(std::integral_constant<int, 128>{});
    } else {  // d_k == 256
      return launch_decode(std::integral_constant<int, 256>{});
    }
  }

  if (recurrent_smem_exceeds_device) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "LinearAttention: recurrent kernel requires ", recurrent_smem_size,
                           " bytes of opt-in shared memory per block, but the device supports ",
                           max_shared_memory_per_block,
                           " bytes. No compatible fallback exists (column kernel requires d_k in {64,128} and d_v % ",
                           kColsPerBlock, " == 0).");
  }

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
        query, key, value, past_state, present_state, decay, beta, output,
        seq_len, q_num_heads, kv_num_heads, n_k_heads, output_hidden, scale,
        needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval, batch_size, state_window);

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
  const size_t smem_size = recurrent_smem_size;

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
      query, key, value, past_state, present_state, decay, beta, output,
      seq_len, q_num_heads, kv_num_heads, n_k_heads, d_k, d_v, output_hidden, scale,
      needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval, batch_size, state_window);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchLinearAttentionKernel<float>(
    cudaStream_t, const float*, const float*, const float*,
    const float*, const float*, float*, const float*, float*,
    int, int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int, int, int, int, size_t, int);

template Status LaunchLinearAttentionKernel<half>(
    cudaStream_t, const half*, const half*, const half*,
    const half*, const half*, half*, const half*, half*,
    int, int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int, int, int, int, size_t, int);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchLinearAttentionKernel<__nv_bfloat16>(
    cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int, int, int, int, size_t, int);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
