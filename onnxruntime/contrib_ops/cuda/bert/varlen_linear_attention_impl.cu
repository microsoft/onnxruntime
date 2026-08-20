// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Packed (ragged) variant of the recurrent linear-attention kernel: every sequence's tokens are
// packed back to back along a single token axis instead of padded into a [batch, seq_len, ...]
// tensor, and cu_seqlens gives each sequence's token range. The per-token recurrence (decay,
// retrieval, delta update, readout) is identical to the dense op; only token addressing and the
// state-window write condition change to use each sequence's own length instead of a shared
// seq_len.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <cstdint>
#include "contrib_ops/cuda/bert/varlen_linear_attention.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
__device__ __forceinline__ float ComputeVarlenLinearAttentionDeltaColumn(
    const T* value,
    const T* beta_in,
    int64_t bt,
    int value_hidden,
    int kv_head,
    int d_v,
    int col,
    int kv_num_heads,
    bool needs_beta,
    bool beta_per_head,
    float retrieval_col) {
  const float value_col = to_float(value[bt * value_hidden + kv_head * d_v + col]);
  if (!needs_beta) {
    return value_col;
  }

  const float beta_value = beta_per_head ? to_float(beta_in[bt * kv_num_heads + kv_head])
                                         : to_float(beta_in[bt]);
  return beta_value * (value_col - retrieval_col);
}

struct VarlenLinearAttentionReadoutHeads {
  int query_head;
  int output_head;
};

__device__ __forceinline__ int GetVarlenLinearAttentionReadoutHeadCount(int q_num_heads, int kv_num_heads) {
  return q_num_heads >= kv_num_heads ? q_num_heads / kv_num_heads : 1;
}

__device__ __forceinline__ VarlenLinearAttentionReadoutHeads GetVarlenLinearAttentionReadoutHeads(
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

// Column-per-thread kernel: this thread owns state column `col` for the whole token loop.
// Reused for both the all-ones decode fast path (kAllOnes=true: token i belongs to sequence i,
// local_len=1, cu_seqlens is never read) and the ragged column-split prefill path (kAllOnes=
// false: start/local_len come from cu_seqlens). Requires d_v % kColsPerBlock == 0 and DK in
// {64, 128}; the dispatcher only reaches this kernel under those conditions.
//
// Grid:  (batch_size, kv_num_heads, d_v / kColsPerBlock)
// Block: (kColsPerBlock)
constexpr int kColsPerBlock = 32;

template <typename T, int DK, bool kAllOnes>
__global__ void VarlenLinearAttentionColKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* past_state,
    T* present_state,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ output,
    const int32_t* __restrict__ cu_seqlens,
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
  const int b = blockIdx.x;
  const int h_kv = blockIdx.y;
  const int tid = threadIdx.x;
  const int col = blockIdx.z * kColsPerBlock + tid;

  int64_t start;
  int local_len;
  if constexpr (kAllOnes) {
    start = b;
    local_len = 1;
  } else {
    start = cu_seqlens[b];
    local_len = cu_seqlens[b + 1] - cu_seqlens[b];
  }

  const int kv_per_k = kv_num_heads / n_k_heads;
  const int h_k = h_kv / kv_per_k;

  // This thread owns column `col`: S[i][col] lives at i*d_v + col (row-major) within window
  // slot W-1, the state after the last token. Window-major, so a slot spans the whole batch
  // (all requests), regardless of individual sequence lengths.
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

  // Per-token broadcasts shared across all columns of this (sequence, h_kv).
  __shared__ float k_sh[DK];
  __shared__ float q_sh[DK];
  __shared__ float g_sh[DK];
  __shared__ float scalar_g;

  const int k_hidden = n_k_heads * DK;
  const int q_hidden = q_num_heads * DK;
  const int v_hidden = kv_num_heads * d_v;

  for (int t = 0; t < local_len; ++t) {
    const int64_t bt = start + t;

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

    const float delta_col = ComputeVarlenLinearAttentionDeltaColumn(
        value, beta_in, bt, v_hidden, h_kv, d_v, col, kv_num_heads, needs_beta, beta_per_head, r_col);

    // State update: S[i][col] = decayed S[i][col] + k[i] * delta_col.
#pragma unroll
    for (int i = 0; i < DK; ++i) {
      s_col[i] += k_sh[i] * delta_col;
    }

    // Emit the recurrent state AFTER processing token t into the right-aligned window slot
    // t + W - local_len (negative => outside the window, dropped). The last token's slot is
    // always W-1 and is written by the coalesced epilogue below. This thread owns column `col`.
    // Layout [W, B, H_kv, DK, d_v] row-major; element (i, col) at base_t + i*d_v + col.
    const int state_slot = t + state_window - local_len;
    if (state_slot >= 0 && t + 1 < local_len) {
      const int64_t base_t = ((int64_t)state_slot * batch_size + b) * slot_stride +
                             (int64_t)h_kv * DK * d_v;
#pragma unroll
      for (int i = 0; i < DK; ++i) {
        present_state[base_t + (int64_t)i * d_v + col] = from_float<T>(s_col[i]);
      }
    }

    // Readout: output = scale * sum_i S[i][col] * q[i].
    const int head_count = GetVarlenLinearAttentionReadoutHeadCount(q_num_heads, kv_num_heads);
    for (int group_index = 0; group_index < head_count; ++group_index) {
      const VarlenLinearAttentionReadoutHeads readout_heads =
          GetVarlenLinearAttentionReadoutHeads(h_kv, q_num_heads, kv_num_heads, group_index);
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
    __syncthreads();  // before next token overwrites k_sh/g_sh
  }

  // Store the updated column back (coalesced).
#pragma unroll
  for (int i = 0; i < DK; ++i) {
    S_present_head[(int64_t)i * d_v] = from_float<T>(s_col[i]);
  }
}

// General fallback: one block per (sequence, kv_head), full [d_k, d_v] state in dynamic shared
// memory, any d_k/d_v. Handles every shape the column kernel cannot (arbitrary DK, or d_v not a
// multiple of kColsPerBlock) and every ragged sequence length.
//
// Grid:  (batch_size, kv_num_heads, 1)
// Block: (round_up(max(d_k, d_v), 32))
template <typename T>
__global__ void VarlenLinearAttentionRecurrentKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* past_state,
    T* present_state,
    const T* __restrict__ decay,
    const T* __restrict__ beta_in,
    T* __restrict__ output,
    const int32_t* __restrict__ cu_seqlens,
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

  const int64_t start = cu_seqlens[b];
  const int local_len = static_cast<int>(cu_seqlens[b + 1] - cu_seqlens[b]);

  const int64_t slot_stride = (int64_t)kv_num_heads * d_k * d_v;
  const int64_t state_offset = ((int64_t)(state_window - 1) * batch_size + b) * slot_stride +
                               (int64_t)h_kv * d_k * d_v;
  const T* S_past = past_state + state_offset;
  T* S_present = present_state + state_offset;

  extern __shared__ float smem[];
  float* S_smem = smem;                       // [d_k * d_v]
  float* k_buf = smem + d_k * d_v;            // [d_k]
  float* s_scratch = smem + d_k * d_v + d_k;  // [max(d_k, d_v)]

  for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
    S_smem[idx] = to_float(S_past[idx]);
  }
  __syncthreads();

  for (int t = 0; t < local_len; ++t) {
    const int64_t bt = start + t;

    float kt_val = 0.0f;
    if (tid < d_k) {
      const int64_t k_offset = bt * (n_k_heads * d_k) + h_k * d_k + tid;
      kt_val = to_float(key[k_offset]);
    }

    bool fused_decay_update = false;
    float fused_exp_g = 1.0f;

    if (needs_decay && needs_retrieval && !decay_per_key_dim) {
      if (tid < d_k) {
        k_buf[tid] = kt_val;
      }
      if (tid == 0) {
        const int64_t g_offset = bt * kv_num_heads + h_kv;
        s_scratch[0] = expf(to_float(decay[g_offset]));
      }
      __syncthreads();

      fused_exp_g = s_scratch[0];
      __syncthreads();

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
      if (needs_decay) {
        if (!decay_per_key_dim) {
          if (tid == 0) {
            const int64_t g_offset = bt * kv_num_heads + h_kv;
            s_scratch[0] = expf(to_float(decay[g_offset]));
          }
          __syncthreads();
        }
        if (tid < d_k) {
          float exp_g;
          if (decay_per_key_dim) {
            const int64_t g_offset = bt * (kv_num_heads * d_k) + h_kv * d_k + tid;
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

    if (needs_beta) {
      float bt_beta;
      if (beta_per_head) {
        bt_beta = to_float(beta_in[bt * kv_num_heads + h_kv]);
      } else {
        bt_beta = to_float(beta_in[bt]);
      }

      if (tid < d_v) {
        const int64_t v_base = bt * (kv_num_heads * d_v) + h_kv * d_v;
        float vj = to_float(value[v_base + tid]);
        s_scratch[tid] = bt_beta * (vj - s_scratch[tid]);
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
        const int64_t v_base = bt * (kv_num_heads * d_v) + h_kv * d_v;
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
    // t + W - local_len (negative => this position falls outside the window and is dropped).
    // The last token's slot is always W-1 and is written by the epilogue below.
    const int state_slot = t + state_window - local_len;
    if (state_slot >= 0 && t + 1 < local_len) {
      const int64_t base_t = ((int64_t)state_slot * batch_size + b) * slot_stride +
                             (int64_t)h_kv * d_k * d_v;
      for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
        present_state[base_t + idx] = from_float<T>(S_smem[idx]);
      }
    }

    // Query readout — output = S^T @ q_t (standard GQA or inverse GQA)
    if (q_num_heads >= kv_num_heads) {
      int heads_per_group = q_num_heads / kv_num_heads;
      for (int g = 0; g < heads_per_group; ++g) {
        if (g > 0) {
          __syncthreads();
        }

        int h_q = h_kv * heads_per_group + g;
        if (tid < d_k) {
          const int64_t q_offset = bt * (q_num_heads * d_k) + h_q * d_k + tid;
          s_scratch[tid] = to_float(query[q_offset]);
        }
        __syncthreads();

        if (tid < d_v) {
          float acc = 0.0f;
          for (int i = 0; i < d_k; ++i) {
            acc += S_smem[i * d_v + tid] * s_scratch[i];
          }
          const int64_t out_offset = bt * output_hidden + h_q * d_v + tid;
          output[out_offset] = from_float<T>(scale * acc);
        }
      }
    } else {
      int h_q = h_kv * q_num_heads / kv_num_heads;
      if (tid < d_k) {
        const int64_t q_offset = bt * (q_num_heads * d_k) + h_q * d_k + tid;
        s_scratch[tid] = to_float(query[q_offset]);
      }
      __syncthreads();

      if (tid < d_v) {
        float acc = 0.0f;
        for (int i = 0; i < d_k; ++i) {
          acc += S_smem[i * d_v + tid] * s_scratch[i];
        }
        const int64_t out_offset = bt * output_hidden + h_kv * d_v + tid;
        output[out_offset] = from_float<T>(scale * acc);
      }
    }

    __syncthreads();
  }

  for (int idx = tid; idx < d_k * d_v; idx += num_threads) {
    S_present[idx] = from_float<T>(S_smem[idx]);
  }
}

}  // namespace

template <typename T>
Status LaunchVarlenLinearAttentionKernel(
    cudaStream_t stream,
    const T* query,
    const T* key,
    const T* value,
    const T* decay,
    const T* beta,
    T* output,
    const T* past_state,
    T* present_state,
    const int32_t* cu_seqlens,
    int batch_size,
    bool all_ones,
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
    int multiprocessor_count,
    int max_threads_per_block,
    size_t max_shared_memory_per_block,
    int state_window) {
  const int output_hidden = std::max(q_num_heads, kv_num_heads) * d_v;
  const bool col_kernel_eligible = (d_k == 64 || d_k == 128) && (d_v % kColsPerBlock) == 0;

  // All-ones decode fast path: every sequence has exactly one token, so cu_seqlens is never
  // read and there is no token loop (the column kernel's `for (t = 0; t < local_len; ++t)`
  // degenerates to a single iteration with start=b).
  if (all_ones && col_kernel_eligible) {
    const dim3 grid(batch_size, kv_num_heads, d_v / kColsPerBlock);
    const dim3 block(kColsPerBlock, 1, 1);
    auto launch = [&](auto dk_tag) -> Status {
      constexpr int DK = decltype(dk_tag)::value;
      VarlenLinearAttentionColKernel<T, DK, true><<<grid, block, 0, stream>>>(
          query, key, value, past_state, present_state, decay, beta, output, cu_seqlens,
          q_num_heads, kv_num_heads, n_k_heads, d_v, output_hidden, scale,
          needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval,
          batch_size, state_window);
      return CUDA_CALL(cudaGetLastError());
    };
    return d_k == 64 ? launch(std::integral_constant<int, 64>{})
                     : launch(std::integral_constant<int, 128>{});
  }

  const size_t recurrent_smem_size =
      (static_cast<size_t>(d_k) * d_v + d_k + std::max(d_k, d_v)) * sizeof(float);
  const bool recurrent_smem_exceeds_device = recurrent_smem_size > max_shared_memory_per_block;
  // The column recurrence avoids materializing the full state matrix in shared memory and exposes
  // independent state columns as blocks. Use it for every compatible shape: the generic kernel's
  // large per-block shared state causes a severe occupancy cliff once batch_size * kv_num_heads
  // crosses the old grid-size heuristic.
  const bool prefill_column_split = col_kernel_eligible;
  (void)multiprocessor_count;

  if (prefill_column_split) {
    const dim3 grid(batch_size, kv_num_heads, d_v / kColsPerBlock);
    const dim3 block(kColsPerBlock, 1, 1);
    auto launch = [&](auto dk_tag) -> Status {
      constexpr int DK = decltype(dk_tag)::value;
      VarlenLinearAttentionColKernel<T, DK, false><<<grid, block, 0, stream>>>(
          query, key, value, past_state, present_state, decay, beta, output, cu_seqlens,
          q_num_heads, kv_num_heads, n_k_heads, d_v, output_hidden, scale,
          needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval,
          batch_size, state_window);
      return CUDA_CALL(cudaGetLastError());
    };
    return d_k == 64 ? launch(std::integral_constant<int, 64>{})
                     : launch(std::integral_constant<int, 128>{});
  }

  if (recurrent_smem_exceeds_device) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "VarlenLinearAttention: recurrent kernel requires ", recurrent_smem_size,
                           " bytes of opt-in shared memory per block, but the device supports ",
                           max_shared_memory_per_block,
                           " bytes. No compatible fallback exists (column kernel requires d_k in "
                           "{64,128} and d_v % ",
                           kColsPerBlock, " == 0).");
  }

  int threads = ((std::max(d_k, d_v) + 31) / 32) * 32;
  if (threads > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "VarlenLinearAttention: max(d_k=", d_k, ", d_v=", d_v,
                           ") exceeds max threads per block (", max_threads_per_block,
                           "). Use a model with smaller head dimensions.");
  }
  const dim3 grid(batch_size, kv_num_heads, 1);
  const dim3 block(threads, 1, 1);

  if (recurrent_smem_size > 48 * 1024) {
    cudaError_t attr_err = cudaFuncSetAttribute(
        VarlenLinearAttentionRecurrentKernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(recurrent_smem_size));
    if (attr_err != cudaSuccess) {
      return CUDA_CALL(attr_err);
    }
  }

  VarlenLinearAttentionRecurrentKernel<T><<<grid, block, recurrent_smem_size, stream>>>(
      query, key, value, past_state, present_state, decay, beta, output, cu_seqlens,
      q_num_heads, kv_num_heads, n_k_heads, d_k, d_v, output_hidden, scale,
      needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval,
      batch_size, state_window);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchVarlenLinearAttentionKernel<float>(
    cudaStream_t stream, const float* query, const float* key, const float* value,
    const float* decay, const float* beta, float* output, const float* past_state,
    float* present_state, const int32_t* cu_seqlens, int batch_size, bool all_ones,
    int q_num_heads, int kv_num_heads, int n_k_heads, int d_k, int d_v, float scale,
    bool needs_decay, bool decay_per_key_dim, bool needs_beta, bool beta_per_head,
    bool needs_retrieval, int multiprocessor_count, int max_threads_per_block,
    size_t max_shared_memory_per_block, int state_window);

template Status LaunchVarlenLinearAttentionKernel<half>(
    cudaStream_t stream, const half* query, const half* key, const half* value,
    const half* decay, const half* beta, half* output, const half* past_state,
    half* present_state, const int32_t* cu_seqlens, int batch_size, bool all_ones,
    int q_num_heads, int kv_num_heads, int n_k_heads, int d_k, int d_v, float scale,
    bool needs_decay, bool decay_per_key_dim, bool needs_beta, bool beta_per_head,
    bool needs_retrieval, int multiprocessor_count, int max_threads_per_block,
    size_t max_shared_memory_per_block, int state_window);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
