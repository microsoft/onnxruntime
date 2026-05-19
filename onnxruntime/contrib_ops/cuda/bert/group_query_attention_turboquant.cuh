// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// Enable TurboQuant KV cache support (Hadamard + Lloyd-Max keys, uniform values).
#define KV_TURBOQUANT_SUPPORTED 1

#include <cuda_fp16.h>

#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention_qdq.cuh"  // for TypeConverter<T>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace turboquant {

// =============================================================================
// TurboQuant CUDA kernels
// =============================================================================
//
// TurboQuant is a near-optimal KV cache quantization scheme. Per-slot K storage
// is a packed Lloyd-Max codebook index sequence + a single fp16 vec_norm.
// V storage is packed uniform asymmetric quant + (scale, zero) fp16 pair.
//
// References:
//   - vLLM Triton implementation:
//     vllm/v1/attention/ops/triton_turboquant_{store,decode}.py
//   - NumPy reference + paper validation:
//     onnxruntime/python/tools/quantization/turboquant_kv/
//
// Layout (per (token, head) slot, head_dim D, key_bits Bk, value_bits Bv):
//
//   K storage:
//     [packed_K_indices: ceil(D * Bk / 8) bytes][vec_norm: 2 bytes fp16]
//
//   V storage:
//     [packed_V_indices: ceil(D * Bv / 8) bytes][v_scale: 2 bytes fp16][v_zero: 2 bytes fp16]
//
// Bit-packing:
//   3-bit: 8 values into 3 bytes. byte0 = (v0) | (v1<<3) | (low2 of v2 << 6)... 24-bit LE word.
//   4-bit: pairs into 1 byte. lo nibble = even index, hi nibble = odd index.
//
// Decode path:
//   q_rot = q @ H            // Walsh-Hadamard, applied once per (layer, step)
//   for each cached token i:
//     y_hat = centroids[indices[i]]   // gather from constant LUT
//     score[i] = vec_norm[i] * dot(q_rot, y_hat)
//   No K reconstruction in original space — Hadamard is orthogonal so
//   dot(q, k) == dot(q@H, k_hat@H) * ||k||.

// -----------------------------------------------------------------------------
// Constants and traits.
// -----------------------------------------------------------------------------

constexpr int kTQThreadsPerBlock = 128;
constexpr int kTQMaxHeadDim = 256;     // Largest head_dim we currently support.
constexpr int kTQMaxCentroids = 16;    // 2 ** max(key_bits) = 2^4.

template <int kKeyBits>
struct TQKeyTraits {
  static_assert(kKeyBits == 3 || kKeyBits == 4, "TurboQuant supports 3- or 4-bit keys");
  static constexpr int kBits = kKeyBits;
  static constexpr int kCentroids = 1 << kKeyBits;
  static constexpr int kPackGroup = (kKeyBits == 3) ? 8 : 2;
  static constexpr int kPackBytes = (kKeyBits == 3) ? 3 : 1;
};

template <int kValueBits>
struct TQValueTraits {
  static_assert(kValueBits == 3 || kValueBits == 4, "TurboQuant supports 3- or 4-bit values");
  static constexpr int kBits = kValueBits;
  static constexpr int kLevels = 1 << kValueBits;
  static constexpr int kPackGroup = (kValueBits == 3) ? 8 : 2;
  static constexpr int kPackBytes = (kValueBits == 3) ? 3 : 1;
};

// Bytes per K slot for given (head_dim, key_bits).
__host__ __device__ inline int TQKeySlotBytes(int head_dim, int key_bits) {
  // ceil(D * Bk / 8) + 2 for vec_norm fp16
  return (head_dim * key_bits + 7) / 8 + 2;
}

// Bytes per V slot for given (head_dim, value_bits).
__host__ __device__ inline int TQValueSlotBytes(int head_dim, int value_bits) {
  // ceil(D * Bv / 8) + 4 for scale + zero fp16
  return (head_dim * value_bits + 7) / 8 + 4;
}

// -----------------------------------------------------------------------------
// Device helpers: bit-packing.
// -----------------------------------------------------------------------------

// Pack 8 3-bit indices [0,7] into 3 bytes. Indices passed as a uint32 word with
// 3 bits per index (already pre-shifted) is the most efficient form.
__device__ inline void TQPack3BitGroup(const uint8_t* idx, uint8_t* out) {
  uint32_t word = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    word |= (static_cast<uint32_t>(idx[i]) & 0x7u) << (i * 3);
  }
  out[0] = static_cast<uint8_t>(word & 0xFFu);
  out[1] = static_cast<uint8_t>((word >> 8) & 0xFFu);
  out[2] = static_cast<uint8_t>((word >> 16) & 0xFFu);
}

// Unpack 3 bytes into 8 3-bit indices.
__device__ inline void TQUnpack3BitGroup(const uint8_t* bytes, uint8_t* idx) {
  const uint32_t word = static_cast<uint32_t>(bytes[0]) |
                        (static_cast<uint32_t>(bytes[1]) << 8) |
                        (static_cast<uint32_t>(bytes[2]) << 16);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    idx[i] = static_cast<uint8_t>((word >> (i * 3)) & 0x7u);
  }
}

// Pack 2 4-bit indices [0,15] into 1 byte. lo = even, hi = odd.
__device__ inline uint8_t TQPack4BitPair(uint8_t lo, uint8_t hi) {
  return static_cast<uint8_t>((lo & 0xFu) | ((hi & 0xFu) << 4));
}

// Unpack 1 byte into 2 4-bit indices.
__device__ inline void TQUnpack4BitPair(uint8_t byte, uint8_t* lo, uint8_t* hi) {
  *lo = byte & 0xFu;
  *hi = (byte >> 4) & 0xFu;
}

// -----------------------------------------------------------------------------
// Device helpers: Lloyd-Max codebook lookup.
// -----------------------------------------------------------------------------

// Binary search the index of the nearest centroid for value `y`.
// `boundaries` is sorted ascending, length = kCentroids - 1.
template <int kCentroids>
__device__ inline uint8_t TQEncodeIndex(float y, const float* boundaries) {
  // For 8 or 16 centroids, an unrolled linear search is faster than binary
  // search and more ALU-friendly than a loop.
  uint8_t idx = 0;
  #pragma unroll
  for (int i = 0; i < kCentroids - 1; ++i) {
    idx += (y > boundaries[i]) ? 1 : 0;
  }
  return idx;
}

// -----------------------------------------------------------------------------
// Device helpers: Hadamard transform.
// -----------------------------------------------------------------------------

// In-shared-memory Walsh-Hadamard transform (FWHT) of length D (power of two).
// Each thread is responsible for a slice of D/blockDim.x elements; we cooperate
// across the warp via __syncthreads after each butterfly stage.
//
// Input: shared array `x[D]`, modified in place to H @ x where H is normalized
// Walsh-Hadamard (so that H @ H^T = I). Caller is responsible for the final
// 1/sqrt(D) scaling — we fold it into the codebook step instead.
template <int kHeadDim>
__device__ inline void TQHadamardInPlace(float* x) {
  // Sylvester FWHT, log2(D) butterfly stages.
  #pragma unroll
  for (int h = 1; h < kHeadDim; h *= 2) {
    int tid = threadIdx.x;
    // Each thread processes a pair (j, j+h) for some j in [0, D/2) where
    // (j / h) is even.
    for (int idx = tid; idx < kHeadDim / 2; idx += blockDim.x) {
      const int j = (idx / h) * 2 * h + (idx % h);
      const float a = x[j];
      const float b = x[j + h];
      x[j] = a + b;
      x[j + h] = a - b;
    }
    __syncthreads();
  }
  // Final normalization by 1/sqrt(D).
  const float inv_sqrt_d = rsqrtf(static_cast<float>(kHeadDim));
  for (int idx = threadIdx.x; idx < kHeadDim; idx += blockDim.x) {
    x[idx] *= inv_sqrt_d;
  }
  __syncthreads();
}

// -----------------------------------------------------------------------------
// Kernel: TurboQuant store (write-time encode + pack of K and V).
// -----------------------------------------------------------------------------
//
// Grid:  (n_tokens, n_kv_heads, batch)
// Block: kTQThreadsPerBlock
//
// Each block handles one (token, head) slot. Steps per block:
//   1. Load K[D] into shared, compute ||k|| via warp reduction.
//   2. Normalize x_hat = K / ||k||.
//   3. FWHT in shared: y = H @ x_hat.
//   4. Encode each y[j] to a Lloyd-Max codebook index using boundaries.
//   5. Pack indices and write to K cache slot.
//   6. Load V[D] into shared, compute (min, max) via warp reduction.
//   7. Quantize V uniformly, pack and write to V cache slot.
//   8. Write vec_norm, v_scale, v_zero as fp16 next to the packed bytes.
//
// All boundaries / centroids constants are passed via global memory pointers
// to avoid CUDA __constant__ limits when launching across kernels with
// different head_dims in the same module.
template <typename T, int kHeadDim, int kKeyBits, int kValueBits>
__global__ void TQStoreKernel(
    const T* __restrict__ K,            // (B, H, S, D) input keys
    const T* __restrict__ V,            // (B, H, S, D) input values
    const float* __restrict__ k_boundaries,  // (kCentroids - 1,) Lloyd-Max boundaries
    uint8_t* __restrict__ key_cache,    // raw bytes for K cache, slot-indexed
    uint8_t* __restrict__ value_cache,  // raw bytes for V cache, slot-indexed
    int batch_size,
    int seq_len,
    int n_kv_heads,
    int slot_bytes_k,
    int slot_bytes_v) {
  // Identify the slot.
  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int s = blockIdx.x;
  if (b >= batch_size || h >= n_kv_heads || s >= seq_len) return;

  using KT = TQKeyTraits<kKeyBits>;
  using VT = TQValueTraits<kValueBits>;

  __shared__ float smem_k[kHeadDim];
  __shared__ float smem_v[kHeadDim];
  __shared__ float k_norm_sq;
  __shared__ float v_min_smem;
  __shared__ float v_max_smem;

  // Load K and V into shared memory; convert from T (fp16/bf16) to fp32 for
  // numerical stability of the rotation + scoring.
  const int kv_stride = seq_len * n_kv_heads * kHeadDim;
  const int slot_offset = ((b * n_kv_heads + h) * seq_len + s) * kHeadDim;
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    smem_k[i] = TypeConverter<T>::to_float(K[slot_offset + i]);
    smem_v[i] = TypeConverter<T>::to_float(V[slot_offset + i]);
  }
  if (threadIdx.x == 0) {
    k_norm_sq = 0.0f;
    v_min_smem = 1e30f;
    v_max_smem = -1e30f;
  }
  __syncthreads();

  // Compute ||k||^2 via partial sums.
  float local_sq = 0.0f;
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    local_sq += smem_k[i] * smem_k[i];
  }
  atomicAdd(&k_norm_sq, local_sq);
  __syncthreads();
  const float vec_norm = sqrtf(k_norm_sq);
  const float inv_norm = (vec_norm > 1e-9f) ? (1.0f / vec_norm) : 1.0f;

  // Normalize K in place.
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    smem_k[i] *= inv_norm;
  }
  __syncthreads();

  // FWHT in shared.
  TQHadamardInPlace<kHeadDim>(smem_k);

  // Encode K via Lloyd-Max boundaries. Use shared memory to store the indices
  // before packing.
  __shared__ uint8_t k_indices[kHeadDim];
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    k_indices[i] = TQEncodeIndex<KT::kCentroids>(smem_k[i], k_boundaries);
  }
  __syncthreads();

  // Pack and write K indices.
  uint8_t* k_slot = key_cache + (((b * n_kv_heads + h) * seq_len + s) * slot_bytes_k);
  if constexpr (kKeyBits == 4) {
    for (int i = threadIdx.x; i < kHeadDim / 2; i += blockDim.x) {
      k_slot[i] = TQPack4BitPair(k_indices[2 * i], k_indices[2 * i + 1]);
    }
  } else if constexpr (kKeyBits == 3) {
    for (int g = threadIdx.x; g < kHeadDim / 8; g += blockDim.x) {
      TQPack3BitGroup(&k_indices[g * 8], k_slot + g * 3);
    }
  }

  // Write vec_norm fp16 right after the packed K indices.
  if (threadIdx.x == 0) {
    const int packed_k_bytes = (kHeadDim * kKeyBits + 7) / 8;
    half h_norm = __float2half(vec_norm);
    *reinterpret_cast<half*>(k_slot + packed_k_bytes) = h_norm;
  }
  __syncthreads();

  // Compute V min/max via partial reductions.
  float local_min = 1e30f;
  float local_max = -1e30f;
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    const float x = smem_v[i];
    local_min = fminf(local_min, x);
    local_max = fmaxf(local_max, x);
  }
  atomicMin(reinterpret_cast<int*>(&v_min_smem), __float_as_int(local_min));
  atomicMax(reinterpret_cast<int*>(&v_max_smem), __float_as_int(local_max));
  __syncthreads();

  const float v_min = v_min_smem;
  const float v_max = v_max_smem;
  const float v_scale = (v_max - v_min) / static_cast<float>(VT::kLevels - 1);
  const float inv_v_scale = (v_scale > 1e-12f) ? (1.0f / v_scale) : 0.0f;

  // Quantize V uniformly into shared `v_indices`.
  __shared__ uint8_t v_indices[kHeadDim];
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    int q = static_cast<int>(rintf((smem_v[i] - v_min) * inv_v_scale));
    q = max(0, min(VT::kLevels - 1, q));
    v_indices[i] = static_cast<uint8_t>(q);
  }
  __syncthreads();

  // Pack and write V indices.
  uint8_t* v_slot = value_cache + (((b * n_kv_heads + h) * seq_len + s) * slot_bytes_v);
  if constexpr (kValueBits == 4) {
    for (int i = threadIdx.x; i < kHeadDim / 2; i += blockDim.x) {
      v_slot[i] = TQPack4BitPair(v_indices[2 * i], v_indices[2 * i + 1]);
    }
  } else if constexpr (kValueBits == 3) {
    for (int g = threadIdx.x; g < kHeadDim / 8; g += blockDim.x) {
      TQPack3BitGroup(&v_indices[g * 8], v_slot + g * 3);
    }
  }

  // Write v_scale and v_zero (= v_min) fp16 right after the packed V indices.
  if (threadIdx.x == 0) {
    const int packed_v_bytes = (kHeadDim * kValueBits + 7) / 8;
    half h_scale = __float2half(v_scale);
    half h_zero = __float2half(v_min);
    *reinterpret_cast<half*>(v_slot + packed_v_bytes) = h_scale;
    *reinterpret_cast<half*>(v_slot + packed_v_bytes + 2) = h_zero;
  }
}

// -----------------------------------------------------------------------------
// Kernel: TurboQuant fused decode score (rotated-space attention).
// -----------------------------------------------------------------------------
//
// Grid:  (n_kv_heads, batch)
// Block: kTQThreadsPerBlock
//
// Computes attention scores for a single decode step (q_len == 1):
//
//   q_rot = q @ H              // rotate query, in shared memory
//   for each cached token i:
//     y_hat = centroids[indices[i]]
//     scores[i] = vec_norm[i] * dot(q_rot, y_hat)
//
// The k_centroids LUT is loaded into shared once and reused for all tokens.
// V is dequantized + softmax-weighted in a second pass (kept separate for
// numerical stability of online softmax).
template <typename T, int kHeadDim, int kKeyBits>
__global__ void TQDecodeScoreKernel(
    const T* __restrict__ Q,             // (B, H_q, 1, D) query
    const uint8_t* __restrict__ key_cache,
    const float* __restrict__ k_centroids,  // (kCentroids,) centroid values
    const int* __restrict__ seq_lens,    // (B,) actual sequence length per batch
    int n_kv_heads,
    int n_q_heads,
    int max_seq_len,
    int slot_bytes_k,
    bool norm_correction,
    float scale,
    float* __restrict__ scores  // (B, H_q, max_seq_len) attention logits
) {
  using KT = TQKeyTraits<kKeyBits>;

  const int b = blockIdx.y;
  const int h = blockIdx.x;
  if (b >= gridDim.y || h >= n_kv_heads) return;

  const int seq_len = seq_lens[b];

  __shared__ float q_rot[kHeadDim];
  __shared__ float centroids[KT::kCentroids];

  // Load and rotate Q for this head. For GQA, q_heads_per_kv = n_q_heads / n_kv_heads;
  // we need to rotate each q-head separately. For brevity, this kernel handles
  // one kv-head and assumes the caller handles q-head indexing externally.
  // (The full implementation will iterate q_heads_per_kv inside the kernel.)
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    q_rot[i] = TypeConverter<T>::to_float(Q[(b * n_q_heads + h) * kHeadDim + i]);
  }
  for (int i = threadIdx.x; i < KT::kCentroids; i += blockDim.x) {
    centroids[i] = k_centroids[i];
  }
  __syncthreads();

  TQHadamardInPlace<kHeadDim>(q_rot);

  // For each token in the cache, gather centroids by index and compute the dot.
  for (int s = blockIdx.z * blockDim.x + threadIdx.x; s < seq_len;
       s += blockDim.x * gridDim.z) {
    const uint8_t* k_slot = key_cache +
                            (((b * n_kv_heads + h) * max_seq_len + s) * slot_bytes_k);
    const int packed_k_bytes = (kHeadDim * kKeyBits + 7) / 8;
    const half h_norm = *reinterpret_cast<const half*>(k_slot + packed_k_bytes);
    const float vec_norm = __half2float(h_norm);

    float dot = 0.0f;
    if constexpr (kKeyBits == 4) {
      for (int i = 0; i < kHeadDim / 2; ++i) {
        uint8_t lo, hi;
        TQUnpack4BitPair(k_slot[i], &lo, &hi);
        dot += q_rot[2 * i] * centroids[lo];
        dot += q_rot[2 * i + 1] * centroids[hi];
      }
    } else if constexpr (kKeyBits == 3) {
      for (int g = 0; g < kHeadDim / 8; ++g) {
        uint8_t idx[8];
        TQUnpack3BitGroup(k_slot + g * 3, idx);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          dot += q_rot[g * 8 + j] * centroids[idx[j]];
        }
      }
    }

    if (norm_correction) {
      // Re-normalize the centroid vector to unit length so the reconstructed
      // key has the correct magnitude.
      float c_sq = 0.0f;
      if constexpr (kKeyBits == 4) {
        for (int i = 0; i < kHeadDim / 2; ++i) {
          uint8_t lo, hi;
          TQUnpack4BitPair(k_slot[i], &lo, &hi);
          c_sq += centroids[lo] * centroids[lo];
          c_sq += centroids[hi] * centroids[hi];
        }
      } else if constexpr (kKeyBits == 3) {
        for (int g = 0; g < kHeadDim / 8; ++g) {
          uint8_t idx[8];
          TQUnpack3BitGroup(k_slot + g * 3, idx);
          #pragma unroll
          for (int j = 0; j < 8; ++j) {
            c_sq += centroids[idx[j]] * centroids[idx[j]];
          }
        }
      }
      dot *= rsqrtf(c_sq);
    }

    scores[((b * n_q_heads + h) * max_seq_len) + s] = dot * vec_norm * scale;
  }
}

// -----------------------------------------------------------------------------
// Kernel: V dequant + softmax-weighted sum (the second decode pass).
// -----------------------------------------------------------------------------
//
// Standard pattern: output[d] = sum_i (softmax_weight[i] * v_dequant[i, d]).
// V dequant is uniform asymmetric: v_dequant = scale * idx + zero.
template <typename T, int kHeadDim, int kValueBits>
__global__ void TQDecodeWeightedSumKernel(
    const float* __restrict__ softmax_weights,  // (B, H_q, S) post-softmax scores
    const uint8_t* __restrict__ value_cache,
    const int* __restrict__ seq_lens,
    int n_kv_heads,
    int n_q_heads,
    int max_seq_len,
    int slot_bytes_v,
    T* __restrict__ output  // (B, H_q, D) attention output
) {
  using VT = TQValueTraits<kValueBits>;

  const int b = blockIdx.y;
  const int h = blockIdx.x;
  if (h >= n_kv_heads) return;

  const int seq_len = seq_lens[b];

  // Each thread accumulates a few D dims.
  for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < seq_len; ++s) {
      const uint8_t* v_slot = value_cache +
                              (((b * n_kv_heads + h) * max_seq_len + s) * slot_bytes_v);
      const int packed_v_bytes = (kHeadDim * kValueBits + 7) / 8;
      const half h_scale = *reinterpret_cast<const half*>(v_slot + packed_v_bytes);
      const half h_zero = *reinterpret_cast<const half*>(v_slot + packed_v_bytes + 2);
      const float v_scale = __half2float(h_scale);
      const float v_zero = __half2float(h_zero);

      uint8_t idx;
      if constexpr (kValueBits == 4) {
        const int byte_idx = d / 2;
        const uint8_t byte = v_slot[byte_idx];
        idx = (d % 2 == 0) ? (byte & 0xFu) : ((byte >> 4) & 0xFu);
      } else if constexpr (kValueBits == 3) {
        const int g = d / 8;
        const int j = d % 8;
        uint8_t group_idx[8];
        TQUnpack3BitGroup(v_slot + g * 3, group_idx);
        idx = group_idx[j];
      }
      const float v = v_scale * static_cast<float>(idx) + v_zero;
      const float w = softmax_weights[((b * n_q_heads + h) * max_seq_len) + s];
      acc += w * v;
    }
    output[((b * n_q_heads + h) * kHeadDim) + d] = static_cast<T>(acc);
  }
}

}  // namespace turboquant
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
