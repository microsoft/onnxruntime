// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"  // must precede attention_data.h for AllocatorPtr
#include "contrib_ops/cuda/bert/group_query_attention_turboquant.cuh"
#include "contrib_ops/cuda/bert/group_query_attention_turboquant_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"  // QkvToContext (fp16 path)
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"        // LaunchRotaryEmbeddingKernel
#include "contrib_ops/cuda/bert/unfused_attention.h"           // GetUnfusedAttentionWorkspaceSize
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"   // mha_fwd_kvcache (FlashAttention)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>  // for nvcuda::wmma (tensor cores) — used by v6 score path

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// =============================================================================
// TurboQuant CUDA orchestration.
//
// Strategy for v1: bulk dequant + standard fp16 attention.
//   1. Encode incoming K/V into the present cache (TurboQuant slot layout).
//   2. Decode the entire present cache back to fp16 K, V buffers in the
//      ORIGINAL (non-rotated) space.
//   3. Hand off to the existing QkvToContext<T, T> (fp16 cache) path.
//
// Memory win: cache stays compressed when not in use. Compute cost: every
// step pays an O(D x S) dequant; no in-rotated-space scoring (deferred to v2).
//
// Layout per (b, h, s) slot in the cache (uint8 last-dim = max(K_slot, V_slot)):
//   K slot: [packed_idx[ceil(D*kbits/8)] | vec_norm fp16 (2B)]
//   V slot: [packed_idx[ceil(D*vbits/8)] | v_scale fp16 (2B) | v_zero fp16 (2B)]
// The K and V tensors are *separate* tensors; both share the same last-dim.
// =============================================================================

namespace {

// v3: device-side fp16 -> fp32 codebook conversion.  Launched as <<<1, 32>>>
// with at most 16 active threads (4-bit) or 8 (3-bit).  Avoids a host sync
// per attention call.
template <typename T>
__global__ void TQConvertCodebookKernel(const T* __restrict__ src,
                                        float* __restrict__ dst,
                                        int n) {
  const int i = threadIdx.x;
  if (i < n) {
    dst[i] = static_cast<float>(src[i]);
  }
}

// -------- Encode: write new K/V tokens into cache slots ---------------------
//
// Grid: (new_seq_len, n_kv_heads, batch_size)
// Block: kTQThreadsPerBlock
//
// K_in / V_in shape (B, H_kv, new_seq, D), fp16.
// k_cache / v_cache shape (B, H_kv, max_seq, slot_last_dim), uint8 — we write
// into slots [past_seq_len, past_seq_len + new_seq_len).
template <typename T, int kHeadDim, int kKeyBits, int kValueBits>
__global__ void TQEncodeKernel(
    const T* __restrict__ K_in,        // (B, H_kv, new_seq, D)
    const T* __restrict__ V_in,        // (B, H_kv, new_seq, D)
    const float* __restrict__ k_codebook,
    int batch_size,
    int new_seq_len,
    int n_kv_heads,
    int max_seq_len,
    int past_seq_len,
    int slot_last_dim,
    uint8_t* __restrict__ k_cache,     // (B, H_kv, max_seq, slot_last_dim)
    uint8_t* __restrict__ v_cache) {   // (B, H_kv, max_seq, slot_last_dim)
  using namespace turboquant;
  using KT = TQKeyTraits<kKeyBits>;
  using VT = TQValueTraits<kValueBits>;
  constexpr int kPackedKBytes = (kHeadDim * kKeyBits + 7) / 8;
  constexpr int kPackedVBytes = (kHeadDim * kValueBits + 7) / 8;

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int s_new = blockIdx.x;
  if (b >= batch_size || h >= n_kv_heads || s_new >= new_seq_len) return;

  const int s_cache = past_seq_len + s_new;
  if (s_cache >= max_seq_len) return;

  __shared__ float smem_k[kHeadDim];
  __shared__ float smem_v[kHeadDim];
  __shared__ float reduce_buf[kTQThreadsPerBlock];
  __shared__ float scalars[3];   // [0]=norm, [1]=v_min, [2]=v_max
  __shared__ uint8_t k_indices[kHeadDim];
  __shared__ uint8_t v_indices[kHeadDim];

  // Read input K/V into shared memory.  ORT's GroupQueryAttention passes K/V
  // in BSNH layout (B, S, N_kv, H_kv) — the projection outputs are flat
  // (B, S, N_kv * H_kv) and stride accordingly.  Earlier this kernel used
  // BNSH stride (b * N_kv * S * H + h * S * H + s * H), which on a real
  // model interleaves K/V values from different (head, seq) pairs and
  // produces near-random encoded slots.  The Llama 0.99 cos sim numbers
  // were measured with a synthetic-Gaussian unit test that didn't exercise
  // this stride; LFM2 inference exposes it because it actually feeds real
  // BSNH-layout K/V through the kernel.
  const int in_off = ((b * new_seq_len + s_new) * n_kv_heads + h) * kHeadDim;
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    smem_k[i] = TypeConverter<T>::to_float(K_in[in_off + i]);
    smem_v[i] = TypeConverter<T>::to_float(V_in[in_off + i]);
  }
  __syncthreads();

  // ||k|| via tree reduction.
  reduce_buf[threadIdx.x] = 0.0f;
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    reduce_buf[threadIdx.x] += smem_k[i] * smem_k[i];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) scalars[0] = sqrtf(reduce_buf[0]);
  __syncthreads();
  const float vec_norm = scalars[0];
  const float inv_norm = (vec_norm > 1e-9f) ? (1.0f / vec_norm) : 1.0f;

  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    smem_k[i] *= inv_norm;
  }
  __syncthreads();

  // FWHT (rotate to TQ scoring space).
  TQHadamardInPlace<kHeadDim>(smem_k);

  // Encode K via Lloyd-Max midpoints (linear search, cheap for ≤16 centroids).
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    float y = smem_k[i];
    uint8_t idx = 0;
    #pragma unroll
    for (int j = 1; j < KT::kCentroids; ++j) {
      float midpoint = 0.5f * (k_codebook[j - 1] + k_codebook[j]);
      idx += (y > midpoint) ? 1 : 0;
    }
    k_indices[i] = idx;
  }
  __syncthreads();

  // Pack K indices and write to cache.
  uint8_t* k_slot = k_cache + (((b * n_kv_heads + h) * max_seq_len + s_cache) * slot_last_dim);
  if constexpr (kKeyBits == 4) {
    for (int i = threadIdx.x; i < kHeadDim / 2; i += blockDim.x) {
      k_slot[i] = TQPack4BitPair(k_indices[2 * i], k_indices[2 * i + 1]);
    }
  } else if constexpr (kKeyBits == 3) {
    for (int g = threadIdx.x; g < kHeadDim / 8; g += blockDim.x) {
      TQPack3BitGroup(&k_indices[g * 8], k_slot + g * 3);
    }
  }

  // Write vec_norm fp16 right after the packed K bytes.  Clamp to a value
  // safely inside fp16 range — for transformer K vectors, ||k|| can run into
  // the thousands per layer, and an overflow to fp16 +inf would propagate as
  // NaN through the rest of attention.
  if (threadIdx.x == 0) {
    float vn = vec_norm;
    if (!isfinite(vn)) vn = 0.0f;
    if (vn > 65000.0f) vn = 65000.0f;
    half h_norm = __float2half(vn);
    *reinterpret_cast<half*>(k_slot + kPackedKBytes) = h_norm;
  }

  // V min/max via tree reduction.
  float local_min = 1e30f, local_max = -1e30f;
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    local_min = fminf(local_min, smem_v[i]);
    local_max = fmaxf(local_max, smem_v[i]);
  }
  reduce_buf[threadIdx.x] = local_min;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) reduce_buf[threadIdx.x] = fminf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) scalars[1] = reduce_buf[0];
  __syncthreads();
  reduce_buf[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) reduce_buf[threadIdx.x] = fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) scalars[2] = reduce_buf[0];
  __syncthreads();

  const float v_min = scalars[1];
  const float v_scale_f = (scalars[2] - v_min) / static_cast<float>(VT::kLevels - 1);
  const float inv_v_scale = (v_scale_f > 1e-12f) ? (1.0f / v_scale_f) : 0.0f;

  // Encode V uniformly.
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    int q = static_cast<int>(rintf((smem_v[i] - v_min) * inv_v_scale));
    q = max(0, min(VT::kLevels - 1, q));
    v_indices[i] = static_cast<uint8_t>(q);
  }
  __syncthreads();

  uint8_t* v_slot = v_cache + (((b * n_kv_heads + h) * max_seq_len + s_cache) * slot_last_dim);
  if constexpr (kValueBits == 4) {
    for (int i = threadIdx.x; i < kHeadDim / 2; i += blockDim.x) {
      v_slot[i] = TQPack4BitPair(v_indices[2 * i], v_indices[2 * i + 1]);
    }
  } else if constexpr (kValueBits == 3) {
    for (int g = threadIdx.x; g < kHeadDim / 8; g += blockDim.x) {
      TQPack3BitGroup(&v_indices[g * 8], v_slot + g * 3);
    }
  }

  // Write v_scale, v_zero fp16 right after the packed V bytes.  Clamp to fp16
  // range: V values for some layers/tokens can exceed 65504, which would
  // saturate to fp16 +inf and cascade into NaN during decode/attention.
  if (threadIdx.x == 0) {
    float vs = v_scale_f;
    float vz = v_min;
    if (!isfinite(vs)) vs = 0.0f;
    if (!isfinite(vz)) vz = 0.0f;
    if (vs > 65000.0f) vs = 65000.0f;
    if (vs < -65000.0f) vs = -65000.0f;
    if (vz > 65000.0f) vz = 65000.0f;
    if (vz < -65000.0f) vz = -65000.0f;
    *reinterpret_cast<half*>(v_slot + kPackedVBytes) = __float2half(vs);
    *reinterpret_cast<half*>(v_slot + kPackedVBytes + 2) = __float2half(vz);
  }
}

// -------- Decode: read entire cache → fp16 K, V in ORIGINAL space ----------
//
// Grid: (total_seq_len, n_kv_heads, batch_size)
// Block: kTQThreadsPerBlock
//
// Reads from k_cache / v_cache (slot layout) for slots [0, total_seq_len).
// Writes K_out / V_out (B, H_kv, total_seq, D), fp16, in ORIGINAL space.
//
// K decode: idx → centroid (fp16) → multiply by vec_norm → result is in
// rotated space → apply H once more (Hadamard is self-inverse) → original space.
template <typename T, int kHeadDim, int kKeyBits, int kValueBits>
__global__ void TQDecodeKernel(
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const float* __restrict__ k_codebook,
    int batch_size,
    int total_seq_len,
    int n_kv_heads,
    int max_seq_len,
    int slot_last_dim,
    bool norm_correction,
    T* __restrict__ K_out,             // (B, H_kv, total_seq, D)
    T* __restrict__ V_out) {           // (B, H_kv, total_seq, D)
  using namespace turboquant;
  using KT = TQKeyTraits<kKeyBits>;
  using VT = TQValueTraits<kValueBits>;
  constexpr int kPackedKBytes = (kHeadDim * kKeyBits + 7) / 8;
  constexpr int kPackedVBytes = (kHeadDim * kValueBits + 7) / 8;

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int s = blockIdx.x;
  if (b >= batch_size || h >= n_kv_heads || s >= total_seq_len) return;

  __shared__ float smem_k[kHeadDim];
  __shared__ float reduce_buf[kTQThreadsPerBlock];
  __shared__ float scalars[1];

  const uint8_t* k_slot = k_cache + (((b * n_kv_heads + h) * max_seq_len + s) * slot_last_dim);
  const uint8_t* v_slot = v_cache + (((b * n_kv_heads + h) * max_seq_len + s) * slot_last_dim);

  // Decode K indices → smem_k (rotated space, scaled by vec_norm).
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    uint8_t idx;
    if constexpr (kKeyBits == 4) {
      uint8_t byte = k_slot[i / 2];
      idx = (i & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
    } else /* kKeyBits == 3 */ {
      // 8 indices per 3 bytes.
      const int g = i / 8;
      const int j = i % 8;
      const uint32_t word = static_cast<uint32_t>(k_slot[g * 3]) |
                            (static_cast<uint32_t>(k_slot[g * 3 + 1]) << 8) |
                            (static_cast<uint32_t>(k_slot[g * 3 + 2]) << 16);
      idx = static_cast<uint8_t>((word >> (j * 3)) & 0x7u);
    }
    smem_k[i] = k_codebook[idx];
  }
  __syncthreads();

  // Optional norm correction: compute ||y_hat|| over the centroid vector and
  // divide each entry by it. Equivalent to renormalizing the unit-vector
  // approximation before scaling by vec_norm.
  if (norm_correction) {
    reduce_buf[threadIdx.x] = 0.0f;
    for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
      reduce_buf[threadIdx.x] += smem_k[i] * smem_k[i];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
      __syncthreads();
    }
    // Guard against degenerate slots whose decoded centroid sum is ~0
    // (e.g. cache slots that were never written because past was zero-padded
    // or the K vector decoded to all-near-zero centroids).  Without the guard
    // rsqrtf(0) = +inf and the subsequent vec_norm * inf = 0/inf = NaN, which
    // propagates through attention and turns the entire model output into NaN.
    const float sum_sq = reduce_buf[0];
    const float nc = (sum_sq > 1e-30f) ? rsqrtf(sum_sq) : 0.0f;
    for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
      smem_k[i] *= nc;
    }
    __syncthreads();
  }

  // Read vec_norm and scale.
  if (threadIdx.x == 0) {
    half h_norm = *reinterpret_cast<const half*>(k_slot + kPackedKBytes);
    scalars[0] = __half2float(h_norm);
  }
  __syncthreads();
  const float vec_norm = scalars[0];

  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    smem_k[i] *= vec_norm;   // now in rotated space, scaled
  }
  __syncthreads();

  // Apply Hadamard once more — H is symmetric self-inverse, so this rotates
  // back to original space.
  TQHadamardInPlace<kHeadDim>(smem_k);

  // Write K_out (original space).
  const int out_off = ((b * n_kv_heads + h) * total_seq_len + s) * kHeadDim;
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    float kv = smem_k[i];
    if (!isfinite(kv)) kv = 0.0f;
    if (kv > 65000.0f) kv = 65000.0f;
    if (kv < -65000.0f) kv = -65000.0f;
    K_out[out_off + i] = static_cast<T>(kv);
  }

  // Decode V: indices → idx * scale + zero.
  __shared__ float v_meta[2];
  if (threadIdx.x == 0) {
    half h_scale = *reinterpret_cast<const half*>(v_slot + kPackedVBytes);
    half h_zero = *reinterpret_cast<const half*>(v_slot + kPackedVBytes + 2);
    v_meta[0] = __half2float(h_scale);
    v_meta[1] = __half2float(h_zero);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    uint8_t idx;
    if constexpr (kValueBits == 4) {
      uint8_t byte = v_slot[i / 2];
      idx = (i & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
    } else /* kValueBits == 3 */ {
      const int g = i / 8;
      const int j = i % 8;
      const uint32_t word = static_cast<uint32_t>(v_slot[g * 3]) |
                            (static_cast<uint32_t>(v_slot[g * 3 + 1]) << 8) |
                            (static_cast<uint32_t>(v_slot[g * 3 + 2]) << 16);
      idx = static_cast<uint8_t>((word >> (j * 3)) & 0x7u);
    }
    float v_hat = v_meta[0] * static_cast<float>(idx) + v_meta[1];
    // Clamp to fp16 range so ±inf can't escape decode and poison attention.
    if (!isfinite(v_hat)) v_hat = 0.0f;
    if (v_hat > 65000.0f) v_hat = 65000.0f;
    if (v_hat < -65000.0f) v_hat = -65000.0f;
    V_out[out_off + i] = static_cast<T>(v_hat);
  }
}

// -------- Static dispatch on (head_dim, key_bits, value_bits) --------------

template <typename T, int kHeadDim, int kKeyBits, int kValueBits>
Status LaunchEncodeDecodeFor(
    cudaStream_t stream,
    int batch_size, int n_kv_heads,
    int new_seq_len, int total_seq_len,
    int max_seq_len, int past_seq_len,
    int slot_last_dim, bool norm_correction,
    const T* K_in, const T* V_in, const float* k_codebook,
    uint8_t* k_cache, uint8_t* v_cache,
    T* K_out, T* V_out) {
  if (new_seq_len > 0) {
    dim3 grid_enc(new_seq_len, n_kv_heads, batch_size);
    dim3 block(turboquant::kTQThreadsPerBlock);
    TQEncodeKernel<T, kHeadDim, kKeyBits, kValueBits><<<grid_enc, block, 0, stream>>>(
        K_in, V_in, k_codebook,
        batch_size, new_seq_len, n_kv_heads, max_seq_len, past_seq_len, slot_last_dim,
        k_cache, v_cache);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return CUDA_CALL(err);
  }
  if (total_seq_len > 0) {
    dim3 grid_dec(total_seq_len, n_kv_heads, batch_size);
    dim3 block(turboquant::kTQThreadsPerBlock);
    TQDecodeKernel<T, kHeadDim, kKeyBits, kValueBits><<<grid_dec, block, 0, stream>>>(
        k_cache, v_cache, k_codebook,
        batch_size, total_seq_len, n_kv_heads, max_seq_len, slot_last_dim,
        norm_correction, K_out, V_out);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return CUDA_CALL(err);
  }
  return Status::OK();
}

// =============================================================================
// v7 — copy fresh fp16 K/V (post-RoPE for K) directly from BSNH input into
// the BNSH-layout K_out / V_out scratch buffers at slots [past_seq, total_seq).
//
// This skips the encode → decode round-trip for the new tokens of THIS forward
// pass — encode still writes those slots into the packed cache (so future
// decode steps can read them) but for THIS turn's attention we use the
// original, exact fp16 values.  Lossless for new tokens; matches v6 exactly
// for past tokens (which still go through the lossy decode path).
//
// Effect at long-context prompt step (past=0, new=S): the decode kernel
// processes zero slots and we just memcpy K/V from BSNH to BNSH layout.
// Effect at decode step (past=S, new=1): decode runs on S past slots, copy
// runs on 1 new slot — almost identical to v6 in that regime.
template <typename T>
__global__ void TQCopyFreshKVKernel(
    const T* __restrict__ K_in_bsnh,   // (B, S_new, H_kv, D)
    const T* __restrict__ V_in_bsnh,   // (B, S_new, H_kv, D)
    T* __restrict__ K_out_bnsh,        // (B, H_kv, total_seq, D) — slots [past, total)
    T* __restrict__ V_out_bnsh,
    int B, int S_new, int H_kv, int total_seq, int past_seq, int D) {
  const int s_new = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (s_new >= S_new || h >= H_kv || b >= B) return;
  const int tid = threadIdx.x;

  // BSNH input stride: ((b * S_new + s_new) * H_kv + h) * D
  const int in_off = ((b * S_new + s_new) * H_kv + h) * D;
  // BNSH output stride: ((b * H_kv + h) * total_seq + (past_seq + s_new)) * D
  const int out_off = ((b * H_kv + h) * total_seq + (past_seq + s_new)) * D;

  for (int i = tid; i < D; i += blockDim.x) {
    K_out_bnsh[out_off + i] = K_in_bsnh[in_off + i];
    V_out_bnsh[out_off + i] = V_in_bsnh[in_off + i];
  }
}

template <typename T>
Status LaunchCopyFreshKV(cudaStream_t stream,
                         const T* K_in_bsnh, const T* V_in_bsnh,
                         T* K_out_bnsh, T* V_out_bnsh,
                         int B, int S_new, int H_kv, int total_seq, int past_seq, int D) {
  if (S_new <= 0) return Status::OK();
  dim3 grid(S_new, H_kv, B);
  dim3 block(min(D, 256));
  TQCopyFreshKVKernel<T><<<grid, block, 0, stream>>>(
      K_in_bsnh, V_in_bsnh, K_out_bnsh, V_out_bnsh,
      B, S_new, H_kv, total_seq, past_seq, D);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status DispatchEncodeDecode(
    cudaStream_t stream,
    int batch_size, int n_kv_heads,
    int new_seq_len, int total_seq_len,
    int max_seq_len, int past_seq_len, int head_size,
    int key_bits, int value_bits,
    int slot_last_dim, bool norm_correction,
    const T* K_in, const T* V_in, const float* k_codebook,
    uint8_t* k_cache, uint8_t* v_cache,
    T* K_out, T* V_out) {
#define TQ_CASE(HD, KB, VB)                                                 \
  if (head_size == (HD) && key_bits == (KB) && value_bits == (VB)) {        \
    return LaunchEncodeDecodeFor<T, (HD), (KB), (VB)>(                      \
        stream, batch_size, n_kv_heads, new_seq_len, total_seq_len,         \
        max_seq_len, past_seq_len, slot_last_dim, norm_correction,          \
        K_in, V_in, k_codebook, k_cache, v_cache, K_out, V_out);            \
  }

  // NOTE: `total_seq_len` here is the count of slots to DECODE.  In v7 the
  // caller passes `past_seq_len` for that argument (decode handles past
  // slots only; new slots are written by LaunchCopyFreshKV separately).
  TQ_CASE(64, 4, 4)
  TQ_CASE(64, 3, 4)
  TQ_CASE(64, 3, 3)
  TQ_CASE(96, 4, 4)
  TQ_CASE(128, 4, 4)
  TQ_CASE(128, 3, 4)
  TQ_CASE(128, 3, 3)
  TQ_CASE(256, 4, 4)
  TQ_CASE(256, 3, 4)

#undef TQ_CASE
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "TurboQuant: unsupported (head_size, key_bits, value_bits) = (",
                         head_size, ", ", key_bits, ", ", value_bits, ")");
}

}  // namespace

// =============================================================================
// v2 attention kernels: simple custom CUDA attention over decoded fp16 K, V.
// =============================================================================
//
// Layout assumptions:
//   query     : [B, S_q, num_heads * head_size]  (BSNH packed, fp16)
//   k_full    : [B, num_kv_heads, total_seq, head_size]  (BNSH, fp16)
//   v_full    : [B, num_kv_heads, total_seq, head_size]  (BNSH, fp16)
//   scores    : [B, num_heads, S_q, total_seq]   (fp32 for softmax precision)
//   output    : [B, S_q, num_heads * head_size]  (BSNH packed, fp16)
//
// GQA: each query head h maps to kv head h_kv = h / (num_heads / num_kv_heads).
//
// Causal mask: token at position s_q (which corresponds to past_seq + s_q in the
// full sequence) attends to s_kv in [0, past_seq + s_q + 1).

template <typename T>
__global__ void TQScoresKernel(
    const T* __restrict__ query,
    const T* __restrict__ k_full,
    float* __restrict__ scores,
    int B, int S_q, int total_seq, int past_seq,
    int num_heads, int num_kv_heads, int head_size,
    float scale,
    int g_size /* = num_heads / num_kv_heads */) {
  // Block: (s_kv_chunk, h, b * S_q + s_q)
  // Each block computes one (b, h, s_q) row of scores against all s_kv.
  int bs = blockIdx.z;
  int b = bs / S_q;
  int s_q = bs % S_q;
  int h = blockIdx.y;
  if (b >= B || h >= num_heads || s_q >= S_q) return;

  int h_kv = h / g_size;
  int q_offset = ((b * S_q + s_q) * num_heads + h) * head_size;
  int k_base = (b * num_kv_heads + h_kv) * total_seq * head_size;
  int s_offset = ((b * num_heads + h) * S_q + s_q) * total_seq;

  // The "current" position of this query in the full sequence (past_seq + s_q).
  int q_pos = past_seq + s_q;

  for (int s_kv = blockIdx.x * blockDim.x + threadIdx.x; s_kv < total_seq;
       s_kv += blockDim.x * gridDim.x) {
    if (s_kv > q_pos) {
      // Causal mask.
      scores[s_offset + s_kv] = -INFINITY;
      continue;
    }
    float acc = 0.0f;
    for (int d = 0; d < head_size; ++d) {
      acc += static_cast<float>(query[q_offset + d]) *
             static_cast<float>(k_full[k_base + s_kv * head_size + d]);
    }
    scores[s_offset + s_kv] = acc * scale;
  }
}

template <typename T>
__global__ void TQSoftmaxRowKernel(
    float* __restrict__ scores,         // in-place; will hold softmax output as fp32 too
    int B, int num_heads, int S_q, int total_seq) {
  // One block per (b, h, s_q). Threads cooperate on softmax over total_seq.
  int bs = blockIdx.z;
  int b = bs / S_q;
  int s_q = bs % S_q;
  int h = blockIdx.y;
  if (b >= B || h >= num_heads || s_q >= S_q) return;

  int s_offset = ((b * num_heads + h) * S_q + s_q) * total_seq;
  float* row = scores + s_offset;

  // Find max.
  float local_max = -INFINITY;
  for (int s = threadIdx.x; s < total_seq; s += blockDim.x) {
    local_max = fmaxf(local_max, row[s]);
  }
  __shared__ float shared_max;
  if (threadIdx.x == 0) shared_max = local_max;
  __syncthreads();
  // Atomic-style reduction with one thread (small total_seq path; for large
  // total_seq the warp reduction would be faster but this is correct).
  __shared__ float reduce_buf[1024];
  reduce_buf[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] = fmaxf(reduce_buf[threadIdx.x], reduce_buf[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float row_max = reduce_buf[0];

  // Compute sum(exp(x - max)) and write exp(x - max) in-place.
  float local_sum = 0.0f;
  for (int s = threadIdx.x; s < total_seq; s += blockDim.x) {
    float v = expf(row[s] - row_max);
    row[s] = v;
    local_sum += v;
  }
  reduce_buf[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float row_sum = reduce_buf[0] > 0.0f ? reduce_buf[0] : 1.0f;

  // Normalize.
  for (int s = threadIdx.x; s < total_seq; s += blockDim.x) {
    row[s] /= row_sum;
  }
}

template <typename T>
__global__ void TQOutputKernel(
    const float* __restrict__ scores,    // [B, num_heads, S_q, total_seq] fp32
    const T* __restrict__ v_full,        // [B, num_kv_heads, total_seq, head_size]
    T* __restrict__ output,              // [B, S_q, num_heads * head_size]
    int B, int S_q, int total_seq,
    int num_heads, int num_kv_heads, int head_size,
    int g_size) {
  int bs = blockIdx.z;
  int b = bs / S_q;
  int s_q = bs % S_q;
  int h = blockIdx.y;
  if (b >= B || h >= num_heads || s_q >= S_q) return;

  int h_kv = h / g_size;
  int v_base = (b * num_kv_heads + h_kv) * total_seq * head_size;
  int s_offset = ((b * num_heads + h) * S_q + s_q) * total_seq;
  int o_offset = ((b * S_q + s_q) * num_heads + h) * head_size;

  for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
    float acc = 0.0f;
    for (int s_kv = 0; s_kv < total_seq; ++s_kv) {
      acc += scores[s_offset + s_kv] *
             static_cast<float>(v_full[v_base + s_kv * head_size + d]);
    }
    // Clamp to fp16 range — large ||V|| layers can otherwise produce inf which
    // turns the rest of the network into NaN.
    if (!isfinite(acc)) acc = 0.0f;
    if (acc > 65000.0f) acc = 65000.0f;
    if (acc < -65000.0f) acc = -65000.0f;
    output[o_offset + d] = static_cast<T>(acc);
  }
}

// =============================================================================
// v4 fused FlashAttention-style kernel.
//
// Replaces the v3.5 split (TQScores -> TQSoftmaxRow -> TQOutput) which
// materialised a [B, num_heads, S_q, total_seq] fp32 scores buffer.  At
// S=4096 that buffer is ~2 GB *per attention call* and dominates HBM traffic
// (we measured prompt-step at p=4096 running 7.4x slower than fp16
// FlashAttention purely because of this).  The fused kernel uses online
// softmax: one block per (b, h, s_q) output, walks K_full/V_full in tiles
// of kBlockK rows, keeps a running (max, sum, accumulator) trio in
// registers, and never writes the score matrix back to HBM.
//
// Algorithm (FlashAttention-2):
//   m = -inf, l = 0, acc = 0
//   for tile in [0..total_seq) step kBlockK:
//     load K_tile, V_tile into smem
//     for s in tile:
//       if causal-masked (s > q_pos): score = -inf
//       else: score = (Q . K_s) * scale
//     m_new = max(m, max_s(score))
//     alpha = exp(m - m_new);  l *= alpha;  acc *= alpha
//     for s: p_s = exp(score_s - m_new);  l += p_s;  acc += p_s * V_s
//     m = m_new
//   acc /= l
//   write acc to output
// =============================================================================
// v5 q-tiled FlashAttention kernel.
//
// Each block handles kBlockQ consecutive Q rows for one (b, h). The K/V
// rows of each tile are loaded ONCE and shared across all kBlockQ queries
// in the block, which reduces K/V HBM reads by kBlockQ-fold compared to
// v4-lite (where each Q row had its own block walking the whole cache).
//
// Threading invariant: blockDim.x == kHeadDim == kBlockK. In phase 1
// (score compute) thread tid plays the role of "K row tid", computing
// score[q][tid] = Q[q] . K_tile[tid] * scale.  In phase 2 (accumulator
// update) thread tid plays the role of "output dim tid", computing
// my_acc[q] += p[q][s] * V_tile[s][tid] for s in 0..kBlockK.
template <typename T, int kHeadDim, int kBlockK, int kBlockQ>
__global__ void TQFlashAttentionQTiledKernel(
    const T* __restrict__ query,
    const T* __restrict__ k_full,
    const T* __restrict__ v_full,
    T* __restrict__ output,
    int B, int S_q, int total_seq, int past_seq,
    int num_heads, int num_kv_heads, int head_size,
    float scale, int g_size) {
  static_assert(kBlockK == kHeadDim, "v5 kernel requires kBlockK == kHeadDim");
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int s_q_block = blockIdx.z;  // index into ceil(S_q / kBlockQ)
  if (b >= B || h >= num_heads) return;

  const int s_q_base = s_q_block * kBlockQ;
  if (s_q_base >= S_q) return;
  const int q_count = min(kBlockQ, S_q - s_q_base);  // last block may be partial

  const int h_kv = h / g_size;
  const int tid = threadIdx.x;

  // Smem layout.
  __shared__ T smem_q[kBlockQ][kHeadDim];
  __shared__ T smem_k[kBlockK][kHeadDim];
  __shared__ T smem_v[kBlockK][kHeadDim];
  __shared__ float smem_scores[kBlockQ][kBlockK];
  __shared__ float s_m[kBlockQ];     // running max per query
  __shared__ float s_l[kBlockQ];     // running denominator per query
  __shared__ float s_alpha[kBlockQ]; // exp(prev_m - new_m), set each tile

  if (tid < kBlockQ) { s_m[tid] = -INFINITY; s_l[tid] = 0.0f; }

  // Load Q rows for this block.  Each thread loads one element per query.
  for (int q = 0; q < q_count; ++q) {
    const int q_off = ((b * S_q + s_q_base + q) * num_heads + h) * kHeadDim;
    smem_q[q][tid] = query[q_off + tid];
  }
  __syncthreads();

  // Per-thread accumulator: one float per query, output dim == tid.
  float my_acc[kBlockQ];
  #pragma unroll
  for (int q = 0; q < kBlockQ; ++q) my_acc[q] = 0.0f;

  for (int tile = 0; tile < total_seq; tile += kBlockK) {
    const int tile_n = min(kBlockK, total_seq - tile);

    // Load K_tile and V_tile.  Thread tid loads K_tile[tid] and V_tile[tid].
    if (tid < tile_n) {
      const int row_off = (b * num_kv_heads + h_kv) * total_seq * kHeadDim
                        + (tile + tid) * kHeadDim;
      #pragma unroll
      for (int i = 0; i < kHeadDim; ++i) {
        smem_k[tid][i] = k_full[row_off + i];
        smem_v[tid][i] = v_full[row_off + i];
      }
    }
    __syncthreads();

    // Phase 1: compute scores. Thread tid computes score[q][tid] for all q.
    if (tid < tile_n) {
      const int s_kv_global = tile + tid;
      #pragma unroll
      for (int q = 0; q < q_count; ++q) {
        const int q_pos = past_seq + s_q_base + q;
        if (s_kv_global > q_pos) {
          smem_scores[q][tid] = -INFINITY;
        } else {
          float dot = 0.0f;
          #pragma unroll
          for (int i = 0; i < kHeadDim; ++i) {
            dot += static_cast<float>(smem_q[q][i]) * static_cast<float>(smem_k[tid][i]);
          }
          smem_scores[q][tid] = dot * scale;
        }
      }
    }
    __syncthreads();

    // Phase 2 prep: per-query running-max bookkeeping.  Thread q (q < kBlockQ)
    // does the serial reduction for query q.
    if (tid < q_count) {
      const int q = tid;
      float tile_max = -INFINITY;
      for (int s = 0; s < tile_n; ++s) tile_max = fmaxf(tile_max, smem_scores[q][s]);
      const float prev_m = s_m[q];
      const float m_new = fmaxf(prev_m, tile_max);
      s_alpha[q] = (prev_m == -INFINITY) ? 0.0f : expf(prev_m - m_new);
      float l_tile = 0.0f;
      for (int s = 0; s < tile_n; ++s) {
        float p = (m_new == -INFINITY) ? 0.0f : expf(smem_scores[q][s] - m_new);
        smem_scores[q][s] = p;
        l_tile += p;
      }
      s_l[q] = s_l[q] * s_alpha[q] + l_tile;
      s_m[q] = m_new;
    }
    __syncthreads();

    // Phase 2: each thread updates its own slice of my_acc for every query.
    if (tid < kHeadDim) {
      #pragma unroll
      for (int q = 0; q < q_count; ++q) {
        float new_contrib = 0.0f;
        for (int s = 0; s < tile_n; ++s) {
          new_contrib += smem_scores[q][s] * static_cast<float>(smem_v[s][tid]);
        }
        my_acc[q] = s_alpha[q] * my_acc[q] + new_contrib;
      }
    }
    __syncthreads();
  }

  // Normalize and write out.  One thread per output dim.
  if (tid < kHeadDim) {
    for (int q = 0; q < q_count; ++q) {
      float out = my_acc[q] / fmaxf(s_l[q], 1e-30f);
      if (!isfinite(out)) out = 0.0f;
      if (out > 65000.0f) out = 65000.0f;
      if (out < -65000.0f) out = -65000.0f;
      const int o_offset = ((b * S_q + s_q_base + q) * num_heads + h) * kHeadDim;
      output[o_offset + tid] = static_cast<T>(out);
    }
  }
}

// =============================================================================
// v6 q-tiled FlashAttention kernel — wmma tensor cores for the Q*K^T phase.
//
// The fragment math operates only on half-precision A/B with float
// accumulators on Ampere/Ada (sm_80+).  We dispatch this kernel only when
// T == half.  For bf16 / future paths, we fall through to v5.
//
// Tile shape: m=16, n=16, k=16 (the canonical Ampere fp16 mma shape).
// For (kBlockQ=32, kBlockK=64, kHeadDim=64) we have:
//   2 q-tiles × 4 s-tiles × 4 k-iterations = 32 mma_sync calls per K/V tile.
// Block has 2 warps (blockDim=64) so each warp owns half the (q-tile, s-tile)
// pairs.  Phase 2 (online softmax + V update) is unchanged from v5.
template <int kHeadDim, int kBlockK, int kBlockQ>
__global__ void TQFlashAttentionWmmaKernel(
    const half* __restrict__ query,
    const half* __restrict__ k_full,
    const half* __restrict__ v_full,
    half* __restrict__ output,
    int B, int S_q, int total_seq, int past_seq,
    int num_heads, int num_kv_heads, int head_size,
    float scale, int g_size) {
  using namespace nvcuda;
  static_assert(kBlockK == kHeadDim, "v6 kernel requires kBlockK == kHeadDim");
  static_assert(kHeadDim % 16 == 0, "v6 needs kHeadDim divisible by 16");
  static_assert(kBlockQ % 16 == 0, "v6 needs kBlockQ divisible by 16");
  static_assert(kBlockK % 16 == 0, "v6 needs kBlockK divisible by 16");

  constexpr int kMmaM = 16;
  constexpr int kMmaN = 16;
  constexpr int kMmaK = 16;
  constexpr int kQTiles = kBlockQ / kMmaM;        // e.g. 32 / 16 = 2
  constexpr int kSTiles = kBlockK / kMmaN;        // e.g. 64 / 16 = 4
  constexpr int kKIters = kHeadDim / kMmaK;       // e.g. 64 / 16 = 4

  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int s_q_block = blockIdx.z;
  if (b >= B || h >= num_heads) return;

  const int s_q_base = s_q_block * kBlockQ;
  if (s_q_base >= S_q) return;
  const int q_count = min(kBlockQ, S_q - s_q_base);

  const int h_kv = h / g_size;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int n_warps = blockDim.x / 32;

  __shared__ half smem_q[kBlockQ][kHeadDim];
  __shared__ half smem_k[kBlockK][kHeadDim];
  __shared__ half smem_v[kBlockK][kHeadDim];
  __shared__ float smem_scores[kBlockQ][kBlockK];
  __shared__ float s_m[kBlockQ];
  __shared__ float s_l[kBlockQ];
  __shared__ float s_alpha[kBlockQ];

  if (tid < kBlockQ) { s_m[tid] = -INFINITY; s_l[tid] = 0.0f; }

  // Vectorised Q load: each thread loads kBlockQ / blockDim Q elements.
  // For blockDim=64 and kBlockQ * kHeadDim = 32 * 64 = 2048 elements, that's
  // 32 elements per thread — covered by the simple per-thread loop below.
  for (int q = 0; q < q_count; ++q) {
    const int q_off = ((b * S_q + s_q_base + q) * num_heads + h) * kHeadDim;
    smem_q[q][tid] = query[q_off + tid];
  }
  __syncthreads();

  float my_acc[kBlockQ];
  #pragma unroll
  for (int q = 0; q < kBlockQ; ++q) my_acc[q] = 0.0f;

  // Causal early-exit cap: the highest q_pos in this Q block is
  // past_seq + s_q_base + q_count - 1.  K rows past that index are masked to
  // -inf for every query in the block, so we don't need to load or compute
  // them at all.  For the prompt step (Q rows uniform over 0..S-1) this
  // halves the work on average.
  const int max_q_pos = past_seq + s_q_base + q_count - 1;
  const int last_useful_kv = min(total_seq - 1, max_q_pos);

  for (int tile = 0; tile <= last_useful_kv; tile += kBlockK) {
    const int tile_n = min(kBlockK, total_seq - tile);

    if (tid < tile_n) {
      const int row_off = (b * num_kv_heads + h_kv) * total_seq * kHeadDim
                        + (tile + tid) * kHeadDim;
      // Vectorised uint4 (8-fp16) loads — see v4-lite kernel comment.
      static_assert(kHeadDim % 8 == 0, "vectorised load needs kHeadDim divisible by 8");
      const uint4* k_v4 = reinterpret_cast<const uint4*>(k_full + row_off);
      const uint4* v_v4 = reinterpret_cast<const uint4*>(v_full + row_off);
      uint4* smem_k_v4 = reinterpret_cast<uint4*>(&smem_k[tid][0]);
      uint4* smem_v_v4 = reinterpret_cast<uint4*>(&smem_v[tid][0]);
      #pragma unroll
      for (int i = 0; i < kHeadDim / 8; ++i) {
        smem_k_v4[i] = k_v4[i];
        smem_v_v4[i] = v_v4[i];
      }
    }
    __syncthreads();

    // ---- Phase 1: scores via wmma tensor cores ------------------------
    // Distribute (q_tile, s_tile) pairs across warps.  Each warp owns one
    // pair at a time and walks the K dimension in kKIters steps.
    constexpr int kTotalPairs = kQTiles * kSTiles;
    for (int pair = warp_id; pair < kTotalPairs; pair += n_warps) {
      const int q_tile = pair / kSTiles;
      const int s_tile = pair % kSTiles;
      const int q_row0 = q_tile * kMmaM;
      const int s_col0 = s_tile * kMmaN;

      wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, half, wmma::col_major> b_frag;
      wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> c_frag;
      wmma::fill_fragment(c_frag, 0.0f);

      #pragma unroll
      for (int kit = 0; kit < kKIters; ++kit) {
        const int k_off = kit * kMmaK;
        // A: smem_q rows [q_row0..q_row0+15], cols [k_off..k_off+15], row_major.
        wmma::load_matrix_sync(a_frag, &smem_q[q_row0][k_off], kHeadDim);
        // B: K^T treated as col_major over (k=head_dim, n=s) where the source
        //    storage is smem_k[s][k] (s outer).  Loading from
        //    &smem_k[s_col0][k_off] with leading dim = kHeadDim gives the
        //    correct stride: column n is at offset n*kHeadDim from base.
        wmma::load_matrix_sync(b_frag, &smem_k[s_col0][k_off], kHeadDim);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      // Store the 16x16 fp32 score tile to smem_scores[q_row0..][s_col0..]
      // with row-major leading dim = kBlockK.
      wmma::store_matrix_sync(&smem_scores[q_row0][s_col0], c_frag, kBlockK, wmma::mem_row_major);
    }
    __syncthreads();

    // Apply scaling + causal mask.  Same thread layout as v5: tid plays "K row".
    if (tid < tile_n) {
      const int s_kv_global = tile + tid;
      #pragma unroll
      for (int q = 0; q < q_count; ++q) {
        const int q_pos = past_seq + s_q_base + q;
        if (s_kv_global > q_pos) {
          smem_scores[q][tid] = -INFINITY;
        } else {
          smem_scores[q][tid] *= scale;
        }
      }
    }
    __syncthreads();

    // ---- Phase 2 prep: online softmax bookkeeping (per query) ---------
    if (tid < q_count) {
      const int q = tid;
      float tile_max = -INFINITY;
      for (int s = 0; s < tile_n; ++s) tile_max = fmaxf(tile_max, smem_scores[q][s]);
      const float prev_m = s_m[q];
      const float m_new = fmaxf(prev_m, tile_max);
      s_alpha[q] = (prev_m == -INFINITY) ? 0.0f : expf(prev_m - m_new);
      float l_tile = 0.0f;
      for (int s = 0; s < tile_n; ++s) {
        float p = (m_new == -INFINITY) ? 0.0f : expf(smem_scores[q][s] - m_new);
        smem_scores[q][s] = p;
        l_tile += p;
      }
      s_l[q] = s_l[q] * s_alpha[q] + l_tile;
      s_m[q] = m_new;
    }
    __syncthreads();

    // ---- Phase 3: V accumulator update --------------------------------
    if (tid < kHeadDim) {
      #pragma unroll
      for (int q = 0; q < q_count; ++q) {
        float new_contrib = 0.0f;
        for (int s = 0; s < tile_n; ++s) {
          new_contrib += smem_scores[q][s] * __half2float(smem_v[s][tid]);
        }
        my_acc[q] = s_alpha[q] * my_acc[q] + new_contrib;
      }
    }
    __syncthreads();
  }

  if (tid < kHeadDim) {
    for (int q = 0; q < q_count; ++q) {
      float out = my_acc[q] / fmaxf(s_l[q], 1e-30f);
      if (!isfinite(out)) out = 0.0f;
      if (out > 65000.0f) out = 65000.0f;
      if (out < -65000.0f) out = -65000.0f;
      const int o_offset = ((b * S_q + s_q_base + q) * num_heads + h) * kHeadDim;
      output[o_offset + tid] = __float2half(out);
    }
  }
}

template <typename T, int kHeadDim, int kBlockK>
__global__ void TQFlashAttentionKernel(
    const T* __restrict__ query,
    const T* __restrict__ k_full,
    const T* __restrict__ v_full,
    T* __restrict__ output,
    int B, int S_q, int total_seq, int past_seq,
    int num_heads, int num_kv_heads, int head_size,
    float scale, int g_size) {
  const int bs = blockIdx.z;
  const int b = bs / S_q;
  const int s_q = bs % S_q;
  const int h = blockIdx.y;
  if (b >= B || h >= num_heads || s_q >= S_q) return;

  const int h_kv = h / g_size;
  const int q_pos = past_seq + s_q;

  const int q_offset = ((b * S_q + s_q) * num_heads + h) * kHeadDim;
  const int k_base = (b * num_kv_heads + h_kv) * total_seq * kHeadDim;
  const int o_offset = ((b * S_q + s_q) * num_heads + h) * kHeadDim;

  // Block layout: blockDim.x == kHeadDim. Each thread owns one head dim.
  // (Works when kBlockK <= kHeadDim, which we ensure by template params.)
  static_assert(kBlockK <= kHeadDim, "kBlockK must be <= kHeadDim for this thread layout");
  const int tid = threadIdx.x;

  __shared__ T smem_q[kHeadDim];
  __shared__ T smem_k[kBlockK][kHeadDim];
  __shared__ T smem_v[kBlockK][kHeadDim];
  __shared__ float smem_scores[kBlockK];   // also used to hold p_s after softmax
  __shared__ float s_m;       // running max
  __shared__ float s_l;       // running denominator
  __shared__ float s_alpha;   // exp(prev_m - new_m), shared each tile

  if (tid == 0) { s_m = -INFINITY; s_l = 0.0f; }

  // Load Q once.  One element per thread.
  if (tid < kHeadDim) smem_q[tid] = query[q_offset + tid];

  float my_acc = 0.0f;   // this thread's slice of the output accumulator (1 dim)

  for (int tile = 0; tile < total_seq; tile += kBlockK) {
    const int tile_end = min(tile + kBlockK, total_seq);
    const int tile_n = tile_end - tile;

    // Vectorised K/V load: each thread loads ONE full row via uint4 (16-byte
    // = 8-fp16 per transaction).  Single coalesced HBM burst per row.
    static_assert(kHeadDim % 8 == 0, "vectorised load needs kHeadDim divisible by 8");
    if (tid < tile_n) {
      const int row_off = k_base + (tile + tid) * kHeadDim;
      const uint4* k_v4 = reinterpret_cast<const uint4*>(k_full + row_off);
      const uint4* v_v4 = reinterpret_cast<const uint4*>(v_full + row_off);
      uint4* smem_k_v4 = reinterpret_cast<uint4*>(&smem_k[tid][0]);
      uint4* smem_v_v4 = reinterpret_cast<uint4*>(&smem_v[tid][0]);
      #pragma unroll
      for (int i = 0; i < kHeadDim / 8; ++i) {
        smem_k_v4[i] = k_v4[i];
        smem_v_v4[i] = v_v4[i];
      }
    }
    __syncthreads();

    // (cp.async double-buffered loads were tried and regressed at our tile
    //  size: per-tile commit_group/wait_group overhead exceeds the overlap
    //  savings when the post-load compute is small.  See git history.)

    // Each thread computes the score for one s in the tile.
    if (tid < tile_n) {
      const int s = tid;
      const int s_kv_global = tile + s;
      float dot = 0.0f;
      #pragma unroll
      for (int i = 0; i < kHeadDim; ++i) {
        dot += static_cast<float>(smem_q[i]) * static_cast<float>(smem_k[s][i]);
      }
      dot *= scale;
      if (s_kv_global > q_pos) dot = -INFINITY;  // causal
      smem_scores[s] = dot;
    }
    __syncthreads();

    // Thread 0 does the running-max bookkeeping for this tile.
    if (tid == 0) {
      float tile_max = -INFINITY;
      for (int s = 0; s < tile_n; ++s) tile_max = fmaxf(tile_max, smem_scores[s]);
      const float prev_m = s_m;
      const float m_new = fmaxf(prev_m, tile_max);
      s_alpha = (prev_m == -INFINITY) ? 0.0f : expf(prev_m - m_new);
      float l_tile = 0.0f;
      for (int s = 0; s < tile_n; ++s) {
        float p = (m_new == -INFINITY) ? 0.0f : expf(smem_scores[s] - m_new);
        smem_scores[s] = p;
        l_tile += p;
      }
      s_l = s_l * s_alpha + l_tile;
      s_m = m_new;
    }
    __syncthreads();

    // All threads update their own slice of the accumulator.
    if (tid < kHeadDim) {
      float new_contrib = 0.0f;
      for (int s = 0; s < tile_n; ++s) {
        new_contrib += smem_scores[s] * static_cast<float>(smem_v[s][tid]);
      }
      my_acc = s_alpha * my_acc + new_contrib;
    }
    __syncthreads();
  }

  // Normalize and write out.
  if (tid < kHeadDim) {
    float out = my_acc / fmaxf(s_l, 1e-30f);
    if (!isfinite(out)) out = 0.0f;
    if (out > 65000.0f) out = 65000.0f;
    if (out < -65000.0f) out = -65000.0f;
    output[o_offset + tid] = static_cast<T>(out);
  }
}

template <typename T>
Status LaunchTQAttention(cudaStream_t stream,
                         int B, int S_q, int total_seq, int past_seq,
                         int num_heads, int num_kv_heads, int head_size,
                         float scale,
                         const T* query, const T* k_full, const T* v_full,
                         float* scores, T* output) {
  const int g_size = num_heads / num_kv_heads;
  (void)scores;  // unused in the fused path

  // v6 wmma kernel for the prompt step on fp16 inputs.  Falls through to v5
  // for bf16 / non-fp16 paths.
  if constexpr (std::is_same<T, half>::value) {
    if (head_size == 64 && S_q > 1) {
      constexpr int kHeadDim = 64;
      constexpr int kBlockK = 64;
      constexpr int kBlockQ = 32;  // larger sizes (64) hurt occupancy at 32K via reg pressure
      dim3 grid(B, num_heads, (S_q + kBlockQ - 1) / kBlockQ);
      dim3 block(kHeadDim);
      TQFlashAttentionWmmaKernel<kHeadDim, kBlockK, kBlockQ><<<grid, block, 0, stream>>>(
          query, k_full, v_full, output, B, S_q, total_seq, past_seq,
          num_heads, num_kv_heads, head_size, scale, g_size);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
      return Status::OK();
    }
  }
  // v5 q-tiled kernel for the prompt step (S_q > 1): each block handles a
  // group of consecutive Q rows, sharing K/V loads across them.  For
  // decode (S_q == 1) we drop to v4-lite (one block per s_q is already
  // optimal there).
  if (head_size == 64 && S_q > 1) {
    constexpr int kHeadDim = 64;
    constexpr int kBlockK = 64;
    constexpr int kBlockQ = 32;  // tuned: bigger values help up to 32, then occupancy drops
    dim3 grid(B, num_heads, (S_q + kBlockQ - 1) / kBlockQ);
    dim3 block(kHeadDim);
    TQFlashAttentionQTiledKernel<T, kHeadDim, kBlockK, kBlockQ><<<grid, block, 0, stream>>>(
        query, k_full, v_full, output, B, S_q, total_seq, past_seq,
        num_heads, num_kv_heads, head_size, scale, g_size);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return Status::OK();
  }
  // (head_dim=128 q-tiling deferred — needs kBlockK split from kHeadDim to
  // fit shared memory.  Falls through to v4-lite below for hd=128 prompts.)
  // v4 fused kernel: decode-step path (S_q == 1) — one block per (b, h),
  // online softmax over K/V tiles. blockDim == kHeadDim.
  if (head_size == 64) {
    constexpr int kHeadDim = 64;
    constexpr int kBlockK = 64;
    dim3 grid(1, num_heads, B * S_q);
    dim3 block(kHeadDim);
    TQFlashAttentionKernel<T, kHeadDim, kBlockK><<<grid, block, 0, stream>>>(
        query, k_full, v_full, output, B, S_q, total_seq, past_seq,
        num_heads, num_kv_heads, head_size, scale, g_size);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return Status::OK();
  }
  if (head_size == 128) {
    constexpr int kHeadDim = 128;
    constexpr int kBlockK = 64;
    dim3 grid(1, num_heads, B * S_q);
    dim3 block(kHeadDim);
    TQFlashAttentionKernel<T, kHeadDim, kBlockK><<<grid, block, 0, stream>>>(
        query, k_full, v_full, output, B, S_q, total_seq, past_seq,
        num_heads, num_kv_heads, head_size, scale, g_size);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return Status::OK();
  }

  // Fallback to the v3.5 split kernels for unsupported head sizes (96, 256, …).
  // Scores kernel.
  {
    dim3 grid((total_seq + 255) / 256, num_heads, B * S_q);
    dim3 block(256);
    TQScoresKernel<T><<<grid, block, 0, stream>>>(
        query, k_full, scores, B, S_q, total_seq, past_seq,
        num_heads, num_kv_heads, head_size, scale, g_size);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }
  // Softmax kernel.
  {
    int block_size = 256;
    while (block_size > total_seq && block_size > 32) block_size /= 2;
    dim3 grid(1, num_heads, B * S_q);
    dim3 block(block_size);
    TQSoftmaxRowKernel<T><<<grid, block, 0, stream>>>(
        scores, B, num_heads, S_q, total_seq);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }
  // Output kernel.
  {
    dim3 grid(1, num_heads, B * S_q);
    dim3 block(64);  // == head_size, simple
    TQOutputKernel<T><<<grid, block, 0, stream>>>(
        scores, v_full, output, B, S_q, total_seq,
        num_heads, num_kv_heads, head_size, g_size);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }
  return Status::OK();
}

// =============================================================================
// LaunchTurboQuantAttention: full attention via bulk dequant + standard fp16.
// =============================================================================

template <typename T, typename U>
Status LaunchTurboQuantAttention(
    const cudaDeviceProp& device_prop,
    Stream* stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T, U>& data) {
  static_assert(std::is_same<U, uint8_t>::value,
                "TurboQuant only supports uint8 cache type (T_CACHE = uint8)");

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream->GetHandle());

  const int B = parameters.batch_size;
  const int H_kv = parameters.kv_num_heads;
  const int new_seq = parameters.sequence_length;
  const int past_seq = parameters.seqlen_past_kv_cache;
  const int total_seq = past_seq + new_seq;
  const int max_seq = parameters.seqlen_present_kv_cache;
  const int D = parameters.head_size;
  const int kbits = parameters.key_quant_bits;
  const int vbits = parameters.value_quant_bits;

  // Cache slot last dim = max(K_slot, V_slot).
  const int k_slot = (D * kbits + 7) / 8 + 2;
  const int v_slot = (D * vbits + 7) / 8 + 4;
  const int slot_last_dim = (k_slot > v_slot) ? k_slot : v_slot;

  // Materialize codebook as fp32 device buffer (small, max 16 entries).
  // v3: do this entirely on-device so we don't force a host sync on every
  // attention call. The codebook itself is constant per session, but we
  // re-derive the fp32 view per call to keep the orchestrator stateless.
  // 16 floats = 64 bytes; this is dominated by the kernel launch latency.
  const int n_centroids = 1 << kbits;
  float* d_codebook = nullptr;
  CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_codebook, n_centroids * sizeof(float), cuda_stream));
  TQConvertCodebookKernel<T><<<1, 32, 0, cuda_stream>>>(data.k_codebook, d_codebook, n_centroids);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  // v2 PATH: encode incoming K/V → present cache (compressed bytes), decode
  // entire present cache → fp16 K/V buffers, run attention math directly via
  // our simple custom kernels (TQScoresKernel + TQSoftmaxRowKernel + TQOutputKernel),
  // then write to data.output. This produces real attention results with real
  // tokens — same output a fp16 GQA would give, modulo the lossy cache.

  // Apply RoPE to incoming Q and K when the GQA op was configured with
  // do_rotary=1 (e.g. LFM2/LFM2.5/Qwen3 ONNX exports inline RoPE inside the
  // GQA op).  The standard QkvToContext path does this via
  // LaunchUnpackRoPEAppend; the TQ path bypasses that, so without this step
  // the cache encodes positionless K and the model produces near-random
  // logits even though no NaN appears.
  T* d_Q_rot = nullptr;
  T* d_K_rot = nullptr;
  int* d_past_seq_lens = nullptr;
  // The shared LaunchRotaryEmbeddingKernel is instantiated for `half` and ORT's
  // `BFloat16` wrapper, NOT for the CUDA `__nv_bfloat16` we use here. Skip the
  // bf16 path — wandler today only ships fp16 LLMs through TurboQuant and
  // adding the bf16 instantiation would require touching the shared rotary
  // template definitions. The fp16 path correctly applies inline RoPE.
  const bool apply_rotary = parameters.do_rotary && std::is_same<T, half>::value;
  if constexpr (std::is_same<T, half>::value) {
   if (apply_rotary) {
    // Q has shape (B, S_q, num_heads * D), K has shape (B, S_q, kv_num_heads * D), both BSNH.
    const size_t q_elems = static_cast<size_t>(B) * new_seq * parameters.num_heads * D;
    const size_t k_elems = static_cast<size_t>(B) * new_seq * H_kv * D;
    CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_Q_rot, q_elems * sizeof(T), cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_K_rot, k_elems * sizeof(T), cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_past_seq_lens, B * sizeof(int), cuda_stream));
    // past_sequence_lengths[b] = past_seq for every batch element.
    {
      std::vector<int> host_past(B, past_seq);
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(d_past_seq_lens, host_past.data(),
                                           B * sizeof(int), cudaMemcpyHostToDevice, cuda_stream));
    }
    const int rotary_dim = (parameters.rotary_dim > 0) ? parameters.rotary_dim : D;
    constexpr int kPositionIdsFormat = 2;  // use past_sequence_lengths[b] + s
    constexpr int kMaxSequenceLength = 1 << 20;  // bounds-check upper limit; only used by formats 0/1
    ORT_RETURN_IF_ERROR((LaunchRotaryEmbeddingKernel<T>(
        cuda_stream, d_Q_rot, data.query, /*position_ids=*/nullptr, d_past_seq_lens,
        data.cos_cache, data.sin_cache, B, new_seq, parameters.num_heads, D,
        rotary_dim, kMaxSequenceLength, kPositionIdsFormat, parameters.rotary_interleaved,
        device_prop.maxThreadsPerBlock, /*is_input_bnsh_format=*/false)));
    ORT_RETURN_IF_ERROR((LaunchRotaryEmbeddingKernel<T>(
        cuda_stream, d_K_rot, data.key, /*position_ids=*/nullptr, d_past_seq_lens,
        data.cos_cache, data.sin_cache, B, new_seq, H_kv, D,
        rotary_dim, kMaxSequenceLength, kPositionIdsFormat, parameters.rotary_interleaved,
        device_prop.maxThreadsPerBlock, /*is_input_bnsh_format=*/false)));
   }
  }

  // Allocate temporary fp16 K/V buffers for the full cache.
  const size_t kv_elements = static_cast<size_t>(B) * H_kv * total_seq * D;
  T *d_K_fp16 = nullptr, *d_V_fp16 = nullptr;
  CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_K_fp16, kv_elements * sizeof(T), cuda_stream));
  CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_V_fp16, kv_elements * sizeof(T), cuda_stream));

  // Copy past_key/past_value content into the first past_seq slots of present_key/value.
  // The encode kernel will only write the new [past_seq, total_seq) slots; the decode
  // kernel reads ALL [0, total_seq) slots, so the past must already live in present.
  // (When past_present_share_buffer is true the same buffer is reused and this copy
  // is a no-op for content but still safe — both pointers refer to the same memory.)
  if (past_seq > 0 && data.past_key != nullptr && data.past_value != nullptr &&
      reinterpret_cast<const void*>(data.past_key) !=
          reinterpret_cast<const void*>(data.present_key)) {
    const size_t past_bytes_per_layer =
        static_cast<size_t>(B) * H_kv * past_seq * slot_last_dim;
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        data.present_key, data.past_key, past_bytes_per_layer,
        cudaMemcpyDeviceToDevice, cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        data.present_value, data.past_value, past_bytes_per_layer,
        cudaMemcpyDeviceToDevice, cuda_stream));
  }

  // v7 path:
  //   1. Encode new K/V into the packed cache for future decode steps to read.
  //   2. Decode runs on PAST slots only — the new slots get filled by step 3.
  //   3. Copy fresh fp16 K/V (post-RoPE for K) from BSNH input straight into
  //      the BNSH K_out / V_out scratch buffer at slots [past_seq, total_seq).
  const T* k_in = apply_rotary ? d_K_rot : data.key;
  Status status = DispatchEncodeDecode<T>(
      cuda_stream, B, H_kv, new_seq, /*decode_count=*/past_seq,
      max_seq, past_seq, D,
      kbits, vbits, slot_last_dim, parameters.norm_correction,
      k_in, data.value, d_codebook,
      reinterpret_cast<uint8_t*>(data.present_key),
      reinterpret_cast<uint8_t*>(data.present_value),
      d_K_fp16, d_V_fp16);

  if (status.IsOK()) {
    Status copy_status = LaunchCopyFreshKV<T>(
        cuda_stream, k_in, data.value, d_K_fp16, d_V_fp16,
        B, new_seq, H_kv, total_seq, past_seq, D);
    if (!copy_status.IsOK()) status = copy_status;
  }

  // Option ε: for first-prompt steps (past_seq == 0, S_q > 1), delegate the
  // attention math to ORT's standard FlashAttention.  We have the EXACT
  // pre-quantization fp16 K/V (in d_K_rot / data.value or data.key /
  // data.value) — running FA on those is much faster than walking the
  // packed cache with our custom kernel.  The encode kernel above has
  // already populated the packed cache for FUTURE decode steps to read.
  //
  // This mirrors vLLM's TurboQuant prefill path (turboquant_attn.py:542):
  // first-chunk prefill never touches the quantized cache; FA runs on the
  // raw new K/V.  We only fall through to the custom TQ kernel for
  // continuation prefill / decode steps where past tokens are in the cache.
  if constexpr (std::is_same<T, half>::value) {
    if (status.IsOK() && past_seq == 0 && new_seq > 1 &&
        data.softmax_lse != nullptr && data.padded_seq_lens != nullptr) {
      // FA "no internal append" mode: pass kcache/vcache fully populated
      // (d_K_fp16 / d_V_fp16 already hold the post-RoPE K and raw V from
      // LaunchCopyFreshKV) plus nullptr for new_k/new_v.  seqlens_k passes
      // the TOTAL length already in kcache (= padded_seq_lens for first
      // prompt) — same convention the standard GQA FlashAttention path uses.
      void* q_in_ptr = const_cast<half*>(reinterpret_cast<const half*>(
          apply_rotary ? d_Q_rot : data.query));
      Status fa_status = onnxruntime::flash::mha_fwd_kvcache(
          device_prop, cuda_stream,
          /*q=*/q_in_ptr,
          /*kcache=*/d_K_fp16,
          /*vcache=*/d_V_fp16,
          /*new_k=*/nullptr,        // already appended via LaunchCopyFreshKV
          /*new_v=*/nullptr,
          /*out=*/data.output,
          /*softmax_lse=*/reinterpret_cast<void*>(data.softmax_lse),
          /*seqlens_k=*/data.padded_seq_lens,
          /*rotary_cos=*/nullptr,   // pre-rotated tensors passed
          /*rotary_sin=*/nullptr,
          /*cache_batch_idx=*/nullptr,
          /*leftpad_k=*/nullptr,
          /*head_sink=*/nullptr,
          /*block_table=*/nullptr,
          B, parameters.num_heads, H_kv, D,
          /*seqlen_q=*/new_seq,
          /*seqlen_k=*/total_seq,
          /*seqlen_k_new=*/0,       // 0 because not appending
          /*rotary_dim=*/0,
          /*scale=*/(parameters.scale != 0.0f) ? parameters.scale
                                               : (1.0f / sqrtf(static_cast<float>(D))),
          /*softcap=*/parameters.softcap,
          /*is_causal=*/true,
          /*is_bf16=*/false,
          /*use_smooth_softmax=*/parameters.use_smooth_softmax,
          /*past_bsnh=*/false,      // d_K_fp16 / d_V_fp16 are BNSH
          /*num_splits=*/parameters.num_splits,
          /*lse_accum=*/reinterpret_cast<void*>(data.softmax_lse_accum),
          /*out_accum=*/reinterpret_cast<void*>(data.out_accum),
          /*local_window=*/parameters.local_window_size - 1,
          /*is_rotary_interleaved=*/false,
          /*is_packed_qkv=*/false);
      cudaFreeAsync(d_codebook, cuda_stream);
      cudaFreeAsync(d_K_fp16, cuda_stream);
      cudaFreeAsync(d_V_fp16, cuda_stream);
      if (d_Q_rot != nullptr) cudaFreeAsync(d_Q_rot, cuda_stream);
      if (d_K_rot != nullptr) cudaFreeAsync(d_K_rot, cuda_stream);
      if (d_past_seq_lens != nullptr) cudaFreeAsync(d_past_seq_lens, cuda_stream);
      return fa_status;
    }
  }

  if (!status.IsOK()) {
    cudaFreeAsync(d_codebook, cuda_stream);
    cudaFreeAsync(d_K_fp16, cuda_stream);
    cudaFreeAsync(d_V_fp16, cuda_stream);
    if (d_Q_rot != nullptr) cudaFreeAsync(d_Q_rot, cuda_stream);
    if (d_K_rot != nullptr) cudaFreeAsync(d_K_rot, cuda_stream);
    if (d_past_seq_lens != nullptr) cudaFreeAsync(d_past_seq_lens, cuda_stream);
    return status;
  }

  // Step 3: allocate scores buffer [B, num_heads, S_q, total_seq] fp32 only
  // for the v3.5 fallback path (head sizes other than 64/128). The v4 fused
  // kernel does online softmax in registers and never touches HBM scores.
  const bool use_fused_attention = (D == 64 || D == 128);
  const size_t scores_elements =
      use_fused_attention ? 0 : (static_cast<size_t>(B) * parameters.num_heads * new_seq * total_seq);
  float* d_scores = nullptr;
  if (scores_elements > 0) {
    CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_scores, scores_elements * sizeof(float), cuda_stream));
  }

  const float scale =
      (parameters.scale != 0.0f) ? parameters.scale : (1.0f / sqrtf(static_cast<float>(D)));

  // Use the RoPE-rotated Q when do_rotary=1.
  const T* q_in = apply_rotary ? d_Q_rot : data.query;
  Status attn_status = LaunchTQAttention<T>(cuda_stream, B, new_seq, total_seq, past_seq,
                                            parameters.num_heads, H_kv, D, scale,
                                            q_in, d_K_fp16, d_V_fp16,
                                            d_scores, data.output);
  status = attn_status;

  cudaFreeAsync(d_codebook, cuda_stream);
  cudaFreeAsync(d_K_fp16, cuda_stream);
  cudaFreeAsync(d_V_fp16, cuda_stream);
  if (d_scores != nullptr) cudaFreeAsync(d_scores, cuda_stream);
  if (d_Q_rot != nullptr) cudaFreeAsync(d_Q_rot, cuda_stream);
  if (d_K_rot != nullptr) cudaFreeAsync(d_K_rot, cuda_stream);
  if (d_past_seq_lens != nullptr) cudaFreeAsync(d_past_seq_lens, cuda_stream);
  return status;
}

#if 0  // V2 code preserved for reference: full fp16-delegation path that calls
       // the existing GQA attention via QkvToContext<T, T>. Currently disabled
       // because the unfused path's PrepareQKV requires more buffer setup than
       // we have here and was hitting illegal-memory-access in attention_transpose.
       // The v1 shortcut above is sufficient to validate end-to-end wiring
       // (encode + decode + structural session.run() success).
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream->GetHandle());
  contrib::GroupQueryAttentionParameters fp_params = parameters;
  fp_params.kv_quant_method = KVQuantMethod::NONE;
  fp_params.k_quant_type = KVQuantizationType::NONE;
  fp_params.v_quant_type = KVQuantizationType::NONE;
  fp_params.kv_cache_bit_width = 0;
  fp_params.is_first_prompt = true;   // tells the kernel "no past, just compute attention from K/V"
  fp_params.sequence_length = total_seq;
  fp_params.total_sequence_length = total_seq;
  fp_params.seqlen_past_kv_cache = 0;
  fp_params.seqlen_present_kv_cache = total_seq;
  fp_params.past_present_share_buffer = false;

  GroupQueryAttentionData<T, T> fp_data{};
  fp_data.query = data.query;
  // Use the decoded K/V as the "current" K/V for first-prompt attention.
  // We need just the new_seq tail (so the kernel only computes attention for
  // the new positions), but the keys are over the full total_seq.
  // For minimal v1 we run "attention over full sequence" and only the last
  // new_seq Q rows contribute to the output that wandler reads — same shape.
  fp_data.key = d_K_fp16;
  fp_data.value = d_V_fp16;
  fp_data.cos_cache = data.cos_cache;
  fp_data.sin_cache = data.sin_cache;
  fp_data.head_sink = data.head_sink;
  fp_data.position_ids = data.position_ids;
  fp_data.output = data.output;

  // We need a present_key / present_value that the fp16 path can write to.
  // Allocate fp16 stub buffers (shape: B * H_kv * total_seq * D).
  T* d_present_k_fp16 = nullptr;
  T* d_present_v_fp16 = nullptr;
  CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_present_k_fp16, kv_elements * sizeof(T), cuda_stream));
  CUDA_RETURN_IF_ERROR(cudaMallocAsync(&d_present_v_fp16, kv_elements * sizeof(T), cuda_stream));
  fp_data.present_key = d_present_k_fp16;
  fp_data.present_value = d_present_v_fp16;
  // past_key / past_value left as nullptr for is_first_prompt path.

  // We also need cublas + the rest of the GroupQueryAttention surrounding the
  // call. For v1 we delegate to QkvToContext directly with a fresh cublas
  // handle from the stream's parent provider — but we don't have access here.
  // Simpler v1: use a global cublas handle obtained via the stream.
  cublasHandle_t cublas;
  cublasCreate(&cublas);
  cublasSetStream(cublas, cuda_stream);

  // Recompute past_seq_lens / total_seq_lens for the inner call.
  // We use the existing buffers from data (already populated by the outer caller).
  fp_data.past_seq_lens = data.past_seq_lens;
  fp_data.total_seq_lens = data.total_seq_lens;
  fp_data.padded_seq_lens = data.padded_seq_lens;
  fp_data.softmax_lse = data.softmax_lse;
  fp_data.softmax_lse_accum = data.softmax_lse_accum;
  fp_data.out_accum = data.out_accum;
  fp_data.qkv_buffer = data.qkv_buffer;
  fp_data.fmha_buffer = data.fmha_buffer;
  fp_data.k = data.k;
  fp_data.v = data.v;
  // Use the unfused path for simplicity: it has fewer constraints and works for
  // any head_size. It's the "math" path that does Q@K^T -> softmax -> @V via
  // cublas + element-wise softmax kernel.
  fp_data.use_flash_attention = false;
  fp_data.use_memory_efficient_attention = false;
  fp_data.use_xqa = false;
  fp_data.use_unfused = true;

  // Allocate unfused scratch buffers.
  const size_t Bs = static_cast<size_t>(B);
  const size_t N_q = static_cast<size_t>(parameters.num_heads);
  const size_t S_q = static_cast<size_t>(total_seq);
  const size_t Hs = static_cast<size_t>(D);
  const size_t S_kv = static_cast<size_t>(total_seq);

  auto align256 = [](size_t v) -> size_t { return ((v + 255) / 256) * 256; };
  const size_t q_bnsh_bytes = align256(Bs * N_q * S_q * Hs * sizeof(T));
  const size_t y_bnsh_bytes = align256(Bs * N_q * S_q * Hs * sizeof(T));
  const size_t ws_bytes = onnxruntime::contrib::cuda::GetUnfusedAttentionWorkspaceSize(
      static_cast<int>(Bs), static_cast<int>(N_q), static_cast<int>(S_q), static_cast<int>(S_kv));

  uint8_t* unfused_scratch = nullptr;
  CUDA_RETURN_IF_ERROR(cudaMallocAsync(&unfused_scratch, q_bnsh_bytes + y_bnsh_bytes + ws_bytes, cuda_stream));
  fp_data.unfused_q_bnsh = reinterpret_cast<T*>(unfused_scratch);
  fp_data.unfused_y_bnsh = reinterpret_cast<T*>(unfused_scratch + q_bnsh_bytes);
  fp_data.unfused_workspace = reinterpret_cast<void*>(unfused_scratch + q_bnsh_bytes + y_bnsh_bytes);

  Status final_status = QkvToContext<T, T>(device_prop, cublas, stream, fp_params, fp_data);

  cublasDestroy(cublas);
  cudaFreeAsync(d_codebook, cuda_stream);
  cudaFreeAsync(d_K_fp16, cuda_stream);
  cudaFreeAsync(d_V_fp16, cuda_stream);
  cudaFreeAsync(d_present_k_fp16, cuda_stream);
  cudaFreeAsync(d_present_v_fp16, cuda_stream);
  cudaFreeAsync(unfused_scratch, cuda_stream);
  return final_status;
#endif  // 0 — v2 fp16-delegation code

template Status LaunchTurboQuantAttention<half, uint8_t>(
    const cudaDeviceProp&, Stream*,
    contrib::GroupQueryAttentionParameters&,
    GroupQueryAttentionData<half, uint8_t>&);

template Status LaunchTurboQuantAttention<__nv_bfloat16, uint8_t>(
    const cudaDeviceProp&, Stream*,
    contrib::GroupQueryAttentionParameters&,
    GroupQueryAttentionData<__nv_bfloat16, uint8_t>&);

// =============================================================================
// Kept for backward compat: roundtrip helper used by validation.
// =============================================================================

template <typename T>
Status LaunchTurboQuantEncodeDecodeRoundtrip(
    const cudaDeviceProp& /*device_prop*/,
    Stream* /*stream*/,
    int /*batch_size*/,
    int /*n_kv_heads*/,
    int /*seq_len*/,
    int /*head_size*/,
    int /*key_bits*/,
    int /*value_bits*/,
    bool /*norm_correction*/,
    const T* /*K*/,
    const T* /*V*/,
    const T* /*k_codebook*/,
    const T* /*hadamard*/,
    T* /*K_recon*/,
    T* /*V_recon*/) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Roundtrip helper deprecated; use LaunchTurboQuantAttention or test via gtest harness.");
}

template Status LaunchTurboQuantEncodeDecodeRoundtrip<half>(
    const cudaDeviceProp&, Stream*, int, int, int, int, int, int, bool,
    const half*, const half*, const half*, const half*, half*, half*);

template Status LaunchTurboQuantEncodeDecodeRoundtrip<__nv_bfloat16>(
    const cudaDeviceProp&, Stream*, int, int, int, int, int, int, bool,
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

// =============================================================================
// C-style entry point retained for tests that may dlopen this lib.
// =============================================================================
extern "C" __attribute__((visibility("default"))) int RunTurboQuantRoundtrip_fp16(
    int /*batch_size*/, int /*n_kv_heads*/, int /*seq_len*/, int /*head_size*/,
    int /*key_bits*/, int /*value_bits*/, int /*norm_correction*/,
    const void* /*d_K*/, const void* /*d_V*/, const void* /*d_codebook*/,
    void* /*d_K_recon*/, void* /*d_V_recon*/,
    void* /*cuda_stream*/) {
  return -1;  // deprecated
}
