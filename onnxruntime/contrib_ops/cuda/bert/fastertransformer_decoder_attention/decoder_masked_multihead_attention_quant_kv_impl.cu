/*
 * The implementation of this file is based on code provided by https://github.com/NVIDIA/FasterTransformer
 *
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modifications Copyright (c) Microsoft.
// Licensed under the MIT License.

// Modifications:
// (1) Removed some code paths from the original implementation that had features which is not supported by
//  corresponding ORT kernel - for example- CrossAttention support, FP8, INT8, supports, etc.
// (2) When dealing with masked tokens, this kernel implementation deviates from FasterTransformer by applying
// mask filter values. Appropriate commentary exists in the code below.

#include "contrib_ops/cuda/bert/rotary_embedding_util.h"
#include "decoder_masked_multihead_attention_impl.h"
#include "decoder_masked_multihead_attention_impl_utils.h"
#include <cfloat>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace decoder_masked_self_attention_details;

template <typename T>
struct TFloatTypeFrom {
};

template <>
struct TFloatTypeFrom<float> {
  using Type = float;
};

template <>
struct TFloatTypeFrom<uint16_t> {
  using Type = half;
};

inline __device__ __half2 DequantizeChar2(const char2 ch2, const float unit_scale) {
  return __float22half2_rn(float2{unit_scale * ch2.x, unit_scale * ch2.y});
}

template <typename TVec>
class QuantVec {};

struct __align__(4) Char2x2 {
  char2 x;
  char2 y;
};

struct __align__(8) Char2x4 {
  char2 x;
  char2 y;
  char2 z;
  char2 w;
};

template <>
class QuantVec<uint32_t> {
 public:
  using Type = char2;
};

template <>
class QuantVec<uint2> {
 public:
  using Type = Char2x2;
};

template <>
class QuantVec<uint4> {
 public:
  using Type = Char2x4;
};

template <typename TVec>
inline __device__ TVec DequantizeVec(const typename QuantVec<TVec>::Type quant_vec_m, const float unit_scale);

template <>
inline __device__ uint32_t DequantizeVec<uint32_t>(const char2 ch2, const float unit_scale) {
  union __align__(4) {
    __half2 h2;
    uint32_t whole;
  }
  uh;
  uh.h2 = DequantizeChar2(ch2, unit_scale);
  return uh.whole;
}

template <>
inline __device__ uint2 DequantizeVec<uint2>(const Char2x2 ch2x2, const float unit_scale) {
  union __align__(8) {
    struct __align__(8) {
      __half2 h2x;
      __half2 h2y;
    };
    uint2 whole;
  }
  vec;
  vec.h2x = DequantizeChar2(ch2x2.x, unit_scale);
  vec.h2y = DequantizeChar2(ch2x2.y, unit_scale);
  return vec.whole;
}

template <>
inline __device__ uint4 DequantizeVec<uint4>(const Char2x4 ch2x4, const float unit_scale) {
  union __align__(16) {
    struct __align__(16) {
      __half2 h2x;
      __half2 h2y;
      __half2 h2z;
      __half2 h2w;
    };
    uint4 whole;
  }
  vec;
  vec.h2x = DequantizeChar2(ch2x4.x, unit_scale);
  vec.h2y = DequantizeChar2(ch2x4.y, unit_scale);
  vec.h2z = DequantizeChar2(ch2x4.z, unit_scale);
  vec.h2w = DequantizeChar2(ch2x4.w, unit_scale);
  return vec.whole;
}

template <typename TVec>
inline __device__ TVec LoadQuantVec(const TVec* q8, const float unit_scale) {
  using TQuantVec = typename QuantVec<TVec>::Type;
  TQuantVec quant_vec = *(const TQuantVec*)q8;
  return DequantizeVec<TVec>(quant_vec, unit_scale);
}

template <typename TFp, typename TVec>
inline __device__ TFp MaxAbsFloat(const TVec v);

template <>
inline __device__ __half MaxAbsFloat(const uint32_t v) {
  union __align__(4) {
    __half2 h2;
    uint32_t whole;
  }
  uvec = {.whole = v};
  const __half2 h2 = __habs2(uvec.h2);
  return __hmax(h2.x, h2.y);
}

template <>
inline __device__ __half MaxAbsFloat(const uint2 v) {
  // make it simple rather than save one op
  return __hmax(MaxAbsFloat<__half, uint32_t>(v.x), MaxAbsFloat<__half, uint32_t>(v.y));
}

template <>
inline __device__ __half MaxAbsFloat(const uint4 v) {
  return __hmax(__hmax(MaxAbsFloat<__half, uint32_t>(v.x), MaxAbsFloat<__half, uint32_t>(v.y)),
                __hmax(MaxAbsFloat<__half, uint32_t>(v.z), MaxAbsFloat<__half, uint32_t>(v.w)));
}

template <typename TVec>
inline __device__ typename QuantVec<TVec>::Type Quantize(const TVec v, const float scale);

template <>
inline __device__ char2 Quantize(const uint32_t v, const float inv_unit_scale) {
  union __align__(4) {
    uint32_t u;
    __half2 h2;
  }
  uh2 = {v};
  float2 f2 = __half22float2(uh2.h2);
  return char2{(char)min(max(-127, __float2int_rn(inv_unit_scale * f2.x)), 127),
               (char)min(max(-127, __float2int_rn(inv_unit_scale * f2.y)), 127)};
}

template <>
inline __device__ Char2x2 Quantize(const uint2 v, const float inv_unit_scale) {
  Char2x2 ch2x2;
  ch2x2.x = Quantize<uint32_t>(v.x, inv_unit_scale);
  ch2x2.y = Quantize<uint32_t>(v.y, inv_unit_scale);
  return ch2x2;
}

template <>
inline __device__ Char2x4 Quantize(const uint4 v, const float inv_unit_scale) {
  Char2x4 ch2x4;
  ch2x4.x = Quantize<uint32_t>(v.x, inv_unit_scale);
  ch2x4.y = Quantize<uint32_t>(v.y, inv_unit_scale);
  ch2x4.z = Quantize<uint32_t>(v.z, inv_unit_scale);
  ch2x4.w = Quantize<uint32_t>(v.w, inv_unit_scale);
  return ch2x4;
}

template <typename TVec>
inline __device__ void QuantizeTo(int8_t* dst, const TVec v, const float inv_unit_scale) {
  using TQuantVec = typename QuantVec<TVec>::Type;
  TQuantVec quant_vec = Quantize(v, inv_unit_scale);
  *(TQuantVec*)dst = quant_vec;
}

template <
    // The type of the inputs. Supported types: half(uint16_t).
    typename T,
    // The hidden dimension per head.
    int head_size,
    // The number of threads per key.
    int THREADS_PER_KEY,
    // The number of threads per value.
    int THREADS_PER_VALUE,
    // The number of threads in a threadblock.
    int THREADS_PER_BLOCK,
    // The type of the scale in memory
    typename TScale>
__global__ void masked_multihead_attention_quant_kv_kernel(DecoderMaskedMultiHeadAttentionQuantKVParams params) {
  // This kernel contains some code that cannot be compiled on CUDA ARCH 5.3 or lower
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
  (void)(params);
#else
  using TQ8 = int8_t;  // quantized value type for K V cache
  using TFp = typename TFloatTypeFrom<T>::Type;

  // Make sure the hidden dimension per head is a multiple of the number of threads per key.
  static_assert(head_size % THREADS_PER_KEY == 0, "");

  // Make sure the hidden dimension per head is a multiple of the number of threads per value.
  static_assert(head_size % THREADS_PER_VALUE == 0, "");

  // The size of a warp.
  constexpr int WARP_SIZE = 32;

  // The number of warps in a threadblock.
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  extern __shared__ char smem_[];

  // The shared memory for the Q*K^T values and partial logits in softmax.
  float* qk_smem = reinterpret_cast<float*>(smem_);

  // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
  char* logits_smem_ = smem_;

  if (sizeof(T) != 4) {
    // For fp16, we have allocated separate memory for logits - use it
    logits_smem_ += (((params.total_sequence_length + 3) / 4) * 16);
  }

  T* logits_smem = reinterpret_cast<T*>(logits_smem_);

  // The shared memory to do the final reduction for the output values. Reuse qk_smem.
  T* out_smem = reinterpret_cast<T*>(smem_);

  // The shared memory buffers for the block-wide reductions. One for max, one for sum.
  __shared__ float red_smem[WARPS_PER_BLOCK * 2];

  // A vector of Q or K elements for the current timestep.
  using Qk_vec_k = typename Qk_vec_k_<T, head_size>::Type;  // with kernel-used precision
  using Qk_vec_m = typename Qk_vec_m_<T, head_size>::Type;  // with memory-used precision

  // Use alignment for safely casting the shared buffers as Qk_vec_k.
  // Shared memory to store Q inputs.
  __shared__ __align__(sizeof(Qk_vec_k)) T q_smem[head_size];

  // The number of elements per vector.
  constexpr int QK_VEC_SIZE = sizeof(Qk_vec_m) / sizeof(T);
  // caller need to check that
  //    * params.quant_kv_block_size is power of 2 and > 0
  //    * params.quant_kv_block_size % QK_VEC_SIZE == 0
  //    * params.quant_kv_block_size % K_VEC_SIZE == 0
  //    * params.quant_kv_block_size % V_VEC_SIZE == 0
  //    * head_size % params.quant_kv_block_size == 0
  const int scales_per_head = head_size / params.quant_kv_block_size;

  // Make sure the hidden size per head is a multiple of the vector size.
  static_assert(head_size % QK_VEC_SIZE == 0, "");

  constexpr int QK_THREAD_COUNT = head_size / QK_VEC_SIZE;
  static_assert(QK_THREAD_COUNT <= THREADS_PER_BLOCK);

  // The layout of the cache is [B, H, head_size/x, L, x] with x == 4/8/16 for FP32/FP16/FP8. Since each thread
  // owns x elements, we have to decompose the linear index into chunks of x values and the posi-
  // tion of the thread in that chunk.

  static_assert(sizeof(T) <= 16);
  static_assert(sizeof(Qk_vec_m) <= 16);

  // The number of elements in a chunk of 16B (that's the x in the above formula).
  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);

  // The number of K vectors in 16B.
  constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec_m);

  // The batch/beam idx
  const int bi = blockIdx.y;

  // The beam idx
  // const int beami = bi % params.beam_width;

  // The "beam-aware" batch idx
  const int bbi = bi / params.beam_width;

  // The head.
  const int hi = blockIdx.x;

  // Combine the batch and the head indices.
  const int bhi = bi * params.num_heads + hi;

  // Combine the "beam-aware" batch idx and the head indices.
  const int bbhi = bbi * params.beam_width * params.num_heads + hi;

  const int input_beam_index = bi % params.beam_width;

  // The thread in the block.
  const int tidx = threadIdx.x;

  // While doing the product Q*K^T for the different keys we track the max.
  float qk_max = -FLT_MAX;

  float qk = 0.0F;

  int qkv_base_offset = params.is_mha && !params.is_packed_qkv
                            ? bi * params.hidden_size + hi * head_size
                            : bi * (3 * params.hidden_size) + hi * head_size;

  const size_t bi_total_seq_length = bi * params.total_sequence_length;

  const size_t bi_max_seq_length = bi * params.max_sequence_length;

  int tlength = params.is_cross_attention ? params.kv_sequence_length : params.past_sequence_length;

  // First QK_THREAD_COUNT load Q and K + the bias values for the current timestep.
  const bool is_active_qk_thread = tidx < QK_THREAD_COUNT;

  // The offset in the Q and K buffer also accounts for the batch.
  int qk_offset = qkv_base_offset + tidx * QK_VEC_SIZE;

  // Trigger the loads from the Q and K buffers.
  Qk_vec_k q;
  zero(q);

  if (is_active_qk_thread) {
    q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&reinterpret_cast<T*>(params.q)[qk_offset]));
  }

  // The offset in the bias buffer.
  int qk_bias_offset = hi * head_size + tidx * QK_VEC_SIZE;

  // Trigger the loads from the Q and K bias buffers.
  if (params.q_bias && is_active_qk_thread) {
    Qk_vec_k q_bias;

    q_bias = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&reinterpret_cast<T*>(params.q_bias)[qk_bias_offset]));

    q = add_vec(q, q_bias);
  }

  TQ8* params_k_cache = reinterpret_cast<TQ8*>(params.k_cache);

  const float inv_sqrt_dh = params.scale;

  if (is_active_qk_thread) {
    // Store the Q values to shared memory.
    *reinterpret_cast<Qk_vec_k*>(&q_smem[tidx * QK_VEC_SIZE]) = q;
  }

  if (!params.is_cross_attention) {
    Qk_vec_k k;

    if (is_active_qk_thread) {
      k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&reinterpret_cast<T*>(params.k)[qk_offset]));

      if (params.k_bias) {
        Qk_vec_k k_bias = vec_conversion<Qk_vec_k, Qk_vec_m>(
            *reinterpret_cast<const Qk_vec_m*>(&reinterpret_cast<T*>(params.k_bias)[qk_bias_offset]));
        k = add_vec(k, k_bias);
      }
    } else {
      zero(k);
    }

    if (params.rotary_embedding_dim > 0) {
      const bool do_rotary = is_active_qk_thread && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

      T* q_smem = reinterpret_cast<T*>(smem_);
      T* k_smem = q_smem + params.rotary_embedding_dim;

      const int half_rotary_dim = params.rotary_embedding_dim / 2;
      const int half_idx = (tidx * QK_VEC_SIZE) / half_rotary_dim;
      const int intra_half_idx = (tidx * QK_VEC_SIZE) % half_rotary_dim;
      const int smem_pitch = half_rotary_dim;

      assert(half_rotary_dim % QK_VEC_SIZE == 0);

      if (do_rotary) {
        *reinterpret_cast<Qk_vec_k*>(q_smem + half_idx * smem_pitch + intra_half_idx) = q;
        *reinterpret_cast<Qk_vec_k*>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
      }

      __syncthreads();

      const int transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
      constexpr int tidx_factor = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;

      if (do_rotary) {
        vec_from_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
        vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

        apply_rotary_embedding(
            q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim, params.t_step);

        write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
        write_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
      }

      __syncthreads();

      if (do_rotary) {
        q = *reinterpret_cast<Qk_vec_k*>(q_smem + half_idx * smem_pitch + intra_half_idx);
        k = *reinterpret_cast<Qk_vec_k*>(k_smem + half_idx * smem_pitch + intra_half_idx);
      }

      __syncthreads();
    }

    float max_abs_k = (float)MaxAbsFloat<TFp, Qk_vec_k>(k);
    // Perform the final reduction to compute the max inside each warp.
    const int qk_threads_per_scale = params.quant_kv_block_size / QK_VEC_SIZE;
    const int qk_threads_per_scale_in_warp = min(qk_threads_per_scale, WARP_SIZE);
    for (int mask = qk_threads_per_scale_in_warp / 2; mask >= 1; mask /= 2) {
      max_abs_k = fmaxf(max_abs_k, __shfl_xor_sync(uint32_t(-1), max_abs_k, mask));
    }

    if (qk_threads_per_scale > WARP_SIZE) {
      const int warp = tidx / WARP_SIZE;
      const int lane = tidx % WARP_SIZE;
      if (lane == 0) {
        red_smem[warp] = max_abs_k;
      }
      __syncthreads();

      // The warps finalize the reduction.
      max_abs_k = ((lane < WARPS_PER_BLOCK) ? red_smem[lane] : -FLT_MAX);

#pragma unroll
      for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
        max_abs_k = fmaxf(max_abs_k, __shfl_xor_sync(uint32_t(-1), max_abs_k, mask));
      }

      // Broadcast to all the threads in the warp.
      max_abs_k = __shfl_sync(uint32_t(-1), max_abs_k, 0);
    }

    if (is_active_qk_thread) {
      // Write the K values to the global memory cache.
      // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
      // system. We designed it this way as it allows much better memory loads (and there are many
      // more loads) + the stores are really "write and forget" since we won't need the ack before
      // the end of the kernel. There's plenty of time for the transactions to complete.

      // The 16B chunk written by the thread.
      int co = tidx / QK_VECS_IN_16B;

      // The position of the thread in that 16B chunk.
      int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

      // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
      int offset = bhi * params.max_sequence_length * head_size + co * params.max_sequence_length * QK_ELTS_IN_16B +
                   tlength * QK_ELTS_IN_16B + ci;
      // Trigger the stores to global memory.
      const float inv_unit_scale_k = (max_abs_k ? (127.0f / max_abs_k) : max_abs_k);
      QuantizeTo(&params_k_cache[offset], k, inv_unit_scale_k);
      if (tidx % qk_threads_per_scale == 0) {
        const int scale_offset = (bhi * params.max_sequence_length + tlength) * scales_per_head + tidx / qk_threads_per_scale;
        *(((TScale*)params.k_scale) + scale_offset) = (TScale)(max_abs_k / 127.0f);
      }

      // Compute \sum_i Q[i] * K^T[i] for the current timestep.
      using Qk_vec_acum = Qk_vec_k;
      qk = dot<Qk_vec_acum, Qk_vec_k>(q, k);

      if (QK_THREAD_COUNT <= WARP_SIZE) {
#pragma unroll
        for (int mask = QK_THREAD_COUNT / 2; mask >= 1; mask /= 2) {
          qk += __shfl_xor_sync(shfl_mask(QK_THREAD_COUNT), qk, mask);
        }
      }
    }

    if (QK_THREAD_COUNT > WARP_SIZE) {
      constexpr int WARPS_PER_RED = (QK_THREAD_COUNT + WARP_SIZE - 1) / WARP_SIZE;
      qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0) {
      // Normalize qk.
      qk *= inv_sqrt_dh;
      if (params.relative_attention_bias != nullptr) {
        qk = add_vec(qk,
                     reinterpret_cast<T*>(params.relative_attention_bias)[hi * params.sequence_length * params.total_sequence_length + tlength]);
      }
      qk_max = qk;
      qk_smem[tlength] = qk;
    }
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The type of queries and keys for the math in the Q*K^T product.
  using K_vec_k = typename K_vec_k_<T, THREADS_PER_KEY>::Type;
  using K_vec_m = typename K_vec_m_<T, THREADS_PER_KEY>::Type;

  // The number of elements per vector.
  constexpr int K_VEC_SIZE = sizeof(K_vec_m) / sizeof(T);

  // Make sure the hidden size per head is a multiple of the vector size.
  static_assert(head_size % K_VEC_SIZE == 0, "");

  // The number of elements per thread.
  constexpr int K_ELTS_PER_THREAD = head_size / THREADS_PER_KEY;

  // The number of vectors per thread.
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

  // The position the first key loaded by each thread from the cache buffer (for this B * H).
  int ko = tidx / THREADS_PER_KEY;

  // The position of the thread in the chunk of keys.
  int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;

  static_assert(head_size == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD);

  // Load the Q values from shared memory. The values are reused during the loop on K.
  K_vec_k q_vec[K_VECS_PER_THREAD];
#pragma unroll
  for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
    q_vec[ii] = *reinterpret_cast<const K_vec_k*>(&q_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
  }

  // The number of timesteps loaded per iteration.
  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;

  // The number of keys per warp.
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  // Base pointer for the beam's batch, before offsetting with indirection buffer
  TQ8* k_cache_batch = &params_k_cache[bbhi * params.max_sequence_length * head_size + ki];

  // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
  int ti_end = ((tlength + K_PER_WARP - 1) / K_PER_WARP) * K_PER_WARP;

  // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
  bool has_beams = params.cache_indir != nullptr && !params.is_cross_attention;
  const int* beam_indices = has_beams ? &params.cache_indir[bi_max_seq_length] : nullptr;

  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    bool is_masked = (params.mask != nullptr) && (params.mask[bi_total_seq_length + ti] == 0) && (ti < tlength);
    const int mapped_beam_index = (has_beams && ti < tlength) ? beam_indices[ti] : input_beam_index;
    const int beam_offset = mapped_beam_index * params.num_heads * params.max_sequence_length * head_size;

    // The keys loaded from the key cache.
    K_vec_k k_vec[K_VECS_PER_THREAD];

    if (ti < tlength) {
      const int mapped_bhi = bbhi + mapped_beam_index * params.num_heads;
      const TScale* scales_in_head = ((const TScale*)params.k_scale) + ((mapped_bhi * params.max_sequence_length + ti) * scales_per_head);
      float unit_scale_k = 0.0f;
      int in_head_elem_idx = ki;
      int renew_scale_elem_idx = 0; // reload scale when in_head_elem_idx >= renew_scale_elem_idx
#pragma unroll
      for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        if (in_head_elem_idx >= renew_scale_elem_idx) {
          int in_head_scale_idx = in_head_elem_idx / params.quant_kv_block_size;
          renew_scale_elem_idx = (in_head_scale_idx + 1) * params.quant_kv_block_size;
          unit_scale_k = (float)scales_in_head[in_head_scale_idx];
        }
        in_head_elem_idx += QK_ELTS_IN_16B;
        int jj = ii * params.max_sequence_length + ti;
        k_vec[ii] = LoadQuantVec((const K_vec_k*)(&k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B]), unit_scale_k);
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        zero(k_vec[ii]);
      }
    }

    // Perform the dot product and normalize qk.
    // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * inv_sqrt_dh;

    // This is a deviation from FasterTransformer kernel implementation
    // but this aligns with ORT's other Attention kernels which strives to
    // mimic PyTorch when dealing with mask filter values
    if (is_masked) {
      qk += params.mask_filter_value;
    }

    // Store the product to shared memory. There's one qk value per timestep. Update the max.
    if (ti < tlength && tidx % THREADS_PER_KEY == 0) {
      if (params.relative_attention_bias != nullptr) {
        qk = add_vec(qk,
                     reinterpret_cast<T*>(params.relative_attention_bias)[hi * params.sequence_length * params.total_sequence_length + ti]);
      }
      qk_max = fmaxf(qk_max, qk);
      qk_smem[ti] = qk;
    }
  }

  // Perform the final reduction to compute the max inside each warp.
  //
  // NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
  // group so it's not needed to run the reduction inside the group (again).
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  // Decompose the thread index into warp and lane.
  const int warp = tidx / WARP_SIZE;
  const int lane = tidx % WARP_SIZE;

  // The warp leader writes the max to shared memory.
  if (lane == 0) {
    red_smem[warp] = qk_max;
  }

  // Make sure the products are in shared memory.
  __syncthreads();

  // The warps finalize the reduction.
  qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  // Broadcast to all the threads in the warp.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Compute the logits and start the sum.
  float sum = 0.f;
  int sum_tlength = params.is_cross_attention ? tlength - 1 : tlength;
  for (int ti = tidx; ti <= sum_tlength; ti += THREADS_PER_BLOCK) {
    // This is a deviation from FasterTransformer kernel implementation
    // but this aligns with ORT's other Attention kernels which strives to
    // mimic PyTorch when dealing with mask filter values
    float logit = __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }

  // Compute the sum.
  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  // Normalize the logits.
  float inv_sum = __fdividef(1.f, sum + 1.e-6f);
  for (int ti = tidx; ti <= sum_tlength; ti += THREADS_PER_BLOCK) {
    float logit = qk_smem[ti] * inv_sum;
    ConvertFromFloat(logits_smem[ti], logit);
  }

  // Put Values part below so we leverage __syncthreads
  // from the previous step

  // The number of elements per vector.
  constexpr int V_VEC_SIZE = head_size / THREADS_PER_VALUE;

  // A vector of V elements for the current timestep.
  using V_vec_k = typename V_vec_k_<T, V_VEC_SIZE>::Type;
  using V_vec_m = typename V_vec_m_<T, V_VEC_SIZE>::Type;

  // The value computed by this thread.
  int vo = tidx / THREADS_PER_VALUE;

  // The hidden dimensions computed by this particular thread.
  int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE;

  // The base pointer for the value in the cache buffer.
  TQ8* params_v_cache = reinterpret_cast<TQ8*>(params.v_cache);

  TQ8* v_cache = &params_v_cache[bhi * params.max_sequence_length * head_size + vi];

  // Base pointer for the beam's batch, before offsetting with indirection buffer
  TQ8* v_cache_batch = &params_v_cache[bbhi * params.max_sequence_length * head_size + vi];

  // The number of values processed per iteration of the loop.
  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

  // One group of threads computes the product(s) for the current timestep.
  V_vec_k v_bias;
  if (params.v_bias && !params.is_cross_attention) {
    zero(v_bias);

    T* params_v_bias = reinterpret_cast<T*>(params.v_bias);

    if (vo == tlength % V_PER_ITER) {
      v_bias = vec_conversion<V_vec_k, V_vec_m>(*reinterpret_cast<const V_vec_m*>(&params_v_bias[hi * head_size + vi]));
    }
  }

  // From previous, before values, step
  // Also make sure the logits are in shared memory.
  __syncthreads();

  using V_vec_acum = typename V_vec_acum_fp32_<V_vec_k>::Type;

  // The partial outputs computed by each thread.
  V_vec_acum out;
  zero(out);

  // Loop over the timesteps to compute the partial outputs.
  for (int ti = vo; ti < tlength; ti += V_PER_ITER) {
    // Fetch offset based on cache_indir when beam sampling
    const int mapped_beam_index = has_beams ? params.cache_indir[bi_max_seq_length + ti] : input_beam_index;
    const int beam_offset = mapped_beam_index * params.num_heads * params.max_sequence_length * head_size;

    const int mapped_bhi = bbhi + mapped_beam_index * params.num_heads;
    const int scale_offset = (mapped_bhi * params.max_sequence_length + ti) * scales_per_head + vi / params.quant_kv_block_size;
    const float unit_scale_v = (float)*(((TScale*)params.v_scale) + scale_offset);

    // Load the values from the cache.
    // V_vec_k v = vec_conversion<V_vec_k, V_vec_m>(*reinterpret_cast<const V_vec_m*>(&v_cache_batch[beam_offset + ti * head_size]));
    V_vec_k v = LoadQuantVec((const V_vec_k*)(&v_cache_batch[beam_offset + ti * head_size]), unit_scale_v);

    // Load the logits from shared memory.
    T logit = logits_smem[ti];
    out = fma(logit, v, out);
  }

  // One group of threads computes the product(s) for the current timestep.
  if ((vo == tlength % V_PER_ITER) && !params.is_cross_attention) {
    const auto v_offset = qkv_base_offset + vi;

    V_vec_k v;
    v = vec_conversion<V_vec_k, V_vec_m>(*reinterpret_cast<const V_vec_m*>(&reinterpret_cast<T*>(params.v)[v_offset]));
    if (params.v_bias) {
      v = add_vec(v, v_bias);
    }

    static_assert(THREADS_PER_VALUE <= WARP_SIZE);
    float max_abs_v = (float)MaxAbsFloat<TFp, V_vec_k>(v);
    const uint32_t group_id = (tidx % WARP_SIZE) / THREADS_PER_VALUE;
    const uint32_t group_masks = ((1u << THREADS_PER_VALUE) - 1) << (group_id * THREADS_PER_VALUE);
#pragma unroll
    for (int mask = THREADS_PER_VALUE / 2; mask >= 1; mask /= 2) {
      max_abs_v = fmaxf(max_abs_v, __shfl_xor_sync(group_masks, max_abs_v, mask, THREADS_PER_VALUE));
    }

    // Store the values with bias back to global memory in the cache for V.
    //*reinterpret_cast<V_vec_m*>(&v_cache[tlength * head_size]) = vec_conversion<V_vec_m, V_vec_k>(v);
    const float inv_unit_scale_v = (max_abs_v ? (127.0f / max_abs_v) : max_abs_v);
    QuantizeTo(&v_cache[tlength * head_size], v, inv_unit_scale_v);
    if (vi % params.quant_kv_block_size == 0) {
      const int scales_per_head = head_size / params.quant_kv_block_size;
      const int scale_offset = (bhi * params.max_sequence_length + tlength) * scales_per_head + vi / params.quant_kv_block_size;
      *(((TScale*)params.v_scale) + scale_offset) = (TScale)(max_abs_v / 127.0f);
    }

    // Initialize the output value with the current timestep.
    out = fma(logits_smem[tlength], v, out);
  }

  // Make sure we can start writing to shared memory.
  __syncthreads();

  // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
  for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2) {
    // The midpoint in the number of active groups.
    int midpoint = active_groups / 2;

    // The upper part of active threads store to shared memory.
    if (vo >= midpoint && vo < active_groups) {
      ConvertFromFloat(*reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * head_size + vi]), out);
    }
    __syncthreads();

    // The bottom warps update their values.
    if (vo < midpoint) {
      out = add_vec(*reinterpret_cast<const V_vec_k*>(&out_smem[vo * head_size + vi]), out);
    }
    __syncthreads();
  }

  // Output the final values.
  T* params_out = reinterpret_cast<T*>(params.out);
  if (vo == 0) {
    ConvertFromFloat(*reinterpret_cast<V_vec_m*>(&params_out[bhi * head_size + vi]), out);
  }
#endif
}

typedef __half TQuantKVScale;

// Template instantiation(s)

#define Instantiate(TQuantKVScale)                                                                                                                                    \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 32, 4, 4, 64, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params);    \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 32, 2, 4, 128, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params);   \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 32, 1, 4, 256, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params);   \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 64, 4, 8, 64, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params);    \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 64, 2, 8, 128, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params);   \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 64, 1, 8, 256, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params);   \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 128, 4, 16, 64, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params);  \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 128, 2, 16, 128, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params); \
  template void __global__ masked_multihead_attention_quant_kv_kernel<uint16_t, 128, 1, 16, 256, TQuantKVScale>(DecoderMaskedMultiHeadAttentionQuantKVParams params);

Instantiate(__half);
Instantiate(float);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
