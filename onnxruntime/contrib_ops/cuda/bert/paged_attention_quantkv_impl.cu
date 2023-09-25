// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "paged_attention_quantkv_impl.cuh"

#include <cassert>
#include <cuda_fp16.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <type_traits>
#include "core/framework/float16.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "contrib_ops/cuda/bert/transformer_common.h"

#include "paged_generic.cuh"
#include "paged_dtype_float16.cuh"
#include "paged_dtype_float32.cuh"
#include "paged_utils.cuh"

using namespace onnxruntime::cuda;
#define CHECK_CUDA(expr) CUDA_RETURN_IF_ERROR(expr)

namespace onnxruntime {
namespace contrib {
namespace cuda {

#include <algorithm>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {

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

inline __device__ __half2 DequantizeChar2(const char2 ch2, const float2 unit_scales) {
  return __float22half2_rn(float2{unit_scales.x * ch2.x, unit_scales.y * ch2.y});
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


template <typename TVec>
inline __device__ TVec DequantizeByScaleVec(const typename QuantVec<TVec>::Type quant_vec_m, const typename FloatVec<TVec> unit_scales);

template <>
inline __device__ uint32_t DequantizeByScaleVec<uint32_t>(const char2 ch2, const float2 unit_scales) {
  union __align__(4) {
    __half2 h2;
    uint32_t whole;
  }
  uh;
  uh.h2 = DequantizeChar2(ch2, unit_scales);
  return uh.whole;
}

template <>
inline __device__ uint2 DequantizeByScaleVec<uint2>(const Char2x2 ch2x2, const Float4_ unit_scales) {
  union __align__(8) {
    struct __align__(8) {
      __half2 h2x;
      __half2 h2y;
    };
    uint2 whole;
  }
  vec;
  vec.h2x = DequantizeChar2(ch2x2.x, unit_scales.x);
  vec.h2y = DequantizeChar2(ch2x2.y, unit_scales.y);
  return vec.whole;
}

template <>
inline __device__ uint4 DequantizeByScaleVec<uint4>(const Char2x4 ch2x4, const Float8_ unit_scales) {
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
  vec.h2x = DequantizeChar2(ch2x4.x, unit_scales.x);
  vec.h2y = DequantizeChar2(ch2x4.y, unit_scales.y);
  vec.h2z = DequantizeChar2(ch2x4.z, unit_scales.z);
  vec.h2w = DequantizeChar2(ch2x4.w, unit_scales.w);
  return vec.whole;
}


template <typename TVec>
inline __device__ TVec LoadQuantVecByScales(const TVec* q8, const typename FloatVec<TVec> unit_scales) {
  using TQuantVec = typename QuantVec<TVec>::Type;
  TQuantVec quant_vec = *(const TQuantVec*)q8;
  return DequantizeByScaleVec<TVec>(quant_vec, unit_scale);
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


template <typename quant_params_elem_t> // float or float16
class KVQuantProcessor {
 public:
  static constexpr int32_t K = 0;
  static constexpr int32_t V = 1;

  using kv_mem_scalar_t = int8_t;
  using kv_quant_param_t = quant_params_elem_t;
  KVQuantProcessor(const kv_quant_param_t* params_cache, int kv_quant_chunk_size, int num_kv_heads, int head_size, int block_size)
  : kv_quant_params_cache_(params_cache), kv_quant_chunk_size_(kv_quant_chunk_size), block_size_(block_size) {
    assert(head_size % kv_quant_chunk_size == 0);
    head_stride_ = block_size * (head_size / kv_quant_chunk_size);
    k_or_v_stride_ = num_kv_heads * head_stride_;
  }

  template <typename VecT>
  inline __device__ VecT LoadAndDequantizeK(
      const VecT* ptr,
      const int k_or_v,
      const int block_id,
      const int in_block_token_id,
      const int head_id,
      const int in_head_idx) {
    const kv_quant_param_t* kv_block = kv_quant_params_cache_ + (int64_t)(block_id * 2 + k_or_v) * k_or_v_stride_;
    float* unit_scale = (float)*(kv_block + (head_id * head_stride_ + (in_head_idx / kv_quant_chunk_size_) * block_size_ + in_block_token_id));
    return LoadQuantVec(ptr, unit_scale);
  }

  const kv_quant_param_t* kv_quant_params_cache_;  // [num_blocks, 2, num_kv_heads, head_size / kv_quant_chunk_size, block_size]
  const int kv_quant_chunk_size_; // for how many consecutive values, calc one scale for them to quant
  const int block_size_;
  const int head_stride_;
  const int k_or_v_stride_;
};

template <
    typename kv_quant_param_t,
    int HEAD_SIZE,
    int BLOCK_SIZE>
inline __device__ void TransposeAndDequantizeV(
    const int8_t* v_blk,                // [HEAD_SIZE / x, BLOCK_SIZE, x = 2 * (WARP_SIZE / BLOCK_SIZE)]
    const quant_scale_t* quant_param_block,  // [HEAD_SIZE / quant_chunk_size, BLOCK_SIZE]
    half v_shm[][HEAD_SIZE][BLOCK_SIZE],// [#warp][HEAD_SIZE][BLOCK_SIZE]
    const int quant_chunk_size) {

  constexpr int VEC_SIZE = 2; // to save half2 in shared memory for alignment
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_VECS_PER_THREAD = HEAD_SIZE / (THREAD_GROUP_SIZE * VEC_SIZE);
  assert(HEAD_SIZE % (THREAD_GROUP_SIZE * VEC_SIZE) == 0);
  constexpr int x = (THREAD_GROUP_SIZE * VEC_SIZE);
  constexpr int head_stride = BLOCK_SIZE * x;

  // Optimize it more later
  const int thread_idx = threadIdx.x & (WARP_SIZE - 1);
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;
  const int in_block_token_idx = (thread_idx & (WARP_SIZE - 1)) / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type; // uint32_t (2 half)

  K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
  for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
    const auto* v_ptr = v_blk  + in_block_token_idx * x;
    const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
    const int offset1 = (vec_idx * VEC_SIZE) / x;
    const int offset2 = (vec_idx * VEC_SIZE) % x;

    const int8_t* vec_ptr = reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
    float* unit_scale = (float)*(kv_block + (head_id * head_stride_ + (in_head_idx / kv_quant_chunk_size_) * block_size_ + in_block_token_id));
    return LoadQuantVec(ptr, unit_scale);

    K_vec[j] =
    if constexpr (!kv_quant_handler_t::IS_QUANTIZED) {
      k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
    } else {
      k_vecs[j] = quant_handler.LoadAndDequantize(
          reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2),
          physical_block_number,
          physical_block_offset,
          kv_head_idx,
          vec_idx * VEC_SIZE);
    }
  }



}


// Grid: (num_heads, num_seqs).
template <
    typename scalar_t,
    typename kv_quant_handler_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void single_query_cached_quant_kv_attention_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
    const int8_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/16, block_size, 16]
    const int8_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size/16, block_size, 16]
    const int* __restrict__ head_mapping,  // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const kv_quant_handler_t kv_quant_handler) {
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_mapping ? head_mapping[head_idx] : head_idx;
  const int kv_block_stride_m = head_mapping ? kv_block_stride : num_heads * HEAD_SIZE * BLOCK_SIZE;
  const int kv_head_stride_m = head_mapping ? kv_head_stride : HEAD_SIZE * BLOCK_SIZE;
  const int seq_idx = blockIdx.y;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  const int num_kv_heads = kv_block_stride_m / kv_head_stride_m;

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4, and the data type is half,
  // then the vector size is 16 / (4 * sizeof(int8_t)) == 4.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(int8_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const int quant_param_head_size = HEAD_SIZE / kv_quant_chunk_size;
  const int kv_quant_param_stride = num_kv_heads * quant_param_head_size * BLOCK_SIZE;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = block_table[block_idx];

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const auto* k_ptr = k_cache + physical_block_number * kv_block_stride_m + kv_head_idx * kv_head_stride_m + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        //k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        k_vecs[j] = quant_handler.LoadAndDequantize(
            reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2),
            kv_quant_handler_t::K,
            physical_block_number,
            physical_block_offset,
            kv_head_idx,
            vec_idx * VEC_SIZE);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // Due to block quantize on the Head elements, the computation is different with non-quant version
  // The number of elements per vector.
  constexpr int V_VEC_SIZE = VEC_SIZE;

  // A vector of V elements for the current timestep.
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

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


  //==================================================================================================================
  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride_m + kv_head_idx * kv_head_stride_m;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j <= V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
