// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

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
// #include "paged_dtype_bfloat16.cuh"
#include "paged_utils.cuh"

using namespace onnxruntime::cuda;

#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)

namespace onnxruntime {
namespace contrib {
namespace cuda {

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

struct __align__(8) Half2x2 {
  half2 x;
  half2 y;
};

template <>
class QuantVec<uint32_t> {
 public:
  using Type = char2;
};

template <>
class QuantVec<half2> {
 public:
  using Type = char2;
};

template <>
class QuantVec<uint2> {
 public:
  using Type = Char2x2;
};

template <>
class QuantVec<Half2x2> {
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
inline __device__ TVec DequantizeByScaleVec(
    const typename QuantVec<TVec>::Type quant_vec_m,
    const typename FloatVec<TVec>::Type unit_scales);

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
inline __device__ TVec LoadQuantVecByScales(const TVec* q8, const typename FloatVec<TVec>::Type unit_scales) {
  using TQuantVec = typename QuantVec<TVec>::Type;
  TQuantVec quant_vec = *(const TQuantVec*)q8;
  return DequantizeByScaleVec<TVec>(quant_vec, unit_scales);
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
inline __device__ __half MaxAbsFloat(const __half2 v) {
  const __half2 h2 = __habs2(v);
  return __hmax(h2.x, h2.y);
}

template <>
inline __device__ __half MaxAbsFloat(const Half2x2 v) {
  // make it simple rather than save one op
  return __hmax(MaxAbsFloat<__half, __half2>(v.x), MaxAbsFloat<__half, __half2>(v.y));
}

template <>
inline __device__ __half MaxAbsFloat(const uint4 v) {
  return __hmax(__hmax(MaxAbsFloat<__half, uint32_t>(v.x), MaxAbsFloat<__half, uint32_t>(v.y)),
                __hmax(MaxAbsFloat<__half, uint32_t>(v.z), MaxAbsFloat<__half, uint32_t>(v.w)));
}

template <typename TVec>
inline __device__ typename QuantVec<TVec>::Type Quantize(const TVec v, const float scale);

template <>
inline __device__ char2 Quantize(const half2 h2, const float inv_unit_scale) {
  float2 f2 = __half22float2(h2);
  return char2{(char)min(max(-127, __float2int_rn(inv_unit_scale * f2.x)), 127),
               (char)min(max(-127, __float2int_rn(inv_unit_scale * f2.y)), 127)};
}

template <>
inline __device__ char2 Quantize(const uint32_t v, const float inv_unit_scale) {
  union __align__(4) {
    uint32_t u;
    __half2 h2;
  }
  uh2 = {v};
  return Quantize(uh2.h2, inv_unit_scale);
}

template <>
inline __device__ Char2x2 Quantize(const uint2 v, const float inv_unit_scale) {
  Char2x2 ch2x2;
  ch2x2.x = Quantize<uint32_t>(v.x, inv_unit_scale);
  ch2x2.y = Quantize<uint32_t>(v.y, inv_unit_scale);
  return ch2x2;
}

template <>
inline __device__ Char2x2 Quantize(const Half2x2 v, const float inv_unit_scale) {
  Char2x2 ch2x2;
  ch2x2.x = Quantize<half2>(v.x, inv_unit_scale);
  ch2x2.y = Quantize<half2>(v.y, inv_unit_scale);
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

template <typename scalar_t>
class NoQuantProcessor {
 public:
  using kv_mem_scalar_t = scalar_t;
  static constexpr bool IS_QUANTIZED = false;
  kv_mem_scalar_t* dummy = nullptr;
};

template <typename quant_params_elem_t>  // float or float16
class KVQuantProcessor {
 public:
  using kv_mem_scalar_t = int8_t;
  using kv_quant_param_t = quant_params_elem_t;
  static constexpr bool IS_QUANTIZED = true;

  KVQuantProcessor(
      const kv_quant_param_t* params_cache,
      int kv_quant_chunk_size,
      int num_kv_heads,
      int head_size,
      int block_size)
      : kv_quant_params_cache_(params_cache),
        kv_quant_chunk_size_(kv_quant_chunk_size),
        block_size_(block_size),
        head_stride_(block_size * (head_size / kv_quant_chunk_size)),
        k_or_v_stride_(num_kv_heads * head_stride_) {
    assert(head_size % kv_quant_chunk_size == 0);
  }

  template <typename VecT>
  inline __device__ VecT LoadAndDequantizeK(
      const VecT* ptr,
      const int block_id,
      const int in_block_token_id,
      const int valid_tokens_in_block,
      const int head_id,
      const int in_head_idx) const {
    const kv_quant_param_t* kv_param_block = kv_quant_params_cache_ + (int64_t)(block_id * 2) * k_or_v_stride_;
    float unit_scale = 0.0f;
    if (in_block_token_id < valid_tokens_in_block) {
      unit_scale = (float)*(kv_param_block + (head_id * head_stride_ + (in_head_idx / kv_quant_chunk_size_) * block_size_ + in_block_token_id));
    }
    return LoadQuantVec(ptr, unit_scale);
  }

  template <typename VecT>
  inline __device__ VecT LoadAndDequantizeV(
      const VecT* ptr,
      const int block_id,
      const int in_block_token_id,
      const int head_id,
      const int in_head_idx) const {
    const kv_quant_param_t* kv_param_block = kv_quant_params_cache_ + (int64_t)(block_id * 2 + 1) * k_or_v_stride_;
    if constexpr (std::is_same<kv_quant_param_t, float>::value) {
      const VecT float_scale_vec = *(const VecT*)(kv_param_block + (head_id * head_stride_ + (in_head_idx / kv_quant_chunk_size_) * block_size_ + in_block_token_id));
      return LoadQuantVecByScales(ptr, float_scale_vec);
    } else {
      const VecT scale_vec = *(const VecT*)(kv_param_block + (head_id * head_stride_ + (in_head_idx / kv_quant_chunk_size_) * block_size_ + in_block_token_id));
      using ScaleFp32Vec = typename FloatVec<VecT>::Type;
      const ScaleFp32Vec float_scale_vec = to_float(scale_vec);
      return LoadQuantVecByScales(ptr, float_scale_vec);
    }
  }

  const kv_quant_param_t* kv_quant_params_cache_;  // [num_blocks, 2, num_kv_heads, head_size / kv_quant_chunk_size, block_size]
  const int kv_quant_chunk_size_;                  // for how many consecutive values, calc one scale for them to quant
  const int block_size_;
  const int head_stride_;
  const int k_or_v_stride_;
};

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// Grid: (num_heads, num_seqs).
template <
    typename scalar_t,
    typename kv_quant_handler_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void single_query_cached_kv_attention_quant_kernel(
    scalar_t* __restrict__ out,                                                // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,                                            // [num_seqs, num_heads, head_size]
    const typename kv_quant_handler_t::kv_mem_scalar_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const typename kv_quant_handler_t::kv_mem_scalar_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* __restrict__ head_mapping,                                      // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const kv_quant_handler_t quant_handler) {
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

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  if constexpr (kv_quant_handler_t::IS_QUANTIZED) {
    assert(quant_handler.kv_quant_chunk_size_ % VEC_SIZE == 0);
  }

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

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const int valid_tokens_in_block = max(0, min(context_len - block_idx * BLOCK_SIZE, BLOCK_SIZE));

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
        if constexpr (!kv_quant_handler_t::IS_QUANTIZED) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else {
          k_vecs[j] = quant_handler.LoadAndDequantizeK(
              reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2),
              physical_block_number,
              physical_block_offset,
              valid_tokens_in_block,
              kv_head_idx,
              vec_idx * VEC_SIZE);
        }
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

    const auto* v_ptr = v_cache + physical_block_number * kv_block_stride_m + kv_head_idx * kv_head_stride_m;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;

        V_vec v_vec;
        if constexpr (!kv_quant_handler_t::IS_QUANTIZED) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          v_vec = quant_handler.LoadAndDequantizeV(
              reinterpret_cast<const V_vec*>(v_ptr + offset),
              physical_block_number,
              physical_block_offset,
              kv_head_idx,
              row_idx);
        }

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

// Prerequest:
//    kv_quant_chunk_size % 4 == 0
//    head_size % 4 == 0
//    head_size % kv_quant_chunk_size == 0
// num_threads_per_chunk = min(32, kv_quant_chunk_size / 4) round up to power 2
// num_chunks_per_warp = 32 / num_threads_per_chunk
// num_warps_per_head = ceil( (head_size / kv_quant_chunk_size) / num_chunks_per_warp )
// grid: [num_tokens, num_heads]
// block: [num_warps_per_head * 32]
__global__ void quantize_reshape_and_cache_kernel(
    const half* __restrict__ key,          // [num_tokens, num_heads, head_size]
    const half* __restrict__ value,        // [num_tokens, num_heads, head_size]
    int8_t* __restrict__ key_cache,        // [num_blocks, num_heads, head_size/x, block_size, x]
    int8_t* __restrict__ value_cache,      // [num_blocks, num_heads, head_size, block_size]
    const int* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    void* __restrict__ kv_param_cache,  // [num_blocks, 2, num_heads, head_size / kv_quant_chunk_size, block_size]
    const int kv_quant_chunk_size,
    bool use_fp32_scales) {
  using Vec = Half2x2;
  using QVec = QuantVec<Vec>::Type;
  constexpr int VEC_SIZE = 4;
  constexpr int MAX_NUM_ITER_PER_CHUNK = 2;

  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;  // in block token idx

  assert(head_size % kv_quant_chunk_size == 0);
  assert(kv_quant_chunk_size % VEC_SIZE == 0);
  assert(head_size % VEC_SIZE == 0);
  assert(x % VEC_SIZE == 0);

  // round it up to power of 2
  int num_threads_per_chunk = min(WARP_SIZE, kv_quant_chunk_size / VEC_SIZE);
  if ((num_threads_per_chunk & (num_threads_per_chunk - 1)) != 0) {
    num_threads_per_chunk = 1 << (32 - __clz(num_threads_per_chunk));
  }
  const int num_chunks_per_head = head_size / kv_quant_chunk_size;
  const int quant_chunk_idx = threadIdx.x / num_threads_per_chunk;
  const int quant_chunk_offset = threadIdx.x % num_threads_per_chunk;

  const int num_h_per_iter_per_chunk = num_threads_per_chunk * VEC_SIZE;
  const int num_iters = (kv_quant_chunk_size + num_h_per_iter_per_chunk - 1) / num_h_per_iter_per_chunk;

  half2 max_kv_fp16{0, 0};
  Vec k_vecs[MAX_NUM_ITER_PER_CHUNK];
  Vec v_vecs[MAX_NUM_ITER_PER_CHUNK];

  int64_t src_k_idx = (int64_t)token_idx * key_stride + head_idx * head_size + quant_chunk_idx * kv_quant_chunk_size;
  int64_t src_v_idx = (int64_t)token_idx * value_stride + head_idx * head_size + quant_chunk_idx * kv_quant_chunk_size;
  for (int iter = 0; iter < num_iters; iter++) {
    int h_in_chunk = quant_chunk_offset * VEC_SIZE + iter * num_h_per_iter_per_chunk;
    if (quant_chunk_idx < num_chunks_per_head && h_in_chunk < kv_quant_chunk_size) {
      k_vecs[iter] = *(const Vec*)(&key[src_k_idx + h_in_chunk]);
      v_vecs[iter] = *(const Vec*)(&value[src_v_idx + h_in_chunk]);
    } else {  // make sure all threads are alive to reduce
      k_vecs[iter] = Vec{{CUDART_ZERO_FP16, CUDART_ZERO_FP16}, {CUDART_ZERO_FP16, CUDART_ZERO_FP16}};
      v_vecs[iter] = k_vecs[iter];
    }
    max_kv_fp16 = __hmax2(max_kv_fp16, __half2{MaxAbsFloat<__half, Half2x2>(k_vecs[iter]), MaxAbsFloat<__half, Half2x2>(v_vecs[iter])});
    for (int mask = num_threads_per_chunk / 2; mask >= 1; mask /= 2) {
      max_kv_fp16 = __hmax2(max_kv_fp16, __shfl_xor_sync(uint32_t(-1), max_kv_fp16, mask));
    }
  }

  float2 max_kv_fp32 = __half22float2(max_kv_fp16);
  float inv_unit_scale_k = max_kv_fp16.x ? (127.0f / max_kv_fp32.x) : 0.0f;
  float inv_unit_scale_v = max_kv_fp16.y ? (127.0f / max_kv_fp32.y) : 0.0f;
  if (quant_chunk_idx < num_chunks_per_head) {

    if (quant_chunk_offset == 0) {
      int stride = block_size * (head_size / kv_quant_chunk_size);
      int64_t k_offset = ((int64_t)block_idx * num_heads * 2 + head_idx) * stride + quant_chunk_idx * block_size + block_offset;
      int64_t v_offset = k_offset + num_heads * stride;
      float unit_scale_k = max_kv_fp32.x / 127.0f;
      float unit_scale_v = max_kv_fp32.y / 127.0f;
      if (use_fp32_scales) {
        *((float*)kv_param_cache + k_offset) = unit_scale_k;
        *((float*)kv_param_cache + v_offset) = unit_scale_v;
      } else {
        *((half*)kv_param_cache + k_offset) = __float2half(unit_scale_k);
        *((half*)kv_param_cache + v_offset) = __float2half(unit_scale_v);
      }
    }

    int64_t tgt_kv_idx = ((int64_t)block_idx * num_heads + head_idx) * head_size * block_size;
    for (int iter = 0; iter < num_iters; iter++) {
      int h_in_chunk = quant_chunk_offset * VEC_SIZE + iter * num_h_per_iter_per_chunk;
      if (h_in_chunk < kv_quant_chunk_size) {
        QVec ch2x2_k = Quantize(k_vecs[iter], inv_unit_scale_k);
        QVec ch2x2_v = Quantize(v_vecs[iter], inv_unit_scale_v);

        int h = quant_chunk_offset * VEC_SIZE + iter * num_h_per_iter_per_chunk + quant_chunk_idx * kv_quant_chunk_size;
        const int tgt_key_idx = tgt_kv_idx + ((h / x) * block_size * x + block_offset * x + (h % x));
        const int tgt_value_idx = tgt_kv_idx + (h * block_size + block_offset);

        *(QVec*)(&key_cache[tgt_key_idx]) = ch2x2_k;
        value_cache[tgt_value_idx] = ch2x2_v.x.x;
        value_cache[tgt_value_idx + 1 * block_size] = ch2x2_v.x.y;
        value_cache[tgt_value_idx + 2 * block_size] = ch2x2_v.y.x;
        value_cache[tgt_value_idx + 3 * block_size] = ch2x2_v.y.y;
      }
    }
  }
}

}  // namespace vllm

#define LAUNCH_ATTENTION_KERNEL(T, QH, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)                 \
  vllm::single_query_cached_kv_attention_quant_kernel<T, QH, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS> \
      <<<grid, block, shared_mem_size, stream>>>(                                          \
          out_ptr,                                                                         \
          query_ptr,                                                                       \
          (const typename QH::kv_mem_scalar_t*)key_cache_ptr,                              \
          (const typename QH::kv_mem_scalar_t*)value_cache_ptr,                            \
          head_mapping_ptr,                                                                \
          scale,                                                                           \
          block_tables_ptr,                                                                \
          context_lens_ptr,                                                                \
          max_num_blocks_per_seq,                                                          \
          alibi_slopes_ptr,                                                                \
          query_stride,                                                                    \
          kv_block_stride,                                                                 \
          kv_head_stride,                                                                  \
          quant_handler)

// TODO(woosuk): Tune NUM_THREADS.
template <
    typename T,
    int BLOCK_SIZE,
    int NUM_THREADS = 128>
void single_query_cached_kv_attention_quant_launcher(
    const cudaStream_t stream,
    T* out,
    const T* query,
    const void* key_cache,
    const void* value_cache,
    const int* head_mapping,
    float scale,
    const int* block_tables,
    const int max_num_blocks_per_seq,
    const int* context_lens,
    int max_context_len,
    const float* alibi_slopes_ptr,
    const int64_t* query_shapes,
    int num_queries_per_kv,
    const void* kv_quant_params,
    int kv_quant_chunk_size,
    int kv_quant_param_dtype) {
    if constexpr (std::is_same<T, uint16_t>::value) {
      int num_seqs = query_shapes[0];
      int num_heads = query_shapes[1];
      int head_size = query_shapes[2];
      // int max_num_blocks_per_seq = 1;            // block_tables.size(1);xxxxxxxxxxxxxxxxxxxxxxxx
      int query_stride = head_size * num_heads;                                       // query.stride(0);
      int kv_block_stride = num_heads * head_size / num_queries_per_kv * BLOCK_SIZE;  // xxxxxxxxxxxxxxxxxxxxxxxx
      int kv_head_stride = head_size * BLOCK_SIZE;                                    // key_cache.stride(1);

      // int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
      assert(head_size % (MAX(WARP_SIZE / BLOCK_SIZE, 1)) == 0);

      T* out_ptr = reinterpret_cast<T*>(out);
      const T* query_ptr = reinterpret_cast<const T*>(query);
      const T* key_cache_ptr = reinterpret_cast<const T*>(key_cache);
      const T* value_cache_ptr = reinterpret_cast<const T*>(value_cache);
      const int* head_mapping_ptr = reinterpret_cast<const int*>(head_mapping);

      const int* block_tables_ptr = block_tables;
      const int* context_lens_ptr = context_lens;

      constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
      int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
      int logits_size = padded_max_context_len * sizeof(float);
      int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
      int shared_mem_size = std::max<int>(logits_size, outputs_size);

      dim3 grid(num_heads, num_seqs);
      dim3 block(NUM_THREADS);
      assert(kv_quant_params != nullptr);

      using QuantHandler = typename vllm::KVQuantProcessor<half>;
      QuantHandler quant_handler(
          (const QuantHandler::kv_quant_param_t*)kv_quant_params,
          kv_quant_chunk_size,
          num_heads / num_queries_per_kv,
          head_size,
          BLOCK_SIZE);
      switch (head_size) {
      // NOTE(woosuk): To reduce the compilation time, we omitted head sizes
      // 32, 160, 192, 256.
      // case 32:
      //   LAUNCH_ATTENTION_KERNEL(T, QuantHandler, 32, BLOCK_SIZE, NUM_THREADS);
      //   break;
      case 64:
        LAUNCH_ATTENTION_KERNEL(T, QuantHandler, 64, BLOCK_SIZE, NUM_THREADS);
        break;
      case 80:
        LAUNCH_ATTENTION_KERNEL(T, QuantHandler, 80, BLOCK_SIZE, NUM_THREADS);
        break;
      case 96:
        LAUNCH_ATTENTION_KERNEL(T, QuantHandler, 96, BLOCK_SIZE, NUM_THREADS);
        break;
      case 128:
        LAUNCH_ATTENTION_KERNEL(T, QuantHandler, 128, BLOCK_SIZE, NUM_THREADS);
        break;
      // case 160:
      //   LAUNCH_ATTENTION_KERNEL(T, QuantHandler, 160, BLOCK_SIZE, NUM_THREADS);
      //   break;
      // case 192:
      //   LAUNCH_ATTENTION_KERNEL(T, QuantHandler, 192, BLOCK_SIZE, NUM_THREADS);
      //   break;
      // case 256:
      //   LAUNCH_ATTENTION_KERNEL(T, QuantHandler, 256, BLOCK_SIZE, NUM_THREADS);
      //   break;
      default:
        // TORCH_CHECK(false, "Unsupported head size: ", head_size);
        abort();
        break;
    }
  } else {
  // Do not support
  abort();
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                       \
  single_query_cached_kv_attention_quant_launcher<T, BLOCK_SIZE>( \
      stream,                                                     \
      (T*)out,                                                    \
      (const T*)query,                                            \
      (const void*)key_cache,                                     \
      (const void*)value_cache,                                   \
      (const int*)head_mapping,                                   \
      scale,                                                      \
      block_tables,                                               \
      max_num_blocks_per_seq,                                     \
      context_lens,                                               \
      max_context_len,                                            \
      alibi_slopes_ptr,                                           \
      query_shapes,                                               \
      num_queries_per_kv,                                         \
      kv_quant_params,                                            \
      kv_quant_chunk_size,                                        \
      kv_quant_param_dtype)

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T) \
  switch (block_size) {                    \
    /* case 1:                         */  \
    /*   CALL_KERNEL_LAUNCHER(T, 1);   */  \
    /*   break;                        */  \
    /* case 2:                         */  \
    /*   CALL_KERNEL_LAUNCHER(T, 2);   */  \
    /*   break;                        */  \
    /* case 4:                         */  \
    /*   CALL_KERNEL_LAUNCHER(T, 4);   */  \
    /*   break;                        */  \
    case 8:                                \
      CALL_KERNEL_LAUNCHER(T, 8);          \
      break;                               \
    case 16:                               \
      CALL_KERNEL_LAUNCHER(T, 16);         \
      break;                               \
    case 32:                               \
      CALL_KERNEL_LAUNCHER(T, 32);         \
      break;                               \
    /* case 64:                        */  \
    /*   CALL_KERNEL_LAUNCHER(T, 64);  */  \
    /*   break;                        */  \
    /* case 128:                       */  \
    /*   CALL_KERNEL_LAUNCHER(T, 128); */  \
    /*   break;                        */  \
    /* case 256:                       */  \
    /*   CALL_KERNEL_LAUNCHER(T, 256); */  \
    /*   break;                        */  \
    default:                               \
      abort();                             \
      break;                               \
  }

void reshape_and_cache_quant(
    const cudaStream_t stream,
    const void* key,          // [num_tokens, num_heads, head_size]
    const void* value,        // [num_tokens, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int* slot_mapping,  // [num_tokens]
    const int64_t* key_shapes,
    const int64_t* value_shapes,
    const int64_t block_size,
    const int vec_x,
    const int dtype,
    void* kv_quant_param,
    const int kv_quant_chunk_size,
    const int kv_quant_param_dtype) {
  int num_tokens = key_shapes[0];
  int num_heads = key_shapes[1];
  int head_size = key_shapes[2];
  // int block_size = key_cache.size(3);
  int x = vec_x;

  int key_stride = key_shapes[1] * key_shapes[2];
  int value_stride = value_shapes[1] * value_shapes[2];

  // static_assert(std::is_same_v<T, MLFloat16>, "Unsupported data type: ");

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  // if constexpr (std::is_same_v<T, MLFloat16>) {
  if (dtype == 1) {
    // round it up to power of 2
    constexpr int VEC_SIZE = 4;
    int num_iters = (kv_quant_chunk_size / VEC_SIZE + (32 - 1)) / 32;
    assert(num_iters == 1 || num_iters == 2);

    int num_threads_per_chunk = std::max(32, kv_quant_chunk_size / VEC_SIZE);
    int pow = (int)log2(num_threads_per_chunk);
    if (num_threads_per_chunk > (1 << pow)) {
      num_threads_per_chunk = (1 << (pow + 1));
    }
    const int num_chunks_per_warp = 32 / num_threads_per_chunk;
    const int num_warps_per_head = ((head_size / kv_quant_chunk_size + num_chunks_per_warp - 1) / num_chunks_per_warp);

    dim3 grid(num_tokens, num_heads);
    dim3 block(num_warps_per_head * 32);
    vllm::quantize_reshape_and_cache_kernel<<<grid, block, 0, stream>>>(
        (const half*)key,
        (const half*)value,
        (int8_t*)key_cache,
        (int8_t*)value_cache,
        slot_mapping,
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x,
        kv_quant_param,
        kv_quant_chunk_size,
        kv_quant_param_dtype == 0);

  }
}

void single_query_cached_kv_attention_quant(
    const cudaStream_t stream,
    const void* out,          // [num_seqs, num_heads, head_size]
    const void* query,        // [num_seqs, num_heads, head_size]
    const void* key_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const void* value_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int* head_mapping,  // [num_heads]
    float scale,
    const int* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int max_num_blocks_per_seq,
    const int* context_lens,  // [num_seqs]
    int block_size,
    int max_context_len,
    const float* __restrict__ alibi_slopes_ptr,
    const int64_t* query_shapes,
    int num_queries_per_kv,
    int dtype,
    const void* kv_quant_params,  // [num_blocks, 2, num_kv_heads, head_size / kv_quant_chunk_size, block_size]
    int kv_quant_chunk_size,
    int kv_quant_param_dtype) {
  // static_assert(std::is_same_v<T, float> || std::is_same_v<T, BFloat16> || std::is_same_v<T, MLFloat16>, "Unsupported data type: ");
  // if constexpr (std::is_same_v<T, float>) {
  if (dtype == 0) {  // float
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
  } else if (dtype == 1) {  // half
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (dtype == 2) {  //} else if constexpr (std::is_same_v<T, BFloat16>) {
    // CALL_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  }
}

#undef CALL_KERNEL_LAUNCHER
#undef LAUNCH_ATTENTION_KERNEL
#undef CALL_KERNEL_LAUNCHER_BLOCK_SIZE
#undef WARP_SIZE
#undef MAX
#undef MIN

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
