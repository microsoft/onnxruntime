/***************************************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include "masked_multihead_attention.h"
#include "masked_multihead_attention_utils.h"
#include <assert.h>
#include <float.h>

//#define MMHA_USE_HMMA_FOR_REDUCTION

// Below are knobs to extend FP32 accumulation for higher FP16 accuracy

// Does not seem to affect the accuracy that much
//#define MMHA_USE_FP32_ACUM_FOR_FMA

// Seems to slightly improve the accuracy
#define MMHA_USE_FP32_ACUM_FOR_OUT

#if 0 && defined(MMHA_USE_FP32_ACUM_FOR_OUT)
// Does not seem to improve the accuracy
//#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#endif

namespace mmha { 

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.
//
// The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use 
// 64, 128 and 256 threads per block.
//
// Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to 
// compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The 
// cache buffer helps with memory accesses and contains keys with bias.
//
// The layout of the cache buffer for the keys is [B, H, Dh/x, L, x] where x == 8 for FP16 and 
// x == 4 for FP32 where the fastest moving dimension (contiguous data) is the rightmost one. The
// values for x are chosen to create chunks of 16 bytes.
//
// The different kernels use 1, 2 or 4 threads per key (THREADS_PER_KEY). The size of the LDGs 
// depends on the number of threads per key. Each thread sums Dh / THREADS_PER_KEY elements. At
// the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
// HMMA instruction (Tensor Core). Each Q * K^T valuey is stored in shared memory in FP32.
//
// After that loop, a parallel softmax is computed accross the different Q * K^T values stored in
// shared memory.
//
// The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
// timesteps are computed by loop iteration. As with the keys, the values are read from a cache 
// except for the current timestep. The layout of the cache buffer for the values is much simpler
// as it is [B, H, L, Dh].
//

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T, int Dh >
struct Qk_vec_ {};

template<> struct Qk_vec_<float,     32> { using Type = float;    };
template<> struct Qk_vec_<float,     64> { using Type = float2;   };
template<> struct Qk_vec_<float,    128> { using Type = float4;   };
template<> struct Qk_vec_<uint16_t,  32> { using Type = uint32_t; };
template<> struct Qk_vec_<uint16_t,  64> { using Type = uint32_t; };
template<> struct Qk_vec_<uint16_t, 128> { using Type = uint2;    };

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T, int THREADS_PER_KEY >
struct K_vec_ {};

template<> struct K_vec_<float,    4> { using Type = float;    };
template<> struct K_vec_<float,    2> { using Type = float2;   };
template<> struct K_vec_<float,    1> { using Type = float4;   };
template<> struct K_vec_<uint16_t, 4> { using Type = uint32_t; };
template<> struct K_vec_<uint16_t, 2> { using Type = uint2;    };
template<> struct K_vec_<uint16_t, 1> { using Type = uint4;    };

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T, int V_VEC_SIZE >
struct V_vec_ {};

template<> struct V_vec_<float,    1> { using Type = float;    };
template<> struct V_vec_<float,    2> { using Type = float2;   };
template<> struct V_vec_<float,    4> { using Type = float4;   };
template<> struct V_vec_<uint16_t, 2> { using Type = uint32_t; };
template<> struct V_vec_<uint16_t, 4> { using Type = uint2;    };
template<> struct V_vec_<uint16_t, 8> { using Type = uint4;    };

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
template< typename T>
struct Qk_vec_acum_fp32_ {};

template<> struct Qk_vec_acum_fp32_<float   > { using Type = float;        };
template<> struct Qk_vec_acum_fp32_<float2  > { using Type = float2;       };
template<> struct Qk_vec_acum_fp32_<float4  > { using Type = float4;       };
//template<> struct Qk_vec_acum_fp32_<uint16_t> { using Type = float;        };
template<> struct Qk_vec_acum_fp32_<uint32_t> { using Type = float2;       };
template<> struct Qk_vec_acum_fp32_<uint2   > { using Type = Float4_;      };

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T>
struct K_vec_acum_fp32_ {};

template<> struct K_vec_acum_fp32_<float   > { using Type = float;        };
template<> struct K_vec_acum_fp32_<float2  > { using Type = float2;       };
template<> struct K_vec_acum_fp32_<float4  > { using Type = float4;       };
template<> struct K_vec_acum_fp32_<uint32_t> { using Type = float2;       };
template<> struct K_vec_acum_fp32_<uint2   > { using Type = Float4_;      };
template<> struct K_vec_acum_fp32_<uint4   > { using Type = Float8_;      };
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
template< typename T >
struct V_vec_acum_fp32_ {};

template<> struct V_vec_acum_fp32_<float   > { using Type = float;    };
template<> struct V_vec_acum_fp32_<float2  > { using Type = float2;   };
template<> struct V_vec_acum_fp32_<float4  > { using Type = float4;   };
template<> struct V_vec_acum_fp32_<uint32_t> { using Type = float2;   };
template<> struct V_vec_acum_fp32_<uint2   > { using Type = Float4_;  };
template<> struct V_vec_acum_fp32_<uint4   > { using Type = Float8_;  };
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int THREADS_PER_KEY, typename K_vec, int N >
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N]) {
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
  using K_vec_acum = typename K_vec_acum_fp32_<K_vec>::Type;
#else
  using K_vec_acum = K_vec;
#endif
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
  #pragma unroll
  for( int ii = 1; ii < N; ++ii ) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
  #pragma unroll
  for( int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2 ) {
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
  }
  return qk;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T, int THREADS_PER_KEY >
struct Qk_dot {
  template< typename K_vec, int N >
  static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N]) {
    return qk_dot_<THREADS_PER_KEY>(q, k);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 hmma_fp32(const uint2 &a, uint32_t b) {
  float4 c; float zero = 0.f;
  asm volatile( \
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n" \
    "    {%0, %1, %2, %3}, \n" \
    "    {%4, %5}, \n" \
    "    {%6}, \n" \
    "    {%7, %7, %7, %7}; \n" \
      \
      : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w) 
      :  "r"(a.x)   "r"(a.y)
      ,  "r"(b)
      ,  "f"(zero));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
  using K_vec_acum = typename K_vec_acum_fp32_<uint32_t>::Type;
#else
  using K_vec_acum = uint32_t;
#endif
  K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q[0], k[0]);
  #pragma unroll
  for( int ii = 1; ii < N; ++ii ) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    uint32_t qk_vec_ = float2_to_half2(qk_vec);
    return hmma_fp32(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
  return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
  return 0.f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Qk_dot<uint16_t, 4> {
  template< int N >
  static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N]) {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA_FOR_REDUCTION)
    return qk_hmma_dot_(q, k);
#else
    return qk_dot_<4>(q, k);
#endif // defined MMHA_USE_HMMA_FOR_REDUCTION
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int WARPS_PER_BLOCK, int WARP_SIZE = 32 >
inline __device__ float block_sum(float *red_smem, float sum) {

  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
  #pragma unroll
  for( int mask = WARP_SIZE / 2; mask >= 1; mask /= 2 ) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if( lane == 0 ) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if( lane < WARPS_PER_BLOCK ) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
  #pragma unroll
  for( int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2 ) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float &dst, float src) {
  dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint16_t &dst, float src) {
  dst = float_to_half(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint32_t &dst, float2 src) {
  dst = float2_to_half2(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint2 &dst, Float4_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(uint4 &dst, Float8_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
  dst.z = float2_to_half2(src.z);
  dst.w = float2_to_half2(src.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float2 &dst, float2 src) {
  dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float4 &dst, float4 src) {
  dst = src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float convert_to_float(float4 u) {
  return u.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float convert_to_float(uint4 u) {
  float2 tmp = half2_to_float2(u.x);
  return tmp.x;
}

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float cast_to_float(float u) {
  return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(float2 u) {
  return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 cast_to_float(float4 u) {
  return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(Float4_ u) {
  return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(Float8_ u) {
  return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(uint32_t u) {
  return half2_to_float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(uint2 u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(uint4 u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  tmp.z = half2_to_float2(u.z);
  tmp.w = half2_to_float2(u.w);
  return tmp;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n-1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
inline size_t smem_size_in_bytes(const Masked_multihead_attention_params<T> &params,
                                 int threads_per_value,
                                 int threads_per_block) {
  // The amount of shared memory needed to store the Q*K^T values in float.
  size_t qk_sz = div_up(params.timestep + 1, 4) * 16;

  // The extra memory needed if we are not using floats for the final logits.
  size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
  if( sizeof(T) != 4 ) {
    logits_sz = div_up(params.seq_length, 4) * 4 * sizeof(T);
  }
#endif

  // The total size needed during softmax.
  size_t softmax_sz = qk_sz + logits_sz;

  // The number of partial rows to reduce in the final reduction.
  int rows_per_red = threads_per_block / threads_per_value;
  // The amount of storage needed to finalize the outputs.
  size_t red_sz = rows_per_red * params.hidden_size_per_head * sizeof(T) / 2;

  // The max.
  return max(softmax_sz, red_sz);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ constexpr uint32_t shfl_mask(int threads) { 
  return threads == 32 ? uint32_t(-1) : (1u << threads) - 1u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< 
  // The type of the inputs. Supported types: float and half.
  typename T, 
  // The hidden dimension per head.
  int Dh,
  // The number of threads per key.
  int THREADS_PER_KEY,
  // The number of threads per value.
  int THREADS_PER_VALUE,
  // The number of threads in a threadblock.
  int THREADS_PER_BLOCK
>
__global__ void masked_multihead_attention_kernel(Masked_multihead_attention_params<T> params) {
  // Make sure the hidden dimension per head is a multiple of the number of threads per key.
  static_assert(Dh % THREADS_PER_KEY == 0, "");
  // Make sure the hidden dimension per head is a multiple of the number of threads per value.
  static_assert(Dh % THREADS_PER_VALUE == 0, "");

  // The size of a warp.
  constexpr int WARP_SIZE = 32;
  // The number of warps in a threadblock.
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  // Use smem_size_in_bytes (above) to determine the amount of shared memory.
  extern __shared__ char smem_[];

  // The shared memory for the Q*K^T values and partial logits in softmax.
  float *qk_smem = reinterpret_cast<float*>(smem_);

  // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
  char *logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
  if( sizeof(T) != 4 ) {
    logits_smem_ += div_up(params.timestep + 1, 4) * 16; //sizeof(float);
  }
  T *logits_smem = reinterpret_cast<T*>(logits_smem_);
#else
  float *logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

  // The shared memory to do the final reduction for the output values. Reuse qk_smem.
  T *out_smem = reinterpret_cast<T*>(smem_);

  // The shared memory buffers for the block-wide reductions. One for max, one for sum.
  __shared__ float red_smem[WARPS_PER_BLOCK * 2];
  // Shared memory to store Q inputs.
  __shared__ T q_smem[Dh];

  // A vector of Q or K elements for the current timestep.
  using Qk_vec = typename Qk_vec_<T, Dh>::Type;
  // The number of elements per vector.
  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
  // Make sure the hidden size per head is a multiple of the vector size.
  static_assert(Dh % QK_VEC_SIZE == 0 && Dh / QK_VEC_SIZE <= WARP_SIZE, "");
  // The number of vectors per warp.
  constexpr int QK_VECS_PER_WARP = Dh / QK_VEC_SIZE;

  // The layout of the cache is [B, H, Dh/x, L, x] with x == 4/8 for FP32/FP16. Since each thread
  // owns x elements, we have to decompose the linear index into chunks of x values and the posi-
  // tion of the thread in that chunk. 

  // The number of elements in a chunk of 16B (that's the x in the above formula).
  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
  // The number of K vectors in 16B.
  constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

  // The batch.
  const int bi = blockIdx.y;
  if(params.finished != nullptr && params.finished[bi] == true) return;
  // The head.
  const int hi = blockIdx.x;
  // Combine the batch and the head indices.
  const int bhi = bi * params.num_heads + hi;
  // The thread in the block. 
  const int tidx = threadIdx.x;

  // While doing the product Q*K^T for the different keys we track the max.
  float qk_max = -FLT_MAX;

  int qkv_base_offset = (params.stride == 0)? bhi*Dh : bi*params.stride + hi*Dh;

  // First QK_VECS_PER_WARP load Q and K + the bias values for the current timestep.
  if( tidx < QK_VECS_PER_WARP ) {

    // The offset in the Q and K buffer also accounts for the batch.
    int qk_offset = qkv_base_offset + tidx*QK_VEC_SIZE;
    // The offset in the bias buffer.
    int qk_bias_offset = hi*Dh + tidx*QK_VEC_SIZE;

    // Trigger the loads from the Q and K buffers.
    Qk_vec q = *reinterpret_cast<const Qk_vec*>(&params.q[qk_offset]);
    Qk_vec k = *reinterpret_cast<const Qk_vec*>(&params.k[qk_offset]);

    // Trigger the loads from the Q and K bias buffers.
    Qk_vec q_bias = *reinterpret_cast<const Qk_vec*>(&params.q_bias[qk_bias_offset]);
    Qk_vec k_bias = *reinterpret_cast<const Qk_vec*>(&params.k_bias[qk_bias_offset]);

    // Computes the Q/K values with bias.
    q = add(q, q_bias);
    k = add(k, k_bias);

    // Store the Q values to shared memory.
    *reinterpret_cast<Qk_vec*>(&q_smem[tidx*QK_VEC_SIZE]) = q;

    // Write the K values to the global memory cache. 
    // 
    // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory 
    // system. We designed it this way as it allows much better memory loads (and there are many 
    // more loads) + the stores are really "write and forget" since we won't need the ack before 
    // the end of the kernel. There's plenty of time for the transactions to complete.

    // The 16B chunk written by the thread.
    int co = tidx / QK_VECS_IN_16B;
    // The position of the thread in that 16B chunk.
    int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

    // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
    int offset = bhi*params.seq_length*Dh + 
                 co*params.seq_length*QK_ELTS_IN_16B + 
                 params.timestep*QK_ELTS_IN_16B +
                 ci; 

    // Trigger the stores to global memory. 
    *reinterpret_cast<Qk_vec*>(&params.k_cache[offset]) = k;

    // Compute \sum_i Q[i] * K^T[i] for the current timestep.
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using Qk_vec_acum = typename Qk_vec_acum_fp32_<Qk_vec>::Type;
#else
    using Qk_vec_acum = Qk_vec;
#endif
    float qk = dot<Qk_vec_acum, Qk_vec>(q, k);
    #pragma unroll
    for( int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2 ) {
      qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
    }

    // Normalize qk.
    qk *= params.inv_sqrt_dh;

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if( tidx == 0 ) {
      qk_max = qk;
      qk_smem[params.timestep] = qk;
    }
  }
  
  // Make sure the data is in shared memory.
  __syncthreads();

  // The type of queries and keys for the math in the Q*K^T product.
  using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
  // The number of elements per vector.
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
  // Make sure the hidden size per head is a multiple of the vector size.
  static_assert(Dh % K_VEC_SIZE == 0, "");
  // The number of elements per thread.
  constexpr int K_ELTS_PER_THREAD = Dh / THREADS_PER_KEY;
  // The number of vectors per thread.
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

  // The position the first key loaded by each thread from the cache buffer (for this B * H).
  int ko = tidx / THREADS_PER_KEY;
  // The position of the thread in the chunk of keys.
  int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;

  // Load the Q values from shared memory. The values are reused during the loop on K.
  K_vec q[K_VECS_PER_THREAD];
  #pragma unroll
  for( int ii = 0; ii < K_VECS_PER_THREAD; ++ii ) {
    q[ii] = *reinterpret_cast<const K_vec*>(&q_smem[ki + ii*THREADS_PER_KEY*K_VEC_SIZE]);
  }

  // The number of timesteps loaded per iteration.
  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  // The number of keys per warp.
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  // The base pointer for the key in the cache buffer.
  T *k_cache = &params.k_cache[bhi*params.seq_length*Dh + ki];

  // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
  int ti_end = div_up(params.timestep, K_PER_WARP) * K_PER_WARP;

  // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
  for( int ti = ko; ti < ti_end; ti += K_PER_ITER ) {

    // The keys loaded from the key cache.
    K_vec k[K_VECS_PER_THREAD];
    #pragma unroll
    for( int ii = 0; ii < K_VECS_PER_THREAD; ++ii ) {
      int jj = ii * params.seq_length + ti; 
      if( ti < params.timestep ) {
        k[ii] = *reinterpret_cast<const K_vec*>(&k_cache[jj*QK_ELTS_IN_16B]);
      }
    }

    // Perform the dot product and normalize qk. 
    // 
    // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k) * params.inv_sqrt_dh;

    bool is_mask = params.is_mask? (ti >= params.input_lengths[bi] && ti < params.max_input_len) : false;

    // Store the product to shared memory. There's one qk value per timestep. Update the max.
    if( ti < params.timestep && tidx % THREADS_PER_KEY == 0 ) {

      qk_max = is_mask? qk_max : fmaxf(qk_max, qk);
      qk_smem[ti] = qk;
    }
  }

  // Perform the final reduction to compute the max inside each warp.
  //
  // NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
  // group so it's not needed to run the reduction inside the group (again).
  #pragma unroll
  for( int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2 ) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  // Decompose the thread index into warp and lane.
  const int warp = tidx / WARP_SIZE;
  const int lane = tidx % WARP_SIZE;

  // The warp leader writes the max to shared memory.
  if( lane == 0 ) {
    red_smem[warp] = qk_max;
  }

  // Make sure the products are in shared memory.
  __syncthreads();

  // The warps finalize the reduction.
  qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
  #pragma unroll
  for( int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2 ) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  // Broadcast to all the threads in the warp.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Compute the logits and start the sum.
  float sum = 0.f;

  for( int ti = tidx; ti <= params.timestep; ti += THREADS_PER_BLOCK ) {
    bool is_mask = params.is_mask? (ti >= params.input_lengths[bi] && ti < params.max_input_len) : false;

    float logit = is_mask? 0.f : __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }

  // Compute the sum.
  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  // Normalize the logits.
  float inv_sum = __fdividef(1.f, sum + 1.e-6f);
  for( int ti = tidx; ti <= params.timestep; ti += THREADS_PER_BLOCK ) { 
    convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
  }

  // Make sure the logits are in shared memory.
  __syncthreads();

  // The number of elements per vector.
  constexpr int V_VEC_SIZE = Dh / THREADS_PER_VALUE;
  // A vector of V elements for the current timestep.
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

  // The value computed by this thread.
  int vo = tidx / THREADS_PER_VALUE;
  // The hidden dimensions computed by this particular thread.
  int vi = tidx % THREADS_PER_VALUE * V_VEC_SIZE;

  // The base pointer for the value in the cache buffer.
  T *v_cache = &params.v_cache[bhi*params.seq_length*Dh + vi];

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif
  // The partial outputs computed by each thread.
  V_vec_acum out; zero(out);

  // The number of values processed per iteration of the loop.
  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;

  // Loop over the timesteps to compute the partial outputs.
  for( int ti = vo; ti < params.timestep; ti += V_PER_ITER ) {

    // Load the values from the cache.
    V_vec v = *reinterpret_cast<const V_vec*>(&v_cache[ti*Dh]);
    // Load the logits from shared memory.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
    float logit = logits_smem[ti];
    out = fma(logit, cast_to_float(v), out);
#else
    T logit = logits_smem[ti];

    // Update the partial sums.
    out = fma(logit, v, out);
#endif
  }

  // One group of threads computes the product(s) for the current timestep.
  if( vo == params.timestep % V_PER_ITER ) {

    // Trigger the loads from the V buffer.
    V_vec v = *reinterpret_cast<const V_vec*>(&params.v[qkv_base_offset + vi]);
    // Trigger the loads from the V bias buffer.
    V_vec v_bias = *reinterpret_cast<const V_vec*>(&params.v_bias[hi*Dh + vi]);

    // Compute the V values with bias.
    v = add(v, v_bias);

    // Store the values with bias back to global memory in the cache for V. 
    *reinterpret_cast<V_vec*>(&v_cache[params.timestep*Dh]) = v;

    // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
    out = fma(logits_smem[params.timestep], cast_to_float(v), out);
#else
    out = fma(logits_smem[params.timestep], v, out);
#endif
  }

  // Make sure we can start writing to shared memory.
  __syncthreads();

  // Run the final reduction amongst the different groups computing different partial outputs.
  #pragma unroll
  for( int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2 ) {

    // The midpoint in the number of active groups.
    int midpoint = active_groups / 2;

    // The upper part of active threads store to shared memory.
    if( vo >= midpoint && vo < active_groups ) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
      convert_from_float(*reinterpret_cast<V_vec*>(&out_smem[(vo - midpoint)*Dh + vi]), out);
#else
      *reinterpret_cast<V_vec*>(&out_smem[(vo - midpoint)*Dh + vi]) = out;
#endif
    }
    __syncthreads();

    // The bottom warps update their values.
    if( vo < midpoint ) {
      out = add(*reinterpret_cast<const V_vec*>(&out_smem[vo*Dh + vi]), out);
    }
    __syncthreads();
  }

  // Output the final values.
  if( vo == 0 ) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    convert_from_float(*reinterpret_cast<V_vec*>(&params.out[bhi*Dh + vi]), out);
#else
    *reinterpret_cast<V_vec*>(&params.out[bhi*Dh + vi]) = out;
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace mmha  

////////////////////////////////////////////////////////////////////////////////////////////////////

#define MMHA_LAUNCH_KERNEL(T, Dh, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK, stream) \
  size_t smem_sz = mmha::smem_size_in_bytes<T>(params, THDS_PER_VALUE, THDS_PER_BLOCK); \
  dim3 grid(params.num_heads, params.batch_size); \
  mmha::masked_multihead_attention_kernel<T, Dh, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK> \
    <<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)

////////////////////////////////////////////////////////////////////////////////////////////////////

template < typename T, int Dh >
void  mmha_launch_kernel(const Masked_multihead_attention_params<T> &params, const cudaStream_t &stream) {
  constexpr int THREADS_PER_VALUE = Dh * sizeof(T) / 16;
  if( params.timestep < 32 ) {
    MMHA_LAUNCH_KERNEL(T, Dh, 4, THREADS_PER_VALUE, 64, stream);
  } else if( params.timestep < 2048 ) {
    MMHA_LAUNCH_KERNEL(T, Dh, 2, THREADS_PER_VALUE, 128, stream);
  } else {
    MMHA_LAUNCH_KERNEL(T, Dh, 1, THREADS_PER_VALUE, 256, stream);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
void masked_multihead_attention_(const Masked_multihead_attention_params<T> &params, const cudaStream_t &stream) {
  switch ( params.hidden_size_per_head ) {
    case 32:
      mmha_launch_kernel<T, 32>(params, stream);
      break;
    case 64:
      mmha_launch_kernel<T, 64>(params, stream);
      break;
    case 128:
      mmha_launch_kernel<T, 128>(params, stream);
      break;
    default:
    assert(false);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<float> &params, const cudaStream_t &stream) {
  masked_multihead_attention_(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void masked_multihead_attention(const Masked_multihead_attention_params<uint16_t> &params, const cudaStream_t &stream) {
  masked_multihead_attention_(params, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#undef MMHA_LAUNCH_KERNEL

