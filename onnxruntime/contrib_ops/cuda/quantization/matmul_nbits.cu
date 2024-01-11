// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_nbits.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ T WarpUniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  p.asInt = __shfl_sync(0xffffffff, (unsigned)p.asInt, 0);
  return p.value;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
// Convert 8 4bits integer stored in one uint32_t to 8 halfs.
// 8 4bits with order 0,1,2,3,4,5,6,7,8 will be converted to 8 halfs with order 0,4,1,5,2,6,3,7
__device__ __forceinline__ void Convert8xInt4To8xHalfs(uint32_t value, half2* half_2x4) {
  uint32_t* h = reinterpret_cast<uint32_t*>(half_2x4);

  // From https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
  // First, we extract the i4s and construct an intermediate fp16 number.
  constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
  constexpr uint32_t kBottomMask = 0x000f000f;
  constexpr uint32_t kTopMask = 0x00f000f0;
  constexpr uint32_t kI4sToF16sMagicNum = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
  // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
  // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
  // elt_67 to fp16 without having to shift them to the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
  // immediately before required.
  const uint32_t top_i4s = value >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(value), "n"(kBottomMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(value), "n"(kTopMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(kBottomMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(kTopMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));

  // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
  // half2 ctor. In this case, I chose performance reliability over code readability.

  // This is the half2 {1024, 1024} represented as an integer.
  constexpr uint32_t kFp16TopMagicNum = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  constexpr uint32_t kOneSixteenth = 0x2c002c00;
  // This is the half2 {-64, -64} represented as an integer.
  constexpr uint32_t kNeg64 = 0xd400d400;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(kFp16TopMagicNum));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(kOneSixteenth), "r"(kNeg64));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(kFp16TopMagicNum));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(kOneSixteenth), "r"(kNeg64));
}

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  constexpr uint32_t kLowHalf2 = 0x5410;
  constexpr uint32_t kHighHalf2 = 0x7632;

  uint4 vec_permuted;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.x) : "r"(vec_a.x), "r"(vec_a.z), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.y) : "r"(vec_a.x), "r"(vec_a.z), "r"(kHighHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.z) : "r"(vec_a.y), "r"(vec_a.w), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.w) : "r"(vec_a.y), "r"(vec_a.w), "r"(kHighHalf2));

  half2 elements[4];  // [04, 15, 26, 37]

  Convert8xInt4To8xHalfs(values_quant, elements);

  half2 v0 = elements[0] * scale_half2 + zp_adjust2;
  half2 v1 = elements[1] * scale_half2 + zp_adjust2;
  half2 v2 = elements[2] * scale_half2 + zp_adjust2;
  half2 v3 = elements[3] * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_permuted.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_permuted.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_permuted.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_permuted.w))));
}
#else
__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  half2 element01 = __halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF));
  half2 element23 = __halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF));
  half2 element45 = __halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF));
  half2 element67 = __halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF));

  half2 v0 = element01 * scale_half2 + zp_adjust2;
  half2 v1 = element23 * scale_half2 + zp_adjust2;
  half2 v2 = element45 * scale_half2 + zp_adjust2;
  half2 v3 = element67 * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_a.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_a.w))));
}
#endif

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, float scale, uint8_t zp, const float* a, float* sums) {
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xF) * scale + zp_adjust;
  float v1 = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  float v2 = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  float v3 = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  float v4 = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  float v5 = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  float v6 = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  float v7 = float((values_quant >> 28) & 0xF) * scale + zp_adjust;

  sums[0] += v0 * a_vec_0.x;
  sums[1] += v1 * a_vec_0.y;
  sums[2] += v2 * a_vec_0.z;
  sums[3] += v3 * a_vec_0.w;
  sums[4] += v4 * a_vec_1.x;
  sums[5] += v5 * a_vec_1.y;
  sums[6] += v6 * a_vec_1.z;
  sums[7] += v7 * a_vec_1.w;
}

constexpr int kColsPerThreadBlock = 8;
constexpr int kWarpSize = 32;

// kernel for 4bits quantized gemv, i.e., computing A(1,K) x B(K, N)
// B(K, N) is quantized blockwise with 4bits and stored as [N, (K + block_size - 1)/block_size, blob]
// The thread block size is (kWarpSize, kColsPerThreadBlock) and grid size is (N/kColsPerThreadBlock, 1)
// Each thread block computes [1, K] x [kColsPerThreadBlock, (K + block_size - 1)/block_size, blob],
//     i.e., computing kColsPerThreadBlock per block and a warp reduce (1, K) x (K)
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloatInt4Kernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int n_block_id = blockIdx.x;
  const int m_id = blockIdx.y;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = n_block_id * kColsPerThreadBlock + warp_id;
  constexpr int k_per_iter = 256;

  extern __shared__ char shared_buffer[];
  // load scale to shared buffer
  T* b_scale_vec = (T*)shared_buffer;
  int offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
  for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  uint8_t* b_zp_vec;
  (void)b_zp_vec;
  if constexpr (has_zero_point) {
    b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + kColsPerThreadBlock * blocks_per_K);
    const int b_zp_k = (blocks_per_K + 1) / 2;
    int zp_offset = n_block_id * kColsPerThreadBlock * b_zp_k;
    for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * b_zp_k; i += kColsPerThreadBlock * kWarpSize) {
      b_zp_vec[2 * i] = (zero_points[zp_offset + i] & 0x0f);
      b_zp_vec[2 * i + 1] = (zero_points[zp_offset + i] >> 4);
    }
    b_zp_vec += warp_id * b_zp_k * 2;
  }
  __syncthreads();

  a_data += m_id * k + (lane_id << 3);

  b_scale_vec += warp_id * blocks_per_K;

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

#define UnRollReduction(unroll_size)                                                              \
  do {                                                                                            \
    constexpr int kUnroll = unroll_size;                                                          \
    constexpr int kUnrollMask = 0xffffffff & (~(kUnroll * k_per_iter - 1));                       \
    for (; k_id < (k & kUnrollMask); k_id += kUnroll * k_per_iter) {                              \
      _Pragma("unroll") for (int i = 0; i < kUnroll; i++) {                                       \
        uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_per_iter / 2 * i)); \
        T scale = b_scale_vec[t_meta_k + k_per_iter / block_size * i];                            \
        uint8_t zp = 8;                                                                           \
        if constexpr (has_zero_point) {                                                           \
          zp = b_zp_vec[t_meta_k + k_per_iter / block_size * i];                                  \
        }                                                                                         \
        AccumulateEightElements(value, scale, zp, a_data + k_id + i * k_per_iter, sums);          \
      }                                                                                           \
      b_data_quant += k_per_iter / 2 * kUnroll;                                                   \
      t_meta_k += k_per_iter / block_size * kUnroll;                                              \
    }                                                                                             \
  } while (false)

  UnRollReduction(16);
  UnRollReduction(4);
  UnRollReduction(1);
#undef UnRollReduction

  // handle reminder
  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[t_meta_k];
    uint8_t zp = 8;
    if constexpr (has_zero_point) {
      zp = b_zp_vec[t_meta_k];
    }
    AccumulateEightElements(value, scale, zp, a_data + k_id, sums);
  }

  float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  // warp reduction
  for (int i = 16; i > 0; i = i / 2) {
    sum += __shfl_down_sync(0xffffffff, sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}  // namespace cuda

template <class T>
bool TryMatMul4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream) {
  if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > 1) {
    return false;
  }
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);
  dim3 threads(kWarpSize, kColsPerThreadBlock);
  int blocks_per_K = (k + block_size - 1) / block_size;
  int shared_mem_size = sizeof(T) * blocks_per_K * kColsPerThreadBlock +
                        (zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

#define MatMulFloatInt4KernelDispatch(block_size)                                              \
  if (nullptr != zero_points) {                                                                \
    MatMulFloatInt4Kernel<T, block_size, true><<<blocks, threads, shared_mem_size, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);        \
  } else {                                                                                     \
    MatMulFloatInt4Kernel<T, block_size, false><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);        \
  }

  if (16 == block_size) {
    MatMulFloatInt4KernelDispatch(16);
  } else if (32 == block_size) {
    MatMulFloatInt4KernelDispatch(32);
  } else if (64 == block_size) {
    MatMulFloatInt4KernelDispatch(64);
  } else if (128 == block_size) {
    MatMulFloatInt4KernelDispatch(128);
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }

#undef MatMulFloatInt4KernelDispatch

  return true;
}

template bool TryMatMul4Bits<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

template bool TryMatMul4Bits<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);


namespace GPTQPacking {
constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;
const int width_element_per_block = 32 * 2;
template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
  if (WarpSize >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (WarpSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (WarpSize >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (WarpSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (WarpSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}
template <typename T, typename ZEROT>
__global__ void MatMulW4A16Kernel(T* out, const T* inA, const uint32_t* inB, const T* scales, const ZEROT* qzeros,
                                  uint32_t groupsize, const uint32_t matrix_M, const uint32_t matrix_k, const uint32_t matrix_N) {
  const uint32_t block_k = ((matrix_k + 31) / 32 + 7) / 8 * 8;

  int bid = blockIdx.x;
  __shared__ float bsum[2][32][32 + 1];
  float sum[2] = {0, 0};
  int y_start = threadIdx.y * block_k;

  half2 res2 = {};
  half2 res2_1 = {};

  const half2* inA_start = (const half2*)(inA + blockIdx.y * matrix_k + y_start);

  int n_offset_x = bid * width_element_per_block + threadIdx.x * 2;

  int start_group_id = (y_start / groupsize);
  int compressed_idx = threadIdx.x % 4;
  half2 scale = ((const half2*)(scales + start_group_id * matrix_N + n_offset_x))[0];
  half2 hzero;
  uint32_t qzero_p;
  if constexpr (std::is_same<ZEROT, uint32_t>::value) {
    qzero_p = qzeros == nullptr ? 0x88888888 : ((qzeros + n_offset_x / 8 + start_group_id * ((matrix_N + 7) / 8)))[0];
    hzero = __halves2half2(
        __int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
        __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
  } else {
    hzero = ((const half2*)(qzeros + start_group_id * matrix_N + n_offset_x))[0];
  }
  half2 scale_h0 = __half2half2(scale.x);
  half2 scale_h1 = __half2half2(scale.y);
  half2 hzero_scale_0 = __half2half2(hzero.x * scale.x);
  half2 hzero_scale_1 = __half2half2(hzero.y * scale.y);

#pragma unroll
  for (int i = 0; i < block_k / 2; i += 4) {  // read half2 * 4
    res2 = {};
    res2_1 = {};
    int k_offset = y_start + i * 2;
    int g_id = k_offset / groupsize;

    if (g_id > start_group_id) {
      scale = ((const half2*)(scales + g_id * matrix_N + n_offset_x))[0];
      if constexpr (std::is_same<ZEROT, uint32_t>::value) {
        qzero_p = ((qzeros + n_offset_x / 8 +
                    g_id * ((matrix_N + 7) / 8)))[0];
        hzero = __halves2half2(
            __int2half_rn((qzero_p >> (8 * (compressed_idx))) & 0xf),
            __int2half_rn(((qzero_p) >> (8 * (compressed_idx) + 4)) & 0xf));
      } else {
        hzero = ((const half2*)(qzeros + g_id * matrix_N + n_offset_x))[0];
      }
      scale_h0 = __half2half2(scale.x);
      scale_h1 = __half2half2(scale.y);
      hzero_scale_0 = __half2half2(hzero.x * scale.x);
      hzero_scale_1 = __half2half2(hzero.y * scale.y);
      start_group_id = g_id;
    }

    const uint32_t* hinB = inB + n_offset_x + k_offset / 8 * matrix_N;
    uint32_t vbInt1 =
        (n_offset_x < matrix_N && (k_offset < matrix_k)) ? hinB[0] : int32_t(0);
    uint32_t vbInt2 = (n_offset_x + 1 < matrix_N && (k_offset < matrix_k))
                          ? (hinB)[1]
                          : int32_t(0);
    half2 vb[8];
    const uint8_t* qweight_p1 = (const uint8_t*)&vbInt1;
    const uint8_t* qweight_p2 = (const uint8_t*)&vbInt2;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      // vb[j] = __halves2half2(__int2half_rn(((vbInt1 >> (j * 8))) & 0xF),
      //                        __int2half_rn(((vbInt1) >> (j*8+4)) & 0xF));
      // vb[j + 4] = __halves2half2(__int2half_rn(((vbInt2)>>(j*8)) & 0xF),
      //                            __int2half_rn((((vbInt2) >> (j*8+4))) &
      //                            0xF));
      vb[j] = __halves2half2(__int2half_rn(((*(qweight_p1 + j))) & 0xF),
                             __int2half_rn(((*(qweight_p1 + j)) >> 4) & 0xF));
      vb[j + 4] =
          __halves2half2(__int2half_rn(((*(qweight_p2 + j))) & 0xF),
                         __int2half_rn((((*(qweight_p2 + j)) >> 4)) & 0xF));
    }

    half2 va[4];
    va[0] = (k_offset < matrix_k) ? ((inA_start))[i] : res2;
    va[1] = (k_offset + 1 < matrix_k) ? ((inA_start))[i + 1] : res2;
    va[2] = (k_offset + 2 < matrix_k) ? ((inA_start))[i + 2] : res2;
    va[3] = (k_offset + 3 < matrix_k) ? ((inA_start))[i + 3] : res2;

#pragma unroll
    for (int j = 0; j < 4; j++) {
      vb[j] = __hfma2(scale_h0, vb[j], -hzero_scale_0);
      res2 = __hfma2(va[j], vb[j], res2);
      vb[4 + j] = __hfma2(scale_h1, vb[4 + j], -hzero_scale_1);
      res2_1 = __hfma2(va[j], vb[4 + j], res2_1);
    }

    sum[0] += __half2float(res2.x) + __half2float(res2.y);
    sum[1] += __half2float(res2_1.x) + __half2float(res2_1.y);
  }
  // sum[0] += __half2float(res2.x);
  // sum[1] +=  __half2float(res2.y);
  bsum[0][threadIdx.x][threadIdx.y] = sum[0];
  bsum[1][threadIdx.x][threadIdx.y] = sum[1];

  __syncthreads();
  sum[0] = 0;
  sum[1] = 0;

#pragma unroll
  for (int i = 0; i < 2; i++) {
    sum[i] = bsum[i][threadIdx.y][threadIdx.x];
    __syncthreads();
    sum[i] = warpReduceSum<32>(sum[i]);
    if (threadIdx.x == 0) {
      out[+blockIdx.y * matrix_N + bid * width_element_per_block +
          threadIdx.y * 2 + i] = __float2half_rn(sum[i]);
    }
  }
}

constexpr int kBlockOutput = 32;
constexpr int kMaxInputBatchInThread = 1;

template <typename scalar_t, int WBITS>
__global__ void MatMulW4A16GidxKernel(const scalar_t* __restrict__ input,
                                      const int32_t* __restrict__ qweight, scalar_t* __restrict__ output,
                                      const scalar_t* __restrict__ scales,
                                      const uint32_t* __restrict__ qzeros,
                                      const int* __restrict__ g_idx, uint32_t mat_m,
                                      uint32_t mat_k, uint32_t mat_n, uint32_t zero_width) {
  const int num_thread_group = kBlockSize / kNumWaves;
  const int thread_num_k = (mat_k + num_thread_group - 1) / num_thread_group;
  const int thread_idx_group = threadIdx.y;
  const int thread_group_start = thread_idx_group * thread_num_k;

  const int compress_group_size = 32 / WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;

  const int weight_x = blockIdx.x * kBlockOutput + threadIdx.x;

  __shared__ float blocksum[kMaxInputBatchInThread][num_thread_group]
                           [kBlockOutput];
  float sum[kMaxInputBatchInThread];
#pragma unroll
  for (int bid = 0; bid < kMaxInputBatchInThread; bid++) {
    sum[bid] = 0;
  }
  const int end_k = min(mat_k, thread_group_start + thread_num_k);
  int input_start_y = blockIdx.y * kMaxInputBatchInThread;
  int input_end_y = min(mat_m, input_start_y + kMaxInputBatchInThread);
  int len_input_y = input_end_y - input_start_y;
  for (int weight_y = thread_group_start; weight_y < end_k; weight_y++) {
    scalar_t input_vec[kMaxInputBatchInThread];
    for (int bid = 0; bid < len_input_y; bid++) {
      input_vec[bid] = input[(input_start_y + bid) * mat_k + weight_y];
    }
    int scale_row = g_idx[weight_y];

    scalar_t scale_v = scales[scale_row * mat_n + weight_x];
    uint32_t zero_v =
        qzeros == nullptr
            ? 0x88888888
            : qzeros[scale_row * zero_width + (weight_x / compress_group_size)];
    int zero_ind = weight_x % compress_group_size;
    uint8_t zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;

    scalar_t scale_zeros = __hmul(scale_v, __ushort2half_rn(zv1));

    uint32_t weight_int = qweight[(weight_y / compress_group_size) * mat_n + weight_x];
    int weight_ind = (weight_y) % compress_group_size;
    uint8_t wv1 = (weight_int >> (weight_ind * WBITS)) & max_num_in_bits;
    scalar_t wv = __ushort2half_rn(wv1);
    scalar_t weight = __hfma(wv, scale_v, -scale_zeros);
    // sum = __hfma(weight, input_v, sum);
    for (int bid = 0; bid < len_input_y; bid++) {
      sum[bid] += __half2float(weight * input_vec[bid]);
    }
  }
  for (int bid = 0; bid < len_input_y; bid++) {
    if constexpr (!std::is_same<scalar_t, float>::value) {
      blocksum[bid][thread_idx_group][threadIdx.x] = sum[bid];  //__half2float(sum);
    } else {
      blocksum[bid][thread_idx_group][threadIdx.x] = sum[bid];
    }
  }
  for (unsigned int s = 1; s < num_thread_group; s *= 2) {
    __syncthreads();
    int index = 2 * s * thread_idx_group;
    if (index < num_thread_group) {
      for (int bid = 0; bid < len_input_y; bid++) {
        blocksum[bid][index][threadIdx.x] +=
            blocksum[bid][index + s][threadIdx.x];
      }
    }
  }
  for (int bid = 0; bid < len_input_y; bid++) {
    if (thread_idx_group == 0) {
      if constexpr (!std::is_same<scalar_t, float>::value) {
        output[(input_start_y + bid) * mat_n + blockIdx.x * kBlockOutput +
               threadIdx.x] = __float2half_rn(blocksum[bid][0][threadIdx.x]);
      } else {
        output[(input_start_y + bid) * mat_n + blockIdx.x * kBlockOutput +
               threadIdx.x] = blocksum[bid][0][threadIdx.x];
      }
    }
  }
}

template <typename ZeroT>
void TryMatMul4Bits(
    cudaStream_t stream,
    const void* input_data,
    const int32_t* qweight_data,
    void* mul_out_data,
    const void* scales_data,
    const ZeroT* zeros_data,
    uint32_t matrix_M,
    uint32_t matrix_k,
    uint32_t matrix_N,
    uint32_t groupsize) {
  const int block_k = ((matrix_k + 31) / 32 + 7) / 8 * 8;

  dim3 gridDim = {(matrix_N + width_element_per_block - 1) / width_element_per_block, matrix_M};
  dim3 blockDim = {32, (matrix_k + block_k - 1) / block_k};
  MatMulW4A16Kernel<half><<<gridDim, blockDim, 0, stream>>>(
      static_cast<half*>(mul_out_data), static_cast<const half*>(input_data),
      reinterpret_cast<const uint32_t*>(qweight_data), static_cast<const half*>(scales_data),
      zeros_data, groupsize, matrix_M, matrix_k, matrix_N);
}

template void TryMatMul4Bits<uint32_t>(
    cudaStream_t stream,
    const void* input_data,
    const int32_t* qweight_data,
    void* mul_out_data,
    const void* scales_data,
    const uint32_t* zeros_data,
    uint32_t matrix_M,
    uint32_t matrix_k,
    uint32_t matrix_N,
    uint32_t groupsize);
template void TryMatMul4Bits<half>(
    cudaStream_t stream,
    const void* input_data,
    const int32_t* qweight_data,
    void* mul_out_data,
    const void* scales_data,
    const half* zeros_data,
    uint32_t matrix_M,
    uint32_t matrix_k,
    uint32_t matrix_N,
    uint32_t groupsize);
template <typename T>
    __forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

void TryMatMul4BitsGidx(
    cudaStream_t stream,
    const void* input,
    const int32_t* qweight,
    void* output,
    const void* scales,
    const int32_t* qzeros,
    const int32_t* g_idx,
    const int64_t* shapes) {
  auto matricx_m = static_cast<uint32_t>(shapes[0]);
  auto matricx_k = static_cast<uint32_t>(shapes[1]);
  auto matricx_n = static_cast<uint32_t>(shapes[2]);
  auto zero_width = static_cast<uint32_t>(shapes[3]);

  dim3 blocks(ceil_div<uint32_t>(matricx_n, kBlockOutput),
              ceil_div<uint32_t>(matricx_m, kMaxInputBatchInThread));
  dim3 threads(kBlockOutput, kBlockSize / kBlockOutput);

  using scalar_t = __half;
  MatMulW4A16GidxKernel<scalar_t, 4><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const scalar_t*>(input), qweight, reinterpret_cast<scalar_t*>(output),
      reinterpret_cast<const scalar_t*>(scales), reinterpret_cast<const uint32_t*>(qzeros), g_idx, matricx_m, matricx_k, matricx_n, zero_width);
}
}
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
