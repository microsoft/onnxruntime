// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  half2 elements[4];  // [04, 15, 26, 37]
  uint32_t* h = reinterpret_cast<uint32_t*>(&elements);

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
  // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
  // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
  // elt_67 to fp16 without having to shift them to the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
  // immediately before required.
  const uint32_t top_i4s = values_quant >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(values_quant), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(values_quant), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

  // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
  // half2 ctor. In this case, I chose performance reliability over code readability.

  // This is the half2 {1024, 1024} represented as an integer.
  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // This is the half2 {-64, -64} represented as an integer.
  static constexpr uint32_t NEG_64 = 0xd400d400;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

  half2 v0 = elements[0] * scale_half2 + zp_adjust2;
  half2 v1 = elements[1] * scale_half2 + zp_adjust2;
  half2 v2 = elements[2] * scale_half2 + zp_adjust2;
  half2 v3 = elements[3] * scale_half2 + zp_adjust2;

  sums[0] += (v0.x * ((*(reinterpret_cast<half2*>(&(vec_a.x)))).x));
  sums[1] += (v1.x * ((*(reinterpret_cast<half2*>(&(vec_a.x)))).y));
  sums[2] += (v2.x * ((*(reinterpret_cast<half2*>(&(vec_a.y)))).x));
  sums[3] += (v3.x * ((*(reinterpret_cast<half2*>(&(vec_a.y)))).y));
  sums[4] += (v0.y * ((*(reinterpret_cast<half2*>(&(vec_a.z)))).x));
  sums[5] += (v1.y * ((*(reinterpret_cast<half2*>(&(vec_a.z)))).y));
  sums[6] += (v2.y * ((*(reinterpret_cast<half2*>(&(vec_a.w)))).x));
  sums[7] += (v3.y * ((*(reinterpret_cast<half2*>(&(vec_a.w)))).y));
}

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
template <class T, int block_size>
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

  // blocks_per_k is the number of scales and zero points on the k dim
  const int b_zp_k = (blocks_per_K + 1) / 2;

  extern __shared__ char shared_buffer[];

  // load scale to shared buffer
  T* b_scale_vec = (T*)shared_buffer;
  uint8_t* b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + kColsPerThreadBlock * blocks_per_K);
  int offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
  for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  int zp_offset = n_block_id * kColsPerThreadBlock * b_zp_k;
  for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * b_zp_k; i += kColsPerThreadBlock * kWarpSize) {
    b_zp_vec[i] = zero_points != nullptr ? zero_points[zp_offset + i] : uint8_t(0x88);
  }
  __syncthreads();

  a_data += m_id * k;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

  const int scale_col_offset = warp_id * blocks_per_K;
  const int zp_col_offset = warp_id * b_zp_k;

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  for (; k_id < (k & 0xffffff00); k_id += k_per_iter) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[scale_col_offset + t_meta_k];
    uint8_t zp = b_zp_vec[zp_col_offset + t_meta_k / 2];
    zp = (t_meta_k & 0x01) ? (zp >> 4) : (zp & 0x0f);
    AccumulateEightElements(value, scale, zp, a_data + k_id + (lane_id << 3), sums);
    t_meta_k += k_per_iter / block_size;  // k index for this thread, points to the scale and zero point
    b_data_quant += k_per_iter / 2;
  }

  // handle reminder
  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[scale_col_offset + t_meta_k];
    uint8_t zp = b_zp_vec[zp_col_offset + t_meta_k / 2];
    zp = (t_meta_k & 0x01) ? (zp >> 4) : (zp & 0x0f);
    AccumulateEightElements(value, scale, zp, a_data + k_id + (lane_id << 3), sums);
  }

  float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  // warp reduction
  for (int i = 16; i > 0; i = i / 2) {
    sum += __shfl_down_sync(0xffffffff, sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}

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
  int blocks_per_thread_block = blocks_per_K * kColsPerThreadBlock;
  int shared_mem_size = sizeof(T) * blocks_per_thread_block + blocks_per_thread_block / 2;
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

  if (16 == block_size) {
    MatMulFloatInt4Kernel<T, 16><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (32 == block_size) {
    MatMulFloatInt4Kernel<T, 32><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (64 == block_size) {
    MatMulFloatInt4Kernel<T, 64><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (128 == block_size) {
    MatMulFloatInt4Kernel<T, 128><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }

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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
