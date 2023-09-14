// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_with_quant_weight.cuh"

using namespace cub;

namespace nvidia_cuda = cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  half2 v0 = __hfma2(__halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF)), scale_half2, zp_adjust2);
  half2 v1 = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF)), scale_half2, zp_adjust2);
  half2 v2 = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF)), scale_half2, zp_adjust2);
  half2 v3 = __hfma2(__halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF)), scale_half2, zp_adjust2);
  v0 = __hmul2(v0, *(reinterpret_cast<half2*>(&(vec_a.x))));
  v1 = __hmul2(v1, *(reinterpret_cast<half2*>(&(vec_a.y))));
  v2 = __hfma2(v2, *(reinterpret_cast<half2*>(&(vec_a.z))), v0);
  v3 = __hfma2(v3, *(reinterpret_cast<half2*>(&(vec_a.w))), v1);
  v3 = __hadd2(v2, v3);
  return float(v3.x) + float(v3.y);
}

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, float scale, uint8_t zp, const float* a) {
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

  v0 = v0 * a_vec_0.x;
  v1 = v1 * a_vec_0.y;
  v2 = v2 * a_vec_0.z;
  v3 = v3 * a_vec_0.w;
  v4 = v4 * a_vec_1.x + v0;
  v5 = v5 * a_vec_1.y + v1;
  v6 = v6 * a_vec_1.z + v2;
  v7 = v7 * a_vec_1.w + v3;
  return v4 + v5 + v6 + v7;
}

// constexpr int BLOCKSIZEN = 8;
// constexpr int BLOCKSIZEN_HALF = 4;

template <class T, int group_size, int BLOCKSIZEN>
__global__ void MatMulFloatInt4Kernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k) {
  int n_block_id = blockIdx.x;
  int m_id = blockIdx.y;
  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int n_id = n_block_id * BLOCKSIZEN + warp_id;
  int group_count = (k + group_size - 1) / group_size;
  int thread_id = warp_id * 32 + lane_id;
  int k_iter = k / 256;

  extern __shared__ char shared_buffer[];

  // load scale to shared buffer
  T* b_scale_vec = (T*)shared_buffer;
  uint8_t* b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + BLOCKSIZEN * group_count);
  int offset = n_block_id * BLOCKSIZEN * group_count;
  for (int i = thread_id; i < BLOCKSIZEN * group_count; i += 32 * BLOCKSIZEN) {
    b_scale_vec[i] = scales_data[offset + i];
    b_zp_vec[i] = zero_points != nullptr ? zero_points[offset + i] : uint8_t(8);
  }
  __syncthreads();

  a_data += m_id * k;
  b_data_quant += n_id * group_count * (group_size / 2);

  float sum = 0.f;
  int k_id = 0;

  constexpr int stages_count = 2;
  constexpr int elements_per_ite = 256;
  constexpr int bytes_per_thread = 4;
  __shared__ alignas(alignof(float4)) T a_shared[stages_count][elements_per_ite];

  nvidia_cuda::pipeline<nvidia_cuda::thread_scope_thread> pipe = nvidia_cuda::make_pipeline();

  int k_step = 0;
  int fetch = 0;
  int shared_offset = std::is_same_v<T, float> ? thread_id : (thread_id << 1);
  for (; k_id < (k & 0xffffff00); k_id += elements_per_ite, k_step++) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + (k_id >> 1) + lane_id * 4));
    // fetch from global to shared
    for (; fetch < k_iter && fetch < (k_step + stages_count); fetch++) {
      pipe.producer_acquire();
      nvidia_cuda::memcpy_async(&a_shared[fetch % 2][shared_offset],
                                a_data + fetch * elements_per_ite + shared_offset,
                                bytes_per_thread,
                                pipe);
      // nvidia_cuda::memcpy_async(group, a_shared_stages[fetch % 2], a_data + fetch * elements_per_ite, sizeof(T) * elements_per_ite, pipeline);
      pipe.producer_commit();
    }

    pipe.consumer_wait();
    __syncthreads();
    T scale = b_scale_vec[warp_id * group_count + (k_id + lane_id * 8) / group_size];
    uint8_t zp = b_zp_vec[warp_id * group_count + (k_id + lane_id * 8) / group_size];
    sum += AccumulateEightElements(value, scale, zp, a_shared[k_step % 2] + (lane_id << 3));
    pipe.consumer_release();
  }

  // handle reminder
  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_iter * 128 + lane_id * 4));
    T scale = b_scale_vec[warp_id * group_count + (k_id + lane_id * 8) / group_size];
    uint8_t zp = b_zp_vec[warp_id * group_count + (k_id + lane_id * 8) / group_size];
    sum += AccumulateEightElements(value, scale, zp, a_data + k_id + (lane_id << 3));
  }

  // warp reduction
  for (int i = 16; i > 0; i = i / 2) {
    sum += __shfl_down_sync(0xffffffff, sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}

template <class T>
bool TryMatMul4BitsWeight(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int group_size,
    cudaStream_t stream) {
  constexpr int block_size_n = std::is_same_v<T, float> ? 8 : 4;
  if (n % block_size_n != 0 || k % 8 != 0) {
    return false;
  }
  dim3 blocks((n + block_size_n - 1) / block_size_n, m);
  dim3 threads(32, block_size_n);
  int shared_mem_size = (sizeof(T) + 1) * ((k + group_size - 1) / group_size * block_size_n);

  if (16 == group_size) {
    MatMulFloatInt4Kernel<T, 16, block_size_n><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (32 == group_size) {
    MatMulFloatInt4Kernel<T, 32, block_size_n><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (64 == group_size) {
    MatMulFloatInt4Kernel<T, 64, block_size_n><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else if (128 == group_size) {
    MatMulFloatInt4Kernel<T, 128, block_size_n><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k);
  } else {
    ORT_THROW("block size ", group_size, " is not supported");
  }

  return true;
}

template bool TryMatMul4BitsWeight<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

template bool TryMatMul4BitsWeight<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
