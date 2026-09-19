// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

using onnxruntime::cuda::CeilDiv;
using onnxruntime::cuda::GridDim;

// A 2-bit weight blob packs four codes per byte, code i at bit offset 2*(i%4) of byte i/4
// (the layout produced by MlasQuantizeBlockwise<T, 2>). One uint32 load therefore covers
// 16 consecutive codes, which is also the widest aligned store (32 B for fp16, 64 B for fp32).
constexpr int kElementsPerThread2b = 16;
constexpr int kElementsPerByte2b = 4;
constexpr uint8_t kDefaultZeroPoint2b = 2;  // 1 << (bits - 1)

__device__ __forceinline__ float ToFloat2b(float v) { return v; }
__device__ __forceinline__ float ToFloat2b(half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat2b(nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ uint8_t UnpackZeroPoint2b(const uint8_t* zero_points, int64_t byte_index, int block_index) {
  return static_cast<uint8_t>((zero_points[byte_index] >> ((block_index & 0x03) << 1)) & 0x03);
}

template <class T>
__device__ __forceinline__ void StoreSixteen2b(const T* src, T* dst) {
  constexpr int kVectors = static_cast<int>(kElementsPerThread2b * sizeof(T) / sizeof(float4));
#pragma unroll
  for (int i = 0; i < kVectors; ++i) {
    reinterpret_cast<float4*>(dst)[i] = reinterpret_cast<const float4*>(src)[i];
  }
}

// Dequantizes 16 codes with a single (scale, zero point) pair. The arithmetic is done in fp32 and
// rounded once on store, which is at least as accurate as the reference CPU dequantizer.
template <class T>
__device__ __forceinline__ void DequantizeSixteen2b(uint32_t values_quant, T scale, float zero_point, T* output) {
  const float scale_f = ToFloat2b(scale);
  const float zp_adjust = -scale_f * zero_point;
  alignas(16) T results[kElementsPerThread2b];
#pragma unroll
  for (int i = 0; i < kElementsPerThread2b; ++i) {
    const float code = static_cast<float>((values_quant >> (2 * i)) & 0x03u);
    results[i] = static_cast<T>(fmaf(code, scale_f, zp_adjust));
  }
  StoreSixteen2b<T>(results, output);
}

template <class T, typename ZeroT>
__global__ void Dequantize2BitsKernel(
    T* output,
    const uint8_t* quant_data,
    const T* scale_data,
    const ZeroT* zero_points,
    int block_size,
    int groups_per_K,
    int groups_per_threadblock,
    int total_groups) {
  const int block_id = blockIdx.x * groups_per_threadblock + ((threadIdx.x * kElementsPerThread2b) / block_size);
  if (block_id >= total_groups) {
    return;
  }
  const int64_t element_offset = static_cast<int64_t>(block_id) * block_size +
                                 ((threadIdx.x * kElementsPerThread2b) & (block_size - 1));
  const uint32_t quant_value = *(reinterpret_cast<const uint32_t*>(quant_data + element_offset / kElementsPerByte2b));
  const T scale = *(scale_data + block_id);

  float zero_point_value;
  if constexpr (std::is_same_v<ZeroT, uint8_t>) {
    const int zero_point_shape_x = (groups_per_K + kElementsPerByte2b - 1) / kElementsPerByte2b;
    const int kb_idx = block_id % groups_per_K;
    const int n_idx = block_id / groups_per_K;
    uint8_t zp = kDefaultZeroPoint2b;
    if (zero_points) {
      zp = UnpackZeroPoint2b(zero_points,
                             static_cast<int64_t>(n_idx) * zero_point_shape_x + kb_idx / kElementsPerByte2b, kb_idx);
    }
    zero_point_value = static_cast<float>(zp);
  } else {
    zero_point_value = zero_points ? ToFloat2b(*(zero_points + block_id))
                                   : static_cast<float>(kDefaultZeroPoint2b);
  }

  DequantizeSixteen2b<T>(quant_value, scale, zero_point_value, output + element_offset);
}

}  // namespace

template <class T, typename ZeroT>
Status Dequantize2Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const ZeroT* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream) {
  // k is padded and equal to groups_per_K * block_size. block_size is a power of two >= 16
  // (enforced by matmul_nbits_helper::CheckInputs), so each thread's 16 codes always fall inside
  // one quantization block and one aligned uint32 of the packed blob.
  ORT_ENFORCE(k % block_size == 0, "k must be a multiplier of block_size");
  ORT_ENFORCE(block_size >= kElementsPerThread2b && (block_size & (block_size - 1)) == 0,
              "2-bit dequantization requires block_size to be a power of two >= 16, got ", block_size);
  ORT_RETURN_IF(reorder_idx != nullptr, "CUDA 2-bit dequantization does not support g_idx (reorder_idx).");
  ORT_ENFORCE(block_size <= 256, "2-bit dequantization block_size must not exceed 256, got ", block_size);

  const int groups_per_K = k / block_size;
  const int total_groups = n * groups_per_K;
  const int groups_per_threadblock = GridDim::maxThreadsPerBlock * kElementsPerThread2b / block_size;
  const int groups_per_grid = CeilDiv(total_groups, groups_per_threadblock);
  dim3 grid_dim(groups_per_grid);
  dim3 block_dim(GridDim::maxThreadsPerBlock);

  Dequantize2BitsKernel<T, ZeroT><<<grid_dim, block_dim, 0, stream>>>(
      output,
      quant_data,
      scales_data,
      zero_points,
      block_size,
      groups_per_K,
      groups_per_threadblock,
      total_groups);

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_DEQUANTIZE_2BITS(T, ZeroT)                                              \
  template Status Dequantize2Bits<T, ZeroT>(T * output, const uint8_t* quant_data,          \
                                            const T* scales_data, const ZeroT* zero_points, \
                                            const int32_t* reorder_idx, int k, int n,       \
                                            int block_size, cudaStream_t stream)

INSTANTIATE_DEQUANTIZE_2BITS(float, uint8_t);
INSTANTIATE_DEQUANTIZE_2BITS(half, uint8_t);
INSTANTIATE_DEQUANTIZE_2BITS(nv_bfloat16, uint8_t);
INSTANTIATE_DEQUANTIZE_2BITS(float, float);
INSTANTIATE_DEQUANTIZE_2BITS(half, half);
INSTANTIATE_DEQUANTIZE_2BITS(nv_bfloat16, nv_bfloat16);

#undef INSTANTIATE_DEQUANTIZE_2BITS

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
