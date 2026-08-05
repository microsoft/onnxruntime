// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ROCm dequantize kernels for MatMulNBits.
//
// These are plain HIP kernels with no arch-specific intrinsics (no dp4a, no
// MFMA, no WMMA), so they compile and run on every ROCm target including
// gfx900 (Vega 10) which has no native dp4a instruction.
//
// Layout (column-wise blockwise quantisation — the standard HuggingFace
// ONNX exporter layout, and the layout used by the Qwen3-TTS model):
//
//   quant_data  : uint8, shape [N, k_blocks, block_size*bits/8]
//                 4-bit: low nibble = element index 2i+0,
//                        high nibble = element index 2i+1 inside each block.
//   scales      : T,     shape [N, k_blocks]
//   zero_points : uint8, shape [N, ceil(k_blocks/2)]  (packed nibbles, optional)
//                 default implicit zero-point = 8 (unsigned 4-bit midpoint)
//
//   output      : T,     shape [N, K_padded]  row-major

#include "contrib_ops/rocm/quantization/dequantize_nbits_rocm.cuh"
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace onnxruntime {
namespace contrib {
namespace rocm {

// ---------------------------------------------------------------------------
// 4-bit dequantize kernel
// Each thread processes 8 output elements (one uint32 = 8 nibbles).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void Dequantize4BitsKernel(
    T* __restrict__ output,
    const unsigned char* __restrict__ quant_data,
    const T* __restrict__ scale_data,
    const unsigned char* __restrict__ zero_points,
    int block_size,
    int groups_per_K,
    int groups_per_tb,
    int total_groups) {
  const int block_id =
      blockIdx.x * groups_per_tb + (threadIdx.x * 8) / block_size;
  if (block_id >= total_groups) return;

  const int element_offset =
      block_id * block_size + (threadIdx.x * 8) % block_size;

  // Single 32-bit load covers 8 nibbles.
  const unsigned int quant_value =
      *reinterpret_cast<const unsigned int*>(quant_data + element_offset / 2);

  const T scale = scale_data[block_id];

  // Zero-point: packed nibbles; default = 8 (midpoint for unsigned 4-bit).
  unsigned char zp_raw = 8u;
  if (zero_points) {
    const int zero_point_shape_x = (groups_per_K + 1) / 2;
    const int kb_idx = block_id % groups_per_K;
    const int n_idx  = block_id / groups_per_K;
    const unsigned char packed =
        zero_points[n_idx * zero_point_shape_x + kb_idx / 2];
    zp_raw = (kb_idx & 1) ? (packed >> 4) : (packed & 0x0fu);
  }

  const T zp_adjust = static_cast<T>(-static_cast<float>(scale) *
                                      static_cast<float>(zp_raw));

  T* out = output + element_offset;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned char nibble =
        static_cast<unsigned char>((quant_value >> (4 * i)) & 0xFu);
    out[i] = static_cast<T>(nibble) * scale + zp_adjust;
  }
}

// ---------------------------------------------------------------------------
// 8-bit dequantize kernel — each thread processes 4 output elements.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void Dequantize8BitsKernel(
    T* __restrict__ output,
    const unsigned char* __restrict__ quant_data,
    const T* __restrict__ scale_data,
    const unsigned char* __restrict__ zero_points,
    int block_size,
    int groups_per_K,
    int groups_per_tb,
    int total_groups) {
  const int block_id =
      blockIdx.x * groups_per_tb + (threadIdx.x * 4) / block_size;
  if (block_id >= total_groups) return;

  const int element_offset =
      block_id * block_size + (threadIdx.x * 4) % block_size;

  const unsigned int quant_value =
      *reinterpret_cast<const unsigned int*>(quant_data + element_offset);

  const T scale = scale_data[block_id];

  unsigned char zp_raw = 128u;
  if (zero_points) {
    const int kb_idx = block_id % groups_per_K;
    const int n_idx  = block_id / groups_per_K;
    zp_raw = zero_points[n_idx * groups_per_K + kb_idx];
  }

  const T zp_adjust = static_cast<T>(-static_cast<float>(scale) *
                                      static_cast<float>(zp_raw));

  T* out = output + element_offset;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned char byte =
        static_cast<unsigned char>((quant_value >> (8 * i)) & 0xFFu);
    out[i] = static_cast<T>(byte) * scale + zp_adjust;
  }
}

// ---------------------------------------------------------------------------
// Launch helpers (definitions of the declarations in the .cuh)
// ---------------------------------------------------------------------------
template <typename T>
hipError_t LaunchDequantize4Bits(
    T* output,
    const unsigned char* quant_data,
    const T* scales,
    const unsigned char* zero_points,
    int K_padded,
    int N,
    int block_size,
    hipStream_t stream) {
  constexpr int kThreadsPerBlock = 256;
  constexpr int kElemsPerThread = 8;

  const int groups_per_K = K_padded / block_size;
  const int total_groups = N * groups_per_K;
  const int groups_per_tb =
      (kThreadsPerBlock * kElemsPerThread) / block_size;
  const int grid = (total_groups + groups_per_tb - 1) / groups_per_tb;

  Dequantize4BitsKernel<T><<<grid, kThreadsPerBlock, 0, stream>>>(
      output, quant_data, scales, zero_points,
      block_size, groups_per_K, groups_per_tb, total_groups);

  return hipGetLastError();
}

template <typename T>
hipError_t LaunchDequantize8Bits(
    T* output,
    const unsigned char* quant_data,
    const T* scales,
    const unsigned char* zero_points,
    int K_padded,
    int N,
    int block_size,
    hipStream_t stream) {
  constexpr int kThreadsPerBlock = 256;
  constexpr int kElemsPerThread = 4;

  const int groups_per_K = K_padded / block_size;
  const int total_groups = N * groups_per_K;
  const int groups_per_tb =
      (kThreadsPerBlock * kElemsPerThread) / block_size;
  const int grid = (total_groups + groups_per_tb - 1) / groups_per_tb;

  Dequantize8BitsKernel<T><<<grid, kThreadsPerBlock, 0, stream>>>(
      output, quant_data, scales, zero_points,
      block_size, groups_per_K, groups_per_tb, total_groups);

  return hipGetLastError();
}

// Explicit instantiations.
template hipError_t LaunchDequantize4Bits<float>(
    float*, const unsigned char*, const float*, const unsigned char*, int, int, int, hipStream_t);
template hipError_t LaunchDequantize4Bits<__half>(
    __half*, const unsigned char*, const __half*, const unsigned char*, int, int, int, hipStream_t);

template hipError_t LaunchDequantize8Bits<float>(
    float*, const unsigned char*, const float*, const unsigned char*, int, int, int, hipStream_t);
template hipError_t LaunchDequantize8Bits<__half>(
    __half*, const unsigned char*, const __half*, const unsigned char*, int, int, int, hipStream_t);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
