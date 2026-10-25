// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_ggml_iq2_xxs.h"
#include "contrib_ops/cuda/quantization/iq2_xxs_grid.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

#include <algorithm>

namespace onnxruntime::contrib::cuda {
namespace {

using namespace onnxruntime::cuda;

template <typename T>
__global__ void MatMulIQ2XXSKernel(const T* a, const uint8_t* b, const T* bias, T* y,
                                   int64_t m, int64_t n, int64_t k_blocks) {
  constexpr int kWarps = 8;
  const int lane = threadIdx.x % 32;
  const int vector = lane % 4;
  const int group = lane / 4;
  const int64_t k = k_blocks * 256;
  // One warp per output. Every lane decodes one eight-value grid vector per superblock.
  for (int64_t output = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
       output < m * n; output += static_cast<int64_t>(gridDim.x) * kWarps) {
    const int64_t row = output / n;
    const int64_t col = output % n;
    float sum = 0.0f;
    for (int64_t block = 0; block < k_blocks; ++block) {
      const uint8_t* packed = b + (col * k_blocks + block) * 66;
      // The 66-byte stride and the two-byte header do not guarantee uint32 alignment.
      const uint16_t scale_bits = static_cast<uint16_t>(packed[0]) |
                                  (static_cast<uint16_t>(packed[1]) << 8);
      const float d = __half2float(__ushort_as_half(scale_bits));
      const uint8_t* record = packed + 2 + group * 8;
      const uint32_t signs_scale = static_cast<uint32_t>(record[4]) |
                                   (static_cast<uint32_t>(record[5]) << 8) |
                                   (static_cast<uint32_t>(record[6]) << 16) |
                                   (static_cast<uint32_t>(record[7]) << 24);
      const unsigned sign_index = (signs_scale >> (7 * vector)) & 127;
      const unsigned signs = sign_index | ((__popc(sign_index) & 1) << 7);
      const float scale = d * (0.5f + static_cast<float>(signs_scale >> 28)) * 0.25f;
      const uint64_t grid = kIQ2XXSGrid[record[vector]];
      const int64_t a_base = row * k + block * 256 + group * 32 + vector * 8;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float magnitude = static_cast<float>((grid >> (8 * j)) & 255);
        // Reconstruct in FP32, then round the complete weight (not its block scale).
        const float weight = to_float<T>(
            from_float<T>(scale * magnitude * ((signs & (1u << j)) ? -1.0f : 1.0f)));
        sum = fmaf(to_float<T>(a[a_base + j]), weight, sum);
      }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
      const T rounded = from_float<T>(sum);
      if (bias != nullptr) {
        y[output] = from_float<T>(to_float<T>(rounded) + to_float<T>(bias[col]));
      } else {
        y[output] = rounded;
      }
    }
  }
}

template <typename T>
void Launch(const void* a, const uint8_t* b, const void* bias, void* y,
            int64_t m, int64_t n, int64_t k_blocks, cudaStream_t stream) {
  const int64_t outputs = m * n;
  const int blocks = static_cast<int>(std::min<int64_t>((outputs - 1) / 8 + 1, 65535));
  MatMulIQ2XXSKernel<<<blocks, 256, 0, stream>>>(
      static_cast<const T*>(a), b, static_cast<const T*>(bias), static_cast<T*>(y),
      m, n, k_blocks);
}

}  // namespace

Status LaunchMatMulGGMLIQ2XXS(const void* a, const uint8_t* b, const void* bias, void* y,
                              int64_t m, int64_t n, int64_t k_blocks,
                              IQ2XXSActivationType type, cudaStream_t stream) {
  switch (type) {
    case IQ2XXSActivationType::Float:
      Launch<float>(a, b, bias, y, m, n, k_blocks, stream);
      break;
    case IQ2XXSActivationType::Float16:
      Launch<half>(a, b, bias, y, m, n, k_blocks, stream);
      break;
    case IQ2XXSActivationType::BFloat16:
      Launch<nv_bfloat16>(a, b, bias, y, m, n, k_blocks, stream);
      break;
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda
