// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_ggml_iq2_xxs.h"
#include "contrib_ops/cuda/quantization/iq2_xxs_decode.cuh"
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
  const int64_t k = k_blocks * 256;
  // One warp per output. Every lane decodes one eight-value grid vector per superblock.
  for (int64_t output = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
       output < m * n; output += static_cast<int64_t>(gridDim.x) * kWarps) {
    const int64_t row = output / n;
    const int64_t col = output % n;
    const float sum = iq2_xxs::Dot(a + row * k, b + col * k_blocks * 66, k_blocks);
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
