// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_ggml_q8_0.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

#include <algorithm>

namespace onnxruntime::contrib::cuda {
namespace {

template <typename T>
__global__ void MatMulQ8Kernel(const T* a, const uint8_t* b, const T* bias, T* y,
                               int64_t m, int64_t n, int64_t k_blocks) {
  constexpr int kWarps = 8;
  const int lane = threadIdx.x % 32;
  const int64_t k = k_blocks * 32;
  for (int64_t output = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
       output < m * n; output += static_cast<int64_t>(gridDim.x) * kWarps) {
    const int64_t row = output / n;
    const int64_t col = output % n;
    float sum = 0.0f;
    for (int64_t block = 0; block < k_blocks; ++block) {
      const uint8_t* packed = b + (col * k_blocks + block) * 34;
      const uint16_t scale_bits = static_cast<uint16_t>(packed[0]) |
                                  (static_cast<uint16_t>(packed[1]) << 8);
      const float d = __half2float(__ushort_as_half(scale_bits));
      const unsigned raw = packed[2 + lane];
      const int q = static_cast<int>(raw) - (raw >= 128 ? 256 : 0);
      // Preserve the FP16 scale; round q*d, not d, to the activation type.
      const float weight = to_float<T>(from_float<T>(static_cast<float>(q) * d));
      sum = fmaf(to_float<T>(a[row * k + block * 32 + lane]), weight, sum);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
      const T rounded = from_float<T>(sum);
      y[output] = bias != nullptr ? from_float<T>(to_float<T>(rounded) + to_float<T>(bias[col]))
                                  : rounded;
    }
  }
}

template <typename T>
void Launch(const void* a, const uint8_t* b, const void* bias, void* y,
            int64_t m, int64_t n, int64_t k_blocks, cudaStream_t stream) {
  const int blocks = static_cast<int>(std::min<int64_t>((m * n - 1) / 8 + 1, 65535));
  MatMulQ8Kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const T*>(a), b, static_cast<const T*>(bias), static_cast<T*>(y),
      m, n, k_blocks);
}

}  // namespace

Status LaunchMatMulGGMLQ8_0(const void* a, const uint8_t* b, const void* bias, void* y,
                            int64_t m, int64_t n, int64_t k_blocks,
                            Q8ActivationType type, cudaStream_t stream) {
  switch (type) {
    case Q8ActivationType::Float:
      Launch<float>(a, b, bias, y, m, n, k_blocks, stream);
      break;
    case Q8ActivationType::Float16:
      Launch<half>(a, b, bias, y, m, n, k_blocks, stream);
      break;
    case Q8ActivationType::BFloat16:
      Launch<nv_bfloat16>(a, b, bias, y, m, n, k_blocks, stream);
      break;
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda
