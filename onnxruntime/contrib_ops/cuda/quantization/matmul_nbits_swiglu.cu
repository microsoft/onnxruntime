// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_nbits_swiglu.h"
#include "contrib_ops/cuda/quantization/matmul_2bits_batched_impl.cuh"
#include "core/providers/cuda/cuda_common.h"

#include <limits>

namespace onnxruntime::contrib::cuda {
namespace {

__global__ void MatMulNBitsSwiGLUEpilogue(const half* projected, half* output, int64_t count, int n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int64_t source = (i / (n / 2)) * n + i % (n / 2);
  output[i] = MatMul2BitsSwiGLU(projected[source], projected[source + n / 2]);
}

}  // namespace

Status LaunchMatMulNBitsSwiGLU(cudaStream_t stream, const void* x, const uint8_t* weight,
                               const void* scales, void* output, int m, int n, int k) {
  dim3 threads(kWarpSize2b, kColsPerThreadBlock2b);
  dim3 blocks(n / (2 * kColsPerThreadBlock2b));
#define LAUNCH_SWIGLU(rows, unroll)                                                             \
  MatMulFloat2bKernelBatched<half, 128, false, rows, 2, unroll, true>                           \
      <<<blocks, threads, 0, stream>>>(static_cast<half*>(output), static_cast<const half*>(x), \
                                       weight, static_cast<const half*>(scales), nullptr,       \
                                       m, n, k, k / 128)
  if (m == 1) {
    LAUNCH_SWIGLU(1, 1);
  } else if (m <= 2) {
    LAUNCH_SWIGLU(2, 2);
  } else if (m <= 4) {
    LAUNCH_SWIGLU(4, 2);
  } else {
    LAUNCH_SWIGLU(8, 2);
  }
#undef LAUNCH_SWIGLU
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchMatMulNBitsSwiGLUEpilogue(cudaStream_t stream, const void* projected,
                                       void* output, int m, int n) {
  const int64_t count = static_cast<int64_t>(m) * (n / 2);
  ORT_RETURN_IF_NOT((count + 255) / 256 <= std::numeric_limits<int>::max(),
                    "SwiGLU epilogue requires too many CUDA blocks");
  const int blocks = static_cast<int>((count + 255) / 256);
  MatMulNBitsSwiGLUEpilogue<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projected), static_cast<half*>(output), count, n);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
