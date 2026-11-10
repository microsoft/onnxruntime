// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "contrib_ops/cuda/quantization/matmul_4bits_batched_impl.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
bool TryMatMulUnalignedBatched4Bits(
    T* output, const T* a_data, const uint8_t* b_data_quant, const T* scales_data,
    int m, int n, int k, cudaStream_t stream) {
  static_assert(std::is_same_v<T, half>);
  if (m < 2 || m > 8 || n < 4 || n % 8 != 4 || k <= 0 || k % 32 != 0) {
    return false;
  }
  const int cta_m = m <= 2 ? 2 : m <= 4 ? 4
                                        : 8;
  dim3 threads(onnxruntime::cuda::GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  dim3 blocks(n / kColsPerThreadBlock + 1, (m + cta_m - 1) / cta_m);
  if (cta_m == 2) {
    MatMulFloat4BatchedKernel<T, 32, false, 2, 1, true><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, nullptr, m, n, k, k / 32);
  } else if (cta_m == 4) {
    MatMulFloat4BatchedKernel<T, 32, false, 4, 1, true><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, nullptr, m, n, k, k / 32);
  } else {
    MatMulFloat4BatchedKernel<T, 32, false, 8, 1, true><<<blocks, threads, 0, stream>>>(
        output, a_data, b_data_quant, scales_data, nullptr, m, n, k, k / 32);
  }
  return true;
}

template bool TryMatMulUnalignedBatched4Bits<half>(
    half*, const half*, const uint8_t*, const half*, int, int, int, cudaStream_t);

template bool TryMatMulBatched4Bits<half>(
    half*, const half*, const uint8_t*, const half*, const uint8_t*,
    int, int, int, int, size_t, cudaStream_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime