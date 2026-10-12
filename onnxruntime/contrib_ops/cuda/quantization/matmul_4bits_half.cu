// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "contrib_ops/cuda/quantization/matmul_4bits_batched.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_m1_impl.cuh"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
bool TryMatMulUnaligned4Bits(
    T* output, const T* a_data, const uint8_t* b_data_quant, const T* scales_data,
    int m, int n, int k, size_t shared_mem_per_block, cudaStream_t stream) {
  static_assert(std::is_same_v<T, half>);
  if (m < 1 || m > 8 || n < 4 || n % 8 != 4 || k <= 0 || k % 32 != 0) {
    return false;
  }
  if (m > 1) {
    return TryMatMulUnalignedBatched4Bits<T>(output, a_data, b_data_quant, scales_data, m, n, k, stream);
  }
  const size_t shared_mem_size = sizeof(T) * (k / 32) * kColsPerThreadBlock;
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }
  dim3 blocks(n / kColsPerThreadBlock + 1, 1);
  dim3 threads(onnxruntime::cuda::GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  MatMulFloat4BitsKernelM1<T, 32, false, true><<<blocks, threads, shared_mem_size, stream>>>(
      output, a_data, b_data_quant, scales_data, nullptr, 1, n, k, k / 32);
  return true;
}

template bool TryMatMulUnaligned4Bits<half>(
    half*, const half*, const uint8_t*, const half*, int, int, int, size_t, cudaStream_t);

template bool TryMatMul4BitsM1<half>(
    half*, const half*, const uint8_t*, const half*, const uint8_t*, int, int, int, size_t, cudaStream_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
