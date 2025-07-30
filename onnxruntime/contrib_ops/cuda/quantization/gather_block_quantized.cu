// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_block_quantized.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T1>
inline __device__ int64_t get_val(const T1* data, int64_t idx, int64_t bits) {
  const uint32_t mask = (1U << bits) - 1;

  const int64_t elems_per_byte = 8 / bits;
  const int64_t byte_idx = idx / elems_per_byte;
  const int64_t bit_offset = (idx % elems_per_byte) * bits;

  const uint8_t byte = reinterpret_cast<const uint8_t*>(data)[byte_idx];
  return (byte >> bit_offset) & mask;
}

template <typename T1, typename T2, typename Tind>
__global__ void GatherBlockQuantizedKernel(
    const T1* data,
    const Tind* indices,
    const T2* scales,
    const T1* zero_points,
    T2* output,
    int64_t after_gather_dim,
    int64_t ind_dim,
    int64_t bits,
    int64_t block_size,
    int64_t gather_axis,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(out_idx, N);

  const int64_t idx_bg = out_idx / (after_gather_dim * ind_dim);
  const int64_t idx_ag = out_idx % after_gather_dim;
  const int64_t idx_axis = (out_idx % (after_gather_dim * ind_dim)) / after_gather_dim;

  const int64_t in_idx = idx_bg * after_gather_dim + idx_axis * after_gather_dim + idx_ag;

  int64_t offset = 0;
  if (zero_points != nullptr) {
    offset = get_val(zero_points, in_idx, bits);
  }

  const int64_t weight = get_val(data, in_idx / block_size, bits);
  output[out_idx] = static_cast<T2>(weight - offset) * scales[in_idx / block_size];
}

template <typename T1, typename T2, typename Tind>
void LaunchGatherBlockQuantizedKernel(const T1* data,
                                      const Tind* indices,
                                      const T2* scales,
                                      const T1* zero_points,
                                      T2* output,
                                      GatherBlockQuantizedParam param) {
  // Require quant_axis is last dim
  int blocksPerGrid = (int)(ceil(static_cast<float>(param.N) / GridDim::maxThreadsPerBlock));

  GatherBlockQuantizedKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, param.stream>>>(data, indices, scales, zero_points, output,
                                                                                              param.after_gather_dim, param.ind_dim, param.bits, param.block_size, param.gather_axis, param.N);
}

template void LaunchGatherBlockQuantizedKernel<uint8_t, float, int32_t>(const uint8_t*, const int32_t*, const float*, const uint8_t*, float*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<uint8_t, float, int64_t>(const uint8_t*, const int64_t*, const float*, const uint8_t*, float*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, float, int32_t>(const UInt4x2*, const int32_t*, const float*, const UInt4x2*, float*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, float, int64_t>(const UInt4x2*, const int64_t*, const float*, const UInt4x2*, float*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, float, int32_t>(const Int4x2*, const int32_t*, const float*, const Int4x2*, float*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, float, int64_t>(const Int4x2*, const int64_t*, const float*, const Int4x2*, float*, GatherBlockQuantizedParam);

template void LaunchGatherBlockQuantizedKernel<uint8_t, half, int32_t>(const uint8_t*, const int32_t*, const half*, const uint8_t*, half*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<uint8_t, half, int64_t>(const uint8_t*, const int64_t*, const half*, const uint8_t*, half*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, half, int32_t>(const UInt4x2*, const int32_t*, const half*, const UInt4x2*, half*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, half, int64_t>(const UInt4x2*, const int64_t*, const half*, const UInt4x2*, half*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, half, int32_t>(const Int4x2*, const int32_t*, const half*, const Int4x2*, half*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, half, int64_t>(const Int4x2*, const int64_t*, const half*, const Int4x2*, half*, GatherBlockQuantizedParam);

template void LaunchGatherBlockQuantizedKernel<uint8_t, BFloat16, int32_t>(const uint8_t*, const int32_t*, const BFloat16*, const uint8_t*, BFloat16*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<uint8_t, BFloat16, int64_t>(const uint8_t*, const int64_t*, const BFloat16*, const uint8_t*, BFloat16*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, BFloat16, int32_t>(const UInt4x2*, const int32_t*, const BFloat16*, const UInt4x2*, BFloat16*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, BFloat16, int64_t>(const UInt4x2*, const int64_t*, const BFloat16*, const UInt4x2*, BFloat16*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, BFloat16, int32_t>(const Int4x2*, const int32_t*, const BFloat16*, const Int4x2*, BFloat16*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, BFloat16, int64_t>(const Int4x2*, const int64_t*, const BFloat16*, const Int4x2*, BFloat16*, GatherBlockQuantizedParam);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
