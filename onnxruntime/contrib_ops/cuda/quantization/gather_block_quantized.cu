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
__device__ inline int64_t get_val(const T1* data, int64_t idx, int64_t bits, bool sign) {
  const uint32_t mask = (1U << bits) - 1;
  const int64_t elems_per_byte = 8 / bits;
  const int64_t byte_idx = idx / elems_per_byte;
  const int64_t bit_offset = (idx % elems_per_byte) * bits;
  const uint8_t byte = reinterpret_cast<const uint8_t*>(data)[byte_idx];
  int64_t val = (byte >> bit_offset) & mask;

  // Sign-extend based on bit width
  if (sign) {
    if (val & (1 << (bits - 1))) {
      val |= -1LL << bits;
    }
  }

  return val;
}

template <typename T1, typename T2, typename Tind>
__global__ void GatherBlockQuantizedKernel(
    const T1* data,  // packed 4-bit codes, one code per element
    const Tind* indices,
    const T2* scales,       // one float scale per block
    const T1* zero_points,  // packed 4-bit zero-points, one per block
    T2* output,
    int64_t after_gather_dim,
    int64_t gather_axis_dim,
    int64_t ind_dim,
    int64_t bits,
    int64_t block_size,
    int64_t gather_axis,
    int64_t N,
    bool sign) {
  int64_t out_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (out_idx >= N) return;

  // compute which input element this thread corresponds to:
  int64_t idx_before = out_idx / (after_gather_dim * ind_dim);
  int64_t idx_after = out_idx % after_gather_dim;
  int64_t idx = (out_idx % (after_gather_dim * ind_dim)) / after_gather_dim;
  int64_t idx_at_g = indices[idx];
  int64_t in_idx = idx_before * gather_axis_dim * after_gather_dim + idx_at_g * after_gather_dim + idx_after;

  int64_t block_id = in_idx / block_size;

  // unpack zero_point for this block:
  int64_t offset = 0;
  if (zero_points) {
    offset = get_val(zero_points, block_id, bits, sign);
  }

  // unpack the raw quantized code for this element:
  int64_t weight = get_val(data, in_idx, bits, sign);

  // apply dequantization:
  output[out_idx] = static_cast<T2>((weight - offset) * scales[block_id]);
}

template <typename T1, typename T2, typename Tind>
void LaunchGatherBlockQuantizedKernel(const T1* data,
                                      const Tind* indices,
                                      const T2* scales,
                                      const T1* zero_points,
                                      T2* output,
                                      bool sign,
                                      GatherBlockQuantizedParam param) {
  // Require quant_axis is last dim
  int blocksPerGrid = (int)(ceil(static_cast<float>(param.N) / GridDim::maxThreadsPerBlock));

  GatherBlockQuantizedKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, param.stream>>>(data, indices, scales, zero_points, output,
                                                                                              param.after_gather_dim, param.gather_axis_dim, param.ind_dim, param.bits, param.block_size, param.gather_axis, param.N, sign);
}

template void LaunchGatherBlockQuantizedKernel<uint8_t, float, int32_t>(const uint8_t*, const int32_t*, const float*, const uint8_t*, float*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<uint8_t, float, int64_t>(const uint8_t*, const int64_t*, const float*, const uint8_t*, float*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, float, int32_t>(const UInt4x2*, const int32_t*, const float*, const UInt4x2*, float*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, float, int64_t>(const UInt4x2*, const int64_t*, const float*, const UInt4x2*, float*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, float, int32_t>(const Int4x2*, const int32_t*, const float*, const Int4x2*, float*, true, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, float, int64_t>(const Int4x2*, const int64_t*, const float*, const Int4x2*, float*, true, GatherBlockQuantizedParam);

template void LaunchGatherBlockQuantizedKernel<uint8_t, half, int32_t>(const uint8_t*, const int32_t*, const half*, const uint8_t*, half*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<uint8_t, half, int64_t>(const uint8_t*, const int64_t*, const half*, const uint8_t*, half*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, half, int32_t>(const UInt4x2*, const int32_t*, const half*, const UInt4x2*, half*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, half, int64_t>(const UInt4x2*, const int64_t*, const half*, const UInt4x2*, half*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, half, int32_t>(const Int4x2*, const int32_t*, const half*, const Int4x2*, half*, true, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, half, int64_t>(const Int4x2*, const int64_t*, const half*, const Int4x2*, half*, true, GatherBlockQuantizedParam);

template void LaunchGatherBlockQuantizedKernel<uint8_t, BFloat16, int32_t>(const uint8_t*, const int32_t*, const BFloat16*, const uint8_t*, BFloat16*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<uint8_t, BFloat16, int64_t>(const uint8_t*, const int64_t*, const BFloat16*, const uint8_t*, BFloat16*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, BFloat16, int32_t>(const UInt4x2*, const int32_t*, const BFloat16*, const UInt4x2*, BFloat16*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<UInt4x2, BFloat16, int64_t>(const UInt4x2*, const int64_t*, const BFloat16*, const UInt4x2*, BFloat16*, false, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, BFloat16, int32_t>(const Int4x2*, const int32_t*, const BFloat16*, const Int4x2*, BFloat16*, true, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Int4x2, BFloat16, int64_t>(const Int4x2*, const int64_t*, const BFloat16*, const Int4x2*, BFloat16*, true, GatherBlockQuantizedParam);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
