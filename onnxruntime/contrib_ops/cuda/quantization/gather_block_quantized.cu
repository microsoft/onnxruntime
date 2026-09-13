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

// Dequantizes a single FP8 element to float. Float8E4M3FN/FNUZ/Float8E5M2/FNUZ all have an
// ORT_HOST_DEVICE `operator float()`, so the generic template body works for all of them; only
// the packed FP4 type needs a specialization (below) to unpack the correct nibble.
template <typename T1>
__device__ inline float dequant_fp_elem(const T1* data, int64_t idx) {
  return static_cast<float>(data[idx]);
}

#if !defined(DISABLE_FLOAT4_TYPES)
template <>
__device__ inline float dequant_fp_elem<Float4E2M1x2>(const Float4E2M1x2* data, int64_t idx) {
  auto pair = data[idx >> 1].ToFloat2();
  return (idx & 1) ? pair.second : pair.first;
}
#endif  // !defined(DISABLE_FLOAT4_TYPES)

template <typename T1, typename T2, typename Tind>
__global__ void GatherBlockQuantizedFpKernel(
    const T1* data,  // FP8 or packed FP4 codes, one code per element (no zero point, symmetric)
    const Tind* indices,
    const T2* scales,  // one scale per block, or a single broadcast scale if scale_size == 1
    T2* output,
    int64_t after_gather_dim,
    int64_t gather_axis_dim,
    int64_t ind_dim,
    int64_t block_size,
    int64_t N,
    int64_t scale_size) {
  int64_t out_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (out_idx >= N) return;

  int64_t idx_before = out_idx / (after_gather_dim * ind_dim);
  int64_t idx_after = out_idx % after_gather_dim;
  int64_t idx = (out_idx % (after_gather_dim * ind_dim)) / after_gather_dim;
  int64_t idx_at_g = indices[idx];
  if (idx_at_g < -gather_axis_dim || idx_at_g >= gather_axis_dim) {
    output[out_idx] = static_cast<T2>(0);
    return;
  }
  if (idx_at_g < 0) {
    idx_at_g += gather_axis_dim;
  }
  int64_t in_idx = idx_before * gather_axis_dim * after_gather_dim + idx_at_g * after_gather_dim + idx_after;

  int64_t block_id = in_idx / block_size;
  int64_t scale_idx = (scale_size == 1) ? 0 : block_id;

  float dq = dequant_fp_elem(data, in_idx);
  output[out_idx] = static_cast<T2>(dq) * scales[scale_idx];
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
  // Keep invalid dynamic indices from participating in device address arithmetic.
  if (idx_at_g < -gather_axis_dim || idx_at_g >= gather_axis_dim) {
    output[out_idx] = static_cast<T2>(0);
    return;
  }
  if (idx_at_g < 0) {
    idx_at_g += gather_axis_dim;
  }
  int64_t in_idx = idx_before * gather_axis_dim * after_gather_dim + idx_at_g * after_gather_dim + idx_after;

  int64_t block_id = in_idx / block_size;

  // unpack zero_point for this block:
  int64_t offset = 0;
  if (zero_points) {
    offset = get_val(zero_points, block_id, bits, sign);
  } else if constexpr (std::is_same_v<T1, uint8_t>) {
    offset = int64_t{1} << (bits - 1);
  }

  // unpack the raw quantized code for this element:
  int64_t weight = get_val(data, in_idx, bits, sign);

  // apply dequantization:
  output[out_idx] = static_cast<T2>(weight - offset) * scales[block_id];
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

  if constexpr (IsFpQuantizedV<T1>) {
    GatherBlockQuantizedFpKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, param.stream>>>(
        data, indices, scales, output,
        param.after_gather_dim, param.gather_axis_dim, param.ind_dim, param.block_size, param.N, param.scale_size);
  } else {
    bool sign = std::is_same<T1, Int4x2>::value;
    GatherBlockQuantizedKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, param.stream>>>(data, indices, scales, zero_points, output,
                                                                                                param.after_gather_dim, param.gather_axis_dim, param.ind_dim, param.bits, param.block_size, param.gather_axis, param.N, sign);
  }
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

#if !defined(DISABLE_FLOAT8_TYPES)
#define INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_FP8(T1)                                                                                                               \
  template void LaunchGatherBlockQuantizedKernel<T1, float, int32_t>(const T1*, const int32_t*, const float*, const T1*, float*, GatherBlockQuantizedParam);          \
  template void LaunchGatherBlockQuantizedKernel<T1, float, int64_t>(const T1*, const int64_t*, const float*, const T1*, float*, GatherBlockQuantizedParam);          \
  template void LaunchGatherBlockQuantizedKernel<T1, half, int32_t>(const T1*, const int32_t*, const half*, const T1*, half*, GatherBlockQuantizedParam);             \
  template void LaunchGatherBlockQuantizedKernel<T1, half, int64_t>(const T1*, const int64_t*, const half*, const T1*, half*, GatherBlockQuantizedParam);             \
  template void LaunchGatherBlockQuantizedKernel<T1, BFloat16, int32_t>(const T1*, const int32_t*, const BFloat16*, const T1*, BFloat16*, GatherBlockQuantizedParam); \
  template void LaunchGatherBlockQuantizedKernel<T1, BFloat16, int64_t>(const T1*, const int64_t*, const BFloat16*, const T1*, BFloat16*, GatherBlockQuantizedParam);

INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_FP8(Float8E4M3FN);
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_FP8(Float8E4M3FNUZ);
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_FP8(Float8E5M2);
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_FP8(Float8E5M2FNUZ);
#undef INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_FP8
#endif  // !defined(DISABLE_FLOAT8_TYPES)

#if !defined(DISABLE_FLOAT4_TYPES)
template void LaunchGatherBlockQuantizedKernel<Float4E2M1x2, float, int32_t>(const Float4E2M1x2*, const int32_t*, const float*, const Float4E2M1x2*, float*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Float4E2M1x2, float, int64_t>(const Float4E2M1x2*, const int64_t*, const float*, const Float4E2M1x2*, float*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Float4E2M1x2, half, int32_t>(const Float4E2M1x2*, const int32_t*, const half*, const Float4E2M1x2*, half*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Float4E2M1x2, half, int64_t>(const Float4E2M1x2*, const int64_t*, const half*, const Float4E2M1x2*, half*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Float4E2M1x2, BFloat16, int32_t>(const Float4E2M1x2*, const int32_t*, const BFloat16*, const Float4E2M1x2*, BFloat16*, GatherBlockQuantizedParam);
template void LaunchGatherBlockQuantizedKernel<Float4E2M1x2, BFloat16, int64_t>(const Float4E2M1x2*, const int64_t*, const BFloat16*, const Float4E2M1x2*, BFloat16*, GatherBlockQuantizedParam);
#endif  // !defined(DISABLE_FLOAT4_TYPES)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
