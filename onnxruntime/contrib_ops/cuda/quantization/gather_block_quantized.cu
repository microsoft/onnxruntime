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
    const T2* scales,  // one scale per block, laid out per `scale_strides`/`scale_broadcast_axis`
    T2* output,
    int64_t after_gather_dim,
    int64_t gather_axis_dim,
    int64_t ind_dim,
    int64_t block_size,
    int64_t N,
    int32_t rank,
    int64_t quantize_axis,
    TArray<int64_t> data_dims,
    TArray<int64_t> scale_strides,
    TArray<int64_t> scale_broadcast_axis) {
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

  // Decompose in_idx (a flat row-major offset into a tensor shaped like `data`) into a per-axis
  // index, so the quantize axis's block boundary resets correctly at every row (i.e. even when
  // data_dims[quantize_axis] is not a multiple of block_size) and so scale broadcasting can be
  // applied independently on any other axis.
  int64_t scale_idx = 0;
  int64_t remaining = in_idx;
  for (int32_t i = rank - 1; i >= 0; --i) {
    int64_t dim = data_dims[i];
    int64_t axis_idx = remaining % dim;
    remaining /= dim;
    int64_t contrib = (i == quantize_axis) ? axis_idx / block_size : axis_idx;
    if (scale_broadcast_axis[i]) {
      contrib = 0;
    }
    scale_idx += contrib * scale_strides[i];
  }

  float dq = dequant_fp_elem(data, in_idx);
  output[out_idx] = static_cast<T2>(dq * static_cast<float>(scales[scale_idx]));
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
Status LaunchGatherBlockQuantizedKernel(const T1* data,
                                        const Tind* indices,
                                        const T2* scales,
                                        const T1* zero_points,
                                        T2* output,
                                        GatherBlockQuantizedParam param) {
  if (param.N == 0) {
    return Status::OK();
  }

  const int64_t blocks = (param.N - 1) / GridDim::maxThreadsPerBlock + 1;
  ORT_RETURN_IF_NOT(blocks <= param.max_blocks_per_grid,
                    "GatherBlockQuantized output is too large for a CUDA grid.");
  const int blocks_per_grid = static_cast<int>(blocks);

  if constexpr (IsFpQuantizedV<T1>) {
    GatherBlockQuantizedFpKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, param.stream>>>(
        data, indices, scales, output,
        param.after_gather_dim, param.gather_axis_dim, param.ind_dim, param.block_size, param.N,
        param.rank, param.quantize_axis, param.data_dims, param.scale_strides, param.scale_broadcast_axis);
  } else {
    bool sign = std::is_same<T1, Int4x2>::value;
    GatherBlockQuantizedKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, param.stream>>>(
        data, indices, scales, zero_points, output,
        param.after_gather_dim, param.gather_axis_dim, param.ind_dim, param.bits,
        param.block_size, param.gather_axis, param.N, sign);
  }

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED(T1, T2, Tind)     \
  template Status LaunchGatherBlockQuantizedKernel<T1, T2, Tind>( \
      const T1*, const Tind*, const T2*, const T1*, T2*, GatherBlockQuantizedParam);

#define INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(T1)        \
  INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED(T1, float, int32_t)    \
  INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED(T1, float, int64_t)    \
  INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED(T1, half, int32_t)     \
  INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED(T1, half, int64_t)     \
  INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED(T1, BFloat16, int32_t) \
  INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED(T1, BFloat16, int64_t)

INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(uint8_t)
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(UInt4x2)
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(Int4x2)

#if !defined(DISABLE_FLOAT8_TYPES)
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(Float8E4M3FN)
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(Float8E4M3FNUZ)
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(Float8E5M2)
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(Float8E5M2FNUZ)
#endif  // !defined(DISABLE_FLOAT8_TYPES)

#if !defined(DISABLE_FLOAT4_TYPES)
INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES(Float4E2M1x2)
#endif  // !defined(DISABLE_FLOAT4_TYPES)

#undef INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED_TYPES
#undef INSTANTIATE_LAUNCH_GATHERBLOCKQUANTIZED

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
