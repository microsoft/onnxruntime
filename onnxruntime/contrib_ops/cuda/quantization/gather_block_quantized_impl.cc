// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_block_quantized_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename Tind>
__device__ __forceinline__ int64_t GetIndexValue(const Tind* indices_data, int64_t offset) {
  return static_cast<int64_t>(indices_data[offset]);
}

template <typename T2, typename Tind, int bits>
__global__ void GatherBlockQuantizedKernel(
    T2* output,
    const uint8_t* data,
    const T2* scales,
    const uint8_t* zero_points,
    const Tind* indices,
    const int64_t block_size,
    const int64_t gather_M,
    const int64_t gather_N,
    const int64_t gather_block_unpacked,
    const int64_t gather_axis_dim,
    const int64_t quantize_axis_dim_unpacked,
    const int64_t quantize_N,
    const int64_t scale_stride_quantize_M,
    const int64_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(output_flat_idx, N);

  constexpr int components = 8 / bits;
  const int64_t data_gather_full_block_unpacked = gather_axis_dim * gather_block_unpacked;

  const int64_t block_idx = output_flat_idx / gather_block_unpacked;
  const int64_t elem_in_block_idx = output_flat_idx % gather_block_unpacked;

  const int64_t gather_M_idx = block_idx / gather_N;
  const int64_t gather_N_idx = block_idx % gather_N;

  int64_t indices_val = GetIndexValue(indices, gather_N_idx);
  if (indices_val < 0) {
    indices_val += gather_axis_dim;
  }

  const int64_t unpacked_data_flat_idx = gather_M_idx * data_gather_full_block_unpacked + indices_val * gather_block_unpacked + elem_in_block_idx;

  // --- Scale and Zero Point Index Calculation ---
  const int64_t quantize_full_block_unpacked = quantize_axis_dim_unpacked * quantize_N;

  const int64_t quantize_M_idx = unpacked_data_flat_idx / quantize_full_block_unpacked;
  const int64_t quantize_axis_coord_unpacked = (unpacked_data_flat_idx % quantize_full_block_unpacked) / quantize_N;
  const int64_t quantize_N_idx = unpacked_data_flat_idx % quantize_N;

  const int64_t scale_quantize_axis_coord = quantize_axis_coord_unpacked / block_size;
  const int64_t scale_flat_idx = quantize_M_idx * scale_stride_quantize_M + scale_quantize_axis_coord * quantize_N + quantize_N_idx;

  // --- Read Quantized Data ---
  int32_t q_val;
  if constexpr (bits == 8) {
    q_val = static_cast<int32_t>(data[unpacked_data_flat_idx]);
  } else {  // bits == 4
    const uint8_t packed_byte = data[unpacked_data_flat_idx / 2];
    q_val = (unpacked_data_flat_idx & 1) ? (packed_byte >> 4) : (packed_byte & 0x0F);
  }

  // --- Read Scale and Zero Point ---
  const T2 scale = scales[scale_flat_idx];
  int32_t zp_val;

  if (zero_points) {
    if constexpr (bits == 8) {
      zp_val = static_cast<int32_t>(zero_points[scale_flat_idx]);
    } else {  // bits == 4
      const uint8_t packed_zp = zero_points[scale_flat_idx / 2];
      zp_val = (scale_flat_idx & 1) ? (packed_zp >> 4) : (packed_zp & 0x0F);
    }
  } else {
    zp_val = (bits == 4) ? 8 : 128;
  }

  // --- Dequantize and Write Output ---
  float result = (static_cast<float>(q_val - zp_val)) * scale;
  output[output_flat_idx] = T2(result);
}

template <typename T2, typename Tind, int bits>
Status LaunchGatherBlockQuantizedKernel(
    cudaStream_t stream,
    T2* output,
    const uint8_t* data,
    const T2* scales,
    const uint8_t* zero_points,
    const Tind* indices,
    const int64_t block_size,
    const int64_t gather_M,
    const int64_t gather_N,
    const int64_t gather_block_unpacked,
    const int64_t gather_axis_dim,
    const int64_t quantize_axis_dim_unpacked,
    const int64_t quantize_N,
    const int64_t scale_stride_quantize_M,
    const int64_t N) {
  auto& device_prop = GetDeviceProp();
  constexpr int threads_per_block = 256;
  const int blocks = static_cast<int>((N + threads_per_block - 1) / threads_per_block);

  GatherBlockQuantizedKernel<T2, Tind, bits><<<blocks, threads_per_block, 0, stream>>>(
      output, data, scales, zero_points, indices, block_size,
      gather_M, gather_N, gather_block_unpacked, gather_axis_dim,
      quantize_axis_dim_unpacked, quantize_N, scale_stride_quantize_M, N);

  return CUDA_CALL(cudaGetLastError());
}

template <typename T2, typename Tind>
Status GatherBlockQuantizedImpl(
    cudaStream_t stream,
    const CudaKernel& kernel,
    const Tensor* data,
    const Tensor* indices,
    const Tensor* scales,
    const Tensor* zero_points,
    Tensor& output,
    const int64_t gather_axis,
    const int64_t quantize_axis,
    const int64_t block_size,
    const int64_t bits) {
  const auto& data_shape = data->Shape();
  const auto& indices_shape = indices->Shape();
  const auto& scales_shape = scales->Shape();

  const int64_t components = (bits == 4) ? 2 : 1;

  // Calculate strides and sizes for the flattened view
  const int64_t gather_M = data_shape.SizeToDimension(gather_axis);
  const int64_t gather_N = indices_shape.Size();
  const int64_t gather_axis_dim = data_shape[gather_axis];

  // Calculate gather_block_unpacked by considering if the quantized axis is part of the block
  int64_t gather_block_unpacked = 1;
  if (data_shape.NumDimensions() > gather_axis + 1) {
      if (quantize_axis > gather_axis) {
          // Manually calculate size to correctly apply components
          gather_block_unpacked = data_shape.GetDims()[quantize_axis] * components;
          for(size_t i = quantize_axis + 1; i < data_shape.NumDimensions(); ++i) {
              gather_block_unpacked *= data_shape.GetDims()[i];
          }
          for(size_t i = gather_axis + 1; i < quantize_axis; ++i) {
              gather_block_unpacked *= data_shape.GetDims()[i];
          }
      } else { // quantize_axis is not in the gather block
          gather_block_unpacked = data_shape.SizeFromDimension(gather_axis + 1);
      }
  }

  const int64_t quantize_N = data_shape.SizeFromDimension(quantize_axis + 1);
  const int64_t quantize_axis_dim_unpacked = data_shape[quantize_axis] * components;
  const int64_t scale_quantize_axis_dim = scales_shape[quantize_axis];
  const int64_t scale_stride_quantize_M = scale_quantize_axis_dim * quantize_N;

  const int64_t N = output.Shape().Size();

  T2* output_ptr = output.template MutableData<T2>();
  const uint8_t* data_ptr = data->template Data<uint8_t>();
  const T2* scales_ptr = scales->template Data<T2>();
  const uint8_t* zp_ptr = zero_points ? zero_points->template Data<uint8_t>() : nullptr;
  const Tind* indices_ptr = indices->template Data<Tind>();

  if (bits == 4) {
    return LaunchGatherBlockQuantizedKernel<T2, Tind, 4>(
        stream, output_ptr, data_ptr, scales_ptr, zp_ptr, indices_ptr, block_size,
        gather_M, gather_N, gather_block_unpacked, gather_axis_dim,
        quantize_axis_dim_unpacked, quantize_N, scale_stride_quantize_M, N);
  } else {  // bits == 8
    return LaunchGatherBlockQuantizedKernel<T2, Tind, 8>(
        stream, output_ptr, data_ptr, scales_ptr, zp_ptr, indices_ptr, block_size,
        gather_M, gather_N, gather_block_unpacked, gather_axis_dim,
        quantize_axis_dim_unpacked, quantize_N, scale_stride_quantize_M, N);
  }
}

// Explicit template instantiations
template Status GatherBlockQuantizedImpl<MLFloat16, int32_t>(cudaStream_t, const CudaKernel&, const Tensor*, const Tensor*, const Tensor*, const Tensor*, Tensor&, int64_t, int64_t, int64_t, int64_t);
template Status GatherBlockQuantizedImpl<MLFloat16, int64_t>(cudaStream_t, const CudaKernel&, const Tensor*, const Tensor*, const Tensor*, const Tensor*, Tensor&, int64_t, int64_t, int64_t, int64_t);
template Status GatherBlockQuantizedImpl<BFloat16, int32_t>(cudaStream_t, const CudaKernel&, const Tensor*, const Tensor*, const Tensor*, const Tensor*, Tensor&, int64_t, int64_t, int64_t, int64_t);
template Status GatherBlockQuantizedImpl<BFloat16, int64_t>(cudaStream_t, const CudaKernel&, const Tensor*, const Tensor*, const Tensor*, const Tensor*, Tensor&, int64_t, int64_t, int64_t, int64_t);

}  // namespace cuda
}  // namespace onnxruntime
