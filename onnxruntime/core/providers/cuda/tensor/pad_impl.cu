// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "pad_impl.h"

namespace onnxruntime {
namespace cuda {

// PadMode enum from core/providers/cpu/tensor/padbase.h, cannot use that header because of nvcc/onnxruntime incompatibility
enum class PadMode : int {
  Constant = 0,
  Reflect,
  Edge,
  Wrap
};

template <typename T, int pad_mode>
__global__ void _PadKernel(
    const size_t shape_rank,
    const TArray<int64_t> input_dims,
    const TArray<int64_t> input_strides,
    const TArray<int64_t> lower_pads,
    const TArray<int64_t> slice_starts,
    const TArray<int64_t> effective_input_dims,
    const T pad_value,
    const T* input_data,
    const TArray<fast_divmod> fdm_output_strides,
    T* output_data,
    const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;
  bool use_pad_value = false;
  for (int dim = 0; dim < shape_rank && !use_pad_value; ++dim) {
    int out_coord, r;
    fdm_output_strides[dim].divmod(output_index, out_coord, r);
    output_index = r;
    int in_coord = 0;
    if (out_coord < lower_pads[dim]) {
      switch ((PadMode)pad_mode) {
        case PadMode::Constant:
          use_pad_value = true;
          break;
        case PadMode::Edge:
          in_coord = 0;
          break;
        case PadMode::Reflect:
          in_coord = lower_pads[dim] - out_coord;
          break;
        case PadMode::Wrap: {
          int64_t eff_len = effective_input_dims[dim];
          if (eff_len == 0) {
            // Should not happen if output size > 0, but safe fallback
            in_coord = 0;
          } else {
            // Match CPU: effective_index = (eff_len - lower_pads[dim] + out_coord) % eff_len
            // then in_coord = -slice_starts[dim] + effective_index (map to input via effective region)
            int64_t effective_index = (eff_len - lower_pads[dim] + static_cast<int64_t>(out_coord)) % eff_len;
            if (effective_index < 0) effective_index += eff_len;
            in_coord = static_cast<int>(-slice_starts[dim] + effective_index);
          }
          break;
        }
      }
    } else if (out_coord >= lower_pads[dim] + input_dims[dim]) {
      switch ((PadMode)pad_mode) {
        case PadMode::Constant:
          use_pad_value = true;
          break;
        case PadMode::Edge:
          in_coord = input_dims[dim] - 1;
          break;
        case PadMode::Reflect:
          in_coord = input_dims[dim] - 2 - (out_coord - (lower_pads[dim] + input_dims[dim]));
          break;
        case PadMode::Wrap: {
          int64_t eff_len = effective_input_dims[dim];
          if (eff_len == 0) {
            in_coord = 0;
          } else {
            // Match CPU: effective_index = (eff_len - lower_pads[dim] + out_coord) % eff_len
            int64_t effective_index = (eff_len - lower_pads[dim] + static_cast<int64_t>(out_coord)) % eff_len;
            if (effective_index < 0) effective_index += eff_len;
            in_coord = static_cast<int>(-slice_starts[dim] + effective_index);
          }
          break;
        }
      }
    } else {
      in_coord = out_coord - lower_pads[dim];
    }
    input_index += input_strides[dim] * in_coord;
  }
  output_data[id] = use_pad_value ? (T)pad_value : input_data[input_index];
}

template <typename T, int pad_mode>
__global__ void _PadNCHWInputWithPaddingAlongHAndWKernel(
    const int64_t n,  // Batch
    const int64_t c,  // Channel
    const int64_t input_height,
    const int64_t output_height,
    const int64_t input_width,
    const int64_t output_width,
    const int64_t pad_height_start,
    const int64_t pad_width_start,
    const int64_t slice_height_start,
    const int64_t slice_width_start,
    const int64_t effective_input_height,
    const int64_t effective_input_width,
    const T pad_value,
    const T* input_data,
    T* output_data,
    const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  const int current_output_width = id % output_width;
  int nc_index = id / output_width;
  const int current_output_height = nc_index % output_height;
  nc_index /= output_height;

  int current_input_height = current_output_height - pad_height_start;
  int current_input_width = current_output_width - pad_width_start;

  switch ((PadMode)pad_mode) {
    case PadMode::Constant:
      output_data[id] = (current_input_height < 0 ||
                         current_input_width < 0 ||
                         current_input_height >= input_height ||
                         current_input_width >= input_width)
                            ? pad_value
                            : input_data[(nc_index * input_height +
                                          current_input_height) *
                                             input_width +
                                         current_input_width];
      break;

    case PadMode::Edge:
      current_input_height = std::max(0, std::min(current_input_height, static_cast<int>(input_height - 1)));
      current_input_width = std::max(0, std::min(current_input_width, static_cast<int>(input_width - 1)));
      output_data[id] = input_data[(nc_index * input_height +
                                    current_input_height) *
                                       input_width +
                                   current_input_width];
      break;

    case PadMode::Reflect:
      current_input_height = std::max(current_input_height, -current_input_height);
      current_input_height = std::min(static_cast<int>(current_input_height),
                                      2 * static_cast<int>(input_height) - current_input_height - 2);

      current_input_width = std::max(current_input_width, -current_input_width);
      current_input_width = std::min(static_cast<int>(current_input_width),
                                     2 * static_cast<int>(input_width) - current_input_width - 2);

      output_data[id] = input_data[(nc_index * input_height +
                                    current_input_height) *
                                       input_width +
                                   current_input_width];
      break;

    case PadMode::Wrap: {
      int64_t h_in, w_in;

      if (effective_input_height == 0) {
        h_in = 0;
      } else {
        // Match CPU: effective_index = (eff_len - lower_pads + out_coord) % eff_len
        int64_t effective_index_h = (effective_input_height - pad_height_start + current_output_height) % effective_input_height;
        if (effective_index_h < 0) effective_index_h += effective_input_height;
        h_in = -slice_height_start + effective_index_h;
      }

      if (effective_input_width == 0) {
        w_in = 0;
      } else {
        // Match CPU: effective_index = (eff_len - lower_pads + out_coord) % eff_len
        int64_t effective_index_w = (effective_input_width - pad_width_start + current_output_width) % effective_input_width;
        if (effective_index_w < 0) effective_index_w += effective_input_width;
        w_in = -slice_width_start + effective_index_w;
      }

      output_data[id] = input_data[(nc_index * input_height + static_cast<int>(h_in)) * input_width +
                                   static_cast<int>(w_in)];
      break;
    }
  }
}

template <typename T>
void PadImpl(
    cudaStream_t stream,
    const size_t shape_rank,
    const TArray<int64_t>& input_dims,
    const TArray<int64_t>& input_strides,
    const TArray<int64_t>& lower_pads,
    const TArray<int64_t>& slice_starts,
    const TArray<int64_t>& effective_input_dims,
    const T pad_value,
    const int pad_mode,
    const T* input_data,
    const TArray<fast_divmod>& fdm_output_strides,
    T* output_data,
    const size_t N) {
  if (N == 0)  // special case where there's a dim value of 0 in the output shape
    return;

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (pad_mode) {
    case 0:
      _PadKernel<T, 0><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_dims, input_strides, lower_pads, slice_starts, effective_input_dims,
          pad_value, input_data, fdm_output_strides, output_data, N);
      break;
    case 1:
      _PadKernel<T, 1><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_dims, input_strides, lower_pads, slice_starts, effective_input_dims,
          pad_value, input_data, fdm_output_strides, output_data, N);
      break;
    case 2:
      _PadKernel<T, 2><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_dims, input_strides, lower_pads, slice_starts, effective_input_dims,
          pad_value, input_data, fdm_output_strides, output_data, N);
      break;
    case 3:
      _PadKernel<T, 3><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_dims, input_strides, lower_pads, slice_starts, effective_input_dims,
          pad_value, input_data, fdm_output_strides, output_data, N);
      break;
  }
}

template <typename T>
void PadNCHWInputWithPaddingAlongHAndWImpl(
    cudaStream_t stream,
    const int64_t n,  // Batch
    const int64_t c,  // Channel
    const int64_t input_height,
    const int64_t output_height,
    const int64_t input_width,
    const int64_t output_width,
    const int64_t pad_height_start,
    const int64_t pad_width_start,
    const int64_t slice_height_start,
    const int64_t slice_width_start,
    const int64_t effective_input_height,
    const int64_t effective_input_width,
    const T pad_value,
    const int pad_mode,
    const T* input_data,
    T* output_data,
    const size_t N) {
  if (N == 0)  // special case where there's a dim value of 0 in the output shape
    return;

  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (pad_mode) {
    case 0:
      _PadNCHWInputWithPaddingAlongHAndWKernel<T, 0><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          n, c, input_height, output_height, input_width, output_width,
          pad_height_start, pad_width_start, slice_height_start, slice_width_start,
          effective_input_height, effective_input_width,
          pad_value, input_data, output_data, N);
      break;
    case 1:
      _PadNCHWInputWithPaddingAlongHAndWKernel<T, 1><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          n, c, input_height, output_height, input_width, output_width,
          pad_height_start, pad_width_start, slice_height_start, slice_width_start,
          effective_input_height, effective_input_width,
          pad_value, input_data, output_data, N);
      break;
    case 2:
      _PadNCHWInputWithPaddingAlongHAndWKernel<T, 2><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          n, c, input_height, output_height, input_width, output_width,
          pad_height_start, pad_width_start, slice_height_start, slice_width_start,
          effective_input_height, effective_input_width,
          pad_value, input_data, output_data, N);
      break;
    case 3:
      _PadNCHWInputWithPaddingAlongHAndWKernel<T, 3><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          n, c, input_height, output_height, input_width, output_width,
          pad_height_start, pad_width_start, slice_height_start, slice_width_start,
          effective_input_height, effective_input_width,
          pad_value, input_data, output_data, N);
      break;
  }
}

#define SPECIALIZED_IMPL(T)                                                                                       \
  template void PadImpl<T>(cudaStream_t stream, const size_t shape_rank,                                          \
                           const TArray<int64_t>& input_dims, const TArray<int64_t>& input_strides,               \
                           const TArray<int64_t>& lower_pads,                                                     \
                           const TArray<int64_t>& slice_starts,                                                   \
                           const TArray<int64_t>& effective_input_dims,                                           \
                           const T pad_value,                                                                     \
                           const int pad_mode,                                                                    \
                           const T* input_data,                                                                   \
                           const TArray<fast_divmod>& fdm_output_strides,                                         \
                           T* output_data,                                                                        \
                           const size_t N);                                                                       \
  template void PadNCHWInputWithPaddingAlongHAndWImpl<T>(cudaStream_t stream, const int64_t n, const int64_t c,   \
                                                         const int64_t input_height, const int64_t output_height, \
                                                         const int64_t input_width, const int64_t output_width,   \
                                                         const int64_t pad_height_start,                          \
                                                         const int64_t pad_width_start,                           \
                                                         const int64_t slice_height_start,                        \
                                                         const int64_t slice_width_start,                         \
                                                         const int64_t effective_input_height,                    \
                                                         const int64_t effective_input_width,                     \
                                                         const T pad_value,                                       \
                                                         const int pad_mode,                                      \
                                                         const T* input_data, T* output_data,                     \
                                                         const size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(bool)
}  // namespace cuda
}  // namespace onnxruntime
