#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/tensor/resize_impl.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
__global__ void _ResizeNearestKernel(const size_t rank,
                                     const int64_t* input_pitches,
                                     const fast_divmod* output_div_pitches,
                                     const float* scales,
                                     const T* input_data,
                                     T* output_data,
                                     const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;

  int div, mod;
  for (int dim = 0; dim < rank; ++dim) {
    output_div_pitches[dim].divmod(output_index, div, mod);
    output_index = mod;
    if (scales[dim] <= 1) {  //downsample
      div = std::ceil(div / scales[dim]);
    } else {  //upsample
      div = div / scales[dim];
    }
    input_index += input_pitches[dim] * div;
  }
  output_data[id] = input_data[input_index];
}

// The following method supports a 4-D input in 'Linear mode'
// that amounts to 'Bilinear' Upsampling/Resizing in the sense that it assumes
// the scale values for the outermost 2 dimensions are 1.
// This is the common use-case where the 4-D input (batched multi-channel images)
// is usually of shape [N, C, H, W] and the scales are [1.0, 1.0, height_scale, width_scale]
template <typename T>
__global__ void _ResizeBilinear4DInputKernel(const int64_t input_dim2,
                                      const int64_t* input_pitches,
                                      const fast_divmod* output_div_pitches,
                                      const float* scales,
                                      const T* input_data,
                                      T* output_data,
                                      const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;

  // For bilinear mode, scales[0]=scales[1]=1
  int mod;
  int index_of_dim0, index_of_dim1, index_of_dim2, index_of_dim3;
  output_div_pitches[0].divmod(id, index_of_dim0, mod);
  output_div_pitches[1].divmod(mod, index_of_dim1, mod);
  output_div_pitches[2].divmod(mod, index_of_dim2, mod);
  index_of_dim3 = mod;
  int index_of_input_dim2, index_of_input_dim3;
  float x_offset_0, y_offset_0, x_offset_1, y_offset_1;
  index_of_input_dim2 = static_cast<int64_t>(index_of_dim2 / scales[2]);
  index_of_input_dim3 = static_cast<int64_t>(index_of_dim3 / scales[3]);
  input_index = index_of_dim0 * input_pitches[0] +
                index_of_dim1 * input_pitches[1] +
                index_of_input_dim2 * input_pitches[2] +
                index_of_input_dim3;

  T x00 = input_data[input_index];
  T x10, x01, x11;

  bool end_of_dim2 = false, end_of_dim3 = false;
  if (index_of_input_dim2 == (input_dim2 - 1)) {
    // It's the end in dimension 2
    x01 = x00;
    end_of_dim2 = true;
  } else {
    x01 = input_data[input_index + input_pitches[2]];
  }

  if (index_of_input_dim3 == (input_pitches[2] - 1)) {
    // It's the end in dimension 3
    x10 = x00;
    x11 = x01;
    end_of_dim3 = true;
  } else {
    x10 = input_data[input_index + 1];
    x11 = end_of_dim2 ? x10 : input_data[input_index + input_pitches[2] + 1];
  }

  y_offset_0 = end_of_dim2 ? 0.5f : index_of_dim2 / scales[2] - index_of_input_dim2;
  y_offset_1 = 1.0f - y_offset_0;
  x_offset_0 = end_of_dim3 ? 0.5f : index_of_dim3 / scales[3] - index_of_input_dim3;
  x_offset_1 = 1.0f - x_offset_0;

  output_data[id] =
      x00 * static_cast<T>(y_offset_1 * x_offset_1) +
      x01 * static_cast<T>(y_offset_0 * x_offset_1) +
      x10 * static_cast<T>(y_offset_1 * x_offset_0) +
      x11 * static_cast<T>(y_offset_0 * x_offset_0);
}

// The following method supports a 2-D input in 'Linear mode'
template <typename T>
__global__ void _ResizeBilinear2DInputKernel(const int64_t input_dim0,
                                             const int64_t* input_pitches,
                                             const fast_divmod* output_div_pitches,
                                             const float* scales,
                                             const T* input_data,
                                             T* output_data,
                                             const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;

  int mod;
  int index_of_dim0, index_of_dim1;
  output_div_pitches[0].divmod(id, index_of_dim0, mod);
  index_of_dim1 = mod;
  int index_of_input_dim0, index_of_input_dim1;
  float x_offset_0, y_offset_0, x_offset_1, y_offset_1;
  index_of_input_dim0 = static_cast<int64_t>(index_of_dim0 / scales[0]);
  index_of_input_dim1 = static_cast<int64_t>(index_of_dim1 / scales[1]);
  input_index = index_of_input_dim0 * input_pitches[0] + index_of_input_dim1;

  T x00 = input_data[input_index];
  T x10, x01, x11;

  bool end_of_dim0 = false, end_of_dim1 = false;
  if (index_of_input_dim0 == (input_dim0 - 1)) {
    // It's the end in dimension 0
    x01 = x00;
    end_of_dim0 = true;
  } else {
    x01 = input_data[input_index + input_pitches[0]];
  }

  if (index_of_input_dim1 == (input_pitches[0] - 1)) {
    // It's the end in dimension 1
    x10 = x00;
    x11 = x01;
    end_of_dim1 = true;
  } else {
    x10 = input_data[input_index + 1];
    x11 = end_of_dim0 ? x10 : input_data[input_index + input_pitches[0] + 1];
  }

  y_offset_0 = end_of_dim0 ? 0.5f : index_of_dim0 / scales[0] - index_of_input_dim0;
  y_offset_1 = 1.0f - y_offset_0;
  x_offset_0 = end_of_dim1 ? 0.5f : index_of_dim1 / scales[1] - index_of_input_dim1;
  x_offset_1 = 1.0f - x_offset_0;

  output_data[id] =
      x00 * static_cast<T>(y_offset_1 * x_offset_1) +
      x01 * static_cast<T>(y_offset_0 * x_offset_1) +
      x10 * static_cast<T>(y_offset_1 * x_offset_0) +
      x11 * static_cast<T>(y_offset_0 * x_offset_0);
}

template <typename T>
void ResizeImpl(const onnxruntime::UpsampleMode upsample_mode,
                const size_t rank,
                const int64_t input_dim2,
                const int64_t* input_pitches,
                const fast_divmod* output_div_pitches,
                const float* scales_vals,
                const T* input_data,
                T* output_data,
                const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  if (onnxruntime::UpsampleMode::NN == upsample_mode) {
    _ResizeNearestKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        rank, input_pitches, output_div_pitches, scales_vals,
        input_data, output_data, N);
  } else if (onnxruntime::UpsampleMode::LINEAR == upsample_mode && rank == 4) {
    _ResizeBilinear4DInputKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        input_dim2, input_pitches, output_div_pitches, scales_vals,
        input_data, output_data, N);
  } else if (onnxruntime::UpsampleMode::LINEAR == upsample_mode && rank == 2) {
    _ResizeBilinear2DInputKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
        input_dim2, input_pitches, output_div_pitches, scales_vals,
        input_data, output_data, N);
  }
}

#define SPECIALIZED_IMPL(T)                                                  \
  template void ResizeImpl<T>(const onnxruntime::UpsampleMode upsample_mode, \
                              const size_t rank,                             \
                              const int64_t input_dim2,                      \
                              const int64_t* input_pitches,                  \
                              const fast_divmod* output_div_pitches,         \
                              const float* scales_vals,                      \
                              const T* input_data,                           \
                              T* output_data,                                \
                              const size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(uint8_t)

}  // namespace cuda
}  // namespace onnxruntime
