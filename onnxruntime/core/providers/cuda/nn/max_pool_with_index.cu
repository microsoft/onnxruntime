// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "max_pool_with_index.h"

#include <cfloat>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
__global__ void MaxPoolWithIndexKernel(
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t depth,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t pooled_depth,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t kernel_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t stride_d,
    int64_t pad_h,
    int64_t pad_w,
    int64_t pad_d,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t dilation_d,
    fast_divmod fdm_c,
    fast_divmod fdm_h,
    fast_divmod fdm_w,
    fast_divmod fdm_d,
    int64_t storage_order,
    const T* p_input,
    int64_t output_size,
    T* p_output,
    int64_t* p_indices) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= output_size) return;

  int d_index, w_index, h_index, c_index, n_index, id_tmp;
  fdm_d.divmod(id, id_tmp, d_index);
  fdm_w.divmod(id_tmp, id_tmp, w_index);
  fdm_h.divmod(id_tmp, id_tmp, h_index);
  fdm_c.divmod(id_tmp, n_index, c_index);

  int64_t d_start = d_index * stride_d - pad_d;
  int64_t w_start = w_index * stride_w - pad_w;
  int64_t h_start = h_index * stride_h - pad_h;

  int64_t d_end = _Min<int64_t>(d_start + (kernel_d - 1) * dilation_d + 1, depth);
  int64_t w_end = _Min<int64_t>(w_start + (kernel_w - 1) * dilation_w + 1, width);
  int64_t h_end = _Min<int64_t>(h_start + (kernel_h - 1) * dilation_h + 1, height);

  d_start = _Max<int64_t>(d_start, 0);
  w_start = _Max<int64_t>(w_start, 0);
  h_start = _Max<int64_t>(h_start, 0);
  int64_t d_index_max = -1;
  int64_t w_index_max = -1;
  int64_t h_index_max = -1;
  int64_t offset = (n_index * channels + c_index) * height * width * depth;
  const T* p_slice = p_input + offset;
  T maxval = p_slice[h_start * width * depth + w_start * depth + d_start] - (T)1;
  for (int64_t d = d_start; d < d_end; d += dilation_d) {
    for (int64_t w = w_start; w < w_end; w += dilation_w) {
      for (int64_t h = h_start; h < h_end; h += dilation_h) {
        if (p_slice[h * width * depth + w * depth + d] > maxval) {
          h_index_max = h;
          w_index_max = w;
          d_index_max = d;
          maxval = static_cast<float>(p_slice[h * width * depth + w * depth + d]);
        }
      }
    }
  }
  p_output[id] = p_input[offset + h_index_max * width * depth + w_index_max * depth + d_index_max];
  if (p_indices) {
    p_indices[id] = storage_order == 0 ? offset + h_index_max * width * depth + w_index_max * depth + d_index_max
                                       : offset + h_index_max + w_index_max * height + d_index_max * width * height;
  }
}

template <typename T>
void MaxPoolWithIndex(
    cudaStream_t stream,
    const TensorShape& input_shape,
    const TensorShape& output_shape,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& stride_shape,
    const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations,
    int64_t storage_order,
    const T* p_input,
    T* p_output,
    int64_t* p_indices) {

  int64_t batchs = input_shape[0];
  int64_t channels = input_shape[1];
  int64_t height = input_shape[2];
  int64_t width = kernel_shape.size() > 1 ? input_shape[3] : 1;
  int64_t depth = kernel_shape.size() > 2 ? input_shape[4] : 1;
  int64_t pooled_height = output_shape[2];
  int64_t pooled_width = kernel_shape.size() > 1 ? output_shape[3] : 1;
  int64_t pooled_depth = kernel_shape.size() > 2 ? output_shape[4] : 1;
  int64_t kernel_h = kernel_shape[0];
  int64_t kernel_w = kernel_shape.size() > 1 ? kernel_shape[1] : 1;
  int64_t kernel_d = kernel_shape.size() > 2 ? kernel_shape[2] : 1;
  int64_t stride_h = stride_shape[0];
  int64_t stride_w = stride_shape.size() > 1 ? stride_shape[1] : 1;
  int64_t stride_d = stride_shape.size() > 2 ? stride_shape[2] : 1;
  //pads in the format of [x1_begin, x2_begin...x1_end, x2_end,...],
  //where xi_begin the number of pixels added at the beginning of axis i
  //and xi_end, the number of pixels added at the end of axis i.
  int64_t pad_h = pads[0];
  int64_t pad_w = pads.size() >= 4 ? pads[1] : 0;
  int64_t pad_d = pads.size() == 6 ? pads[2] : 0;
  int64_t dilation_h = dilations[0];
  int64_t dilation_w = dilations.size() >= 2 ? dilations[1] : 1;
  int64_t dilation_d = dilations.size() == 3 ? dilations[2] : 1;
  int64_t output_size = output_shape.Size();

  fast_divmod fdm_c(static_cast<int>(channels));
  fast_divmod fdm_h(static_cast<int>(pooled_height));
  fast_divmod fdm_w(static_cast<int>(pooled_width));
  fast_divmod fdm_d(static_cast<int>(pooled_depth));

  int blocksPerGrid = (int)((output_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  MaxPoolWithIndexKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      batchs,
      channels,
      height,
      width,
      depth,
      pooled_height,
      pooled_width,
      pooled_depth,
      kernel_h,
      kernel_w,
      kernel_d,
      stride_h,
      stride_w,
      stride_d,
      pad_h,
      pad_w,
      pad_d,
      dilation_h,
      dilation_w,
      dilation_d,
      fdm_c,
      fdm_h,
      fdm_w,
      fdm_d,
      storage_order,
      p_input,
      output_size,
      p_output,
      p_indices);
}

#define INSTANTIATEMAXPOOLWITHINDEX(T)          \
  template void MaxPoolWithIndex<T>(            \
      cudaStream_t stream,                \
      const TensorShape& input_shape,           \
      const TensorShape& output_shape,          \
      const std::vector<int64_t>& kernel_shape, \
      const std::vector<int64_t>& stride_shape, \
      const std::vector<int64_t>& pads,         \
      const std::vector<int64_t>& dilations,    \
      int64_t storage_order,                    \
      const T* p_input,                         \
      T* p_output,                              \
      int64_t* p_indices);

INSTANTIATEMAXPOOLWITHINDEX(float)
INSTANTIATEMAXPOOLWITHINDEX(double)
INSTANTIATEMAXPOOLWITHINDEX(half)
INSTANTIATEMAXPOOLWITHINDEX(int8_t)
INSTANTIATEMAXPOOLWITHINDEX(uint8_t)

}  // namespace cuda
}  // namespace onnxruntime
