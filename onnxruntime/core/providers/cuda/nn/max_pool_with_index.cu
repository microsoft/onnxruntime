// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "max_pool_with_index.h"

#include <cfloat>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
template <typename T, bool Layout>
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

  auto compute_offset =
    [height, width, depth, channels](int n_index, int c_index, int h_index, int w_index, int d_index) -> int64_t {
    if constexpr (Layout == LAYOUT_NCHW) {
      return (((n_index * channels + c_index) * height + h_index) * width + w_index) * depth + d_index;
    } else if constexpr (Layout == LAYOUT_NHWC) {
      return (((n_index * height + h_index) * width + w_index) * depth + d_index) * channels + c_index;
    }
  };

  int d_index, w_index, h_index, c_index, n_index, id_tmp;
  if constexpr (Layout == LAYOUT_NCHW) {
    fdm_d.divmod(id, id_tmp, d_index);
    fdm_w.divmod(id_tmp, id_tmp, w_index);
    fdm_h.divmod(id_tmp, id_tmp, h_index);
    fdm_c.divmod(id_tmp, n_index, c_index);
  } else if constexpr (Layout == LAYOUT_NHWC) {
    fdm_c.divmod(id, id_tmp, c_index);
    fdm_d.divmod(id_tmp, id_tmp, d_index);
    fdm_w.divmod(id_tmp, id_tmp, w_index);
    fdm_h.divmod(id_tmp, n_index, h_index);
  }

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
  int64_t offset = compute_offset(n_index, c_index, 0, 0, 0);
  const T* p_slice = p_input + offset;
  T maxval = p_slice[compute_offset(0, 0, h_start, w_start, d_start)] - (T)1;
  for (int64_t d = d_start; d < d_end; d += dilation_d) {
    for (int64_t w = w_start; w < w_end; w += dilation_w) {
      for (int64_t h = h_start; h < h_end; h += dilation_h) {
        auto pool_offset = compute_offset(0, 0, h, w, d);
        if (p_slice[pool_offset] > maxval) {
          h_index_max = h;
          w_index_max = w;
          d_index_max = d;
          maxval = static_cast<float>(p_slice[pool_offset]);
        }
      }
    }
  }
  p_output[id] = p_input[offset + compute_offset(0, 0, h_index_max, w_index_max, d_index_max)];

  if (p_indices) {
    if constexpr (Layout == LAYOUT_NCHW) {
      p_indices[id] = storage_order == 0 ? offset + h_index_max * width * depth + w_index_max * depth + d_index_max
                                         : offset + h_index_max + w_index_max * height + d_index_max * width * height;
    } else if constexpr (Layout == LAYOUT_NHWC) {
      // The tests currently have to be provided in NHWC layout so that tests do not fail. When converting between
      // layouts, does it make sense to do an index conversion as well?
      // Storing indices in NHWC layout isn't critical as they are supposed to be used by Unpooling operations
      // which currently assume that indices reference to Tensors in NHWC layout.
      int64_t id_nchw = 
        (((n_index * channels + c_index) * pooled_height + h_index) * pooled_width + w_index) * pooled_depth + d_index;
      int64_t offset_nchw = (n_index * channels + c_index) * width * height * depth;

      p_indices[id_nchw] = (storage_order == 0)
                               ? offset_nchw + h_index_max * width * depth + w_index_max * depth + d_index_max
                               : offset_nchw + h_index_max + w_index_max * height + d_index_max * width * height;
    }
  }
}

template <typename T, bool Layout>
void MaxPoolWithIndex(
    cudaStream_t stream,
    const TensorShape& input_shape,
    const TensorShape& output_shape,
    const gsl::span<const int64_t>& kernel_shape,
    const gsl::span<const int64_t>& stride_shape,
    const gsl::span<const int64_t>& pads,
    const gsl::span<const int64_t>& dilations,
    int64_t storage_order,
    const T* p_input,
    T* p_output,
    int64_t* p_indices) {
  int64_t batchs, channels, height, width, depth;
  int64_t pooled_height, pooled_width, pooled_depth;
  if constexpr (Layout == LAYOUT_NCHW) {
    batchs = input_shape[0];
    channels = input_shape[1];
    height = input_shape[2];
    width = kernel_shape.size() > 1 ? input_shape[3] : 1;
    depth = kernel_shape.size() > 2 ? input_shape[4] : 1;

    pooled_height = output_shape[2];
    pooled_width = kernel_shape.size() > 1 ? output_shape[3] : 1;
    pooled_depth = kernel_shape.size() > 2 ? output_shape[4] : 1;
  } else if constexpr (Layout == LAYOUT_NHWC) {
    batchs = input_shape[0];
    height = input_shape[1];
    width = kernel_shape.size() > 1 ? input_shape[2] : 1;
    depth = kernel_shape.size() > 2 ? input_shape[3] : 1;
    channels = input_shape[input_shape.NumDimensions() - 1];

    pooled_height = output_shape[1];
    pooled_width = kernel_shape.size() > 1 ? output_shape[2] : 1;
    pooled_depth = kernel_shape.size() > 2 ? output_shape[3] : 1;
  }
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
  MaxPoolWithIndexKernel<T, Layout><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
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

#define INSTANTIATEMAXPOOLWITHINDEX(T, Layout)      \
  template void MaxPoolWithIndex<T, Layout>(        \
      cudaStream_t stream,                          \
      const TensorShape& input_shape,               \
      const TensorShape& output_shape,              \
      const gsl::span<const int64_t>& kernel_shape, \
      const gsl::span<const int64_t>& stride_shape, \
      const gsl::span<const int64_t>& pads,         \
      const gsl::span<const int64_t>& dilations,    \
      int64_t storage_order,                        \
      const T* p_input,                             \
      T* p_output,                                  \
      int64_t* p_indices);

INSTANTIATEMAXPOOLWITHINDEX(float, LAYOUT_NCHW)
INSTANTIATEMAXPOOLWITHINDEX(double, LAYOUT_NCHW)
INSTANTIATEMAXPOOLWITHINDEX(half, LAYOUT_NCHW)
INSTANTIATEMAXPOOLWITHINDEX(int8_t, LAYOUT_NCHW)
INSTANTIATEMAXPOOLWITHINDEX(uint8_t, LAYOUT_NCHW)

#ifdef ENABLE_CUDA_NHWC_OPS
INSTANTIATEMAXPOOLWITHINDEX(float, LAYOUT_NHWC)
INSTANTIATEMAXPOOLWITHINDEX(double, LAYOUT_NHWC)
INSTANTIATEMAXPOOLWITHINDEX(half, LAYOUT_NHWC)
INSTANTIATEMAXPOOLWITHINDEX(int8_t, LAYOUT_NHWC)
INSTANTIATEMAXPOOLWITHINDEX(uint8_t, LAYOUT_NHWC)
#endif

}  // namespace cuda
}  // namespace onnxruntime
