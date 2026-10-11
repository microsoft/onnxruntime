// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "avg_pool_impl.h"

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

// Accumulate half in float for precision; keep native type otherwise.
template <typename T>
struct AveragePoolAccumulator {
  using type = T;
};
template <>
struct AveragePoolAccumulator<half> {
  using type = float;
};
template <>
struct AveragePoolAccumulator<BFloat16> {
  using type = float;
};

template <typename T, bool Layout>
__global__ void AveragePoolWithPadKernel(
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
    int64_t pad_h_head,
    int64_t pad_w_head,
    int64_t pad_d_head,
    int64_t pad_h_tail,
    int64_t pad_w_tail,
    int64_t pad_d_tail,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t dilation_d,
    fast_divmod fdm_c,
    fast_divmod fdm_h,
    fast_divmod fdm_w,
    fast_divmod fdm_d,
    bool count_include_pad,
    const T* p_input,
    int64_t output_size,
    T* p_output) {
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

  // Window bounds mirror the CPU AveragePool{1,2,3}DTask reference exactly.
  int64_t h_start = h_index * stride_h - pad_h_head;
  int64_t w_start = w_index * stride_w - pad_w_head;
  int64_t d_start = d_index * stride_d - pad_d_head;

  int64_t h_end = _Min<int64_t>(h_start + kernel_h * dilation_h, height + pad_h_tail);
  int64_t w_end = _Min<int64_t>(w_start + kernel_w * dilation_w, width + pad_w_tail);
  int64_t d_end = _Min<int64_t>(d_start + kernel_d * dilation_d, depth + pad_d_tail);

  using AccT = typename AveragePoolAccumulator<T>::type;
  AccT acc = static_cast<AccT>(0);
  int64_t counted = 0;

  int64_t offset = compute_offset(n_index, c_index, 0, 0, 0);
  const T* p_slice = p_input + offset;
  for (int64_t h = h_start; h < h_end; h += dilation_h) {
    if (h < 0 || h >= height) continue;
    for (int64_t w = w_start; w < w_end; w += dilation_w) {
      if (w < 0 || w >= width) continue;
      for (int64_t d = d_start; d < d_end; d += dilation_d) {
        if (d < 0 || d >= depth) continue;
        acc += static_cast<AccT>(p_slice[compute_offset(0, 0, h, w, d)]);
        ++counted;
      }
    }
  }

  AccT result = static_cast<AccT>(0);
  if (counted > 0) {
    if (count_include_pad) {
      int64_t divisor = (1 + (h_end - h_start - 1) / dilation_h) *
                        (1 + (w_end - w_start - 1) / dilation_w) *
                        (1 + (d_end - d_start - 1) / dilation_d);
      result = acc / static_cast<AccT>(divisor);
    } else {
      result = acc / static_cast<AccT>(counted);
    }
  }
  p_output[id] = static_cast<T>(result);
}

template <typename T, bool Layout>
void AveragePoolWithPad(
    cudaStream_t stream,
    const TensorShape& input_shape,
    const TensorShape& output_shape,
    const gsl::span<const int64_t>& kernel_shape,
    const gsl::span<const int64_t>& stride_shape,
    const gsl::span<const int64_t>& pads,
    const gsl::span<const int64_t>& dilations,
    bool count_include_pad,
    const T* p_input,
    T* p_output) {
  int64_t channels, height, width, depth;
  int64_t pooled_height, pooled_width, pooled_depth;
  if constexpr (Layout == LAYOUT_NCHW) {
    channels = input_shape[1];
    height = input_shape[2];
    width = kernel_shape.size() > 1 ? input_shape[3] : 1;
    depth = kernel_shape.size() > 2 ? input_shape[4] : 1;

    pooled_height = output_shape[2];
    pooled_width = kernel_shape.size() > 1 ? output_shape[3] : 1;
    pooled_depth = kernel_shape.size() > 2 ? output_shape[4] : 1;
  } else if constexpr (Layout == LAYOUT_NHWC) {
    height = input_shape[1];
    width = kernel_shape.size() > 1 ? input_shape[2] : 1;
    depth = kernel_shape.size() > 2 ? input_shape[3] : 1;
    channels = input_shape[input_shape.NumDimensions() - 1];

    pooled_height = output_shape[1];
    pooled_width = kernel_shape.size() > 1 ? output_shape[2] : 1;
    pooled_depth = kernel_shape.size() > 2 ? output_shape[3] : 1;
  }

  const int64_t rank = static_cast<int64_t>(kernel_shape.size());
  int64_t kernel_h = kernel_shape[0];
  int64_t kernel_w = rank > 1 ? kernel_shape[1] : 1;
  int64_t kernel_d = rank > 2 ? kernel_shape[2] : 1;
  int64_t stride_h = stride_shape[0];
  int64_t stride_w = rank > 1 ? stride_shape[1] : 1;
  int64_t stride_d = rank > 2 ? stride_shape[2] : 1;

  // pads: [x1_begin,...,xN_begin, x1_end,...,xN_end]. Begin at [i], end at [rank + i].
  int64_t pad_h_head = pads[0];
  int64_t pad_w_head = rank > 1 ? pads[1] : 0;
  int64_t pad_d_head = rank > 2 ? pads[2] : 0;
  int64_t pad_h_tail = pads[rank + 0];
  int64_t pad_w_tail = rank > 1 ? pads[rank + 1] : 0;
  int64_t pad_d_tail = rank > 2 ? pads[rank + 2] : 0;

  int64_t dilation_h = dilations[0];
  int64_t dilation_w = rank > 1 ? dilations[1] : 1;
  int64_t dilation_d = rank > 2 ? dilations[2] : 1;

  int64_t output_size = output_shape.Size();
  if (output_size == 0) return;

  fast_divmod fdm_c(static_cast<int>(channels));
  fast_divmod fdm_h(static_cast<int>(pooled_height));
  fast_divmod fdm_w(static_cast<int>(pooled_width));
  fast_divmod fdm_d(static_cast<int>(pooled_depth));

  int blocksPerGrid = (int)((output_size + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock);
  AveragePoolWithPadKernel<T, Layout><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
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
      pad_h_head,
      pad_w_head,
      pad_d_head,
      pad_h_tail,
      pad_w_tail,
      pad_d_tail,
      dilation_h,
      dilation_w,
      dilation_d,
      fdm_c,
      fdm_h,
      fdm_w,
      fdm_d,
      count_include_pad,
      p_input,
      output_size,
      p_output);
}

#define INSTANTIATE_AVERAGEPOOLWITHPAD(T, Layout)   \
  template void AveragePoolWithPad<T, Layout>(      \
      cudaStream_t stream,                          \
      const TensorShape& input_shape,               \
      const TensorShape& output_shape,              \
      const gsl::span<const int64_t>& kernel_shape, \
      const gsl::span<const int64_t>& stride_shape, \
      const gsl::span<const int64_t>& pads,         \
      const gsl::span<const int64_t>& dilations,    \
      bool count_include_pad,                       \
      const T* p_input,                             \
      T* p_output);

INSTANTIATE_AVERAGEPOOLWITHPAD(float, LAYOUT_NCHW)
INSTANTIATE_AVERAGEPOOLWITHPAD(double, LAYOUT_NCHW)
INSTANTIATE_AVERAGEPOOLWITHPAD(half, LAYOUT_NCHW)
INSTANTIATE_AVERAGEPOOLWITHPAD(BFloat16, LAYOUT_NCHW)

#ifdef ENABLE_CUDA_NHWC_OPS
INSTANTIATE_AVERAGEPOOLWITHPAD(float, LAYOUT_NHWC)
INSTANTIATE_AVERAGEPOOLWITHPAD(double, LAYOUT_NHWC)
INSTANTIATE_AVERAGEPOOLWITHPAD(half, LAYOUT_NHWC)
INSTANTIATE_AVERAGEPOOLWITHPAD(BFloat16, LAYOUT_NHWC)
#endif

}  // namespace cuda
}  // namespace onnxruntime
