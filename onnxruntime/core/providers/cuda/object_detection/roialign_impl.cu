/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* Modifications Copyright (c) Microsoft. */

#include "roialign_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__device__ AccumulationType_t<T> bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    AccumulationType_t<T> y,
    AccumulationType_t<T> x,
    const bool is_mode_avg,
    const int index /* index for debug only*/) {
  using TAcc = AccumulationType_t<T>;

  // deal with cases that inverse elements are out of feature map boundary
  if (y < static_cast<TAcc>(-1.0f) || y > static_cast<TAcc>(height) ||
      x < static_cast<TAcc>(-1.0f) || x > static_cast<TAcc>(width)) {
    // empty
    return static_cast<TAcc>(0.0f);
  }

  if (y <= static_cast<TAcc>(0.0f)) {
    y = static_cast<TAcc>(0.0f);
  }
  if (x <= static_cast<TAcc>(0.0f)) {
    x = static_cast<TAcc>(0.0f);
  }

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = static_cast<TAcc>(y_low);
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = static_cast<TAcc>(x_low);
  } else {
    x_high = x_low + 1;
  }

  TAcc ly = y - static_cast<TAcc>(y_low);
  TAcc lx = x - static_cast<TAcc>(x_low);
  TAcc hy = static_cast<TAcc>(1.0f) - ly;
  TAcc hx = static_cast<TAcc>(1.0f) - lx;
  // do bilinear interpolation
  TAcc v1 = static_cast<TAcc>(bottom_data[y_low * width + x_low]);
  TAcc v2 = static_cast<TAcc>(bottom_data[y_low * width + x_high]);
  TAcc v3 = static_cast<TAcc>(bottom_data[y_high * width + x_low]);
  TAcc v4 = static_cast<TAcc>(bottom_data[y_high * width + x_high]);
  TAcc w1 = hy * hx;
  TAcc w2 = hy * lx;
  TAcc w3 = ly * hx;
  TAcc w4 = ly * lx;

  TAcc val = is_mode_avg
                 ? (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)             // mode Avg
                 : max(max(max(w1 * v1, w2 * v2), w3 * v3), w4 * v4);  // mode Max

  return val;
}

template <typename T>
__global__ void RoIAlignForward(
    const int64_t nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio,
    const T* bottom_rois,
    int64_t roi_cols,
    T* top_data,
    const bool is_mode_avg,
    const bool half_pixel,
    const int64_t* batch_indices_ptr,
    const int64_t batch_size) {
  using TAcc = AccumulationType_t<T>;

  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // RoI could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    const auto roi_batch_ind = batch_indices_ptr[n];
    // Validate batch_indices values are within [0, batch_size).
    // If the index is out of range, we set the output to 0 for this RoI element.
    if (roi_batch_ind < 0 || roi_batch_ind >= batch_size) {
      CUDA_KERNEL_ASSERT(false && "batch_indices values are out of range");
      top_data[index] = static_cast<T>(0.0f);
      continue;
    }

    // Do not using rounding; this implementation detail is critical
    const TAcc spatial_scale_acc = static_cast<TAcc>(spatial_scale);
    const TAcc roi_offset = half_pixel ? static_cast<TAcc>(0.5f) : static_cast<TAcc>(0.0f);
    TAcc roi_start_w = static_cast<TAcc>(offset_bottom_rois[0]) * spatial_scale_acc - roi_offset;
    TAcc roi_start_h = static_cast<TAcc>(offset_bottom_rois[1]) * spatial_scale_acc - roi_offset;
    TAcc roi_end_w = static_cast<TAcc>(offset_bottom_rois[2]) * spatial_scale_acc - roi_offset;
    TAcc roi_end_h = static_cast<TAcc>(offset_bottom_rois[3]) * spatial_scale_acc - roi_offset;

    TAcc roi_width = roi_end_w - roi_start_w;
    TAcc roi_height = roi_end_h - roi_start_h;
    if (!half_pixel) {  // backward compatibility
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, static_cast<TAcc>(1.0f));
      roi_height = max(roi_height, static_cast<TAcc>(1.0f));
    }
    const TAcc bin_size_h = roi_height / static_cast<TAcc>(pooled_height);
    const TAcc bin_size_w = roi_width / static_cast<TAcc>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : static_cast<int>(_Ceil(roi_height / static_cast<TAcc>(pooled_height)));  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(_Ceil(roi_width / static_cast<TAcc>(pooled_width)));

    // We do average (integral) pooling inside a bin
    const int grid_count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
    const TAcc count = static_cast<TAcc>(grid_count);  // e.g. = 4

    TAcc output_val = static_cast<TAcc>(0.0f);
    bool max_flag = false;
    for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g., iy = 0, 1
    {
      const TAcc y = roi_start_h + static_cast<TAcc>(ph) * bin_size_h +
                     (static_cast<TAcc>(iy) + static_cast<TAcc>(0.5f)) * bin_size_h /
                         static_cast<TAcc>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const TAcc x = roi_start_w + static_cast<TAcc>(pw) * bin_size_w +
                       (static_cast<TAcc>(ix) + static_cast<TAcc>(0.5f)) * bin_size_w /
                           static_cast<TAcc>(roi_bin_grid_w);

        const TAcc val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, is_mode_avg, index);

        if (is_mode_avg) {
          output_val += val;
        } else {
          if (!max_flag) {
            output_val = val;
            max_flag = true;
          } else {
            output_val = max(output_val, val);
          }
        }
      }
    }
    if (is_mode_avg) {
      output_val /= count;
    }

    top_data[index] = static_cast<T>(output_val);
  }
}

template <typename T>
void RoiAlignImpl(
    cudaStream_t stream,
    const int64_t nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio,
    const T* bottom_rois,
    int64_t roi_cols,
    T* top_data,
    const bool is_mode_avg,
    const bool half_pixel,
    const int64_t* batch_indices_ptr,
    const int64_t batch_size) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(nthreads) / GridDim::maxThreadsPerBlock));
  RoIAlignForward<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      nthreads,
      bottom_data,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      bottom_rois,
      roi_cols,
      top_data,
      is_mode_avg,
      half_pixel,
      batch_indices_ptr,
      batch_size);
}

#define SPECIALIZED_IMPL(T)             \
  template void RoiAlignImpl<T>(        \
      cudaStream_t stream,              \
      const int64_t nthreads,           \
      const T* bottom_data,             \
      const T spatial_scale,            \
      const int64_t channels,           \
      const int64_t height,             \
      const int64_t width,              \
      const int64_t pooled_height,      \
      const int64_t pooled_width,       \
      const int64_t sampling_ratio,     \
      const T* bottom_rois,             \
      int64_t roi_cols,                 \
      T* top_data,                      \
      const bool is_mode_avg,           \
      const bool half_pixel,            \
      const int64_t* batch_indices_ptr, \
      const int64_t batch_size);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(BFloat16)

}  // namespace cuda
}  // namespace onnxruntime
