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

namespace onnxruntime {
namespace cuda {

template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const bool is_mode_avg,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = is_mode_avg
            ? (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)  // mode Avg
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
    const int64_t* batch_indices_ptr) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // RoI could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    const auto roi_batch_ind = batch_indices_ptr[n];

    bool continuous_coordinate = false;
    // Do not using rounding; this implementation detail is critical
    T roi_offset = continuous_coordinate ? T(0.5) : T(0);
    T roi_start_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale - roi_offset;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale - roi_offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!continuous_coordinate) { // backward compatiblity
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : roundf(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : roundf(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    bool max_flag = false;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
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

    top_data[index] = output_val;
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
  const int64_t* batch_indices_ptr) {
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
      batch_indices_ptr);
}

#define SPECIALIZED_IMPL(T)                     \
  template void RoiAlignImpl<T>(                \
        cudaStream_t stream,              \
        const int64_t nthreads,                 \
        const T* bottom_data,                   \
        const T spatial_scale,                  \
        const int64_t channels,                 \
        const int64_t height,                   \
        const int64_t width,                    \
        const int64_t pooled_height,            \
        const int64_t pooled_width,             \
        const int64_t sampling_ratio,           \
        const T* bottom_rois,                   \
        int64_t roi_cols,                       \
        T* top_data,                            \
        const bool is_mode_avg,                 \
        const int64_t* batch_indices_ptr);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)

} // namespace cuda
} // namespace onnxruntime
