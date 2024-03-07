// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "grid_sample_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ T GsDenormalize(T n, int64_t length, bool align_corners) {
  T x = {};
  if (align_corners) {  // align_corners: true => [-1, 1] to [0, length - 1]
    x = (n + static_cast<T>(1)) / static_cast<T>(2) * (length - 1);
  } else {  // align_corners: false => [-1, 1] to [-0.5, length - 0.5]
    x = ((n + static_cast<T>(1)) * length - static_cast<T>(1)) / static_cast<T>(2);
  }
  return x;
}


template <typename T>
__device__ T GsReflect(T x, float x_min, float x_max) {
  float fx = static_cast<float>(x);
  float dx = {};
  float range = x_max - x_min;
  if (fx < x_min) {
    dx = x_min - fx;
    int n = static_cast<int>(dx / range);
    float r = dx - n * range;
    if (n % 2 == 0) {
      fx = x_min + r;
    } else {
      fx = x_max - r;
    }
  } else if (fx > x_max) {
    dx = fx - x_max;
    int n = static_cast<int>(dx / range);
    float r = dx - n * range;
    if (n % 2 == 0) {
      fx = x_max - r;
    } else {
      fx = x_min + r;
    }
  }
  // else fallthrough
  return static_cast<T>(fx);
}

template <typename T, bool Layout>
__device__ T PixelAtGrid(const T* input_data, int64_t bIdx, int64_t cIdx, int64_t y, int64_t x,
                         int64_t padding_mode, int64_t N, int64_t C, int64_t H, int64_t W, float border[4]) {
  T pixel = 0.0f;

  auto PixelOffset = [bIdx, cIdx, C, H, W](int64_t x, int64_t y) -> int64_t {
    return Layout == LAYOUT_NCHW
       ? (bIdx * C * H * W + cIdx * H * W + y * W + x)
       : (bIdx * H * W * C + y * W * C + x * C + cIdx);
  };

  if (padding_mode == 0) {  // zeros
    if (x >= 0 && x < W && y >= 0 && y < H) {
      pixel = input_data[PixelOffset(x, y)];
    }
  } else if (padding_mode == 1) {  // border
    x = max((int64_t)0, min((int64_t)W - 1, (int64_t)x));
    y = max((int64_t)0, min((int64_t)H - 1, (int64_t)y));
    pixel = input_data[PixelOffset(x, y)];
  } else {  // Reflection
    x = (int64_t)GsReflect<T>(x, border[0], border[2]);
    y = (int64_t)GsReflect<T>(y, border[1], border[3]);
    pixel = input_data[PixelOffset(x, y)];
  }
  return pixel;
}

__device__ void GsGetCubicCoeffs(float x, float coeffs[4]) {
  float cubic_alpha = -0.75f;
  x = abs(x);
  coeffs[0] = (((cubic_alpha * (x + 1) - 5 * cubic_alpha) * (x + 1) + 8 * cubic_alpha) * (x + 1) - 4 * cubic_alpha);
  coeffs[1] = (((cubic_alpha + 2) * x - (cubic_alpha + 3)) * x * x + 1);
  coeffs[2] = (((cubic_alpha + 2) * (1 - x) - (cubic_alpha + 3)) * (1 - x) * (1 - x) + 1);
  coeffs[3] = (((cubic_alpha * (2 - x) - 5 * cubic_alpha) * (2 - x) + 8 * cubic_alpha) * (2 - x) - 4 * cubic_alpha);
}

template <typename T>
__device__ T GsBicubicInterpolate(T p[4][4], float x, float y) {
  float v[4] = {};
  float coeffs[4] = {};
  GsGetCubicCoeffs(x, coeffs);
  for (int64_t i = 0; i < 4; i++) {
    v[i] = coeffs[0] * p[i][0] + coeffs[1] * p[i][1] + coeffs[2] * p[i][2] + coeffs[3] * p[i][3];
  }
  GsGetCubicCoeffs(y, coeffs);
  T pixel = static_cast<T>(coeffs[0] * v[0] + coeffs[1] * v[1] + coeffs[2] * v[2] + coeffs[3] * v[3]);
  return pixel;
}

template <typename T, bool Layout>
__global__ void _GridSampleKernel(
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t N,
    const int64_t C,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(idx, N * C * H_out * W_out);
    // extract batch index, channel index, y index, x index for current thread
    int BIdx, yIdx, xIdx, cIdx;
    if constexpr (Layout == LAYOUT_NCHW) {
      BIdx = idx / (C * H_out * W_out);
      int tmpBCnt = BIdx * (C * H_out * W_out);

      cIdx = (idx - tmpBCnt) / (H_out * W_out);
      int tmpCCnt = tmpBCnt + cIdx * (H_out * W_out);

      yIdx = (idx - tmpCCnt) / W_out;
      int tmpHCnt = tmpCCnt + yIdx * W_out;

      xIdx = (idx - tmpHCnt);
    } else {
      static_assert(Layout == LAYOUT_NHWC, "Unsupported layout");

      BIdx = idx / (H_out * W_out * C);
      int tmpBCnt = BIdx * (H_out * W_out * C);

      yIdx = (idx - tmpBCnt) / (W_out * C);
      int tmpHCnt = tmpBCnt + yIdx * (W_out * C);

      xIdx = (idx - tmpHCnt) / C;
      int tmpWCnt = tmpHCnt + xIdx * C;

      cIdx = (idx - tmpWCnt);
    }

    int grid_idx = BIdx * H_out * W_out + yIdx * W_out + xIdx;
    T grid_X = grid_data[grid_idx * 2 + 0];
    T grid_Y = grid_data[grid_idx * 2 + 1];
    int outIdx = idx;

    T grid_x_imgSpace = GsDenormalize(grid_X, W_in, align_corners == 1);
    T grid_y_imgSpace = GsDenormalize(grid_Y, H_in, align_corners == 1);
    if (mode == 1) {  //nearest
      grid_x_imgSpace = nearbyint(grid_x_imgSpace);
      grid_y_imgSpace = nearbyint(grid_y_imgSpace);
    }
    float x_min = -0.5f;
    float x_max = W_in - 0.5f;
    float y_min = -0.5f;
    float y_max = H_in - 0.5f;

    if (align_corners) {
      x_min = 0.0f;
      x_max = W_in - 1.0;
      y_min = 0.0f;
      y_max = H_in - 1.0f;
    }
    float border[] = {x_min, y_min, x_max, y_max};  // l-t-r-b
    if (grid_x_imgSpace < x_min || grid_x_imgSpace > x_max ||
        grid_y_imgSpace < y_min || grid_y_imgSpace > y_max) { // out of bound
      if (padding_mode == 1) {  // border
        // Clamping must not be done here, see #10607
        // grid_x_imgSpace = max(0.0f, min(grid_x_imgSpace, W_in - 1.0f));
        // grid_y_imgSpace = max(0.0f, min(grid_y_imgSpace, H_in - 1.0f));
      } else if (padding_mode == 2) {  // reflection
        grid_x_imgSpace = GsReflect(grid_x_imgSpace, x_min, x_max);
        grid_y_imgSpace = GsReflect(grid_y_imgSpace, y_min, y_max);
      }
    }

    if (mode == 0) {  // bilinear
      int x1 = floor(grid_x_imgSpace);
      int y1 = floor(grid_y_imgSpace);
      int x2 = x1 + 1;
      int y2 = y1 + 1;
      T w_lt = 0.0f;
      T w_rt = 0.0f;
      T w_lb = 0.0f;
      T w_rb = 0.0f;

      T w_r = grid_x_imgSpace - x1;
      T w_l = 1.0f - w_r;
      T w_b = grid_y_imgSpace - y1;
      T w_t = 1.0f - w_b;

      w_lt = w_t * w_l;
      w_rt = w_t * w_r;
      w_lb = w_b * w_l;
      w_rb = w_b * w_r;

      T lt_v = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y1, x1, padding_mode, N, C, H_in, W_in, border);
      T rt_v = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y1, x2, padding_mode, N, C, H_in, W_in, border);
      T lb_v = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y2, x1, padding_mode, N, C, H_in, W_in, border);
      T rb_v = PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y2, x2, padding_mode, N, C, H_in, W_in, border);
      T interpoV = w_lt * lt_v + w_rt * rt_v + w_lb * lb_v + w_rb * rb_v;
      output_data[outIdx] = interpoV;
      return;
    }
    if (mode == 1) {  // nearest
      int x_n = grid_x_imgSpace;
      int y_n = grid_y_imgSpace;
      output_data[outIdx] =
        PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, y_n, x_n, padding_mode, N, C, H_in, W_in, border);
      return;
    }
    if (mode == 2) {  // bicubic
      int64_t x0 = static_cast<int64_t>(std::floor(grid_x_imgSpace)) - 1;  // top-left corner of the bbox
      int64_t y0 = static_cast<int64_t>(std::floor(grid_y_imgSpace)) - 1;
      T p[4][4] = {};  // [H][W]
      for (int64_t h = 0; h < 4; h++) {
        for (int64_t w = 0; w < 4; w++) {
          p[h][w] = 
            PixelAtGrid<T, Layout>(input_data, BIdx, cIdx, h + y0, w + x0, padding_mode, N, C, H_in, W_in, border);
        }
      }
      T dx = grid_x_imgSpace - x0 - 1;
      T dy = grid_y_imgSpace - y0 - 1;
      output_data[outIdx] = GsBicubicInterpolate(p, dx, dy);
    }
}

template <typename T, bool IsNHWC>
void GridSampleImpl(
    cudaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims[4],
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  using Ch = Channels<IsNHWC>;

  int blocksPerGrid = static_cast<int>(
    ceil(static_cast<T>(dims[Ch::N] * dims[Ch::C] * H_out * W_out) / GridDim::maxThreadsPerBlock));
  _GridSampleKernel<T, IsNHWC><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, grid_data, mode, padding_mode, align_corners, 
      dims[Ch::N], dims[Ch::C], dims[Ch::H], dims[Ch::W],
      H_out, W_out, output_data);
}

#define SPECIALIZED_IMPL(T, IsNHWC)                                                                                    \
  template void GridSampleImpl<T, IsNHWC>(cudaStream_t stream, const T* input_data, const T* grid_data,                \
                                          const int64_t mode, const int64_t padding_mode, const int64_t align_corners, \
                                          const int64_t[4], const int64_t H_out, const int64_t W_out, T* output_data);

SPECIALIZED_IMPL(float, false)  // NCHW
SPECIALIZED_IMPL(float, true)   // NHWC

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
