// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "grid_sample_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
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

template <typename T>
__device__ T PixelAtGrid(const T* input_data, int64_t bIdx, int64_t cIdx, int64_t y, int64_t x,
    int64_t padding_mode, int64_t N, int64_t C, int64_t H, int64_t W, float border[4]) {
  T pixel = 0.0f;
  if (padding_mode == 0) {  // zeros
    if (x >= 0 && x < W && y >= 0 && y < H) {
      pixel = input_data[bIdx * C * H * W + cIdx * H * W + y * W + x];
    }
  } else if (padding_mode == 1) {  //border
    x = std::clamp<int64_t>(x, 0, W - 1);
    y = std::clamp<int64_t>(y, 0, H - 1);
    pixel = input_data[bIdx * C * H * W + cIdx * H * W + y * W + x];
  } else {  // Reflection
    x = (int64_t) GsReflect<T>(x, border[0], border[2]);
    y = (int64_t) GsReflect<T>(y, border[1], border[3]);
    pixel = input_data[bIdx * C * H * W + cIdx * H * W + y * W + x];
  }
  return pixel;
}

template <typename T>
__device__ T PixelAtGrid3D(const T* input_data, int64_t bIdx, int64_t cIdx, int64_t z, int64_t y, int64_t x,
                           int64_t padding_mode, int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, float border[6]) {
  T pixel = 0.0f;
  if (padding_mode == 0) {  // zeros
    if (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D) {
      pixel = input_data[bIdx * C * D * H * W + cIdx * D * H * W + z * H * W + y * W + x];
    }
  } else if (padding_mode == 1) {  // border
    x = std::clamp<int64_t>(x, 0, W - 1);
    y = std::clamp<int64_t>(y, 0, H - 1);
    z = std::clamp<int64_t>(z, 0, D - 1);
    pixel = input_data[bIdx * C * D * H * W + cIdx * D * H * W + z * H * W + y * W + x];
  } else {  // Reflection
    x = (int64_t)GsReflect<T>(x, border[0], border[3]);
    y = (int64_t)GsReflect<T>(y, border[1], border[4]);
    z = (int64_t)GsReflect<T>(z, border[2], border[5]);
    pixel = input_data[bIdx * C * D * H * W + cIdx * D * H * W + z * H * W + y * W + x];
  }
  return pixel;
}

__device__ void GsGetCubicCoeffs(float x, float coeffs[4])
{
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

template <typename T>
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
    int BIdx = idx / (C * H_out * W_out );
    int tmpBCnt = BIdx * (C * H_out * W_out);

    int cIdx = (idx - tmpBCnt) / (H_out * W_out);
    int tmpCCnt = tmpBCnt + cIdx * (H_out * W_out);

    int yIdx = (idx - tmpCCnt) / W_out;
    int tmpHCnt = tmpCCnt + yIdx * W_out;

    int xIdx = (idx - tmpHCnt);

    int grid_idx = BIdx * H_out * W_out + yIdx * W_out + xIdx;
    T grid_X = grid_data[grid_idx * 2 + 0];
    T grid_Y = grid_data[grid_idx * 2 + 1];
    int outIdx = idx;

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

    T grid_x_imgSpace = GsDenormalize(grid_X, W_in, align_corners == 1);
    T grid_y_imgSpace = GsDenormalize(grid_Y, H_in, align_corners == 1);

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

      T lt_v = PixelAtGrid(input_data, BIdx, cIdx, y1, x1, padding_mode, N, C, H_in, W_in, border);
      T rt_v = PixelAtGrid(input_data, BIdx, cIdx, y1, x2, padding_mode, N, C, H_in, W_in, border);
      T lb_v = PixelAtGrid(input_data, BIdx, cIdx, y2, x1, padding_mode, N, C, H_in, W_in, border);
      T rb_v = PixelAtGrid(input_data, BIdx, cIdx, y2, x2, padding_mode, N, C, H_in, W_in, border);
      T interpoV = w_lt * lt_v + w_rt * rt_v + w_lb * lb_v + w_rb * rb_v;
      output_data[outIdx] = interpoV;
      return;
    }
    if (mode == 1) {  // nearest
      grid_x_imgSpace = nearbyint(grid_x_imgSpace);
      grid_y_imgSpace = nearbyint(grid_y_imgSpace);
      int x_n = grid_x_imgSpace;
      int y_n = grid_y_imgSpace;
      output_data[outIdx] = PixelAtGrid(input_data, BIdx, cIdx, y_n, x_n, padding_mode, N, C, H_in, W_in, border);
      return;
    }
    if (mode == 2) {  // bicubic
      int64_t x0 = static_cast<int64_t>(std::floor(grid_x_imgSpace)) - 1;  // top-left corner of the bbox
      int64_t y0 = static_cast<int64_t>(std::floor(grid_y_imgSpace)) - 1;
      T p[4][4] = {};  // [H][W]
      for (int64_t h = 0; h < 4; h++) {
        for (int64_t w = 0; w < 4; w++) {
          p[h][w] = PixelAtGrid(input_data, BIdx, cIdx, h + y0, w + x0, padding_mode, N, C, H_in, W_in, border);
        }
      }
      T dx = grid_x_imgSpace - x0 - 1;
      T dy = grid_y_imgSpace - y0 - 1;
      output_data[outIdx] = GsBicubicInterpolate(p, dx, dy);
    }
}

template <typename T>
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
  int blocksPerGrid = (int)(ceil(static_cast<T>(dims[0] * dims[1] * H_out * W_out) / GridDim::maxThreadsPerBlock));
  _GridSampleKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, grid_data, mode, padding_mode, align_corners, dims[0], dims[1], dims[2], dims[3], H_out, W_out, output_data);
}

#define SPECIALIZED_IMPL(T) \
  template void GridSampleImpl<T>(cudaStream_t stream, const T* input_data, const T* grid_data, \
                                  const int64_t mode, const int64_t padding_mode, const int64_t align_corners, \
                                  const int64_t[4], const int64_t H_out, const int64_t W_out, T* output_data);

SPECIALIZED_IMPL(float)

template <typename T>
__global__ void _GridSampleKernel3D(
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t N,
    const int64_t C,
    const int64_t D_in,
    const int64_t H_in,
    const int64_t W_in,
    const int64_t D_out,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(idx, N * C * D_out * H_out * W_out);
  // extract batch index, channel index, y index, x index for current thread
  int BIdx = idx / (C * D_out * H_out * W_out);
  int tmpBCnt = BIdx * (C * D_out * H_out * W_out);

  int cIdx = (idx - tmpBCnt) / (D_out * H_out * W_out);
  int tmpCCnt = tmpBCnt + cIdx * (D_out * H_out * W_out);

  int zIdx = (idx - tmpCCnt) / (H_out * W_out);
  int tmpDCnt = tmpCCnt + zIdx * H_out * W_out;

  int yIdx = (idx - tmpDCnt) / W_out;
  int tmpHCnt = tmpDCnt + yIdx * W_out;

  int xIdx = (idx - tmpHCnt);

  int grid_idx = BIdx * D_out * H_out * W_out + zIdx * H_out * W_out + yIdx * W_out + xIdx;
  T grid_X = grid_data[grid_idx * 3 + 0];
  T grid_Y = grid_data[grid_idx * 3 + 1];
  T grid_Z = grid_data[grid_idx * 3 + 2];
  int outIdx = idx;

  float x_min = -0.5f;
  float x_max = W_in - 0.5f;
  float y_min = -0.5f;
  float y_max = H_in - 0.5f;
  float z_min = -0.5f;
  float z_max = D_in - 0.5f;

  if (align_corners) {
      x_min = 0.0f;
      x_max = W_in - 1.0;
      y_min = 0.0f;
      y_max = H_in - 1.0f;
      z_min = 0.0f;
      z_max = D_in - 1.0f;
  }
  float border[] = {x_min, y_min, z_min, x_max, y_max, z_max};  // l-t-n-r-b-f

  T grid_x_imgSpace = GsDenormalize(grid_X, W_in, align_corners == 1);
  T grid_y_imgSpace = GsDenormalize(grid_Y, H_in, align_corners == 1);
  T grid_z_imgSpace = GsDenormalize(grid_Z, D_in, align_corners == 1);

  if (mode == 0) {  // trilinear
      int x1 = floor(grid_x_imgSpace);
      int y1 = floor(grid_y_imgSpace);
      int z1 = floor(grid_z_imgSpace);
      int x2 = x1 + 1;
      int y2 = y1 + 1;
      int z2 = z1 + 1;
      T dx2 = static_cast<T>(x2) - grid_x_imgSpace;
      T dx1 = grid_x_imgSpace - static_cast<T>(x1);
      T dy2 = static_cast<T>(y2) - grid_y_imgSpace;
      T dy1 = grid_y_imgSpace - static_cast<T>(y1);
      T dz2 = static_cast<T>(z2) - grid_z_imgSpace;
      T dz1 = grid_z_imgSpace - static_cast<T>(z1);

      T p111 = PixelAtGrid3D(input_data, BIdx, cIdx, z1, y1, x1, padding_mode, N, C, D_in, H_in, W_in, border);
      T p112 = PixelAtGrid3D(input_data, BIdx, cIdx, z1, y1, x2, padding_mode, N, C, D_in, H_in, W_in, border);
      T p121 = PixelAtGrid3D(input_data, BIdx, cIdx, z1, y2, x1, padding_mode, N, C, D_in, H_in, W_in, border);
      T p122 = PixelAtGrid3D(input_data, BIdx, cIdx, z1, y2, x2, padding_mode, N, C, D_in, H_in, W_in, border);
      T Y_gridpoint_z1 = dy2 * (dx2 * p111 + dx1 * p112) + dy1 * (dx2 * p121 + dx1 * p122);

      T p211 = PixelAtGrid3D(input_data, BIdx, cIdx, z2, y1, x1, padding_mode, N, C, D_in, H_in, W_in, border);
      T p212 = PixelAtGrid3D(input_data, BIdx, cIdx, z2, y1, x2, padding_mode, N, C, D_in, H_in, W_in, border);
      T p221 = PixelAtGrid3D(input_data, BIdx, cIdx, z2, y2, x1, padding_mode, N, C, D_in, H_in, W_in, border);
      T p222 = PixelAtGrid3D(input_data, BIdx, cIdx, z2, y2, x2, padding_mode, N, C, D_in, H_in, W_in, border);
      T Y_gridpoint_z2 = dy2 * (dx2 * p211 + dx1 * p212) + dy1 * (dx2 * p221 + dx1 * p222);
      output_data[outIdx] = dz2 * Y_gridpoint_z1 + dz1 * Y_gridpoint_z2;
      return;
  }
  if (mode == 1) {  // nearest
      T x = static_cast<T>(std::nearbyint(static_cast<T>(grid_x_imgSpace)));
      T y = static_cast<T>(std::nearbyint(static_cast<T>(grid_y_imgSpace)));
      T z = static_cast<T>(std::nearbyint(static_cast<T>(grid_z_imgSpace)));
      output_data[outIdx] = PixelAtGrid3D(input_data, BIdx, cIdx, static_cast<int64_t>(z), static_cast<int64_t>(y), static_cast<int64_t>(x),
                                          padding_mode, N, C, D_in, H_in, W_in, border);
      return;
  }
}

template <typename T>
void GridSampleImpl3D(
    cudaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const int64_t dims[5],
    const int64_t D_out,
    const int64_t H_out,
    const int64_t W_out,
    T* output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<T>(dims[0] * dims[1] * D_out * H_out * W_out) / GridDim::maxThreadsPerBlock));
  _GridSampleKernel3D<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, grid_data, mode, padding_mode, align_corners, dims[0], dims[1], dims[2], dims[3], dims[4], D_out, H_out, W_out, output_data);
}

#define SPECIALIZED_IMPL_3D(T)                                                                                    \
  template void GridSampleImpl3D<T>(cudaStream_t stream, const T* input_data, const T* grid_data,                \
                                  const int64_t mode, const int64_t padding_mode, const int64_t align_corners, \
                                  const int64_t[5], const int64_t D_out, const int64_t H_out, const int64_t W_out, T* output_data);

SPECIALIZED_IMPL_3D(float)

}  // namespace cuda
}  // namespace onnxruntime
