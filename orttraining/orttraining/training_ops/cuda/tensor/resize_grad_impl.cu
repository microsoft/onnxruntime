// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Contents of this file are derived from the pytorch cuda implementation of
// the upsample_bilinear2d_backward implementation at:
// https://github.com/pytorch/pytorch/blob/ce50132748f652ed6079c3db8008a6817594dbae/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu

#include "orttraining/training_ops/cuda/tensor/resize_grad_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"

namespace onnxruntime::cuda {

namespace {

constexpr int NumThreadsPerBlock = GridDim::maxThreadsPerBlock;

}  // namespace

__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t h,
    const size_t w) {
  return (nc * height + h) * width + w;
}

template <typename T>
__device__ __forceinline__ static T AreaPixelComputeSourceIndex(
    T scale,
    int dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    T src_idx = scale * (dst_index + static_cast<T>(0.5)) -
                static_cast<T>(0.5);
    return (!cubic && src_idx < static_cast<T>(0))
               ? static_cast<T>(0)
               : src_idx;
  }
}

template <typename T, typename AccT>
__global__ void UpsampleGrad(const int64_t nc, const int64_t input_height,
                             const int64_t input_width, const int64_t output_height,
                             const int64_t output_width, const AccT rheight,
                             const AccT rwidth, const bool align_corners,
                             const T* dY_data, T* dX_data) {
  const size_t dy_numel = nc * output_width * output_height;
  const size_t dx_numel = nc * input_width * input_height;
  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x;
       index < dy_numel;
       index += blockDim.x * gridDim.x) {
    size_t index_temp = index;
    const int w2 = index_temp % output_width;  // 0:width2-1
    index_temp /= output_width;
    const int h2 = index_temp % output_height;  // 0:height2-1
    const size_t nc = index_temp / output_height;

    const AccT h1r = AreaPixelComputeSourceIndex<AccT>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < input_height - 1) ? 1 : 0;
    const AccT h1lambda = h1r - h1;
    const AccT h0lambda = static_cast<AccT>(1) - h1lambda;

    const AccT w1r = AreaPixelComputeSourceIndex<AccT>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < input_width - 1) ? 1 : 0;
    const AccT w1lambda = w1r - w1;
    const AccT w0lambda = static_cast<AccT>(1) - w1lambda;

    const T d2val = dY_data[index];
    AtomicAdd(
        dX_data,
        idx(nc, input_height, input_width, h1, w1),
        dx_numel,
        static_cast<T>(h0lambda * w0lambda) * d2val);
    AtomicAdd(
        dX_data,
        idx(nc, input_height, input_width, h1, w1 + w1p),
        dx_numel,
        static_cast<T>(h0lambda * w1lambda) * d2val);
    AtomicAdd(
        dX_data,
        idx(nc, input_height, input_width, h1 + h1p, w1),
        dx_numel,
        static_cast<T>(h1lambda * w0lambda) * d2val);
    AtomicAdd(
        dX_data,
        idx(nc, input_height, input_width, h1 + h1p, w1 + w1p),
        dx_numel,
        static_cast<T>(h1lambda * w1lambda) * d2val);
  }
}

template <typename T>
T AreaPixelComputeScale(int64_t input_size, int64_t output_size, bool align_corners,
                        const std::optional<float>& scale) {
  if (align_corners) {
    if (output_size <= 1) {
      return T{0};
    }
    return static_cast<T>(input_size - 1) / static_cast<T>(output_size - 1);
  } else {
    if (scale.has_value()) {
      return static_cast<T>(T{1.0} / *scale);
    } else {
      return static_cast<T>(input_size) / static_cast<T>(output_size);
    }
  }
}

template <typename T>
void ResizeGradImpl(cudaStream_t stream, int64_t input_height,
                    int64_t input_width, int64_t output_height,
                    int64_t output_width, int64_t batch_size,
                    int64_t channels, bool align_corners,
                    const std::optional<float>& scale_height,
                    const std::optional<float>& scale_width,
                    const T* dY_data, T* dX_data) {
  float rheight = AreaPixelComputeScale<float>(input_height, output_height, align_corners, scale_height);
  float rwidth = AreaPixelComputeScale<float>(input_width, output_width, align_corners, scale_width);

  const size_t output_numel = batch_size * channels * output_height * output_width;
  int blocks_per_grid = (int)(ceil(static_cast<float>(output_numel) / NumThreadsPerBlock));
  UpsampleGrad<T><<<blocks_per_grid, NumThreadsPerBlock, 0, stream>>>(
      batch_size * channels, input_height, input_width, output_height, output_width,
      rheight, rwidth, align_corners, dY_data, dX_data);
}

#define SPECIALIZED_RESIZEGRAD_IMPL(T)                                        \
  template void ResizeGradImpl<T>(cudaStream_t stream, int64_t input_height,  \
                                  int64_t input_width, int64_t output_height, \
                                  int64_t output_width, int64_t batch_size,   \
                                  int64_t channels, bool align_corners,       \
                                  const std::optional<float>& scale_height,   \
                                  const std::optional<float>& scale_width,    \
                                  const T* dY_data, T* dX_data);

SPECIALIZED_RESIZEGRAD_IMPL(half)
SPECIALIZED_RESIZEGRAD_IMPL(float)
SPECIALIZED_RESIZEGRAD_IMPL(double)

#undef SPECIALIZED_RESIZEGRAD_IMPL

}  // namespace onnxruntime::cuda
