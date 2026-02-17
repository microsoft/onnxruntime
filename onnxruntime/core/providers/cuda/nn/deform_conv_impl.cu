// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CUDA implementation of DeformConv: deformable im2col kernel + bilinear interpolation.
// Reference: torchvision deform_conv2d_kernel.cu, ONNX DeformConv spec.

#include "deform_conv_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/common/float16.h"
#include <type_traits>
#include <algorithm>
#include <limits>

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kDeformConvThreadsPerBlock = 256;

// Calculate grid size with a safety limit to prevent overflow.
// Since we use grid-stride loops in kernels, limiting the grid size is safe.
inline int GetGridSize(size_t n, size_t threads_per_block) {
  size_t blocks_needed = (n + threads_per_block - 1) / threads_per_block;
  return static_cast<int>(std::min(blocks_needed, static_cast<size_t>(std::numeric_limits<int>::max())));
}

// Bilinear interpolation at (h, w). Returns 0 if out of bounds (ONNX spec).
template <typename T>
__device__ __inline__ T BilinearInterpolate(
    const T* in,
    int64_t height,
    int64_t width,
    T h,
    T w) {
  if (h <= static_cast<T>(-1) || h >= height || w <= static_cast<T>(-1) || w >= width) {
    return static_cast<T>(0);
  }
  int h_low = static_cast<int>(_Floor(h));
  int w_low = static_cast<int>(_Floor(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - static_cast<T>(h_low);
  T lw = w - static_cast<T>(w_low);
  T hh = static_cast<T>(1) - lh;
  T hw = static_cast<T>(1) - lw;

  T v1 = (h_low >= 0 && w_low >= 0) ? __ldg(in + h_low * width + w_low) : static_cast<T>(0);
  T v2 = (h_low >= 0 && w_high < width) ? __ldg(in + h_low * width + w_high) : static_cast<T>(0);
  T v3 = (h_high < height && w_low >= 0) ? __ldg(in + h_high * width + w_low) : static_cast<T>(0);
  T v4 = (h_high < height && w_high < width) ? __ldg(in + h_high * width + w_high) : static_cast<T>(0);

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// FP16/BF16: coordinate and weight math in float to avoid precision loss.
template <typename T>
struct DeformConvUseFloatCoords : std::false_type {};
template <>
struct DeformConvUseFloatCoords<half> : std::true_type {};
template <>
struct DeformConvUseFloatCoords<BFloat16> : std::true_type {};

// __ldg has no overload for BFloat16*; use 16-bit load + FromBits. Other types use __ldg directly.
template <typename T>
__device__ __inline__ T DeformConvLdg(const T* p) {
  return __ldg(p);
}
template <>
__device__ __inline__ BFloat16 DeformConvLdg<BFloat16>(const BFloat16* p) {
  return BFloat16::FromBits(__ldg(reinterpret_cast<const uint16_t*>(p)));
}

__device__ __inline__ half BilinearInterpolate(
    const half* in,
    int64_t height,
    int64_t width,
    float h,
    float w) {
  if (h <= -1.0f || h >= height || w <= -1.0f || w >= width) {
    return __float2half(0.0f);
  }
  int h_low = static_cast<int>(floorf(h));
  int w_low = static_cast<int>(floorf(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - static_cast<float>(h_low);
  float lw = w - static_cast<float>(w_low);
  float hh = 1.0f - lh;
  float hw = 1.0f - lw;

  float v1 = (h_low >= 0 && w_low >= 0) ? __half2float(__ldg(in + h_low * width + w_low)) : 0.0f;
  float v2 = (h_low >= 0 && w_high < width) ? __half2float(__ldg(in + h_low * width + w_high)) : 0.0f;
  float v3 = (h_high < height && w_low >= 0) ? __half2float(__ldg(in + h_high * width + w_low)) : 0.0f;
  float v4 = (h_high < height && w_high < width) ? __half2float(__ldg(in + h_high * width + w_high)) : 0.0f;

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return __float2half(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

__device__ __inline__ BFloat16 BilinearInterpolate(
    const BFloat16* in,
    int64_t height,
    int64_t width,
    float h,
    float w) {
  if (h <= -1.0f || h >= height || w <= -1.0f || w >= width) {
    return BFloat16(0.0f);
  }
  int h_low = static_cast<int>(floorf(h));
  int w_low = static_cast<int>(floorf(w));
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - static_cast<float>(h_low);
  float lw = w - static_cast<float>(w_low);
  float hh = 1.0f - lh;
  float hw = 1.0f - lw;

  float v1 = (h_low >= 0 && w_low >= 0) ? static_cast<float>(DeformConvLdg(in + h_low * width + w_low)) : 0.0f;
  float v2 = (h_low >= 0 && w_high < width) ? static_cast<float>(DeformConvLdg(in + h_low * width + w_high)) : 0.0f;
  float v3 = (h_high < height && w_low >= 0) ? static_cast<float>(DeformConvLdg(in + h_high * width + w_low)) : 0.0f;
  float v4 = (h_high < height && w_high < width) ? static_cast<float>(DeformConvLdg(in + h_high * width + w_high)) : 0.0f;

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return BFloat16(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

// 1D parallel: each thread handles (in_c, out_y, out_x, out_b), inner loop over kH x kW.
// num_kernels = C * out_h * out_w * parallel_imgs.
// Col layout row-major: rows = C*kH*kW, cols = parallel_imgs*out_h*out_w.
// data_col[col_row_idx * col_stride + c_col] with col_stride = parallel_imgs*out_h*out_w.
template <typename T, typename IndexT>
__global__ void DeformableIm2ColKernel(
    IndexT num_kernels,
    const T* __restrict__ input,
    const T* __restrict__ offset,
    const T* __restrict__ mask,
    int64_t height,
    int64_t width,
    int64_t weight_h,
    int64_t weight_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t channels,
    int64_t offset_group,
    DivMod<IndexT> out_h_div,
    DivMod<IndexT> out_w_div,
    DivMod<IndexT> parallel_imgs_div,
    DivMod<IndexT> channel_per_offset_grp_div,
    bool use_mask,
    T* __restrict__ data_col) {
  
  // Reconstruct dimensions from DivMod objects
  const int64_t out_h = out_h_div.d_;
  const int64_t out_w = out_w_div.d_;
  const int64_t parallel_imgs = parallel_imgs_div.d_;
  
  const int64_t out_size = out_h * out_w;
  const int64_t col_stride = parallel_imgs * out_size;

  for (IndexT index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels; index += blockDim.x * gridDim.x) {
    IndexT val = index;
    IndexT out_x, out_y, out_b, in_c;
    
    // Fast division/modulo to recover coordinates
    out_w_div.divmod(val, val, out_x);
    out_h_div.divmod(val, val, out_y);
    parallel_imgs_div.divmod(val, in_c, out_b);

    IndexT offset_grp, dummy;
    channel_per_offset_grp_div.divmod(in_c, offset_grp, dummy);

    const T* input_ptr = input + out_b * (channels * height * width) + in_c * (height * width);
    const T* offset_ptr = offset + out_b * (offset_group * 2 * weight_h * weight_w * out_size) +
                          offset_grp * (2 * weight_h * weight_w * out_size);
    const T* mask_ptr = use_mask ? (mask + out_b * (offset_group * weight_h * weight_w * out_size) +
                                    offset_grp * (weight_h * weight_w * out_size))
                                : nullptr;

    const int64_t c_col = out_b * out_size + out_y * out_w + out_x;

    // Use float for coordinate math when T is half or BFloat16 to avoid precision loss.
    using CoordT = typename std::conditional<DeformConvUseFloatCoords<T>::value, float, T>::type;

#pragma unroll
    for (int64_t i = 0; i < weight_h; ++i) {
#pragma unroll
      for (int64_t j = 0; j < weight_w; ++j) {
        const int64_t mask_idx = i * weight_w + j;
        const int64_t offset_idx = 2 * mask_idx;

        T mask_val = static_cast<T>(1);
        if (use_mask) {
          mask_val = DeformConvLdg(mask_ptr + mask_idx * out_size + out_y * out_w + out_x);
        }

        const int64_t offset_h_idx = (offset_idx)*out_size + out_y * out_w + out_x;
        const int64_t offset_w_idx = (offset_idx + 1) * out_size + out_y * out_w + out_x;
        const CoordT offset_h = static_cast<CoordT>(DeformConvLdg(offset_ptr + offset_h_idx));
        const CoordT offset_w = static_cast<CoordT>(DeformConvLdg(offset_ptr + offset_w_idx));

        const CoordT h_im = static_cast<CoordT>(out_y * stride_h - pad_h + i * dilation_h) + offset_h;
        const CoordT w_im = static_cast<CoordT>(out_x * stride_w - pad_w + j * dilation_w) + offset_w;

        T val = static_cast<T>(0);
        if (mask_val != static_cast<T>(0)) {
          val = BilinearInterpolate(input_ptr, height, width, h_im, w_im);
        }

        const int64_t col_row_idx = (in_c * weight_h * weight_w) + (i * weight_w + j);
        data_col[col_row_idx * col_stride + c_col] = val * mask_val;
      }
    }
  }
}

// Bias add: Y[n,m,oh,ow] += B[m]. Layout NCHW.
template <typename T>
__global__ void DeformConvAddBiasKernel(T* Y, const T* B, int64_t N, int64_t M, int64_t out_h, int64_t out_w) {
  int64_t out_size = out_h * out_w;
  int64_t total = N * M * out_size;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    int64_t m = (idx / out_size) % M;
    Y[idx] += DeformConvLdg(B + m);
  }
}

// Copy GEMM output (row-major [M_per_group, cur_parallel*output_image_size]) into NCHW Y_g.
// src(c, j) with j = b_idx*output_image_size + pos -> dst[b_idx*M*output_image_size + c*output_image_size + pos].
template <typename T>
__global__ void CopyGemmOutputRowMajorToNCHWKernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t M,
    int64_t M_per_group,
    int64_t output_image_size,
    int64_t cur_parallel) {
  int64_t total = cur_parallel * M_per_group * output_image_size;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
    int64_t pos = idx % output_image_size;
    int64_t c = (idx / output_image_size) % M_per_group;
    int64_t b_idx = idx / (output_image_size * M_per_group);
    int64_t j = b_idx * output_image_size + pos;
    // src index for row-major: c * (cur_parallel * output_image_size) + j
    dst[b_idx * M * output_image_size + c * output_image_size + pos] = src[c * (cur_parallel * output_image_size) + j];
  }
}

}  // namespace

template <typename T>
void DeformConvAddBiasImpl(cudaStream_t stream, T* Y, const T* B, int64_t N, int64_t M, int64_t out_h, int64_t out_w) {
  int64_t total = N * M * out_h * out_w;
  if (total <= 0) return;
  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
  DeformConvAddBiasKernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(Y, B, N, M, out_h, out_w);
}

template <typename T>
void DeformConvCopyGemmOutputRowMajorToNCHW(
    cudaStream_t stream,
    const T* gemm_output,
    T* Y_g,
    int64_t M,
    int64_t M_per_group,
    int64_t output_image_size,
    int64_t cur_parallel) {
  int64_t total = cur_parallel * M_per_group * output_image_size;
  if (total <= 0) return;
  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
  CopyGemmOutputRowMajorToNCHWKernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
      gemm_output, Y_g, M, M_per_group, output_image_size, cur_parallel);
}

template <typename T>
void DeformConvIm2ColImpl(
    cudaStream_t stream,
    const T* input,
    const T* offset,
    const T* mask,
    T* col_buffer,
    int64_t parallel_imgs,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kH,
    int64_t kW,
    int64_t out_h,
    int64_t out_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t offset_group,
    bool use_mask) {
  const int64_t num_kernels = static_cast<int64_t>(C) * out_h * out_w * parallel_imgs;
  if (num_kernels <= 0) {
    return;
  }

  const int64_t col_numel = static_cast<int64_t>(C) * kH * kW * parallel_imgs * out_h * out_w;
  const bool use_64bit = (num_kernels > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) ||
                        (col_numel > static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  int blocks = GetGridSize(static_cast<size_t>(num_kernels), kDeformConvThreadsPerBlock);

  if (use_64bit) {
    DeformableIm2ColKernel<T, int64_t><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
        num_kernels,
        input,
        offset,
        mask,
        H,
        W,
        kH,
        kW,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        C, // channels is C
        offset_group,
        DivMod<int64_t>(out_h),
        DivMod<int64_t>(out_w),
        DivMod<int64_t>(parallel_imgs),
        DivMod<int64_t>(C / offset_group),
        use_mask,
        col_buffer);
  } else {
    DeformableIm2ColKernel<T, int32_t><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
        static_cast<int32_t>(num_kernels),
        input,
        offset,
        mask,
        H,
        W,
        kH,
        kW,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        C, // channels is C
        offset_group,
        DivMod<int32_t>(static_cast<int32_t>(out_h)),
        DivMod<int32_t>(static_cast<int32_t>(out_w)),
        DivMod<int32_t>(static_cast<int32_t>(parallel_imgs)),
        DivMod<int32_t>(static_cast<int32_t>(C / offset_group)),
        use_mask,
        col_buffer);
  }
}

#define INST_DeformConvIm2ColImpl(T) \
  template void DeformConvIm2ColImpl<T>(cudaStream_t, const T*, const T*, const T*, T*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool);

INST_DeformConvIm2ColImpl(float)
INST_DeformConvIm2ColImpl(double)
INST_DeformConvIm2ColImpl(half)
INST_DeformConvIm2ColImpl(BFloat16)

template void DeformConvCopyGemmOutputRowMajorToNCHW<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, int64_t);
template void DeformConvCopyGemmOutputRowMajorToNCHW<double>(cudaStream_t, const double*, double*, int64_t, int64_t, int64_t, int64_t);
template void DeformConvCopyGemmOutputRowMajorToNCHW<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, int64_t);
template void DeformConvCopyGemmOutputRowMajorToNCHW<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, int64_t, int64_t, int64_t, int64_t);

template void DeformConvAddBiasImpl<float>(cudaStream_t, float*, const float*, int64_t, int64_t, int64_t, int64_t);
template void DeformConvAddBiasImpl<double>(cudaStream_t, double*, const double*, int64_t, int64_t, int64_t, int64_t);
template void DeformConvAddBiasImpl<half>(cudaStream_t, half*, const half*, int64_t, int64_t, int64_t, int64_t);
template void DeformConvAddBiasImpl<BFloat16>(cudaStream_t, BFloat16*, const BFloat16*, int64_t, int64_t, int64_t, int64_t);

// Delegate ORT type to CUDA type (e.g. MLFloat16 -> half); avoids repeating three identical specializations.
#define DELEGATE_DEFORM_CONV_IMPL(ORT_T, CUDA_T)                                           \
  template <>                                                                              \
  void DeformConvIm2ColImpl<ORT_T>(cudaStream_t stream, const ORT_T* input,              \
                                   const ORT_T* offset, const ORT_T* mask, ORT_T* col_buffer, \
                                   int64_t parallel_imgs, int64_t C, int64_t H, int64_t W, \
                                   int64_t kH, int64_t kW, int64_t out_h, int64_t out_w,   \
                                   int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w, \
                                   int64_t dilation_h, int64_t dilation_w, int64_t offset_group, bool use_mask) { \
    DeformConvIm2ColImpl<CUDA_T>(stream, reinterpret_cast<const CUDA_T*>(input),           \
                                 reinterpret_cast<const CUDA_T*>(offset),                  \
                                 mask ? reinterpret_cast<const CUDA_T*>(mask) : nullptr,   \
                                 reinterpret_cast<CUDA_T*>(col_buffer),                    \
                                 parallel_imgs, C, H, W, kH, kW, out_h, out_w,            \
                                 pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, \
                                 offset_group, use_mask);                                  \
  }                                                                                       \
  template <>                                                                              \
  void DeformConvCopyGemmOutputRowMajorToNCHW<ORT_T>(cudaStream_t stream,                 \
                                                     const ORT_T* gemm_output, ORT_T* Y_g, \
                                                     int64_t M, int64_t M_per_group,       \
                                                     int64_t output_image_size, int64_t cur_parallel) { \
    DeformConvCopyGemmOutputRowMajorToNCHW<CUDA_T>(stream,                                  \
                                                 reinterpret_cast<const CUDA_T*>(gemm_output), \
                                                 reinterpret_cast<CUDA_T*>(Y_g),            \
                                                 M, M_per_group, output_image_size, cur_parallel); \
  }                                                                                       \
  template <>                                                                              \
  void DeformConvAddBiasImpl<ORT_T>(cudaStream_t stream, ORT_T* Y, const ORT_T* B,        \
                                   int64_t N, int64_t M, int64_t out_h, int64_t out_w) {  \
    DeformConvAddBiasImpl<CUDA_T>(stream, reinterpret_cast<CUDA_T*>(Y),                    \
                                  reinterpret_cast<const CUDA_T*>(B), N, M, out_h, out_w); \
  }

DELEGATE_DEFORM_CONV_IMPL(MLFloat16, half)

}  // namespace cuda
}  // namespace onnxruntime
