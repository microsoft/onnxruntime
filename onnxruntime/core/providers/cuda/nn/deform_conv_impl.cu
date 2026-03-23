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

template <int N>
struct DeformConvKSize {
  static constexpr int value = N;
};

// Calculate grid size with a safety limit to prevent overflow.
// Since we use grid-stride loops in kernels, limiting the grid size is safe.
inline int GetGridSize(size_t n, size_t threads_per_block) {
  size_t blocks_needed = (n + threads_per_block - 1) / threads_per_block;
  return static_cast<int>(std::min(blocks_needed, static_cast<size_t>(std::numeric_limits<int>::max())));
}

// __ldg has no overload for BFloat16*; use 16-bit load + FromBits. Other types use __ldg directly.
template <typename T>
__device__ __inline__ T DeformConvLdg(const T* p) {
  return __ldg(p);
}
template <>
__device__ __inline__ BFloat16 DeformConvLdg<BFloat16>(const BFloat16* p) {
  return BFloat16::FromBits(__ldg(reinterpret_cast<const uint16_t*>(p)));
}

// Traits for bilinear interpolation math:
// - ComputeT: type used for coordinate/weight math (float for half/BFloat16, T otherwise)
// - Load:     load one element and convert to ComputeT
// - ToResult: convert ComputeT result back to T
// - Zero:     zero value of T
template <typename T>
struct DeformConvBilinearTraits {
  using ComputeT = T;

  __device__ static __inline__ ComputeT Load(const T* p) {
    return __ldg(p);
  }

  __device__ static __inline__ T ToResult(ComputeT v) {
    return v;
  }

  __device__ static __inline__ T Zero() {
    return static_cast<T>(0);
  }
};

template <>
struct DeformConvBilinearTraits<half> {
  using ComputeT = float;

  __device__ static __inline__ ComputeT Load(const half* p) {
    return __half2float(__ldg(p));
  }

  __device__ static __inline__ half ToResult(ComputeT v) {
    return __float2half(v);
  }

  __device__ static __inline__ half Zero() {
    return __float2half(0.0f);
  }
};

template <>
struct DeformConvBilinearTraits<BFloat16> {
  using ComputeT = float;

  __device__ static __inline__ ComputeT Load(const BFloat16* p) {
    return static_cast<float>(DeformConvLdg(p));
  }

  __device__ static __inline__ BFloat16 ToResult(ComputeT v) {
    return BFloat16(v);
  }

  __device__ static __inline__ BFloat16 Zero() {
    return BFloat16(0.0f);
  }
};

// Bilinear interpolation at (h, w). Returns 0 if out of bounds (ONNX spec).
// Indices h_low, w_low, h_high, w_high use int (not int64_t) to reduce register pressure and
// improve occupancy in the hot path. Limitation: (H+1)*W must not exceed INT_MAX; this is
// validated on the host side in DeformConvValidateAndParse to guarantee index math in int
// does not overflow. For half/BFloat16, coordinate and weight math use float via
// DeformConvBilinearTraits to avoid precision loss. We keep floor() results in CoordT and
// cast to int only for indices (h_low/w_low), which avoids unnecessary CoordT->int->CoordT
// round trips when computing lh/lw/hh/hw.
template <typename T>
__device__ __inline__ T BilinearInterpolate(
    const T* in,
    int height,
    int width,
    typename DeformConvBilinearTraits<T>::ComputeT h,
    typename DeformConvBilinearTraits<T>::ComputeT w) {
  using Traits = DeformConvBilinearTraits<T>;
  using CoordT = typename Traits::ComputeT;

  // [Optimization 1]: Early exit for clearly out-of-bounds (skip floor() for OOB case).
  if (h <= static_cast<CoordT>(-1) || h >= height || w <= static_cast<CoordT>(-1) || w >= width) {
    return Traits::Zero();
  }

  // [Optimization 2]: Keep floor result in T; cast to int only for indices. Avoids float->int->float in lh/lw.
  CoordT h_floor = _Floor(h);
  CoordT w_floor = _Floor(w);
  int h_low = static_cast<int>(h_floor);
  int w_low = static_cast<int>(w_floor);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  CoordT lh = h - h_floor;
  CoordT lw = w - w_floor;
  CoordT hh = static_cast<CoordT>(1) - lh;
  CoordT hw = static_cast<CoordT>(1) - lw;

  // [Optimization 3]: Avoid a second multiply for base_high.
  // Original code computed both bases as:
  //   base_low  = h_low  * width;
  //   base_high = h_high * width;
  // Since h_high = h_low + 1, we can rewrite base_high as base_low + width and
  // save one integer multiply in the hot path:
  //   base_low  = h_low  * width;
  //   base_high = base_low + width;
  int base_low = h_low * width;
  int base_high = base_low + width;

  CoordT v1 = (h_low >= 0 && w_low >= 0) ? Traits::Load(in + base_low + w_low) : static_cast<CoordT>(0);
  CoordT v2 = (h_low >= 0 && w_high < width) ? Traits::Load(in + base_low + w_high) : static_cast<CoordT>(0);
  CoordT v3 = (h_high < height && w_low >= 0) ? Traits::Load(in + base_high + w_low) : static_cast<CoordT>(0);
  CoordT v4 = (h_high < height && w_high < width) ? Traits::Load(in + base_high + w_high) : static_cast<CoordT>(0);

  CoordT w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return Traits::ToResult(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

// kH/kW = -1 means dynamic (runtime); >= 0 means compile-time constant for loop unrolling.
template <typename T, typename IndexT, int kH = -1, int kW = -1>
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
  constexpr bool is_fixed = (kH >= 0 && kW >= 0);
  const int64_t h_dim = is_fixed ? kH : weight_h;
  const int64_t w_dim = is_fixed ? kW : weight_w;

  // Reconstruct dimensions from DivMod objects
  const int64_t out_h = out_h_div.d_;
  const int64_t out_w = out_w_div.d_;
  const int64_t parallel_imgs = parallel_imgs_div.d_;

  const int64_t out_size = out_h * out_w;
  // The stride for data_col is (parallel_imgs * out_h * out_w)
  const int64_t col_stride = parallel_imgs * out_size;

  using CoordT = typename DeformConvBilinearTraits<T>::ComputeT;

  for (IndexT index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels; index += blockDim.x * gridDim.x) {
    IndexT val = index;
    IndexT out_x, out_y, out_b, in_c;

    // Fast division/modulo to recover coordinates
    out_w_div.divmod(val, val, out_x);
    out_h_div.divmod(val, val, out_y);
    parallel_imgs_div.divmod(val, in_c, out_b);

    // [Optimization 3] Avoid expensive division if offset_group is 1 (very common case).
    IndexT offset_grp = 0;
    if (offset_group > 1) {
      IndexT dummy;
      channel_per_offset_grp_div.divmod(in_c, offset_grp, dummy);
    }

    // [Optimization 2] Common Subexpression Elimination (CSE) & Pointer Arithmetic
    // Pre-calculate base pointers to reduce integer arithmetic inside the inner loops.

    // 1. Input pointer base for this batch and channel.
    const T* input_ptr = input + static_cast<int64_t>(out_b) * (channels * height * width) + static_cast<int64_t>(in_c) * (height * width);

    // 2. Spatial index in the output feature map.
    const int64_t spatial_idx = static_cast<int64_t>(out_y) * out_w + static_cast<int64_t>(out_x);

    // 3. Offset pointer base calculation.
    // Layout: (N, offset_groups, 2*KH*KW, OH, OW)
    // We pre-calculate the pointer to the start of the specific (n, g) block, plus spatial_idx.
    const int64_t offset_group_block_size = 2 * h_dim * w_dim * out_size;
    const T* offset_ptr_base = offset + (static_cast<int64_t>(out_b) * offset_group + static_cast<int64_t>(offset_grp)) * offset_group_block_size + spatial_idx;

    // 4. Mask pointer base calculation (if used).
    // Layout: (N, offset_groups, KH*KW, OH, OW)
    const T* mask_ptr_base = nullptr;
    if (use_mask) {
      const int64_t mask_group_block_size = h_dim * w_dim * out_size;
      mask_ptr_base = mask + (static_cast<int64_t>(out_b) * offset_group + static_cast<int64_t>(offset_grp)) * mask_group_block_size + spatial_idx;
    }

    // 5. Output pointer base calculation.
    // data_col Layout: (C * KH * KW, N * OH * OW)
    // The current thread writes to the column `c_col` = (b * OH * OW) + spatial_idx.
    // The starting row for this channel is `in_c * KH * KW`.
    const int64_t c_col = static_cast<int64_t>(out_b) * out_size + spatial_idx;
    T* data_col_ptr_base = data_col + (static_cast<int64_t>(in_c) * h_dim * w_dim) * col_stride + c_col;

    // 6. Pre-calculate invariant coordinate parts.
    // Use float for coordinate math when T is half or BFloat16 to avoid precision loss.
    const CoordT base_h_im = static_cast<CoordT>(out_y * stride_h - pad_h);
    const CoordT base_w_im = static_cast<CoordT>(out_x * stride_w - pad_w);

    auto process_kernel_point = [&](int64_t i, int64_t j) {
      const int64_t kernel_idx = i * w_dim + j;
      T mask_val = static_cast<T>(1);
      if (use_mask) {
        // Access mask using pre-calculated base and stride.
        mask_val = DeformConvLdg(mask_ptr_base + kernel_idx * out_size);
      }

      // Calculate offset pointers relative to the base.
      // The offset tensor stores (y_offset, x_offset) pairs for each kernel weight.
      // Stride between y_offset and x_offset is `out_size`.
      const int64_t offset_offset_idx = (2 * kernel_idx) * out_size;

      const CoordT offset_h = static_cast<CoordT>(DeformConvLdg(offset_ptr_base + offset_offset_idx));
      const CoordT offset_w = static_cast<CoordT>(DeformConvLdg(offset_ptr_base + offset_offset_idx + out_size));

      const CoordT h_im = base_h_im + static_cast<CoordT>(i * dilation_h) + offset_h;
      const CoordT w_im = base_w_im + static_cast<CoordT>(j * dilation_w) + offset_w;

      // height/width are validated on host (DeformConvValidateAndParse) so int is safe here.
      T val = BilinearInterpolate(input_ptr,
                                  static_cast<int>(height),
                                  static_cast<int>(width),
                                  h_im,
                                  w_im);

      // Match CPU path: always interpolate then apply mask to keep branch-free hot loop.
      data_col_ptr_base[kernel_idx * col_stride] = val * mask_val;
    };

    if constexpr (is_fixed) {
#pragma unroll
      for (int i = 0; i < kH; ++i) {
#pragma unroll
        for (int j = 0; j < kW; ++j) {
          process_kernel_point(i, j);
        }
      }
    } else {
      for (int64_t i = 0; i < weight_h; ++i) {
        for (int64_t j = 0; j < weight_w; ++j) {
          process_kernel_point(i, j);
        }
      }
    }
  }
}

// Bias add: Y[n,m,oh,ow] += B[m]. Layout NCHW.
template <typename T>
__global__ void DeformConvAddBiasKernel(
    T* Y,
    const T* B,
    DivMod<int64_t> spatial_div,  // For dividing by (H * W)
    DivMod<int64_t> channel_div,  // For dividing by M (channel count)
    int64_t total_elements) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
    int64_t val = idx;
    int64_t batch_channel_idx, pixel_idx;

    // 1. First decomposition: decompose idx into (batch_channel_idx, pixel_idx)
    // Equivalent to: batch_channel_idx = idx / (H*W); pixel_idx = idx % (H*W);
    spatial_div.divmod(val, batch_channel_idx, pixel_idx);

    int64_t batch_idx, channel_idx;

    // 2. Second decomposition: decompose batch_channel_idx into (batch_idx, channel_idx)
    // Equivalent to: channel_idx = batch_channel_idx % M;
    // We only need channel_idx (i.e. m)
    channel_div.divmod(batch_channel_idx, batch_idx, channel_idx);
    (void)batch_idx;  // Only channel_idx is needed

    // channel_idx is what we need (i.e. m)
    Y[idx] += DeformConvLdg(B + channel_idx);
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
Status DeformConvAddBiasImpl(cudaStream_t stream, T* Y, const T* B, int64_t N, int64_t M, int64_t out_h, int64_t out_w) {
  int64_t total = N * M * out_h * out_w;
  if (total <= 0) return Status::OK();

  // 1. Prepare divisor
  int64_t out_size = out_h * out_w;

  // 2. Create FastDivMod object (note: ensure int64_t version of DivMod is used here)
  DivMod<int64_t> spatial_div(out_size);
  DivMod<int64_t> channel_div(M);

  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);

  // 3. Pass DivMod objects
  DeformConvAddBiasKernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
      Y,
      B,
      spatial_div,
      channel_div,
      total);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status DeformConvCopyGemmOutputRowMajorToNCHW(
    cudaStream_t stream,
    const T* gemm_output,
    T* Y_g,
    int64_t M,
    int64_t M_per_group,
    int64_t output_image_size,
    int64_t cur_parallel) {
  int64_t total = cur_parallel * M_per_group * output_image_size;
  if (total <= 0) return Status::OK();
  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
  CopyGemmOutputRowMajorToNCHWKernel<T><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
      gemm_output, Y_g, M, M_per_group, output_image_size, cur_parallel);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status DeformConvIm2ColImpl(
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
    return Status::OK();
  }

  const int64_t col_numel = static_cast<int64_t>(C) * kH * kW * parallel_imgs * out_h * out_w;
  const bool use_64bit = (num_kernels > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) ||
                         (col_numel > static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  int blocks = GetGridSize(static_cast<size_t>(num_kernels), kDeformConvThreadsPerBlock);

  auto launch = [&](auto kH_tag, auto kW_tag) {
    constexpr int KH = decltype(kH_tag)::value;
    constexpr int KW = decltype(kW_tag)::value;
    if (use_64bit) {
      DeformableIm2ColKernel<T, int64_t, KH, KW><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
          num_kernels, input, offset, mask, H, W, kH, kW, pad_h, pad_w,
          stride_h, stride_w, dilation_h, dilation_w, C, offset_group,
          DivMod<int64_t>(out_h), DivMod<int64_t>(out_w), DivMod<int64_t>(parallel_imgs),
          DivMod<int64_t>(C / offset_group), use_mask, col_buffer);
    } else {
      DeformableIm2ColKernel<T, int32_t, KH, KW><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
          static_cast<int32_t>(num_kernels), input, offset, mask, H, W, kH, kW, pad_h, pad_w,
          stride_h, stride_w, dilation_h, dilation_w, C, offset_group,
          DivMod<int32_t>(static_cast<int32_t>(out_h)),
          DivMod<int32_t>(static_cast<int32_t>(out_w)),
          DivMod<int32_t>(static_cast<int32_t>(parallel_imgs)),
          DivMod<int32_t>(static_cast<int32_t>(C / offset_group)),
          use_mask, col_buffer);
    }
  };

  if (kH == 1 && kW == 1) {
    launch(DeformConvKSize<1>{}, DeformConvKSize<1>{});
  } else if (kH == 3 && kW == 3) {
    launch(DeformConvKSize<3>{}, DeformConvKSize<3>{});
  } else if (kH == 5 && kW == 5) {
    launch(DeformConvKSize<5>{}, DeformConvKSize<5>{});
  } else {
    launch(DeformConvKSize<-1>{}, DeformConvKSize<-1>{});
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INST_DeformConvIm2ColImpl(T) \
  template Status DeformConvIm2ColImpl<T>(cudaStream_t, const T*, const T*, const T*, T*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool)

INST_DeformConvIm2ColImpl(float);
INST_DeformConvIm2ColImpl(double);
INST_DeformConvIm2ColImpl(half);
INST_DeformConvIm2ColImpl(BFloat16);

template Status DeformConvCopyGemmOutputRowMajorToNCHW<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<double>(cudaStream_t, const double*, double*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvCopyGemmOutputRowMajorToNCHW<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, int64_t, int64_t, int64_t, int64_t);

template Status DeformConvAddBiasImpl<float>(cudaStream_t, float*, const float*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<double>(cudaStream_t, double*, const double*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<half>(cudaStream_t, half*, const half*, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<BFloat16>(cudaStream_t, BFloat16*, const BFloat16*, int64_t, int64_t, int64_t, int64_t);

// Delegate ORT type to CUDA type (e.g. MLFloat16 -> half); avoids repeating three identical specializations.
#define DELEGATE_DEFORM_CONV_IMPL(ORT_T, CUDA_T)                                                                    \
  template <>                                                                                                       \
  Status DeformConvIm2ColImpl<ORT_T>(cudaStream_t stream, const ORT_T* input,                                       \
                                     const ORT_T* offset, const ORT_T* mask, ORT_T* col_buffer,                     \
                                     int64_t parallel_imgs, int64_t C, int64_t H, int64_t W,                        \
                                     int64_t kH, int64_t kW, int64_t out_h, int64_t out_w,                          \
                                     int64_t pad_h, int64_t pad_w, int64_t stride_h, int64_t stride_w,              \
                                     int64_t dilation_h, int64_t dilation_w, int64_t offset_group, bool use_mask) { \
    return DeformConvIm2ColImpl<CUDA_T>(stream, reinterpret_cast<const CUDA_T*>(input),                             \
                                        reinterpret_cast<const CUDA_T*>(offset),                                    \
                                        mask ? reinterpret_cast<const CUDA_T*>(mask) : nullptr,                     \
                                        reinterpret_cast<CUDA_T*>(col_buffer),                                      \
                                        parallel_imgs, C, H, W, kH, kW, out_h, out_w,                               \
                                        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,                   \
                                        offset_group, use_mask);                                                    \
  }                                                                                                                 \
  template <>                                                                                                       \
  Status DeformConvCopyGemmOutputRowMajorToNCHW<ORT_T>(cudaStream_t stream,                                         \
                                                       const ORT_T* gemm_output, ORT_T* Y_g,                        \
                                                       int64_t M, int64_t M_per_group,                              \
                                                       int64_t output_image_size, int64_t cur_parallel) {           \
    return DeformConvCopyGemmOutputRowMajorToNCHW<CUDA_T>(stream,                                                   \
                                                          reinterpret_cast<const CUDA_T*>(gemm_output),             \
                                                          reinterpret_cast<CUDA_T*>(Y_g),                           \
                                                          M, M_per_group, output_image_size, cur_parallel);         \
  }                                                                                                                 \
  template <>                                                                                                       \
  Status DeformConvAddBiasImpl<ORT_T>(cudaStream_t stream, ORT_T * Y, const ORT_T* B,                               \
                                      int64_t N, int64_t M, int64_t out_h, int64_t out_w) {                         \
    return DeformConvAddBiasImpl<CUDA_T>(stream, reinterpret_cast<CUDA_T*>(Y),                                      \
                                         reinterpret_cast<const CUDA_T*>(B), N, M, out_h, out_w);                   \
  }

// BFloat16 is not delegated: ORT's BFloat16 is the same type used in device code (ToCudaType<BFloat16> in
// cuda_common.h), so the explicit instantiations above (INST_DeformConvIm2ColImpl(BFloat16), etc.) suffice.
DELEGATE_DEFORM_CONV_IMPL(MLFloat16, half)

}  // namespace cuda
}  // namespace onnxruntime
