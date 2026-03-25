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

template <typename... Values>
inline bool Needs64BitIndex(Values... values) {
  constexpr int64_t kInt32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
  return ((static_cast<int64_t>(values) > kInt32Max) || ...);
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
template <typename T, typename IndexT, int kH = -1, int kW = -1, bool UseMask = false>
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
    T* __restrict__ data_col) {
  constexpr bool is_fixed = (kH >= 0 && kW >= 0);
  const int64_t h_dim_i64 = is_fixed ? kH : weight_h;
  const int64_t w_dim_i64 = is_fixed ? kW : weight_w;
  const IndexT h_dim = static_cast<IndexT>(h_dim_i64);
  const IndexT w_dim = static_cast<IndexT>(w_dim_i64);

  // Reconstruct dimensions from DivMod objects
  const IndexT out_h = out_h_div.d_;
  const IndexT out_w = out_w_div.d_;
  const IndexT parallel_imgs = parallel_imgs_div.d_;

  const IndexT out_size = out_h * out_w;
  // The stride for data_col is (parallel_imgs * out_h * out_w)
  const IndexT col_stride = parallel_imgs * out_size;
  const int64_t out_size_i64 = static_cast<int64_t>(out_size);
  const int64_t col_stride_i64 = static_cast<int64_t>(col_stride);
  const int64_t channel_hw_i64 = static_cast<int64_t>(height) * static_cast<int64_t>(width);
  const int64_t batch_input_stride_i64 = static_cast<int64_t>(channels) * channel_hw_i64;
  const int64_t offset_group_block_size_i64 = static_cast<int64_t>(2) * h_dim_i64 * w_dim_i64 * out_size_i64;
  const int64_t mask_group_block_size_i64 = h_dim_i64 * w_dim_i64 * out_size_i64;
  const int height_i = static_cast<int>(height);
  const int width_i = static_cast<int>(width);

  using CoordT = typename DeformConvBilinearTraits<T>::ComputeT;

  for (IndexT index = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x; index < num_kernels; index += static_cast<IndexT>(blockDim.x) * gridDim.x) {
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
    const IndexT channel_hw = static_cast<IndexT>(channel_hw_i64);
    const IndexT batch_input_stride = static_cast<IndexT>(batch_input_stride_i64);
    const IndexT input_base = out_b * batch_input_stride + in_c * channel_hw;
    const T* input_ptr = input + static_cast<int64_t>(input_base);

    // 2. Spatial index in the output feature map.
    const IndexT spatial_idx = static_cast<IndexT>(out_y * out_w + out_x);

    // 3. Offset pointer base calculation.
    // Layout: (N, offset_groups, 2*KH*KW, OH, OW)
    // We pre-calculate the pointer to the start of the specific (n, g) block, plus spatial_idx.
    const IndexT offset_group_idx = static_cast<IndexT>(offset_group);
    const IndexT ng = out_b * offset_group_idx + offset_grp;  // n * offset_group + g
    const IndexT offset_group_block_size = static_cast<IndexT>(offset_group_block_size_i64);
    const IndexT offset_base = ng * offset_group_block_size + spatial_idx;
    const T* offset_ptr_base =
        offset + static_cast<int64_t>(offset_base);

    // 4. Mask pointer base calculation (if used).
    // Layout: (N, offset_groups, KH*KW, OH, OW)
    const T* mask_ptr_base = nullptr;
    if constexpr (UseMask) {
      const IndexT mask_group_block_size = static_cast<IndexT>(mask_group_block_size_i64);
      const IndexT mask_base = ng * mask_group_block_size + spatial_idx;
      mask_ptr_base =
          mask + static_cast<int64_t>(mask_base);
    }

    // 5. Output pointer base calculation.
    // data_col Layout: (C * KH * KW, N * OH * OW)
    // The current thread writes to the column `c_col` = (b * OH * OW) + spatial_idx.
    // The starting row for this channel is `in_c * KH * KW`.
    const IndexT c_col = out_b * out_size + spatial_idx;
    const IndexT row_base = static_cast<IndexT>((in_c * h_dim) * w_dim);
    T* data_col_ptr_base =
        data_col + static_cast<int64_t>(row_base) * col_stride_i64 + static_cast<int64_t>(c_col);

    // 6. Pre-calculate invariant coordinate parts.
    // Use float for coordinate math when T is half or BFloat16 to avoid precision loss.
    const CoordT base_h_im = static_cast<CoordT>(out_y * stride_h - pad_h);
    const CoordT base_w_im = static_cast<CoordT>(out_x * stride_w - pad_w);

    auto process_kernel_point = [&](IndexT i, IndexT j) {
      // Keep hot-loop indexing in IndexT to reduce 64-bit integer instruction chains.
      // Safety: launch-side gating uses use_64bit (num_kernels/col_numel) and shared shape validation
      // ((H+1)*W <= INT_MAX), so int32 path has bounded index products here.
      const IndexT kernel_idx = static_cast<IndexT>(i * w_dim + j);
      T mask_val = static_cast<T>(1);
      if constexpr (UseMask) {
        // Access mask using pre-calculated base and stride.
        mask_val = DeformConvLdg(mask_ptr_base + kernel_idx * out_size);
      }

      // Calculate offset pointers relative to the base.
      // The offset tensor stores (y_offset, x_offset) pairs for each kernel weight.
      // Stride between y_offset and x_offset is `out_size`.
      const IndexT offset_offset_idx = static_cast<IndexT>(2 * kernel_idx) * out_size;

      const CoordT offset_h = static_cast<CoordT>(DeformConvLdg(offset_ptr_base + offset_offset_idx));
      const CoordT offset_w = static_cast<CoordT>(DeformConvLdg(offset_ptr_base + offset_offset_idx + out_size));

      const CoordT h_im = base_h_im + static_cast<CoordT>(i * dilation_h) + offset_h;
      const CoordT w_im = base_w_im + static_cast<CoordT>(j * dilation_w) + offset_w;

      // height/width are validated on host (DeformConvValidateAndParse) so int is safe here.
      T val = BilinearInterpolate(input_ptr, height_i, width_i, h_im, w_im);

      // Match CPU path: always interpolate then apply mask to keep branch-free hot loop.
      data_col_ptr_base[kernel_idx * col_stride] = val * mask_val;
    };

    if constexpr (is_fixed) {
      if constexpr (kH * kW <= 9) {
        // For 1x1 and 3x3, fully unroll both loops
#pragma unroll
        for (int i = 0; i < kH; ++i) {
#pragma unroll
          for (int j = 0; j < kW; ++j) {
            process_kernel_point(static_cast<IndexT>(i), static_cast<IndexT>(j));
          }
        }
      } else {
        // For 5x5 (25 iterations), fully unrolling causes register spilling and I-cache thrashing.
        // Unroll only the inner loop to balance loop overhead and instruction footprint.
        for (int i = 0; i < kH; ++i) {
#pragma unroll
          for (int j = 0; j < kW; ++j) {
            process_kernel_point(static_cast<IndexT>(i), static_cast<IndexT>(j));
          }
        }
      }
    } else {
      const IndexT weight_h_idx = static_cast<IndexT>(weight_h);
      const IndexT weight_w_idx = static_cast<IndexT>(weight_w);
      for (IndexT i = 0; i < weight_h_idx; ++i) {
        for (IndexT j = 0; j < weight_w_idx; ++j) {
          process_kernel_point(static_cast<IndexT>(i), static_cast<IndexT>(j));
        }
      }
    }
  }
}

// Bias add: Y[n,m,oh,ow] += B[m]. Layout NCHW.
template <typename T, typename IndexT>
__global__ void DeformConvAddBiasKernel(
    T* Y,
    const T* B,
    DivMod<IndexT> spatial_div,  // For dividing by (H * W)
    DivMod<IndexT> channel_div,  // For dividing by M (channel count)
    IndexT total_elements) {
  for (IndexT idx = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total_elements;
       idx += static_cast<IndexT>(blockDim.x) * gridDim.x) {
    IndexT val = idx;
    IndexT batch_channel_idx, pixel_idx;

    // 1. First decomposition: decompose idx into (batch_channel_idx, pixel_idx)
    // Equivalent to: batch_channel_idx = idx / (H*W); pixel_idx = idx % (H*W);
    spatial_div.divmod(val, batch_channel_idx, pixel_idx);

    // 2. Second decomposition: decompose batch_channel_idx into (batch_idx, channel_idx)
    // Equivalent to: channel_idx = batch_channel_idx % M;
    // We only need channel_idx (i.e. m)
    IndexT batch_idx, channel_idx;
    channel_div.divmod(batch_channel_idx, batch_idx, channel_idx);
    ORT_UNUSED_PARAMETER(batch_idx);  // Only channel_idx is needed

    // channel_idx is what we need (i.e. m)
    Y[idx] += DeformConvLdg(B + channel_idx);
  }
}

// 2D path only when N*M <= max_grid_y. If !Needs64BitIndex(total, out_size, M), then total <= INT32_MAX so
// batch_channel_idx * spatial_size + pixel_idx < total fits int32; use IndexT=int32_t. Otherwise int64_t.
template <typename T, typename IndexT>
__global__ void DeformConvAddBias2DKernel(T* Y, const T* B, IndexT spatial_size, int32_t channels) {
  // blockIdx.y maps to batch_channel_idx (N * M)
  const IndexT batch_channel_idx = static_cast<IndexT>(blockIdx.y);
  const IndexT channel_idx = batch_channel_idx % static_cast<IndexT>(channels);
  T bias_val = DeformConvLdg(B + channel_idx);

  const IndexT pixel_idx =
      static_cast<IndexT>(blockIdx.x) * static_cast<IndexT>(blockDim.x) + static_cast<IndexT>(threadIdx.x);
  if (pixel_idx < spatial_size) {
    Y[batch_channel_idx * spatial_size + pixel_idx] += bias_val;
  }
}

}  // namespace

template <typename T>
Status DeformConvAddBiasImpl(cudaStream_t stream, T* Y, const T* B, int64_t N, int64_t M, int64_t out_h, int64_t out_w, int64_t max_grid_y) {
  int64_t total = N * M * out_h * out_w;
  if (total <= 0) return Status::OK();

  // 1. Prepare divisor
  const int64_t out_size = out_h * out_w;
  const int64_t batch_channels = N * M;
  // For 1D DivMod kernel only: int32 fast path vs int64. Orthogonal to 2D launch (gridDim.y limit).
  const bool use_64bit = Needs64BitIndex(total, out_size, M);

  // Fast 2D launch path: map blockIdx.y to (N*M) to avoid per-thread DivMod in bias add.
  // Keep this path only when max_grid_y is reasonably large (>32); very small y-dimension limits
  // provide too little parallelism in y and often don't justify the extra path/launch logic.
  if (max_grid_y > 32 && batch_channels <= static_cast<int64_t>(max_grid_y)) {
    dim3 block(kDeformConvThreadsPerBlock);
    dim3 grid(static_cast<unsigned int>(GetGridSize(static_cast<size_t>(out_size), block.x)),
              static_cast<unsigned int>(batch_channels));
    const int32_t m_i32 = static_cast<int32_t>(M);
    if (use_64bit) {
      DeformConvAddBias2DKernel<T, int64_t><<<grid, block, 0, stream>>>(Y, B, out_size, m_i32);
    } else {
      DeformConvAddBias2DKernel<T, int32_t><<<grid, block, 0, stream>>>(
          Y, B, static_cast<int32_t>(out_size), m_i32);
    }
    return CUDA_CALL(cudaGetLastError());
  }

  int blocks = GetGridSize(static_cast<size_t>(total), kDeformConvThreadsPerBlock);
  if (use_64bit) {
    // 2. Create FastDivMod object (note: ensure int64_t version of DivMod is used here)
    // 3. Pass DivMod objects
    DeformConvAddBiasKernel<T, int64_t><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
        Y, B,
        DivMod<int64_t>(out_size),
        DivMod<int64_t>(M),
        total);
  } else {
    // 2. Create FastDivMod object
    // 3. Pass DivMod objects
    DeformConvAddBiasKernel<T, int32_t><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
        Y, B,
        DivMod<int32_t>(static_cast<int32_t>(out_size)),
        DivMod<int32_t>(static_cast<int32_t>(M)),
        static_cast<int32_t>(total));
  }
  return CUDA_CALL(cudaGetLastError());
}

// Determine if we need to fall back to 64-bit integer arithmetic in the CUDA kernel.
// 32-bit arithmetic is significantly faster and uses fewer registers.
// We check if any of the intermediate index calculations could exceed INT32_MAX (~2.14 billion).
// The most likely variable to exceed this is `col_numel`:
// col_numel = C * kH * kW * parallel_imgs * out_h * out_w
//
// Examples of when 64-bit fallback is triggered (col_numel > 2,147,483,647):
// - High Resolution (1K): C=256, kH=3, kW=3, parallel_imgs=1, out_h=1024, out_w=1024
//   col_numel = 256 * 3 * 3 * 1 * 1024 * 1024 = 2,415,919,104 (> 2.14B)
// - Large Kernel & Batch: C=128, kH=5, kW=5, parallel_imgs=11, out_h=256, out_w=256
//   col_numel = 128 * 5 * 5 * 11 * 256 * 256 = 2,306,867,200 (> 2.14B)
// - Massive Channels: C=4096, kH=3, kW=3, parallel_imgs=1, out_h=256, out_w=256
//   col_numel = 4096 * 3 * 3 * 1 * 256 * 256 = 2,415,919,104 (> 2.14B)
// - 3D-like Large Kernel: C=512, kH=7, kW=7, parallel_imgs=1, out_h=512, out_w=512
//   col_numel = 512 * 7 * 7 * 1 * 512 * 512 = 6,576,668,672 (> 2.14B)
//
// Example of a safe 32-bit case:
// - Typical ResNet: C=256, kH=3, kW=3, parallel_imgs=32, out_h=128, out_w=128
//   col_numel = 256 * 3 * 3 * 32 * 128 * 128 = 1,207,959,552 (< 2.14B)
//
// In practice, due to the 2GB hard limit on temp memory allocation in GetDeformConvEffectiveMaxTempBytes(),
// col_numel will almost never exceed INT32_MAX without OOMing first.
inline bool CheckDeformConvNeeds64BitIndex(
    int64_t num_kernels, int64_t C, int64_t H, int64_t W, int64_t kH, int64_t kW, int64_t out_h, int64_t out_w,
    int64_t parallel_imgs, int64_t offset_group) {
  const int64_t col_numel = static_cast<int64_t>(C) * kH * kW * parallel_imgs * out_h * out_w;
  const int64_t offset_inner_size = static_cast<int64_t>(2) * kH * kW * out_h * out_w;
  const int64_t mask_inner_size = kH * kW * out_h * out_w;
  const int64_t offset_numel = parallel_imgs * offset_group * offset_inner_size;
  const int64_t mask_numel = parallel_imgs * offset_group * mask_inner_size;
  const int64_t channel_hw = H * W;
  const int64_t batch_input_stride = C * channel_hw;
  const int64_t input_numel = parallel_imgs * batch_input_stride;
  const int64_t out_size = out_h * out_w;
  const int64_t col_stride = parallel_imgs * out_size;
  const int64_t max_col_write_idx = static_cast<int64_t>(kH * kW - 1) * col_stride;

  return Needs64BitIndex(num_kernels, col_numel, offset_inner_size, mask_inner_size, offset_numel, mask_numel,
                         max_col_write_idx, channel_hw, batch_input_stride, input_numel, offset_group);
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

  const bool use_64bit = CheckDeformConvNeeds64BitIndex(num_kernels, C, H, W, kH, kW, out_h, out_w, parallel_imgs, offset_group);

  int blocks = GetGridSize(static_cast<size_t>(num_kernels), kDeformConvThreadsPerBlock);

  auto launch = [&](auto kH_tag, auto kW_tag, auto use_mask_tag) {
    constexpr int KH = decltype(kH_tag)::value;
    constexpr int KW = decltype(kW_tag)::value;
    constexpr bool UseMask = decltype(use_mask_tag)::value;
    if (use_64bit) {
      DeformableIm2ColKernel<T, int64_t, KH, KW, UseMask><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
          num_kernels, input, offset, mask, H, W, kH, kW, pad_h, pad_w,
          stride_h, stride_w, dilation_h, dilation_w, C, offset_group,
          DivMod<int64_t>(out_h), DivMod<int64_t>(out_w), DivMod<int64_t>(parallel_imgs),
          DivMod<int64_t>(C / offset_group), col_buffer);
    } else {
      DeformableIm2ColKernel<T, int32_t, KH, KW, UseMask><<<blocks, kDeformConvThreadsPerBlock, 0, stream>>>(
          static_cast<int32_t>(num_kernels), input, offset, mask, H, W, kH, kW, pad_h, pad_w,
          stride_h, stride_w, dilation_h, dilation_w, C, offset_group,
          DivMod<int32_t>(static_cast<int32_t>(out_h)),
          DivMod<int32_t>(static_cast<int32_t>(out_w)),
          DivMod<int32_t>(static_cast<int32_t>(parallel_imgs)),
          DivMod<int32_t>(static_cast<int32_t>(C / offset_group)),
          col_buffer);
    }
  };

  auto launch_with_mask = [&](auto kH_tag, auto kW_tag) {
    if (use_mask) {
      launch(kH_tag, kW_tag, std::integral_constant<bool, true>{});
    } else {
      launch(kH_tag, kW_tag, std::integral_constant<bool, false>{});
    }
  };

  if (kH == 1 && kW == 1) {
    launch_with_mask(DeformConvKSize<1>{}, DeformConvKSize<1>{});
  } else if (kH == 3 && kW == 3) {
    launch_with_mask(DeformConvKSize<3>{}, DeformConvKSize<3>{});
  } else if (kH == 5 && kW == 5) {
    launch_with_mask(DeformConvKSize<5>{}, DeformConvKSize<5>{});
  } else {
    launch_with_mask(DeformConvKSize<-1>{}, DeformConvKSize<-1>{});
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INST_DeformConvIm2ColImpl(T) \
  template Status DeformConvIm2ColImpl<T>(cudaStream_t, const T*, const T*, const T*, T*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool)

INST_DeformConvIm2ColImpl(float);
INST_DeformConvIm2ColImpl(double);
INST_DeformConvIm2ColImpl(half);
INST_DeformConvIm2ColImpl(BFloat16);

template Status DeformConvAddBiasImpl<float>(cudaStream_t, float*, const float*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<double>(cudaStream_t, double*, const double*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<half>(cudaStream_t, half*, const half*, int64_t, int64_t, int64_t, int64_t, int64_t);
template Status DeformConvAddBiasImpl<BFloat16>(cudaStream_t, BFloat16*, const BFloat16*, int64_t, int64_t, int64_t, int64_t, int64_t);

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
  Status DeformConvAddBiasImpl<ORT_T>(cudaStream_t stream, ORT_T * Y, const ORT_T* B,                               \
                                      int64_t N, int64_t M, int64_t out_h, int64_t out_w, int64_t max_grid_y) {     \
    return DeformConvAddBiasImpl<CUDA_T>(stream, reinterpret_cast<CUDA_T*>(Y),                                      \
                                         reinterpret_cast<const CUDA_T*>(B), N, M, out_h, out_w, max_grid_y);       \
  }

// BFloat16 is not delegated: ORT's BFloat16 is the same type used in device code (ToCudaType<BFloat16> in
// cuda_common.h), so the explicit instantiations above (INST_DeformConvIm2ColImpl(BFloat16), etc.) suffice.
DELEGATE_DEFORM_CONV_IMPL(MLFloat16, half)

}  // namespace cuda
}  // namespace onnxruntime
