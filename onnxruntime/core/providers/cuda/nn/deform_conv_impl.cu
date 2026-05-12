// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CUDA device code for DeformConv: deformable im2col kernel(s) and bias-add kernel.
// Host orchestration and GEMM: `deform_conv.cc` (pipeline described there, aligned with CPU `nn/deform_conv.cc`).
//
// This file corresponds to CPU step (1) on GPU: each thread contributes im2col entries by sampling X with
// bilinear interpolation at offset positions (+ optional mask), instead of CPU's precomputed AoSoA plan + fill.
//
// Reference: torchvision deform_conv2d_kernel.cu, ONNX DeformConv.
//
// ONNX shapes (this EP; batch chunk = parallel_imgs):
//   X     [parallel_imgs, C, H, W]
//   offset[parallel_imgs, offset_group * 2*kH*kW, out_h, out_w]  — per (n, oh, ow), channels are
//         (dy, dx) pairs for kernel taps in order (i=0..kH-1, j=0..kW-1): ch = 2*(i*kW+j) for dy, +1 for dx.
//   mask  [parallel_imgs, offset_group * kH*kW, out_h, out_w]      — optional; ch = i*kW+j.
//   col   row-major [C * kH * kW, parallel_imgs * out_h * out_w]; GEMM uses this as in deform_conv.cc.
//
// Sampling (same as CPU / typical DCN): for output (oh, ow), kernel tap (i, j),
//   h_ref = oh * stride_h - pad_h + i * dilation_h + Δh(oh,ow,i,j)
//   w_ref = ow * stride_w - pad_w + j * dilation_w + Δw(oh,ow,i,j)
// then bilinear sample X at (h_ref, w_ref); multiply by mask if present.

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

inline bool ProductExceedsInt32Max(std::initializer_list<int64_t> factors) {
  constexpr int64_t kInt32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
  int64_t acc = 1;
  for (int64_t v : factors) {
    // DeformConv dimensions are expected to be non-negative after validation.
    // If violated unexpectedly, conservatively force the 64-bit kernel path.
    if (v < 0) return true;
    if (v == 0) return false;
    if (acc > kInt32Max / v) return true;
    acc *= v;
  }
  return false;
}

// __ldg has no overload for BFloat16*; use 16-bit load + FromBits. Other types use __ldg directly.
template <typename T>
__device__ __inline__ T DeformConvLdg(const T* __restrict__ p) {
  return __ldg(p);
}
template <>
__device__ __inline__ BFloat16 DeformConvLdg<BFloat16>(const BFloat16* __restrict__ p) {
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

  __device__ static __inline__ ComputeT Load(const T* __restrict__ p) {
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

  __device__ static __inline__ ComputeT Load(const half* __restrict__ p) {
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

  __device__ static __inline__ ComputeT Load(const BFloat16* __restrict__ p) {
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
//
// Historical note: before switching to branchless masked loads, this workload had the following
// "edge sample" ratio (counts = samples with >=1 OOB neighbor / total bilinear samples).
// The numbers remain useful as boundary-hit context, but no longer imply control-flow divergence.
// Example workload only; not a benchmark or representative ratio.
//   kernel 1x1: 1.3746%  (2421 / 176128)
//   kernel 3x3: 1.4833%  (11756 / 792576)
//   kernel 7x7: 4.7593%  (52537 / 1103872)
// Current implementation always issues safe-address loads and masks invalid neighbors to zero.
// Offsets are often spatially smooth, so nearby threads still tend to exhibit similar validity patterns.
template <typename T>
__device__ __inline__ T BilinearInterpolate(
    const T* __restrict__ in,
    int height,
    int width,
    typename DeformConvBilinearTraits<T>::ComputeT h,
    typename DeformConvBilinearTraits<T>::ComputeT w) {
  using Traits = DeformConvBilinearTraits<T>;
  using CoordT = typename Traits::ComputeT;

  // [Optimization 1]: Early exit for clearly out-of-bounds (skip floor() and neighbor loads for OOB case).
  // Semantics guardrail: if sample point is outside [-1, H) x [-1, W), ONNX bilinear contribution is exactly 0.
  // Why keep this even with branchless masked loads below:
  //   - The branchless path guarantees safe addressing and correct masked zero, but still pays floor/weight math
  //     and four global loads.
  //   - This early return avoids all of that work for clearly OOB samples.
  // About divergence: mixed in/out-of-bound warps can diverge here, but OOB lanes terminate immediately while
  // in-bound lanes continue useful work; in practice this often wins unless OOB distribution is highly random
  // and branch hit-rate is very high.
  if (h <= static_cast<CoordT>(-1) || h >= height || w <= static_cast<CoordT>(-1) || w >= width) {
    return Traits::Zero();
  }

  // [Optimization 2]: Keep floor result in CoordT; cast to int only for indices. Avoids float->int->float in lh/lw.
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

  // [Optimization 3]: Branchless neighbor loads via "safe address + one-sided clamp".
  // Given the early return above, coordinates are in (-1, H) x (-1, W), so each index only needs one-sided clamp:
  //   h_low in [-1, H-1], h_high in [0, H], w_low in [-1, W-1], w_high in [0, W].
  // We always load from legal addresses; validity is applied by 2D neighbor masks below.
  // CUDA compilers usually lower this to predicated/selp-style code without control-flow branches.
  const int safe_h_low = max(0, h_low);
  const int safe_h_high = min(h_high, height - 1);
  const int safe_w_low = max(0, w_low);
  const int safe_w_high = min(w_high, width - 1);

  // [Optimization 4]: One-sided validity checks under the same invariant.
  // Keep 2D neighbor masks (m1..m4), algebraically equivalent to masking invalid neighbor terms to zero.
  // Use one/zero ternaries directly in CoordT to encourage selp.f32/f16 generation.
  const CoordT one = static_cast<CoordT>(1);
  const CoordT zero = static_cast<CoordT>(0);
  const CoordT m1 = (h_low >= 0 && w_low >= 0) ? one : zero;
  const CoordT m2 = (h_low >= 0 && w_high < width) ? one : zero;
  const CoordT m3 = (h_high < height && w_low >= 0) ? one : zero;
  const CoordT m4 = (h_high < height && w_high < width) ? one : zero;

  const int safe_base_low = safe_h_low * width;
  const int safe_base_high = safe_h_high * width;

  const CoordT v1 = Traits::Load(in + safe_base_low + safe_w_low) * m1;
  const CoordT v2 = Traits::Load(in + safe_base_low + safe_w_high) * m2;
  const CoordT v3 = Traits::Load(in + safe_base_high + safe_w_low) * m3;
  const CoordT v4 = Traits::Load(in + safe_base_high + safe_w_high) * m4;

  // [Optimization 5]: Factor bilinear into horizontal blends on two rows, then vertical blend.
  // Algebraically equivalent to w1*v1 + w2*v2 + w3*v3 + w4*v4 with w1..w4 from hh/hw/lh/lw;
  // this form tends to produce fewer independent multiplies and friendlier FFMA scheduling.
  CoordT top = hw * v1 + lw * v2;
  CoordT bottom = hw * v3 + lw * v4;
  return Traits::ToResult(hh * top + lh * bottom);
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
  // Aliasing contract for this kernel:
  // - input/offset/mask are read-only and may alias each other,
  // - data_col is write-only and must not overlap any input buffer.
  constexpr bool is_fixed = (kH >= 0 && kW >= 0);
  const int64_t h_dim_i64 = is_fixed ? kH : weight_h;
  const int64_t w_dim_i64 = is_fixed ? kW : weight_w;
  const IndexT h_dim = static_cast<IndexT>(h_dim_i64);
  const IndexT w_dim = static_cast<IndexT>(w_dim_i64);

  // Linear thread index `index` encodes (in_c, out_b, out_y, out_x) with x fastest:
  //   index = out_x + out_w * (out_y + out_h * (out_b + parallel_imgs * in_c))
  // Unroll: divmod by out_w -> out_x; by out_h -> out_y; by parallel_imgs -> out_b, in_c.
  const IndexT out_h = out_h_div.d_;
  const IndexT out_w = out_w_div.d_;
  const IndexT parallel_imgs = parallel_imgs_div.d_;

  const IndexT out_size = out_h * out_w;
  // The stride for data_col is (parallel_imgs * out_h * out_w)
  const IndexT col_stride = parallel_imgs * out_size;  // columns span one spatial map per image in the chunk
  const int64_t out_size_i64 = static_cast<int64_t>(out_size);
  const int64_t col_stride_i64 = static_cast<int64_t>(col_stride);
  const int64_t channel_hw_i64 = static_cast<int64_t>(height) * static_cast<int64_t>(width);
  const int64_t batch_input_stride_i64 = static_cast<int64_t>(channels) * channel_hw_i64;
  // One (n, offset_group g) slice of `offset` in linear memory: 2*kH*kW planes of shape (out_h, out_w).
  const int64_t offset_group_block_size_i64 = static_cast<int64_t>(2) * h_dim_i64 * w_dim_i64 * out_size_i64;
  // Same for `mask`: kH*kW planes of (out_h, out_w).
  [[maybe_unused]] const int64_t mask_group_block_size_i64 = UseMask ? (h_dim_i64 * w_dim_i64 * out_size_i64) : int64_t{0};
  const int height_i = static_cast<int>(height);
  const int width_i = static_cast<int>(width);

  using CoordT = typename DeformConvBilinearTraits<T>::ComputeT;

  for (IndexT index = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x; index < num_kernels; index += static_cast<IndexT>(blockDim.x) * gridDim.x) {
    IndexT val = index;
    IndexT out_x, out_y, out_b, in_c;

    out_w_div.divmod(val, val, out_x);
    out_h_div.divmod(val, val, out_y);
    parallel_imgs_div.divmod(val, in_c, out_b);

    // [Im2Col] offset_group==1: channel_per_offset_grp_div is unused; skip divmod.
    IndexT offset_grp = 0;
    if (offset_group > 1) {
      IndexT dummy;
      channel_per_offset_grp_div.divmod(in_c, offset_grp, dummy);
    }

    // [Im2Col] CSE: base pointers for this thread (one output pixel × input channel).

    // 1. Input X: NCHW; offset to (out_b, in_c) is out_b * (C*H*W) + in_c * (H*W).
    const IndexT channel_hw = static_cast<IndexT>(channel_hw_i64);
    const IndexT batch_input_stride = static_cast<IndexT>(batch_input_stride_i64);
    const IndexT input_base = out_b * batch_input_stride + in_c * channel_hw;
    const T* __restrict__ input_ptr = input + static_cast<int64_t>(input_base);

    // 2. Spatial index in the output feature map.
    const IndexT spatial_idx = static_cast<IndexT>(out_y * out_w + out_x);

    // 3. Offset: linear index to (dy,dx) channel 0 at (out_y, out_x) for image out_b, deformable group offset_grp.
    //   ng = out_b * offset_group + offset_grp
    //   offset_base = ng * (2*kH*kW*out_h*out_w) + (out_y*out_w + out_x)
    const IndexT offset_group_idx = static_cast<IndexT>(offset_group);
    const IndexT ng = out_b * offset_group_idx + offset_grp;
    const IndexT offset_group_block_size = static_cast<IndexT>(offset_group_block_size_i64);
    const IndexT offset_base = ng * offset_group_block_size + spatial_idx;
    const T* __restrict__ offset_ptr_base = offset + static_cast<int64_t>(offset_base);

    // 4. Mask: same as offset but kH*kW planes: mask_base = ng * (kH*kW*out_h*out_w) + spatial_idx.
    const T* __restrict__ mask_ptr_base = nullptr;
    if constexpr (UseMask) {
      const IndexT mask_group_block_size = static_cast<IndexT>(mask_group_block_size_i64);
      const IndexT mask_base = ng * mask_group_block_size + spatial_idx;
      mask_ptr_base =
          mask + static_cast<int64_t>(mask_base);
    }

    // 5. col_buffer row-major: row r = in_c * (kH*kW) + kernel_flat; column c_col = out_b * out_h*out_w + spatial_idx.
    //    Element (r, c_col) at col_buffer[r * col_stride + c_col].
    const IndexT c_col = out_b * out_size + spatial_idx;
    const IndexT row_base = static_cast<IndexT>((in_c * h_dim) * w_dim);
    T* __restrict__ data_col_ptr_base =
        data_col + static_cast<int64_t>(row_base) * col_stride_i64 + static_cast<int64_t>(c_col);

    // 6. Undilated top-left of the kernel anchor for this output pixel: base_* = out_* * stride_* - pad_*.
    //    Row i / col j add i*dilation_h / j*dilation_w before applying offsets (see run_deform_row).
    const CoordT base_h_im = static_cast<CoordT>(out_y * stride_h - pad_h);
    const CoordT base_w_im = static_cast<CoordT>(out_x * stride_w - pad_w);

    // Per (output location, channel): one sample from offset/mask tensors and bilinear input.
    auto process_kernel_point = [&](const T* __restrict__ offset_h_ptr, const T* __restrict__ offset_w_ptr,
                                    const T* __restrict__ mask_ptr, T* __restrict__ data_col_ptr, CoordT h_base,
                                    CoordT w_base) {
      T mask_val = static_cast<T>(1);
      if constexpr (UseMask) {
        mask_val = DeformConvLdg(mask_ptr);
      }

      const CoordT offset_h = static_cast<CoordT>(DeformConvLdg(offset_h_ptr));
      const CoordT offset_w = static_cast<CoordT>(DeformConvLdg(offset_w_ptr));

      const CoordT h_im = h_base + offset_h;
      const CoordT w_im = w_base + offset_w;

      // height/width are validated on host (DeformConvValidateAndParse) so int is safe here.
      T val = BilinearInterpolate(input_ptr, height_i, width_i, h_im, w_im);

      // Match CPU path: always interpolate then apply mask to keep branch-free hot loop.
      *data_col_ptr = val * mask_val;
    };

    // One row of kernel weights (fixed kW or runtime weight_w): compute row base once, then walk j with pointer
    // adds only (no kernel_idx * stride rebuild each j). Shared by compile-time and dynamic kernel sizes.
    // Along the kernel row, dy/dx planes are spaced by out_h*out_w; each (dy,dx) pair spans 2*out_size elements.
    const IndexT offset_pair_stride = static_cast<IndexT>(2) * out_size;
    auto run_deform_row = [&](IndexT row_kernel_base, CoordT h_base, IndexT row_width) {
      CoordT w_base = base_w_im;
      const IndexT offset_elem_offset = static_cast<IndexT>(2 * row_kernel_base) * out_size;
      const T* __restrict__ offset_h_ptr = offset_ptr_base + offset_elem_offset;
      const T* __restrict__ offset_w_ptr = offset_h_ptr + out_size;
      const T* __restrict__ mask_ptr = nullptr;
      if constexpr (UseMask) {
        mask_ptr = mask_ptr_base + row_kernel_base * out_size;
      }
      T* __restrict__ data_col_ptr = data_col_ptr_base + row_kernel_base * col_stride;

      auto step_kernel_point = [&]() {
        process_kernel_point(offset_h_ptr, offset_w_ptr, mask_ptr, data_col_ptr, h_base, w_base);
        offset_h_ptr += offset_pair_stride;
        offset_w_ptr += offset_pair_stride;
        if constexpr (UseMask) {
          mask_ptr += out_size;
        }
        data_col_ptr += col_stride;
        w_base += static_cast<CoordT>(dilation_w);
      };

      // Small fixed kernels: unroll inner j so codegen matches the old fully-unrolled 1x1/3x3 path.
      if constexpr (is_fixed && kH * kW <= 9) {
#pragma unroll
        for (IndexT j = 0; j < row_width; ++j) {
          step_kernel_point();
        }
      } else {
        for (IndexT j = 0; j < row_width; ++j) {
          step_kernel_point();
        }
      }
    };

    if constexpr (is_fixed) {
      if constexpr (kH * kW <= 9) {
        // For 1x1 and 3x3, unroll the outer i loop; inner j uses run_deform_row with #pragma unroll there.
#pragma unroll
        for (int i = 0; i < kH; ++i) {
          const IndexT i_idx = static_cast<IndexT>(i);
          run_deform_row(i_idx * w_dim, base_h_im + static_cast<CoordT>(i_idx * dilation_h), w_dim);
        }
      } else {
        // Larger fixed kernels (including 7x7): keep both outer i and inner j rolled to limit register
        // pressure from the heavy bilinear body. 7x7 still benefits from launch-time kH/kW constants
        // without inner #pragma unroll.
        for (int i = 0; i < kH; ++i) {
          const IndexT i_idx = static_cast<IndexT>(i);
          run_deform_row(i_idx * w_dim, base_h_im + static_cast<CoordT>(i_idx * dilation_h), w_dim);
        }
      }
    } else {
      const IndexT weight_h_idx = static_cast<IndexT>(weight_h);
      const IndexT weight_w_idx = static_cast<IndexT>(weight_w);
      for (IndexT i = 0; i < weight_h_idx; ++i) {
        const IndexT row_base_idx = static_cast<IndexT>(i * weight_w_idx);
        run_deform_row(row_base_idx, base_h_im + static_cast<CoordT>(i * dilation_h), weight_w_idx);
      }
    }
  }
}

// Bias add: Y[n,m,oh,ow] += B[m]. Y linear row-major NCHW: idx = n*(M*HW) + m*HW + (oh*W+ow).
template <typename T, typename IndexT>
__global__ void DeformConvAddBiasKernel(
    T* __restrict__ Y,
    const T* __restrict__ B,
    DivMod<IndexT> spatial_div,  // For dividing by (H * W)
    DivMod<IndexT> channel_div,  // For dividing by M (channel count)
    IndexT total_elements) {
  for (IndexT idx = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total_elements;
       idx += static_cast<IndexT>(blockDim.x) * gridDim.x) {
    IndexT val = idx;
    IndexT batch_channel_idx, pixel_idx;

    // idx -> (batch_channel_idx, pixel_idx) with pixel_idx = oh*out_w+ow fastest.
    spatial_div.divmod(val, batch_channel_idx, pixel_idx);

    // batch_channel_idx = n*M + m  ->  bias index is m = batch_channel_idx % M.
    IndexT batch_idx, channel_idx;
    channel_div.divmod(batch_channel_idx, batch_idx, channel_idx);
    ORT_UNUSED_PARAMETER(batch_idx);

    Y[idx] += DeformConvLdg(B + channel_idx);
  }
}

// 2D launch: blockIdx.y -> batch_channel_idx in [0, N*M), threadIdx -> pixel_idx in [0, out_h*out_w).
// Indexing: Y[batch_channel_idx * spatial_size + pixel_idx]. Pick IndexT from Needs64BitIndex like the 1D kernel.
template <typename T, typename IndexT>
__global__ void DeformConvAddBias2DKernel(T* __restrict__ Y, const T* __restrict__ B, IndexT spatial_size,
                                          int32_t channels) {
  // blockIdx.y maps to batch_channel_idx (N * M)
  const IndexT batch_channel_idx = static_cast<IndexT>(blockIdx.y);
  const IndexT channel_idx = batch_channel_idx % static_cast<IndexT>(channels);
  T bias_val = DeformConvLdg(B + channel_idx);

  const IndexT pixel_idx = static_cast<IndexT>(blockIdx.x) * static_cast<IndexT>(blockDim.x) + static_cast<IndexT>(threadIdx.x);
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
  // Use it only when the device allows enough grid rows: below ~32 blocks in y, the extra
  // parallelism (warps scheduled across blockIdx.y) is often too small to outweigh maintaining
  // a second launch + kernel variant; the threshold is a heuristic—revisit if future GPUs change
  // occupancy sweet spots or typical batch×channel counts.
  constexpr int kMinGridYForBias2DPath = 32;
  if (max_grid_y > kMinGridYForBias2DPath && batch_channels <= static_cast<int64_t>(max_grid_y)) {
    dim3 block(kDeformConvThreadsPerBlock);
    dim3 grid(static_cast<unsigned int>(GetGridSize(static_cast<size_t>(out_size), block.x)),
              static_cast<unsigned int>(batch_channels));
    const int32_t m_i32 = static_cast<int32_t>(M);
    if (use_64bit) {
      DeformConvAddBias2DKernel<T, int64_t><<<grid, block, 0, stream>>>(Y, B, out_size, m_i32);
    } else {
      DeformConvAddBias2DKernel<T, int32_t><<<grid, block, 0, stream>>>(Y, B, static_cast<int32_t>(out_size), m_i32);
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
  if (Needs64BitIndex(num_kernels, C, H, W, kH, kW, out_h, out_w, parallel_imgs, offset_group)) {
    return true;
  }

  // Check potentially large products without evaluating intermediate multiplications.
  return ProductExceedsInt32Max({C, kH, kW, parallel_imgs, out_h, out_w}) ||                // col_numel
         ProductExceedsInt32Max({2, kH, kW, out_h, out_w}) ||                               // offset_inner_size
         ProductExceedsInt32Max({kH, kW, out_h, out_w}) ||                                  // mask_inner_size
         ProductExceedsInt32Max({parallel_imgs, offset_group, 2, kH, kW, out_h, out_w}) ||  // offset_numel
         ProductExceedsInt32Max({parallel_imgs, offset_group, kH, kW, out_h, out_w}) ||     // mask_numel
         ProductExceedsInt32Max({H, W}) ||                                                  // channel_hw
         ProductExceedsInt32Max({C, H, W}) ||                                               // batch_input_stride
         ProductExceedsInt32Max({parallel_imgs, C, H, W});                                  // input_numel
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

  auto launch_with_mask = [&](auto k_size_tag) {
    if (use_mask) {
      launch(k_size_tag, k_size_tag, std::integral_constant<bool, true>{});
    } else {
      launch(k_size_tag, k_size_tag, std::integral_constant<bool, false>{});
    }
  };

  // Keep template specializations for the most common kernel sizes in modern models.
  // 5x5 is intentionally not specialized: it is less common in current architectures and is often
  // replaced by stacked 3x3 blocks (similar receptive field with better optimization flexibility).
  if (kH == 1 && kW == 1) {
    launch_with_mask(DeformConvKSize<1>{});
  } else if (kH == 3 && kW == 3) {
    launch_with_mask(DeformConvKSize<3>{});
  } else if (kH == 7 && kW == 7) {
    launch_with_mask(DeformConvKSize<7>{});
  } else {
    launch_with_mask(DeformConvKSize<-1>{});
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
