// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CPU implementation of DeformConv (deformable convolution 2D).

#include "deform_conv.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/common/common.h"
#include "core/util/math_cpuonly.h"
#include "core/common/narrow.h"
#include "core/util/force_inline.h"
#include "core/util/math.h"

namespace onnxruntime {

namespace {
constexpr uint8_t kTopLeftMask = 1u << 0;
constexpr uint8_t kTopRightMask = 1u << 1;
constexpr uint8_t kBottomLeftMask = 1u << 2;
constexpr uint8_t kBottomRightMask = 1u << 3;
constexpr uint8_t kAllNeighborsMask = kTopLeftMask | kTopRightMask | kBottomLeftMask | kBottomRightMask;
constexpr int kTile = 256;
constexpr size_t kMaxSamplingPlanWorkspaceBytes = 256ull * 1024ull * 1024ull;
constexpr int64_t kStreamingSpatialTileSize = 4096;

namespace sampling_plan_internal {

template <typename T>
struct BilinearSamplePlanArrays;

constexpr size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

inline void AddAlignedRegion(size_t alignment, size_t bytes, size_t max_size_t, size_t& bytes_total) {
  bytes_total = AlignUp(bytes_total, alignment);
  ORT_ENFORCE(bytes_total <= max_size_t - bytes, "Sampling plan byte size overflows size_t.");
  bytes_total += bytes;
}

template <typename T>
size_t ComputeSamplingPlanWorkspaceBytes(size_t plan_size, size_t max_size_t,
                                         size_t& idx_bytes, size_t& weight_bytes, size_t& flag_bytes) {
  ORT_ENFORCE(plan_size <= max_size_t / sizeof(int32_t), "Sampling plan idx bytes overflows size_t.");
  ORT_ENFORCE(plan_size <= max_size_t / sizeof(T), "Sampling plan weight bytes overflows size_t.");
  ORT_ENFORCE(plan_size <= max_size_t / sizeof(uint8_t), "Sampling plan flag bytes overflows size_t.");
  idx_bytes = plan_size * sizeof(int32_t);
  weight_bytes = plan_size * sizeof(T);
  flag_bytes = plan_size * sizeof(uint8_t);

  size_t bytes_total = 0;
  for (int i = 0; i < 4; ++i) {
    AddAlignedRegion(alignof(int32_t), idx_bytes, max_size_t, bytes_total);
  }
  for (int i = 0; i < 4; ++i) {
    AddAlignedRegion(alignof(T), weight_bytes, max_size_t, bytes_total);
  }
  AddAlignedRegion(alignof(uint8_t), flag_bytes, max_size_t, bytes_total);
  return bytes_total;
}

template <typename T>
void InitializeSamplingPlanViewsFromBuffer(uint8_t* plan_base,
                                           size_t bytes_total,
                                           size_t idx_bytes,
                                           size_t weight_bytes,
                                           size_t flag_bytes,
                                           BilinearSamplePlanArrays<T>& sampling_plan) {
  size_t cursor = 0;
  auto take_region = [&](size_t alignment, size_t bytes) -> uint8_t* {
    cursor = AlignUp(cursor, alignment);
    uint8_t* ptr = plan_base + cursor;
    cursor += bytes;
    return ptr;
  };

  sampling_plan.idx00 = reinterpret_cast<int32_t*>(take_region(alignof(int32_t), idx_bytes));
  sampling_plan.idx01 = reinterpret_cast<int32_t*>(take_region(alignof(int32_t), idx_bytes));
  sampling_plan.idx10 = reinterpret_cast<int32_t*>(take_region(alignof(int32_t), idx_bytes));
  sampling_plan.idx11 = reinterpret_cast<int32_t*>(take_region(alignof(int32_t), idx_bytes));
  sampling_plan.w00 = reinterpret_cast<T*>(take_region(alignof(T), weight_bytes));
  sampling_plan.w01 = reinterpret_cast<T*>(take_region(alignof(T), weight_bytes));
  sampling_plan.w10 = reinterpret_cast<T*>(take_region(alignof(T), weight_bytes));
  sampling_plan.w11 = reinterpret_cast<T*>(take_region(alignof(T), weight_bytes));
  sampling_plan.flags = reinterpret_cast<uint8_t*>(take_region(alignof(uint8_t), flag_bytes));

  ORT_ENFORCE(cursor <= bytes_total, "Sampling plan buffer layout exceeds allocated workspace.");
}

template <typename T>
struct BilinearSamplePlanArrays {
  int32_t* idx00 = nullptr;
  int32_t* idx01 = nullptr;
  int32_t* idx10 = nullptr;
  int32_t* idx11 = nullptr;
  T* w00 = nullptr;
  T* w01 = nullptr;
  T* w10 = nullptr;
  T* w11 = nullptr;
  uint8_t* flags = nullptr;
};

template <bool RangeMode, typename T>
void BuildBilinearSamplingPlanImpl(
    const T* ptr_offset_h,
    const T* ptr_offset_w,
    int height,
    int width,
    int64_t height_col,
    int64_t width_col,
    int64_t stride_h,
    int64_t stride_w,
    T base_h,
    T base_w,
    int64_t spatial_begin,
    int64_t spatial_count,
    BilinearSamplePlanArrays<T>& plan,
    int64_t plan_base) {
  auto process_sample = [&](int64_t spatial_idx, int64_t pidx) {
    const int64_t h_col = spatial_idx / width_col;
    const int64_t w_col = spatial_idx - h_col * width_col;

    const T h_im = h_col * stride_h + base_h + ptr_offset_h[spatial_idx];
    const T w_im = w_col * stride_w + base_w + ptr_offset_w[spatial_idx];

    if (h_im <= static_cast<T>(-1) || h_im >= height || w_im <= static_cast<T>(-1) || w_im >= width) {
      plan.idx00[pidx] = plan.idx01[pidx] = plan.idx10[pidx] = plan.idx11[pidx] = 0;
      plan.w00[pidx] = plan.w01[pidx] = plan.w10[pidx] = plan.w11[pidx] = static_cast<T>(0);
      plan.flags[pidx] = 0;
      return;
    }

    const T h_floor = std::floor(h_im);
    const T w_floor = std::floor(w_im);
    const int h_low = static_cast<int>(h_floor);
    const int w_low = static_cast<int>(w_floor);
    const int h_high = h_low + 1;
    const int w_high = w_low + 1;

    const T lh = h_im - h_floor;
    const T lw = w_im - w_floor;
    const T hh = static_cast<T>(1) - lh;
    const T hw = static_cast<T>(1) - lw;

    plan.w00[pidx] = hh * hw;
    plan.w01[pidx] = hh * lw;
    plan.w10[pidx] = lh * hw;
    plan.w11[pidx] = lh * lw;

    uint8_t mask = 0;
    if (static_cast<unsigned>(h_low) < static_cast<unsigned>(height - 1) &&
        static_cast<unsigned>(w_low) < static_cast<unsigned>(width - 1)) {
      mask = kAllNeighborsMask;
    } else {
      if (h_low >= 0 && w_low >= 0) {
        mask |= kTopLeftMask;
      }
      if (h_low >= 0 && w_high < width) {
        mask |= kTopRightMask;
      }
      if (h_high < height && w_low >= 0) {
        mask |= kBottomLeftMask;
      }
      if (h_high < height && w_high < width) {
        mask |= kBottomRightMask;
      }
    }

    const int base_low = h_low * width;
    const int base_high = h_high * width;
    plan.idx00[pidx] = base_low + w_low;
    plan.idx01[pidx] = base_low + w_high;
    plan.idx10[pidx] = base_high + w_low;
    plan.idx11[pidx] = base_high + w_high;
    plan.flags[pidx] = mask;
  };

  if constexpr (RangeMode) {
    for (int64_t local_idx = 0; local_idx < spatial_count; ++local_idx) {
      process_sample(spatial_begin + local_idx, plan_base + local_idx);
    }
  } else {
    for (int64_t h_tile_begin = 0; h_tile_begin < height_col; h_tile_begin += kTile) {
      const int64_t h_tile_end = std::min<int64_t>(h_tile_begin + kTile, height_col);
      for (int64_t h_col = h_tile_begin; h_col < h_tile_end; ++h_col) {
        for (int64_t w_col = 0; w_col < width_col; ++w_col) {
          const int64_t spatial_idx = h_col * width_col + w_col;
          process_sample(spatial_idx, plan_base + spatial_idx);
        }
      }
    }
  }
}

template <bool RangeMode, typename T>
void BuildAllBilinearSamplingPlansImpl(
    const T* data_offset,
    int height,
    int width,
    int64_t kernel_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t offset_groups,
    int64_t output_size,
    int64_t height_col,
    int64_t width_col,
    int64_t kernel_size,
    int64_t spatial_begin,
    int64_t spatial_count,
    BilinearSamplePlanArrays<T>& sampling_plan,
    concurrency::ThreadPool* thread_pool) {
  const int64_t plan_rows = offset_groups * kernel_size;
  const double plan_parallel_cost = static_cast<double>(RangeMode ? spatial_count : output_size) * 12.0;
  concurrency::ThreadPool::TryParallelFor(
      thread_pool,
      static_cast<std::ptrdiff_t>(plan_rows),
      plan_parallel_cost,
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (ptrdiff_t plan_row = begin; plan_row < end; ++plan_row) {
          const int64_t row = static_cast<int64_t>(plan_row);
          const int64_t offset_grp = row / kernel_size;
          const int64_t kernel_idx = row % kernel_size;
          const int64_t i = kernel_idx / kernel_w;
          const int64_t j = kernel_idx % kernel_w;

          const int64_t offset_base = offset_grp * 2 * kernel_size + 2 * kernel_idx;
          const T* ptr_offset_h = data_offset + offset_base * output_size;
          const T* ptr_offset_w = data_offset + (offset_base + 1) * output_size;
          const T base_h = -pad_h + static_cast<T>(i) * dilation_h;
          const T base_w = -pad_w + static_cast<T>(j) * dilation_w;

          if constexpr (RangeMode) {
            BuildBilinearSamplingPlanImpl<true>(
                ptr_offset_h, ptr_offset_w,
                height, width, 0, width_col,
                stride_h, stride_w, base_h, base_w,
                spatial_begin, spatial_count,
                sampling_plan, row * spatial_count);
          } else {
            BuildBilinearSamplingPlanImpl<false>(
                ptr_offset_h, ptr_offset_w,
                height, width, height_col, width_col,
                stride_h, stride_w, base_h, base_w,
                0, 0, sampling_plan, row * output_size);
          }
        }
      });
}

template <typename T>
ORT_FORCEINLINE T EvalSampleFromPlan(const T* im_ptr,
                                     const BilinearSamplePlanArrays<T>& sampling_plan,
                                     int64_t pidx) {
  const uint8_t flag = sampling_plan.flags[pidx];
  if (flag == kAllNeighborsMask) {
    return sampling_plan.w00[pidx] * im_ptr[sampling_plan.idx00[pidx]] +
           sampling_plan.w01[pidx] * im_ptr[sampling_plan.idx01[pidx]] +
           sampling_plan.w10[pidx] * im_ptr[sampling_plan.idx10[pidx]] +
           sampling_plan.w11[pidx] * im_ptr[sampling_plan.idx11[pidx]];
  }

  T val = static_cast<T>(0);
  if (flag != 0) {
    if (flag & kTopLeftMask) {
      val += sampling_plan.w00[pidx] * im_ptr[sampling_plan.idx00[pidx]];
    }
    if (flag & kTopRightMask) {
      val += sampling_plan.w01[pidx] * im_ptr[sampling_plan.idx01[pidx]];
    }
    if (flag & kBottomLeftMask) {
      val += sampling_plan.w10[pidx] * im_ptr[sampling_plan.idx10[pidx]];
    }
    if (flag & kBottomRightMask) {
      val += sampling_plan.w11[pidx] * im_ptr[sampling_plan.idx11[pidx]];
    }
  }
  return val;
}

template <bool RangeMode, typename T, bool UseMask>
void FillColRowFromSamplingPlanImpl(
    const T* im_ptr,
    const BilinearSamplePlanArrays<T>& sampling_plan,
    int64_t plan_row_base,
    int64_t spatial_begin,
    int64_t spatial_count,
    int64_t mask_row_base,
    const T* ptr_mask,
    T* col_ptr) {
  if constexpr (RangeMode) {
    for (int64_t local_idx = 0; local_idx < spatial_count; ++local_idx) {
      const int64_t pidx = plan_row_base + local_idx;
      T val = EvalSampleFromPlan(im_ptr, sampling_plan, pidx);

      if constexpr (UseMask) {
        val *= ptr_mask[mask_row_base + local_idx];
      }

      col_ptr[spatial_begin + local_idx] = val;
    }
  } else {
    for (int64_t tile_begin = 0; tile_begin < spatial_count; tile_begin += kTile) {
      const int64_t tile_end = std::min<int64_t>(tile_begin + kTile, spatial_count);
      for (int64_t local_idx = tile_begin; local_idx < tile_end; ++local_idx) {
        const int64_t pidx = plan_row_base + local_idx;
        T val = EvalSampleFromPlan(im_ptr, sampling_plan, pidx);

        if constexpr (UseMask) {
          val *= ptr_mask[local_idx];
        }

        col_ptr[local_idx] = val;
      }
    }
  }
}

// Deformable Im2Col for a SINGLE image.
// Streamed=false builds full plan once; Streamed=true builds per spatial tile.
template <bool Streamed, typename T, bool UseMask>
void DeformableIm2colPlanned(
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    int height, int width,
    int64_t kernel_h, int64_t kernel_w,
    int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t channels,
    int64_t offset_groups,
    int64_t height_col, int64_t width_col,
    BilinearSamplePlanArrays<T>& sampling_plan,
    int64_t spatial_tile_size,
    T* data_col,
    concurrency::ThreadPool* thread_pool) {
  const int64_t channel_per_offset_group = channels / offset_groups;
  const int64_t kernel_size = kernel_h * kernel_w;
  const int64_t output_size = height_col * width_col;

  const int64_t effective_tile = Streamed ? spatial_tile_size : output_size;
  for (int64_t spatial_begin = 0; spatial_begin < output_size; spatial_begin += effective_tile) {
    const int64_t spatial_count = std::min<int64_t>(effective_tile, output_size - spatial_begin);
    if constexpr (Streamed) {
      BuildAllBilinearSamplingPlansImpl<true>(
          data_offset, height, width,
          kernel_w, pad_h, pad_w,
          stride_h, stride_w,
          dilation_h, dilation_w,
          offset_groups,
          output_size, height_col, width_col, kernel_size,
          spatial_begin, spatial_count,
          sampling_plan, thread_pool);
    } else {
      BuildAllBilinearSamplingPlansImpl<false>(
          data_offset, height, width,
          kernel_w, pad_h, pad_w,
          stride_h, stride_w,
          dilation_h, dilation_w,
          offset_groups,
          output_size, height_col, width_col, kernel_size,
          0, 0,
          sampling_plan, thread_pool);
    }

    const double parallel_cost = static_cast<double>(spatial_count) * (UseMask ? 12.0 : 10.0);
    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        static_cast<std::ptrdiff_t>(channels * kernel_size),
        parallel_cost,
        [&](ptrdiff_t begin, ptrdiff_t end) {
          for (ptrdiff_t idx = begin; idx < end; ++idx) {
            const int64_t j = static_cast<int64_t>(idx) % kernel_w;
            const int64_t i = (static_cast<int64_t>(idx) / kernel_w) % kernel_h;
            const int64_t c_im = static_cast<int64_t>(idx) / kernel_size;
            const int64_t offset_grp = c_im / channel_per_offset_group;

            T* col_ptr = data_col + static_cast<int64_t>(idx) * output_size;
            const T* im_ptr = data_im + c_im * static_cast<int64_t>(height) * width;
            const int64_t plan_stride = Streamed ? spatial_count : output_size;
            const int64_t plan_row_base = (offset_grp * kernel_size + (i * kernel_w + j)) * plan_stride;
            [[maybe_unused]] const T* ptr_mask = nullptr;
            const int64_t mask_row_base = (offset_grp * kernel_size + i * kernel_w + j) * output_size + spatial_begin;
            if constexpr (UseMask) {
              ptr_mask = data_mask;
            }

            if constexpr (Streamed) {
              FillColRowFromSamplingPlanImpl<true, T, UseMask>(
                  im_ptr, sampling_plan, plan_row_base,
                  spatial_begin, spatial_count, mask_row_base, ptr_mask, col_ptr);
            } else {
              FillColRowFromSamplingPlanImpl<false, T, UseMask>(
                  im_ptr, sampling_plan, plan_row_base,
                  0, output_size, 0, ptr_mask, col_ptr);
            }
          }
        });

    if constexpr (!Streamed) {
      break;
    }
  }
}

template <bool Streamed, typename T>
ORT_FORCEINLINE void DispatchDeformableIm2colPlanned(
    bool use_mask,
    const T* X_curr,
    const T* offset_curr,
    const T* mask_curr,
    int H,
    int W_in,
    int64_t kH,
    int64_t kW,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t C,
    int64_t offset_group,
    int64_t out_h,
    int64_t out_w,
    BilinearSamplePlanArrays<T>& sampling_plan,
    int64_t tile_size,
    T* col_buffer_ptr,
    concurrency::ThreadPool* thread_pool) {
  if (use_mask) {
    DeformableIm2colPlanned<Streamed, T, true>(
        X_curr, offset_curr, mask_curr,
        H, W_in, kH, kW,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        C, offset_group, out_h, out_w, sampling_plan, tile_size,
        col_buffer_ptr, thread_pool);
  } else {
    DeformableIm2colPlanned<Streamed, T, false>(
        X_curr, offset_curr, nullptr,
        H, W_in, kH, kW,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
        C, offset_group, out_h, out_w, sampling_plan, tile_size,
        col_buffer_ptr, thread_pool);
  }
}

}  // namespace sampling_plan_internal

}  // namespace

template <typename T>
Status DeformConv<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto* offset = context->Input<Tensor>(2);
  const auto* B = context->Input<Tensor>(3);     // optional
  const auto* mask = context->Input<Tensor>(4);  // optional

  DeformConvParams params;
  ORT_RETURN_IF_ERROR(DeformConvValidateAndParse(
      attrs_,
      X->Shape(),
      W->Shape(),
      offset->Shape(),
      B ? &B->Shape() : nullptr,
      mask ? &mask->Shape() : nullptr,
      params));

  const int64_t N = params.N;
  const int64_t C = params.C;
  const int64_t H = params.H;
  const int64_t W_in = params.W_in;
  const int64_t M = params.M;
  const int64_t kH = params.kH;
  const int64_t kW = params.kW;
  const int64_t pad_h = params.pad_h;
  const int64_t pad_w = params.pad_w;
  const int64_t stride_h = params.stride_h;
  const int64_t stride_w = params.stride_w;
  const int64_t dilation_h = params.dilation_h;
  const int64_t dilation_w = params.dilation_w;
  const int64_t group = params.group;
  const int64_t offset_group = params.offset_group;
  const int64_t out_h = params.out_h;
  const int64_t out_w = params.out_w;
  const bool use_mask = params.use_mask;

  // Allocate output tensor [N, M, out_h, out_w].
  const TensorShape Y_shape({N, M, out_h, out_w});
  Tensor* Y = context->Output(0, Y_shape);
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  // Precompute common sizes for the im2col + GEMM pipeline.
  const int64_t kernel_size = kH * kW;
  const int64_t output_image_size = out_h * out_w;
  const int64_t input_image_size = H * W_in;
  const int64_t kernel_dim = C / group * kernel_size;  // K dimension for GEMM: C/group * kH * kW
  const int64_t plan_rows = offset_group * kernel_size;

  ORT_ENFORCE(plan_rows >= 0 && output_image_size >= 0, "Invalid plan dimensions.");
  const size_t max_size_t = std::numeric_limits<size_t>::max();
  ORT_ENFORCE(plan_rows == 0 || static_cast<uint64_t>(output_image_size) <= (max_size_t / static_cast<size_t>(plan_rows)),
              "Sampling plan size overflows size_t.");
  const size_t plan_size = static_cast<size_t>(plan_rows) * static_cast<size_t>(output_image_size);

  // Col buffer: shape [C*kH*kW, out_h*out_w]. Allocate per-image (process one image at a time)
  // to reduce peak memory when N is large; im2col is implemented per-image anyway.
  const int64_t col_buffer_size = (C * kernel_size) * output_image_size;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  auto col_buffer = IAllocator::MakeUniquePtr<T>(alloc, SafeInt<size_t>(col_buffer_size));
  size_t idx_bytes = 0;
  size_t weight_bytes = 0;
  size_t flag_bytes = 0;
  const size_t bytes_total_full = sampling_plan_internal::ComputeSamplingPlanWorkspaceBytes<T>(
      plan_size, max_size_t, idx_bytes, weight_bytes, flag_bytes);
  const bool use_streamed_plan = bytes_total_full > kMaxSamplingPlanWorkspaceBytes;
  const int64_t spatial_tile_size = std::min<int64_t>(kStreamingSpatialTileSize, output_image_size);

  ORT_ENFORCE(static_cast<uint64_t>(H) * static_cast<uint64_t>(W_in) <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
              "DeformConv requires H*W to fit in int for sampling indices.");

  const size_t plan_size_for_alloc = use_streamed_plan
                                         ? static_cast<size_t>(plan_rows) * static_cast<size_t>(spatial_tile_size)
                                         : plan_size;
  const size_t bytes_total = sampling_plan_internal::ComputeSamplingPlanWorkspaceBytes<T>(
      plan_size_for_alloc, max_size_t, idx_bytes, weight_bytes, flag_bytes);
  auto plan_buffer = IAllocator::MakeUniquePtr<uint8_t>(alloc, SafeInt<size_t>(bytes_total));
  sampling_plan_internal::BilinearSamplePlanArrays<T> sampling_plan{};
  sampling_plan_internal::InitializeSamplingPlanViewsFromBuffer<T>(
      plan_buffer.get(), bytes_total, idx_bytes, weight_bytes, flag_bytes, sampling_plan);

  const T* Xdata = X->Data<T>();
  const T* Wdata = W->Data<T>();
  const T* offset_data = offset->Data<T>();
  const T* mask_data = use_mask ? mask->Data<T>() : nullptr;
  T* Ydata = Y->MutableData<T>();
  const T* Bdata = (B != nullptr) ? B->Data<T>() : nullptr;

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
  // Process each image in the batch.
  for (int64_t n = 0; n < N; ++n) {
    // Step 1: Deformable Im2Col for image n.
    // Gather deformed samples into col buffer for GEMM.
    const T* X_curr = Xdata + n * (C * input_image_size);
    const T* offset_curr = offset_data + n * (offset_group * 2 * kernel_size * output_image_size);
    const T* mask_curr = use_mask ? (mask_data + n * (offset_group * kernel_size * output_image_size)) : nullptr;
    T* col_buffer_ptr = col_buffer.get();

    if (use_streamed_plan) {
      sampling_plan_internal::DispatchDeformableIm2colPlanned<true, T>(
          use_mask, X_curr, offset_curr, mask_curr,
          static_cast<int>(H), static_cast<int>(W_in), kH, kW,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          C, offset_group, out_h, out_w, sampling_plan, spatial_tile_size,
          col_buffer_ptr, thread_pool);
    } else {
      sampling_plan_internal::DispatchDeformableIm2colPlanned<false, T>(
          use_mask, X_curr, offset_curr, mask_curr,
          static_cast<int>(H), static_cast<int>(W_in), kH, kW,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          C, offset_group, out_h, out_w, sampling_plan, output_image_size,
          col_buffer_ptr, thread_pool);
    }

    // Step 2: GEMM for each group. Y = W * Col (per group).
    for (int64_t g = 0; g < group; ++g) {
      // Weight for group g: shape [M/group, C/group, kH, kW], row-major.
      const T* weight_g = Wdata + g * (M / group) * kernel_dim;

      // Col rows for group g: layout [C*kH*kW, out_h*out_w], group g spans rows [g*kernel_dim, (g+1)*kernel_dim).
      const T* col_g = col_buffer_ptr + g * kernel_dim * output_image_size;

      // Output slice for group g: [n, g*M/group:(g+1)*M/group, out_h, out_w].
      T* Y_g = Ydata + n * M * output_image_size + g * (M / group) * output_image_size;

      // GEMM: Y = W * Col. W [M/group, kernel_dim], Col [kernel_dim, output_image_size].
      math::Gemm<T>(
          CblasNoTrans,
          CblasNoTrans,
          narrow<ptrdiff_t>(M / group),          // M
          narrow<ptrdiff_t>(output_image_size),  // N
          narrow<ptrdiff_t>(kernel_dim),         // K
          static_cast<T>(1),                     // alpha
          weight_g,                              // A
          col_g,                                 // B
          static_cast<T>(0),                     // beta
          Y_g,                                   // C
          thread_pool,
          nullptr);  // mlas_backend_kernel_selector_config
    }
  }

  // Step 3: Add bias if provided (broadcast over spatial dimensions).
  if (Bdata != nullptr) {
    int64_t total_work = N * M;
    concurrency::ThreadPool::TryParallelFor(
        thread_pool, static_cast<std::ptrdiff_t>(total_work), static_cast<double>(output_image_size),
        [&](ptrdiff_t first, ptrdiff_t last) {
          for (ptrdiff_t idx = first; idx < last; ++idx) {
            int64_t n = idx / M;
            int64_t m = idx % M;
            T* Y_ptr = Ydata + n * M * output_image_size + m * output_image_size;
            // Eigen vectorized add: Y_ptr += Bdata[m] over all spatial positions.
            EigenVectorArrayMap<T>(Y_ptr, narrow<ptrdiff_t>(output_image_size)) += Bdata[m];
          }
        });
  }

  return Status::OK();
}

// Explicit template instantiation for float and double
template class DeformConv<float>;
template class DeformConv<double>;

#define REGISTER_DEFORMCONV_KERNEL_TYPED(T)                       \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DeformConv, 19, 21, T,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DeformConv<T>);                                             \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                 \
      DeformConv, 22, T,                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DeformConv<T>)

REGISTER_DEFORMCONV_KERNEL_TYPED(float)
REGISTER_DEFORMCONV_KERNEL_TYPED(double)

#undef REGISTER_DEFORMCONV_KERNEL_TYPED

}  // namespace onnxruntime
