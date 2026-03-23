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
constexpr size_t kMaxSamplingPlanWorkspaceBytes = 256ull * 1024ull * 1024ull;
constexpr int64_t kStreamingSpatialTileSize = 4096;

namespace sampling_plan_internal {

constexpr int64_t kPlanAoSoALanes = 8;

template <typename T>
struct alignas(64) BilinearSamplePlanBlock {
  int32_t idx[4][kPlanAoSoALanes];
  T w[4][kPlanAoSoALanes];
  uint8_t flags[kPlanAoSoALanes];
};

template <typename T>
struct BilinearSamplePlanArrays {
  BilinearSamplePlanBlock<T>* blocks = nullptr;
  int64_t sample_offset = 0;
};

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
  const size_t lanes = static_cast<size_t>(kPlanAoSoALanes);
  ORT_ENFORCE(plan_size <= max_size_t - (lanes - 1), "Sampling plan block count overflows size_t.");
  const size_t block_count = (plan_size + lanes - 1) / lanes;
  ORT_ENFORCE(block_count <= max_size_t / sizeof(BilinearSamplePlanBlock<T>), "Sampling plan bytes overflows size_t.");

  idx_bytes = block_count * sizeof(BilinearSamplePlanBlock<T>);
  weight_bytes = 0;
  flag_bytes = 0;

  size_t bytes_total = 0;
  AddAlignedRegion(alignof(BilinearSamplePlanBlock<T>), idx_bytes, max_size_t, bytes_total);
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

  ORT_UNUSED_PARAMETER(weight_bytes);
  ORT_UNUSED_PARAMETER(flag_bytes);
  sampling_plan.blocks = reinterpret_cast<BilinearSamplePlanBlock<T>*>(
      take_region(alignof(BilinearSamplePlanBlock<T>), idx_bytes));
  sampling_plan.sample_offset = 0;

  ORT_ENFORCE(cursor <= bytes_total, "Sampling plan buffer layout exceeds allocated workspace.");
}

ORT_FORCEINLINE int64_t PlanGlobalIndex(int64_t sample_offset, int64_t pidx) {
  return sample_offset + pidx;
}

ORT_FORCEINLINE int64_t PlanBlockIndex(int64_t global_idx) {
  return global_idx / kPlanAoSoALanes;
}

ORT_FORCEINLINE int64_t PlanLaneIndex(int64_t global_idx) {
  return global_idx % kPlanAoSoALanes;
}

template <typename T>
ORT_FORCEINLINE void PlanStoreSample(BilinearSamplePlanArrays<T>& plan, int64_t pidx,
                                     int32_t idx00, int32_t idx01, int32_t idx10, int32_t idx11,
                                     T w00, T w01, T w10, T w11, uint8_t flag) {
  const int64_t global_idx = PlanGlobalIndex(plan.sample_offset, pidx);
  const int64_t block = PlanBlockIndex(global_idx);
  const int64_t lane = PlanLaneIndex(global_idx);
  auto& dst = plan.blocks[block];
  dst.idx[0][lane] = idx00;
  dst.idx[1][lane] = idx01;
  dst.idx[2][lane] = idx10;
  dst.idx[3][lane] = idx11;
  dst.w[0][lane] = w00;
  dst.w[1][lane] = w01;
  dst.w[2][lane] = w10;
  dst.w[3][lane] = w11;
  dst.flags[lane] = flag;
}

template <typename T>
struct LoadedPlanSample {
  int32_t idx[4];
  T w[4];
  uint8_t flag;
};

template <typename T>
ORT_FORCEINLINE LoadedPlanSample<T> PlanLoadSample(const BilinearSamplePlanArrays<T>& plan, int64_t pidx) {
  const int64_t global_idx = PlanGlobalIndex(plan.sample_offset, pidx);
  const int64_t block = PlanBlockIndex(global_idx);
  const int64_t lane = PlanLaneIndex(global_idx);
  const auto& src = plan.blocks[block];
  LoadedPlanSample<T> out{
      {src.idx[0][lane], src.idx[1][lane], src.idx[2][lane], src.idx[3][lane]},
      {src.w[0][lane], src.w[1][lane], src.w[2][lane], src.w[3][lane]},
      src.flags[lane]};
  return out;
}

template <typename T>
ORT_FORCEINLINE BilinearSamplePlanArrays<T> SlicePlanArrays(BilinearSamplePlanArrays<T>& sampling_plan, int64_t base) {
  return BilinearSamplePlanArrays<T>{sampling_plan.blocks, sampling_plan.sample_offset + base};
}

template <typename T>
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
    BilinearSamplePlanArrays<T>& plan) {
  auto process_sample = [&](int64_t h_col, int64_t w_col, int64_t spatial_idx, int64_t pidx) {
    const T h_im = h_col * stride_h + base_h + ptr_offset_h[spatial_idx];
    const T w_im = w_col * stride_w + base_w + ptr_offset_w[spatial_idx];

    if (h_im <= static_cast<T>(-1) || h_im >= height || w_im <= static_cast<T>(-1) || w_im >= width) {
      PlanStoreSample(plan, pidx, 0, 0, 0, 0,
                      static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), 0);
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

    const T plan_w00 = hh * hw;
    const T plan_w01 = hh * lw;
    const T plan_w10 = lh * hw;
    const T plan_w11 = lh * lw;

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
    PlanStoreSample(
        plan, pidx,
        base_low + w_low,
        base_low + w_high,
        base_high + w_low,
        base_high + w_high,
        plan_w00, plan_w01, plan_w10, plan_w11, mask);
  };

  ORT_ENFORCE(spatial_begin >= 0 && spatial_count >= 0 && spatial_begin + spatial_count <= height_col * width_col,
              "Invalid spatial range for sampling plan.");

  const int64_t end = spatial_begin + spatial_count;
  int64_t local_idx = 0;
  const int64_t first_h = spatial_begin / width_col;
  const int64_t last_h = (end - 1) / width_col;
  for (int64_t h_col = first_h; h_col <= last_h; ++h_col) {
    const int64_t w_begin = (h_col == first_h) ? (spatial_begin - h_col * width_col) : 0;
    const int64_t w_end = (h_col == last_h) ? (end - h_col * width_col) : width_col;
    int64_t spatial_idx = h_col * width_col + w_begin;
    for (int64_t w_col = w_begin; w_col < w_end; ++w_col) {
      process_sample(h_col, w_col, spatial_idx, local_idx);
      ++spatial_idx;
      ++local_idx;
    }
  }

  ORT_ENFORCE(local_idx == spatial_count, "Sampling plan fill count mismatch.");
}

template <typename T>
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
  const double plan_parallel_cost = static_cast<double>(spatial_count) * 12.0;
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

          const int64_t plan_row_base = row * spatial_count;
          BilinearSamplePlanArrays<T> row_plan = SlicePlanArrays(sampling_plan, plan_row_base);

          BuildBilinearSamplingPlanImpl(
              ptr_offset_h, ptr_offset_w,
              height, width, height_col, width_col,
              stride_h, stride_w, base_h, base_w,
              spatial_begin, spatial_count,
              row_plan);
        }
      });
}

template <typename T>
ORT_FORCEINLINE T EvalSampleFromPlan(const T* im_ptr,
                                     const BilinearSamplePlanArrays<T>& sampling_plan,
                                     int64_t pidx) {
  const LoadedPlanSample<T> sample = PlanLoadSample(sampling_plan, pidx);
  const uint8_t flag = sample.flag;
  const int32_t idx00 = sample.idx[0];
  const int32_t idx01 = sample.idx[1];
  const int32_t idx10 = sample.idx[2];
  const int32_t idx11 = sample.idx[3];
  const T w00 = sample.w[0];
  const T w01 = sample.w[1];
  const T w10 = sample.w[2];
  const T w11 = sample.w[3];
  if (flag == kAllNeighborsMask) {
    const Eigen::Matrix<T, 4, 1> weights(w00, w01, w10, w11);
    const Eigen::Matrix<T, 4, 1> samples(im_ptr[idx00], im_ptr[idx01], im_ptr[idx10], im_ptr[idx11]);
    return weights.dot(samples);
  }

  T val = static_cast<T>(0);
  if (flag != 0) {
    if (flag & kTopLeftMask) {
      val += w00 * im_ptr[idx00];
    }
    if (flag & kTopRightMask) {
      val += w01 * im_ptr[idx01];
    }
    if (flag & kBottomLeftMask) {
      val += w10 * im_ptr[idx10];
    }
    if (flag & kBottomRightMask) {
      val += w11 * im_ptr[idx11];
    }
  }
  return val;
}

template <typename T>
struct DeformableIm2colContext {
  const T* data_im = nullptr;
  const T* data_offset = nullptr;
  const T* data_mask = nullptr;
  int height = 0;
  int width = 0;
  int64_t kernel_h = 0;
  int64_t kernel_w = 0;
  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t dilation_h = 0;
  int64_t dilation_w = 0;
  int64_t channels = 0;
  int64_t offset_groups = 0;
  int64_t height_col = 0;
  int64_t width_col = 0;
  int64_t spatial_tile_size = 0;
  BilinearSamplePlanArrays<T>* sampling_plan = nullptr;
  T* data_col = nullptr;
  concurrency::ThreadPool* thread_pool = nullptr;
};

template <typename T, bool UseMask>
void FillColRowFromSamplingPlanImpl(
    const T* im_ptr,
    const BilinearSamplePlanArrays<T>& sampling_plan,
    int64_t spatial_begin,
    int64_t spatial_count,
    int64_t mask_row_base,
    const T* ptr_mask,
    T* col_ptr) {
  for (int64_t local_idx = 0; local_idx < spatial_count; ++local_idx) {
    T val = EvalSampleFromPlan(im_ptr, sampling_plan, local_idx);
    if constexpr (UseMask) {
      val *= ptr_mask[mask_row_base + local_idx];
    }
    col_ptr[spatial_begin + local_idx] = val;
  }
}

// Deformable Im2Col for a SINGLE image. A single implementation handles
// both full-plan and streamed-plan modes via spatial_tile_size.
template <typename T, bool UseMask>
void DeformableIm2colPlanned(const DeformableIm2colContext<T>& ctx) {
  ORT_ENFORCE(ctx.sampling_plan != nullptr, "sampling_plan must not be null.");
  ORT_ENFORCE(ctx.data_col != nullptr, "data_col must not be null.");
  const int64_t channel_per_offset_group = ctx.channels / ctx.offset_groups;
  const int64_t kernel_size = ctx.kernel_h * ctx.kernel_w;
  const int64_t output_size = ctx.height_col * ctx.width_col;

  for (int64_t spatial_begin = 0; spatial_begin < output_size; spatial_begin += ctx.spatial_tile_size) {
    const int64_t spatial_count = std::min<int64_t>(ctx.spatial_tile_size, output_size - spatial_begin);
    BuildAllBilinearSamplingPlansImpl(
        ctx.data_offset, ctx.height, ctx.width,
        ctx.kernel_w, ctx.pad_h, ctx.pad_w,
        ctx.stride_h, ctx.stride_w,
        ctx.dilation_h, ctx.dilation_w,
        ctx.offset_groups,
        output_size, ctx.height_col, ctx.width_col, kernel_size,
        spatial_begin, spatial_count,
        *ctx.sampling_plan, ctx.thread_pool);

    const double parallel_cost = static_cast<double>(spatial_count) * (UseMask ? 12.0 : 10.0);
    concurrency::ThreadPool::TryParallelFor(
        ctx.thread_pool,
        static_cast<std::ptrdiff_t>(ctx.channels * kernel_size),
        parallel_cost,
        [&](ptrdiff_t begin, ptrdiff_t end) {
          for (ptrdiff_t idx = begin; idx < end; ++idx) {
            const int64_t j = static_cast<int64_t>(idx) % ctx.kernel_w;
            const int64_t i = (static_cast<int64_t>(idx) / ctx.kernel_w) % ctx.kernel_h;
            const int64_t c_im = static_cast<int64_t>(idx) / kernel_size;
            const int64_t offset_grp = c_im / channel_per_offset_group;

            T* col_ptr = ctx.data_col + static_cast<int64_t>(idx) * output_size;
            const T* im_ptr = ctx.data_im + c_im * static_cast<int64_t>(ctx.height) * ctx.width;
            const int64_t row = offset_grp * kernel_size + i * ctx.kernel_w + j;
            const int64_t plan_row_base = row * spatial_count;
            BilinearSamplePlanArrays<T> row_plan = SlicePlanArrays(*ctx.sampling_plan, plan_row_base);
            [[maybe_unused]] const T* ptr_mask = nullptr;
            const int64_t mask_row_base = row * output_size + spatial_begin;
            if constexpr (UseMask) {
              ptr_mask = ctx.data_mask;
            }

            FillColRowFromSamplingPlanImpl<T, UseMask>(
                im_ptr, row_plan, spatial_begin, spatial_count, mask_row_base, ptr_mask, col_ptr);
          }
        });
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
  size_t bytes_total = bytes_total_full;
  if (use_streamed_plan) {
    bytes_total = sampling_plan_internal::ComputeSamplingPlanWorkspaceBytes<T>(
        plan_size_for_alloc, max_size_t, idx_bytes, weight_bytes, flag_bytes);
  }
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

    const int64_t tile_size = use_streamed_plan ? spatial_tile_size : output_image_size;
    sampling_plan_internal::DeformableIm2colContext<T> im2col_ctx{
        X_curr,
        offset_curr,
        mask_curr,
        static_cast<int>(H),
        static_cast<int>(W_in),
        kH,
        kW,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        C,
        offset_group,
        out_h,
        out_w,
        tile_size,
        &sampling_plan,
        col_buffer_ptr,
        thread_pool};
    if (use_mask) {
      sampling_plan_internal::DeformableIm2colPlanned<T, true>(im2col_ctx);
    } else {
      sampling_plan_internal::DeformableIm2colPlanned<T, false>(im2col_ctx);
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
            int64_t n_idx = idx / M;
            int64_t m_idx = idx % M;
            T* Y_ptr = Ydata + n_idx * M * output_image_size + m_idx * output_image_size;
            // Eigen vectorized add: Y_ptr += Bdata[m_idx] over all spatial positions.
            EigenVectorArrayMap<T>(Y_ptr, narrow<ptrdiff_t>(output_image_size)) += Bdata[m_idx];
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
