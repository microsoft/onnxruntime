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
#include <vector>

#include "core/common/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/force_inline.h"
#include "core/util/math.h"

namespace onnxruntime {

namespace {

constexpr int64_t kPlanAoSoALanes = 8;

ORT_FORCEINLINE size_t CheckedMulSizeT(size_t a, size_t b, size_t max_size_t, const char* err) {
  ORT_ENFORCE(a == 0 || b <= max_size_t / a, err);
  return a * b;
}

ORT_FORCEINLINE void CheckedBatchSpan(size_t n, size_t stride, size_t max_size_t, const char* err) {
  ORT_ENFORCE(n == 0 || stride <= max_size_t / n, err);
}

struct CpuDeformConvStrides {
  size_t x_batch_stride = 0;
  size_t y_batch_stride = 0;
  size_t w_group_stride = 0;
  size_t col_group_stride = 0;
  size_t y_group_stride = 0;
  size_t offset_batch_stride = 0;
  size_t mask_batch_stride = 0;
};

struct CpuDeformConvExecutionDims {
  int64_t plan_rows = 0;
  int64_t padded_spatial_count = 0;
  size_t block_count = 0;
  int64_t im2col_rows = 0;
  int64_t total_work = 0;
  int64_t col_buffer_size = 0;
};

inline CpuDeformConvExecutionDims ComputeCpuDeformConvExecutionDims(const DeformConvParams& params,
                                                                    const DeformConvCommonDims& common_dims,
                                                                    int64_t ptrdiff_max,
                                                                    size_t max_size_t) {
  const int64_t int64_max = std::numeric_limits<int64_t>::max();

  ORT_ENFORCE(params.offset_group <= int64_max / common_dims.kernel_size, "plan_rows overflows int64.");
  const int64_t plan_rows = params.offset_group * common_dims.kernel_size;

  ORT_ENFORCE(plan_rows > 0 && common_dims.output_image_size > 0, "Invalid plan dimensions.");
  ORT_ENFORCE(common_dims.output_image_size <= int64_max - (kPlanAoSoALanes - 1),
              "output_image_size is too large and will overflow.");
  const int64_t padded_spatial_count = (common_dims.output_image_size + kPlanAoSoALanes - 1) /
                                       kPlanAoSoALanes * kPlanAoSoALanes;
  const size_t blocks_per_row = static_cast<size_t>(padded_spatial_count) / kPlanAoSoALanes;
  ORT_ENFORCE(blocks_per_row <= (max_size_t / static_cast<size_t>(plan_rows)),
              "Sampling plan size overflows size_t.");
  const size_t block_count = static_cast<size_t>(plan_rows) * blocks_per_row;

  ORT_ENFORCE(plan_rows <= ptrdiff_max, "plan_rows exceeds ptrdiff_t range.");
  ORT_ENFORCE(params.C <= int64_max / common_dims.kernel_size, "im2col row count overflows int64.");
  const int64_t im2col_rows = params.C * common_dims.kernel_size;
  ORT_ENFORCE(im2col_rows <= ptrdiff_max, "im2col row count exceeds ptrdiff_t range.");
  ORT_ENFORCE(params.N <= int64_max / params.M, "N*M overflows int64.");
  const int64_t total_work = params.N * params.M;
  ORT_ENFORCE(total_work <= ptrdiff_max, "bias work size exceeds ptrdiff_t range.");
  ORT_ENFORCE(im2col_rows <= int64_max / common_dims.output_image_size, "col_buffer_size overflows int64.");
  const int64_t col_buffer_size = im2col_rows * common_dims.output_image_size;

  return CpuDeformConvExecutionDims{
      plan_rows,
      padded_spatial_count,
      block_count,
      im2col_rows,
      total_work,
      col_buffer_size};
}

inline CpuDeformConvStrides ComputeCpuDeformConvStrides(const DeformConvParams& params,
                                                        const DeformConvCommonDims& common_dims,
                                                        size_t max_size_t) {
  const size_t c_size = static_cast<size_t>(params.C);
  const size_t m_size = static_cast<size_t>(params.M);
  const size_t n_size = static_cast<size_t>(params.N);
  const size_t group_size = static_cast<size_t>(params.group);
  const size_t offset_group_size = static_cast<size_t>(params.offset_group);
  const size_t input_image_size_sz = static_cast<size_t>(common_dims.input_image_size);
  const size_t output_image_size_sz = static_cast<size_t>(common_dims.output_image_size);
  const size_t kernel_size_sz = static_cast<size_t>(common_dims.kernel_size);
  const size_t kernel_dim_sz = static_cast<size_t>(common_dims.kernel_dim);
  const size_t m_per_group_sz = m_size / group_size;

  CpuDeformConvStrides strides;
  strides.x_batch_stride = CheckedMulSizeT(c_size, input_image_size_sz, max_size_t, "X batch stride overflows size_t.");
  strides.y_batch_stride = CheckedMulSizeT(m_size, output_image_size_sz, max_size_t, "Y batch stride overflows size_t.");
  strides.w_group_stride = CheckedMulSizeT(m_per_group_sz, kernel_dim_sz, max_size_t, "weight group stride overflows size_t.");
  strides.col_group_stride = CheckedMulSizeT(kernel_dim_sz, output_image_size_sz, max_size_t, "col group stride overflows size_t.");
  strides.y_group_stride = CheckedMulSizeT(m_per_group_sz, output_image_size_sz, max_size_t, "Y group stride overflows size_t.");

  const size_t offset_rows = CheckedMulSizeT(
      CheckedMulSizeT(offset_group_size, kernel_size_sz, max_size_t, "offset rows overflows size_t."),
      static_cast<size_t>(2), max_size_t, "offset rows overflows size_t.");
  strides.offset_batch_stride = CheckedMulSizeT(offset_rows, output_image_size_sz, max_size_t, "offset batch stride overflows size_t.");

  const size_t mask_rows = CheckedMulSizeT(offset_group_size, kernel_size_sz, max_size_t, "mask rows overflows size_t.");
  strides.mask_batch_stride = CheckedMulSizeT(mask_rows, output_image_size_sz, max_size_t, "mask batch stride overflows size_t.");

  CheckedBatchSpan(n_size, strides.x_batch_stride, max_size_t, "X batch indexing overflows size_t.");
  CheckedBatchSpan(n_size, strides.y_batch_stride, max_size_t, "Y batch indexing overflows size_t.");
  CheckedBatchSpan(n_size, strides.offset_batch_stride, max_size_t, "offset batch indexing overflows size_t.");
  if (params.use_mask) {
    CheckedBatchSpan(n_size, strides.mask_batch_stride, max_size_t, "mask batch indexing overflows size_t.");
  }

  return strides;
}

namespace sampling_plan_internal {

template <typename T>
struct KernelMetaEntry {
  size_t offset_base_delta = 0;  // 2 * kernel_idx
  T base_h{};
  T base_w{};
};

template <typename T>
struct alignas(64) BilinearSamplePlanBlock {
  int32_t idx[4][kPlanAoSoALanes];
  T w[4][kPlanAoSoALanes];
};

template <typename T>
struct DeformableIm2colContext {
  const T* data_im = nullptr;
  const T* data_offset = nullptr;
  const T* data_mask = nullptr;
  int height = 0;
  int width = 0;
  int64_t kernel_h = 0;
  int64_t kernel_w = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t channels = 0;
  int64_t offset_groups = 0;
  int64_t height_col = 0;
  int64_t width_col = 0;
  int64_t padded_spatial_count = 0;
  const KernelMetaEntry<T>* kernel_meta = nullptr;
  BilinearSamplePlanBlock<T>* sampling_plan_blocks = nullptr;
  T* data_col = nullptr;
  concurrency::ThreadPool* thread_pool = nullptr;
};

template <typename T>
ORT_FORCEINLINE void PlanStoreSample(BilinearSamplePlanBlock<T>* blocks, int64_t pidx,
                                     int32_t idx00, int32_t idx01, int32_t idx10, int32_t idx11,
                                     T w00, T w01, T w10, T w11) {
  const int64_t block = pidx / kPlanAoSoALanes;
  const int64_t lane = pidx % kPlanAoSoALanes;
  auto& dst = blocks[block];
  dst.idx[0][lane] = idx00;
  dst.idx[1][lane] = idx01;
  dst.idx[2][lane] = idx10;
  dst.idx[3][lane] = idx11;
  dst.w[0][lane] = w00;
  dst.w[1][lane] = w01;
  dst.w[2][lane] = w10;
  dst.w[3][lane] = w11;
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
    BilinearSamplePlanBlock<T>* plan_blocks) {
  auto process_sample = [&](int64_t h_col, int64_t w_col, int64_t local_idx) {
    const T h_im = h_col * stride_h + base_h + ptr_offset_h[local_idx];
    const T w_im = w_col * stride_w + base_w + ptr_offset_w[local_idx];

    if (h_im <= static_cast<T>(-1) || h_im >= height || w_im <= static_cast<T>(-1) || w_im >= width) {
      PlanStoreSample(plan_blocks, local_idx, 0, 0, 0, 0,
                      static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
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

    T plan_w00 = hh * hw;
    T plan_w01 = hh * lw;
    T plan_w10 = lh * hw;
    T plan_w11 = lh * lw;

    const int base_low = h_low * width;
    const int base_high = h_high * width;
    int32_t idx00 = base_low + w_low;
    int32_t idx01 = base_low + w_high;
    int32_t idx10 = base_high + w_low;
    int32_t idx11 = base_high + w_high;

    if (static_cast<unsigned>(h_low) < static_cast<unsigned>(height - 1) &&
        static_cast<unsigned>(w_low) < static_cast<unsigned>(width - 1)) {
      PlanStoreSample(
          plan_blocks, local_idx,
          idx00, idx01, idx10, idx11,
          plan_w00, plan_w01, plan_w10, plan_w11);
    } else {
      const bool v00 = (h_low >= 0 && w_low >= 0);
      const bool v01 = (h_low >= 0 && w_high < width);
      const bool v10 = (h_high < height && w_low >= 0);
      const bool v11 = (h_high < height && w_high < width);

      plan_w00 = v00 ? plan_w00 : static_cast<T>(0);
      plan_w01 = v01 ? plan_w01 : static_cast<T>(0);
      plan_w10 = v10 ? plan_w10 : static_cast<T>(0);
      plan_w11 = v11 ? plan_w11 : static_cast<T>(0);

      idx00 = v00 ? idx00 : 0;
      idx01 = v01 ? idx01 : 0;
      idx10 = v10 ? idx10 : 0;
      idx11 = v11 ? idx11 : 0;

      PlanStoreSample(
          plan_blocks, local_idx,
          idx00, idx01, idx10, idx11,
          plan_w00, plan_w01, plan_w10, plan_w11);
    }
  };

  int64_t local_idx = 0;
  for (int64_t h_col = 0; h_col < height_col; ++h_col) {
    for (int64_t w_col = 0; w_col < width_col; ++w_col) {
      process_sample(h_col, w_col, local_idx);
      ++local_idx;
    }
  }
}

template <typename T>
void BuildAllBilinearSamplingPlansImpl(
    const T* data_offset,
    int height,
    int width,
    int64_t stride_h,
    int64_t stride_w,
    int64_t offset_groups,
    size_t output_size,
    int64_t padded_spatial_count,
    int64_t height_col,
    int64_t width_col,
    int64_t kernel_size,
    const KernelMetaEntry<T>* kernel_meta,
    BilinearSamplePlanBlock<T>* sampling_plan_blocks,
    concurrency::ThreadPool* thread_pool) {
  const int64_t plan_rows = offset_groups * kernel_size;
  ORT_ENFORCE(kernel_meta != nullptr, "kernel_meta must not be null.");
  const size_t kernel_size_sz = static_cast<size_t>(kernel_size);
  const size_t offset_group_stride = static_cast<size_t>(2) * kernel_size_sz;
  const double plan_parallel_cost = static_cast<double>(output_size) * 12.0;
  concurrency::ThreadPool::TryParallelFor(
      thread_pool,
      static_cast<std::ptrdiff_t>(plan_rows),
      plan_parallel_cost,
      [&](ptrdiff_t begin, ptrdiff_t end) {
        int64_t row = static_cast<int64_t>(begin);
        int64_t offset_grp = row / kernel_size;
        int64_t kernel_idx = row % kernel_size;
        size_t offset_grp_base = static_cast<size_t>(offset_grp) * offset_group_stride;

        for (ptrdiff_t plan_row = begin; plan_row < end; ++plan_row, ++row) {
          const auto& kernel_meta_entry = kernel_meta[static_cast<size_t>(kernel_idx)];

          const size_t offset_base = offset_grp_base + kernel_meta_entry.offset_base_delta;
          const T* ptr_offset_h = data_offset + offset_base * output_size;
          const T* ptr_offset_w = data_offset + (offset_base + 1) * output_size;

          const size_t plan_row_base = static_cast<size_t>(row) * static_cast<size_t>(padded_spatial_count);
          BilinearSamplePlanBlock<T>* row_plan = sampling_plan_blocks + (plan_row_base / kPlanAoSoALanes);

          BuildBilinearSamplingPlanImpl(
              ptr_offset_h, ptr_offset_w,
              height, width, height_col, width_col,
              stride_h, stride_w, kernel_meta_entry.base_h, kernel_meta_entry.base_w,
              row_plan);

          if (++kernel_idx == kernel_size) {
            kernel_idx = 0;
            ++offset_grp;
            offset_grp_base += offset_group_stride;
          }
        }
      });
}

template <typename T, bool UseMask>
void FillColRowFromSamplingPlanImpl(
    const T* im_ptr,
    const BilinearSamplePlanBlock<T>* plan_blocks,
    int64_t spatial_count,
    size_t mask_row_base,
    const T* ptr_mask,
    T* col_ptr) {
  const int64_t block_count = spatial_count / kPlanAoSoALanes;
  const int64_t tail_count = spatial_count % kPlanAoSoALanes;

  int64_t local_idx = 0;
  for (int64_t b = 0; b < block_count; ++b) {
    const auto& block = plan_blocks[b];
    for (int lane = 0; lane < kPlanAoSoALanes; ++lane) {
      T val = block.w[0][lane] * im_ptr[block.idx[0][lane]] +
              block.w[1][lane] * im_ptr[block.idx[1][lane]] +
              block.w[2][lane] * im_ptr[block.idx[2][lane]] +
              block.w[3][lane] * im_ptr[block.idx[3][lane]];
      if constexpr (UseMask) {
        val *= ptr_mask[mask_row_base + local_idx];
      }
      col_ptr[local_idx] = val;
      ++local_idx;
    }
  }

  if (tail_count > 0) {
    const auto& block = plan_blocks[block_count];
    for (int lane = 0; lane < tail_count; ++lane) {
      T val = block.w[0][lane] * im_ptr[block.idx[0][lane]] +
              block.w[1][lane] * im_ptr[block.idx[1][lane]] +
              block.w[2][lane] * im_ptr[block.idx[2][lane]] +
              block.w[3][lane] * im_ptr[block.idx[3][lane]];
      if constexpr (UseMask) {
        val *= ptr_mask[mask_row_base + local_idx];
      }
      col_ptr[local_idx] = val;
      ++local_idx;
    }
  }
}

// Deformable Im2Col for a SINGLE image.
template <typename T, bool UseMask>
void DeformableIm2colPlanned(const DeformableIm2colContext<T>& ctx) {
  ORT_ENFORCE(ctx.sampling_plan_blocks != nullptr, "sampling_plan_blocks must not be null.");
  ORT_ENFORCE(ctx.data_col != nullptr, "data_col must not be null.");
  const int64_t channel_per_offset_group = ctx.channels / ctx.offset_groups;
  const int64_t kernel_size = ctx.kernel_h * ctx.kernel_w;
  const int64_t output_size = ctx.height_col * ctx.width_col;
  const size_t output_size_sz = static_cast<size_t>(output_size);

  BuildAllBilinearSamplingPlansImpl(
      ctx.data_offset, ctx.height, ctx.width,
      ctx.stride_h, ctx.stride_w,
      ctx.offset_groups,
      output_size_sz, ctx.padded_spatial_count, ctx.height_col, ctx.width_col, kernel_size,
      ctx.kernel_meta,
      ctx.sampling_plan_blocks, ctx.thread_pool);

  const double parallel_cost = static_cast<double>(output_size) * (UseMask ? 12.0 : 10.0);
  concurrency::ThreadPool::TryParallelFor(
      ctx.thread_pool,
      static_cast<std::ptrdiff_t>(ctx.channels * kernel_size),
      parallel_cost,
      [&](ptrdiff_t begin, ptrdiff_t end) {
        int64_t c_im = begin / kernel_size;
        int64_t rem = begin % kernel_size;
        int64_t i = rem / ctx.kernel_w;
        int64_t j = rem % ctx.kernel_w;

        for (ptrdiff_t idx = begin; idx < end; ++idx) {
          const int64_t offset_grp = c_im / channel_per_offset_group;

          T* col_ptr = ctx.data_col + static_cast<int64_t>(idx) * output_size;
          const T* im_ptr = ctx.data_im + c_im * static_cast<int64_t>(ctx.height) * ctx.width;
          const int64_t row = offset_grp * kernel_size + i * ctx.kernel_w + j;
          const size_t plan_row_base = static_cast<size_t>(row) * static_cast<size_t>(ctx.padded_spatial_count);
          const BilinearSamplePlanBlock<T>* row_plan = ctx.sampling_plan_blocks + (plan_row_base / kPlanAoSoALanes);

          const T* ptr_mask = nullptr;
          size_t mask_row_base = 0;
          if constexpr (UseMask) {
            ptr_mask = ctx.data_mask;
            mask_row_base = static_cast<size_t>(row) * output_size_sz;
          }

          FillColRowFromSamplingPlanImpl<T, UseMask>(
              im_ptr, row_plan, output_size, mask_row_base, ptr_mask, col_ptr);

          if (++j == ctx.kernel_w) {
            j = 0;
            if (++i == ctx.kernel_h) {
              i = 0;
              ++c_im;
            }
          }
        }
      });
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

  // 1) Shared (CPU/CUDA) runtime bounds + derived dimensions.
  const int64_t ptrdiff_max = static_cast<int64_t>(std::numeric_limits<std::ptrdiff_t>::max());

  DeformConvCommonDims common_dims;
  ORT_RETURN_IF_ERROR(DeformConvValidateAndComputeCommonDims(params, common_dims));
  const int64_t kernel_size = common_dims.kernel_size;
  const int64_t output_image_size = common_dims.output_image_size;
  const int64_t input_image_size = common_dims.input_image_size;
  const int64_t kernel_dim = common_dims.kernel_dim;  // K dimension for GEMM: C/group * kH * kW
  const size_t max_size_t = std::numeric_limits<size_t>::max();
  const CpuDeformConvExecutionDims exec_dims = ComputeCpuDeformConvExecutionDims(params, common_dims, ptrdiff_max, max_size_t);
  const int64_t padded_spatial_count = exec_dims.padded_spatial_count;
  const size_t block_count = exec_dims.block_count;
  const int64_t total_work = exec_dims.total_work;
  std::vector<sampling_plan_internal::KernelMetaEntry<T>> kernel_meta(static_cast<size_t>(kernel_size));
  for (int64_t kernel_idx = 0; kernel_idx < kernel_size; ++kernel_idx) {
    const int64_t i = kernel_idx / kW;
    const int64_t j = kernel_idx % kW;
    auto& entry = kernel_meta[static_cast<size_t>(kernel_idx)];
    entry.offset_base_delta = static_cast<size_t>(2) * static_cast<size_t>(kernel_idx);
    entry.base_h = static_cast<T>(-pad_h + i * dilation_h);
    entry.base_w = static_cast<T>(-pad_w + j * dilation_w);
  }
  // Col buffer: shape [C*kH*kW, out_h*out_w]. Allocate per-image (process one image at a time)
  // to reduce peak memory when N is large; im2col is implemented per-image anyway.
  const int64_t col_buffer_size = exec_dims.col_buffer_size;

  // 4) Pointer-stride precompute (validated once, reused in hot loop).
  const CpuDeformConvStrides strides = ComputeCpuDeformConvStrides(params, common_dims, max_size_t);
  const size_t output_image_size_sz = static_cast<size_t>(output_image_size);
  const size_t x_batch_stride = strides.x_batch_stride;
  const size_t y_batch_stride = strides.y_batch_stride;
  const size_t w_group_stride = strides.w_group_stride;
  const size_t col_group_stride = strides.col_group_stride;
  const size_t y_group_stride = strides.y_group_stride;
  const size_t offset_batch_stride = strides.offset_batch_stride;
  const size_t mask_batch_stride = strides.mask_batch_stride;

  // 5) GEMM call-site bounds (checked once outside group loop).
  ORT_ENFORCE((M / group) <= ptrdiff_max, "GEMM M dimension exceeds ptrdiff_t range.");
  ORT_ENFORCE(output_image_size <= ptrdiff_max, "GEMM N dimension exceeds ptrdiff_t range.");
  ORT_ENFORCE(kernel_dim <= ptrdiff_max, "GEMM K dimension exceeds ptrdiff_t range.");

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  auto col_buffer = IAllocator::MakeUniquePtr<T>(alloc, SafeInt<size_t>(col_buffer_size));

  ORT_ENFORCE(static_cast<uint64_t>(H) * static_cast<uint64_t>(W_in) <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
              "DeformConv requires H*W to fit in int for sampling indices.");

  auto plan_blocks = IAllocator::MakeUniquePtr<sampling_plan_internal::BilinearSamplePlanBlock<T>>(alloc, SafeInt<size_t>(block_count));

  const T* Xdata = X->Data<T>();
  const T* Wdata = W->Data<T>();
  const T* offset_data = offset->Data<T>();
  const T* mask_data = use_mask ? mask->Data<T>() : nullptr;
  T* Ydata = Y->MutableData<T>();
  const T* Bdata = (B != nullptr) ? B->Data<T>() : nullptr;

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
  // Process each image in the batch.
  for (int64_t n = 0; n < N; ++n) {
    const size_t n_idx = static_cast<size_t>(n);
    // Step 1: Deformable Im2Col for image n.
    // Gather deformed samples into col buffer for GEMM.
    const T* X_curr = Xdata + n_idx * x_batch_stride;
    const T* offset_curr = offset_data + n_idx * offset_batch_stride;
    const T* mask_curr = use_mask ? (mask_data + n_idx * mask_batch_stride) : nullptr;
    T* col_buffer_ptr = col_buffer.get();

    sampling_plan_internal::DeformableIm2colContext<T> im2col_ctx{
        X_curr, offset_curr, mask_curr,
        static_cast<int>(H), static_cast<int>(W_in),
        kH, kW, stride_h, stride_w, C, offset_group, out_h, out_w,
        padded_spatial_count,
        kernel_meta.data(),
        plan_blocks.get(),
        col_buffer_ptr,
        thread_pool};
    if (use_mask) {
      sampling_plan_internal::DeformableIm2colPlanned<T, true>(im2col_ctx);
    } else {
      sampling_plan_internal::DeformableIm2colPlanned<T, false>(im2col_ctx);
    }

    // Step 2: GEMM for each group. Y = W * Col (per group).
    for (int64_t g = 0; g < group; ++g) {
      const size_t g_idx = static_cast<size_t>(g);
      // Weight for group g: shape [M/group, C/group, kH, kW], row-major.
      const T* weight_g = Wdata + g_idx * w_group_stride;

      // Col rows for group g: layout [C*kH*kW, out_h*out_w], group g spans rows [g*kernel_dim, (g+1)*kernel_dim).
      const T* col_g = col_buffer_ptr + g_idx * col_group_stride;

      // Output slice for group g: [n, g*M/group:(g+1)*M/group, out_h, out_w].
      T* Y_g = Ydata + n_idx * y_batch_stride + g_idx * y_group_stride;

      // GEMM: Y = W * Col. W [M/group, kernel_dim], Col [kernel_dim, output_image_size].
      math::Gemm<T>(
          CblasNoTrans,
          CblasNoTrans,
          static_cast<ptrdiff_t>(M / group),          // M
          static_cast<ptrdiff_t>(output_image_size),  // N
          static_cast<ptrdiff_t>(kernel_dim),         // K
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
    concurrency::ThreadPool::TryParallelFor(
        thread_pool, static_cast<std::ptrdiff_t>(total_work), static_cast<double>(output_image_size),
        [&](ptrdiff_t first, ptrdiff_t last) {
          for (ptrdiff_t idx = first; idx < last; ++idx) {
            int64_t n_idx = idx / M;
            int64_t m_idx = idx % M;
            const size_t n_idx_sz = static_cast<size_t>(n_idx);
            const size_t m_idx_sz = static_cast<size_t>(m_idx);
            T* Y_ptr = Ydata + n_idx_sz * y_batch_stride + m_idx_sz * output_image_size_sz;
            // Eigen vectorized add: Y_ptr += Bdata[m_idx] over all spatial positions.
            EigenVectorArrayMap<T>(Y_ptr, static_cast<ptrdiff_t>(output_image_size)) += Bdata[m_idx];
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
