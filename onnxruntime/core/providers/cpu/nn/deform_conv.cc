// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CPU implementation of DeformConv (deformable convolution 2D).

#include "deform_conv.h"

#include <cmath>

#include "core/common/common.h"
#include "core/util/math_cpuonly.h"
#include "core/common/narrow.h"
#include "core/util/math.h"

namespace onnxruntime {

namespace {
// Bilinear interpolation at (h, w). Out-of-bounds samples return 0 (ONNX spec).
// Indices use int (not int64_t) to reduce register pressure and improve occupancy in the hot path.
// Limitation: height and width must not exceed INT_MAX, or casting floor(h)/floor(w) to int may overflow.
// Acceptable in practice: deformable convolution spatial dimensions are typically well below INT_MAX.
template <typename T>
T BilinearInterpolate(const T* in, int height, int width, T h, T w) {
  // [Optimization 1]: Early exit for clearly out-of-bounds (skip floor() for OOB case).
  if (h <= static_cast<T>(-1) || h >= height || w <= static_cast<T>(-1) || w >= width) {
    return static_cast<T>(0);
  }

  // [Optimization 2]: Keep floor result in T; cast to int only for indices. Avoids float->int->float in lh/lw.
  const T h_floor = std::floor(h);
  const T w_floor = std::floor(w);
  const int h_low = static_cast<int>(h_floor);
  const int w_low = static_cast<int>(w_floor);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const T lh = h - h_floor;
  const T lw = w - w_floor;
  const T hh = static_cast<T>(1) - lh;
  const T hw = static_cast<T>(1) - lw;

  // Fast path: all 4 corners in bounds (h in [0, height-1), w in [0, width-1)).
  // Most sampling points in deformable conv fall here; avoids 4 per-corner branches.
  // [Optimization 3]: Use unsigned comparison to avoid branch on negative height/width.
  if (static_cast<unsigned>(h_low) < static_cast<unsigned>(height - 1) &&
      static_cast<unsigned>(w_low) < static_cast<unsigned>(width - 1)) {
    const int base_low = h_low * width;
    const int base_high = h_high * width;
    return hh * hw * in[base_low + w_low] +
           hh * lw * in[base_low + w_high] +
           lh * hw * in[base_high + w_low] +
           lh * lw * in[base_high + w_high];
  }

  // Slow path: near boundary (one or more of the 4 corners may be out of bounds).
  const int base_low = h_low * width;
  const int base_high = h_high * width;
  const T v1 = (h_low >= 0 && w_low >= 0) ? in[base_low + w_low] : static_cast<T>(0);
  const T v2 = (h_low >= 0 && w_high < width) ? in[base_low + w_high] : static_cast<T>(0);
  const T v3 = (h_high < height && w_low >= 0) ? in[base_high + w_low] : static_cast<T>(0);
  const T v4 = (h_high < height && w_high < width) ? in[base_high + w_high] : static_cast<T>(0);
  return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

// Deformable Im2Col for a SINGLE image.
// Converts the input image into a matrix suitable for GEMM by sampling with learned offsets.
// Output 'data_col' shape: [C_in * kH * kW, H_out * W_out]
// When UseMask=false, pass nullptr for data_mask; compiler eliminates dead code for mask.
template <typename T, bool UseMask>
void DeformableIm2col(
    const T* data_im,                        // Input image [C, H, W]
    const T* data_offset,                    // Offset [offset_groups * 2 * kH * kW, H_out, W_out]
    const T* data_mask,                      // Mask [offset_groups * kH * kW, H_out, W_out] (nullptr when UseMask=false)
    int height, int width,                   // Input spatial dimensions (validated H*W <= INT_MAX)
    int64_t kernel_h, int64_t kernel_w,      // Kernel dimensions
    int64_t pad_h, int64_t pad_w,            // Padding (begin) for H and W
    int64_t stride_h, int64_t stride_w,      // Stride for H and W
    int64_t dilation_h, int64_t dilation_w,  // Dilation for H and W
    int64_t channels,                        // Input channels
    int64_t offset_groups,                   // Number of offset groups (channels shared per group)
    int64_t height_col, int64_t width_col,   // Output spatial dimensions (H_out, W_out)
    T* data_col,                             // Output buffer for im2col result
    concurrency::ThreadPool* thread_pool) {
  const int64_t channel_per_offset_group = channels / offset_groups;
  const int64_t kernel_size = kernel_h * kernel_w;
  const int64_t output_size = height_col * width_col;

  // Parallelize over (channel, kernel_position) so each task processes one full row of data_col.
  // This yields channels*kernel_size tasks, better CPU utilization and cache-friendly sequential writes.
  concurrency::ThreadPool::TryParallelFor(
      thread_pool,
      static_cast<std::ptrdiff_t>(channels * kernel_size),
      static_cast<double>(output_size) * 10.0,
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (ptrdiff_t idx = begin; idx < end; ++idx) {
          // Decompose idx into (c_im, i, j): which channel and kernel position.
          const int64_t j = static_cast<int64_t>(idx) % kernel_w;
          const int64_t i = (static_cast<int64_t>(idx) / kernel_w) % kernel_h;
          const int64_t c_im = static_cast<int64_t>(idx) / kernel_size;
          const int64_t offset_grp = c_im / channel_per_offset_group;

          // Output row: one (channel, kernel_pos) across all spatial locations.
          T* col_ptr = data_col + static_cast<int64_t>(idx) * output_size;
          const T* im_ptr = data_im + c_im * static_cast<int64_t>(height) * width;

          // Offset tensor layout: [offset_grp, 2*kH*kW, H_out, W_out] flattened.
          // For (i,j) we use channel indices 2*(i*kW+j) and 2*(i*kW+j)+1 for offset_h, offset_w.
          // Precompute pointers to avoid offset_base * output_size multiplication in inner loop.
          const int64_t offset_base =
              offset_grp * 2 * kernel_size + 2 * (i * kernel_w + j);
          const T* ptr_offset_h = data_offset + offset_base * output_size;
          const T* ptr_offset_w = data_offset + (offset_base + 1) * output_size;

          // Base terms for h_im, w_im: invariant in inner loop (i, j fixed).
          const T base_h = -pad_h + static_cast<T>(i) * dilation_h;
          const T base_w = -pad_w + static_cast<T>(j) * dilation_w;

          // Mask pointer; only used when UseMask=true (compiler removes when false).
          [[maybe_unused]] const T* ptr_mask = nullptr;
          if constexpr (UseMask) {
            ptr_mask = data_mask + (offset_grp * kernel_size + i * kernel_w + j) * output_size;
          }

          // Loop over output spatial positions.
          for (int64_t h_col = 0; h_col < height_col; ++h_col) {
            for (int64_t w_col = 0; w_col < width_col; ++w_col) {
              const int64_t spatial_idx = h_col * width_col + w_col;

              const T offset_h = ptr_offset_h[spatial_idx];
              const T offset_w = ptr_offset_w[spatial_idx];

              // Deformed sampling coordinates (fractional, for bilinear interpolation).
              const T h_im = h_col * stride_h + base_h + offset_h;
              const T w_im = w_col * stride_w + base_w + offset_w;

              // Sample input at deformed location; returns 0 if out of bounds.
              T val = BilinearInterpolate(im_ptr, height, width, h_im, w_im);

              // Modulate by mask when UseMask=true; compiled away when false.
              // Design choice: we always interpolate then multiply, rather than skip when mask==0.
              // Rationale: (1) Skipping adds a branch; unpredictable mask values cause misprediction
              // penalties (~15-20 cycles). (2) Straight-line code vectorizes better; conditional
              // skip blocks SIMD. (3) Multiplying by 0 is cheap when vectorized. In typical DCN
              // usage (moderate mask density), the unconditional path usually wins.
              if constexpr (UseMask) {
                val *= ptr_mask[spatial_idx];
              }

              col_ptr[spatial_idx] = val;
            }
          }
        }
      });
}

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

  // Col buffer: shape [C*kH*kW, out_h*out_w]. Allocate per-image (process one image at a time)
  // to reduce peak memory when N is large; im2col is implemented per-image anyway.
  const int64_t col_buffer_size = (C * kernel_size) * output_image_size;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  auto col_buffer = IAllocator::MakeUniquePtr<T>(alloc, SafeInt<size_t>(col_buffer_size));

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

    // Dispatch to template instantiation: UseMask=true or false eliminates branch in hot loop.
    // Note: pad_h, pad_w are begin-side paddings for coordinate mapping; pad_h_end/pad_w_end
    // affect only output size (already baked into out_h, out_w), not im2col sampling.
    if (use_mask) {
      DeformableIm2col<T, true>(
          X_curr, offset_curr, mask_curr,
          static_cast<int>(H), static_cast<int>(W_in), kH, kW,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          C, offset_group, out_h, out_w,
          col_buffer_ptr, thread_pool);
    } else {
      DeformableIm2col<T, false>(
          X_curr, offset_curr, nullptr,
          static_cast<int>(H), static_cast<int>(W_in), kH, kW,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
          C, offset_group, out_h, out_w,
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
