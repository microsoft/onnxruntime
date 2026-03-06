// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CPU implementation of DeformConv (deformable convolution 2D).

#include "deform_conv.h"

#include <cmath>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/util/math.h"

namespace onnxruntime {

namespace {

// Bilinear interpolation at (h, w). Returns 0 if out of bounds.
template <typename T>
T BilinearInterpolate(const T* in, int64_t height, int64_t width, T h, T w) {
  // Check boundaries
  if (h <= static_cast<T>(-1) || h >= height || w <= static_cast<T>(-1) || w >= width) {
    return static_cast<T>(0);
  }

  const int h_low = static_cast<int>(std::floor(h));
  const int w_low = static_cast<int>(std::floor(w));
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const T lh = h - static_cast<T>(h_low);
  const T lw = w - static_cast<T>(w_low);
  const T hh = static_cast<T>(1) - lh;
  const T hw = static_cast<T>(1) - lw;

  const T v1 = (h_low >= 0 && w_low >= 0) ? in[h_low * width + w_low] : static_cast<T>(0);
  const T v2 = (h_low >= 0 && w_high < width) ? in[h_low * width + w_high] : static_cast<T>(0);
  const T v3 = (h_high < height && w_low >= 0) ? in[h_high * width + w_low] : static_cast<T>(0);
  const T v4 = (h_high < height && w_high < width) ? in[h_high * width + w_high] : static_cast<T>(0);

  const T w1 = hh * hw;
  const T w2 = hh * lw;
  const T w3 = lh * hw;
  const T w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

// Deformable Im2Col for a SINGLE image.
// Converts the input image into a matrix suitable for GEMM.
// Output 'data_col' shape: [C_in * kH * kW, H_out * W_out]
template <typename T>
void DeformableIm2col(
    const T* data_im,                        // Input image [C, H, W]
    const T* data_offset,                    // Offset [offset_groups * 2 * kH * kW, H_out, W_out]
    const T* data_mask,                      // Mask [offset_groups * kH * kW, H_out, W_out] (optional)
    int64_t height, int64_t width,           // Input dimensions
    int64_t kernel_h, int64_t kernel_w,      // Kernel dimensions
    int64_t pad_h, int64_t pad_w,            // Padding
    int64_t stride_h, int64_t stride_w,      // Stride
    int64_t dilation_h, int64_t dilation_w,  // Dilation
    int64_t channels,                        // Input channels
    int64_t offset_groups,                   // Number of offset groups
    int64_t height_col, int64_t width_col,   // Output dimensions
    bool use_mask,                           // Use mask
    T* data_col,                             // Output buffer
    concurrency::ThreadPool* thread_pool) {
  const int64_t channel_per_offset_group = channels / offset_groups;

  // Loop order optimized for cache locality:
  // Outer loop: Channels
  // Inner loop: Spatial locations (c_col)
  // This ensures sequential access to data_col and better locality for data_im.

  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(channels), 1.0,
      [&](ptrdiff_t c_im_start, ptrdiff_t c_im_end) {
        for (int64_t c_im = c_im_start; c_im < c_im_end; ++c_im) {
          const int64_t offset_grp = c_im / channel_per_offset_group;

          for (int64_t c_col = 0; c_col < height_col * width_col; ++c_col) {
            const int64_t w_col = c_col % width_col;
            const int64_t h_col = c_col / width_col;

            // Iterate over kernel window
            for (int64_t i = 0; i < kernel_h; ++i) {
              for (int64_t j = 0; j < kernel_w; ++j) {
                // Calculate the index in the offset/mask tensors.
                // The offset tensor is organized as: (offset_groups, 2 * kH * kW, H_out, W_out).
                // Flattened offset channel index relative to the start of the tensor:
                // base = offset_grp * (2 * kH * kW).
                // specific = 2 * (i * kW + j).

                const int64_t data_offset_h_ptr =
                    ((offset_grp * (2 * kernel_h * kernel_w) + 2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;

                const int64_t data_offset_w_ptr =
                    ((offset_grp * (2 * kernel_h * kernel_w) + 2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;

                const int64_t data_mask_ptr =
                    ((offset_grp * (kernel_h * kernel_w) + (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;

                const T offset_h = data_offset[data_offset_h_ptr];
                const T offset_w = data_offset[data_offset_w_ptr];

                T val = static_cast<T>(0);
                T mask_val = static_cast<T>(1);
                if (use_mask) {
                  mask_val = data_mask[data_mask_ptr];
                }

                // Only compute interpolation if mask is not zero (optimization)
                if (mask_val != 0) {
                  const T h_im = h_col * stride_h - pad_h + i * dilation_h + offset_h;
                  const T w_im = w_col * stride_w - pad_w + j * dilation_w + offset_w;

                  // Map (c_im, h_im, w_im) back to input
                  // data_im is [C, H, W]
                  const T* data_im_ptr = data_im + c_im * (height * width);
                  val = BilinearInterpolate(data_im_ptr, height, width, h_im, w_im);
                }

                // Assign to data_col
                // The layout of data_col row is: [Channel, KernelH, KernelW] flattened.
                // Row index: c_im * (kH * kW) + i * kW + j
                const int64_t col_row_idx = (c_im * kernel_h * kernel_w) + (i * kernel_w + j);

                data_col[col_row_idx * (height_col * width_col) + c_col] = val * mask_val;
              }
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
  ORT_RETURN_IF_ERROR(DeformConvValidateAndParse(attrs_, X->Shape(), W->Shape(), offset->Shape(), mask ? &mask->Shape() : nullptr, params));

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

  // Allocate Output
  const TensorShape Y_shape({N, M, out_h, out_w});
  Tensor* Y = context->Output(0, Y_shape);
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  // Common sizes
  const int64_t kernel_size = kH * kW;
  const int64_t output_image_size = out_h * out_w;
  const int64_t input_image_size = H * W_in;
  const int64_t kernel_dim = C / group * kernel_size;  // The "K" dimension for GEMM (per group)

  // Total col buffer size: (C * kH * kW) * (out_h * out_w)
  // We allocate this per image to save memory compared to batch allocation if N is large,
  // or simply because Im2Col is easier to implement per-image.
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

  // Main Loop: Iterate over Batch
  for (int64_t n = 0; n < N; ++n) {
    // 1. Perform Im2Col for the current image n
    // Pointers for current image
    const T* X_curr = Xdata + n * (C * input_image_size);
    const T* offset_curr = offset_data + n * (offset_group * 2 * kernel_size * output_image_size);
    const T* mask_curr = use_mask ? (mask_data + n * (offset_group * kernel_size * output_image_size)) : nullptr;
    T* col_buffer_ptr = col_buffer.get();

    // DeformableIm2col only needs pad_h, pad_w (begin-side pads) for coordinate mapping.
    // pad_h_end and pad_w_end are used in out_h/out_w computation (params) but do not affect
    // the im2col sampling logic; they only influence output dimensions.
    DeformableIm2col<T>(
        X_curr,
        offset_curr,
        mask_curr,
        H, W_in,
        kH, kW,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        C,
        offset_group,
        out_h, out_w,
        use_mask,
        col_buffer_ptr,
        thread_pool);

    // 2. Perform GEMM for each group
    for (int64_t g = 0; g < group; ++g) {
      // Weight pointer for group g
      // Weight shape: [M, C/group, kH, kW].
      // Stride for group g is (M/group) * (C/group * kH * kW).
      const T* weight_g = Wdata + g * (M / group) * kernel_dim;

      // Col buffer pointer for group g
      // Col buffer shape: [C * kH * kW, output_image_size]
      // We need the rows corresponding to group g.
      // Row stride: output_image_size
      // Group stride: (C/group * kH * kW) * output_image_size
      const T* col_g = col_buffer_ptr + g * kernel_dim * output_image_size;

      // Output pointer for group g
      // Output shape: [N, M, out_h, out_w]
      // Current image offset: n * M * output_image_size
      // Group offset: g * (M/group) * output_image_size
      T* Y_g = Ydata + n * M * output_image_size + g * (M / group) * output_image_size;

      // Y = W * Col
      // W matrix: [M/group, kernel_dim]
      // Col matrix: [kernel_dim, output_image_size]
      // Y matrix: [M/group, output_image_size]
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

  // 3. Add Bias if present
  if (Bdata != nullptr) {
    int64_t total_work = N * M;
    concurrency::ThreadPool::TryParallelFor(
        thread_pool, static_cast<std::ptrdiff_t>(total_work), static_cast<double>(output_image_size),
        [&](ptrdiff_t first, ptrdiff_t last) {
          for (ptrdiff_t idx = first; idx < last; ++idx) {
            int64_t n = idx / M;
            int64_t m = idx % M;
            T* Y_ptr = Ydata + n * M * output_image_size + m * output_image_size;
            T bias_val = Bdata[m];
            for (int64_t i = 0; i < output_image_size; ++i) {
              Y_ptr[i] += bias_val;
            }
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
