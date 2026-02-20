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
    const T* data_im,       // Input image [C, H, W]
    const T* data_offset,   // Offset [offset_groups * 2 * kH * kW, H_out, W_out]
    const T* data_mask,     // Mask [offset_groups * kH * kW, H_out, W_out] (optional)
    int64_t height, int64_t width,            // Input dimensions
    int64_t kernel_h, int64_t kernel_w,       // Kernel dimensions
    int64_t pad_h, int64_t pad_w,             // Padding
    int64_t stride_h, int64_t stride_w,       // Stride
    int64_t dilation_h, int64_t dilation_w,   // Dilation
    int64_t channels,                         // Input channels
    int64_t offset_groups,                    // Number of offset groups
    int64_t height_col, int64_t width_col,    // Output dimensions
    bool use_mask,
    T* data_col) {                            // Output buffer

  const int64_t channel_per_offset_group = channels / offset_groups;

  // We iterate over the output matrix columns (spatial locations)
  // and fill the matrix rows (channels * kernels).
  // Note: Parallelization can be applied here over 'c_col' (spatial index).

  for (int64_t c_col = 0; c_col < height_col * width_col; ++c_col) {
    const int64_t w_col = c_col % width_col;
    const int64_t h_col = c_col / width_col;

    // For each spatial location (h_col, w_col), we iterate over all input channels
    for (int64_t c_im = 0; c_im < channels; ++c_im) {
      const int64_t offset_grp = c_im / channel_per_offset_group;

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
}

}  // namespace

template <typename T>
Status DeformConv<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto* offset = context->Input<Tensor>(2);
  const auto* B = context->Input<Tensor>(3);    // optional
  const auto* mask = context->Input<Tensor>(4);  // optional

  const auto& X_shape = X->Shape();
  const auto& W_shape = W->Shape();
  const auto& offset_shape = offset->Shape();

  // Validate Input Shapes
  const int64_t N = X_shape[0];
  const int64_t C = X_shape[1];
  const int64_t H = X_shape[2];
  const int64_t W_in = X_shape[3];

  const int64_t M = W_shape[0];       // out channels
  // Handle kernel shape inference
  const int64_t kH = attrs_.kernel_shape.size() >= 1 ? attrs_.kernel_shape[0] : W_shape[2];
  const int64_t kW = attrs_.kernel_shape.size() >= 2 ? attrs_.kernel_shape[1] : W_shape[3];

  int64_t pad_h = 0;
  int64_t pad_w = 0;
  int64_t pad_h_end = 0;
  int64_t pad_w_end = 0;
  if (attrs_.pads.size() >= 4) {
    pad_h = attrs_.pads[0];
    pad_w = attrs_.pads[1];
    pad_h_end = attrs_.pads[2];
    pad_w_end = attrs_.pads[3];
  }

  const int64_t stride_h = attrs_.strides.empty() ? 1 : attrs_.strides[0];
  const int64_t stride_w = attrs_.strides.size() < 2 ? 1 : attrs_.strides[1];
  const int64_t dilation_h = attrs_.dilations.empty() ? 1 : attrs_.dilations[0];
  const int64_t dilation_w = attrs_.dilations.size() < 2 ? 1 : attrs_.dilations[1];
  const int64_t group = attrs_.group;
  const int64_t offset_group = attrs_.offset_group;

  // Validate input shapes
  ORT_RETURN_IF_NOT(stride_h > 0 && stride_w > 0, "Strides must be positive.");
  ORT_RETURN_IF_NOT(dilation_h > 0 && dilation_w > 0, "Dilations must be positive.");
  ORT_RETURN_IF_NOT(kH > 0 && kW > 0, "Kernel shape must be positive.");
  ORT_RETURN_IF_NOT(group > 0, "group must be positive");
  ORT_RETURN_IF_NOT(offset_group > 0, "offset_group must be positive");

  const int64_t out_h = (H + pad_h + pad_h_end - dilation_h * (kH - 1) - 1) / stride_h + 1;
  const int64_t out_w = (W_in + pad_w + pad_w_end - dilation_w * (kW - 1) - 1) / stride_w + 1;

  // Checks
  ORT_RETURN_IF_NOT(W_shape.NumDimensions() == 4, "Weight must be 4D.");
  ORT_RETURN_IF_NOT(offset_shape.NumDimensions() == 4, "Offset must be 4D.");
  ORT_RETURN_IF_NOT(offset_shape[1] == offset_group * 2 * kH * kW,
                    "Offset channel count must be offset_group * 2 * kH * kW.");
  ORT_RETURN_IF_NOT(offset_shape[2] == out_h, "Offset spatial height must match output oH.");
  ORT_RETURN_IF_NOT(offset_shape[3] == out_w, "Offset spatial width must match output oW.");
  ORT_RETURN_IF_NOT(C % offset_group == 0, "Input channels must be divisible by offset_group.");
  ORT_RETURN_IF_NOT(C == W_shape[1] * group, "Input channels must match weight in channels * group.");
  ORT_RETURN_IF_NOT(M % group == 0, "Output channels must be divisible by group.");

  const bool use_mask = (mask != nullptr);
  if (use_mask) {
    ORT_RETURN_IF_NOT(mask->Shape().NumDimensions() == 4, "Mask must be 4D.");
    ORT_RETURN_IF_NOT(mask->Shape()[1] == offset_group * kH * kW, "Mask channel count must be offset_group * kH * kW.");
    ORT_RETURN_IF_NOT(mask->Shape()[2] == out_h, "Mask spatial height must match output oH.");
    ORT_RETURN_IF_NOT(mask->Shape()[3] == out_w, "Mask spatial width must match output oW.");
  }

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
  const int64_t kernel_dim = C / group * kernel_size; // The "K" dimension for GEMM (per group)

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
        col_buffer_ptr);

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
      math::GemmEx<T>(
          CblasNoTrans,
          CblasNoTrans,
          narrow<ptrdiff_t>(M / group),        // M
          narrow<ptrdiff_t>(output_image_size),// N
          narrow<ptrdiff_t>(kernel_dim),       // K
          static_cast<T>(1),                   // alpha
          weight_g,                            // A
          narrow<int>(kernel_dim),             // lda
          col_g,                               // B
          narrow<int>(output_image_size),      // ldb
          static_cast<T>(0),                   // beta
          Y_g,                                 // C
          narrow<int>(output_image_size),      // ldc
          thread_pool);
    }
  }

  // 3. Add Bias if present
  if (Bdata != nullptr) {
    for (int64_t n = 0; n < N; ++n) {
      T* Y_curr = Ydata + n * M * output_image_size;
      for (int64_t m = 0; m < M; ++m) {
        T bias_val = Bdata[m];
        for (int64_t i = 0; i < output_image_size; ++i) {
          Y_curr[m * output_image_size + i] += bias_val;
        }
      }
    }
  }

  return Status::OK();
}

// Explicit template instantiation for float and double
template class DeformConv<float>;

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    DeformConv,
    19,
    21,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 2)  // offset
        .InputMemoryType(OrtMemTypeCPUInput, 4),  // optional mask
    DeformConv<float>);

ONNX_CPU_OPERATOR_KERNEL(
    DeformConv,
    22,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 4),
    DeformConv<float>);

}  // namespace onnxruntime
