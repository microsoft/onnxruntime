// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CUDA implementation of DeformConv (deformable convolution 2D).

#include "core/providers/shared_library/provider_api.h"
#include "deform_conv.h"
#include "deform_conv_impl.h"

#include "core/common/narrow.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kMaxParallelImgs = 32;

int GetGreatestDivisorBelowBound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

}  // namespace

template <typename T>
Status DeformConv<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto* offset = context->Input<Tensor>(2);
  const auto* B = context->Input<Tensor>(3);
  const auto* mask = context->Input<Tensor>(4);

  const auto& X_shape = X->Shape();
  const auto& W_shape = W->Shape();
  const auto& offset_shape = offset->Shape();

  const int64_t N = X_shape[0];
  const int64_t C = X_shape[1];
  const int64_t H = X_shape[2];
  const int64_t W_in = X_shape[3];

  const int64_t M = W_shape[0];
  const int64_t kH = attrs_.kernel_shape.size() >= 1 ? attrs_.kernel_shape[0] : W_shape[2];
  const int64_t kW = attrs_.kernel_shape.size() >= 2 ? attrs_.kernel_shape[1] : W_shape[3];

  int64_t pad_h = 0, pad_w = 0, pad_h_end = 0, pad_w_end = 0;
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

  const int64_t out_h = (H + pad_h + pad_h_end - dilation_h * (kH - 1) - 1) / stride_h + 1;
  const int64_t out_w = (W_in + pad_w + pad_w_end - dilation_w * (kW - 1) - 1) / stride_w + 1;

  ORT_RETURN_IF_NOT(W_shape.NumDimensions() == 4, "Weight must be 4D.");
  ORT_RETURN_IF_NOT(offset_shape.NumDimensions() == 4, "Offset must be 4D.");
  ORT_RETURN_IF_NOT(offset_shape[1] == offset_group * 2 * kH * kW,
                    "Offset channel count must be offset_group * 2 * kH * kW.");
  ORT_RETURN_IF_NOT(offset_shape[2] == out_h && offset_shape[3] == out_w,
                    "Offset spatial dims must match output.");
  ORT_RETURN_IF_NOT(C % offset_group == 0, "Input channels must be divisible by offset_group.");
  ORT_RETURN_IF_NOT(C == W_shape[1] * group, "Input channels must match weight in channels * group.");
  ORT_RETURN_IF_NOT(M % group == 0, "Output channels must be divisible by group.");

  const bool use_mask = (mask != nullptr);
  if (use_mask) {
    ORT_RETURN_IF_NOT(mask->Shape().NumDimensions() == 4, "Mask must be 4D.");
    ORT_RETURN_IF_NOT(mask->Shape()[1] == offset_group * kH * kW, "Mask channel count invalid.");
    ORT_RETURN_IF_NOT(mask->Shape()[2] == out_h && mask->Shape()[3] == out_w, "Mask spatial dims must match output.");
  }

  Tensor* Y = context->Output(0, {N, M, out_h, out_w});
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t kernel_size = kH * kW;
  const int64_t output_image_size = out_h * out_w;
  const int64_t input_image_size = H * W_in;
  const int64_t kernel_dim = (C / group) * kernel_size;

  // Calculate memory usage per image to avoid OOM with large images
  // col_buffer: C * kernel_size * output_image_size
  // gemm_output_buffer: (M / group) * output_image_size
  // We use a safe max(1, ...) for bytes_per_image to avoid division by zero in edge cases
  const size_t bytes_per_image = SafeInt<size_t>(output_image_size) * (C * kernel_size + M / group) * sizeof(T);

  // Heuristic: limit temp memory to 256MB per chunk to balance parallelism and memory usage.
  // For small images, this allows up to kMaxParallelImgs (32).
  // For large images (4K/8K), this restricts parallelism to 1 to prevent OOM.
  constexpr size_t kMaxTempMemSize = 256 * 1024 * 1024;
  const int max_parallel_imgs_mem = std::max(1, static_cast<int>(kMaxTempMemSize / std::max(size_t(1), bytes_per_image)));
  const int target_parallel_imgs = std::min(kMaxParallelImgs, max_parallel_imgs_mem);

  const int n_parallel_imgs = GetGreatestDivisorBelowBound(static_cast<int>(N), target_parallel_imgs);
  const int64_t col_stride = static_cast<int64_t>(n_parallel_imgs) * output_image_size;
  const int64_t col_buffer_size = (C * kernel_size) * col_stride;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  auto col_buffer = IAllocator::MakeUniquePtr<T>(alloc, SafeInt<size_t>(col_buffer_size));
  // Removed col_transposed allocation as we avoid physical transpose.
  auto gemm_output_buffer = IAllocator::MakeUniquePtr<T>(alloc, SafeInt<size_t>((M / group) * col_stride));

  const T* Xdata = X->Data<T>();
  const T* Wdata = W->Data<T>();
  const T* offset_data = offset->Data<T>();
  const T* mask_data = use_mask ? mask->Data<T>() : nullptr;
  T* Ydata = Y->MutableData<T>();
  const T* Bdata = (B != nullptr) ? B->Data<T>() : nullptr;

  cudaStream_t stream = Stream(context);
  cublasHandle_t cublas = GetCublasHandle(context);
  const cudaDeviceProp& device_prop = GetDeviceProp();
  CudaT alpha = ToCudaType<T>::FromFloat(1.0f);
  CudaT beta = ToCudaType<T>::FromFloat(0.0f);

  for (int64_t b = 0; b < N; b += n_parallel_imgs) {
    const int cur_parallel = static_cast<int>(std::min(static_cast<int64_t>(n_parallel_imgs), N - b));
    const int64_t cur_out_size = static_cast<int64_t>(cur_parallel) * output_image_size;

    const T* X_block = Xdata + b * (C * input_image_size);
    const T* offset_block = offset_data + b * (offset_group * 2 * kernel_size * output_image_size);
    const T* mask_block = use_mask ? (mask_data + b * (offset_group * kernel_size * output_image_size)) : nullptr;

    DeformConvIm2ColImpl<T>(
        stream,
        X_block,
        offset_block,
        mask_block,
        col_buffer.get(),
        cur_parallel,
        C,
        H,
        W_in,
        kH,
        kW,
        out_h,
        out_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        offset_group,
        use_mask);

    for (int64_t g = 0; g < group; ++g) {
      const T* W_g = Wdata + g * (M / group) * kernel_dim;
      const T* col_g = col_buffer.get() + g * kernel_dim * col_stride;
      T* Y_g = Ydata + b * M * output_image_size + g * (M / group) * output_image_size;

      // Avoid physical transpose by using cuBLAS OP_N/OP_N logic.
      // We want Y = W * Col.
      // W is [M/group, kernel_dim] (Row-Major).
      // Col is [kernel_dim, cur_out_size] (Row-Major).
      // We compute Y^T = Col^T * W^T.
      // Col^T (Col-Major [cur_out_size, kernel_dim]) is exactly Col (Row-Major [kernel_dim, cur_out_size]) in memory.
      // W^T (Col-Major [kernel_dim, M/group]) is exactly W (Row-Major [M/group, kernel_dim]) in memory.
      // Result Y^T is Col-Major [cur_out_size, M/group].
      // In memory, Y^T (Col-Major) is exactly Y (Row-Major [M/group, cur_out_size]).
      // So we get Y in Row-Major layout.

      // A = Col (Row-Major [kernel_dim, cur_out_size]) -> interpreted as Col-Major [cur_out_size, kernel_dim].
      // B = W (Row-Major [M/group, kernel_dim]) -> interpreted as Col-Major [kernel_dim, M/group].
      // C = A * B = Col^T * W^T = Y^T.
      // C is Col-Major [cur_out_size, M/group].
      // m = cur_out_size, n = M/group, k = kernel_dim.
      // lda = cur_out_size.
      // ldb = kernel_dim.
      // ldc = cur_out_size.

      CUBLAS_RETURN_IF_ERROR((cublasGemmHelper(
          cublas,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          narrow<int>(cur_out_size),
          narrow<int>(M / group),
          narrow<int>(kernel_dim),
          &alpha,
          reinterpret_cast<const CudaT*>(col_g),
          narrow<int>(cur_out_size),
          reinterpret_cast<const CudaT*>(W_g),
          narrow<int>(kernel_dim),
          &beta,
          reinterpret_cast<CudaT*>(gemm_output_buffer.get()),
          narrow<int>(cur_out_size),
          device_prop,
          UseTF32())));

      // The output gemm_output_buffer is now Row-Major [M/group, cur_out_size].
      // We need to copy it to Y_g (NCHW).
      DeformConvCopyGemmOutputRowMajorToNCHW<T>(
          stream,
          gemm_output_buffer.get(),
          Y_g,
          M,
          M / group,
          output_image_size,
          cur_parallel);
    }
  }

  if (Bdata != nullptr) {
    DeformConvAddBiasImpl<T>(stream, Ydata, Bdata, N, M, out_h, out_w);
  }

  return Status::OK();
}

#define REGISTER_KERNEL_TYPED(T)                                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                        \
      DeformConv,                                                                                 \
      kOnnxDomain,                                                                                \
      19,                                                                                         \
      21,                                                                                         \
      T,                                                                                          \
      kCudaExecutionProvider,                                                                     \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),        \
      DeformConv<T>);                                                                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                  \
      DeformConv,                                                                                 \
      kOnnxDomain,                                                                                \
      22,                                                                                         \
      T,                                                                                          \
      kCudaExecutionProvider,                                                                    \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),        \
      DeformConv<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

}  // namespace cuda
}  // namespace onnxruntime
