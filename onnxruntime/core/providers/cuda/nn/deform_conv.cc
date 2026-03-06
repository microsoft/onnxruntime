// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CUDA implementation of DeformConv (deformable convolution 2D).

#include "core/providers/shared_library/provider_api.h"
#include "deform_conv.h"
#include "deform_conv_impl.h"

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kMaxParallelImgs = 32;

// Returns the greatest divisor of n that is <= bound. Used to choose uniform batch chunk sizes.
// Fast path: if n % bound == 0 (common for batch 32/64/128), return immediately.
// When n >= bound^2, linear scan from bound down is O(bound). Otherwise divisor enumeration
// from 1 to sqrt(n) is O(sqrt(n)). Uses integer comparison (no sqrt) for branch decision.
int GetGreatestDivisorBelowBound(int n, int bound) {
  if (bound <= 0 || n <= 0) return 1;
  if (n % bound == 0) return bound;  // Fast path: batch is multiple of target

  // n >= bound^2 <=> bound <= sqrt(n) => linear scan is cheaper
  if (static_cast<int64_t>(n) >= static_cast<int64_t>(bound) * bound) {
    for (int k = bound - 1; k > 1; --k) {
      if (n % k == 0) return k;
    }
  } else {
    // n < bound^2 <=> bound > sqrt(n) => divisor enumeration is cheaper
    int best = 1;
    for (int i = 1; static_cast<int64_t>(i) * i <= static_cast<int64_t>(n); ++i) {
      if (n % i != 0) continue;
      const int q = n / i;
      if (q <= bound && q > best) best = q;
      if (i <= bound && i > best) best = i;
    }
    return best;
  }
  return 1;
}

// Returns effective max temp memory (bytes) for DeformConv batching.
// Uses 90% of free GPU memory with tiered cap; fallback 256MB if cudaMemGetInfo fails.
// Mirrors Conv's approach (conv_8.h); tiered limits avoid OOM on smaller GPUs.
// Called only when input/weight shapes change (see UpdateState).
size_t GetDeformConvEffectiveMaxTempBytes() {
  constexpr size_t kDefaultFallback = 256ULL * 1024 * 1024;
  constexpr size_t kMinTempMemSize = 32ULL * 1024 * 1024;

  size_t free_mem = 0, total_mem = 0;
  if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess || free_mem == 0) {
    return kDefaultFallback;
  }
  free_mem = static_cast<size_t>(static_cast<double>(free_mem) * 0.9);  // 10% fragmentation buffer

  size_t tier_cap;
  if (free_mem > 16ULL * 1024 * 1024 * 1024) {
    tier_cap = 2ULL * 1024 * 1024 * 1024;  // 16GB+ free → 2GB
  } else if (free_mem > 8ULL * 1024 * 1024 * 1024) {
    tier_cap = 1ULL * 1024 * 1024 * 1024;  // 8-16GB → 1GB
  } else if (free_mem > 4ULL * 1024 * 1024 * 1024) {
    tier_cap = 512ULL * 1024 * 1024;  // 4-8GB → 512MB
  } else if (free_mem > 2ULL * 1024 * 1024 * 1024) {
    tier_cap = 256ULL * 1024 * 1024;  // 2-4GB → 256MB
  } else {
    tier_cap = 128ULL * 1024 * 1024;  // <2GB → 128MB
  }

  return std::max(kMinTempMemSize, std::min(tier_cap, free_mem));
}

}  // namespace

template <typename T>
Status DeformConv<T>::UpdateState(OpKernelContext* context,
                                  const DeformConvParams& params,
                                  int& n_parallel_imgs) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto x_dims = X->Shape().AsShapeVector();
  const auto w_dims = W->Shape().AsShapeVector();

  bool input_dims_changed = (state_.last_x_dims != x_dims);
  bool w_dims_changed = (state_.last_w_dims != w_dims);

  if (input_dims_changed || w_dims_changed) {
    if (input_dims_changed) {
      state_.last_x_dims = gsl::make_span(x_dims);
    }
    if (w_dims_changed) {
      state_.last_w_dims = gsl::make_span(w_dims);
    }

    const int64_t kernel_size = params.kH * params.kW;
    const int64_t output_image_size = params.out_h * params.out_w;
    const size_t bytes_per_image = SafeInt<size_t>(output_image_size) * (params.C * kernel_size + params.M / params.group) * sizeof(T);
    const size_t effective_max_temp = GetDeformConvEffectiveMaxTempBytes();
    const int max_parallel_imgs_mem = std::max(1, static_cast<int>(effective_max_temp / std::max(size_t(1), bytes_per_image)));
    const int target_parallel_imgs = std::min(kMaxParallelImgs, max_parallel_imgs_mem);
    state_.cached_n_parallel_imgs = GetGreatestDivisorBelowBound(static_cast<int>(params.N), target_parallel_imgs);
  }

  n_parallel_imgs = state_.cached_n_parallel_imgs;
  return Status::OK();
}

template <typename T>
Status DeformConv<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto* offset = context->Input<Tensor>(2);
  const auto* B = context->Input<Tensor>(3);
  const auto* mask = context->Input<Tensor>(4);

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

  Tensor* Y = context->Output(0, {N, M, out_h, out_w});
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  int n_parallel_imgs;
  {
    std::lock_guard<std::mutex> lock(state_.mutex);
    ORT_RETURN_IF_ERROR(UpdateState(context, params, n_parallel_imgs));
  }

  const int64_t kernel_size = kH * kW;
  const int64_t output_image_size = out_h * out_w;
  const int64_t input_image_size = H * W_in;
  const int64_t kernel_dim = (C / group) * kernel_size;

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

    ORT_RETURN_IF_ERROR(DeformConvIm2ColImpl<T>(
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
        use_mask));

    // GEMM layout trick: compute Y = W * Col without physical transpose.
    //
    // Our data is row-major: W [M/group, kernel_dim], Col [kernel_dim, cur_out_size], Y [M/group, cur_out_size].
    // cuBLAS is column-major. Key insight: row-major A[M,K] in memory equals column-major A^T[K,M].
    // We compute Y^T = Col^T * W^T by passing Col as A and W as B, both OP_N (no transpose):
    //   - Col (row [kernel_dim, cur_out_size]) -> cuBLAS interprets as col-major [cur_out_size, kernel_dim] = Col^T
    //   - W (row [M/group, kernel_dim]) -> cuBLAS interprets as col-major [kernel_dim, M/group] = W^T
    //   - C = A*B = Col^T * W^T = (W*Col)^T = Y^T; C is col-major [cur_out_size, M/group] = Y in row-major
    //
    // m=cur_out_size, n=M/group, k=kernel_dim; lda=cur_out_size, ldb=kernel_dim, ldc=cur_out_size.
    //
    // cur_parallel==1: cur_out_size==output_image_size, C layout (pos, channel) matches NCHW Y_g[0,ch,pos] -> write
    // directly into Y_g. Use strided batched for all groups in one call.
    // cur_parallel>1: layouts differ -> write to gemm_output_buffer, then DeformConvCopyGemmOutputRowMajorToNCHW.

    const bool gemm_writes_directly = (cur_parallel == 1);
    if (gemm_writes_directly) {
      // Strided batched: one call for all groups. Strides between batches:
      const int64_t stride_col = kernel_dim * col_stride;   // = kernel_dim * output_image_size when cur_parallel==1
      const int64_t stride_weight = (M / group) * kernel_dim;
      const int64_t stride_y = (M / group) * output_image_size;
      CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
          cublas,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          narrow<int>(output_image_size),
          narrow<int>(M / group),
          narrow<int>(kernel_dim),
          &alpha,
          reinterpret_cast<const CudaT*>(col_buffer.get()),
          narrow<int>(output_image_size),
          stride_col,
          reinterpret_cast<const CudaT*>(Wdata),
          narrow<int>(kernel_dim),
          stride_weight,
          &beta,
          reinterpret_cast<CudaT*>(Ydata + b * M * output_image_size),
          narrow<int>(output_image_size),
          stride_y,
          narrow<int>(group),
          device_prop,
          UseTF32()));
    } else {
      // cur_parallel>1: GEMM output layout differs from NCHW; write to buffer then copy per group.
      for (int64_t g = 0; g < group; ++g) {
        const T* W_g = Wdata + g * (M / group) * kernel_dim;
        const T* col_g = col_buffer.get() + g * kernel_dim * col_stride;
        T* Y_g = Ydata + b * M * output_image_size + g * (M / group) * output_image_size;

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

        ORT_RETURN_IF_ERROR(DeformConvCopyGemmOutputRowMajorToNCHW<T>(
            stream,
            gemm_output_buffer.get(),
            Y_g,
            M,
            M / group,
            output_image_size,
            cur_parallel));
      }
    }
  }

  if (Bdata != nullptr) {
    ORT_RETURN_IF_ERROR(DeformConvAddBiasImpl<T>(stream, Ydata, Bdata, N, M, out_h, out_w));
  }

  return Status::OK();
}

#define REGISTER_DEFORMCONV_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      DeformConv,                                                                          \
      kOnnxDomain,                                                                         \
      19,                                                                                  \
      21,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DeformConv<T>);                                                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      DeformConv,                                                                          \
      kOnnxDomain,                                                                         \
      22,                                                                                  \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DeformConv<T>);

REGISTER_DEFORMCONV_KERNEL_TYPED(float)
REGISTER_DEFORMCONV_KERNEL_TYPED(double)
REGISTER_DEFORMCONV_KERNEL_TYPED(MLFloat16)
REGISTER_DEFORMCONV_KERNEL_TYPED(BFloat16)

#undef REGISTER_DEFORMCONV_KERNEL_TYPED

}  // namespace cuda
}  // namespace onnxruntime
