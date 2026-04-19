// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CUDA implementation of DeformConv (deformable convolution 2D).
// High-level pipeline matches CPU `nn/deform_conv.cc`: im2col then grouped GEMM then optional bias;
// this file hosts the EP and batch chunking; device kernels live in `deform_conv_impl.cu`.
//
// High-level pipeline (batch may be chunked for col_buffer memory; see GetNParallelImgs):
//   (1) Deformable im2col per chunk: DeformConvIm2ColImpl launches GPU kernels that fill col_buffer
//       (bilinear sampling + optional mask fused in threads; no separate sampling plan like CPU).
//   (2) Grouped strided batched GEMM: Y = W * Col via cuBLAS (row-major vs column-major mapping in ComputeInternal).
//   (3) Optional bias: add B[m] to each output channel map (DeformConvAddBiasImpl).
//
// Main difference vs CPU path: CPU builds an AoSoA bilinear plan once per image then reuses it across channels;
// CUDA recomputes bilinear samples in the im2col kernel while walking offset/mask tensors.

#include "core/providers/shared_library/provider_api.h"
#include "deform_conv.h"
#include "deform_conv_impl.h"

#include <algorithm>

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kMaxParallelImgs = 32;

// ceil(numer / denom) for numer >= 0, denom > 0 (integer, no floating point).
// Avoid (numer + denom - 1) / denom: numer near INT_MAX overflows signed int (UB in C++).
inline int CeilDiv(int numer, int denom) {
  return numer / denom + (numer % denom != 0 ? 1 : 0);
}

// Chooses DeformConv batch chunk size k (images per outer-loop iteration) given batch N and
// a hard cap T from temp-memory budget (target_parallel_imgs).
//
// Goals (in order):
//   1) Minimize the number of outer rounds I = ceil(N / k). Under k <= T, the minimum achievable
//      I is I* = ceil(N / min(N, T)) — take the largest allowed step min(N, T), same as always
//      using k = T when N > T, or one round when N <= T.
//   2) Among all k with ceil(N/k) == I*, pick k = ceil(N / I*) so chunk sizes are as balanced as
//      possible (last chunk is only slightly smaller than full chunks). k need not divide N; choosing
//      k = ceil(N / I*) instead of always k = T often shrinks col_buffer stride when a full-T last
//      chunk would leave a much smaller tail.
//
// Closed form: k_cap = min(N, T), I = ceil(N / k_cap), return ceil(N / I).
inline int GetDeformConvParallelChunkSize(int N, int T) {
  if (N <= 0 || T <= 0) return 1;
  const int k_cap = std::min(N, T);
  const int num_rounds = CeilDiv(N, k_cap);
  return CeilDiv(N, num_rounds);
}

// Returns the maximum temp memory (bytes) allowed for DeformConv's im2col + GEMM buffers.
// Uses a fraction of total GPU memory to avoid OOM while leaving room for weights, activations,
// and other ops. No CUDA API is called; total_global_mem is expected from cached device props.
//
// Formula:
//   budget = total_global_mem * kFraction
//   return clamp(budget, kMin, kMax)
// with kFraction = 0.1 (10%), kMin = 32 MiB, kMax = 2 GiB.
//
// Example results (effective_max_temp after clamp):
//   GPU              | totalGlobalMem | effective_max_temp
//   -----------------|----------------|--------------------
//   A100 80GB        | 80 GiB         | 2 GiB  (capped)
//   RTX 5080 16GB    | 16 GiB         | 1.6 GiB
//   RTX 4090 24GB    | 24 GiB         | 2 GiB  (capped)
//   RTX 3080 10GB    | 10 GiB         | 1 GiB
//   GTX 1060 6GB     | 6 GiB          | 614.4 MiB
//   GTX 1050 4GB     | 4 GiB          | 409.6 MiB
//   Jetson 2GB       | 2 GiB          | 204.8 MiB
size_t GetDeformConvEffectiveMaxTempBytes(size_t total_global_mem) {
  constexpr double kFraction = 0.1;
  constexpr size_t kMin = 32ULL * 1024 * 1024;
  constexpr size_t kMax = 2ULL * 1024 * 1024 * 1024;
  size_t budget = static_cast<size_t>(static_cast<double>(total_global_mem) * kFraction);
  return std::clamp(budget, kMin, kMax);
}

// Returns how many images to process in parallel per batch chunk for DeformConv.
//
// Temp budget → cap T (see below). Chunk size k = GetDeformConvParallelChunkSize(N, T): minimize
// outer-loop rounds first, then balance chunk sizes via ceil(N / ceil(N / min(N,T))).
// The host loop still uses cur_parallel = min(k, N - b), so k need not divide N.
//
// Formulas:
//   kernel_size / output_image_size come from validated common dims
//   bytes_per_image = output_image_size * C * kernel_size * sizeof(T)
//     (temp bytes per image: im2col col buffer only; GEMM writes directly to Y)
//   max_parallel_imgs_mem = max(1, floor(effective_max_temp / bytes_per_image))
//   target_parallel_imgs T = min(kMaxParallelImgs, max_parallel_imgs_mem)
//   return GetDeformConvParallelChunkSize(N, T)
template <typename T>
int GetNParallelImgs(const DeformConvParams& params, int64_t kernel_size, int64_t output_image_size, size_t total_global_mem) {
  const size_t effective_max_temp = GetDeformConvEffectiveMaxTempBytes(total_global_mem);
  const size_t bytes_per_image = SafeInt<size_t>(output_image_size) * params.C * kernel_size * sizeof(T);
  const int max_parallel_imgs_mem = std::max(1, static_cast<int>(effective_max_temp / std::max(size_t(1), bytes_per_image)));
  const int target_parallel_imgs = std::min(kMaxParallelImgs, max_parallel_imgs_mem);
  return GetDeformConvParallelChunkSize(narrow<int>(params.N), target_parallel_imgs);
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

  Tensor* Y = context->Output(0, {N, M, out_h, out_w});
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  DeformConvCommonDims common_dims;
  ORT_RETURN_IF_ERROR(DeformConvValidateAndComputeCommonDims(params, common_dims));
  const int64_t kernel_size = common_dims.kernel_size;
  const int64_t output_image_size = common_dims.output_image_size;
  const int64_t input_image_size = common_dims.input_image_size;
  const int64_t kernel_dim = common_dims.kernel_dim;
  const int n_parallel_imgs = GetNParallelImgs<T>(params, kernel_size, output_image_size, GetDeviceProp().totalGlobalMem);

  const int64_t col_stride = static_cast<int64_t>(n_parallel_imgs) * output_image_size;
  const int64_t col_buffer_size = (C * kernel_size) * col_stride;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  auto col_buffer = IAllocator::MakeUniquePtr<T>(alloc, SafeInt<size_t>(col_buffer_size));

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
    // Stride per full image along N: offset [N, offset_group*2*kH*kW, OH, OW] -> offset_group * 2*kH*kW * OH*OW floats.
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
    // Per batch image: m=output_image_size, n=M/group, k=kernel_dim; lda=cur_out_size, ldb=kernel_dim,
    // ldc=output_image_size (row-major Y slice [M/group, OH*OW]).
    //
    // cur_parallel==1: one strided-batched GEMM over all groups (single launch).
    // cur_parallel>1: per group, strided-batched GEMM with batch_count=cur_parallel; each batch writes one image
    // directly into NCHW Y (strideC = M * output_image_size), avoiding a temp buffer + scatter kernel.

    if (cur_parallel == 1) {
      // col_buffer is packed per iteration with the current chunk width (cur_out_size).
      // Using outer-scope col_stride (based on n_parallel_imgs) breaks tail chunks where
      // cur_out_size != col_stride (including one-image tails) when group > 1.
      const int64_t stride_col = kernel_dim * cur_out_size;
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
      const int64_t stride_a_col = output_image_size;
      const int64_t stride_b = 0;
      const int64_t stride_c_y = M * output_image_size;
      for (int64_t g = 0; g < group; ++g) {
        const T* W_g = Wdata + g * (M / group) * kernel_dim;
        const T* col_g = col_buffer.get() + g * kernel_dim * cur_out_size;
        T* Y_g = Ydata + b * M * output_image_size + g * (M / group) * output_image_size;

        CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
            cublas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            narrow<int>(output_image_size),
            narrow<int>(M / group),
            narrow<int>(kernel_dim),
            &alpha,
            reinterpret_cast<const CudaT*>(col_g),
            narrow<int>(cur_out_size),
            stride_a_col,
            reinterpret_cast<const CudaT*>(W_g),
            narrow<int>(kernel_dim),
            stride_b,
            &beta,
            reinterpret_cast<CudaT*>(Y_g),
            narrow<int>(output_image_size),
            stride_c_y,
            narrow<int>(cur_parallel),
            device_prop,
            UseTF32()));
      }
    }
  }

  if (Bdata != nullptr) {
    ORT_RETURN_IF_ERROR(DeformConvAddBiasImpl<T>(stream, Ydata, Bdata, N, M, out_h, out_w,
                                                 static_cast<int64_t>(device_prop.maxGridSize[1])));
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
      DeformConv<T>)

REGISTER_DEFORMCONV_KERNEL_TYPED(float);
REGISTER_DEFORMCONV_KERNEL_TYPED(double);
REGISTER_DEFORMCONV_KERNEL_TYPED(MLFloat16);

// BFloat16 only for opset 22; opset 19-21 do not support BFloat16.
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DeformConv,
    kOnnxDomain,
    22,
    BFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<BFloat16>()),
    DeformConv<BFloat16>);

#undef REGISTER_DEFORMCONV_KERNEL_TYPED

}  // namespace cuda
}  // namespace onnxruntime
