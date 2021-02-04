// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Gemm,                                                       \
      kOnnxDomain,                                                \
      7,                                                          \
      8,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Gemm,                                                       \
      kOnnxDomain,                                                \
      9,                                                          \
      10,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Gemm,                                                       \
      kOnnxDomain,                                                \
      11,                                                         \
      12,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Gemm,                                                       \
      kOnnxDomain,                                                \
      13,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_KERNEL_TYPED(BFloat16)
#endif

template <typename T>
Status Gemm<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const auto* X = ctx->Input<Tensor>(0);
  const auto* W = ctx->Input<Tensor>(1);
  const auto* B = ctx->Input<Tensor>(2);
  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(X->Shape(), trans_A_, W->Shape(), trans_B_, B != nullptr ? B->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  int K = gsl::narrow_cast<int>(helper.K());
  auto* Y = ctx->Output(0, {M, N});
  CudaT* out_data = reinterpret_cast<CudaT*>(Y->template MutableData<T>());

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);
  auto& device_prop = GetDeviceProp();
  // broadcast bias if needed and is present
  if (beta_ != 0 && B != nullptr) {
    auto& b_shape = B->Shape();
    const CudaT* b_data = reinterpret_cast<const CudaT*>(B->template Data<T>());
    if (b_shape.Size() == 1) {
      // if B is (), (1,) or (1, 1), broadcast the scalar
      CUBLAS_RETURN_IF_ERROR(cublasCopyHelper(
          CublasHandle(),
          M * N,
          b_data,
          0,
          out_data,
          1));
    } else if (b_shape.NumDimensions() == 1 || b_shape[0] == 1) {
      // B is (N,) or (1, N), broadcast using Y(N,M) = 1 * B(N,1) x ones(1,M) + 0 * Y
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          CublasHandle(),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          N, M, 1,
          /*alpha*/ &one,
          b_data, N,
          GetConstOnes<CudaT>(M), 1,
          /*beta*/ &zero,
          out_data, N, device_prop));
    } else if (b_shape.NumDimensions() == 2 && b_shape[1] == 1) {
      // B is (M, 1), broadcast using Y(N,M) = 1 * ones(N,1) x B(1,M) + 0 * Y
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          CublasHandle(),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          N, M, 1,
          /*alpha*/ &one,
          GetConstOnes<CudaT>(N), N,
          b_data, 1,
          /*beta*/ &zero,
          out_data, N, device_prop));
    } else {
      // B is (M, N), no broadcast needed.
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(out_data, b_data, M * N * sizeof(T), cudaMemcpyDeviceToDevice));
    }
  }

  CudaT alpha = ToCudaType<T>::FromFloat(alpha_);
  CudaT beta = ToCudaType<T>::FromFloat(beta_);
  // Gemm, note that CUDA assumes col-major, so Y(N,M) = alpha * op(W) x op(X) + beta * Y
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      CublasHandle(),
      trans_B_ ? CUBLAS_OP_T : CUBLAS_OP_N,
      trans_A_ ? CUBLAS_OP_T : CUBLAS_OP_N,
      N, M, K,
      &alpha,
      reinterpret_cast<const CudaT*>(W->template Data<T>()),
      (trans_B_ ? K : N),
      reinterpret_cast<const CudaT*>(X->template Data<T>()),
      (trans_A_ ? M : K),
      // ideally we need to set the output buffer contents to 0 if bias is missing,
      // but passing 0 for beta is cheaper and it will ignore any junk in the output buffer
      B != nullptr ? &beta : &zero,
      out_data, N, device_prop));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
