// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/gemm_fast_gelu.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl.h"
#include "contrib_ops/rocm/bert/transformer_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GemmFastGelu,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GemmFastGelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
GemmFastGelu<T>::GemmFastGelu(const OpKernelInfo& op_kernel_info) : RocmKernel(op_kernel_info) {
  const TransformerOptions* options = TransformerOptions::GetInstance();
  use_half2_ = !options->DisableHalf2();
}

template <typename T>
Status GemmFastGelu<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToHipType<T>::MappedType HipT;

  const auto* X = ctx->Input<Tensor>(0);
  const auto* W = ctx->Input<Tensor>(1);
  const auto* bias = ctx->Input<Tensor>(2);

  GemmHelper helper(X->Shape(), 0, W->Shape(), 0, TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  int K = gsl::narrow_cast<int>(helper.K());
  auto* Y = ctx->Output(0, {M, N});
  auto gemm_buffer = GetScratchBuffer<T>(M * N);

  HipT zero = ToHipType<T>::FromFloat(0.0f);
  HipT alpha = ToHipType<T>::FromFloat(1.0f);
  // Gemm, note that HIP assumes col-major, so Y(N,M) = alpha * op(W) x op(X) + beta * Y
  ROCBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
      RocblasHandle(),
      rocblas_operation_none,
      rocblas_operation_none,
      N, M, K,
      &alpha,
      reinterpret_cast<const HipT*>(W->template Data<T>()),
      N,
      reinterpret_cast<const HipT*>(X->template Data<T>()),
      K,
      // ideally we need to set the output buffer contents to 0 if bias is missing,
      // but passing 0 for beta is cheaper and it will ignore any junk in the output buffer
      &zero,
      reinterpret_cast<HipT*>(gemm_buffer.get()), N));

  int64_t fast_gelu_input_length = Y->Shape().Size();
  int64_t bias_length = (nullptr == bias) ? 0 : bias->Shape().Size();
  typedef typename ToHipType<T>::MappedType HipT;

  if (!LaunchFastGeluKernel<HipT>(GetDeviceProp(),
                                   Stream(),
                                   static_cast<int>(fast_gelu_input_length),
                                   static_cast<int>(bias_length),
                                   reinterpret_cast<HipT*>(gemm_buffer.get()),
                                   (nullptr != bias) ? reinterpret_cast<const HipT*>(bias->template Data<T>()) : nullptr,
                                   reinterpret_cast<HipT*>(Y->template MutableData<T>()),
                                   use_half2_)) {
    HIP_CALL(hipGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
