// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <iostream>
#include "core/providers/cann/math/gemm.h"

using onnxruntime::common::Status;
namespace onnxruntime {
namespace cann {

template <typename T>
Status Gemm<T>::ComputeInternal(OpKernelContext* context) const {
  const auto* A = context->Input<Tensor>(0);
  const auto* B = context->Input<Tensor>(1);
  const auto* C = context->Input<Tensor>(2);

  GemmHelper helper(A->Shape(), trans_A_, B->Shape(), trans_B_, C != nullptr ? C->Shape() : TensorShape({}));
  if (!helper.State().IsOK())
    return helper.State();

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  int K = gsl::narrow_cast<int>(helper.K());

  auto* Y = context->Output(0, {M, N});

  // broadcast C if needed.
  if (beta_ != 0 && C != nullptr) {
    if (C->Shape().Size() == 1) {
      // C is (), (1,) or (1, 1), fill the scalar to Y
      ORT_RETURN_IF_ERROR(Fill<T>(Y, const_cast<void*>(C->DataRaw())));
    } else if (C->Shape() == Y->Shape()) {
      // C is (M, N), no broadcast needed.
      CANN_RETURN_IF_ERROR(aclrtMemcpyAsync(Y->MutableDataRaw(),
                                            Y->SizeInBytes(),
                                            const_cast<void*>(C->DataRaw()),
                                            Y->SizeInBytes(),
                                            ACL_MEMCPY_DEVICE_TO_DEVICE,
                                            Stream()));
    } else {
      // others, broadcast needed.
      ORT_RETURN_IF_ERROR(Broadcast<T>(C, Y, Y->MutableDataRaw()));
    }
  }

  const aclDataType aclType = getACLType<T>();

  T alpha = ToCannType<T>::FromFloat(alpha_);
  T beta = ToCannType<T>::FromFloat(beta_);
  IAllocatorUniquePtr<void> pAlpha = GetScratchBuffer<void>(sizeof(T));
  IAllocatorUniquePtr<void> pBeta = GetScratchBuffer<void>(sizeof(T));
  CANN_RETURN_IF_ERROR(aclrtMemcpy(pAlpha.get(), sizeof(T), &alpha, sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));
  CANN_RETURN_IF_ERROR(aclrtMemcpy(pBeta.get(), sizeof(T), &beta, sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE));

  ORT_RETURN_IF_ERROR(aclrtblasGemmEx(
      trans_A_ ? ACL_TRANS_T : ACL_TRANS_N,
      trans_B_ ? ACL_TRANS_T : ACL_TRANS_N,
      ACL_TRANS_N,
      M,
      N,
      K,
      pAlpha.get(),
      const_cast<void*>(A->DataRaw()), -1, aclType,
      const_cast<void*>(B->DataRaw()), -1, aclType,
      pBeta.get(),
      Y->MutableDataRaw(), -1, aclType,
      ACL_COMPUTE_HIGH_PRECISION,
      Stream()));

  return Status::OK();
}

#define REGISTER_GEMM_VERSIONED_TYPED_KERNEL(startver, endver, T)                          \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                 \
      Gemm,                                                                                \
      kOnnxDomain,                                                                         \
      startver,                                                                            \
      endver,                                                                              \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

#define REGISTER_GEMM_TYPED_KERNEL(ver, T)                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      Gemm,                                                                                \
      kOnnxDomain,                                                                         \
      ver,                                                                                 \
      T,                                                                                   \
      kCannExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Gemm<T>);

REGISTER_GEMM_VERSIONED_TYPED_KERNEL(7, 8, MLFloat16)
REGISTER_GEMM_VERSIONED_TYPED_KERNEL(9, 10, MLFloat16)
REGISTER_GEMM_VERSIONED_TYPED_KERNEL(11, 12, MLFloat16)
REGISTER_GEMM_TYPED_KERNEL(13, MLFloat16)

}  // namespace cann
}  // namespace onnxruntime
