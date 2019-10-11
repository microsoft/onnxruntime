// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/cpu/math/matmul.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "matmul_helper.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    1, 8,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    1, 8,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

// opset 9 supports more types
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    9,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    MatMul<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    9,
    uint32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint32_t>()),
    MatMul<uint32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    9,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    MatMul<int64_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MatMul,
    9,
    uint64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint64_t>()),
    MatMul<uint64_t>);

template <typename T>
Status MatMul<T>::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  concurrency::ThreadPool* thread_pool = ctx_internal->GetOperatorThreadPool();

  const auto* left_X = ctx->Input<Tensor>(0);
  const auto* right_X = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape()));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  size_t max_len = helper.OutputOffsets().size();
  for (size_t i = 0; i < max_len; i++) {
    math::MatMul<T>(
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()),
        left_X->template Data<T>() + helper.LeftOffsets()[i],
        right_X->template Data<T>() + helper.RightOffsets()[i],
        Y->template MutableData<T>() + helper.OutputOffsets()[i], thread_pool);
  }

  return Status::OK();
}

}  // namespace onnxruntime
