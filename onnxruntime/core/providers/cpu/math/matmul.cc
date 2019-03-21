// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "matmul_helper.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    1, 9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MatMul<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    1, 9,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    MatMul<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9, 9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    MatMul<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9, 9,
    uint32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint32_t>()),
    MatMul<uint32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9, 9,
    int64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    MatMul<int64_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MatMul,
    9, 9,
    uint64_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint64_t>()),
    MatMul<uint64_t>);

template <typename T>
Status MatMul<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape()));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // TODO: replace it with GemmBatch for performance, it's OK for now as GemmBatch unrolls as well
  size_t max_len = helper.OutputOffsets().size();
  for (size_t i = 0; i < max_len; i++) {
    math::Gemm<T, CPUMathUtil>(
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()),
        /* alpha */ 1.0f,
        left_X->template Data<T>() + helper.LeftOffsets()[i],
        right_X->template Data<T>() + helper.RightOffsets()[i],
        /* beta */ 0.0f,
        Y->template MutableData<T>() + helper.OutputOffsets()[i],
        &CPUMathUtil::Instance());
  }

  return Status::OK();
}

}  // namespace onnxruntime
