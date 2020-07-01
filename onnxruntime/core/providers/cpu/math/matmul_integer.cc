// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types_internal.h"
#include "core/providers/cpu/math/matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/qmath.h"
#include "core/providers/common.h"

namespace onnxruntime {

// only register this operator if low precision computation is enabled.
ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<uint8_t, uint8_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<uint8_t, int8_t>);

template <typename T1, typename T2>
Status MatMulInteger<T1, T2>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate zero points
  T1 a_offset = 0;
  T2 b_offset = 0;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *a_zero_point->template Data<T1>();
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    b_offset = *b_zero_point->template Data<T2>();
  }

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    QGemm(static_cast<int>(helper.M()),
          static_cast<int>(helper.N()),
          static_cast<int>(helper.K()),
          a->template Data<T1>() + helper.LeftOffsets()[i],
          static_cast<int>(helper.K()),
          a_offset,
          b->template Data<T2>() + helper.RightOffsets()[i],
          static_cast<int>(helper.N()),
          b_offset,
          y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
          static_cast<int>(helper.N()),
          thread_pool);
  }
  return Status::OK();
}

}  // namespace onnxruntime
