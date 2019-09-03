// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/qmath.h"
#include "core/providers/common.h"

namespace onnxruntime {

// only register this operator if low precision computation is enabled.
ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<uint8_t, uint8_t, int32_t>);

template <>
Status MatMulInteger<uint8_t, uint8_t, int32_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate zero points
  uint8_t a_offset = 0;
  uint8_t b_offset = 0;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = static_cast<int32_t>(*a_zero_point->template Data<uint8_t>());
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    b_offset = static_cast<int32_t>(*b_zero_point->template Data<uint8_t>());
  }

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    QGemmu8u8_s32(static_cast<int>(helper.M()),
                  static_cast<int>(helper.N()),
                  static_cast<int>(helper.K()),
                  a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                  static_cast<int>(helper.K()),
                  a_offset,
                  b->template Data<uint8_t>() + helper.RightOffsets()[i],
                  static_cast<int>(helper.N()),
                  b_offset,
                  y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
                  static_cast<int>(helper.N()),
                  nullptr);
  }

  return Status::OK();
}
}  // namespace onnxruntime
