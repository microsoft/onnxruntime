// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/providers/common.h"

namespace onnxruntime {

class MatMulInteger final : public OpKernel {
 public:
  MatMulInteger(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger);

Status MatMulInteger::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const auto* a = ctx->Input<Tensor>(0);
  const auto* b = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  // validate zero points
  uint8_t a_offset = 0;
  uint8_t b_offset = 0;
  const auto* a_zero_point = ctx->Input<Tensor>(2);
  if (a_zero_point != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *a_zero_point->template Data<uint8_t>();
  }
  const auto* b_zero_point = ctx->Input<Tensor>(3);
  if (b_zero_point != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    b_offset = *static_cast<const uint8_t*>(b_zero_point->DataRaw());
  }

  const auto* a_data = a->template Data<uint8_t>();
  const auto* b_data = static_cast<const uint8_t*>(b->DataRaw());
  const bool b_is_signed = b->IsDataType<int8_t>();
  auto* y_data = y->template MutableData<int32_t>();

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    QGemm(static_cast<int>(helper.M()),
          static_cast<int>(helper.N()),
          static_cast<int>(helper.K()),
          a_data + helper.LeftOffsets()[i],
          static_cast<int>(helper.K()),
          a_offset,
          b_data + helper.RightOffsets()[i],
          static_cast<int>(helper.N()),
          b_offset,
          b_is_signed,
          y_data + helper.OutputOffsets()[i],
          static_cast<int>(helper.N()),
          thread_pool);
  }
  return Status::OK();
}

}  // namespace onnxruntime
