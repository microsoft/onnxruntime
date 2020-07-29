// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/matmul_integer16.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger16,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int16_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int16_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger16<int16_t, int16_t, int32_t>);

template <>
Status MatMulInteger16<int16_t, int16_t, int32_t>::Compute(OpKernelContext* ctx) const {
  auto A = ctx->Input<Tensor>(0);
  auto B = ctx->Input<Tensor>(1);
  ORT_ENFORCE(A != nullptr && B != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0)
    return Status::OK();

  for (int i = 0; i < static_cast<int>(helper.OutputOffsets().size()); i++) {
    EigenCastGEMM<int16_t, int16_t, int32_t>(
        A->template Data<int16_t>() + helper.LeftOffsets()[i],
        B->template Data<int16_t>() + helper.RightOffsets()[i],
        Y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
