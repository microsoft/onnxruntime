// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "matmul_helper.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
  MatMul,
  1,
  9,
  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
  MatMul<float>);

template <>
Status MatMul<float>::Compute(OpKernelContext* ctx) const {
  const Tensor* left_X = ctx->Input<Tensor>(0);
  const Tensor* right_X = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ONNXRUNTIME_RETURN_IF_ERROR(helper.Compute(left_X->Shape(), right_X->Shape()));

  Tensor* Y = ctx->Output(0, helper.OutputShape());

  // TODO: replace it with GemmBatch for performance, it's OK for now as GemmBatch unrolls as well
  for (int i = 0; i < helper.OutputOffsets().size(); i++) {
    math::Gemm<float, CPUMathUtil>(
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(helper.M()),
        static_cast<int>(helper.N()),
        static_cast<int>(helper.K()),
        /* alpha */ 1.0f,
        left_X->template Data<float>() + helper.LeftOffsets()[i],
        right_X->template Data<float>() + helper.RightOffsets()[i],
        /* beta */ 0.0f,
        Y->template MutableData<float>() + helper.OutputOffsets()[i],
        &CPUMathUtil::Instance());
  }

  return Status::OK();
}

}  // namespace onnxruntime
