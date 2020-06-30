// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer_to_float.h"

#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_MATMUL_INTEGER_TO_FLOAT(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      MatMulIntegerToFloat,                                           \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),  \
      MatMulIntegerToFloat<uint8_t, T>);

REGISTER_MATMUL_INTEGER_TO_FLOAT(int8_t)
REGISTER_MATMUL_INTEGER_TO_FLOAT(uint8_t)

template <typename T1, typename T2>
Status MatMulIntegerToFloat<T1, T2>::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  const Tensor* a_scale_tensor = ctx->Input<Tensor>(2);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale_tensor),
              "MatmulInteger : input A scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");

  float a_scale = *a_scale_tensor->template Data<float>();

  const Tensor* b_scale_tensor = ctx->Input<Tensor>(3);
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale_tensor),
              "MatmulInteger : input B scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");

  float b_scale = *b_scale_tensor->template Data<float>();

  float multiplier = a_scale * b_scale;

  // validate zero points
  T1 a_zp = 0;
  const Tensor* a_zp_tensor = ctx->Input<Tensor>(4);
  if (a_zp_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zp_tensor),
                "MatmulInteger : input A zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    a_zp = *a_zp_tensor->template Data<T1>();
  }

  T2 b_zp = 0;
  const Tensor* b_zp_tensor = ctx->Input<Tensor>(5);
  if (b_zp_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zp_tensor),
                "MatmulInteger : input B zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    b_zp = *b_zp_tensor->template Data<T2>();
  }

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  const Tensor* bias_tensor = ctx->Input<Tensor>(6);

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();
  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    QGemm(static_cast<int>(helper.M()),
          static_cast<int>(helper.N()),
          static_cast<int>(helper.K()),
          a->template Data<T1>() + helper.LeftOffsets()[i],
          static_cast<int>(helper.K()),
          a_zp,
          b->template Data<T2>() + helper.RightOffsets()[i],
          static_cast<int>(helper.N()),
          b_zp,
          Y->template MutableData<float>() + helper.OutputOffsets()[i],
          static_cast<int>(helper.N()),
          &multiplier,
          nullptr != bias_tensor ? bias_tensor->Data<float>() : nullptr,
          thread_pool);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
