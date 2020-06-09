// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types_internal.h"
#include "core/providers/cpu/math/matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"
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
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<uint8_t>);

template <typename T1>
Status MatMulInteger<T1>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  const auto* a = ctx->Input<Tensor>(0);
  const auto* b = ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  auto* y = ctx->Output(0, helper.OutputShape());

  MLAS_GEMM_U8X8_PARAMETERS gemm_parameters = {};

  gemm_parameters.BTypeIsSigned = b->IsDataType<int8_t>();

  gemm_parameters.M = static_cast<size_t>(helper.M());
  gemm_parameters.N = static_cast<size_t>(helper.N());
  gemm_parameters.K = static_cast<size_t>(helper.K());

  gemm_parameters.lda = gemm_parameters.K;
  gemm_parameters.ldb = gemm_parameters.N;
  gemm_parameters.ldc = gemm_parameters.N;

  // validate optional zero points
  const auto* a_zero_point = ctx->Input<Tensor>(2);
  if (a_zero_point != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    gemm_parameters.offa = *a_zero_point->template Data<T1>();
  }
  const auto* b_zero_point = ctx->Input<Tensor>(3);
  if (b_zero_point != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    gemm_parameters.offb = *static_cast<const uint8_t*>(b_zero_point->DataRaw());
  }

  const auto* a_data = a->template Data<T1>();
  const auto* b_data = static_cast<const uint8_t*>(b->DataRaw());
  auto* y_data = y->template MutableData<int32_t>();

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    gemm_parameters.A = a_data + helper.LeftOffsets()[i];
    gemm_parameters.B = b_data + helper.RightOffsets()[i];
    gemm_parameters.C = y_data + helper.OutputOffsets()[i];

    QGemm(gemm_parameters, thread_pool);
  }
  return Status::OK();
}

}  // namespace onnxruntime
