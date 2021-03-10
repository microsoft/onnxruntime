// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer_base.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

namespace onnxruntime {

class MatMulInteger final : public MatMulIntegerBase {
 public:
  MatMulInteger(const OpKernelInfo& info) : MatMulIntegerBase(info) {}

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
  const auto* a = ctx->Input<Tensor>(0);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(1);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b ? b->Shape() : b_shape_));
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

  MLAS_GEMM_U8X8_PARAMETERS gemm_params;
  gemm_params.M = static_cast<size_t>(helper.M());
  gemm_params.N = static_cast<size_t>(helper.N());
  gemm_params.K = static_cast<size_t>(helper.K());
  gemm_params.lda = gemm_params.K;
  gemm_params.ZeroPointA = a_offset;
  gemm_params.ldb = gemm_params.N;
  gemm_params.ZeroPointB = &b_offset;
  gemm_params.ldc = gemm_params.N;

  const auto* a_data = a->template Data<uint8_t>();
  auto* y_data = y->template MutableData<int32_t>();

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    gemm_params.A = a_data + helper.LeftOffsets()[i];
    if (packed_b_) {
      gemm_params.B = packed_b_.get();
      gemm_params.BIsPacked = true;
      gemm_params.BIsSigned = b_is_signed_;
    } else if (b != nullptr) {
      gemm_params.B = static_cast<const uint8_t*>(b->DataRaw()) +  + helper.RightOffsets()[i];
      gemm_params.BIsSigned = b->IsDataType<int8_t>();
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input B should not be null.");
    }
    gemm_params.C = y_data + helper.OutputOffsets()[i];
    MlasGemm(&gemm_params, ctx->GetOperatorThreadPool());
  }

  return Status::OK();
}

}  // namespace onnxruntime
