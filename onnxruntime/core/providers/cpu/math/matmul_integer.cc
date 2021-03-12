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

  enum InIdx : int {
      A = 0,
      B = 1,
      Azero = 2,
      Bzero = 3
  };

  enum OutIdx : int { Y = 0 };

 protected:
  int GetBIdx() override { return InIdx::B; }
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
  MatMulComputeHelper helper;
  const auto* a = ctx->Input<Tensor>(InIdx::A);

  const uint8_t* b_data;
  bool b_signed;
  if (packed_b_) {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_));
    b_data = static_cast<const uint8_t*>(packed_b_.get());
    b_signed = b_is_signed_;
  } else {
    const Tensor* b = ctx->Input<Tensor>(InIdx::B);
    if (b == nullptr) {
      // the framework has checks to ensure this won't happen,
      // just need this to shutup static analysis.
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Required input B can not be null!");
    }
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
    b_data = static_cast<const uint8_t*>(b->DataRaw());
    b_signed = b->IsDataType<int8_t>();
  }

  Tensor* y = ctx->Output(OutIdx::Y, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  // validate zero points
  uint8_t a_offset = 0;
  uint8_t b_offset = 0;
  const auto* a_zero_point = ctx->Input<Tensor>(InIdx::Azero);
  if (a_zero_point != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *a_zero_point->template Data<uint8_t>();
  }
  const auto* b_zero_point = ctx->Input<Tensor>(InIdx::Bzero);
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
  gemm_params.BIsPacked = bool(packed_b_);
  gemm_params.BIsSigned = b_signed;

  const auto* a_data = a->template Data<uint8_t>();
  auto* y_data = y->template MutableData<int32_t>();

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    gemm_params.A = a_data + helper.LeftOffsets()[i];
    gemm_params.B = b_data + (gemm_params.BIsPacked ? 0UL : helper.RightOffsets()[i]);
    gemm_params.C = y_data + helper.OutputOffsets()[i];
    MlasGemm(&gemm_params, ctx->GetOperatorThreadPool());
  }

  return Status::OK();
}

}  // namespace onnxruntime
