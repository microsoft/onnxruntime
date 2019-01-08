// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#define ENABLE_LOW_PRECISION_COMPUTATION

#if defined(ENABLE_LOW_PRECISION_COMPUTATION)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#include "../../cmake/external/gemmlowp/public/gemmlowp.h"
#endif  // USE_LOW_PRECISION_LIB

namespace onnxruntime {
namespace contrib {

#ifdef ENABLE_LOW_PRECISION_COMPUTATION
// only register this operator if low precision computation is enabled.
ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<uint8_t, uint8_t, int32_t>);

Status GemmlowpMultiply(OpKernelContext* ctx, const uint8_t* lhs_data, const uint8_t* rhs_data,
                        int32_t* result_data, const int lhs_offset, const int rhs_offset,
                        int m, int n, int k) {
  const std::tuple<> empty_pipeline = {};
  // TODO exp ColMajor order for rhs and result. That may be faster
  const auto matOrder = gemmlowp::MapOrder::RowMajor;
  gemmlowp::MatrixMap<const std::uint8_t, matOrder> lhs(lhs_data, m, k);
  gemmlowp::MatrixMap<const std::uint8_t, matOrder> rhs(rhs_data, k, n);
  gemmlowp::MatrixMap<std::int32_t, matOrder> result(result_data, m, n);

  gemmlowp::GemmContext gemm_context;
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, -lhs_offset, -rhs_offset, empty_pipeline);

  return Status::OK();
}
#endif

void ZeropointValidationHelper(const Tensor* lhs_zero_point, const Tensor* rhs_zero_point, int M, int K) {
  ORT_ENFORCE(lhs_zero_point->Shape().NumDimensions() == 0 || lhs_zero_point->Shape().NumDimensions() == 1,
              "zero_point must be a scalar or a 1D tensor");
  ORT_ENFORCE(rhs_zero_point->Shape().NumDimensions() == 0 || rhs_zero_point->Shape().NumDimensions() == 1,
              "zero_point must be a scalar or a 1D tensor");
  if (lhs_zero_point->Shape().NumDimensions() == 1) {
    ORT_ENFORCE(lhs_zero_point->Shape().Size() == M,
                "when lhs_zero_point is 1D tensor, size should be equal to rows for lhs matrix");
  }
  if (rhs_zero_point->Shape().NumDimensions() == 1) {
    ORT_ENFORCE(lhs_zero_point->Shape().Size() == K,
                "when rhs_zero_point is 1D tensor, size should be equal to rows for lhs matrix");
  }
}

Status MatMulInteger<uint8_t, uint8_t, int32_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(2);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate zero points
  auto a_zero_point = ctx->Input<Tensor>(1);
  auto b_zero_point = ctx->Input<Tensor>(3);
  ZeropointValidationHelper(a_zero_point, b_zero_point,
                            static_cast<int>(helper.M()), static_cast<int>(helper.K()));

#ifdef ENABLE_LOW_PRECISION_COMPUTATION
  for (int i = 0; i < helper.OutputOffsets().size(); i++) {
    GemmlowpMultiply(ctx,
                     a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                     b->template Data<uint8_t>() + helper.RightOffsets()[i],
                     y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
                     *a_zero_point->template Data<uint8_t>(),
                     *b_zero_point->template Data<uint8_t>(),
                     static_cast<int>(helper.M()),
                     static_cast<int>(helper.N()),
                     static_cast<int>(helper.K()));
  }
#endif

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime