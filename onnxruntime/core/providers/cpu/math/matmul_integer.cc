// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "core/providers/cpu/math/matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/gemmlowp_common_wrapper.h"

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

Status GemmlowpMultiply(const uint8_t* lhs_data, const uint8_t* rhs_data,
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

template<>
Status MatMulInteger<uint8_t, uint8_t, int32_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate zero points
  int32_t a_offset = 0;
  int32_t b_offset = 0;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(a_zero_point->Shape().NumDimensions() == 0 ||
        (a_zero_point->Shape().NumDimensions() == 1 && a_zero_point->Shape().GetDims().size() == 1),
        "Currently only scalar zero_point is supported. TODO: add per channel zero point support.");
    a_offset = static_cast<int32_t>(*a_zero_point->template Data<uint8_t>());
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(b_zero_point->Shape().NumDimensions() == 0 ||
        (b_zero_point->Shape().NumDimensions() == 1 && b_zero_point->Shape().GetDims().size() == 1),
        "Currently only scalar zero_point is supported. TODO: add per channel zero point support.");
    b_offset = static_cast<int32_t>(*b_zero_point->template Data<uint8_t>());
  }

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    GemmlowpMultiply(a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                     b->template Data<uint8_t>() + helper.RightOffsets()[i],
                     y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
                     a_offset,
                     b_offset,
                     static_cast<int>(helper.M()),
                     static_cast<int>(helper.N()),
                     static_cast<int>(helper.K()));
  }

  return Status::OK();
}
}  // namespace onnxruntime
