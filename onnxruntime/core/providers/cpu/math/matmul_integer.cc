// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "core/providers/cpu/math/matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/gemmlowp_common_wrapper.h"
#ifdef USE_MKLML
#include <mkl_cblas.h>
#endif

namespace onnxruntime {

// only register this operator if low precision computation is enabled.
ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger);

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
  gemm_context.set_max_num_threads(0);
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, -lhs_offset, -rhs_offset, empty_pipeline);

  return Status::OK();
}

template <typename T>
T GetZeroPoint(const Tensor& zeropoint_input) {
  ORT_ENFORCE(zeropoint_input.Shape().NumDimensions() == 0 || (zeropoint_input.Shape().NumDimensions() == 1 && zeropoint_input.Shape().GetDims().size() == 1),
              "Currently only scalar zero_point is supported. TODO: add per channel zero point support.");
  auto zero_point = static_cast<int32_t>(*zeropoint_input.template Data<T>());
  return zero_point;
}

Status MatMulInteger::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  int32_t a_offset = 0;
  int32_t b_offset = 0;

  if (a->DataType() == DataTypeImpl::GetType<std::uint8_t>() && b->DataType() == DataTypeImpl::GetType<std::uint8_t>()) {
    // validate zero points
    if (has_a_zero_point_) {
      auto a_zero_point = ctx->Input<Tensor>(2);
      a_offset = GetZeroPoint<uint8_t>(*a_zero_point);
    }
    if (has_b_zero_point_) {
      auto b_zero_point = ctx->Input<Tensor>(3);
      b_offset = GetZeroPoint<uint8_t>(*b_zero_point);
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
  } else if (a->DataType() == DataTypeImpl::GetType<std::uint8_t>() && b->DataType() == DataTypeImpl::GetType<std::int8_t>()) {
#ifdef USE_MKLML
    // validate zero points
    if (has_a_zero_point_) {
      auto a_zero_point = ctx->Input<Tensor>(2);
      a_offset = GetZeroPoint<uint8_t>(*a_zero_point);
    }
    if (has_b_zero_point_) {
      auto b_zero_point = ctx->Input<Tensor>(3);
      b_offset = GetZeroPoint<int8_t>(*b_zero_point);
    }

    //ORT_ENFORCE(a_offset == 0 && b_offset == 0, "MKLML only supports zero point == 0");

    MKL_INT32 co = 0;
    for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
      cblas_gemm_s8u8s32(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_OFFSET::CblasFixOffset,
                         static_cast<int>(helper.N()), static_cast<int>(helper.M()), static_cast<int>(helper.K()),
                         1, b->template Data<int8_t>() + helper.RightOffsets()[i], static_cast<int>(helper.N()),
                         0, a->template Data<uint8_t>() + helper.LeftOffsets()[i], static_cast<int>(helper.K()), 0, 0, y->template MutableData<int32_t>() + helper.OutputOffsets()[i], static_cast<int>(helper.N()), &co);
    }
#else
    ORT_THROW("uint8 * int8 for matmul integer op is only supported when use_mklml build option is set.");
#endif
  }

  return Status::OK();
}
}  // namespace onnxruntime
