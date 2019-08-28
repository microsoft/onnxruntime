// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/matmul_integer.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/qmath.h"
#include "core/providers/common.h"

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

Status MatMulInteger::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  if (a->DataType() == DataTypeImpl::GetType<std::uint8_t>() &&
      b->DataType() == DataTypeImpl::GetType<std::uint8_t>()) {
    // validate zero points
    uint8_t a_offset = 0;
    uint8_t b_offset = 0;
    if (has_a_zero_point_) {
      auto a_zero_point = ctx->Input<Tensor>(2);
      ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                  "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
      a_offset = static_cast<int32_t>(*a_zero_point->template Data<uint8_t>());
    }
    if (has_b_zero_point_) {
      auto b_zero_point = ctx->Input<Tensor>(3);
      ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                  "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
      b_offset = static_cast<int32_t>(*b_zero_point->template Data<uint8_t>());
    }

    for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
      QGemmu8u8_s32(static_cast<int>(helper.M()),
                    static_cast<int>(helper.N()),
                    static_cast<int>(helper.K()),
                    a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                    static_cast<int>(helper.K()),
                    a_offset,
                    b->template Data<uint8_t>() + helper.RightOffsets()[i],
                    static_cast<int>(helper.N()),
                    b_offset,
                    y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
                    static_cast<int>(helper.N()),
                    nullptr);
    }
  } else {
    if (has_a_zero_point_ || has_b_zero_point_) {
      // currently zero point is only supported in Gemmlowp path above
      // in future, the selection of Eigen/Gemmlowp/mklml/etc. should be in a common math library like SGEMM

      auto IsZeroPointTensorAllZero = [](OpKernelContext* ctx, int input_idx) -> bool {
        auto t = ctx->Input<Tensor>(input_idx);
        ORT_ENFORCE(t->Shape().NumDimensions() <= 1 && t->Shape().Size() == 1,
                    "Currently only scalar zero_point is supported. TODO: add per channel zero point support.");
        ORT_ENFORCE(t->DataType() == DataTypeImpl::GetType<int8_t>() ||
                    t->DataType() == DataTypeImpl::GetType<uint8_t>());
        auto data = reinterpret_cast<const int8_t*>(t->DataRaw());
        auto vec = std::vector<int8_t>(data, data + t->Shape().Size());
        return std::all_of(vec.begin(), vec.end(), [](int8_t v) { return v == 0; });
      };

      if ((has_a_zero_point_ && !IsZeroPointTensorAllZero(ctx, 2)) ||
          (has_b_zero_point_ && !IsZeroPointTensorAllZero(ctx, 3))) {
        ORT_NOT_IMPLEMENTED("MatMulInteger: Unsupported input types with zero point");
      }
    }

#define HANDLE_TYPES_WITH_EIGEN(T1, T2, T3)                                     \
  if (a->DataType() == DataTypeImpl::GetType<T1>() &&                           \
      b->DataType() == DataTypeImpl::GetType<T2>() &&                           \
      y->DataType() == DataTypeImpl::GetType<T3>()) {                           \
    for (int i = 0; i < static_cast<int>(helper.OutputOffsets().size()); i++) { \
      EigenCastGEMM<T1, T2, T3>(                                                \
          a->template Data<T1>() + helper.LeftOffsets()[i],                     \
          b->template Data<T2>() + helper.RightOffsets()[i],                    \
          y->template MutableData<T3>() + helper.OutputOffsets()[i],            \
          static_cast<int>(helper.M()),                                         \
          static_cast<int>(helper.N()),                                         \
          static_cast<int>(helper.K()));                                        \
    }                                                                           \
    return Status::OK();                                                        \
  }

    HANDLE_TYPES_WITH_EIGEN(uint8_t, int8_t, int32_t);
    HANDLE_TYPES_WITH_EIGEN(int8_t, uint8_t, int32_t);
    HANDLE_TYPES_WITH_EIGEN(int8_t, int8_t, int32_t);
    ORT_ENFORCE(false, "Unexpected types: a = ", a->DataType(), ", b = ", b->DataType());
  }
  return Status::OK();
}
}  // namespace onnxruntime
