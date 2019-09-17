// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/quantize_linear_matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/common.h"

namespace onnxruntime {

// only register this operator if low precision computation is enabled.
ONNX_OPERATOR_KERNEL_EX(
    QLinearMatMul,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearMatMul<uint8_t, uint8_t, uint8_t>);

template <>
Status QLinearMatMul<uint8_t, uint8_t, uint8_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(3);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate offsets
  auto a_offset = ctx->Input<Tensor>(2);
  auto b_offset = ctx->Input<Tensor>(5);
  auto y_offset = ctx->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_offset),
              "QLinearMatmul : weight zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  auto a_scale = ctx->Input<Tensor>(1);
  auto b_scale = ctx->Input<Tensor>(4);
  auto y_scale = ctx->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale),
              "QLinearMatmul : weight scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  auto a_scale_data = *(a_scale->template Data<float>());
  auto b_scale_data = *(b_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  const float real_multiplier = (a_scale_data * b_scale_data) / y_scale_data;
  int32_t integer_multiplier;
  int right_shift;
  QuantizeMultiplier(real_multiplier, &integer_multiplier, &right_shift);

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    GemmlowpMultiplyu8u8_u8(a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                            b->template Data<uint8_t>() + helper.RightOffsets()[i],
                            y->template MutableData<uint8_t>() + helper.OutputOffsets()[i],
                            *a_offset->template Data<uint8_t>(),
                            *b_offset->template Data<uint8_t>(),
                            *y_offset->template Data<uint8_t>(),
                            static_cast<int>(helper.M()),
                            static_cast<int>(helper.N()),
                            static_cast<int>(helper.K()),
                            integer_multiplier,
                            right_shift);
  }

  return Status::OK();
}
}  // namespace onnxruntime
