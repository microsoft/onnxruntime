// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "core/providers/cpu/math/quantize_linear_matmul.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/gemmlowp_common_wrapper.h"

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

Status GemmlowpMultiply(const uint8_t* lhs_data, const uint8_t* rhs_data, uint8_t* result_data,
                        const int lhs_offset, const int rhs_offset, const int result_offset,
                        int m, int n, int k, int32_t int_multiplier, int32_t right_shift) {
  gemmlowp::OutputStageQuantizeDownInt32ByFixedPoint quantize_down_stage;
  quantize_down_stage.result_offset_after_shift = result_offset;
  quantize_down_stage.result_fixedpoint_multiplier = int_multiplier;
  quantize_down_stage.result_shift = right_shift;
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
  const auto& output_pipeline = std::make_tuple(quantize_down_stage, saturating_cast_stage);

  // TODO exp ColMajor order for rhs and result. That may be faster
  const auto matOrder = gemmlowp::MapOrder::RowMajor;
  gemmlowp::MatrixMap<const std::uint8_t, matOrder> lhs(lhs_data, m, k);
  gemmlowp::MatrixMap<const std::uint8_t, matOrder> rhs(rhs_data, k, n);
  gemmlowp::MatrixMap<std::uint8_t, matOrder> result(result_data, m, n);

  gemmlowp::GemmContext gemm_context;
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, -lhs_offset, -rhs_offset, output_pipeline);

  return Status::OK();
}

void QuantizeMultiplier(float fp_multiplier, std::int32_t* integer_multiplier, int* right_shift) {
  uint32_t* fp_as_bits = reinterpret_cast<uint32_t*>(&fp_multiplier);
  auto current_exponent = (*fp_as_bits >> 23);
  // bring multiplier in [.5,1) range and calculate the shift
  auto bumped_multiplier_as_bits =
      (*fp_as_bits & UINT32_C(0x007fffff)) | UINT32_C(0x3f000000);
  float* bumped_multiplier =
      reinterpret_cast<float*>(&bumped_multiplier_as_bits);
  auto shift = 126 - current_exponent;
  // convert to fixed point number
  std::int64_t int_multiplier =
      static_cast<std::int64_t>(std::round(*bumped_multiplier * (1ll << 31)));

  *integer_multiplier = static_cast<int32_t>(int_multiplier);
  *right_shift = shift;
}

void ScaleAndZeropointPairValidationHelper(const Tensor* scale, const Tensor* zeropoint) {
  ORT_ENFORCE(scale->Shape().NumDimensions() == 0 ||
      (scale->Shape().NumDimensions() == 1 && scale->Shape().GetDims().size() == 1),
      "scale must be a scalar");
  ORT_ENFORCE(zeropoint->Shape().NumDimensions() == 0 ||
      (zeropoint->Shape().NumDimensions() == 1 && zeropoint->Shape().GetDims().size() == 1),
      "zeropoint must be a scalar");
}

template<>
Status QLinearMatMul<uint8_t, uint8_t, uint8_t>::Compute(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(3);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate scale and zero points
  auto a_scale = ctx->Input<Tensor>(1);
  auto a_zero_point = ctx->Input<Tensor>(2);
  ScaleAndZeropointPairValidationHelper(a_scale, a_zero_point);
  auto b_scale = ctx->Input<Tensor>(4);
  auto b_zero_point = ctx->Input<Tensor>(5);
  ScaleAndZeropointPairValidationHelper(b_scale, b_zero_point);
  auto y_scale = ctx->Input<Tensor>(6);
  auto y_zero_point = ctx->Input<Tensor>(7);
  ScaleAndZeropointPairValidationHelper(y_scale, y_zero_point);

  auto a_scale_data = *(a_scale->template Data<float>());
  auto b_scale_data = *(b_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  const float real_multiplier = (a_scale_data * b_scale_data) / y_scale_data;
  int32_t integer_multiplier;
  int right_shift;
  QuantizeMultiplier(real_multiplier, &integer_multiplier, &right_shift);

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    GemmlowpMultiply(a->template Data<uint8_t>() + helper.LeftOffsets()[i],
                     b->template Data<uint8_t>() + helper.RightOffsets()[i],
                     y->template MutableData<uint8_t>() + helper.OutputOffsets()[i],
                     *a_zero_point->template Data<uint8_t>(),
                     *b_zero_point->template Data<uint8_t>(),
                     *y_zero_point->template Data<uint8_t>(),
                     static_cast<int>(helper.M()),
                     static_cast<int>(helper.N()),
                     static_cast<int>(helper.K()),
                     integer_multiplier,
                     right_shift);
  }

  return Status::OK();
}
}  // namespace onnxruntime
