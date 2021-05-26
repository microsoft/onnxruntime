// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear_matmul.h"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    QLinearMatMul,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearMatMul);

Status QLinearMatMul::Compute(OpKernelContext* ctx) const {
  const auto* a = ctx->Input<Tensor>(IN_A);
  const auto* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  // validate offsets
  const auto* a_offset = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  const auto* b_offset = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  const auto* y_offset = ctx->Input<Tensor>(IN_Y_ZERO_POINT);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsBQuantParamSupported(b_offset->Shape(), b ? b->Shape() : b_shape_),
              "QLinearMatmul : weight zero point must be a scalar, 1D tensor of size 1, or last to second dimension is 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  const auto* a_scale = ctx->Input<Tensor>(IN_A_SCALE);
  const auto* b_scale = ctx->Input<Tensor>(IN_B_SCALE);
  const auto* y_scale = ctx->Input<Tensor>(IN_Y_SCALE);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsBQuantParamSupported(b_scale->Shape(), b ? b->Shape() : b_shape_),
              "QLinearMatmul : weight scale must be a scalar, 1D tensor of size 1, or last to second dimension is 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  MatMulComputeHelper helper;
  const uint8_t* b_data;
  bool b_is_signed;
  if (nullptr != b) {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape(), &b_scale->Shape(), &b_offset->Shape()));
    b_data = static_cast<const uint8_t*>(b->DataRaw());
    b_is_signed = b->IsDataType<int8_t>();
  } else {
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape_, &b_scale->Shape(), &b_offset->Shape()));
    b_data = static_cast<const uint8_t*>(packed_b_.get());
    b_is_signed = b_is_signed_;
  }

  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  const auto* b_scale_data = b_scale->template Data<float>();
  auto a_scale_data = *(a_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  const int64_t output_scale_size = b_scale->Shape().Size();
  std::vector<float> output_scales(output_scale_size);
  for (int64_t i = 0; i < output_scale_size; i++) {
    output_scales[i] = (a_scale_data * b_scale_data[i] / y_scale_data);
  }

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
  auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) *
                                       static_cast<size_t>(helper.M()) * static_cast<size_t>(helper.N()));
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());

  MLAS_GEMM_U8X8_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.BIsSigned = b_is_signed;

  MLAS_GEMM_U8X8_DATA_PARAMS gemm_params;
  gemm_params.lda = gemm_shape.K;
  gemm_params.ZeroPointA = *a_offset->template Data<uint8_t>();
  gemm_params.ldb = gemm_shape.N;
  gemm_params.C = gemm_output;
  gemm_params.ldc = gemm_shape.N;
  gemm_params.BIsPacked = bool(packed_b_);
  gemm_params.PerColumnZeroPoints = !IsScalarOr1ElementVector(b_offset);

  auto b_zp_data = static_cast<const uint8_t*>(b_offset->DataRaw());
  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    gemm_params.A = a->template Data<uint8_t>() + helper.LeftOffsets()[i];
    gemm_params.B = b_data + helper.RightOffsets()[i];
    gemm_params.ZeroPointB = b_zp_data + helper.RightZeroPointOffsets()[i];

    MlasGemm(gemm_shape, gemm_params, ctx->GetOperatorThreadPool());

    //TODO!! consider making this a post processor, so that we can parallize this loop
    MlasRequantizeOutput(
        gemm_output,
        static_cast<size_t>(helper.N()),
        y->template MutableData<uint8_t>() + helper.OutputOffsets()[i],
        static_cast<size_t>(helper.N()),
        nullptr,
        output_scales.data() + helper.RightScaleOffsets()[i],
        output_scales.size() > 1,
        *y_offset->template Data<uint8_t>(),
        0,0,
        static_cast<size_t>(helper.M()),
        static_cast<size_t>(helper.N())
    );

  }

  return Status::OK();
}

}  // namespace onnxruntime
