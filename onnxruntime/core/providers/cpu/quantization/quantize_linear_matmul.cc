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
// uint8_t kernel supports weight being either uint8_t or int8_t
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearMatMul,
    kOnnxDomain,
    10,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearMatMul);

// int8_t kernel only supports weight being int8_t
#define REGISTER_QLINEARMATMUL_INT8_KERNEL()                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      QLinearMatMul,                                                    \
      kOnnxDomain,                                                      \
      10,                                                               \
      int8_t,                                                           \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())  \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>()), \
      QLinearMatMul);

REGISTER_QLINEARMATMUL_INT8_KERNEL();

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

#if defined(_M_ARM64) || defined(__aarch64__)
  std::vector<int32_t> pre_shifts;
  std::vector<int32_t> multipliers;
  std::vector<int32_t> post_shifts;
  if (use_fixed_point_requant_) {
    pre_shifts.resize(output_scales.size());
    multipliers.resize(output_scales.size());
    post_shifts.resize(output_scales.size());
    MlasFloatToFixedPoint(output_scales.data(),
                          multipliers.data(),
                          pre_shifts.data(),
                          post_shifts.data(),
                          output_scales.size());
  }
#endif

  const size_t num_gemms = helper.OutputOffsets().size();
  MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.AIsSigned = a->IsDataType<int8_t>();
  gemm_shape.BIsSigned = b_is_signed;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
  auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(gemm_shape.M) *
                                       gemm_shape.N * sizeof(int32_t) * num_gemms);
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());

  std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_params(num_gemms);
  std::vector<MLAS_REQUANT_PARAM> requant_params(num_gemms);
  std::vector<MLAS_QGEMM_REQUANT_OUTPUT_PROCESSOR> requant_procs;
  requant_procs.reserve(num_gemms);

  bool is_output_signed = y->IsDataType<int8_t>();
  int32_t output_offset = is_output_signed ? *(static_cast<const int8_t*>(y_offset->DataRaw()))
                                           : *(static_cast<const uint8_t*>(y_offset->DataRaw()));
  auto b_zp_data = static_cast<const uint8_t*>(b_offset->DataRaw());
  for (size_t i = 0; i < num_gemms; i++) {
    gemm_params[i].A = static_cast<const uint8_t*>(a->DataRaw()) + helper.LeftOffsets()[i];
    gemm_params[i].lda = gemm_shape.K;
    gemm_params[i].ZeroPointA = *(static_cast<const uint8_t*>(a_offset->DataRaw()));

    gemm_params[i].B = b_data + helper.RightOffsets()[i];
    gemm_params[i].ldb = gemm_shape.N;
    gemm_params[i].BIsPacked = bool(packed_b_);
    gemm_params[i].ZeroPointB = b_zp_data + helper.RightZeroPointOffsets()[i];

    gemm_params[i].C = gemm_output + (gemm_shape.M * gemm_shape.N * i);
    gemm_params[i].ldc = gemm_shape.N;

    gemm_params[i].PerColumnZeroPoints = !IsScalarOr1ElementVector(b_offset);

    requant_params[i].Size = output_scales.size();
    requant_params[i].ZeroPoint = output_offset;
#if defined(_M_ARM64) || defined(__aarch64__)
    if (use_fixed_point_requant_) {
      requant_params[i].RequantRoundKind = MLAS_ROUND_KIND::MlasRoundHalfUp;
      requant_params[i].PreShift = pre_shifts.data() + helper.RightScaleOffsets()[i];
      requant_params[i].Multiplier = multipliers.data() + helper.RightScaleOffsets()[i];
      requant_params[i].PostShift = post_shifts.data() + helper.RightScaleOffsets()[i];
    } else {
      requant_params[i].RequantRoundKind = MLAS_ROUND_KIND::MlasRoundHalfEven;
      requant_params[i].Scale = output_scales.data() + helper.RightScaleOffsets()[i];
    }
#else
    requant_params[i].RequantRoundKind = MLAS_ROUND_KIND::MlasRoundHalfEven;
    requant_params[i].Scale = output_scales.data() + helper.RightScaleOffsets()[i];
#endif
    requant_procs.emplace_back(static_cast<uint8_t*>(y->MutableDataRaw()) + helper.OutputOffsets()[i],
                               static_cast<size_t>(helper.N()),
                               nullptr,
                               &requant_params[i],
                               is_output_signed);
    gemm_params[i].OutputProcessor = &(requant_procs[i]);
  }

  MlasGemmBatch(gemm_shape, gemm_params.data(), num_gemms, ctx->GetOperatorThreadPool());

  return Status::OK();
}

}  // namespace onnxruntime
