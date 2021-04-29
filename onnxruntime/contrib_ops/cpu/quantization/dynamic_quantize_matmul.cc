// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/math/matmul_integer_base.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {

class MatMulIntegerToFloatBase : public MatMulIntegerBase {
 public:
  MatMulIntegerToFloatBase(const OpKernelInfo& info) : MatMulIntegerBase(info) {
  }

  enum OutputTensors : int { OUT_Y = 0 };

 protected:
  Status ComputeCommon(OpKernelContext* ctx,
                       const uint8_t* a_data,
                       const TensorShape& a_shape,
                       float a_scale,
                       uint8_t a_zp,
                       const Tensor* b_tensor,
                       const Tensor* b_scale,
                       const Tensor* b_zp,
                       const Tensor* bias_tensor) const;
};

Status MatMulIntegerToFloatBase::ComputeCommon(OpKernelContext* ctx,
                                               const uint8_t* a_data,
                                               const TensorShape& a_shape,
                                               float a_scale,
                                               uint8_t a_zp,
                                               const Tensor* b_tensor,
                                               const Tensor* b_scale_tensor,
                                               const Tensor* b_zp_tensor,
                                               const Tensor* bias_tensor) const {
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape, packed_b_ ? b_shape_ : b_tensor->Shape()));
  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->template MutableData<float>();
  const auto* bias_data = bias_tensor != nullptr ? bias_tensor->Data<float>() : nullptr;

  // process zero point of B
  bool is_b_zp_per_column = false;
  uint8_t b_zp_default = 0;
  const uint8_t* b_zp_ptr = nullptr;
  if (b_zp_tensor != nullptr) {
    is_b_zp_per_column = !IsScalarOr1ElementVector(b_zp_tensor);
    b_zp_ptr = static_cast<const uint8_t*>(b_zp_tensor->DataRaw());
  }

  // process scale of B
  bool is_b_scale_per_column = !IsScalarOr1ElementVector(b_scale_tensor);
  const float* b_scale_data = b_scale_tensor->template Data<float>();

  // calculate multiplier
  std::vector<float> multipliers(b_scale_data, b_scale_data + b_scale_tensor->Shape().Size());
  std::for_each(multipliers.begin(), multipliers.end(), [&a_scale](float& multiplier) {
    return multiplier *= a_scale;
  });

  // batch gemm
  MLAS_GEMM_U8X8_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = static_cast<size_t>(helper.M());
  gemm_shape.N = static_cast<size_t>(helper.N());
  gemm_shape.K = static_cast<size_t>(helper.K());
  gemm_shape.BIsSigned = packed_b_ ? b_is_signed_ : b_tensor->IsDataType<int8_t>();

  const size_t num_gemms = helper.OutputOffsets().size();
  std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> gemm_scale_procs;
  gemm_scale_procs.reserve(num_gemms);
  std::vector<MLAS_GEMM_U8X8_DATA_PARAMS> gemm_data_vec(num_gemms);

  for (size_t gemm_idx = 0; gemm_idx < num_gemms; gemm_idx++) {
    gemm_scale_procs.emplace_back(y_data + helper.OutputOffsets()[gemm_idx],
                                  gemm_shape.N,
                                  multipliers.data(),
                                  bias_data,
                                  MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                  is_b_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);
    auto& params = gemm_data_vec[gemm_idx];
    params.OutputProcessor = &(gemm_scale_procs[gemm_idx]);
    params.A = a_data + helper.LeftOffsets()[gemm_idx];
    params.lda = gemm_shape.K;
    params.ZeroPointA = a_zp;
    params.BIsPacked = bool(packed_b_);
    params.B = bool(packed_b_) ? packed_b_.get() : static_cast<const uint8_t*>(b_tensor->DataRaw()) + helper.RightOffsets()[gemm_idx];
    params.ldb = gemm_shape.N;
    params.ZeroPointB = nullptr != b_zp_ptr ? b_zp_ptr : &b_zp_default;
    params.PerColumnZeroPoints = is_b_zp_per_column;
    params.C = reinterpret_cast<int32_t*>(y_data + helper.OutputOffsets()[gemm_idx]);
    params.ldc = gemm_shape.N;
  }

  MlasGemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms, ctx->GetOperatorThreadPool());

  return Status::OK();
}

class DynamicQuantizeMatMul final : public MatMulIntegerToFloatBase {
 public:
  DynamicQuantizeMatMul(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_B_SCALE = 2,
    IN_B_ZERO_POINT = 3,
    IN_BIAS = 4
  };

 protected:
  int GetBIdx() override { return IN_B; }
};

class MatMulIntegerToFloat final : public MatMulIntegerToFloatBase {
 public:
  MatMulIntegerToFloat(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_SCALE = 2,
    IN_B_SCALE = 3,
    IN_A_ZERO_POINT = 4,
    IN_B_ZERO_POINT = 5,
    IN_BIAS = 6
  };

 protected:
  int GetBIdx() override { return IN_B; }
};

Status DynamicQuantizeMatMul::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  const Tensor* b_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);

  // calculate quantization parameter of a
  const float* a_data = a->template Data<float>();
  int64_t num_of_elements = a->Shape().Size();

  float a_scale;
  uint8_t a_zero_point;
  GetQuantizationParameter(a_data, num_of_elements, a_scale, a_zero_point, ctx->GetOperatorThreadPool());

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  uint8_t* a_data_quant = static_cast<uint8_t*>(allocator->Alloc(SafeInt<size_t>(num_of_elements) * sizeof(uint8_t)));
  BufferUniquePtr a_buffer_quant_holder(a_data_quant, BufferDeleter(allocator));

  ParQuantizeLinear(a_data, a_data_quant, num_of_elements, a_scale, a_zero_point, ctx->GetOperatorThreadPool());

  return ComputeCommon(ctx,
                       a_data_quant,
                       a->Shape(),
                       a_scale,
                       a_zero_point,
                       b,
                       b_scale_tensor,
                       b_zp_tensor,
                       ctx->Input<Tensor>(IN_BIAS));
}

Status MatMulIntegerToFloat::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  // validate scale of A
  const Tensor* a_scale_tensor = ctx->Input<Tensor>(IN_A_SCALE);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale_tensor),
              "MatMulIntegerToFloat : input A scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
  float a_scale = *a_scale_tensor->template Data<float>();

  // validate zero point of A
  uint8_t a_zero_point = 0;
  const Tensor* a_zero_point_tensor = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  if (a_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point_tensor),
                "MatMulIntegerToFloat : input A zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    a_zero_point = *a_zero_point_tensor->Data<uint8_t>();
  }

  const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  const Tensor* b_zp_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);

  return ComputeCommon(ctx,
                       a->Data<uint8_t>(),
                       a->Shape(),
                       a_scale,
                       a_zero_point,
                       b,
                       b_scale_tensor,
                       b_zp_tensor,
                       ctx->Input<Tensor>(IN_BIAS));
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DynamicQuantizeMatMul,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
    DynamicQuantizeMatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

}  // namespace contrib
}  // namespace onnxruntime
