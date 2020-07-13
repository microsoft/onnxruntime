// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {

class MatMulIntegerToFloatBase : public OpKernel {
 public:
  MatMulIntegerToFloatBase(const OpKernelInfo& info) : OpKernel(info) {
    TryPackWeights(info);
  }

 protected:
  void TryPackWeights(const OpKernelInfo& info);
  Status ComputeCommon(OpKernelContext* ctx,
                       const uint8_t* a_data,
                       const TensorShape& a_shape,
                       uint8_t a_zero_point,
                       const Tensor* b,
                       uint8_t b_zero_point,
                       float multiplier,
                       const Tensor* bias_tensor) const;

#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
  BufferUniquePtr packed_b_;
#endif
};

void MatMulIntegerToFloatBase::TryPackWeights(const OpKernelInfo& info) {
#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
  // Check if the weights tensor is constant.
  const Tensor* b;
  if (!info.TryGetConstantInput(1, &b)) {
    return;
  }

  // Only handle the common case of a 2D weight matrix. Additional matrices
  // could be handled by stacking the packed buffers.
  const auto& b_shape = b->Shape();
  if (b_shape.NumDimensions() != 2) {
    return;
  }

  const size_t K = static_cast<size_t>(b_shape[0]);
  const size_t N = static_cast<size_t>(b_shape[1]);

  const auto* b_data = static_cast<const uint8_t*>(b->DataRaw());
  const bool b_is_signed = b->IsDataType<int8_t>();

  const size_t packed_b_size = MlasGemmPackBSize(N, K, b_is_signed);
  if (packed_b_size == 0) {
    return;
  }

  auto alloc = info.GetAllocator(0, OrtMemTypeDefault);
  auto* packed_b_data = alloc->Alloc(packed_b_size);
  packed_b_ = BufferUniquePtr(packed_b_data, BufferDeleter(alloc));
  MlasGemmPackB(N, K, b_data, N, b_is_signed, packed_b_data);
#else
  ORT_UNUSED_PARAMETER(info);
#endif
}

Status MatMulIntegerToFloatBase::ComputeCommon(OpKernelContext* ctx,
                                               const uint8_t* a_data,
                                               const TensorShape& a_shape,
                                               uint8_t a_zero_point,
                                               const Tensor* b,
                                               uint8_t b_zero_point,
                                               float multiplier,
                                               const Tensor* bias_tensor) const {
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape, b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  const auto* b_data = static_cast<const uint8_t*>(b->DataRaw());
  const bool b_is_signed = b->IsDataType<int8_t>();
  auto* y_data = y->template MutableData<float>();
  const auto* bias_data = bias_tensor != nullptr ? bias_tensor->Data<float>() : nullptr;

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
    if (packed_b_) {
      MlasGemm(static_cast<size_t>(helper.M()),
               static_cast<size_t>(helper.N()),
               static_cast<size_t>(helper.K()),
               a_data + helper.LeftOffsets()[i],
               static_cast<size_t>(helper.K()),
               a_zero_point,
               packed_b_.get(),
               b_zero_point,
               b_is_signed,
               y_data + helper.OutputOffsets()[i],
               static_cast<size_t>(helper.N()),
               &multiplier,
               bias_data,
               thread_pool);
      continue;
    }
#endif
    QGemm(static_cast<int>(helper.M()),
          static_cast<int>(helper.N()),
          static_cast<int>(helper.K()),
          a_data + helper.LeftOffsets()[i],
          static_cast<int>(helper.K()),
          a_zero_point,
          b_data + helper.RightOffsets()[i],
          static_cast<int>(helper.N()),
          b_zero_point,
          b_is_signed,
          y_data + helper.OutputOffsets()[i],
          static_cast<int>(helper.N()),
          &multiplier,
          bias_data,
          thread_pool);
  }

  return Status::OK();
}

class DynamicQuantizeMatMul final : public MatMulIntegerToFloatBase {
 public:
  DynamicQuantizeMatMul(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

class MatMulIntegerToFloat final : public MatMulIntegerToFloatBase {
 public:
  MatMulIntegerToFloat(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;
};

static void GetQuantizationParameter(const float* data, int64_t num_of_elements, float& scale, uint8_t& zp) {
  // find input range min and max
  float min, max;
  MlasFindMinMaxElement(data, &min, &max, num_of_elements);

  // ensure the input range includes zero
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);

  // find scale and zero point
  uint8_t qmin = 0;
  uint8_t qmax = 255;
  scale = max == min ? 1.0f : (max - min) / (qmax - qmin);

  float initial_zero_point = qmin - min / scale;
  zp = static_cast<uint8_t>(RoundHalfToEven(std::max(float(qmin), std::min(float(qmax), initial_zero_point))));
}

Status DynamicQuantizeMatMul::Compute(OpKernelContext* ctx) const {
  const auto* a = ctx->Input<Tensor>(0);
  const auto* b = ctx->Input<Tensor>(1);

  const auto* b_scale_tensor = ctx->Input<Tensor>(2);
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale_tensor),
              "DynamicQuantizeMatMul : input B scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
  float b_scale = *b_scale_tensor->template Data<float>();

  const auto* b_zero_point_tensor = ctx->Input<Tensor>(3);
  uint8_t b_zero_point = 0;
  if (b_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point_tensor),
                "DynamicQuantizeMatMul : input B zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    b_zero_point = *static_cast<const uint8_t*>(b_zero_point_tensor->DataRaw());
  }

  // calculate quantization parameter of a
  const auto* a_data = a->template Data<float>();
  int64_t num_of_elements = a->Shape().Size();

  float a_scale;
  uint8_t a_zero_point;
  GetQuantizationParameter(a_data, num_of_elements, a_scale, a_zero_point);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  uint8_t* a_data_quant = static_cast<uint8_t*>(allocator->Alloc(SafeInt<size_t>(num_of_elements) * sizeof(uint8_t)));
  BufferUniquePtr a_buffer_quant_holder(a_data_quant, BufferDeleter(allocator));
  // quantize the data
  MlasQuantizeLinear(a_data, a_data_quant, num_of_elements, a_scale, a_zero_point);

  return ComputeCommon(ctx,
                       a_data_quant,
                       a->Shape(),
                       a_zero_point,
                       b,
                       b_zero_point,
                       a_scale * b_scale,
                       ctx->Input<Tensor>(4));
}

Status MatMulIntegerToFloat::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);

  const Tensor* a_scale_tensor = ctx->Input<Tensor>(2);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale_tensor),
              "MatMulIntegerToFloat : input A scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
  float a_scale = *a_scale_tensor->template Data<float>();

  const Tensor* b_scale_tensor = ctx->Input<Tensor>(3);
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale_tensor),
              "MatMulIntegerToFloat : input B scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
  float b_scale = *b_scale_tensor->template Data<float>();

  // validate zero points
  uint8_t a_zero_point = 0;
  const Tensor* a_zero_point_tensor = ctx->Input<Tensor>(4);
  if (a_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point_tensor),
                "MatMulIntegerToFloat : input A zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    a_zero_point = *a_zero_point_tensor->Data<uint8_t>();
  }

  uint8_t b_zero_point = 0;
  const Tensor* b_zero_point_tensor = ctx->Input<Tensor>(5);
  if (b_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point_tensor),
                "MatMulIntegerToFloat : input B zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    b_zero_point = *static_cast<const uint8_t*>(b_zero_point_tensor->DataRaw());
  }

  return ComputeCommon(ctx,
                       a->Data<uint8_t>(),
                       a->Shape(),
                       a_zero_point,
                       b,
                       b_zero_point,
                       a_scale * b_scale,
                       ctx->Input<Tensor>(6));
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
