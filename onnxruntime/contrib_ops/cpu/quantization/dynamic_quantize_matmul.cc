// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamic_quantize_matmul.h"

#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {

#define REGISTER_DYNAMIC_QUANTIZE_MATMUL(T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      DynamicQuantizeMatMul,                                          \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCpuExecutionProvider,                                          \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      DynamicQuantizeMatMul<T>);

REGISTER_DYNAMIC_QUANTIZE_MATMUL(int8_t)
REGISTER_DYNAMIC_QUANTIZE_MATMUL(uint8_t)

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

template <typename T>
Status DynamicQuantizeMatMul<T>::Compute(OpKernelContext* ctx) const {
  auto* a = ctx->Input<Tensor>(0);
  auto* b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  auto* b_scale_tensor = ctx->Input<Tensor>(2);
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale_tensor),
              "MatmulInteger : input B scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");

  float b_scale = *b_scale_tensor->template Data<float>();

  auto* b_zp_tensor = ctx->Input<Tensor>(3);
  T b_zp = 0;
  if (b_zp_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zp_tensor),
                "MatmulInteger : input B zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    b_zp = *b_zp_tensor->template Data<T>();
  }

  // calculate quantization parameter of a
  const auto* a_data = a->template Data<float>();
  int64_t num_of_elements = a->Shape().Size();

  float a_scale;
  uint8_t a_zp;
  GetQuantizationParameter(a_data, num_of_elements, a_scale, a_zp);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  uint8_t* a_data_quant = static_cast<uint8_t*>(allocator->Alloc(SafeInt<size_t>(num_of_elements) * sizeof(uint8_t)));
  BufferUniquePtr a_buffer_quant_holder(a_data_quant, BufferDeleter(allocator));
  // quantize the data
  MlasQuantizeLinear(a_data, a_data_quant, num_of_elements, a_scale, a_zp);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));

  const auto* b_data = b->template Data<T>();

  Tensor* y = ctx->Output(0, helper.OutputShape());
  auto* y_data = y->template MutableData<float>();

  const float multiplier = a_scale * b_scale;

  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();
  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    QGemm(static_cast<int>(helper.M()),
          static_cast<int>(helper.N()),
          static_cast<int>(helper.K()),
          a_data_quant + helper.LeftOffsets()[i],
          static_cast<int>(helper.K()),
          a_zp,
          b_data + helper.RightOffsets()[i],
          static_cast<int>(helper.N()),
          b_zp,
          y_data + helper.OutputOffsets()[i],
          static_cast<int>(helper.N()),
          &multiplier,
          nullptr,
          thread_pool);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
