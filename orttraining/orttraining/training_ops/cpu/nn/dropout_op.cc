// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/nn/dropout_op.h"
#include "core/providers/cpu/nn/dropout_op.h"
#include <chrono>
#include <random>
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {
namespace {
constexpr float k_default_ratio{0.5f};

template <typename T2>
float GetRatioOrDefault(const Tensor* ratio_tensor) {
  if (ratio_tensor) {
    ORT_ENFORCE(ratio_tensor->Shape().Size() == 1, "ratio input should have a single value.");
#ifdef _WIN32
#pragma warning(disable : 4244)
#endif
    const float ratio_value = *ratio_tensor->Data<T2>();
    ORT_ENFORCE(0.0f <= ratio_value && ratio_value < 1.0f, "ratio must be in the range [0, 1)");
    return ratio_value;
  }
  return k_default_ratio;
}
}  // namespace

// Dropout
#define REGISTER_KERNEL_TYPED(OpName, VER, T1, T2, Trainable)         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      OpName,                                                         \
      kOnnxDomain,                                                    \
      VER,                                                            \
      T1##_##T2,                                                      \
      kCpuExecutionProvider,                                          \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()), \
      onnxruntime::Dropout<T1, T2, Trainable>);

#define REGISTER_GRADIENT_KERNEL_TYPED(OpName, T1, T2)                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      OpName,                                                         \
      kMSDomain,                                                      \
      1,                                                              \
      T1##_##T2,                                                      \
      kCpuExecutionProvider,                                          \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()), \
      DropoutGrad<T1, T2>);

// DropoutGrad
// REVIEW(codemzs): ConstEigenVectorArrayMap.cast<MLFLoat16) does not seem to be supported.
// However these types work on GPU implementation.
// REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, MLFloat16)
// REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, float)
// REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, double)

REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, float, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, float, float)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, float, double)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, double, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, double, float)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, double, double)

template <typename T1, typename T2>
Status DropoutGrad<T1, T2>::Compute(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  auto dY_span = dY->DataAsSpan<T1>();
  const Tensor* mask = context->Input<Tensor>(1);
  auto mask_span = mask->DataAsSpan<bool>();
  const Tensor* ratio = context->Input<Tensor>(2);  // optional
  const float ratio_value = GetRatioOrDefault<T2>(ratio);
  const auto& dY_shape = dY->Shape();
  Tensor* dX = context->Output(0, dY_shape);
  auto dX_span = dX->MutableDataAsSpan<T1>();

  ORT_ENFORCE(mask->Shape() == dY_shape, "dY and mask should have the same shape");
  ORT_ENFORCE(dX->Shape() == dY_shape, "dY and dX should have the same shape");

  if (ratio_value == 0.0f) {
    if (dY_span.data() != dX_span.data()) {
      std::copy(dY_span.begin(), dY_span.end(), dX_span.begin());
    }
  } else if (ratio_value < 1.0f) {
    ConstEigenVectorArrayMap<T1> dY_arr(dY_span.data(), dY_span.size());
    ConstEigenVectorArrayMap<bool> mask_arr(mask_span.data(), mask_span.size());
    EigenVectorArrayMap<T1> dX_arr(dX_span.data(), dX_span.size());

    dX_arr = mask_arr.cast<T1>() * dY_arr / (1.0f - ratio_value);
  }

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
