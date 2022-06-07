// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/math/scale.h"
#include "core/framework/math.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_SCALE_KERNEL_TYPED(T, ScaleT)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      Scale,                                                                \
      kMSDomain,                                                            \
      1,                                                                    \
      T##_##ScaleT,                                                         \
      kCpuExecutionProvider,                                                \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("ScaleT", DataTypeImpl::GetTensorType<ScaleT>()), \
      Scale<T, ScaleT>);

template <typename T, typename ScaleT>
Scale<T, ScaleT>::Scale(const OpKernelInfo& info) : OpKernel(info) {
  int64_t scale_down;
  info.GetAttrOrDefault("scale_down", &scale_down, static_cast<int64_t>(0));
  scale_down_ = (scale_down != 0);
}

template <typename T, typename ScaleT>
Status Scale<T, ScaleT>::Compute(OpKernelContext* context) const {
  const Tensor* scale_tensor = context->Input<Tensor>(1);
  ORT_ENFORCE(scale_tensor->Shape().Size() == 1, "Scale input should have a single value.");
  float scale_value = static_cast<float>(*scale_tensor->Data<ScaleT>());
  ORT_ENFORCE(scale_value != 0.0f, "Scale value must not be 0.");

  if (scale_down_) {
    scale_value = 1.0f / scale_value;
  }

  const auto& input_tensor = *context->Input<Tensor>(0);
  auto& output_tensor = *context->Output(0, input_tensor.Shape());
  EigenMap<T>(output_tensor) = static_cast<T>(scale_value) * EigenMap<T>(input_tensor);
  return Status::OK();
}

REGISTER_SCALE_KERNEL_TYPED(float, float)
REGISTER_SCALE_KERNEL_TYPED(float, double)
REGISTER_SCALE_KERNEL_TYPED(float, int64_t)
REGISTER_SCALE_KERNEL_TYPED(float, int32_t)
REGISTER_SCALE_KERNEL_TYPED(double, float)
REGISTER_SCALE_KERNEL_TYPED(double, double)
REGISTER_SCALE_KERNEL_TYPED(double, int64_t)
REGISTER_SCALE_KERNEL_TYPED(double, int32_t)

}  // namespace contrib
}  // namespace onnxruntime
