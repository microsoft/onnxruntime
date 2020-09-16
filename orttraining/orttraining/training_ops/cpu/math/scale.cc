// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scale.h"
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
Status Scale<T, ScaleT>::Compute(OpKernelContext* context) const {
  const Tensor* scale_tensor = context->Input<Tensor>(1);
  ORT_ENFORCE(scale_tensor->Shape().Size() == 1, "Scale input should have a single value.");
  const float scale_value = static_cast<float>(*scale_tensor->Data<ScaleT>());
  ORT_ENFORCE(scale_value != 0.0f, "Scale value must not be 0.");
  const T inverse_scale_value = static_cast<T>(1.0f / scale_value);

  const auto& input_tensor = *context->Input<Tensor>(0);
  auto& output_tensor = *context->Output(0, input_tensor.Shape());
  auto* input = input_tensor.template Data<T>();
  auto* output = output_tensor.template MutableData<T>();
  const auto size = input_tensor.Shape().Size();
  for (int64_t i = 0; i < size; ++i, ++output, ++input) {
    *output = (*input) * inverse_scale_value;
  }

  return Status::OK();
}

REGISTER_SCALE_KERNEL_TYPED(float, float)
REGISTER_SCALE_KERNEL_TYPED(float, int64_t)

}  // namespace contrib
}  // namespace onnxruntime
