// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/upsample.h"
#include <math.h>  //for fabs

using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Upsample<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Upsample<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    uint8_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    Upsample<uint8_t>);

template <typename T>
Status Upsample<T>::BaseCompute(OpKernelContext* context, const std::vector<float>& scales) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);

  const std::vector<int64_t>& dims = X->Shape().GetDims();
  if (dims.size() != scales.size()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Upsample: input tensor's dimension does not match the scales.");
  }

  std::vector<int64_t> Y_dims;
  for (std::size_t i = 0; i < dims.size(); i++) {
    Y_dims.push_back(static_cast<int64_t>(scales[i] * dims[i]));
  }
  Tensor* Y = context->Output(0, Y_dims);

  switch (mode_) {
    case UpsampleMode::NN:
      return UpsampleNearest<T>(X->template Data<T>(), Y->template MutableData<T>(), X->Shape(), Y->Shape(), scales);
    case UpsampleMode::LINEAR: {
      //What's the correct behavior of linear mode is not clear right now,
      //Only support bilinear with 4D tensor to keep consistent with previous behavior
      if (dims.size() != 4)
        return Status(ONNXRUNTIME, FAIL, "Upsample: linear mode upsample only support 4-D tensor with NCHW layout");

      const int64_t batch_size = dims[0], num_channels = dims[1];
      const int64_t input_height = dims[2], input_width = dims[3];

      upsampleBilinear(batch_size, num_channels, input_height, input_width,
                       scales[2], scales[3], X->template Data<T>(), Y->template MutableData<T>());
      return Status::OK();
    }
    default:
      return Status(ONNXRUNTIME, FAIL, "Upsample: unexpected mode");
  }
}

template <typename T>
Status Upsample<T>::Compute(OpKernelContext* context) const {
  if (OpKernel::Node().InputDefs().size() == 1 || scales_cached_) {
    return BaseCompute(context, scales_);
  }

  const Tensor* scales = context->Input<Tensor>(1);
  ORT_ENFORCE(scales != nullptr);
  int64_t scales_size = scales->Shape().Size();
  std::vector<float> scales_arrary(scales_size);
  ParseScalesData(scales, scales_arrary);
  return BaseCompute(context, scales_arrary);
}

}  // namespace onnxruntime
