// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nchwc_pool.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    NchwcMaxPool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcPool<float, MaxPool<1>>);

template <>
Status NchwcPool<float, MaxPool<1>>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  size_t input_dims = x_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_dims >= 3, "Input dimension cannot be less than 3.");

  size_t pooling_dims = input_dims - 2;
  if (pooling_dims > 3) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported pooling size.");
  }
  if (!global_pooling_) {
    ORT_RETURN_IF_NOT(pooling_dims == kernel_shape_.size(), "kernel_shape num_dims is not compatible with X num_dims.");
  }

  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> output_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads);
  Tensor* Y = context->Output(0, output_dims);

  MlasPoolNchwc(MlasMaximumPooling,
                pooling_dims,
                X->Shape().GetDims().data(),
                global_pooling_ ? nullptr : kernel_shape_.data(),
                nullptr,
                global_pooling_ ? nullptr : pads.data(),
                global_pooling_ ? nullptr : strides_.data(),
                output_dims.data(),
                X->template Data<float>(),
                Y->template MutableData<float>());

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
