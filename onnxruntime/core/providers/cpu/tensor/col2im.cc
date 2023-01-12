// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/col2im.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Col2Im,
    18,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Col2Im<float>);

template <typename T>
Status Col2Im<T>::Compute(OpKernelContext* context) const {
  const auto* col_tensor = context->Input<Tensor>(0);
  const auto* image_shape = context->Input<Tensor>(1);
  const auto* kernel_shape = context->Input<Tensor>(2);

  size_t image_dim_size = image_shape->Shape().Size();
  TensorShapeVector pads = col2im_attrs_.pads;
  TensorShapeVector dilations = col2im_attrs_.dilations;
  TensorShapeVector strides = col2im_attrs_.strides;

  if (dilations.empty()) {
    dilations.resize(image_dim_size, 1);
  }
  if (pads.empty()) {
    pads.resize(image_dim_size * 2, 0);
  }
  if (strides.empty()) {
    strides.resize(image_dim_size, 1);
  }

  int64_t image_shape_size = 1;
  int64_t kernel_shape_size = 1;
  TensorShapeVector adjusted_kernel_shape_dims;
  for (auto i = 0; i < image_shape->Shape().Size(); ++i) {
    image_shape_size *= image_shape->Data<int64_t>()[i];
    kernel_shape_size *= kernel_shape->Data<int64_t>()[i];
    adjusted_kernel_shape_dims.push_back(dilations[i] * (kernel_shape->Data<int64_t>()[i] - 1) + 1);
  }
  TensorShape col_shape = col_tensor->Shape();
  const auto N = col_shape[0];
  const int64_t C = col_shape[1] / kernel_shape_size;
  const int64_t col_stride = C * image_shape_size;
  TensorShape adjusted_kernel_shape(adjusted_kernel_shape_dims);
  const int64_t col_data_stride = col_shape.SizeFromDimension(1);

  TensorShapeVector batched_image_shape_dims, adjusted_image_shape_dims;
  batched_image_shape_dims.insert(batched_image_shape_dims.begin(), {N, C});
  for (auto i = 0; i < image_shape->Shape()[0]; ++i) {
    batched_image_shape_dims.push_back(image_shape->Data<int64_t>()[i]);
    adjusted_image_shape_dims.push_back(image_shape->Data<int64_t>()[i] - adjusted_kernel_shape[i] + 1);
  }
  TensorShape batched_image_shape(batched_image_shape_dims);
  T* image_data = context->Output(0, batched_image_shape)->template MutableData<T>();

  const T* col_data = col_tensor->template Data<T>();
  for (auto image_id = 0; image_id < N; ++image_id) {
    if (image_shape->Shape()[0] == 2) {
      math::Col2im<T, CPUMathUtil, StorageOrder::NCHW>(
          col_data + image_id * col_data_stride,
          C,
          image_shape->Data<int64_t>()[0],
          image_shape->Data<int64_t>()[1],
          kernel_shape->Data<int64_t>()[0],
          kernel_shape->Data<int64_t>()[1],
          dilations[0],
          dilations[1],
          pads[0],
          pads[1],
          pads[2],
          pads[3],
          strides[0],
          strides[1],
          image_data + image_id * col_stride,
          &CPUMathUtil::Instance());
    } else {
      math::Col2imNd<T, CPUMathUtil, StorageOrder::NCHW>(
          col_data + image_id * col_data_stride,
          image_shape->Data<int64_t>(),
          adjusted_image_shape_dims.data(),
          kernel_shape_size * C,
          image_shape_size,
          adjusted_kernel_shape.GetDims().data(),
          strides.data(),
          dilations.data(),
          pads.data(),
          image_shape->Shape().Size(),
          image_data + image_id * col_stride,
          &CPUMathUtil::Instance());
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
