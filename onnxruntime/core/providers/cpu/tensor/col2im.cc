// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/col2im.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

// math::Col2im and math::Col2imNd only support float data type
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

  size_t image_dim_number = onnxruntime::narrow<size_t>(image_shape->Shape().Size());
  TensorShapeVector dilations;
  if (dilations_.empty()) {
    dilations.resize(image_dim_number, 1);
  } else {
    ORT_ENFORCE(dilations_.size() == image_dim_number, "size of 'dilations' attribute, if provided, should equal to the number of image dimmensions.");
    dilations = dilations_;
  }

  TensorShapeVector pads;
  if (pads_.empty()) {
    pads.resize(image_dim_number * 2, 0);
  } else {
    ORT_ENFORCE(pads_.size() == 2 * image_dim_number, "size of 'pads' attribute, if provided, should equal to twice the number of image dimmensions.");
    pads = pads_;
  }

  TensorShapeVector strides;
  if (strides_.empty()) {
    strides.resize(image_dim_number, 1);
  } else {
    ORT_ENFORCE(strides_.size() == image_dim_number, "size of 'strides' attribute, if provided, should equal to the number of image dimmensions.");
    strides = strides_;
  }

  int64_t image_shape_size = 1;
  int64_t kernel_shape_size = 1;
  TensorShapeVector adjusted_kernel_shape_dims;
  auto image_dims = image_shape->Data<int64_t>();
  auto kernel_dims = kernel_shape->Data<int64_t>();
  for (size_t i = 0; i < image_dim_number; ++i) {
    image_shape_size *= image_dims[i];
    kernel_shape_size *= kernel_dims[i];
    adjusted_kernel_shape_dims.push_back(dilations[i] * (kernel_dims[i] - 1) + 1);
  }
  TensorShape col_shape = col_tensor->Shape();
  const auto N = col_shape[0];
  const int64_t C = col_shape[1] / kernel_shape_size;
  const int64_t col_stride = C * image_shape_size;
  TensorShape adjusted_kernel_shape(adjusted_kernel_shape_dims);
  const int64_t col_data_stride = col_shape.SizeFromDimension(1);

  TensorShapeVector batched_image_shape_dims, adjusted_image_shape_dims;
  batched_image_shape_dims.insert(batched_image_shape_dims.begin(), {N, C});
  for (size_t i = 0; i < image_dim_number; ++i) {
    batched_image_shape_dims.push_back(image_dims[i]);
    adjusted_image_shape_dims.push_back(image_dims[i] - adjusted_kernel_shape[i] + 1);
  }
  TensorShape batched_image_shape(batched_image_shape_dims);
  T* image_data = context->Output(0, batched_image_shape)->template MutableData<T>();

  const T* col_data = col_tensor->template Data<T>();
  for (auto image_id = 0; image_id < N; ++image_id) {
    if (image_dim_number == 2) {
      math::Col2im<T, CPUMathUtil, StorageOrder::NCHW>(
          col_data + image_id * col_data_stride,
          C,
          image_dims[0],
          image_dims[1],
          kernel_dims[0],
          kernel_dims[1],
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
          image_dims,
          adjusted_image_shape_dims.data(),
          kernel_shape_size * C,
          image_shape_size * C,
          adjusted_kernel_shape.GetDims().data(),
          strides.data(),
          dilations.data(),
          pads.data(),
          image_dim_number,
          image_data + image_id * col_stride,
          &CPUMathUtil::Instance());
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
