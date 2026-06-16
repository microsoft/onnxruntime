// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/col2im.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"

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

  ORT_RETURN_IF_NOT(image_shape->Shape().NumDimensions() == 1,
                    "'image_shape' input must be a 1-D tensor.");
  ORT_RETURN_IF_NOT(kernel_shape->Shape().NumDimensions() == 1,
                    "'block_shape' input must be a 1-D tensor.");
  ORT_RETURN_IF_NOT(image_shape->Shape().Size() == kernel_shape->Shape().Size(),
                    "'image_shape' and 'block_shape' must have the same number of elements.");

  size_t image_dim_number = onnxruntime::narrow<size_t>(image_shape->Shape().Size());
  ORT_RETURN_IF_NOT(image_dim_number > 0, "'image_shape' must have at least one element.");

  TensorShapeVector dilations;
  if (dilations_.empty()) {
    dilations.resize(image_dim_number, 1);
  } else {
    ORT_RETURN_IF_NOT(dilations_.size() == image_dim_number,
                      "size of 'dilations' attribute, if provided, should equal to the number of image dimensions.");
    dilations = dilations_;
  }

  TensorShapeVector pads;
  if (pads_.empty()) {
    pads.resize(image_dim_number * 2, 0);
  } else {
    ORT_RETURN_IF_NOT(pads_.size() == 2 * image_dim_number,
                      "size of 'pads' attribute, if provided, should equal to twice the number of image dimensions.");
    pads = pads_;
  }

  TensorShapeVector strides;
  if (strides_.empty()) {
    strides.resize(image_dim_number, 1);
  } else {
    ORT_RETURN_IF_NOT(strides_.size() == image_dim_number,
                      "size of 'strides' attribute, if provided, should equal to the number of image dimensions.");
    strides = strides_;
  }

  SafeInt<int64_t> image_shape_size = 1;
  SafeInt<int64_t> kernel_shape_size = 1;
  SafeInt<int64_t> expected_col_blocks = 1;
  TensorShapeVector adjusted_kernel_shape_dims;
  TensorShapeVector sliding_block_shape_dims;
  auto image_dims = image_shape->Data<int64_t>();
  auto kernel_dims = kernel_shape->Data<int64_t>();
  for (size_t i = 0; i < image_dim_number; ++i) {
    ORT_RETURN_IF_NOT(image_dims[i] > 0, "All 'image_shape' values must be positive.");
    ORT_RETURN_IF_NOT(kernel_dims[i] > 0, "All 'block_shape' values must be positive.");
    ORT_RETURN_IF_NOT(strides[i] > 0, "All stride values must be positive.");
    ORT_RETURN_IF_NOT(dilations[i] > 0, "All dilation values must be positive.");
    image_shape_size *= image_dims[i];
    kernel_shape_size *= kernel_dims[i];
    const int64_t adjusted_kernel = SafeInt<int64_t>(dilations[i]) * (kernel_dims[i] - 1) + 1;
    adjusted_kernel_shape_dims.push_back(adjusted_kernel);
    const int64_t padded_extent = SafeInt<int64_t>(image_dims[i]) + pads[i] + pads[i + image_dim_number];
    ORT_RETURN_IF_NOT(padded_extent >= adjusted_kernel,
                      "Padded image extent is smaller than the dilated kernel for spatial dimension ", i, ".");
    const int64_t sliding_blocks = (padded_extent - adjusted_kernel) / strides[i] + 1;
    sliding_block_shape_dims.push_back(sliding_blocks);
    expected_col_blocks *= sliding_blocks;
  }
  ORT_RETURN_IF_NOT(kernel_shape_size > 0, "kernel_shape_size must be positive");

  TensorShape col_shape = col_tensor->Shape();
  ORT_RETURN_IF_NOT(col_shape.NumDimensions() == 3,
                    "'input' tensor must be 3-D with shape (N, C * prod(block_shape), L).");
  ORT_RETURN_IF_NOT(col_shape[1] > 0 && col_shape[1] % static_cast<int64_t>(kernel_shape_size) == 0,
                    "'input' dim[1] (", col_shape[1],
                    ") must be a positive multiple of prod(block_shape) (", static_cast<int64_t>(kernel_shape_size), ").");
  ORT_RETURN_IF_NOT(col_shape[2] == static_cast<int64_t>(expected_col_blocks),
                    "'input' dim[2] (", col_shape[2],
                    ") does not match the number of sliding blocks (", static_cast<int64_t>(expected_col_blocks),
                    ") implied by 'image_shape', 'block_shape', 'pads', 'strides', and 'dilations'.");

  const auto N = col_shape[0];
  const int64_t C = col_shape[1] / static_cast<int64_t>(kernel_shape_size);
  const int64_t col_stride = SafeInt<int64_t>(C) * image_shape_size;
  TensorShape adjusted_kernel_shape(adjusted_kernel_shape_dims);
  const int64_t col_data_stride = col_shape.SizeFromDimension(1);

  TensorShapeVector batched_image_shape_dims;
  batched_image_shape_dims.insert(batched_image_shape_dims.begin(), {N, C});
  for (size_t i = 0; i < image_dim_number; ++i) {
    batched_image_shape_dims.push_back(image_dims[i]);
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
          sliding_block_shape_dims.data(),
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
