// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/nn/pool_gradient_op.h"
#include "core/common/narrow.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include <algorithm>

using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace contrib {

std::vector<VectorInt64> InferOutputShapes(OpKernelInfo info) {
  std::vector<VectorInt64> output_tensor_shapes = {};

  auto& node = info.node();
  auto output_defs = node.OutputDefs();
  auto outputCount = output_defs.size();

  for (size_t outputIndex = 0; outputIndex < outputCount; outputIndex++) {
    output_tensor_shapes.push_back({});
    if (!output_defs[outputIndex]->Exists())
      continue;

    auto shape = output_defs[outputIndex]->Shape();
    for (auto dim : shape->dim()) {
      output_tensor_shapes[outputIndex].push_back(dim.dim_value());
    }
  }
  return output_tensor_shapes;
}

ONNX_CPU_OPERATOR_KERNEL(
    MaxPoolGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MaxPoolGrad<float>);

template <typename T>
Status MaxPoolGrad<T>::Compute(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* indices = context->Input<Tensor>(1);
  ORT_ENFORCE(dY->Shape() == indices->Shape(), "The shape of dY and indices does not match in MaxPoolGrad.");

  TensorShape dX_shape(output_tensor_shapes_[0]);
  dX_shape[0] = dY->Shape()[0];
  Tensor* dX = context->Output(0, dX_shape);

  const T* dY_data = dY->template Data<T>();
  const int64_t* indices_data = indices->template Data<int64_t>();
  T* dX_data = dX->template MutableData<T>();

  EigenVectorMap<T>(dX_data, narrow<Eigen::Index>(dX_shape.Size())).setZero();

  for (int64_t i = 0; i < dY->Shape().Size(); ++i) {
    T* p_dX_data = dX_data + indices_data[i];
    *p_dX_data += dY_data[i];
  }

  return Status::OK();
}

template <typename T>
Status AveragePoolGrad<T>::Compute3DAveragePoolGrad(OpKernelContext* context) const {
  const TensorShape dX_shape = TensorShape::FromExistingBuffer(output_tensor_shapes_[0]);
  Tensor* dX = context->Output(0, dX_shape);
  T* dX_data = dX->template MutableData<T>();

  const Tensor* dY = context->Input<Tensor>(0);
  const TensorShape& dY_shape = dY->Shape();
  const T* dY_data = dY->template Data<T>();

  auto channels = dX_shape[0] * dX_shape[1];

  int64_t kernel_first_dim = kernel_shape_[0];
  int64_t kernel_second_dim = kernel_shape_[1];
  int64_t kernel_third_dim = kernel_shape_[2];

  int64_t stride_first_dim = strides_[0];
  int64_t stride_second_dim = strides_[1];
  int64_t stride_third_dim = strides_[2];

  int64_t dX_elements_in_each_channel = dX_shape[2] * dX_shape[3] * dX_shape[4];
  int64_t dY_elements_in_each_channel = dY_shape[2] * dY_shape[3] * dY_shape[4];

  for (int cur_channel = 0; cur_channel < channels; cur_channel++) {
    ConstEigenArrayMap<T> dY_arr = ConstEigenArrayMap<T>(dY_data, narrow<Eigen::Index>(dY_shape[3]),
                                                         narrow<Eigen::Index>(dY_shape[2] * dY_shape[4]));
    EigenArrayMap<T> dX_arr = EigenArrayMap<T>(dX_data, narrow<Eigen::Index>(dX_shape[3]),
                                               narrow<Eigen::Index>(dX_shape[2] * dX_shape[4]));

    for (int d = 0; d < dY_shape[4]; ++d) {
      const int64_t p = std::max<int64_t>(d * stride_third_dim - pads_[2], 0);
      const int64_t a = std::min<int64_t>(d * stride_third_dim - pads_[2] + kernel_third_dim, dX_shape[4]);

      for (int h = 0; h < dY_shape[2]; ++h) {
        const Eigen::Index t = narrow<Eigen::Index>(std::max<int64_t>(h * stride_first_dim - pads_[0], 0));
        const Eigen::Index b =
            narrow<Eigen::Index>(std::min<int64_t>(h * stride_first_dim - pads_[0] + kernel_first_dim, dX_shape[2]));

        for (int w = 0; w < dY_shape[3]; ++w) {
          const Eigen::Index l = narrow<Eigen::Index>(std::max<int64_t>(w * stride_second_dim - pads_[1], 0));
          const Eigen::Index r =
              narrow<Eigen::Index>(std::min<int64_t>(w * stride_second_dim - pads_[1] + kernel_second_dim,
                                                     dX_shape[3]));
          const Eigen::Index dy_index = narrow<Eigen::Index>(d * dY_shape[2] * dY_shape[3] + h * dY_shape[3] + w);
          const T scale = T(1.0f) /
                          static_cast<T>(count_include_pad_ ? kernel_first_dim * kernel_second_dim * kernel_third_dim
                                                            : (a - p) * (b - t) * (r - l));

          for (int64_t i = p; i < a; ++i) {
            dX_arr.block(l, narrow<Eigen::Index>(i * dX_shape[2] + t), r - l, b - t) = dY_arr(dy_index) * scale;
          }
        }
      }
    }

    dY_data += dY_elements_in_each_channel;
    dX_data += dX_elements_in_each_channel;
  }

  return Status::OK();
}

template <typename T>
Status AveragePoolGrad<T>::Compute2DAveragePoolGrad(OpKernelContext* context) const {
  const TensorShape dX_shape = TensorShape::FromExistingBuffer(output_tensor_shapes_[0]);
  Tensor* dX = context->Output(0, dX_shape);
  T* dX_data = dX->template MutableData<T>();

  const Tensor* dY = context->Input<Tensor>(0);
  const TensorShape& dY_shape = dY->Shape();
  const T* dY_data = dY->template Data<T>();

  auto channels = dX_shape[0] * dX_shape[1];

  int64_t kernel_first_dim = kernel_shape_[0];
  int64_t kernel_second_dim = kernel_shape_[1];

  int64_t stride_first_dim = strides_[0];
  int64_t stride_second_dim = strides_[1];

  int64_t dX_elements_in_each_channel = dX_shape[2] * dX_shape[3];
  int64_t dY_elements_in_each_channel = dY_shape[2] * dY_shape[3];

  T* dX_ptr = dX_data;
  const T* dY_ptr = dY_data;

  for (int cur_channel = 0; cur_channel < channels; cur_channel++) {
    ConstEigenArrayMap<T> dY_arr = ConstEigenArrayMap<T>(dY_ptr, narrow<Eigen::Index>(dY_shape[3]),
                                                         narrow<Eigen::Index>(dY_shape[2]));
    EigenArrayMap<T> dX_arr = EigenArrayMap<T>(dX_ptr, narrow<Eigen::Index>(dX_shape[3]),
                                               narrow<Eigen::Index>(dX_shape[2]));

    for (int h = 0; h < dY_shape[2]; ++h) {
      const Eigen::Index t = narrow<Eigen::Index>(std::max<int64_t>(h * stride_first_dim - pads_[0], 0));
      const Eigen::Index b =
          narrow<Eigen::Index>(std::min<int64_t>(h * stride_first_dim - pads_[0] + kernel_first_dim, dX_shape[2]));

      for (int w = 0; w < dY_shape[3]; ++w) {
        const Eigen::Index l = narrow<Eigen::Index>(std::max<int64_t>(w * stride_second_dim - pads_[1], 0));
        const Eigen::Index r =
            narrow<Eigen::Index>(std::min<int64_t>(w * stride_second_dim - pads_[1] + kernel_second_dim, dX_shape[3]));

        const Eigen::Index dy_index = narrow<Eigen::Index>(h * dY_shape[3] + w);
        const T scale = T(1.0f) / static_cast<T>(count_include_pad_ ? kernel_first_dim * kernel_second_dim : (b - t) * (r - l));
        dX_arr.block(l, t, r - l, b - t) += dY_arr(dy_index) * scale;
      }
    }

    // move to next channel
    dY_ptr += dY_elements_in_each_channel;
    dX_ptr += dX_elements_in_each_channel;
  }

  return Status::OK();
}
template <typename T>
Status AveragePoolGrad<T>::Compute1DAveragePoolGrad(OpKernelContext* context) const {
  const TensorShape dX_shape = TensorShape::FromExistingBuffer(output_tensor_shapes_[0]);
  Tensor* dX = context->Output(0, dX_shape);
  T* dX_data = dX->template MutableData<T>();

  const Tensor* dY = context->Input<Tensor>(0);
  const TensorShape& dY_shape = dY->Shape();
  const T* dY_data = dY->template Data<T>();

  auto channels = dX_shape[0] * dX_shape[1];  // total channels across all batches
  int64_t kernel_size = kernel_shape_[0];
  int64_t stride_size = strides_[0];

  float* dX_ptr = dX_data;
  const float* dY_ptr = dY_data;

  for (int cur_channel = 0; cur_channel < channels; cur_channel++) {  // for every channel group
    ConstEigenArrayMap<T> dY_arr = ConstEigenArrayMap<T>(dY_ptr, narrow<Eigen::Index>(dY_shape[2]), 1);
    EigenArrayMap<T> dX_arr = EigenArrayMap<T>(dX_ptr, narrow<Eigen::Index>(dX_shape[2]), 1);

    for (int dy_index = 0; dy_index < dY_shape[2]; ++dy_index) {
      const auto left_index = narrow<Eigen::Index>(std::max<int64_t>(dy_index * stride_size - pads_[0], 0));
      const auto right_index = narrow<Eigen::Index>(std::min<int64_t>(dy_index * stride_size - pads_[0] + kernel_size,
                                                                      dX_shape[2]));

      const T scale = T(1.0f) / static_cast<T>(count_include_pad_ ? kernel_size : right_index - left_index);

      dX_arr.col(0).segment(left_index, right_index - left_index) += dY_arr(dy_index) * scale;
    }

    // move to the next channel
    dY_ptr += dY_shape[2];
    dX_ptr += dX_shape[2];
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    AveragePoolGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    AveragePoolGrad<float>);

// only StorageOrder::NCHW supported
template <typename T>
Status AveragePoolGrad<T>::Compute(OpKernelContext* context) const {
  const TensorShape dX_shape = TensorShape::FromExistingBuffer(output_tensor_shapes_[0]);
  Tensor* dX = context->Output(0, dX_shape);
  T* dX_data = dX->template MutableData<T>();
  EigenVectorMap<T>(dX_data, narrow<Eigen::Index>(dX_shape.Size())).setZero();

  switch (dX_shape.NumDimensions()) {
    case 3:
      ORT_RETURN_IF_ERROR(Compute1DAveragePoolGrad(context));
      break;

    case 4:
      ORT_RETURN_IF_ERROR(Compute2DAveragePoolGrad(context));
      break;

    case 5:
      ORT_RETURN_IF_ERROR(Compute3DAveragePoolGrad(context));
      break;

    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size : ");
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
