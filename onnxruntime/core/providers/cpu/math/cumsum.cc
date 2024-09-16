// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>

#include "cumsum.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime;

namespace onnxruntime {

namespace cumsum_op {
Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) {
  if (!axis_tensor)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor must be provided to the CumSum op");

  if (axis_tensor->Shape().NumDimensions() > 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor should be 0D or 1D");

  if (axis_tensor->IsDataType<int32_t>()) {
    axis_out = static_cast<int64_t>(axis_tensor->Data<int32_t>()[0]);
  } else if (axis_tensor->IsDataType<int64_t>()) {
    axis_out = axis_tensor->Data<int64_t>()[0];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor should be of type `int32_t` or `int64_t`");
  }

  axis_out = HandleNegativeAxis(axis_out, input_rank);

  return Status::OK();
}

}  // namespace cumsum_op

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    CumSum,
    11,
    13,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    CumSum,
    11,
    13,
    double,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<double>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    CumSum,
    11,
    13,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    CumSum,
    11,
    13,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum<int64_t>);

// Opset 14 kernels
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumSum,
    14,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumSum,
    14,
    double,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<double>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumSum,
    14,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    CumSum,
    14,
    int64_t,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                      DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum<int64_t>);

template <typename T>
CumSum<T>::CumSum(const OpKernelInfo& info) : OpKernel(info), exclusive_(), reverse_() {
  int64_t exclusive = 0;
  auto status = info.GetAttr("exclusive", &exclusive);
  if (status.IsOK()) {
    if (exclusive == 1 || exclusive == 0) {
      exclusive_ = exclusive;
    } else {
      ORT_ENFORCE("attribute exclusive can only be 0 or 1");
    }
  }
  int64_t reverse = 0;
  status = info.GetAttr("reverse", &reverse);
  if (status.IsOK()) {
    if (reverse == 1 || reverse == 0) {
      reverse_ = reverse;
    } else {
      ORT_ENFORCE("attribute reverse can only be 0 or 1");
    }
  }
}

template <typename T>
Status CumSum<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);   // input tensor
  size_t rank = input->Shape().NumDimensions();  // the rank of the input/output
  if (rank == 0)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot apply CumSum operator on a scalar");

  const Tensor* axis_tensor = ctx->Input<Tensor>(1);  // axis input tensor

  TensorShape output_shape(input->Shape());
  auto& output_tensor = *ctx->Output(0, output_shape);  // output tensor

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  int64_t axis_input = 0;
  ORT_THROW_IF_ERROR(cumsum_op::GetAxis(axis_tensor, rank, axis_input));

  // we solve the problem by using the identity that(in the case of exclusive)
  // 1) out[upper_dims...][0][lower_dims...] = 0
  // 2) out[upper_dims...][i][lower_dims...] =
  //      in[upper_dims...][i-1][lower_dims...] + out[upper_dims...][i-1][lower_dims...]
  // we loop through the [upper_dims...] and start applying the identity in each slice
  // since the [lower_dims...] are adjecent in memory, so we can add them like vectors

  const auto input_shape = input->Shape().GetDims();
  const size_t axis = onnxruntime::narrow<size_t>(axis_input);
  const int64_t dim = input->Shape()[axis];  // dimension size for the axis
  const int64_t upper_dim_count =            // number of slices we can walk through iteratively
      std::accumulate(input_shape.begin(), input_shape.begin() + axis, static_cast<int64_t>(1), std::multiplies<int64_t>());
  const int64_t lower_dim_size =  // sizes of the slices we can treat as 1D arrays
      std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  if (!reverse_) {
    const auto* input_iter = input->Data<T>();
    auto* output_iter = output_tensor.MutableData<T>();
    const auto* prev_output_iter = output_iter;

    if (exclusive_) {
      for (int64_t outer = 0; outer < upper_dim_count; outer++) {
        prev_output_iter = output_iter;
        for (int64_t inner = 0; inner < lower_dim_size; inner++) {
          *(output_iter++) = 0;
        }
        for (int64_t cum_axis = 1; cum_axis < dim; cum_axis++) {
          for (int64_t inner = 0; inner < lower_dim_size; inner++) {
            *(output_iter++) = *(prev_output_iter++) + *(input_iter++);
          }
        }
        input_iter += lower_dim_size;
      }
    } else {
      for (int64_t outer = 0; outer < upper_dim_count; outer++) {
        prev_output_iter = output_iter;
        for (int64_t inner = 0; inner < lower_dim_size; inner++) {
          *(output_iter++) = *(input_iter++);
        }
        for (int64_t cum_axis = 1; cum_axis < dim; cum_axis++) {
          for (int64_t inner = 0; inner < lower_dim_size; inner++) {
            *(output_iter++) = *(prev_output_iter++) + *(input_iter++);
          }
        }
      }
    }
  } else {
    // in this case the logic is mostly the same, but we start from the end
    const auto* input_iter = input->Data<T>() + input->Shape().Size();
    auto* output_iter = output_tensor.MutableData<T>() + output_shape.Size();
    const auto* prev_output_iter = output_iter;

    if (exclusive_) {
      for (int64_t outer = upper_dim_count - 1; outer >= 0; outer--) {
        prev_output_iter = output_iter;
        for (int64_t inner = lower_dim_size - 1; inner >= 0; inner--) {
          *(--output_iter) = 0;
        }
        for (int64_t cum_axis = dim - 1; cum_axis > 0; cum_axis--) {
          for (int64_t inner = lower_dim_size - 1; inner >= 0; inner--) {
            *(--output_iter) = *(--prev_output_iter) + *(--input_iter);
          }
        }
        input_iter -= lower_dim_size;
      }
    } else {
      for (int64_t outer = upper_dim_count - 1; outer >= 0; outer--) {
        prev_output_iter = output_iter;
        for (int64_t inner = lower_dim_size - 1; inner >= 0; inner--) {
          *(--output_iter) = *(--input_iter);
        }
        for (int64_t cum_axis = dim - 1; cum_axis > 0; cum_axis--) {
          for (int64_t inner = lower_dim_size - 1; inner >= 0; inner--) {
            *(--output_iter) = *(--prev_output_iter) + *(--input_iter);
          }
        }
      }
    }
  }

  return Status::OK();
}

};  // namespace onnxruntime
