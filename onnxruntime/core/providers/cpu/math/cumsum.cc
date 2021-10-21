// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cumsum.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime;

namespace {
// static section
std::vector<int64_t> GetStarts(int64_t rank, int64_t axis, int64_t index) {
  std::vector<int64_t> starts(rank, 0);
  starts[axis] = index;
  return starts;
}
template <typename T>
void ZeroOutSliceAtIndex(Tensor& output, int64_t rank, int64_t axis, int64_t index,
                         const std::vector<int64_t>& slice_dims, const std::vector<int64_t>& steps, const int64_t slice_size) {
  T zero{};
  auto output_starts(GetStarts(rank, axis, index));
  WritableSliceIterator<T> output_iterator(output, output_starts, slice_dims, steps);
  for (int64_t k = 0; k < slice_size; ++k, ++output_iterator) {
    *output_iterator = zero;
  }
}
template <typename T>
void CopySlices(const Tensor& input, Tensor& output,
                const std::vector<int64_t>& input_starts, const std::vector<int64_t>& output_starts,
                const std::vector<int64_t>& slice_dims, const std::vector<int64_t>& steps, const int64_t slice_size) {
  SliceIterator<T> input_iterator(input, input_starts, slice_dims, steps);
  WritableSliceIterator<T> output_iterator(output, output_starts, slice_dims, steps);
  for (int64_t k = 0; k < slice_size; ++k, ++output_iterator, ++input_iterator) {
    *output_iterator = *input_iterator;
  }
}
template <typename T>
void SumSlices(const Tensor& input, Tensor& output,
               const std::vector<int64_t>& input_starts, const std::vector<int64_t>& output_starts, const std::vector<int64_t>& previous_output_starts,
               const std::vector<int64_t>& slice_dims, const std::vector<int64_t>& steps, const int64_t slice_size) {
  SliceIterator<T> input_iterator(input, input_starts, slice_dims, steps);
  WritableSliceIterator<T> output_iterator(output, output_starts, slice_dims, steps);
  SliceIterator<T> previous_output_iterator(output, previous_output_starts, slice_dims, steps);
  for (int64_t k = 0; k < slice_size; ++k, ++output_iterator, ++input_iterator, ++previous_output_iterator) {
    *output_iterator = *input_iterator + *previous_output_iterator;
  }
}
}  // namespace

namespace onnxruntime {

namespace cumsum_op {
Status GetAxis(const Tensor* axis_tensor, int64_t input_rank, int64_t& axis_out) {
  if (!axis_tensor)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor must be provided to the CumSum op");

  if (axis_tensor->Shape().NumDimensions() > 1)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor should be 0D or 1D");

  if (axis_tensor->IsDataType<int32_t>()) {
    axis_out = static_cast<int64_t>(axis_tensor->template Data<int32_t>()[0]);
  } else if (axis_tensor->IsDataType<int64_t>()) {
    axis_out = axis_tensor->template Data<int64_t>()[0];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Axis tensor should be of type `int32_t` or `int64_t`");
  }

  axis_out = HandleNegativeAxis(axis_out, input_rank);

  return Status::OK();
}

}  // namespace cumsum_op

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(CumSum,
                               11,
                               13,
                               float,
                               KernelDefBuilder()
                                   .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                                   .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
                               CumSum<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(CumSum,
                               11,
                               13,
                               double,
                               KernelDefBuilder()
                                   .TypeConstraint("T", DataTypeImpl::GetTensorType<double>())
                                   .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
                               CumSum<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(CumSum,
                               11,
                               13,
                               int32_t,
                               KernelDefBuilder()
                                   .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
                                   .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
                               CumSum<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(CumSum,
                               11,
                               13,
                               int64_t,
                               KernelDefBuilder()
                                   .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
                                   .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
                               CumSum<int64_t>);

// Opset 14 kernels
ONNX_CPU_OPERATOR_TYPED_KERNEL(CumSum,
                               14,
                               float,
                               KernelDefBuilder()
                                   .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
                                   .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
                               CumSum<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(CumSum,
                               14,
                               double,
                               KernelDefBuilder()
                                   .TypeConstraint("T", DataTypeImpl::GetTensorType<double>())
                                   .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
                               CumSum<double>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(CumSum,
                               14,
                               int32_t,
                               KernelDefBuilder()
                                   .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>())
                                   .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
                               CumSum<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(CumSum,
                               14,
                               int64_t,
                               KernelDefBuilder()
                                   .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
                                   .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
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
  const Tensor* input = ctx->Input<Tensor>(0);                       // input tensor
  auto rank = static_cast<int64_t>(input->Shape().NumDimensions());  // the rank of the input/output
  if (rank == 0)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot apply CumSum operator on a scalar");

  const Tensor* axis_tensor = ctx->Input<Tensor>(1);  // axis input tensor

  TensorShape output_shape(input->Shape());
  auto& output_tensor = *ctx->Output(0, output_shape);  // output tensor

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  int64_t axis = 0;
  ORT_THROW_IF_ERROR(cumsum_op::GetAxis(axis_tensor, rank, axis));

  auto dim(output_tensor.Shape()[axis]);    // dimension size for the axis
  TensorShape slice_shape(input->Shape());  // the shape of one slice of input/output for the given value of the axis
  slice_shape[axis] = 1;
  auto slice_size(slice_shape.Size());     // total number of elements in each slice
  auto slice_dims(slice_shape.GetDims());  // dim array for the slice

  std::vector<int64_t> steps(rank, 1);  // steps for the slice -- always set to 1

  if (!reverse_) {
    int64_t index(0);  // the index we use as we walkthrough the given axis
    // If (exclusive == true) the first slice is always 0
    if (exclusive_) {
      ::ZeroOutSliceAtIndex<T>(output_tensor, rank, axis, index, slice_dims, steps, slice_size);
      ++index;
    }
    {
      // The next slice is a copy of the input (if exclusive == false then this is the first slice)
      auto input_starts(::GetStarts(rank, axis, 0));
      auto output_starts(::GetStarts(rank, axis, index));
      ::CopySlices<T>(*input, output_tensor, input_starts, output_starts, slice_dims, steps, slice_size);
      ++index;
    }

    for (; index < dim; ++index) {
      // Each output slice is the sum of corresponding input slice and the previous output slice
      auto input_starts(::GetStarts(rank, axis, exclusive_ ? index - 1 : index));
      auto output_starts(::GetStarts(rank, axis, index));
      auto previous_starts(::GetStarts(rank, axis, index - 1));
      ::SumSlices<T>(*input, output_tensor, input_starts, output_starts, previous_starts,
                     slice_dims, steps, slice_size);
    }
  } else {
    //_reverse == true
    int64_t index(dim - 1);  // the index we use as we walkthrough the given axis
    // If (exclusive == true) the first slice is always 0
    if (exclusive_) {
      ::ZeroOutSliceAtIndex<T>(output_tensor, rank, axis, index, slice_dims, steps, slice_size);
      --index;
    }
    {
      // The next slice is a copy of the input (if exclusive == false then this is the first slice)
      auto input_starts(::GetStarts(rank, axis, dim - 1));
      auto output_starts(::GetStarts(rank, axis, index));
      ::CopySlices<T>(*input, output_tensor, input_starts, output_starts, slice_dims, steps, slice_size);
      --index;
    }

    for (; index >= 0; --index) {
      // Each output slice is the sum of corresponding input slice and the previous output slice
      auto input_starts(::GetStarts(rank, axis, exclusive_ ? index + 1 : index));
      auto output_starts(::GetStarts(rank, axis, index));
      auto previous_starts(::GetStarts(rank, axis, index + 1));
      ::SumSlices<T>(*input, output_tensor, input_starts, output_starts, previous_starts,
                     slice_dims, steps, slice_size);
    }
  }

  return Status::OK();
}

};  // namespace onnxruntime
