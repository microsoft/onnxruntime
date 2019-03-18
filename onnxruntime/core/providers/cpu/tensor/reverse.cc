// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse.h"
#include <utility>

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
Reverse,
10,
KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
Reverse);

Status handle_scalar_tensor(const Tensor* input, Tensor* output, const MLDataType& dtype) {
  // for scalar tensors, "reversing" is just copying the input buffer to the output buffer
  // string scalar
  if (dtype == DataTypeImpl::GetType<std::string>()) {
    const std::string* src = input->template Data<std::string>();
    std::string* dst = output->template MutableData<std::string>();
    if (src != dst)
      std::copy(src, src + 1, dst);
    return Status::OK();
  }

  // non-string scalars
  const void* source = input->DataRaw(dtype);
  void* target = output->MutableDataRaw(dtype);
  if (target != source)
    memcpy(target, source, 1 * dtype->Size());

  return Status::OK();
}

template <int rank>
void ReverseImpl(const void* input_buffer, void* output_buffer, const std::vector<int64_t>& dims, const std::vector<int64_t>& reverse_axes, const MLDataType& dtype) {
  const auto& eigen_reverse_axes = vector_to_eigen_array<rank>(reverse_axes);
  if (dtype == DataTypeImpl::GetType<float>()) {
    EigenTensorMapPair<float, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<float, rank>(static_cast<const float*>(input_buffer), static_cast<float*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<double>()) {
    EigenTensorMapPair<double, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<double, rank>(static_cast<const double*>(input_buffer), static_cast<double*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<int8_t>()) {
    EigenTensorMapPair<int8_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<int8_t, rank>(static_cast<const int8_t*>(input_buffer), static_cast<int8_t*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<int16_t>()) {
    EigenTensorMapPair<int16_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<int16_t, rank>(static_cast<const int16_t*>(input_buffer), static_cast<int16_t*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<int32_t>()) {
    EigenTensorMapPair<int32_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<int32_t, rank>(static_cast<const int32_t*>(input_buffer), static_cast<int32_t*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<int64_t>()) {
    EigenTensorMapPair<int64_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<int64_t, rank>(static_cast<const int64_t*>(input_buffer), static_cast<int64_t*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<uint8_t>()) {
    EigenTensorMapPair<uint8_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<uint8_t, rank>(static_cast<const uint8_t*>(input_buffer), static_cast<uint8_t*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<uint16_t>()) {
    EigenTensorMapPair<uint16_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<uint16_t, rank>(static_cast<const uint16_t*>(input_buffer), static_cast<uint16_t*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<uint32_t>()) {
    EigenTensorMapPair<uint32_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<uint32_t, rank>(static_cast<const uint32_t*>(input_buffer), static_cast<uint32_t*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<uint64_t>()) {
    EigenTensorMapPair<uint64_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<uint64_t, rank>(static_cast<const uint64_t*>(input_buffer), static_cast<uint64_t*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<bool>()) {
    EigenTensorMapPair<bool, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<bool, rank>(static_cast<const bool*>(input_buffer), static_cast<bool*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
  } else {
    ORT_THROW("Unsupported input datatype for Reverse operator. Got ", dtype);
  }
}

template <int rank>
void ReverseImplStringType(const std::string* input_buffer, std::string* output_buffer, const std::vector<int64_t>& dims, const std::vector<int64_t>& reverse_axes) {
  const auto& eigen_reverse_axes = vector_to_eigen_array<rank>(reverse_axes);
  EigenTensorMapPair<std::string, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<std::string, rank>(static_cast<const std::string*>(input_buffer), static_cast<std::string*>(output_buffer), dims);
  eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
}

Status Reverse::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* input = p_op_kernel_context->Input<Tensor>(0);
  const auto& input_shape = input->Shape();
  const auto& input_dims = input_shape.GetDims();
  const auto& input_rank = input_shape.NumDimensions();
  auto* output = p_op_kernel_context->Output(0, input_shape);
  const auto& dtype = input->DataType();

  if (input_rank == 0)
    return handle_scalar_tensor(input, output, dtype);

  if (input_rank > 8)
    ORT_THROW("Reverse operator is not implemented for input tensors with 9 or more dimensions (rank >= 9)");

  // non-default axes - validate them for rank related correctness
  std::vector<int64_t> copy_attr_axes_ = attr_axes_;
  if (copy_attr_axes_.size() > 0) {
    if (copy_attr_axes_.size() > input_rank)
      ORT_THROW("Number of elements in axes attribute exceeds the input tensor rank in Reverse operator");
    for (const auto& axis : copy_attr_axes_) {
      if (static_cast<size_t>(axis) >= input_rank || static_cast<size_t>(axis) < -input_rank)
        ORT_THROW("Elements in axes attribute are outside bounds of the input tensor's rank in Reverse operator");
    }
  }

  const void* source = input->DataRaw(dtype);
  void* target = output->MutableDataRaw(dtype);

  // process the output tensor's buffer
  switch (input_rank) {
    case 1:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<1>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, copy_attr_axes_);
      else
        ReverseImpl<1>(source, target, input_dims, copy_attr_axes_, dtype);
      break;
    case 2:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<2>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, copy_attr_axes_);
      else
        ReverseImpl<2>(source, target, input_dims, copy_attr_axes_, dtype);
      break;
    case 3:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<3>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, copy_attr_axes_);
      else
        ReverseImpl<3>(source, target, input_dims, copy_attr_axes_, dtype);
      break;
    case 4:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<4>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, copy_attr_axes_);
      else
        ReverseImpl<4>(source, target, input_dims, copy_attr_axes_, dtype);
      break;
    case 5:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<5>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, copy_attr_axes_);
      else
        ReverseImpl<5>(source, target, input_dims, copy_attr_axes_, dtype);
      break;
    case 6:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<6>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, copy_attr_axes_);
      else
        ReverseImpl<6>(source, target, input_dims, copy_attr_axes_, dtype);
      break;
    case 7:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<7>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, copy_attr_axes_);
      else
        ReverseImpl<7>(source, target, input_dims, copy_attr_axes_, dtype);
      break;
    case 8:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<8>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, copy_attr_axes_);
      else
        ReverseImpl<8>(source, target, input_dims, copy_attr_axes_, dtype);
      break;
    default:
      ORT_THROW("Reverse operator is not implemented for input tensors with 9 or more dimensions (rank)");
  }
  return Status::OK();
}
}  // namespace onnxruntime