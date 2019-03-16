// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse.h"
#include "core\util\eigen_common_wrapper.h"
#include "core\util\math_cpuonly.h"
#include <utility>
#include "core/framework/utils.h"

namespace onnxruntime {

template <typename T, int rank>
using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, rank, Eigen::RowMajor>>;

template <typename T, int rank>
using ConstEigenTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, rank, Eigen::RowMajor>>;

template <typename T, int rank>
using EigenTensorMapPair = std::pair<ConstEigenTensorMap<T, rank>, EigenTensorMap<T, rank>>;

template <typename T, int rank>
inline EigenTensorMapPair<T, rank> buffers_to_eigen_tensormaps(const T* input_buffer, T* output_buffer, const std::vector<int64_t>& dims) {
  if (rank == 1)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0])));
  else if (rank == 2)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1])));
  else if (rank == 3)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2])));
  else if (rank == 4)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3])));
  else if (rank == 5)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4])));
  else if (rank == 6)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5])));
  else if (rank == 7)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6])));
  else if (rank == 8)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7])));
  else
    ORT_THROW("unsupported conversion from raw buffers to Eigen tensors");
}

template <int rank>
void ReverseImpl(const void* input_buffer, void* output_buffer, const std::vector<int64_t>& dims, const std::vector<bool>& reverse_axes, const MLDataType& dtype) {
    auto eigen_reverse_axes = Eigen::array<bool, rank>();
    for (int i = 0; i < rank; ++i)
      eigen_reverse_axes[i] = reverse_axes[i];

	if (dtype == DataTypeImpl::GetType<uint8_t>()) {
      EigenTensorMapPair<uint8_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<uint8_t, rank>(static_cast<const uint8_t*>(input_buffer), static_cast<uint8_t*>(output_buffer), dims);
      eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    } 
	if (dtype == DataTypeImpl::GetType<uint16_t>()) {
      EigenTensorMapPair<uint16_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<uint16_t, rank>(static_cast<const uint16_t*>(input_buffer), static_cast<uint16_t*>(output_buffer), dims);
      eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    } 
	if (dtype == DataTypeImpl::GetType<uint32_t>()) {
      EigenTensorMapPair<uint32_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<uint32_t, rank>(static_cast<const uint32_t*>(input_buffer), static_cast<uint32_t*>(output_buffer), dims);
      eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    }
    if (dtype == DataTypeImpl::GetType<uint64_t>()) {
      EigenTensorMapPair<uint64_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<uint64_t, rank>(static_cast<const uint64_t*>(input_buffer), static_cast<uint64_t*>(output_buffer), dims);
        eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    }
    if (dtype == DataTypeImpl::GetType<int8_t>()) {
      EigenTensorMapPair<int8_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<int8_t, rank>(static_cast<const int8_t*>(input_buffer), static_cast<int8_t*>(output_buffer), dims);
      eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    }
    if (dtype == DataTypeImpl::GetType<int16_t>()) {
      EigenTensorMapPair<int16_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<int16_t, rank>(static_cast<const int16_t*>(input_buffer), static_cast<int16_t*>(output_buffer), dims);
      eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    }
    if (dtype == DataTypeImpl::GetType<int32_t>()) {
      EigenTensorMapPair<int32_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<int32_t, rank>(static_cast<const int32_t*>(input_buffer), static_cast<int32_t*>(output_buffer), dims);
      eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    }
    if (dtype == DataTypeImpl::GetType<int64_t>()) {
      EigenTensorMapPair<int64_t, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<int64_t, rank>(static_cast<const int64_t*>(input_buffer), static_cast<int64_t*>(output_buffer), dims);
      eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    } 
	if (dtype == DataTypeImpl::GetType<float>()) {
	EigenTensorMapPair<float, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<float, rank>(static_cast<const float*>(input_buffer), static_cast<float*>(output_buffer), dims);
    eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
	}
    if (dtype == DataTypeImpl::GetType<double>()) {
        EigenTensorMapPair<double, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<double, rank>(static_cast<const double*>(input_buffer), static_cast<double*>(output_buffer), dims);
        eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    }
    if (dtype == DataTypeImpl::GetType<bool>()) {
      EigenTensorMapPair<bool, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<bool, rank>(static_cast<const bool*>(input_buffer), static_cast<bool*>(output_buffer), dims);
      eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
    }
}

template <int rank>
void ReverseImplStringType(const std::string* input_buffer, std::string* output_buffer, const std::vector<int64_t>& dims, const std::vector<bool>& reverse_axes) {
  EigenTensorMapPair<std::string, rank> eigen_tensormaps_pair = buffers_to_eigen_tensormaps<std::string, rank>(static_cast<const std::string*>(input_buffer), static_cast<std::string*>(output_buffer), dims);
  eigen_tensormaps_pair.second = eigen_tensormaps_pair.first.reverse(eigen_reverse_axes);
}

Status Reverse::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* input = p_op_kernel_context->Input<Tensor>(0);
  const auto& input_shape = input->Shape();
  const auto& input_dims = input_shape.GetDims();
  const auto& input_rank = input_shape.NumDimensions();

  // TODO: Handle scalar Tensor
  if (input_rank > 8)
    ORT_THROW("Reverse operator is not implemented for input tensors with 9 or more dimensions (rank)");

  std::vector<bool> attr_axes_bool(input_rank, false);
  // non-default axes - process them and validate for correctness along the way
  if (attr_axes_.size() > 0) {
    if (attr_axes_.size() > input_rank)
      ORT_THROW("Number of elements in axes attribute exceeds the input tensor rank in Reverse operator");
    for (const auto& axis : attr_axes_) {
      if (static_cast<size_t>(axis) >= input_rank || static_cast<size_t>(axis) < -input_rank)
        ORT_THROW("Elements in axes attribute are outside bounds of the input tensor's rank in Reverse operator");
      attr_axes_bool[axis >= 0 ? axis : input_rank + axis] = true;
    }
  }
  // default axes - reverse all axes as per spec
  else
    std::fill(attr_axes_bool.begin(), attr_axes_bool.begin() + attr_axes_bool.size(), true);

  // create output
  auto* output = p_op_kernel_context->Output(0, input_shape);
  const auto& dtype = input->DataType();
  const void* source = input->DataRaw(dtype);
  void* target = output->MutableDataRaw(dtype);

  // process the output tensor's buffer
  switch (input_rank) {
    case 1:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<1>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, attr_axes_bool);
      else
        ReverseImpl<1>(source, target, input_dims, attr_axes_bool, dtype);
      break;
    case 2:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<2>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, attr_axes_bool);
      else
        ReverseImpl<2>(source, target, input_dims, attr_axes_bool, dtype);
      break;
    case 3:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<3>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, attr_axes_bool);
      else
		ReverseImpl<3>(source, target, input_dims, attr_axes_bool, dtype);
      break;
    case 4:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<4>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, attr_axes_bool);
      else
		ReverseImpl<4>(source, target, input_dims, attr_axes_bool, dtype);
      break;
    case 5:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<5>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, attr_axes_bool);
      else
	    ReverseImpl<5>(source, target, input_dims, attr_axes_bool, dtype);
      break;
    case 6:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<6>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, attr_axes_bool);
      else
	    ReverseImpl<6>(source, target, input_dims, attr_axes_bool, dtype);
      break;
    case 7:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<7>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, attr_axes_bool);
      else
	    ReverseImpl<7>(source, target, input_dims, attr_axes_bool, dtype);
      break;
    case 8:
      if (dtype == DataTypeImpl::GetType<std::string>())
        ReverseImplStringType<8>(input->template Data<std::string>(), output->template MutableData<std::string>(), input_dims, attr_axes_bool);
      else
	      ReverseImpl<8>(source, target, input_dims, attr_axes_bool, dtype);
      break;
    default:
      ORT_THROW("Reverse operator is not implemented for input tensors with 9 or more dimensions (rank)");
  }

  //const std::string* src = input->template Data<std::string>();
  //std::string* dst = output->template MutableData<std::string>();

  return Status::OK();
}
}  // namespace onnxruntime