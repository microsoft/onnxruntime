// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse.h"
#include <utility>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Reverse,
    kMSDomain,
    1,
    kCpuExecutionProvider,
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
    const ConstEigenTensorMap<float, rank> input = buffer_as_const_eigen_tensor<float, rank>(static_cast<const float*>(input_buffer), dims);
    EigenTensorMap<float, rank> output = buffer_as_eigen_tensor<float, rank>(static_cast<float*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<double>()) {
    const ConstEigenTensorMap<double, rank> input = buffer_as_const_eigen_tensor<double, rank>(static_cast<const double*>(input_buffer), dims);
    EigenTensorMap<double, rank> output = buffer_as_eigen_tensor<double, rank>(static_cast<double*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<int8_t>()) {
    const ConstEigenTensorMap<int8_t, rank> input = buffer_as_const_eigen_tensor<int8_t, rank>(static_cast<const int8_t*>(input_buffer), dims);
    EigenTensorMap<int8_t, rank> output = buffer_as_eigen_tensor<int8_t, rank>(static_cast<int8_t*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<int16_t>()) {
    const ConstEigenTensorMap<int16_t, rank> input = buffer_as_const_eigen_tensor<int16_t, rank>(static_cast<const int16_t*>(input_buffer), dims);
    EigenTensorMap<int16_t, rank> output = buffer_as_eigen_tensor<int16_t, rank>(static_cast<int16_t*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<int32_t>()) {
    const ConstEigenTensorMap<int32_t, rank> input = buffer_as_const_eigen_tensor<int32_t, rank>(static_cast<const int32_t*>(input_buffer), dims);
    EigenTensorMap<int32_t, rank> output = buffer_as_eigen_tensor<int32_t, rank>(static_cast<int32_t*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<int64_t>()) {
    const ConstEigenTensorMap<int64_t, rank> input = buffer_as_const_eigen_tensor<int64_t, rank>(static_cast<const int64_t*>(input_buffer), dims);
    EigenTensorMap<int64_t, rank> output = buffer_as_eigen_tensor<int64_t, rank>(static_cast<int64_t*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<uint8_t>()) {
    const ConstEigenTensorMap<uint8_t, rank> input = buffer_as_const_eigen_tensor<uint8_t, rank>(static_cast<const uint8_t*>(input_buffer), dims);
    EigenTensorMap<uint8_t, rank> output = buffer_as_eigen_tensor<uint8_t, rank>(static_cast<uint8_t*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<uint16_t>()) {
    const ConstEigenTensorMap<uint16_t, rank> input = buffer_as_const_eigen_tensor<uint16_t, rank>(static_cast<const uint16_t*>(input_buffer), dims);
    EigenTensorMap<uint16_t, rank> output = buffer_as_eigen_tensor<uint16_t, rank>(static_cast<uint16_t*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<uint32_t>()) {
    const ConstEigenTensorMap<uint32_t, rank> input = buffer_as_const_eigen_tensor<uint32_t, rank>(static_cast<const uint32_t*>(input_buffer), dims);
    EigenTensorMap<uint32_t, rank> output = buffer_as_eigen_tensor<uint32_t, rank>(static_cast<uint32_t*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<uint64_t>()) {
    const ConstEigenTensorMap<uint64_t, rank> input = buffer_as_const_eigen_tensor<uint64_t, rank>(static_cast<const uint64_t*>(input_buffer), dims);
    EigenTensorMap<uint64_t, rank> output = buffer_as_eigen_tensor<uint64_t, rank>(static_cast<uint64_t*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<bool>()) {
    const ConstEigenTensorMap<bool, rank> input = buffer_as_const_eigen_tensor<bool, rank>(static_cast<const bool*>(input_buffer), dims);
    EigenTensorMap<bool, rank> output = buffer_as_eigen_tensor<bool, rank>(static_cast<bool*>(output_buffer), dims);
    output = input.reverse(eigen_reverse_axes);
  } else {
    ORT_THROW("Unsupported input datatype for Reverse operator. Got ", dtype);
  }
}

template <int rank>
void ReverseImplStringType(const std::string* input_buffer, std::string* output_buffer, const std::vector<int64_t>& dims, const std::vector<int64_t>& reverse_axes) {
  const auto& eigen_reverse_axes = vector_to_eigen_array<rank>(reverse_axes);
  const ConstEigenTensorMap<std::string, rank> input = buffer_as_const_eigen_tensor<std::string, rank>(static_cast<const std::string*>(input_buffer), dims);
  EigenTensorMap<std::string, rank> output = buffer_as_eigen_tensor<std::string, rank>(static_cast<std::string*>(output_buffer), dims);
  output = input.reverse(eigen_reverse_axes);
}

template <int rank>
void ReverseImplMLFloat16Type(const MLFloat16* input_buffer, MLFloat16* output_buffer, const TensorShape& shape, const std::vector<int64_t>& reverse_axes) {
    ORT_ENFORCE(allocator != nullptr, "ORT needs access to allocator to process float16 data in Reverse operator");
    const int64_t len = shape.Size();
    ORT_ENFORCE(len > 0, "Need atleast one float16 element to be processed in Reverse operator");
    void* buffer = allocator->AllocArray(sizeof(float), len);
    ORT_ENFORCE(buffer, "ORT cannot allocate enough memory to process float16 data in Reverse operator");
    Tensor tmp_tensor(DataTypeImpl::GetType<float>(), shape, buffer, allocator->Info());

    allocator->Free(buffer);
  }
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

  // non-default axes - validate them for rank related correctness
  std::vector<int64_t> copy_attr_axes_ = attr_axes_;
  std::unordered_set<int64_t> axes_elements;
  if (copy_attr_axes_.size() > 0) {
    if (copy_attr_axes_.size() > input_rank)
      ORT_THROW("Number of elements in axes attribute exceeds the input tensor rank in Reverse operator");
    for (const auto& axis : copy_attr_axes_) {
      if (axis >= static_cast<int64_t>(input_rank) || axis < static_cast<int64_t>(-input_rank))
        ORT_THROW("Elements in axes attribute are outside bounds of the input tensor's rank in Reverse operator");
       // check for implicit dupes - (e.g.) axes contains -1 and 4 for an input tensor of rank 5
	   // TODO: Do we need checks like these ? It could unnecessarily affect run-time perf
	  if (axis < 0) {
        if (axes_elements.find(axis + input_rank) != axes_elements.end())
          ORT_THROW("axes attribute has implicit dupes - a negative value corresponding to an existing positive value in the axes attribute" 
			        " for the rank of the input tensor ", input_rank);
        axes_elements.insert(axis + input_rank);      
	  } else {
        if (axes_elements.find(axis) != axes_elements.end())
            ORT_THROW("axes attribute has implicit dupes - a negative value corresponding to an existing positive value in the axes attribute"
                      " for the rank of the input tensor ", input_rank);
        axes_elements.insert(axis);        
	  }
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
    default:
      ORT_THROW("Reverse operator is not implemented for input tensors with 9 or more dimensions (rank)");
  }
  return Status::OK();
}
}
}  // namespace onnxruntime 