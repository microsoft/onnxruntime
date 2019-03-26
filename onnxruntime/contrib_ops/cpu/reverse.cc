// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse.h"
#include <utility>
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/utils.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Reverse,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Reverse);

// some templated aliases
template <typename T, int rank>
using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, rank, Eigen::RowMajor, Eigen::DenseIndex>>;

template <typename T, int rank>
using ConstEigenTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, rank, Eigen::RowMajor, Eigen::DenseIndex>>;

// utility helpers specific to Reverse
template <int rank>
Eigen::array<bool, rank> generate_bool_reverse_axes(const std::vector<int64_t>& reverse_axes) {
  Eigen::array<bool, rank> eigen_reverse_axes;

  // default axes - reverse all axes as per spec
  if (reverse_axes.size() == 0) {
    for (int i = 0; i < rank; ++i) {
      eigen_reverse_axes[i] = true;
    }
    return eigen_reverse_axes;
  }

  // explicit axes given
  eigen_reverse_axes.fill(false);
  for (int i = 0; i < reverse_axes.size(); ++i) {
    const auto& dim = reverse_axes[i];
    eigen_reverse_axes[dim >= 0 ? dim : dim + rank] = true;
  }
  return eigen_reverse_axes;
}

template <int rank>
Eigen::DSizes<Eigen::DenseIndex, rank> dims_as_eigen_dsizes(const std::vector<int64_t> dims) {
  Eigen::DSizes<Eigen::DenseIndex, rank> eigen_dsizes;
  for (int i = 0; i < rank; ++i) {
    eigen_dsizes[i] = static_cast<Eigen::DenseIndex>(dims[i]);
  }
  return eigen_dsizes;
}

template <typename T, int rank>
EigenTensorMap<T, rank> buffer_as_eigen_tensor(T* buffer, const std::vector<int64_t>& dims) {
  return EigenTensorMap<T, rank>(buffer, dims_as_eigen_dsizes<rank>(dims));
}

template <typename T, int rank>
ConstEigenTensorMap<T, rank> buffer_as_const_eigen_tensor(const T* buffer, const std::vector<int64_t>& dims) {
  return ConstEigenTensorMap<T, rank>(buffer, dims_as_eigen_dsizes<rank>(dims));
}

Status handle_scalar_tensor(const Tensor* input_tensor, Tensor* output_tensor, const MLDataType& dtype) {
  // for scalar tensors, "reversing" is just copying the input buffer to the output buffer

  // string scalar
  if (dtype == DataTypeImpl::GetType<std::string>()) {
    const std::string* src = input_tensor->template Data<std::string>();
    std::string* dst = output_tensor->template MutableData<std::string>();
    dst[0] = src[0];
    return Status::OK();
  }

  // MLFLoat16 scalar
  if (dtype == DataTypeImpl::GetType<MLFloat16>()) {
    const auto* src = input_tensor->Data<MLFloat16>();
    auto* dst = output_tensor->template MutableData<MLFloat16>();
    dst[0] = src[0];
    return Status::OK();
  }

  // BFLoat16 scalar
  if (dtype == DataTypeImpl::GetType<BFloat16>()) {
    const auto* src = input_tensor->Data<BFloat16>();
    auto* dst = output_tensor->template MutableData<BFloat16>();
    dst[0] = src[0];
    return Status::OK();
  }

  // non-string and non-float16 scalars - just copy the raw bytes
  const void* src = input_tensor->DataRaw(dtype);
  void* dst = output_tensor->MutableDataRaw(dtype);
  if (src != dst)
    memcpy(dst, src, 1 * dtype->Size());

  return Status::OK();
}

template <int rank>
void ReverseImplMLFloat16Type(const OpKernelContext* ctx, const Tensor* input_tensor, 
Tensor* output_tensor, const TensorShape& shape, const std::vector<int64_t>& reverse_axes) {
  AllocatorPtr allocator;
  ctx->GetTempSpaceAllocator(&allocator);
  ORT_ENFORCE(allocator != nullptr, "Temporary memory allocator failed");

  const int64_t len = shape.Size();
  ORT_ENFORCE(len > 0, "Need atleast one float16 element to be processed in Reverse operator");

  // allocate intermediate buffers to be used for processing
  float* input_buffer = static_cast<float*>(allocator->AllocArray(sizeof(float), len));
  float* output_buffer = static_cast<float*>(allocator->AllocArray(sizeof(float), len));
  ORT_ENFORCE(input_buffer && output_buffer, "Could not allocate enough memory "
	                                         "to process float16 data in Reverse operator");

  const auto& dims = shape.GetDims();
  const auto& eigen_reverse_axes = generate_bool_reverse_axes<rank>(reverse_axes);

  // fill the intermediate input buffer with the values in the input tensor
  const auto& span = gsl::make_span(input_tensor->Data<MLFloat16>(), len);
  std::transform(span.cbegin(), span.cend(), input_buffer, [](const MLFloat16& val) 
  { return math::halfToFloat(val.val); });

  // process the intermediate input buffer and use the intermediate output buffer to hold the output
  EigenTensorMap<float, rank> eigen_tensor_input = buffer_as_eigen_tensor<float, rank>(input_buffer, dims);
  EigenTensorMap<float, rank> eigen_tensor_output = buffer_as_eigen_tensor<float, rank>(output_buffer, dims);

  eigen_tensor_output = eigen_tensor_input.reverse(eigen_reverse_axes);

  // fill output tensor's values with results in intermediate output buffer
  auto* output_data = output_tensor->template MutableData<MLFloat16>();
  std::transform(output_buffer, output_buffer + len, output_data, [](const float& val) 
  { return MLFloat16(math::floatToHalf(val)); });

  // free the intermediate buffers
  allocator->Free(input_buffer);
  allocator->Free(output_buffer);
}

template <int rank>
void ReverseImplBFloat16Type(const OpKernelContext* ctx, const Tensor* input_tensor, 
Tensor* output_tensor, const TensorShape& shape, const std::vector<int64_t>& reverse_axes) {
  AllocatorPtr allocator;
  ctx->GetTempSpaceAllocator(&allocator);
  ORT_ENFORCE(allocator != nullptr, "Temporary memory allocator failed");

  const int64_t len = shape.Size();
  ORT_ENFORCE(len > 0, "Need atleast one float16 element to be processed in Reverse operator");

  // allocate intermediate buffers to be used for processing
  float* input_buffer = static_cast<float*>(allocator->AllocArray(sizeof(float), len));
  float* output_buffer = static_cast<float*>(allocator->AllocArray(sizeof(float), len));
  ORT_ENFORCE(input_buffer && output_buffer, "Could not allocate enough memory " 
	                                         "to process float16 data in Reverse operator");

  const auto& dims = shape.GetDims();
  const auto& eigen_reverse_axes = generate_bool_reverse_axes<rank>(reverse_axes);

  // fill the intermediate input buffer with the values in the input tensor
  const auto& span = gsl::make_span(input_tensor->Data<BFloat16>(), len);
  std::transform(span.cbegin(), span.cend(), input_buffer, [](const BFloat16& val) { return val.ToFloat(); });

  // process the intermediate input buffer and use the intermediate output buffer to hold the output
  EigenTensorMap<float, rank> eigen_tensor_input = buffer_as_eigen_tensor<float, rank>(input_buffer, dims);
  EigenTensorMap<float, rank> eigen_tensor_output = buffer_as_eigen_tensor<float, rank>(output_buffer, dims);

  eigen_tensor_output = eigen_tensor_input.reverse(eigen_reverse_axes);

  // fill output tensor's values with results in intermediate output buffer
  auto* output_data = output_tensor->template MutableData<BFloat16>();
  std::transform(output_buffer, output_buffer + len, output_data, [](const float& val) { return BFloat16(val); });

  // free the intermediate buffers
  allocator->Free(input_buffer);
  allocator->Free(output_buffer);
}

#define ProcessType(type, ml_data_type, rank, dims)                                                              \
  const ConstEigenTensorMap<type, rank> input =                                                                  \
  buffer_as_const_eigen_tensor<type, rank>(static_cast<const type*>(input_tensor->DataRaw(ml_data_type)), dims); \
  EigenTensorMap<type, rank> output =                                                                            \
  buffer_as_eigen_tensor<type, rank>(static_cast<type*>(output_tensor->MutableDataRaw(ml_data_type)), dims);     \
  output = input.reverse(eigen_reverse_axes);

template <int rank>
void ReverseImpl(const OpKernelContext* ctx, const Tensor* input_tensor, Tensor* output_tensor, 
const TensorShape& shape, const std::vector<int64_t>& reverse_axes) {
  const auto& dtype = input_tensor->DataType();
  const auto& dims = shape.GetDims();
  const auto& eigen_reverse_axes = generate_bool_reverse_axes<rank>(reverse_axes);

  if (dtype == DataTypeImpl::GetType<float>()) {
    ProcessType(float, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<double>()) {
    ProcessType(double, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<bool>()) {
    ProcessType(bool, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<int8_t>()) {
    ProcessType(int8_t, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<int16_t>()) {
    ProcessType(int16_t, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<int32_t>()) {
    ProcessType(int32_t, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<int64_t>()) {
    ProcessType(int64_t, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<uint8_t>()) {
    ProcessType(uint8_t, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<uint16_t>()) {
    ProcessType(uint16_t, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<uint32_t>()) {
    ProcessType(uint32_t, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<uint64_t>()) {
    ProcessType(uint64_t, dtype, rank, dims)
  } else if (dtype == DataTypeImpl::GetType<std::string>()) {
    const ConstEigenTensorMap<std::string, rank> input = 
    buffer_as_const_eigen_tensor<std::string, rank>(static_cast<const std::string*>(input_tensor->template Data<std::string>()), dims);
    EigenTensorMap<std::string, rank> output = 
	buffer_as_eigen_tensor<std::string, rank>(static_cast<std::string*>(output_tensor->template MutableData<std::string>()), dims);
    output = input.reverse(eigen_reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<MLFloat16>()) {
    ReverseImplMLFloat16Type<rank>(ctx, input_tensor, output_tensor, shape, reverse_axes);
  } else if (dtype == DataTypeImpl::GetType<BFloat16>()) {
    ReverseImplBFloat16Type<rank>(ctx, input_tensor, output_tensor, shape, reverse_axes);
  } else {
    ORT_THROW("Unsupported input datatype for Reverse operator. Got ", dtype);
  }
}

Status Reverse::Compute(OpKernelContext* p_op_kernel_context) const {
  const auto* input = p_op_kernel_context->Input<Tensor>(0);
  const auto& shape = input->Shape();
  const auto& rank = shape.NumDimensions();
  auto* output = p_op_kernel_context->Output(0, shape);
  const auto& dtype = input->DataType();

  if (rank == 0)
    return handle_scalar_tensor(input, output, dtype);

  // non-default axes - validate them for rank related correctness
  std::vector<int64_t> copy_attr_axes_ = attr_axes_;
  std::unordered_set<int64_t> axes_elements;
  if (copy_attr_axes_.size() > 0) {
    if (copy_attr_axes_.size() > rank)
      ORT_THROW("Number of elements in axes attribute exceeds the input tensor rank in Reverse operator");
    for (const auto& axis : copy_attr_axes_) {
      if (axis >= static_cast<int64_t>(rank) || axis < static_cast<int64_t>(-rank))
        ORT_THROW("Elements in axes attribute are outside bounds of the input tensor's rank in Reverse operator");
      // check for implicit and explicit dupes in axes attribute
      // explicit dupes (e.g.) 0 and 0
      // implicit dupes (e.g.) -1 and 1 for an input tensor of rank 2
      if (axis < 0) {
        if (axes_elements.find(axis + rank) != axes_elements.end())
          ORT_THROW("axes attribute has duplicates in Reverse operator");
        axes_elements.insert(axis + rank);
      } else {
        if (axes_elements.find(axis) != axes_elements.end())
          ORT_THROW("axes attribute has duplicates in Reverse operator");
        axes_elements.insert(axis);
      }
    }
  }

  // process the output tensor's buffer
  switch (rank) {
    case 1:
      ReverseImpl<1>(p_op_kernel_context, input, output, shape, copy_attr_axes_);
      break;
    case 2:
      ReverseImpl<2>(p_op_kernel_context, input, output, shape, copy_attr_axes_);
      break;
    case 3:
      ReverseImpl<3>(p_op_kernel_context, input, output, shape, copy_attr_axes_);
      break;
    case 4:
      ReverseImpl<4>(p_op_kernel_context, input, output, shape, copy_attr_axes_);
      break;
    case 5:
      ReverseImpl<5>(p_op_kernel_context, input, output, shape, copy_attr_axes_);
      break;
    case 6:
      ReverseImpl<6>(p_op_kernel_context, input, output, shape, copy_attr_axes_);
      break;
    case 7:
      ReverseImpl<7>(p_op_kernel_context, input, output, shape, copy_attr_axes_);
      break;
    case 8:
      ReverseImpl<8>(p_op_kernel_context, input, output, shape, copy_attr_axes_);
      break;
    default:
      ORT_THROW("Reverse operator is not implemented for input tensors with rank 9 or more. "
		        "Got input tensor of rank: ", rank);
  }
  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime