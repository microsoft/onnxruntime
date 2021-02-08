// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cumsum.h"
#include "cumsum_impl.h"
#include "core/providers/cpu/math/cumsum.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    11, 13,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(1)  // 'axis' needs to be on CPU
        .TypeConstraint("T", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>(),
                                 DataTypeImpl::GetTensorType<uint32_t>(),
                                 DataTypeImpl::GetTensorType<uint64_t>(),
                                 DataTypeImpl::GetTensorType<float>(),
                                 DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum);

ONNX_OPERATOR_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    14,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(1)  // 'axis' needs to be on CPU
        .TypeConstraint("T", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>(),
                                 DataTypeImpl::GetTensorType<uint32_t>(),
                                 DataTypeImpl::GetTensorType<uint64_t>(),
                                 DataTypeImpl::GetTensorType<float>(),
                                 DataTypeImpl::GetTensorType<double>(),
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) // MLFloat16 is added in opset 14
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum);

Status CumSum::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);                       // input tensor
  auto rank = static_cast<int64_t>(input->Shape().NumDimensions());  // the rank of the input/output
  if (rank == 0)
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Cannot apply CumSum operator on a scalar");

  const Tensor* axis_tensor = ctx->Input<Tensor>(1);  // axis input tensor

  int64_t axis = 0;
  ORT_THROW_IF_ERROR(cumsum_op::GetAxis(axis_tensor, rank, axis));

  TensorShape output_shape(input->Shape());
  auto& output = *ctx->Output(0, output_shape);  // output tensor

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  const auto& input_dims = input->Shape().GetDims();

  int64_t current_dim = rank - 1;
  int64_t input_stride_along_axis = 1;

  // axis (and by extension current_dim) can never be negative as this is validated much before
  // so no need to add the extra check to make sure current_dim is within bounds of the vector size
  while (current_dim > axis) {
    input_stride_along_axis *= input_dims[current_dim--];
  }

  fast_divmod fast_divmod_input_dim_along_axis(static_cast<int>(input_dims[axis]));
  fast_divmod fast_divmod_input_stride_along_axis(static_cast<int>(input_stride_along_axis));

  if (input->IsDataType<float>()) {
    CumSumImpl(Stream(), reinterpret_cast<const typename ToCudaType<float>::MappedType*>(input->Data<float>()),
               fast_divmod_input_dim_along_axis,
               fast_divmod_input_stride_along_axis,
               reinterpret_cast<typename ToCudaType<float>::MappedType*>(output.MutableData<float>()),
               output_shape.Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<double>()) {
    CumSumImpl(Stream(), reinterpret_cast<const typename ToCudaType<double>::MappedType*>(input->Data<double>()),
               fast_divmod_input_dim_along_axis,
               fast_divmod_input_stride_along_axis,
               reinterpret_cast<typename ToCudaType<double>::MappedType*>(output.MutableData<double>()),
               output_shape.Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<int32_t>()) {
    CumSumImpl(Stream(), reinterpret_cast<const typename ToCudaType<int32_t>::MappedType*>(input->Data<int32_t>()),
               fast_divmod_input_dim_along_axis,
               fast_divmod_input_stride_along_axis,
               reinterpret_cast<typename ToCudaType<int32_t>::MappedType*>(output.MutableData<int32_t>()),
               output_shape.Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<int64_t>()) {
    CumSumImpl(Stream(), reinterpret_cast<const typename ToCudaType<int64_t>::MappedType*>(input->Data<int64_t>()),
               fast_divmod_input_dim_along_axis,
               fast_divmod_input_stride_along_axis,
               reinterpret_cast<typename ToCudaType<int64_t>::MappedType*>(output.MutableData<int64_t>()),
               output_shape.Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<uint32_t>()) {
    CumSumImpl(Stream(), reinterpret_cast<const typename ToCudaType<uint32_t>::MappedType*>(input->Data<uint32_t>()),
               fast_divmod_input_dim_along_axis,
               fast_divmod_input_stride_along_axis,
               reinterpret_cast<typename ToCudaType<uint32_t>::MappedType*>(output.MutableData<uint32_t>()),
               output_shape.Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<uint64_t>()) {
    CumSumImpl(Stream(), reinterpret_cast<const typename ToCudaType<uint64_t>::MappedType*>(input->Data<uint64_t>()),
               fast_divmod_input_dim_along_axis,
               fast_divmod_input_stride_along_axis,
               reinterpret_cast<typename ToCudaType<uint64_t>::MappedType*>(output.MutableData<uint64_t>()),
               output_shape.Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<MLFloat16>()) {
    CumSumImpl(Stream(), reinterpret_cast<const typename ToCudaType<MLFloat16>::MappedType*>(input->Data<MLFloat16>()),
               fast_divmod_input_dim_along_axis,
               fast_divmod_input_stride_along_axis,
               reinterpret_cast<typename ToCudaType<MLFloat16>::MappedType*>(output.MutableData<MLFloat16>()),
               output_shape.Size(),
               exclusive_,
               reverse_);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported input data type to the CumSum op: ",
                           input->DataType());
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
