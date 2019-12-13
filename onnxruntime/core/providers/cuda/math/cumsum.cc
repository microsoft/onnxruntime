// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cumsum.h"
#include "cumsum_impl.h"
#include "core/providers/cpu/math/cumsum.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    11,
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
                                 DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    CumSum);

Status CumSum::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);                             // input tensor
  const auto rank = static_cast<int64_t>(input->Shape().NumDimensions());  // the rank of the input/output
  const Tensor* axis_tensor = ctx->Input<Tensor>(1);                       // axis input tensor

  int64_t axis;
  auto status = cumsum_op::GetAxis(axis_tensor, rank, axis);
  if (!status.IsOK())
    return status;

  TensorShape output_shape(input->Shape());
  auto& output = *ctx->Output(0, output_shape);  // output tensor

  // output tensor's size is 0, nothing to fill - return
  if (output_shape.Size() == 0)
    return Status::OK();

  const auto& input_dims = input->Shape().GetDims();
  TensorPitches input_strides(input_dims);

  if (input->IsDataType<float>()) {
    CumSumImpl(reinterpret_cast<const typename ToCudaType<float>::MappedType*>(input->Data<float>()),
               input_dims[axis],
               input_strides[axis],
               reinterpret_cast<typename ToCudaType<float>::MappedType*>(output.MutableData<float>()),
               output_shape.Size(),
               input->DataType()->Size(),
               exclusive_,
               reverse_);

  } else if (input->IsDataType<double>()) {
    CumSumImpl(reinterpret_cast<const typename ToCudaType<double>::MappedType*>(input->Data<double>()),
               input_dims[axis],
               input_strides[axis],
               reinterpret_cast<typename ToCudaType<double>::MappedType*>(output.MutableData<double>()),
               output_shape.Size(),
               input->DataType()->Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<int32_t>()) {
    CumSumImpl(reinterpret_cast<const typename ToCudaType<int32_t>::MappedType*>(input->Data<int32_t>()),
               input_dims[axis],
               input_strides[axis],
               reinterpret_cast<typename ToCudaType<int32_t>::MappedType*>(output.MutableData<int32_t>()),
               output_shape.Size(),
               input->DataType()->Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<int64_t>()) {
    CumSumImpl(reinterpret_cast<const typename ToCudaType<int64_t>::MappedType*>(input->Data<int64_t>()),
               input_dims[axis],
               input_strides[axis],
               reinterpret_cast<typename ToCudaType<int64_t>::MappedType*>(output.MutableData<int64_t>()),
               output_shape.Size(),
               input->DataType()->Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<uint32_t>()) {
    CumSumImpl(reinterpret_cast<const typename ToCudaType<uint32_t>::MappedType*>(input->Data<uint32_t>()),
               input_dims[axis],
               input_strides[axis],
               reinterpret_cast<typename ToCudaType<uint32_t>::MappedType*>(output.MutableData<uint32_t>()),
               output_shape.Size(),
               input->DataType()->Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<uint64_t>()) {
    CumSumImpl(reinterpret_cast<const typename ToCudaType<uint64_t>::MappedType*>(input->Data<uint64_t>()),
               input_dims[axis],
               input_strides[axis],
               reinterpret_cast<typename ToCudaType<uint64_t>::MappedType*>(output.MutableData<uint64_t>()),
               output_shape.Size(),
               input->DataType()->Size(),
               exclusive_,
               reverse_);
  } else if (input->IsDataType<MLFloat16>()) {
    CumSumImpl(reinterpret_cast<const typename ToCudaType<MLFloat16>::MappedType*>(input->Data<MLFloat16>()),
               input_dims[axis],
               input_strides[axis],
               reinterpret_cast<typename ToCudaType<MLFloat16>::MappedType*>(output.MutableData<MLFloat16>()),
               output_shape.Size(),
               input->DataType()->Size(),
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
