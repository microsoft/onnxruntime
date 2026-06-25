// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bitcast_op.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

#include <cstring>

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    BitCast,
    26,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                               DataTypeImpl::GetTensorType<double>(),
                               DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<int16_t>(),
                               DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>(),
                               DataTypeImpl::GetTensorType<uint16_t>(),
                               DataTypeImpl::GetTensorType<uint32_t>(),
                               DataTypeImpl::GetTensorType<uint64_t>(),
                               DataTypeImpl::GetTensorType<MLFloat16>(),
                               DataTypeImpl::GetTensorType<BFloat16>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),
                               DataTypeImpl::GetTensorType<double>(),
                               DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<int16_t>(),
                               DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>(),
                               DataTypeImpl::GetTensorType<uint16_t>(),
                               DataTypeImpl::GetTensorType<uint32_t>(),
                               DataTypeImpl::GetTensorType<uint64_t>(),
                               DataTypeImpl::GetTensorType<MLFloat16>(),
                               DataTypeImpl::GetTensorType<BFloat16>()})
        .MayInplace(0, 0),
    BitCast);

BitCast::BitCast(const OpKernelInfo& info) : OpKernel(info) {
  int64_t to;
  Status status = info.GetAttr("to", &to);
  ORT_ENFORCE(status.IsOK(), "Attribute 'to' is not set.");
  to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);
}

Status BitCast::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  ORT_ENFORCE(input != nullptr, "BitCast: input tensor is null.");

  const size_t input_element_size = input->DataType()->Size();

  const auto* output_type = DataTypeImpl::TensorTypeFromONNXEnum(to_);
  ORT_RETURN_IF_NOT(output_type != nullptr,
                    "BitCast: unsupported target type (ONNX enum value: ", to_, ").");
  const size_t output_element_size = output_type->GetElementType()->Size();

  ORT_RETURN_IF_NOT(input_element_size == output_element_size,
                    "BitCast requires input and output types to have the same bit-width. ",
                    "Input element size: ", input_element_size, " bytes, ",
                    "output element size: ", output_element_size, " bytes.");

  Tensor* output = context->Output(0, input->Shape());

  const size_t num_bytes = input->SizeInBytes();
  if (num_bytes > 0) {
    const void* src = input->DataRaw();
    void* dst = output->MutableDataRaw();
    if (src != dst) {
      std::memcpy(dst, src, num_bytes);
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
