// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/dequantize_packed.h"

namespace onnxruntime {
namespace cuda {

Status DequantizePacked::ComputeInternal(OpKernelContext* context) const {
  const auto* input = context->Input<Tensor>(0);
  ORT_RETURN_IF_ERROR(attributes_.ValidateInput(input->Shape()));
  auto* output = context->Output(0, TensorShape(attributes_.shape));
  if (attributes_.elements == 0) return Status::OK();
  return LaunchDequantizePacked(Stream(context), input->Data<uint8_t>(), output->MutableDataRaw(),
                                attributes_.format.id, attributes_.output_dtype, attributes_.elements);
}

ONNX_OPERATOR_KERNEL_EX(
    DequantizePacked, kOnnxDomain, 29, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>(),
                              DataTypeImpl::GetTensorType<BFloat16>(),
                              DataTypeImpl::GetTensorType<double>()}),
    DequantizePacked);

}  // namespace cuda
}  // namespace onnxruntime
