// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_impl.h"
#include "core/providers/cuda/tensor/gather.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1, 10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    11, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

// explicit negative axis support
ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    Gather);

Status Gather::ComputeInternal(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));

  if (p.output_tensor->Shape().Size() == 0) {
    return Status::OK();
  }

  const TensorShape& input_shape = p.input_tensor->Shape();

  const int64_t block_size = input_shape.SizeFromDimension(p.axis + 1);
  size_t N = p.indices_tensor->Shape().Size();
  const int64_t output_block_size = N * block_size;
  const int64_t indices_max = input_shape[p.axis];

  const void* input_data = p.input_tensor->DataRaw();
  void* output_data = p.output_tensor->MutableDataRaw();

  const fast_divmod divmod_block_size(gsl::narrow_cast<int>(block_size));

  const size_t element_size = p.input_tensor->DataType()->Size();

  const int64_t input_block_size = input_shape.SizeFromDimension(p.axis);

  const fast_divmod divmod_output_block_size(gsl::narrow_cast<int>(output_block_size));

  // CUDA Kernel implementation supports element sizes of:
  // int8_t, int16_t, int32_t and int64_t which covers all supported
  // types since there is no computations necessary just data movement
  if (p.indices_tensor->IsDataType<int64_t>()) {
    GatherImpl(
        Stream(),
        input_block_size,
        indices_max,
        divmod_output_block_size,
        divmod_block_size,
        p.indices_tensor->DataRaw(),
        input_data,
        element_size,
        output_data,
        p.output_tensor->Shape().Size());
    return Status::OK();
  }

  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
