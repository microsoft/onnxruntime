// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_elements.h"
#include "gather_elements_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    GatherElements,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElements);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GatherElements,
    kOnnxDomain,
    11, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElements);

Status GatherElements::ComputeInternal(OpKernelContext* context) const {
  // Process input data tensor
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto& input_shape = input_tensor->Shape();
  const auto& input_dims = input_shape.GetDims();
  const int64_t input_rank = static_cast<int64_t>(input_dims.size());

  // Process indices tensor
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto& indices_shape = indices_tensor->Shape();
  const auto& indices_dims = indices_shape.GetDims();
  const int32_t indices_rank = static_cast<int32_t>(indices_dims.size());
  const int64_t indices_size = indices_shape.Size();

  // Handle negative axis if any
  const int64_t axis = HandleNegativeAxis(axis_, input_rank);

  // Validate input shapes and ranks (invoke the static method in the CPU GatherElements kernel that hosts the shared checks)
  auto status = onnxruntime::GatherElements::ValidateInputShapes(input_shape, indices_shape, axis);
  if (!status.IsOK())
    return status;

  // create output tensor
  auto* output_tensor = context->Output(0, indices_shape);

  // if there are no elements in 'indices' - nothing to process
  if (indices_shape.Size() == 0)
    return Status::OK();

  TensorPitches input_strides(input_dims);
  TArray<int64_t> gpu_input_strides(input_strides);

  TArray<fast_divmod> fdm_indices_strides(indices_rank);
  TensorPitches indices_strides(indices_dims);
  for (auto i = 0; i < indices_rank; i++) {
    fdm_indices_strides[i] = fast_divmod(gsl::narrow_cast<int>(indices_strides[i]));
  }

  const size_t element_size = input_tensor->DataType()->Size();
  const size_t index_element_size = indices_tensor->DataType()->Size();

  if (indices_tensor->IsDataType<int32_t>() ||
      indices_tensor->IsDataType<int64_t>()) {
    GatherElementsImpl(
        Stream(),
        input_rank,
        input_tensor->DataRaw(),
        input_dims[axis],
        gpu_input_strides,
        indices_tensor->DataRaw(),
        indices_size,
        fdm_indices_strides,
        axis,
        output_tensor->MutableDataRaw(),
        element_size,
        index_element_size);
    return Status::OK();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GatherElements op: Type of 'indices' must be int32 or int64");
  }
}

}  // namespace cuda
}  // namespace onnxruntime
