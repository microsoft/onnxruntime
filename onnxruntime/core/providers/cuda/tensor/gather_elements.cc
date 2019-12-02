// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_elements.h"
#include "gather_elements_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    GatherElements,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder()
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
  const int64_t input_size = input_shape.Size();

  // Process indices tensor
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto& indices_shape = indices_tensor->Shape();
  const auto& indices_dims = indices_shape.GetDims();
  const int64_t indices_rank = static_cast<int64_t>(indices_dims.size());
  const int64_t indices_size = indices_shape.Size();

  // Handle negative axis if any
  const int64_t axis = static_cast<int64_t>(HandleNegativeAxis(axis_, input_rank));

  // Validate input shapes and ranks (invoke the static method in the CPU GatherElements kernel that hosts the shared checks)
  auto status = onnxruntime::GatherElements::ValidateInputShapes(input_shape, indices_shape, axis);
  if (!status.IsOK())
    return status;

  // create output tensor
  auto* output_tensor = context->Output(0, TensorShape(indices_shape));

  // if there are no elements in 'indices' - nothing to process
  if (indices_shape.Size() == 0)
    return Status::OK();

  TensorPitches input_strides(input_dims);
  CudaAsyncBuffer<int64_t> gpu_input_strides(this, input_strides);

  CudaAsyncBuffer<fast_divmod> fdm_indices_strides(this, indices_rank);
  ORT_ENFORCE(CalculateFdmStrides(fdm_indices_strides.CpuSpan(), indices_dims));

  ORT_RETURN_IF_ERROR(gpu_input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_indices_strides.CopyToGpu());

  size_t element_size = input_tensor->DataType()->Size();

  if (indices_tensor->IsDataType<int32_t>()) {
    const int32_t* indices_data = indices_tensor->template Data<int32_t>();
    GatherElementsImpl<int32_t>(
        input_rank,
        input_tensor->DataRaw(),
        input_size,
        input_dims[axis],
        gpu_input_strides.GpuPtr(),
        indices_data,
        indices_size,
        fdm_indices_strides.GpuPtr(),
        axis,
        output_tensor->MutableDataRaw(),
        element_size);
    return Status::OK();
  } else if (indices_tensor->IsDataType<int64_t>()) {
    const int64_t* indices_data = indices_tensor->template Data<int64_t>();
    GatherElementsImpl<int64_t>(
        input_rank,
        input_tensor->DataRaw(),
        input_size,
        input_dims[axis],
        gpu_input_strides.GpuPtr(),
        indices_data,
        indices_size,
        fdm_indices_strides.GpuPtr(),
        axis,
        output_tensor->MutableDataRaw(),
        element_size);
    return Status::OK();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "GatherElements op: Type of 'indices' must be int32 or int64");
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "String type is not supported yet for the GatherElements op");
}

}  // namespace cuda
}  // namespace onnxruntime
