// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/scatter_nd.h"
#include "core/providers/cuda/tensor/scatter_nd_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(ScatterND,
                                  kOnnxDomain,
                                  11, 12,
                                  kCudaExecutionProvider,
                                  KernelDefBuilder()
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .MayInplace(0, 0),
                                  ScatterND);

ONNX_OPERATOR_KERNEL_EX(ScatterND,
                        kOnnxDomain,
                        13,
                        kCudaExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .MayInplace(0, 0),
                        ScatterND);

Status ScatterND::ComputeInternal(OpKernelContext* context) const {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto* updates_tensor = context->Input<Tensor>(2);

  const auto& input_shape = input_tensor->Shape();
  const auto& indices_shape = indices_tensor->Shape();
  const auto& updates_shape = updates_tensor->Shape();

  // Validate input shapes
  ValidateShapes(input_shape, indices_shape, updates_shape);

  auto* output_tensor = context->Output(0, input_shape);

  const void* input_data = input_tensor->DataRaw();
  void* output_data = output_tensor->MutableDataRaw();

  size_t element_size = input_tensor->DataType()->Size();

  if (input_data != output_data) {
    // TODO: Run benchmarks to determine if a dedicated kernel doing data copy will be faster than invoking cudaMemcpy ?
    cudaMemcpyAsync(output_data, input_data, input_tensor->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream());
  }

  // Bail out early
  if (indices_shape.Size() == 0) {
    return Status::OK();
  }

  auto last_index_dimension = indices_shape[indices_shape.NumDimensions() - 1];

  // We need element counts for each dimension and the input dim value for each dimension
  // for the range [0, last_index_dimension).
  // To avoid multiple GPU data transfers, we combine this into one array and send it through
  TensorPitches input_strides(input_shape);
  std::vector<int64_t> element_counts_and_input_dims(last_index_dimension * 2, 0LL);
  for (int64_t i = 0; i < last_index_dimension; ++i) {
    element_counts_and_input_dims[i] = input_strides[i];
    element_counts_and_input_dims[i + last_index_dimension] = input_shape[i];
  }
  CudaAsyncBuffer<int64_t> element_counts_and_input_dims_gpu(this, element_counts_and_input_dims);
  element_counts_and_input_dims_gpu.CopyToGpu();

  ORT_RETURN_IF_ERROR(ScatterNDImpl(
      Stream(),
      output_data,
      element_size,
      indices_shape.Size() / static_cast<size_t>(last_index_dimension),
      indices_tensor->Data<int64_t>(),  // only int64_t is supported for indices as per the onnx spec
      last_index_dimension,
      element_counts_and_input_dims_gpu.GpuPtr(),
      updates_tensor->DataRaw(),
      input_shape.SizeFromDimension(last_index_dimension)));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
