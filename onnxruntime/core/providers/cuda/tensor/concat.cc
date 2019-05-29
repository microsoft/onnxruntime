// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "concat.h"

namespace onnxruntime {
namespace cuda {
ONNX_OPERATOR_KERNEL_EX(
    Concat,
    kOnnxDomain,
    4,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Concat);

Status Concat::ComputeInternal(OpKernelContext* ctx) const {
  auto input_count = Node().InputArgCount().front();

  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, input_count, p));

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0)
    return Status::OK();

  int64_t output_offset = 0;
  auto element_bytes = p.output_tensor->DataType()->Size();
  for (int input_index = 0; input_index < input_count; input_index++) {
    const auto& prep = p.inputs[input_index];
    // No data in this tensor - so skip it
    if (prep.num_elements == 0)
        continue;
    // Copy the data across. For every 'input_axis_pitch' values copied, we move over by the 'output_axis_pitch'
    CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(
        static_cast<uint8_t*>(p.output_tensor->MutableDataRaw()) + output_offset * element_bytes,
        p.output_axis_pitch * element_bytes,
        prep.tensor->DataRaw(),
        prep.axis_pitch * element_bytes,
        prep.axis_pitch * element_bytes,
        prep.num_elements / prep.axis_pitch,
        cudaMemcpyDeviceToDevice));

    output_offset += prep.axis_pitch;
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
