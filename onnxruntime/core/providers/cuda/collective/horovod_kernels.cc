// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/collective/horovod_kernels.h"
#include "core/providers/cpu/collective/horovod_kernels.h"
#include <thread>

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    HorovodAllReduce,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
      .Alias(0, 0)
      .OutputMemoryType<OrtMemTypeCPUOutput>(1)
      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    HorovodAllReduce);

ONNX_OPERATOR_KERNEL_EX(
    HorovodBarrier,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
      .Alias(0, 0)
      .SetDefaultInputsMemoryType(OrtMemTypeCPUInput)
      .InputMemoryType<OrtMemTypeDefault>(0)
      .OutputMemoryType<OrtMemTypeDefault>(0)
      .OutputMemoryType<OrtMemTypeCPUOutput>(1)
      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    HorovodBarrier);

Status HorovodBarrier::ComputeInternal(OpKernelContext* context) const {
  std::vector<const Tensor*> ready_inputs;
  for (int i = 1; i < context->InputCount(); i++) {
    const Tensor* ready_input = context->Input<Tensor>(i);
    if (*ready_input->template Data<bool>() == false) {
      ready_inputs.push_back(ready_input);
    }
  }

  while (ready_inputs.size() > 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    ready_inputs.erase(std::remove_if(ready_inputs.begin(), ready_inputs.end(), [](const Tensor* ready_input) {
      const bool* ready = ready_input->template Data<bool>();
      return *ready;
    }), ready_inputs.end());
  }

  const Tensor& input_tensor = *context->Input<Tensor>(0);
  const void* input_data = input_tensor.DataRaw();
  const auto& input_shape = input_tensor.Shape();

  Tensor& output_tensor = *context->Output(0, input_shape);
  void* output_data = output_tensor.MutableDataRaw();

  if (output_data != input_data) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, input_tensor.SizeInBytes(), cudaMemcpyDeviceToDevice));
  }

  Tensor& ready_tensor = *context->Output(1, {});
  bool* ready_data = ready_tensor.template MutableData<bool>();
  *ready_data = true;
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
