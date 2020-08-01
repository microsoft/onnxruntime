// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <thread>

#include "horovod_kernels.h"
#include "ready_event.h"

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

Status HorovodAllReduce::ComputeInternal(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(ConvertStatus(horovod::common::CheckInitialized()));

  auto device_id = context->GetDeviceId();
  auto allocator = Info().GetAllocator(device_id, OrtMemTypeDefault);
 
  const Tensor* input_tensor = context->Input<Tensor>(0);

  std::shared_ptr<ORTTensor> hvd_input;
  void* data = nullptr;
  //bugbug
  if (adasum_type_ == training::AdasumReductionType::CpuReduction) {
    allocator = Info().GetAllocator(0, OrtMemTypeCPU);
    //IAllocatorUniquePtr<uint8_t> buffer = AllocateBufferOnCPUPinned<uint8_t>(
    //    static_cast<size_t>(input_tensor->SizeInBytes()));
    data = malloc(sizeof(uint8_t) * static_cast<size_t>(input_tensor->SizeInBytes()));

    ORT_ENFORCE(cudaMemcpy(data, input_tensor->DataRaw(),
                           input_tensor->SizeInBytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
    const auto data_type = input_tensor->DataType();
    const auto shape = input_tensor->Shape();
    OrtMemoryInfo cpuinfo{onnxruntime::CPU, OrtDeviceAllocator};
    auto cpu_tensor = onnxruntime::make_unique<Tensor>(data_type, shape, (void*)(data), cpuinfo);
    //std::unique_ptr<Tensor> cpu_tensor = onnxruntime::make_unique<Tensor>(data_type, shape, allocator);
    //ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(*input_tensor, *cpu_tensor));

    hvd_input = std::make_shared<ORTTensor>(cpu_tensor.release());
    cpu_tensor.reset();
  }
  else {
    hvd_input = std::make_shared<ORTTensor>(input_tensor);
  }

  //auto hvd_ready_event = (adasum_type_ == training::AdasumReductionType::CpuReduction) ? nullptr : std::make_shared<ORTReadyEvent>(device_id);
  auto hvd_ready_event = std::make_shared<ORTReadyEvent>(device_id);
  auto hvd_context = std::make_shared<ORTOpContext>(allocator);
  auto hvd_output = std::make_shared<ORTTensor>(context->Output(0, input_tensor->Shape()));
  Tensor* ready_tensor = context->Output(1, {});
  bool* ready = ready_tensor->MutableData<bool>();
  *ready = false;
  if (adasum_type_ == training::AdasumReductionType::CpuReduction) {
    device_id = -1;
  }

  auto output_data = hvd_output->mutable_data();
  size_t copy_size = static_cast<size_t>(hvd_output->size());
  ORT_RETURN_IF_ERROR(
      ConvertStatus(
          EnqueueTensorAllreduce(
              hvd_context, hvd_input, hvd_input, hvd_ready_event,
              unique_name, device_id,
              [ready, output_data, data, copy_size, this, device_id](const horovod::common::Status& /*status*/) {
                // TODO: handle failures during Allreduce
                // https://aiinfra.visualstudio.com/Lotus/_workitems/edit/3936
                // bugbug
                if(adasum_type_ == training::AdasumReductionType::CpuReduction) {
                  ORT_ENFORCE(cudaMemcpy(output_data, data,
                              copy_size, cudaMemcpyHostToDevice) == cudaSuccess);
                  if(data != nullptr) {
                    free(data);
                  }
                }
                *ready = true;
              },
              reduce_op_)));

  return Status::OK();
}

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
                       }),
                       ready_inputs.end());
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
