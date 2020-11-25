// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "horovod_kernels.h"
#include "orttraining/models/runner/training_util.h"
#include <thread>

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_KERNEL(
    HorovodAllReduce,
    9,
    KernelDefBuilder()
      .Alias(0, 0)
      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
    HorovodAllReduce);

ONNX_CPU_OPERATOR_KERNEL(
    HorovodBarrier,
    9,
    KernelDefBuilder()
      .Alias(0, 0)
      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
      .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
    HorovodBarrier);

Status HorovodAllReduce::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(ConvertStatus(horovod::common::CheckInitialized()));

  auto device_id = context->GetDeviceId();
  auto allocator = training::TrainingUtil::GetCpuAllocator();

  const Tensor* input_tensor = context->Input<Tensor>(0);
  auto hvd_input = std::make_shared<ORTTensor>(input_tensor);
  auto hvd_context = std::make_shared<ORTOpContext>(allocator);
  auto hvd_output = std::make_shared<ORTTensor>(context->Output(0, input_tensor->Shape()));
  Tensor* ready_tensor = context->Output(1, {});
  bool* ready = ready_tensor->MutableData<bool>();

  *ready = false;

  ORT_RETURN_IF_ERROR (
    ConvertStatus(
      EnqueueTensorAllreduce(
        hvd_context, hvd_input, hvd_output, 0/*ready_event*/,
        unique_name, device_id,
        [ready](const horovod::common::Status& /*status*/) {
          // TODO: handle failures during Allreduce
          // https://aiinfra.visualstudio.com/Lotus/_workitems/edit/3936
          *ready = true;
        }, reduce_op_)
    )
  );

  return Status::OK();
}

Status HorovodBarrier::Compute(OpKernelContext* context) const {
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
    memcpy(output_data, input_data, input_tensor.SizeInBytes());
  }

  Tensor& ready_tensor = *context->Output(1, {});
  bool* ready_data = ready_tensor.template MutableData<bool>();
  *ready_data = true;
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
