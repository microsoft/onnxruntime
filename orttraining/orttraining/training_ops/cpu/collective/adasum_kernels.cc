// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef USE_MPI
#include "orttraining/training_ops/cpu/collective/adasum_kernels.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/training_ops/communication_common.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    AdasumAllReduce,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .VariadicAlias(0, 0)  // outputs and inputs are mapped one to one
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes()),
    AdasumAllReduce);

Status AdasumAllReduce::Compute(OpKernelContext* context) const {
  // Get tensor count
  const int num_tensors = context->InputCount();
  std::vector<int> tensor_element_counts;
  std::vector<size_t> tensor_offsets;
  std::vector<size_t> tensor_sizes;

  int64_t total_recv_buffer_len = 0;
  ComputeTensorSizeAndBufferLength(context,
                                   tensor_element_counts,
                                   tensor_offsets,
                                   tensor_sizes,
                                   total_recv_buffer_len);
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // Allocate fusion buffer.
  auto data_buffer = allocator->Alloc(total_recv_buffer_len);
  BufferUniquePtr data_buffer_ptr(data_buffer, BufferDeleter(allocator));

  for (int i = 0; i < num_tensors; ++i) {
    const Tensor* x_tensor = context->Input<Tensor>(i);
    memcpy((uint8_t*)data_buffer + tensor_offsets[i], x_tensor->DataRaw(),
           tensor_sizes[i]);
  }

  auto recv_buffer = allocator->Alloc(total_recv_buffer_len);
  BufferUniquePtr recv_buffer_ptr(recv_buffer, BufferDeleter(allocator));

  ORT_RETURN_IF_ERROR(adasum_reducer_->DispatchFusedAllreduce(data_buffer, recv_buffer, tensor_element_counts,
                                                              1,                                                                                                        // start level
                                                              training::MPIContext::GetInstance().GetMPIGroup(training::WorkerGroupType::GlobalParallel).communicator,  // communicator
                                                              0,                                                                                                        // tag
                                                              adasum_reducer_->GetReductionComms(),                                                                     // reduction_comms
                                                              context->Input<Tensor>(0)->DataType()));
  for (int i = 0; i < num_tensors; i++) {
    Tensor* y_tensor = context->Output(i, context->Input<Tensor>(i)->Shape());
    memcpy(y_tensor->MutableDataRaw(), (uint8_t*)data_buffer + tensor_offsets[i], tensor_sizes[i]);
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
#endif  // USE_MPI
