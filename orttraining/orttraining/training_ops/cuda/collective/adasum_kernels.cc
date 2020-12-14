// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef USE_MPI
#include "orttraining/training_ops/cuda/collective/adasum_kernels.h"
#include "orttraining/training_ops/communication_common.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/training_ops/cuda/optimizer/common.h"

namespace onnxruntime {
namespace cuda {

//bugbug
Status AdasumAllReduce::ComputeInternal(OpKernelContext* context) const {

  const Tensor* is_gradient_finite = context->Input<Tensor>(0);
  // Get tensor count
  const int num_tensors = context->InputCount() - 1;

  if (is_gradient_finite) {
    const bool is_finite = *(is_gradient_finite->template Data<bool>());
    if (!is_finite) {
      for (int i = 0; i < num_tensors; i++) {
        const Tensor* x_tensor = context->Input<Tensor>(i + 1);
        Tensor* y_tensor = context->Output(i, x_tensor->Shape());
        if (x_tensor->DataRaw() != y_tensor->MutableDataRaw()) {
          CUDA_CALL(cudaMemcpy(y_tensor->MutableDataRaw(), x_tensor->DataRaw(),
                            x_tensor->SizeInBytes(), cudaMemcpyDeviceToDevice));
        }
      }
      std::cout<<"#######not finite, skip doing adasum allreduce"<<std::endl;
      return Status::OK();
    }
  }

  int vhdd_start_level = 1;
  // if (adasum_reduce_algo_ == training::AdasumReductionType::GpuHierarchicalReduction) {
  //   vhdd_start_level = training::DistributedRunContext::GetInstance()
  //                                                      .GroupSize(training::WorkerGroupType::NodeLocalDataParallel);
  // }
  std::vector<int> tensor_element_counts;
  std::vector<size_t> tensor_offsets;
  std::vector<size_t> tensor_sizes;

  int64_t total_recv_buffer_len = 0;

  ComputeTensorSizeAndBufferLength(context,
                                   tensor_element_counts,
                                   tensor_offsets,
                                   tensor_sizes,
                                   total_recv_buffer_len);

  // Allocate temp scratch buffer in cpu space.
  AllocatorPtr allocator;
  allocator = Info().GetAllocator(0, OrtMemTypeCPU);
  auto data_buffer = allocator->Alloc(total_recv_buffer_len);
  BufferUniquePtr data_buffer_ptr(data_buffer, BufferDeleter(allocator));

  for (int i = 0; i < num_tensors; ++i) {
    const Tensor* x_tensor = context->Input<Tensor>(i + 1);
    CUDA_CALL(cudaMemcpy((uint8_t*)data_buffer_ptr.get() + tensor_offsets[i], x_tensor->DataRaw(),
                      tensor_sizes[i], cudaMemcpyDeviceToHost));
  }

  auto recv_buffer = allocator->Alloc(total_recv_buffer_len);
  BufferUniquePtr recv_buffer_ptr(recv_buffer, BufferDeleter(allocator));

  //bugbug
  std::cout<<"##########VHDD start level is: "<<vhdd_start_level<<std::endl;
  if(training::MPIContext::GetInstance().GetLocalRank() == 0 ||
     adasum_reduce_algo_ == training::AdasumReductionType::CpuReduction) {
    std::cout<<"##########adasum gpu kernel DispatchFusedAllreduce"<<std::endl;
    ORT_RETURN_IF_ERROR(adasum_reducer_->DispatchFusedAllreduce((void*)data_buffer, recv_buffer, tensor_element_counts,
                            vhdd_start_level, // start level
                            adasum_reduce_algo_ == training::AdasumReductionType::GpuHierarchicalReduction
                            ? training::MPIContext::GetInstance()
                                                  .GetMPIGroup(training::WorkerGroupType::CrossNodeDataParallel)
                                                  .communicator
                            : training::MPIContext::GetInstance()
                                                  .GetMPIGroup(training::WorkerGroupType::GlobalParallel)
                                                  .communicator, // communicator
                            0, // tag
                            adasum_reducer_->GetReductionComms(), // reduction_comms
                            context->Input<Tensor>(1)->DataType()));
     }
  if(adasum_reduce_algo_ == training::AdasumReductionType::GpuHierarchicalReduction) {
    std::cout<<"##########Broadcast result to ranks"<<std::endl;
    int input_count = total_recv_buffer_len / context->Input<Tensor>(1)->DataType()->Size();
    MPI_CHECK(MPI_Bcast(data_buffer, input_count, training::GetMPIDataType(context->Input<Tensor>(1)->DataType()),
                0, /*local root rank*/
                training::MPIContext::GetInstance().GetMPIGroup(training::WorkerGroupType::NodeLocalDataParallel)
                                                   .communicator));
  }
  for (int i = 0; i < num_tensors; i++) {
    Tensor* y_tensor = context->Output(i, context->Input<Tensor>(i + 1)->Shape());
    CUDA_CALL(cudaMemcpy(y_tensor->MutableDataRaw(), (uint8_t*)data_buffer + tensor_offsets[i],
                      tensor_sizes[i], cudaMemcpyHostToDevice));
  }
  return Status::OK();
}
//bugbug
ONNX_OPERATOR_KERNEL_EX(
    AdasumAllReduce,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)   /* CPU variable */
        .VariadicAlias(1, 0)  // outputs and inputs are mapped one to one with offset by 1
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("TBool", DataTypeImpl::GetTensorType<bool>()),
    AdasumAllReduce);

}  // namespace cuda
}  // namespace onnxruntime
#endif // USE_MPI
