// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_HOROVOD

#include "send.h"

namespace onnxruntime {
namespace hip {

#define REGISTER_SEND_KERNEL_TYPED(T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      Send,                                                              \
      kMSDomain,                                                         \
      1,                                                                 \
      T,                                                                 \
      kHipExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .InputMemoryType<OrtMemTypeCPUInput>(0)   /* CPU variable */   \
          .OutputMemoryType<OrtMemTypeCPUOutput>(0) /* CPU variable */   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("TBool", DataTypeImpl::GetTensorType<bool>()), \
      Send<T>);

REGISTER_SEND_KERNEL_TYPED(MLFloat16)
REGISTER_SEND_KERNEL_TYPED(float)
REGISTER_SEND_KERNEL_TYPED(double)

void HIPRT_CB HostSend(void* args) {
  CommInfo_t* info = reinterpret_cast<CommInfo_t*>(args);
  int mpi_code  = MPI_Send(info->buffer, info->size, MPI_CHAR, info->rank, info->tag, MPI_COMM_WORLD);
  ORT_ENFORCE(mpi_code == MPI_SUCCESS, "MPI Send fails.");
}

template <typename T>
Status Send<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Check if control signal is true.
  const Tensor* input_signal_tensor = ctx->Input<Tensor>(0);
  const bool* input_signal = input_signal_tensor->template Data<bool>();
  ORT_ENFORCE(*input_signal, "Input control signal of Send must be true before executing the node.");

  // Start the communication.
  const Tensor* x_tensor = ctx->Input<Tensor>(1);
  size_t tensor_shape_size = static_cast<size_t>(x_tensor->Shape().NumDimensions());;
  size_t tensor_size = static_cast<size_t>(x_tensor->Shape().Size());

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  ORT_ENFORCE(world_rank == src_, "Sending data from rank ", world_rank, " but the expected data is on rank ", src_, ".");

  // TODO: we need to avoid copying data to host after enabling HIP-aware MPI in the build.
  IAllocatorUniquePtr<T> buffer = AllocateBufferOnCPUPinned<T>(tensor_size);
  ORT_ENFORCE(hipMemcpy(
    buffer.get(), x_tensor->Data<T>(), tensor_size * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);

  // Prepare communication information.
  std::vector<int64_t> tensor_shape(x_tensor->Shape().GetDims());
  CommInfo_t info_shape_size{&tensor_shape_size, sizeof(size_t), static_cast<int>(dst_), static_cast<int>(tag_)};
  CommInfo_t info_shape{tensor_shape.data(), tensor_shape_size * sizeof(int64_t), static_cast<int>(dst_), static_cast<int>(tag_)};
  CommInfo_t info_data{buffer.get(), tensor_size * sizeof(T), static_cast<int>(dst_), static_cast<int>(tag_)};

  // Enqueue communication functions to GPU stream.
  hipStream_t commStream;
  hipStreamCreate(&commStream);
  hipLaunchHostFunc(commStream, HostSend, &info_shape_size);
  hipLaunchHostFunc(commStream, HostSend, &info_shape);
  hipLaunchHostFunc(commStream, HostSend, &info_data);
  hipStreamSynchronize(commStream);
  hipStreamDestroy(commStream);

  // Communication is done, so output control signal can be set to true.
  Tensor* output_signal_tensor = ctx->Output(0, {});
  bool* output_signal = output_signal_tensor->MutableData<bool>();
  *output_signal = true;

  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime

#endif