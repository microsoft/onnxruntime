// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_HOROVOD

#include "recv.h"

namespace onnxruntime {
namespace hip {

#define REGISTER_RECV_KERNEL_TYPED(T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      Recv,                                                              \
      kMSDomain,                                                         \
      1,                                                                 \
      T,                                                                 \
      kHipExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .InputMemoryType<OrtMemTypeCPUInput>(0)  /* CPU variable */    \
          .OutputMemoryType<OrtMemTypeCPUOutput>(0)  /* CPU variable */  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("TBool", DataTypeImpl::GetTensorType<bool>()), \
      Recv<T>);

REGISTER_RECV_KERNEL_TYPED(MLFloat16)
REGISTER_RECV_KERNEL_TYPED(float)
REGISTER_RECV_KERNEL_TYPED(double)

void HIPRT_CB HostRecv(void* args) {
  CommInfo_t* info = reinterpret_cast<CommInfo_t*>(args);
  int mpi_code  = MPI_Recv(info->buffer, info->size, MPI_CHAR, info->rank, info->tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ORT_ENFORCE(mpi_code == MPI_SUCCESS, "MPI Recv fails.");
}

template <typename T>
Status Recv<T>::ComputeInternal(OpKernelContext* ctx) const {
  // Check if control signal is true.
  const Tensor* input_signal_tensor = ctx->Input<Tensor>(0);
  const bool* input_signal = input_signal_tensor->template Data<bool>();
  ORT_ENFORCE(*input_signal, "Input control signal of Recv must be true before executing the node.");

  // Start the communication.
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  ORT_ENFORCE(world_rank == dst_, "Rank ", world_rank, " is reciving data but the expected reciving rank is ", dst_, ".");

  hipStream_t commStream;
  hipStreamCreate(&commStream);

  // Receive rank.
  size_t tensor_shape_size = 0;
  CommInfo_t info_shape_size{&tensor_shape_size, sizeof(size_t), static_cast<int>(src_), static_cast<int>(tag_)};
  hipLaunchHostFunc(commStream, HostRecv, &info_shape_size);
  hipStreamSynchronize(commStream);
  
  // Receive shape.
  std::vector<int64_t> tensor_shape(tensor_shape_size);
  CommInfo_t info_shape{tensor_shape.data(), tensor_shape.size() * sizeof(int64_t), static_cast<int>(src_), static_cast<int>(tag_)};
  hipLaunchHostFunc(commStream, HostRecv, &info_shape);
  hipStreamSynchronize(commStream);

  Tensor* x_tensor = ctx->Output(1, tensor_shape);
  T* x = x_tensor->MutableData<T>();

  // Receive actual data.
  IAllocatorUniquePtr<T> buffer = AllocateBufferOnCPUPinned<T>(x_tensor->Shape().Size());
  CommInfo_t info_data{buffer.get(), x_tensor->Shape().Size() * sizeof(T), static_cast<int>(src_), static_cast<int>(tag_)};
  hipLaunchHostFunc(commStream, HostRecv, &info_data);
  hipStreamSynchronize(commStream);
  hipStreamDestroy(commStream);

  // TODO: we need to avoid copying data to host after enabling HIP-aware MPI in the build.
  hipMemcpy(x, buffer.get(), sizeof(T) * x_tensor->Shape().Size(), hipMemcpyHostToDevice);

  // Set first output after communication is done.
  Tensor* output_signal_tensor = ctx->Output(0, {});
  bool* output_signal = output_signal_tensor->template MutableData<bool>();
  *output_signal = true;

  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime

#endif