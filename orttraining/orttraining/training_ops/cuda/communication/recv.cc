// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_NCCL) || defined(USE_HOROVOD)

#include "orttraining/training_ops/cuda/communication/recv.h"
#include "orttraining/training_ops/cuda/communication/common.h"
#include "core/profile/profile.h"
#include <mpi.h>

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Recv,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)   /* CPU variable */
        .InputMemoryType<OrtMemTypeCPUInput>(1)   /* CPU variable */
        .OutputMemoryType<OrtMemTypeCPUOutput>(0) /* CPU variable */
        .TypeConstraint("TBool", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
    Recv);

Status Recv::ComputeInternal(OpKernelContext* ctx) const {
  // Check if control signal is true.
  const Tensor* input_signal_tensor = ctx->Input<Tensor>(0);
  const bool* input_signal = input_signal_tensor->template Data<bool>();
  ORT_ENFORCE(*input_signal, "Input control signal of Recv must be true before executing the node.");

  // Extract remote rank
  const Tensor* remote_rank_tensor = ctx->Input<Tensor>(1);
  const int64_t* remote_rank = remote_rank_tensor->template Data<int64_t>();
  const int src = static_cast<int>(*remote_rank);

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator preRange(
    "PreRecv-" + std::to_string(src), profile::Color::Green);
  // Begin of preparation for receiving data.
  preRange.Begin();
#endif

  // Create buffers
  const int tensor_num = static_cast<int>(element_types_.size());
  // TODO move the following variables to member variables for extending life-time
  // if we want to make the entire call async
  std::vector<size_t> prefix_tensor_shape_sizes(tensor_num);
  std::vector<int64_t> aggregated_tensor_shapes;
  size_t aggregated_aligned_tensor_bytes = 0;

  // Start communication
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  ORT_ENFORCE(world_rank != src, "Receive data from rank ", src, " on the rank ", world_rank, ".");

  // Receive shape sizes and aggregated size
  CommInfo_t info_shape_sizes{prefix_tensor_shape_sizes.data(),
                              tensor_num * static_cast<int>(sizeof(size_t)),
                              src,
                              static_cast<int>(tag_)};
  CommInfo_t info_aggregated_size{&aggregated_aligned_tensor_bytes,
                                  static_cast<int>(sizeof(size_t)),
                                  src,
                                  static_cast<int>(tag_)};


  int mpi_code = 0;

  // Directly use CPU to wait MPI_Recv. We cannot use GPU callback because
  // MPI_Recv may block the entire GPU until it returns.
  mpi_code = MPI_Recv(
    info_shape_sizes.buffer, info_shape_sizes.size, MPI_CHAR,
    info_shape_sizes.rank, info_shape_sizes.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ORT_ENFORCE(mpi_code == MPI_SUCCESS, "MPI Recv fails.");

#ifdef ENABLE_NVTX_PROFILE
  // This range object includes the first MPI_Recv which receives a scalar.  
  // It means we count the MPI's initialization in pre-recv stage.
  preRange.End();
#endif

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator recvRange(
    "Recv-" + std::to_string(src), profile::Color::Green);
  // Begin of major communication tasks.
  // The first MPI_Recv is not included because we don't want to
  // count waiting time before setting up the actual communication.
  recvRange.Begin();
#endif

  mpi_code = MPI_Recv(
    info_aggregated_size.buffer, info_aggregated_size.size, MPI_CHAR,
    info_aggregated_size.rank, info_aggregated_size.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ORT_ENFORCE(mpi_code == MPI_SUCCESS, "MPI Recv fails.");

  // Prepare receive shapes and data buffer
  aggregated_tensor_shapes.resize(prefix_tensor_shape_sizes[tensor_num - 1]);
  IAllocatorUniquePtr<char> buffer =
      AllocateBufferOnCPUPinned<char>(static_cast<size_t>(aggregated_aligned_tensor_bytes));
  CommInfo_t info_shapes{aggregated_tensor_shapes.data(),
                         static_cast<int>(aggregated_tensor_shapes.size()) * static_cast<int>(sizeof(int64_t)),
                         src,
                         static_cast<int>(tag_)};
  CommInfo_t info_data{buffer.get(),
                       static_cast<int>(aggregated_aligned_tensor_bytes),
                       src,
                       static_cast<int>(tag_)};

  // Directly use CPU to wait MPI_Recv. We cannot use GPU callback because
  // MPI_Recv may block the entire GPU until it returns.
  mpi_code = MPI_Recv(
    info_shapes.buffer, info_shapes.size, MPI_CHAR,
    info_shapes.rank, info_shapes.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ORT_ENFORCE(mpi_code == MPI_SUCCESS, "MPI Recv fails.");
  mpi_code = MPI_Recv(
    info_data.buffer, info_data.size, MPI_CHAR,
    info_data.rank, info_data.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  ORT_ENFORCE(mpi_code == MPI_SUCCESS, "MPI Recv fails.");

#ifdef ENABLE_NVTX_PROFILE
  // End of actual communication.
  recvRange.End();
#endif

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator postRange(
    "PostRecv-" + std::to_string(src), profile::Color::Green);
  postRange.Begin();
#endif

  // Create Tensors
  size_t begin = 0;
  size_t tensor_offset_in_bytes = 0;
  for (int i = 0; i < tensor_num; ++i) {
    std::vector<int64_t> tensor_shape(aggregated_tensor_shapes.begin() + begin,
                                      aggregated_tensor_shapes.begin() + prefix_tensor_shape_sizes[i]);
    begin = prefix_tensor_shape_sizes[i];

    Tensor* x_tensor = ctx->Output(i + 1, tensor_shape);
    // Find the next aligned offset in the tensor buffer to meet alignment requirement
    tensor_offset_in_bytes = GetAggregatedAlignedAddress(tensor_offset_in_bytes);

    // Keep the sync copy in the previous design
    // TODO they can be moved to async call after global stream becoming accessible
    ORT_ENFORCE(cudaMemcpy(x_tensor->MutableDataRaw(), buffer.get() + tensor_offset_in_bytes,
                           x_tensor->SizeInBytes(), cudaMemcpyHostToDevice) == cudaSuccess);
    tensor_offset_in_bytes += x_tensor->SizeInBytes();
  }

  // Set first output after communication is done.
  Tensor* output_signal_tensor = ctx->Output(0, {});
  bool* output_signal = output_signal_tensor->template MutableData<bool>();
  *output_signal = true;

#ifdef ENABLE_NVTX_PROFILE
  postRange.End();
#endif

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

#endif
