// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_HOROVOD

#include "orttraining/training_ops/hip/communication/send.h"
#include "orttraining/training_ops/hip/communication/common.h"
#include <mpi.h>
#include <limits>

namespace onnxruntime {
namespace hip {

ONNX_OPERATOR_KERNEL_EX(
    Send,
    kMSDomain,
    1,
    kHipExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)   /* CPU variable */
        .InputMemoryType<OrtMemTypeCPUInput>(1)   /* CPU variable */
        .OutputMemoryType<OrtMemTypeCPUOutput>(0) /* CPU variable */
        .TypeConstraint("TBool", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
    Send);

void HIPRT_CB HostSend(void* args) {
  CommInfo_t* info = reinterpret_cast<CommInfo_t*>(args);
  int mpi_code = MPI_Send(info->buffer, info->size, MPI_CHAR, info->rank, info->tag, MPI_COMM_WORLD);
  ORT_ENFORCE(mpi_code == MPI_SUCCESS, "MPI Send fails.");
}

Status Send::ComputeInternal(OpKernelContext* ctx) const {
  // Check if control signal is true.
  const Tensor* input_signal_tensor = ctx->Input<Tensor>(0);
  const bool* input_signal = input_signal_tensor->template Data<bool>();
  ORT_ENFORCE(*input_signal, "Input control signal of Send must be true before executing the node.");

  // Extract remote rank
  const Tensor* remote_rank_tensor = ctx->Input<Tensor>(1);
  const int64_t* remote_rank = remote_rank_tensor->template Data<int64_t>();
  const int dst = static_cast<int>(*remote_rank);

  // Create buffers
  const int tensor_num = static_cast<int>(element_types_.size());
  // TODO move the following variables to member variables for extending life-time
  // if we want to make the entire call async
  std::vector<size_t> prefix_tensor_shape_sizes;
  std::vector<int64_t> aggregated_tensor_shapes;
  size_t aggregated_aligned_tensor_bytes = 0;
  // tensor_offsets_in_bytes[i] is the starting byte of the i-th tensor in the send tensor buffer
  std::vector<size_t> tensor_offsets_in_bytes;
  // tensor_sizes_in_bytes[i] = (# of elements in the i-th tensor) * sizeof(the i-th tensor's element type)
  std::vector<size_t> tensor_sizes_in_bytes;

  // Compute tensor shapes and sizes
  size_t prefix_tensor_shape_size_sum = 0;
  for (int i = 0; i < tensor_num; ++i) {
    const Tensor* x_tensor = ctx->Input<Tensor>(i + 2);
    prefix_tensor_shape_size_sum += x_tensor->Shape().NumDimensions();
    prefix_tensor_shape_sizes.push_back(prefix_tensor_shape_size_sum);
    aggregated_tensor_shapes.insert(aggregated_tensor_shapes.end(),
                                    x_tensor->Shape().GetDims().begin(),
                                    x_tensor->Shape().GetDims().end());

    // Find the next aligned offset in the tensor buffer to meet alignment requirement
    aggregated_aligned_tensor_bytes = GetAggregatedAlignedAddress(aggregated_aligned_tensor_bytes);
    tensor_offsets_in_bytes.push_back(aggregated_aligned_tensor_bytes);
    aggregated_aligned_tensor_bytes += x_tensor->SizeInBytes();
    tensor_sizes_in_bytes.push_back(x_tensor->SizeInBytes());
  }

  // Start communication
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  ORT_ENFORCE(world_rank != dst, "Sending data to rank ", dst, " on the rank ", world_rank, ".");

  IAllocatorUniquePtr<char> buffer = AllocateBufferOnCPUPinned<char>(
      static_cast<size_t>(aggregated_aligned_tensor_bytes));

  // Keep the sync copy in the previous design
  // TODO they can be moved to async call after global stream becoming accessible
  for (int i = 0; i < tensor_num; ++i) {
    const Tensor* x_tensor = ctx->Input<Tensor>(i + 2);
    ORT_ENFORCE(hipMemcpy(buffer.get() + tensor_offsets_in_bytes[i], x_tensor->Data<void>(),
                           tensor_sizes_in_bytes[i], hipMemcpyDeviceToHost) == hipSuccess);
  }

  // Prepare MPI communication info
  int tensor_num_in_bytes = tensor_num * static_cast<int>(sizeof(size_t));
  ORT_ENFORCE(tensor_num_in_bytes < INT_MAX,
              "Total tensor number larger than MPI size limit");
  CommInfo_t info_shape_sizes{prefix_tensor_shape_sizes.data(),
                              tensor_num_in_bytes,
                              dst,
                              static_cast<int>(tag_)};

  ORT_ENFORCE(aggregated_aligned_tensor_bytes < INT_MAX,
              "Aggregated tensor size larger than MPI size limit");
  CommInfo_t info_aggregated_size{&aggregated_aligned_tensor_bytes,
                                  static_cast<int>(sizeof(size_t)),
                                  dst,
                                  static_cast<int>(tag_)};

  int total_tensor_dim_in_bytes = static_cast<int>(aggregated_tensor_shapes.size()) * static_cast<int>(sizeof(int64_t));
  ORT_ENFORCE(total_tensor_dim_in_bytes < INT_MAX,
              "Total dimensions of tensors larger than MPI size limit");
  CommInfo_t info_shapes{aggregated_tensor_shapes.data(),
                         total_tensor_dim_in_bytes,
                         dst,
                         static_cast<int>(tag_)};

  CommInfo_t info_data{buffer.get(),
                       static_cast<int>(aggregated_aligned_tensor_bytes),
                       dst,
                       static_cast<int>(tag_)};

  // Enqueue communication functions to a GPU stream.
  // Keep the local stream in the previous design
  // TODO they can be moved to a new global stream after global streams becoming accessible
  hipStream_t commStream;
  hipStreamCreate(&commStream);

  hipLaunchHostFunc(commStream, HostSend, &info_shape_sizes);
  hipLaunchHostFunc(commStream, HostSend, &info_aggregated_size);
  hipLaunchHostFunc(commStream, HostSend, &info_shapes);
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
