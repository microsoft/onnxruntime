// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_NCCL) || defined(USE_HOROVOD)

#include "orttraining/training_ops/cuda/communication/send.h"
#include "orttraining/training_ops/cuda/communication/common.h"
#include "core/profile/profile.h"
#include "core/providers/cuda/cuda_common.h"
#include <limits>
#include <mpi.h>

#include "orttraining/core/framework/mpi_setup.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Send,
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
    Send);

void Send::SendShapeInfo(
    const int dst,
    const int num_tensors,  // Number of sent tensors.
    size_t aggregated_aligned_tensor_bytes,
    std::vector<size_t> prefix_tensor_shape_sizes,
    std::vector<int64_t> aggregated_tensor_shapes) const {
  const int num_tensors_in_bytes = num_tensors * static_cast<int>(sizeof(size_t));
  ORT_ENFORCE(num_tensors_in_bytes < INT_MAX,
              "Total tensor number larger than MPI size limit");

  CommInfo_t info_shape_sizes{prefix_tensor_shape_sizes.data(),
                              num_tensors_in_bytes,
                              dst,
                              static_cast<int>(tag_)};
  ORT_ENFORCE(aggregated_aligned_tensor_bytes < INT_MAX,
              "Aggregated tensor size larger than MPI size limit");

  CommInfo_t info_aggregated_size{&aggregated_aligned_tensor_bytes,
                                  static_cast<int>(sizeof(size_t)),
                                  dst,
                                  static_cast<int>(tag_)};

  int total_tensor_dim_in_bytes = static_cast<int>(
                                      aggregated_tensor_shapes.size()) *
                                  static_cast<int>(sizeof(int64_t));
  ORT_ENFORCE(total_tensor_dim_in_bytes < INT_MAX,
              "Total dimensions of tensors larger than MPI size limit");

  CommInfo_t info_shapes{aggregated_tensor_shapes.data(),
                         total_tensor_dim_in_bytes,
                         dst,
                         static_cast<int>(tag_)};

  // Directly use CPU to wait MPI_Send. We cannot use GPU callback because
  // MPI_Send may block the entire GPU until it returns.
  MPI_CHECK(MPI_Send(
      info_shape_sizes.buffer, info_shape_sizes.size, MPI_CHAR,
      info_shape_sizes.rank, info_shape_sizes.tag, MPI_COMM_WORLD));

  MPI_CHECK(MPI_Send(
      info_aggregated_size.buffer, info_aggregated_size.size, MPI_CHAR,
      info_aggregated_size.rank, info_aggregated_size.tag, MPI_COMM_WORLD));

  MPI_CHECK(MPI_Send(
      info_shapes.buffer, info_shapes.size, MPI_CHAR,
      info_shapes.rank, info_shapes.tag, MPI_COMM_WORLD));
}

void Send::SendData(
    OpKernelContext* ctx,
    const int dst,
    const int num_tensors,
    size_t aggregated_aligned_tensor_bytes,
    std::vector<size_t> tensor_offsets_in_bytes,
    std::vector<size_t> tensor_sizes_in_bytes) const {
#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator memcpyRange(
      "SendMemcpy-" + std::to_string(dst), profile::Color::Red);
  // Begin of major communication tasks.
  // The previous MPI_Send's are not included because we don't want to
  // count waiting time before setting up the actual communication.
  memcpyRange.Begin();
#endif

  IAllocatorUniquePtr<char> buffer = AllocateBufferOnCPUPinned<char>(
      aggregated_aligned_tensor_bytes);

  for (int i = 0; i < num_tensors; ++i) {
    const Tensor* tensor = ctx->Input<Tensor>(i + 2);
    CUDA_CALL(cudaMemcpy(buffer.get() + tensor_offsets_in_bytes[i], tensor->DataRaw(),
                         tensor_sizes_in_bytes[i], cudaMemcpyDeviceToHost));
  }

#ifdef ENABLE_NVTX_PROFILE
  memcpyRange.End();
#endif

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator sendRange(
      "Send-" + std::to_string(dst), profile::Color::Red);
  // Begin of major communication tasks.
  // The previous MPI_Send's are not included because we don't want to
  // count waiting time before setting up the actual communication.
  sendRange.Begin();
#endif

  CommInfo_t info_data{buffer.get(),
                       static_cast<int>(aggregated_aligned_tensor_bytes),
                       dst,
                       static_cast<int>(tag_)};

  MPI_CHECK(MPI_Send(
      info_data.buffer, info_data.size, MPI_CHAR,
      info_data.rank, info_data.tag, MPI_COMM_WORLD));

#ifdef ENABLE_NVTX_PROFILE
  // End of major communication tasks.
  sendRange.End();
#endif
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

  // Same-rank communication is not allowed because we currently don't have async Send/Recv.
  int world_rank;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  ORT_ENFORCE(world_rank != dst, "Sending data to rank ", dst, " on the rank ", world_rank, ".");

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator preRange(
      "PreSend-" + std::to_string(dst), profile::Color::Red);
  // Begin of preparation for sending data. This time range includes
  // the time for sending a scalar.
  preRange.Begin();
#endif

  const int num_tensors = static_cast<int>(element_types_.size());
  std::vector<size_t> tensor_sizes_in_bytes;
  std::vector<TensorShape> tensor_shapes;
  GetTensorShapesAndSizes(
      true,
      2,            // First sent tensor's index.
      num_tensors,  // Number of tensors to send
      ctx,
      tensor_sizes_in_bytes,
      tensor_shapes);

  // TODO move the following variables to member variables for extending life-time
  // if we want to make the entire call async
  size_t aggregated_aligned_tensor_bytes = 0;
  std::vector<size_t> prefix_tensor_shape_sizes;
  std::vector<int64_t> aggregated_tensor_shapes;
  // tensor_offsets_in_bytes[i] is the starting byte of the i-th tensor in the send tensor buffer
  std::vector<size_t> tensor_offsets_in_bytes;

  // Extract information needed for copying input tensors into a big buffer.
  // Only that big buffer will be sent.
  ComputeShapeRelatedInfo(
      tensor_sizes_in_bytes,
      tensor_shapes,
      aggregated_aligned_tensor_bytes,
      prefix_tensor_shape_sizes,
      aggregated_tensor_shapes,
      tensor_offsets_in_bytes);

  bool all_shapes_inferred = true;
  for (int i = 0; i < num_tensors; ++i) {
    TensorShape inferred_shape;
    auto shape_inferred = ctx->TryGetInferredInputShape(i + 2, inferred_shape);
    if (!shape_inferred) {
      all_shapes_inferred = false;
      break;
    }
  }

  // Communicate shape information when it cannot be inferred.
  if (!all_shapes_inferred) {
    SendShapeInfo(dst, num_tensors, aggregated_aligned_tensor_bytes, prefix_tensor_shape_sizes, aggregated_tensor_shapes);
  }
#ifdef ENABLE_NVTX_PROFILE
  // End of data preparation and shape communication.
  preRange.End();
#endif

  // Send tensors.
  SendData(ctx, dst, num_tensors, aggregated_aligned_tensor_bytes, tensor_offsets_in_bytes, tensor_sizes_in_bytes);

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator postRange(
      "PostSend-" + std::to_string(dst), profile::Color::Red);
  // Begin of post communication tasks.
  postRange.Begin();
#endif

  // Communication is done, so output control signal can be set to true.
  Tensor* output_signal_tensor = ctx->Output(0, {});
  bool* output_signal = output_signal_tensor->MutableData<bool>();
  *output_signal = true;

#ifdef ENABLE_NVTX_PROFILE
  // End of post communication tasks.
  postRange.End();
#endif

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

#endif
