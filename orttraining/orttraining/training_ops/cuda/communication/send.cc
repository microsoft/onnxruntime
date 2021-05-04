// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_USE_NCCL) || defined(USE_MPI)

#include "orttraining/training_ops/cuda/communication/send.h"
#include "orttraining/training_ops/communication_common.h"
#include "orttraining/training_ops/cuda/communication/nccl_service.h"
#include "core/profile/profile.h"
#include "core/profile/context.h"
#include "core/providers/cuda/cuda_check_memory.h"
#include "core/providers/cuda/cuda_common.h"
#include <mpi.h>

#include "orttraining/core/framework/communication/mpi/mpi_context.h"

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

void Send::SendData(
    OpKernelContext* ctx,
    const int dst,
    const int num_tensors,
    size_t aggregated_aligned_tensor_bytes,
    std::vector<size_t> tensor_offsets_in_bytes,
    std::vector<size_t> tensor_sizes_in_bytes) const {
#ifdef ENABLE_NVTX_PROFILE
  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());

  profile::NvtxRangeCreator memcpyRange(
      "Batch-" + tag +
          " SendMemcpy-" + std::to_string(dst),
      profile::Color::Red);
  // Begin of major communication tasks.
  // The previous MPI_Send's are not included because we don't want to
  // count waiting time before setting up the actual communication.
  memcpyRange.Begin();
#endif

#if defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
  IAllocatorUniquePtr<char> buffer = GetScratchBuffer<char>(aggregated_aligned_tensor_bytes);
#else
  IAllocatorUniquePtr<char> buffer = AllocateBufferOnCPUPinned<char>(
      aggregated_aligned_tensor_bytes);
#endif

  for (int i = 0; i < num_tensors; ++i) {
    const Tensor* tensor = ctx->Input<Tensor>(i + 2);
#ifndef NDEBUG
    CheckIfMemoryOnCurrentGpuDevice(tensor->DataRaw());
#endif

#if defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
    CUDA_CALL(cudaMemcpyAsync(buffer.get() + tensor_offsets_in_bytes[i], tensor->DataRaw(),
                         tensor_sizes_in_bytes[i], cudaMemcpyDeviceToDevice, Stream()));
#else
    CUDA_CALL(cudaMemcpyAsync(buffer.get() + tensor_offsets_in_bytes[i], tensor->DataRaw(),
                         tensor_sizes_in_bytes[i], cudaMemcpyDeviceToHost, Stream()));
#endif
  }

#ifdef ENABLE_NVTX_PROFILE
  memcpyRange.End();
#endif

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator sendRange(
      "Batch-" + tag +
          " Send-" + std::to_string(dst),
      profile::Color::Red);
  // Begin of major communication tasks.
  // The previous MPI_Send's are not included because we don't want to
  // count waiting time before setting up the actual communication.
  sendRange.Begin();
#endif

  CommInfo_t info_data{buffer.get(),
                       static_cast<int>(aggregated_aligned_tensor_bytes),
                       dst,
                       static_cast<int>(tag_)};

#if defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
#ifndef NDEBUG
  CheckIfMemoryOnCurrentGpuDevice(info_data.buffer);
#endif

  auto& nccl_service = cuda::NcclService::GetInstance();
  nccl_service.SubmitSendAndWait(info_data.buffer, info_data.size, info_data.rank);
#elif defined(USE_MPI)
  MPI_CHECK(MPI_Send(
      info_data.buffer, info_data.size, MPI_CHAR,
      info_data.rank, info_data.tag, MPI_COMM_WORLD));
#else
  ORT_THROW("Failed to send to rank: ", info_data.rank);
#endif

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
#ifdef USE_MPI
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
#endif
  ORT_ENFORCE(world_rank != dst, "Sending data to rank ", dst, " on the rank ", world_rank, ".");

#ifdef ENABLE_NVTX_PROFILE
  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());

  profile::NvtxRangeCreator preRange(
      "Batch-" + tag +
          " PreSend-" + std::to_string(dst),
      profile::Color::Red);
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
#ifdef USE_MPI
    SendShapeInfo(dst, tag_, num_tensors, aggregated_aligned_tensor_bytes, prefix_tensor_shape_sizes, aggregated_tensor_shapes);
#else
    ORT_THROW("ORT must be built with MPI to send shape info.");
#endif
  }
#ifdef ENABLE_NVTX_PROFILE
  // End of data preparation and shape communication.
  preRange.End();
#endif

  // Send tensors.
  SendData(ctx, dst, num_tensors, aggregated_aligned_tensor_bytes, tensor_offsets_in_bytes, tensor_sizes_in_bytes);

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator postRange(
      "Batch-" + tag +
          " PostSend-" + std::to_string(dst),
      profile::Color::Red);
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