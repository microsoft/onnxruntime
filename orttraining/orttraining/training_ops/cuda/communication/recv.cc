// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_NCCL) || defined(USE_HOROVOD)

#include "orttraining/training_ops/cuda/communication/recv.h"
#include "orttraining/training_ops/cuda/communication/common.h"
#include "orttraining/training_ops/cuda/communication/nccl_service.h"
#include "core/profile/profile.h"
#include "core/profile/context.h"
#include "core/providers/cuda/cuda_common.h"
#include <string>
#include <mpi.h>

#include "orttraining/core/framework/mpi_context.h"

namespace onnxruntime {
namespace cuda {

void Recv::ReceiveShapeInfo(
    const int src,
    const int num_tensors,
    size_t& aggregated_aligned_tensor_bytes,
    std::vector<size_t>& prefix_tensor_shape_sizes,
    std::vector<int64_t>& aggregated_tensor_shapes) const {
  // Resize vector so that the following .data() returns meaningful pointer.
  prefix_tensor_shape_sizes.resize(num_tensors);
  CommInfo_t info_shape_sizes{prefix_tensor_shape_sizes.data(),
                              num_tensors * static_cast<int>(sizeof(size_t)),
                              src,
                              static_cast<int>(tag_)};
  CommInfo_t info_aggregated_size{&aggregated_aligned_tensor_bytes,
                                  static_cast<int>(sizeof(size_t)),
                                  src,
                                  static_cast<int>(tag_)};
  // Directly use CPU to wait MPI_Recv. We cannot use GPU callback because
  // MPI_Recv may block the entire GPU until it returns.
  MPI_CHECK(MPI_Recv(
      info_shape_sizes.buffer, info_shape_sizes.size, MPI_CHAR,
      info_shape_sizes.rank, info_shape_sizes.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

  MPI_CHECK(MPI_Recv(
      info_aggregated_size.buffer, info_aggregated_size.size, MPI_CHAR,
      info_aggregated_size.rank, info_aggregated_size.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

  // prefix_tensor_shape_sizes's last element is the number of total dimensions.
  // If a 3-D tensor and a 2-D tensor are sent, its value is 2 + 3 = 5.
  aggregated_tensor_shapes.resize(prefix_tensor_shape_sizes[num_tensors - 1]);
  CommInfo_t info_shapes{aggregated_tensor_shapes.data(),
                         static_cast<int>(aggregated_tensor_shapes.size()) * static_cast<int>(sizeof(int64_t)),
                         src,
                         static_cast<int>(tag_)};
  MPI_CHECK(MPI_Recv(
      info_shapes.buffer, info_shapes.size, MPI_CHAR,
      info_shapes.rank, info_shapes.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
}

void Recv::ReceiveData(
    const int num_tensors,
    std::vector<Tensor*> received_tensors,
    const int src,
    const size_t aggregated_aligned_tensor_bytes,
    IAllocatorUniquePtr<char>& buffer) const {
#ifdef ENABLE_NVTX_PROFILE
  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());

  profile::NvtxRangeCreator recvRange(
      "Batch-" + tag +
      " Recv-" + std::to_string(src), profile::Color::Green);
  // Begin of major communication tasks.
  // The first MPI_Recv is not included because we don't want to
  // count waiting time before setting up the actual communication.
  recvRange.Begin();
#endif

#if defined(USE_NCCL) && defined(USE_NCCL_P2P)
  buffer = GetScratchBuffer<char>(aggregated_aligned_tensor_bytes);
#else
  buffer = AllocateBufferOnCPUPinned<char>(static_cast<size_t>(aggregated_aligned_tensor_bytes));
#endif

  CommInfo_t info_data{buffer.get(),
                       static_cast<int>(aggregated_aligned_tensor_bytes),
                       src,
                       static_cast<int>(tag_)};

  // The following NCCL call is equivalent to the following MPI call. User can
  // uncomment the MPI call to debug.
#if defined(USE_NCCL) && defined(USE_NCCL_P2P)
  auto& nccl_service = cuda::NcclService::GetInstance();
  nccl_service.SubmitRecvAndWait(info_data.buffer, info_data.size, info_data.rank);
#else
  MPI_CHECK(MPI_Recv(
  info_data.buffer, info_data.size, MPI_CHAR,
  info_data.rank, info_data.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
#endif

#ifdef ENABLE_NVTX_PROFILE
  // End of actual communication.
  recvRange.End();
#endif

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator memcpyRange(
      "Batch-" + tag +
      " RecvMemcpy-" + std::to_string(src), profile::Color::Green);
  // Begin of host-to-device memory copy.
  memcpyRange.Begin();
#endif

  // Copy tensors from buffer to outputs.
  size_t tensor_offset_in_bytes = 0;
  for (int i = 0; i < num_tensors; ++i) {
    Tensor* tensor = received_tensors[i];

    // Find the next aligned offset in the tensor buffer to meet alignment requirement
    tensor_offset_in_bytes = GetAggregatedAlignedAddress(tensor_offset_in_bytes);

    // Copy data out from buffer.
#if defined(USE_NCCL) && defined(USE_NCCL_P2P)
    CUDA_CALL(cudaMemcpyAsync(tensor->MutableDataRaw(), buffer.get() + tensor_offset_in_bytes,
                              tensor->SizeInBytes(), cudaMemcpyHostToDevice));
#else
    CUDA_CALL(cudaMemcpyAsync(tensor->MutableDataRaw(), buffer.get() + tensor_offset_in_bytes,
                              tensor->SizeInBytes(), cudaMemcpyDeviceToDevice));
#endif
    tensor_offset_in_bytes += tensor->SizeInBytes();
  }

#if defined(USE_NCCL) && defined(USE_NCCL_P2P)
#else
  AddDeferredReleaseCPUPtr(buffer.release());
#endif

#ifdef ENABLE_NVTX_PROFILE
  // End of host-to-device copy.
  memcpyRange.End();
#endif
}

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
  auto& profile_context = profile::Context::GetInstance();
  const auto tag = profile_context.GetThreadTagOrDefault(std::this_thread::get_id());

  profile::NvtxRangeCreator preRange(
      "Batch-" + tag +
      " PreRecv-" + std::to_string(src), profile::Color::Green);
  // Begin of preparation for receiving data.
  preRange.Begin();
#endif

  // Start communication
  int world_rank;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  ORT_ENFORCE(world_rank != src, "Receive data from rank ", src, " on the rank ", world_rank, ".");

  const int num_tensors = static_cast<int>(element_types_.size());
  std::vector<size_t> tensor_sizes_in_bytes;
  std::vector<TensorShape> tensor_shapes;
  // TODO move the following variables to member variables for extending life-time
  // if we want to make the entire call async
  size_t aggregated_aligned_tensor_bytes = 0;
  std::vector<size_t> prefix_tensor_shape_sizes;
  std::vector<int64_t> aggregated_tensor_shapes;
  // tensor_offsets_in_bytes[i] is the starting byte of the i-th tensor in the send tensor buffer
  std::vector<size_t> tensor_offsets_in_bytes;

  // Whether shapes are statically inferrable.
  bool all_shapes_inferred = true;
  // At iteration i, the i-th received tensor is processed.
  for (int i = 0; i < num_tensors; ++i) {
    TensorShape inferred_shape;
    // The first input is a boolean control signal. We only check actual received tensors.
    auto shape_inferred = ctx->TryGetInferredOutputShape(i + 1, inferred_shape);
    if (!shape_inferred) {
      all_shapes_inferred = false;
      break;
    }
  }

  std::vector<Tensor*> received_tensors(num_tensors);
  if (all_shapes_inferred) {
    // Create outputs before communication because all shapes are inferred.
    for (int i = 0; i < num_tensors; ++i) {
      TensorShape inferred_shape;
      // The first input is a boolean control signal. We only work on actual received tensors before that.
      ORT_ENFORCE(ctx->TryGetInferredOutputShape(i + 1, inferred_shape));
      // If shape is statically inferred, we declare output here and
      // access its shape from operator's context in GetTensorShapesAndSizes(...).
      received_tensors[i] = ctx->Output(i + 1, inferred_shape);
    }

    GetTensorShapesAndSizes(
        false,        // value of "is_index_input". Received tensors are "output"s so this flag is "false".
        1,            // First received tensor's index.
        num_tensors,  // Number of tensors to received.
        ctx,
        tensor_sizes_in_bytes,
        tensor_shapes);

    // Extract information needed for copying input tensors from a big buffer
    // to individual locations.
    // Only that big buffer will be received through MPI.
    ComputeShapeRelatedInfo(
        tensor_sizes_in_bytes,
        tensor_shapes,
        aggregated_aligned_tensor_bytes,
        prefix_tensor_shape_sizes,
        aggregated_tensor_shapes,
        tensor_offsets_in_bytes);
  } else {
    ReceiveShapeInfo(
        src,
        num_tensors,
        aggregated_aligned_tensor_bytes,
        prefix_tensor_shape_sizes,
        aggregated_tensor_shapes);

    // Create output tensors. Unlike the case where we can infer output shapes before communication,
    // we need to create outputs after receiving shapes.
    size_t begin = 0;
    for (int i = 0; i < num_tensors; ++i) {
      std::vector<int64_t> tensor_shape(aggregated_tensor_shapes.begin() + begin,
                                        aggregated_tensor_shapes.begin() + prefix_tensor_shape_sizes[i]);
      received_tensors[i] = ctx->Output(i + 1, tensor_shape);
      // Move the "begin" to the beginning dimension of the next received tensor.
      begin = prefix_tensor_shape_sizes[i];
    }
  }

#ifdef ENABLE_NVTX_PROFILE
  // This range object includes the first MPI_Recv which receives a scalar.
  // It means we count the MPI's initialization in pre-recv stage.
  preRange.End();
#endif

  // At this stage, all shape information (either inferred locally or received from the source process)
  // required to receive tensors are ready.
  // Create buffer and receive data.
  IAllocatorUniquePtr<char> buffer;
  ReceiveData(num_tensors, received_tensors, src, aggregated_aligned_tensor_bytes, buffer);

#ifdef ENABLE_NVTX_PROFILE
  profile::NvtxRangeCreator postRange(
      "Batch-" + tag +
      " PostRecv-" + std::to_string(src), profile::Color::Green);
  postRange.Begin();
#endif

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