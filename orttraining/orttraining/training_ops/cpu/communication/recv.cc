// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_MPI)

#include "orttraining/training_ops/cpu/communication/recv.h"

#include <mpi.h>

#include "orttraining/training_ops/communication_common.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"

namespace onnxruntime {
namespace contrib {

void Recv::ReceiveData(
    const int num_tensors,
    std::vector<Tensor*> received_tensors,
    const int src,
    const size_t aggregated_aligned_tensor_bytes,
    std::vector<char>& buffer) const {
  buffer.reserve(aggregated_aligned_tensor_bytes);
  CommInfo_t info_data{buffer.data(),
                       static_cast<int>(aggregated_aligned_tensor_bytes),
                       src,
                       static_cast<int>(tag_)};

  MPI_CHECK(MPI_Recv(
      info_data.buffer, info_data.size, MPI_CHAR,
      info_data.rank, info_data.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

  // Copy tensors from buffer to outputs.
  size_t tensor_offset_in_bytes = 0;
  for (int i = 0; i < num_tensors; ++i) {
    Tensor* tensor = received_tensors[i];

    // Find the next aligned offset in the tensor buffer to meet alignment requirement
    tensor_offset_in_bytes = GetAggregatedAlignedAddress(tensor_offset_in_bytes);

    // Copy data out from buffer.
    assert(tensor_offset_in_bytes + tensor->SizeInBytes() <= aggregated_aligned_tensor_bytes);
    memcpy(tensor->MutableDataRaw(), buffer.data() + tensor_offset_in_bytes, tensor->SizeInBytes());

    tensor_offset_in_bytes += tensor->SizeInBytes();
  }
  assert(tensor_offset_in_bytes == aggregated_aligned_tensor_bytes);
}

ONNX_OPERATOR_KERNEL_EX(
    Recv,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeDefault, 0)  /* CPU variable */
        .InputMemoryType(OrtMemTypeDefault, 1)  /* CPU variable */
        .OutputMemoryType(OrtMemTypeDefault, 0) /* CPU variable */
        .TypeConstraint("TBool", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("TInt64", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
    Recv);

Status Recv::Compute(OpKernelContext* ctx) const {
  // Check if control signal is true.
  const Tensor* input_signal_tensor = ctx->Input<Tensor>(0);
  const bool* input_signal = input_signal_tensor->template Data<bool>();
  ORT_ENFORCE(*input_signal, "Input control signal of Recv must be true before executing the node.");

  // Extract remote rank
  const Tensor* remote_rank_tensor = ctx->Input<Tensor>(1);
  const int64_t* remote_rank = remote_rank_tensor->template Data<int64_t>();
  const int src = static_cast<int>(*remote_rank);

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
#ifdef USE_MPI
    ReceiveShapeInfo(
        src,
        tag_,
        num_tensors,
        aggregated_aligned_tensor_bytes,
        prefix_tensor_shape_sizes,
        aggregated_tensor_shapes);
#else
    ORT_THROW("ORT must be built with MPI to send shape info.");
#endif
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

  // At this stage, all shape information (either inferred locally or received from the source process)
  // required to receive tensors are ready.
  // Create buffer and receive data.
  std::vector<char> buffer;
  ReceiveData(num_tensors, received_tensors, src, aggregated_aligned_tensor_bytes, buffer);

  // Set first output after communication is done.
  Tensor* output_signal_tensor = ctx->Output(0, {});
  bool* output_signal = output_signal_tensor->template MutableData<bool>();
  *output_signal = true;

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime

#endif