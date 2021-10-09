// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#endif
#pragma once

#include "orttraining/core/framework/communication/mpi/mpi_context.h"

namespace onnxruntime {

typedef struct {
  // Pointer to ONNX tensor's data on CPU.
  // It should be removed once we enable CUDA-aware MPI.
  void* buffer;
  // The size of buffer's content in bytes.
  int size;
  // Dst rank for Send and src rank for Recv.
  int rank;
  // Message's tag.
  int tag;
} CommInfo_t;

// This function returns the next multiple of "alignment" where "alignment" is 256 below.
inline size_t GetAggregatedAlignedAddress(size_t old_addr) {
  constexpr size_t alignment = 256;
  size_t new_addr = (old_addr + alignment - 1) / alignment * alignment;
  return new_addr;
}

// This function extracts shapes and sizes for tensors indexed by
// begin, begin + 1, ..., begin + count - 1.
// tensor_sizes[i]: i-th indexed tensor's size.
// tensor_shapes[i]: i-th indexed tensor's shape.
inline void GetTensorShapesAndSizes(
    const bool is_index_input,                  // Index inputs from the context if true. Otherwise, use outputs.
    const int begin,                            // Index of the first sent/received tensor in the context.
    const int count,                            // Number of sent/received tensors.
    OpKernelContext* ctx,                       // The context.
    std::vector<size_t>& tensor_sizes,          // tensor_sizes[i] is the size of i-th sent/received tensor in byte.
    std::vector<TensorShape>& tensor_shapes) {  // tensor_shapes[i] is the i-th sent/received tensor.

  // Helper function to retrieve input or output tensor.
  auto get_tensor = [&](const int index) -> const Tensor* {
    if (is_index_input) {
      return ctx->Input<Tensor>(begin + index);
    } else {
      return ctx->Output<Tensor>(begin + index);
    }
  };

  // Get tensors and shapes for indexed tensors from context.
  tensor_sizes.resize(count);
  tensor_shapes.resize(count);
  for (int i = 0; i < count; ++i) {
    const Tensor* tensor = get_tensor(i);
    tensor_sizes[i] = tensor->SizeInBytes();
    tensor_shapes[i] = tensor->Shape();
  }
}

// Compute shape-related information from given tensor shapes.
inline void ComputeShapeRelatedInfo(
    // tensor_sizes[i] is the size of i-th sent/received tensor in byte.
    const std::vector<size_t> tensor_sizes,
    // tensor_shapes[i] is the i-th sent/received tensor.
    const std::vector<TensorShape> tensor_shapes,
    // The size in bytes if we concatenate all tensors into one single tensor.
    // It may be larger than the original size due to memory alignment.
    size_t& aggregated_aligned_tensor_bytes,
    // aggregated_tensor_shapes[prefix_tensor_shape_sizes[i]] is the first dimension of the i-th tensor.
    // aggregated_tensor_shapes[prefix_tensor_shape_sizes[i + 1]] is the element after the last dimension of the i-th tensor.
    std::vector<size_t>& prefix_tensor_shape_sizes,
    // This field is the concatenation of all received tensors' shapes.
    // Assume that there are two tensors A and B with rank NA and NB, respectively.
    // aggregated_tensor_shapes = [A_shape[0], A_shape[1], ..., A_shape[NA-1], B_shape[0], B_shape[1], ..., B_shape[NB-1]].
    std::vector<int64_t>& aggregated_tensor_shapes,
    // tensor_offsets_in_bytes[i] is the offset of the starting byte of the i-th tensor in the communicated tensor buffer.
    // That is, i-th tensor's first element is tensor_buffer[tensor_offsets_in_bytes[i]].
    std::vector<size_t>& tensor_offsets_in_bytes) {
  // Initialize outputs.
  aggregated_aligned_tensor_bytes = 0;
  prefix_tensor_shape_sizes.resize(0);
  aggregated_tensor_shapes.resize(0);
  tensor_offsets_in_bytes.resize(0);

  // Compute shape information.
  size_t prefix_tensor_shape_size_sum = 0;
  for (int i = 0; static_cast<size_t>(i) < tensor_shapes.size(); ++i) {
    const auto& shape = tensor_shapes[i];
    prefix_tensor_shape_size_sum += shape.NumDimensions();
    prefix_tensor_shape_sizes.push_back(prefix_tensor_shape_size_sum);
    aggregated_tensor_shapes.insert(aggregated_tensor_shapes.end(),
                                    shape.GetDims().begin(),
                                    shape.GetDims().end());

    // aggregated_aligned_tensor_bytes is the first non-occupied address.
    // Starting form  aggregated_aligned_tensor_bytes, we find the next aligned offset in the
    // tensor buffer to meet alignment requirement.
    aggregated_aligned_tensor_bytes = GetAggregatedAlignedAddress(aggregated_aligned_tensor_bytes);
    tensor_offsets_in_bytes.push_back(aggregated_aligned_tensor_bytes);
    aggregated_aligned_tensor_bytes += tensor_sizes[i];
  }
}

#ifdef USE_MPI

inline void SendShapeInfo(
    const int dst,
    const int64_t tag,      // mpi send tag
    const int num_tensors,  // Number of sent tensors.
    size_t aggregated_aligned_tensor_bytes,
    std::vector<size_t> prefix_tensor_shape_sizes,
    std::vector<int64_t> aggregated_tensor_shapes) {
  const int num_tensors_in_bytes = num_tensors * static_cast<int>(sizeof(size_t));
  ORT_ENFORCE(num_tensors_in_bytes < INT_MAX,
              "Total tensor number larger than MPI size limit");

  CommInfo_t info_shape_sizes{prefix_tensor_shape_sizes.data(),
                              num_tensors_in_bytes,
                              dst,
                              static_cast<int>(tag)};
  ORT_ENFORCE(aggregated_aligned_tensor_bytes < INT_MAX,
              "Aggregated tensor size larger than MPI size limit");

  CommInfo_t info_aggregated_size{&aggregated_aligned_tensor_bytes,
                                  static_cast<int>(sizeof(size_t)),
                                  dst,
                                  static_cast<int>(tag)};

  int total_tensor_dim_in_bytes = static_cast<int>(
                                      aggregated_tensor_shapes.size()) *
                                  static_cast<int>(sizeof(int64_t));
  ORT_ENFORCE(total_tensor_dim_in_bytes < INT_MAX,
              "Total dimensions of tensors larger than MPI size limit");

  CommInfo_t info_shapes{aggregated_tensor_shapes.data(),
                         total_tensor_dim_in_bytes,
                         dst,
                         static_cast<int>(tag)};

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

inline void ReceiveShapeInfo(
    const int src,
    const int64_t tag,  // mpi recv tag
    const int num_tensors,
    size_t& aggregated_aligned_tensor_bytes,
    std::vector<size_t>& prefix_tensor_shape_sizes,
    std::vector<int64_t>& aggregated_tensor_shapes) {
  // Resize vector so that the following .data() returns meaningful pointer.
  prefix_tensor_shape_sizes.resize(num_tensors);
  CommInfo_t info_shape_sizes{prefix_tensor_shape_sizes.data(),
                              num_tensors * static_cast<int>(sizeof(size_t)),
                              src,
                              static_cast<int>(tag)};
  CommInfo_t info_aggregated_size{&aggregated_aligned_tensor_bytes,
                                  static_cast<int>(sizeof(size_t)),
                                  src,
                                  static_cast<int>(tag)};
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
                         static_cast<int>(tag)};
  MPI_CHECK(MPI_Recv(
      info_shapes.buffer, info_shapes.size, MPI_CHAR,
      info_shapes.rank, info_shapes.tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
}
#endif  // USE_MPI

inline void ComputeTensorSizeAndBufferLength(OpKernelContext* context,
                                             std::vector<int>& tensor_element_counts,
                                             std::vector<size_t>& tensor_offsets,
                                             std::vector<size_t>& tensor_sizes,
                                             int64_t& total_buffer_len) {
  size_t size_in_bytes = 0;
  const int num_tensors = context->InputCount();
  for (int i = 0; i < num_tensors; ++i) {
    const Tensor* x_tensor = context->Input<Tensor>(i);
    tensor_offsets.push_back(size_in_bytes);

    size_in_bytes = x_tensor->SizeInBytes();
    total_buffer_len += size_in_bytes;

    tensor_sizes.push_back(size_in_bytes);
    tensor_element_counts.push_back((int)x_tensor->Shape().Size());
  }
}
}  // namespace onnxruntime
