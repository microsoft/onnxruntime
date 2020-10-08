// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#pragma once

namespace onnxruntime {
namespace cuda {

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

}  // namespace cuda
}  // namespace onnxruntime
