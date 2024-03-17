// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_reshape.h"
#include "sharding.h"
#include "nccl_kernels.h"
#include "mpi_include.h"

// ORT system.
#include "core/framework/sharding_spec.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/cuda_check_memory.h"

// std C++.
#include <iostream>

using namespace onnxruntime::distributed;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

// Return true if src_shape[src_begin:src_end] is the same as
// dst_shape[dst_begin:dst_end]. Otherwise, return false.
// TODO: replace std::vector with gsl::span.
bool CompareSubVectors(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& dst_shape,
    size_t src_begin, size_t src_end,
    size_t dst_begin, size_t dst_end) {
  if (src_end - src_begin != dst_end - dst_begin) {
    // Sub-vectors have different lengths.
    return false;
  }
  for (size_t src_index = src_begin, dst_index = dst_begin;
       src_index < src_end && dst_index < dst_end;
       ++src_index, ++dst_index) {
    if (src_shape[src_index] != dst_shape[dst_index]) {
      // Sub-vectors have different elements.
      return false;
    }
  }
  // Sub-vectors have same length and same elements.
  return true;
}

// TODO: replace std::vector with gsl::span.
std::tuple<bool, size_t, size_t, size_t> IsTwoAxisFusion(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& dst_shape) {
  // Return values:
  // - bool: whether two consecutive axes are fused.
  // - size_t: the axis in destination shape formed by fusing two source axes.
  // - size_t: the first axis fused.
  // - size_t: the length of fusion. In two-axis fusion considered by this
  //   function, the length of fusion is always 2.
  const size_t src_rank = src_shape.size();
  const size_t dst_rank = dst_shape.size();
  if (src_rank < 2 || dst_rank < 1) {
    return std::make_tuple(false, -1, -1, -1);
  }
  if (src_rank - 1 != dst_rank) {
    return std::make_tuple(false, -1, -1, -1);
  }
  for (size_t i_src = 0; i_src < src_rank; ++i_src) {
    if (i_src + 1 > src_rank - 1) {
      // We are at src_shape[i] and we need
      // src_shape[i + 1] to fuse.
      // If we are at the last axis, we cannot fuse.
      break;
    }
    const int64_t prod = src_shape[i_src] * src_shape[i_src + 1];

    for (size_t i_dst = 0; i_dst < dst_rank; ++i_dst) {
      // Check if shape[i_src:i_src+2] (i.e., shape[i_src] and shape[i_src+1])
      // for source tensor are fused into shape[i_dst] for destination tensor.
      if (prod != dst_shape[i_dst]) {
        continue;
      }
      // Check if corresponding dimensions before fusion area
      // are the same.
      const bool prefix_shape_match = CompareSubVectors(
          src_shape,
          dst_shape,
          // Represent src_shape[0:i_src].
          0, i_src,
          // Represent dst_shape[0:i_dst].
          0, i_dst);
      const bool suffix_shape_match = CompareSubVectors(
          src_shape,
          dst_shape,
          // Represent src_shape[i_src+2:].
          i_src + 2, src_rank,
          // Represent dst_shape[i_dst+1:].
          i_dst + 1, dst_rank);
      if (prefix_shape_match && suffix_shape_match) {
        return std::make_tuple(
            true, i_dst, i_src, 2);
      }
    }
  }
  return std::make_tuple(false, 0, 0, 0);
}

std::tuple<bool, size_t, size_t, size_t> IsTwoAxisDecomposition(
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& dst_shape) {
  // Return values:
  // - bool: whether one source axis is decomposed into two consecutive destination axes.
  // - size_t: the axis in source shape decomposed into two consecutive destination axes.
  // - size_t: the first axis the source axis decomposed into.
  // - size_t: the number of decomposed axes. It's always 2 in this function.
  return IsTwoAxisFusion(dst_shape, src_shape);
}

std::vector<int64_t> RepeatVector(const std::vector<int64_t>& vec, int64_t repeat) {
  std::vector<int64_t> new_vec;
  for (int64_t i = 0; i < repeat; ++i) {
    new_vec.insert(new_vec.end(), vec.begin(), vec.end());
  }
  return new_vec;
}

DeviceMesh CreateInterleaveDeviceMesh(
    const DeviceMesh& source_mesh, const int64_t repeat) {
  // Given a 1-D device mesh [0, 1] and repeat=2,
  // return 1-D device mesh [0, 1, 0, 1].
  if (source_mesh.device_mesh_shape.size() != 1) {
    throw std::runtime_error("Source mesh shape 1-D.");
  }

  // Mesh to return.
  DeviceMesh new_mesh;

  std::vector<int64_t>& elements = new_mesh.device_mesh_elements;
  for (int64_t i = 0; i < repeat; ++i) {
    elements.insert(
        elements.end(),
        source_mesh.device_mesh_elements.begin(),
        source_mesh.device_mesh_elements.end());
  }

  // source mesh must be 1-D so we only care its 1st dimension.
  new_mesh.device_mesh_shape.push_back(source_mesh.device_mesh_shape[0] * repeat);

  return new_mesh;
}

std::tuple<bool, TensorPartitionSpec> ComputeNativeSpecForTwoAxisFusion(
    const TensorPartitionSpec& src_spec,
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& dst_shape,
    const int64_t fused_axis_in_src,
    const int64_t fusion_axis_in_dst) {
  // TODO(wechi): use device mesh stride to support non-1 stride.
  // Example: S[0]R, shape=[2, 3], device_mesh=[0, 1] -> S[0], shape = [6], device_mesh=[0, 1]
  // Example: RS[0], shape=[2, 3], device_mesh=[0, 1] -> S[0], shape = [6], device_mesh=[0, 1, 0, 1]
  // Example: S[0]RR, shape=[2, 3, 5], device_mesh=[0, 1] -> S[0]R, shape = [2, 15], device_mesh=[0, 1]
  ORT_ENFORCE(src_spec.CountShardingAxes() == 1, "Tensor to be reshaped has too many sharding axes.");
  ORT_ENFORCE(src_spec.device_mesh.device_mesh_shape.size() == 1, "Source device mesh be 1-D.");

  if (src_spec.HasNoShard()) {
    return std::make_tuple(true, TensorPartitionSpec::CreateAllReplica(dst_shape.size(), src_spec.device_mesh));
  } else if (src_spec.HasShard() && src_spec.OnlyShardAxis(fused_axis_in_src)) {
    // Example: S[0]R, shape=[2, 3], device_mesh=[0, 1] -> S[0], shape = [6], device_mesh=[0, 1]
    // Example 1:
    //  - logical input shape: [2, 8]
    //  - logical output shape: [16]
    //  - input sharding spec: S[0]R, device_mesh=[0, 1]
    // 1. Device allocation of the original input tensor:
    //  - Logical tensor.
    //    [[0, 0, 0, 0, 0, 0, 0, 0], (device assignment)
    //     [1, 1, 1, 1, 1, 1, 1, 1]]
    //    [[ 0,  1,  2,  3,  4,  5,  6,  7], (values)
    //     [ 8,  9, 10, 11, 12, 13, 14, 15]]
    //  - Device 0's local tensor (shape: [2, 4]).
    //    [[ 0,  1,  2,  3,  4,  5,  6,  7]]
    //  - Device 1's local tensor (shape: [2, 4]).
    //    [[ 8,  9, 10, 11, 12, 13, 14, 15]]
    // 2. Deduce local output shape:
    //  - In the logical Reshape, the 1st and 2nd logical axes are fused,
    //    so are the corresponding local axes.
    //  - Local output shape: [8] by fusing both axes in shape [2, 4].
    // 3. Run local reshape (reshape from shape [2, 4] to shape [8]):
    //  - Device 0's local output tensor.
    //    [ 0,  1,  2,  3,  4,  5,  6,  7]
    //  - Device 1's local output tensor.
    //    [ 8,  9, 10, 11, 12, 13, 14, 15]
    // 4. Determine native output sharding spec from local output tensors.
    //  - Logical output tensor:
    //    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    //  - Device assignment by comparing local tensors and logical output tensor:
    //    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1,  1,  1,  1,  1,  1,  1]
    //  - S[0] with device_mesh = [0, 1] = input device mesh.
    // 5. Native output sharding spec:
    //  - S[0] with device_mesh [0, 1]
    //
    // Example 2:
    //  - logical input shape: [8, 2]
    //  - logical output shape: [16]
    //  - input sharding spec: S[0]R, device_mesh=[0, 1]
    // 1. Device allocation of the original input tensor:
    //  - Logical tensor.
    //    [[0, 0], (device assignment)
    //     [0, 0],
    //     [0, 0],
    //     [0, 0],
    //     [1, 1],
    //     [1, 1],
    //     [1, 1],
    //     [1, 1]]
    //    [[ 0,  1], (values)
    //     [ 2,  3],
    //     [ 4,  5],
    //     [ 6,  7],
    //     [ 8,  9],
    //     [10, 11],
    //     [12, 13],
    //     [14, 15]]
    //  - Device 0's local tensor (shape: [4, 2]).
    //    [[ 0,  1],
    //     [ 2,  3],
    //     [ 4,  5],
    //     [ 6,  7]]
    //  - Device 1's local tensor (shape: [4, 2]).
    //    [[ 8,  9],
    //     [10, 11],
    //     [12, 13],
    //     [14, 15]]
    // 2. Deduce local output shape:
    //  - In the logical Reshape, the 1st and 2nd logical axes are fused,
    //    so are the corresponding local axes.
    //  - Local output shape: [8] by fusing both axes in shape [4, 2].
    // 3. Run local reshape (reshape from shape [4, 2] to shape [8]):
    //  - Device 0's local output tensor.
    //    [ 0,  1,  2,  3,  4,  5,  6,  7]
    //  - Device 1's local output tensor.
    //    [ 8,  9, 10, 11, 12, 13, 14, 15]
    // 4. Determine native output sharding spec from local output tensors.
    //  - Logical output tensor:
    //    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    //  - Device assignment by comparing local tensors and logical output tensor:
    //    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1,  1,  1,  1,  1,  1,  1]
    //  - S[0] with device_mesh = [0, 1] = input device mesh.
    // 5. Native output sharding spec:
    //  - S[0] with device_mesh [0, 1]
    //
    // Example 3:
    //  - logical input shape: [8, 2]
    //  - logical output shape: [16]
    //  - input sharding spec: S[0]R, device_mesh=[0, 1, 0, 1]
    // 1. Device allocation of the original input tensor:
    //  - Logical tensor.
    //    [[0, 0], (device assignment)
    //     [0, 0],
    //     [1, 1],
    //     [1, 1],
    //     [0, 0],
    //     [0, 0],
    //     [1, 1],
    //     [1, 1]]
    //    [[ 0,  1], (values)
    //     [ 2,  3],
    //     [ 4,  5],
    //     [ 6,  7],
    //     [ 8,  9],
    //     [10, 11],
    //     [12, 13],
    //     [14, 15]]
    //  - Device 0's local tensor (shape: [4, 2]).
    //    [[ 0,  1],
    //     [ 2,  3],
    //     [ 8,  9],
    //     [10, 11]]
    //  - Device 1's local tensor (shape: [4, 2]).
    //    [[ 4,  5],
    //     [ 6,  7],
    //     [12, 13],
    //     [14, 15]]
    // 2. Deduce local output shape:
    //  - In the logical Reshape, the 1st and 2nd logical axes are fused,
    //    so are the corresponding local axes.
    //  - Local output shape: [8] by fusing both axes in shape [4, 2].
    // 3. Run local reshape (reshape from shape [4, 2] to shape [8]):
    //  - Device 0's local output tensor.
    //    [ 0,  1,  2,  3,  8,  9, 10, 11]
    //  - Device 1's local output tensor.
    //    [ 4,  5,  6,  7, 12, 13, 14, 15]
    // 4. Determine native output sharding spec from local output tensors.
    //  - Logical output tensor:
    //    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    //  - Device assignment by comparing local tensors and logical output tensor:
    //    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0,  0,  0,  1,  1,  1,  1]
    //  - S[0] with device_mesh = [0, 1] = input device mesh.
    // 5. Native output sharding spec:
    //  - S[0] with device_mesh [0, 1, 0, 1]

    // Reuse original device mesh but shard the fusion axis in output tensor.
    auto dst_spec = TensorPartitionSpec::CreateOneTensorAxisOneDeviceMeshAxisSharding(
        dst_shape.size(), src_spec.device_mesh, fusion_axis_in_dst, /* 1-D mesh */ 0);
    return std::make_tuple(true, dst_spec);
  } else if (src_spec.HasShard() && src_spec.OnlyShardAxis(fused_axis_in_src + 1)) {
    // Example 1 of determining native output sharding spec:
    //  - logical input shape: [3, 4]
    //  - logical output shape: [12]
    //  - input sharding spec: RS[0], device_mesh=[0, 1, 0, 1]
    // 1. Device allocation of the original input tensor:
    //  - Logical tensor.
    //    [[0, 1, 0, 1], (device assignment)
    //     [0, 1, 0, 1],
    //     [0, 1, 0, 1]]
    //    [[0, 1, 2, 3], (values)
    //     [4, 5, 6, 7],
    //     [8, 9, 10, 11]],
    //  - Device 0's local tensor.
    //    [[0, 0],
    //     [0, 0],
    //     [0, 0]]
    //    [[0, 2],
    //     [4, 6],
    //     [8, 10]],
    //  - Device 1's local tensor.
    //    [[1, 1],
    //     [1, 1],
    //     [1, 1]]
    //    [[1, 3],
    //     [5, 7],
    //     [9, 11]],
    // 2. Deduce local output shape:
    //  - In the logical Reshape, the 1st and 2nd logical axes are fused,
    //    so are the corresponding local axes.
    //  - Local output shape: [6] by fusing both axes in shape [3, 2].
    // 3. Run local reshape (reshape from [3, 2] to [6]):
    //  - Device 0's local output tensor.
    //    [0, 0, 0, 0, 0, 0]
    //    [0, 2, 4, 6, 8, 10]
    //  - Device 1's local output tensor.
    //    [1, 1, 1, 1, 1, 1]
    //    [1, 3, 5, 7, 9, 11]
    // 4. Determine native output sharding spec by comparing local output tensors and logical tensor.
    //  - Logical output tensor:
    //    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    //  - S[0] with device_mesh = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] = [0, 1, 0, 1] * (first fused dimension).
    // 5. Native output sharding spec:
    //  - S[0] with device_mesh = [0, 1, 0, 1] * (first fused dimension) = [0, 1, 0, 1] * 3
    //
    // Example 2 of determining native output sharding spec:
    //  - logical input shape: [3, 8]
    //  - logical output shape: [24]
    //  - input sharding spec: RS[0], device_mesh=[0, 1, 0, 1]
    // 1. Device allocation of the original input tensor:
    //  - Logical tensor.
    //    [[0, 0, 1, 1, 0, 0, 1, 1], (device assignment)
    //     [0, 0, 1, 1, 0, 0, 1, 1],
    //     [0, 0, 1, 1, 0, 0, 1, 1]]
    //    [[ 0,  1,  2,  3,  4,  5,  6,  7], (values)
    //     [ 8,  9, 10, 11, 12, 13, 14, 15],
    //     [16, 17, 18, 19, 20, 21, 22, 23]]
    //  - Device 0's local tensor (shape: [3, 4]).
    //    [[0, 0, 0, 0],
    //     [0, 0, 0, 0],
    //     [0, 0, 0, 0]]
    //    [[ 0,  1,  4,  5],
    //     [ 8,  9, 12, 13],
    //     [16, 17, 20, 21]]
    //  - Device 1's local tensor (shape: [3, 4]).
    //    [[1, 1, 1, 1],
    //     [1, 1, 1, 1],
    //     [1, 1, 1, 1]]
    //    [[ 2,  3,  6,  7],
    //     [10, 11, 14, 15],
    //     [18, 19, 22, 23]]
    // 2. Deduce local output shape:
    //  - In the logical Reshape, the 1st and 2nd logical axes are fused,
    //    so are the corresponding local axes.
    //  - Local output shape: [12] by fusing both axes in shape [3, 4].
    // 3. Run local reshape (reshape from [3, 4] to [12]):
    //  - Device 0's local output tensor .
    //    [0, 1, 4, 5,  8,  9, 12, 13, 16, 17, 20, 21]
    //  - Device 1's local output tensor .
    //    [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23]
    // 4. Determine native output sharding spec from local output tensors.
    //  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    //  - [0, 0, 1, 1, 0, 0, 1, 1, 0, 0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1]
    //  - S[0] with device_mesh = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] = .
    // 5. Native output sharding spec:
    //  - S[0] with device_mesh = [0, 1, 0, 1] * (first fused dimension) = [0, 1, 0, 1] * 3
    //
    // Example 3:
    //  - logical input shape: [2, 8]
    //  - logical output shape: [16]
    //  - input sharding spec: RS[0], device_mesh=[0, 1, 0, 1]
    // 1. Device allocation of the original input tensor:
    //  - Logical tensor.
    //    [[0, 0, 1, 1, 0, 0, 1, 1], (device assignment)
    //     [0, 0, 1, 1, 0, 0, 1, 1]]
    //    [[ 0,  1,  2,  3,  4,  5,  6,  7], (values)
    //     [ 8,  9, 10, 11, 12, 13, 14, 15]]
    //  - Device 0's local tensor (shape: [2, 4]).
    //    [[0, 0, 0, 0],
    //     [0, 0, 0, 0]]
    //    [[ 0,  1,  4,  5],
    //     [ 8,  9, 12, 13]]
    //  - Device 1's local tensor (shape: [2, 4]).
    //    [[1, 1, 1, 1],
    //     [1, 1, 1, 1]]
    //    [[ 2,  3,  6,  7],
    //     [10, 11, 14, 15]]
    // 2. Deduce local output shape:
    //  - In the logical Reshape, the 1st and 2nd logical axes are fused,
    //    so are the corresponding local axes.
    //  - Local output shape: [8] by fusing both axes in shape [2, 4].
    // 3. Run local reshape (reshape from [2, 4] to [8]):
    //  - Device 0's local output tensor .
    //    [ 0,  1,  4,  5,  8,  9, 12, 13]
    //  - Device 1's local output tensor .
    //    [ 2,  3,  6,  7, 10, 11, 14, 15]
    // 4. Determine native output sharding spec from local output tensors.
    //  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    //  - [0, 0, 1, 1, 0, 0, 1, 1, 0, 0,  1,  1,  0,  0,  1,  1]
    //  - S[0] with device_mesh = [0, 1, 0, 1, 0, 1, 0, 1] = [0, 1, 0, 1] * (first fused dimension).
    // 5. Native output sharding spec:
    //  - S[0] with device_mesh = [0, 1, 0, 1] * (first fused dimension) = [0, 1, 0, 1] * 2
    //
    // Example 4:
    //  - logical input shape: [2, 8]
    //  - logical output shape: [16]
    //  - input sharding spec: RS[0], device_mesh=[0, 1]
    // 1. Device allocation of the original input tensor:
    //  - Logical tensor.
    //    [[0, 0, 0, 0, 1, 1, 1, 1], (device assignment)
    //     [0, 0, 0, 0, 1, 1, 1, 1]]
    //    [[ 0,  1,  2,  3,  4,  5,  6,  7], (values)
    //     [ 8,  9, 10, 11, 12, 13, 14, 15]]
    //  - Device 0's local tensor (shape: [2, 4]).
    //    [[0, 0, 0, 0],
    //     [0, 0, 0, 0]]
    //    [[ 0,  1,  2,  3],
    //     [ 8,  9, 10, 11]]
    //  - Device 1's local tensor (shape: [2, 4]).
    //    [[1, 1, 1, 1],
    //     [1, 1, 1, 1]]
    //    [[ 4,  5,  6,  7],
    //     [12, 13, 14, 15]]
    // 2. Deduce local output shape:
    //  - In the logical Reshape, the 1st and 2nd logical axes are fused,
    //    so are the corresponding local axes.
    //  - Local output shape: [8] by fusing both axes in shape [2, 4].
    // 3. Run local reshape (reshape from [2, 4] to [8]):
    //  - Device 0's local output tensor .
    //    [ 0,  1,  2,  3,  8,  9, 10, 11]
    //  - Device 1's local output tensor .
    //    [ 4,  5,  6,  7, 12, 13, 14, 15]
    // 4. Determine native output sharding spec from local output tensors.
    //  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    //  - [0, 0, 0, 0, 1, 1, 1, 1, 0, 0,  0,  0,  1,  1,  1,  1]
    //  - S[0] with device_mesh = [0, 1, 0, 1] = [0, 1] * (first fused dimension).
    // 5. Native output sharding spec:
    //  - S[0] with device_mesh = [0, 1] * (first fused dimension) = [0, 1] * 2 = [0, 1, 0, 1]

    // The output device mesh is the repeats of the original device.
    // Let's use Python syntax. If the original device mesh is [0, 1, 0, 1], and
    // the first fused dimension is 3, then the output device mesh is [0, 1, 0, 1] * 3.
    auto dst_device_mesh = DeviceMesh::Create1D(
        src_spec.device_mesh.device_mesh_elements,
        src_shape[fused_axis_in_src]);
    // Sharding happens in the fusion axis with the new device mesh.
    auto dst_spec = TensorPartitionSpec::CreateOneTensorAxisOneDeviceMeshAxisSharding(
        dst_shape.size(), dst_device_mesh, fusion_axis_in_dst, /* 1-D mesh */ 0);
    return std::make_tuple(true, dst_spec);
  } else if (src_spec.HasShard() && (src_spec.GetPartitionAxis() < fused_axis_in_src || src_spec.GetPartitionAxis() > fused_axis_in_src + 1)) {
    // It's two-axis fusion but the fused axes is not sharded.
    // Example: S[0]RR, shape=[2, 3, 5], device_mesh=[0, 1] -> S[0]R, shape = [2, 15], device_mesh=[0, 1]
    auto dst_spec = TensorPartitionSpec::CreateByDropAxes(
        src_spec, {fused_axis_in_src + 1});
    return std::make_tuple(true, dst_spec);
  } else {
    return std::make_tuple(false, TensorPartitionSpec());
  }
}

// Arguments:
//  - device_elements: a vector of device IDs.
//    It should only contain unique device IDs or
//    repeats of a list of unique device IDs. Otherwise,
//    (0, 0) is returned.
// Returns:
//  - count per device ID (all device IDs should have the same count)
//  - number of unique device IDs
// Examples:
//  - [0, 1] -> (2, 1)
//  - [0, 1, 2, 0, 1, 2] -> (2, 3)
std::tuple<int64_t, int64_t> ComputeRepeatAndRepeatStride(
    const std::vector<int64_t>& device_elements) {
  int64_t first_device_id = device_elements.at(0);
  int64_t first_device_id_count = 0;
  for (size_t i = 0; i < device_elements.size(); ++i) {
    if (device_elements.at(i) == first_device_id) {
      ++first_device_id_count;
    }
  }
  size_t repeat_stride = device_elements.size() / first_device_id_count;

  // Check if the device mesh pattern is supported.
  // Supported examples: [0, 1, 2] and [0, 1, 0, 1, 0, 1].
  // Unsupported examples: [0, 1, 2, 1, 2, 0] and [0, 1, 2, 0].
  for (size_t repeat = 0; repeat < first_device_id_count; ++repeat) {
    for (size_t device_id = 0; device_id < repeat_stride; ++device_id) {
      ORT_ENFORCE(
          device_elements.at(repeat * repeat_stride + device_id) == device_elements.at(device_id),
          "Unsupported device mesh pattern.");
    }
  }

  // If device_mesh=[0, 1, 2, 0, 1, 2], returns (2, 3), which means
  //  - each device repeats twice for "2" in (2, 3).
  //  - there are 3 unique devices for "3" in (2, 3).
  return std::make_tuple(first_device_id_count, repeat_stride);
}

std::tuple<bool, TensorPartitionSpec> ComputeNativeSpecForTwoAxisDecomposition(
    const TensorPartitionSpec& src_spec,
    const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& dst_shape,
    const int64_t decomposed_axis_in_src,
    const int64_t decomposition_axis_in_dst) {
  // TODO(wechi): use device mesh stride to support non-1 stride.
  // Example: S[0], shape=[8], device_mesh=[0, 1] -> S[0]R
  // Example: S[0], shape=[8], device_mesh=[0, 1] -> RS[0]
  // Example: S[0], shape=[8], device_mesh=[0, 1, 0, 1] -> S[0]R
  // Example: S[0], shape=[8], device_mesh=[0, 1, 0, 1] -> RS[0]
  // Example: RS[0]R, shape=[8], device_mesh=[0, 1] -> RS[0]RR
  // Example: RS[0]R, shape=[8], device_mesh=[0, 1] -> RRS[0]R
  if (src_spec.CountShardingAxes() != 1) {
    throw std::runtime_error("Too many sharding axes.");
  }
  if (src_spec.device_mesh.device_mesh_shape.size() != 1) {
    throw std::runtime_error("Source device mesh be 1-D.");
  }

  if (src_spec.HasNoShard()) {
    return std::make_tuple(true, TensorPartitionSpec::CreateAllReplica(dst_shape.size(), src_spec.device_mesh));
  } else if (src_spec.OnlyShardAxis(decomposed_axis_in_src)) {
    const int64_t device_stride = src_shape[decomposed_axis_in_src] / src_spec.device_mesh.device_mesh_shape[0];
    if (device_stride >= dst_shape[decomposition_axis_in_dst + 1] && device_stride % dst_shape[decomposition_axis_in_dst + 1] == 0) {
      // Since 2nd decomposition dimension is a factor of device stride,
      // Sharding happens at 1st decomposition axis in dst.
      // device_stride = 10
      // S[0], shape=[20], device=[0, 1] -> S[0]R, shape=[2, 10], device=[0, 1]
      //
      // device_stride = 8
      // S[0], shape=[16], device=[0, 1] -> RS[0], shape=[1, 16], device=[0, 1]
      //
      // device_stride = 8
      // S[0], shape=[16], device=[0, 1] -> S[0]R, shape=[4, 4], device=[0, 1]
      std::vector<AxisPartitionSpec> dst_axis_specs;
      for (size_t src_axis = 0; src_axis < src_shape.size(); ++src_axis) {
        if (src_axis != decomposed_axis_in_src) {
          // Sharding spec is copied if the axis is not decomposed.
          // E.g, shape [5, 6] -> Reshape -> shape [5, 3, 2]
          // The spec for "5" is copied.
          dst_axis_specs.push_back(AxisPartitionSpec::CreateCopy(src_spec.GetAxisSpec(src_axis)));
        } else if (dst_shape[decomposition_axis_in_dst] == 1) {
          // S[0] -> RS[0]
          // E.g., shape [5] -> Reshape -> shape [1, 5]
          // The spec for "5" is copied and "1" is replica.
          // This reshape only adds a dummy new axis without affecting
          // the underlying sharding status.
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
        } else {
          // S[0] -> S[0]R
          // E.g., shape [5] -> Reshape -> shape [5, 1]
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
        }
      }
      // Now, we know sharding happens at decomposed_axis_in_src axis in destination tensor.
      // - effective_device_stride along decomposed_axis_in_src: device_stride / dst_shape[decomposed_axis_in_src + 1]
      // - The original device patterns repeats: dst_shape[decomposed_axis_in_src] / effective_device_stride times.
      const int64_t effective_device_stride = device_stride / dst_shape[decomposed_axis_in_src + 1];
      // How many times a device ID changes along decomposed_axis_in_src axis in destination tensor.
      const int64_t number_of_device_changes = dst_shape[decomposed_axis_in_src] / effective_device_stride;
      if ((size_t)number_of_device_changes != src_spec.device_mesh.device_mesh_elements.size()) {
        throw std::runtime_error("Not supported. Resharding is required.");
      }
      auto dst_device_mesh = CreateInterleaveDeviceMesh(
          src_spec.device_mesh, 1);
      return std::make_tuple(true, TensorPartitionSpec::Create(dst_axis_specs, dst_device_mesh));
    } else if (dst_shape[decomposition_axis_in_dst + 1] > device_stride && dst_shape[decomposition_axis_in_dst + 1] % device_stride == 0) {
      // Since 2nd decomposition dimension is a multiple of device stride,
      // sharding happens at 2nd decomposition axis in dst.
      // stride = 4
      // S[0], shape=[8], device=[0, 1] -> S[0]R, shape=[4, 2], device=[0, 1]
      //
      // stride = 8
      // S[0], shape=[32], device=[0, 1, 0, 1] -> RS[0], shape=[2, 16], device=[0, 1]
      std::vector<AxisPartitionSpec> dst_axis_specs;
      // How many times a device ID appears.
      // E.g., [0, 1, 0, 1, 0, 1] -> 3
      int64_t repeats = 0;
      // Number of unique devices.
      // E.g., [0, 1, 0, 1, 0, 1] -> 2
      int64_t repeat_stride = 0;
      DeviceMesh dst_device_mesh;
      std::tie(repeats, repeat_stride) = ComputeRepeatAndRepeatStride(src_spec.device_mesh.device_mesh_elements);
      for (size_t src_axis = 0; src_axis < src_shape.size(); ++src_axis) {
        if (src_axis != decomposed_axis_in_src) {
          dst_axis_specs.push_back(AxisPartitionSpec::CreateCopy(src_spec.GetAxisSpec(src_axis)));
        } else if (dst_shape[decomposition_axis_in_dst] == 1) {
          // S[0] -> RS[0]
          // E.g., shape [5] -> Reshape -> shape [1, 5]
          // In this case "1" is added as a dummy axis without affecting
          // the underlying sharding status, so we just copy the spec
          // for input "5" to output "5".
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          dst_device_mesh = src_spec.device_mesh;
        } else if (dst_shape[decomposition_axis_in_dst + 1] == 1) {
          // S[0] -> S[0]R
          // E.g., shape [5] -> Reshape -> shape [5, 1]
          // In this case "1" is added as a dummy axis without affecting
          // the underlying sharding status, so we just copy the spec
          // for input "5" to output "5".
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_device_mesh = src_spec.device_mesh;
        } else if (repeats == 1 && dst_shape[decomposition_axis_in_dst + 1] == device_stride * repeat_stride) {
          // S[0] -> RS[0]
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          dst_device_mesh = src_spec.device_mesh;
        } else if (repeats != 1 && dst_shape[decomposition_axis_in_dst + 1] % (device_stride * repeat_stride) == 0) {
          // S[0] -> RS[0]
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          // Extract [0, 1] from [0, 1, 0, 1].
          std::vector<int64_t> unique_device_mesh_elements(
              src_spec.device_mesh.device_mesh_elements.begin(),
              src_spec.device_mesh.device_mesh_elements.begin() + repeat_stride);
          // Compute new repeats.
          // Example of repeats change from 2 to 1:
          //  [16]-shape tensor                      [2, 8]-shape tensor
          //  with 1-D device mesh     -> Reshape -> with 1-D device mesh
          //  [0, 1, 0, 1] (repeats=2)               [0, 1] (repeats=1)
          const int64_t new_repeat = dst_shape[decomposition_axis_in_dst + 1] / (device_stride * repeat_stride);
          dst_device_mesh.device_mesh_shape.push_back(repeat_stride);
          dst_device_mesh.device_mesh_elements = RepeatVector(unique_device_mesh_elements, new_repeat);
        } else {
          throw std::runtime_error("Not supported. Resharding is required.");
        }
      }
      return std::make_tuple(true, TensorPartitionSpec::Create(dst_axis_specs, dst_device_mesh));
    } else {
      // Not supported. Resharding is required.
      return std::make_tuple(false, TensorPartitionSpec());
    }
  } else {
    // Source tensor is sharded on non-decomposed axis.
    std::vector<AxisPartitionSpec> dst_axis_specs;
    for (size_t src_axis = 0; src_axis < src_shape.size(); ++src_axis) {
      if (src_axis != decomposed_axis_in_src) {
        dst_axis_specs.push_back(AxisPartitionSpec::CreateCopy(src_spec.GetAxisSpec(src_axis)));
      } else {
        // R -> RR
        dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
        dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
      }
    }

    return std::make_tuple(true, TensorPartitionSpec::Create(dst_axis_specs, src_spec.device_mesh));
  }
}

// Arguments:
//  global_data_shape: logical shape of Reshape's 1st input.
//  global_shape_span: logical content of Reshape's 2nd input.
// Returns:
//  logical shape of Reshape's output.
inline TensorShape InferDistributedReshapeLogicalOutputShape(
    const TensorShape& global_data_shape,
    const gsl::span<const int64_t>& global_shape_span,
    const int64_t allow_zero) {
  return onnxruntime::cuda::InferReshapeOutputShape(
      global_data_shape,
      global_shape_span,
      allow_zero);
}

template <typename T>
DistributedReshape<T>::DistributedReshape(const OpKernelInfo& info) : DistributedKernel(info) {
  allow_zero_ = info.GetAttrOrDefault("allowzero", static_cast<int64_t>(0));
}

template <typename T>
Status DistributedReshape<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(context != nullptr);
  auto data_tensor = context->Input<Tensor>(0);
  auto shape_tensor = context->Input<Tensor>(1);
  const auto& data_sharding_spec = input_shard_specs_.at(0);
  const auto& shape_sharding_spec = input_shard_specs_.at(1);
  const auto& output_sharding_spec = output_shard_specs_.at(0);

  if (data_sharding_spec.HasNoShard() && shape_sharding_spec.HasNoShard() && output_sharding_spec.HasNoShard()) {
    // Case: all inputs and outputs are not sharded.
    const auto target_shape = onnxruntime::cuda::InferReshapeOutputShape(
        data_tensor,
        shape_tensor,
        allow_zero_);

    auto output_tensor = context->Output(0, target_shape);

    // Copy data from input from output.
    return FuncReshape(
        this,
        context,
        data_tensor,
        shape_tensor,
        allow_zero_,
        output_tensor);
  } else {
    ORT_ENFORCE(shape_sharding_spec.HasNoShard(),
                "Shape tensor should not be sharded because it will trigger communication. "
                "If sharding shape is needed, please request this feature on Github.");
    ORT_ENFORCE(shape_tensor->Shape().NumDimensions() == 1, "Shape must be a 1-D tensor.");
    const auto original_data_shape = ComputeOriginShape(data_tensor->Shape(), data_sharding_spec);
    const auto original_output_shape = InferDistributedReshapeLogicalOutputShape(
        original_data_shape,
        shape_tensor->template DataAsSpan<int64_t>(),
        allow_zero_);

    // TODO: remove below code after replacing std::vector with TensorShape in other APIs.
    std::vector<int64_t> src_shape(original_data_shape.GetDims().begin(), original_data_shape.GetDims().end());
    std::vector<int64_t> dst_shape(original_output_shape.GetDims().begin(), original_output_shape.GetDims().end());

    // Case: Two axis fusion
    bool is_two_axis_fusion = false;
    size_t two_axis_fusion_axis_in_dst = 0;
    size_t two_axis_fusion_first_fused_axis_in_src = 0;
    size_t two_axis_fusion_fused_axis_count = 0;
    std::tie(
        is_two_axis_fusion,
        two_axis_fusion_axis_in_dst,
        two_axis_fusion_first_fused_axis_in_src,
        two_axis_fusion_fused_axis_count) = IsTwoAxisFusion(src_shape, dst_shape);

    if (is_two_axis_fusion) {
      bool is_supported = false;
      TensorPartitionSpec native_dst_spec;
      std::tie(is_supported, native_dst_spec) = ComputeNativeSpecForTwoAxisFusion(
          data_sharding_spec,
          src_shape,
          dst_shape,
          two_axis_fusion_first_fused_axis_in_src,
          two_axis_fusion_axis_in_dst);

      if (is_supported && native_dst_spec == output_sharding_spec) {
        // In this case, we can apply Reshape with local shape on local tensor without resharding.
        // Those local output tensors match the output tensors defined by
        // sharding the logical tensor following the native sharding spec.
        TensorShape local_shape = ComputeShardShape(original_output_shape, native_dst_spec);
        auto output_tensor = context->Output(0, local_shape);
        return FuncReshape(
            this,
            context,
            data_tensor,
            shape_tensor,
            allow_zero_,
            output_tensor);
      } else {
        // TODO: Reshape outputs from `native_dst_spec` to `output_sharding_spec`.
        return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Encounter unsupported reshape pattern.");
      }
    }

    // Case: Two axis decomposition
    bool is_two_axis_decomposition = false;
    size_t two_axis_decomposition_decomposed_axis_in_src = 0;
    size_t two_axis_decomposition_first_factor_axis_in_dst = 0;
    size_t two_axis_decomposition_factor_axis_count_in_dst = 0;
    std::tie(
        is_two_axis_decomposition,
        two_axis_decomposition_decomposed_axis_in_src,
        two_axis_decomposition_first_factor_axis_in_dst,
        two_axis_decomposition_factor_axis_count_in_dst) = IsTwoAxisDecomposition(src_shape, dst_shape);

    if (is_two_axis_decomposition) {
      bool is_supported = false;
      TensorPartitionSpec native_dst_spec;
      std::tie(is_supported, native_dst_spec) = ComputeNativeSpecForTwoAxisDecomposition(
          data_sharding_spec,
          src_shape,
          dst_shape,
          two_axis_decomposition_decomposed_axis_in_src,
          two_axis_decomposition_first_factor_axis_in_dst);

      if (is_supported && native_dst_spec == output_sharding_spec) {
        // In this case, we can apply Reshape with local shape on local tensor without resharding.
        // Those local output tensors match the output tensors defined by
        // sharding the logical tensor following the native sharding spec.
        TensorShape local_shape = ComputeShardShape(original_output_shape, native_dst_spec);
        auto output_tensor = context->Output(0, local_shape);
        return FuncReshape(
            this,
            context,
            data_tensor,
            shape_tensor,
            allow_zero_,
            output_tensor);
      } else {
        // TODO: Reshape outputs from `native_dst_spec` to `output_sharding_spec`.
        return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Encounter unsupported reshape pattern.");
      }
    }
  }

  return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Encounter unsupported reshape pattern.");
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReshape,
    kMSDomain,
    1,
    int64_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReshape<int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReshape,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReshape<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReshape,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReshape<MLFloat16>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
