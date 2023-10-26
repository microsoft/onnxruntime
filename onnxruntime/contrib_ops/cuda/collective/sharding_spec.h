// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"

#include <iostream>
#include <sstream>
#include <vector>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

class DeviceMesh {
 public:
  // [Device Mesh and Tensor Sharding for Tensor Parallel]
  // Device mesh is a tensor of device indices.
  // A tensor can then be partitioned along specific mesh axes.
  //
  // Assume we have 4 GPUs indexed by 0, 1, 2, and 3.
  // Let's consider some examples.
  //  1. 1D device mesh [0, 1, 2, 3]. In this case,
  //     device_mesh_shape is [4] and device_mesh_elements
  //     is [0, 1, 2, 3].
  //     If we want to shard a 2-D tensor along its axis 1, the
  //     corresponding sharding spec is a string "RS[0]".
  //  2. 2D device mesh [[0, 1], [2, 3]]. In this case,
  //     device_mesh_shape is [2, 2] and device_mesh_elements
  //     is [0, 1, 2, 3].
  //     If we want to shard a 2-D tensor's
  //     rows along mesh axis 1 and
  //     columns along mesh axis 0, the
  //     corresponding sharding spec is a string "S[1]S[0]".
  //     If that 2-D tensor's value is np.array([[5, 6], [7, 8]]),
  //     GPU 0/1/2/3 owns 5/7/6/8.  Below is a visualization the sharding
  //     proccess.
  //     - Start with a 2-D device mesh [[0, 1], [2, 3]] and
  //       a 2-D tensor [[5, 6], [7, 8]]
  //       - GPU: [[0, 1], [2, 3]], Tensor: [[5, 6], [7, 8]]
  //     - Split GPU mesh along axis 1 and tensor along
  //       axis 0 for "S[1]" in "S[1]S[0]"
  //       - GPU: [[0], [2]], Tensor: [[5, 6]]
  //         GPU: [[1], [3]], Tensor: [[7, 8]]
  //     - Split GPU mesh along axis 0 and tensor along
  //       axis 1 for "S[0]" in "S[1]S[0]"
  //       - GPU: [[0]], Tensor: [[5]]
  //       - GPU: [[2]], Tensor: [[6]]
  //       - GPU: [[1]], Tensor: [[7]]
  //       - GPU: [[3]], Tensor: [[8]]

  // Actual shape of device mesh represented by `device_mesh_elements`.
  std::vector<int64_t> device_mesh_shape;

  // Flattened device mesh.
  std::vector<int64_t> device_mesh_elements;

  // Helper to debug and generate error message; e.g.,
  // "DeviceMesh{Shape: [2,2,], Elements: [0,1,2,3,]}".
  std::string ToString() const {
    std::ostringstream os;
    os << "DeviceMesh{Shape: [";
    for (const auto& shape : device_mesh_shape)
      os << shape << ",";
    os << "], Elements: [";
    for (const auto& element : device_mesh_elements)
      os << element << ",";
    os << "]}";
    return os.str();
  }

  // Call this in GDB to visualize the mesh.
  void Print() const {
    std::cout << ToString() << std::endl;
  }
};

class AxisPartitionSpec {
  // [Device Mesh and Tensor Sharding for Tensor Parallel]
  // This class is the in-memory representation of
  //  1. if a tensor is sharded or not (aka replica), and
  //  2. which tensor axis is shard by which device mesh axis.
  // Let's consider sharding 2-D tensor along column axis on
  // device mesh [0, 1] as an example.
  // The required sharding spec RS[0] can be represented by
  // - AxisPartitionSpec(Condition::Replica, -1)
  // - AxisPartitionSpec(Condition::Shard, 0)
 public:
  // Status of a tensor axis.
  // A tensor axis can be either sharded or replicated
  // along a device mesh axis.
  enum class Condition { Replica,
                         Shard };

  // This field tells if a tensor axis is sharded or not.
  Condition cond;

  // If a tensor axis is sharded, this field tells which device
  // mesh axis to distribute the shards along.
  // If a tensor axis is not sharded, this field is ignored.
  int device_mesh_axis;

  // A helper to construct a replica spec for a tensor axis.
  static AxisPartitionSpec CreateReplica() {
    return AxisPartitionSpec(Condition::Replica, -1);
  }

  // A helper to construct a sharding spec for a tensor axis.
  // This tensor axis is sharded along `device_mesh_axis` in device mesh.
  static AxisPartitionSpec CreateShard(int device_mesh_axis) {
    return AxisPartitionSpec(Condition::Shard, device_mesh_axis);
  }

  // A normal ctor.
  // TODO(wechi): Consider to hide it and revise the `public` members/functions
  // exposed to the user.
  AxisPartitionSpec(Condition cond_, int device_mesh_axis_) : device_mesh_axis(device_mesh_axis_), cond(cond_) {}

  // Helper to debug and generate error message; e.g.,
  // "RS[0]".
  std::string ToString() const {
    std::ostringstream os;
    os << (cond == Condition::Replica ? "R" : "S");
    if (cond == Condition::Shard) os << "[" << device_mesh_axis << "]";
    return os.str();
  }

  // Call this in GDB to visualize the spec.
  void Print() const {
    std::cout << ToString() << std::endl;
  }
};

// Return true if `axis` is a valid axis index for a tensor of rank `rank`.
// Negative `axis` is allowed (e.g., -1 for the last axis).
void ValidateAxisIndex(const int64_t axis, const int64_t rank);

class TensorPartitionSpec {
  // [Device Mesh and Tensor Sharding for Tensor Parallel]
  // TensorPartitionSpec holds a collection of AxisPartitionSpec and an
  // associated DeviceMesh. It is responsible for determining how a tensor
  // should be partitioned across a device mesh.
  //
  // Example 1: RS[0]
  // In this scenario, `axis_specs` would contain two `AxisPartitionSpec` objects.
  // - The first object is a Replica, denoting that the first axis of the tensor is
  //   not sharded but is instead replicated.
  // - The second object is a Shard along the 0-th axis of the device mesh. It denotes
  //   that the second axis of the tensor is sharded along the first axis of the
  //   device mesh.
  //
  // Example 2: S[0]RR
  // In this scenario, `axis_specs` would contain three `AxisPartitionSpec` objects.
  // - The first object is a Shard along the 0-th axis of the device mesh, indicating
  //   that the first axis of the tensor is sharded along the first axis of the
  //   device mesh.
  // - The second and third objects are Replicas, indicating that the second and third
  //   axes of the tensor are not sharded but are instead replicated.
 public:
  // axis_specs[i]: AxisPartitionSpec for tensor axis i. For a 2-D tensor,
  //                axis_specs[0] is for row axis and axis_specs[1] is for
  //                column axis. axis_specs[i].device_mesh_axis = j means that
  //                tensor axis i is sharded along device mesh axis j.
  std::vector<AxisPartitionSpec> axis_specs;

  // device_mesh: DeviceMesh for sharding the associated tensor.
  // Read [Device Mesh and Tensor Sharding for Tensor Parallel] in DeviceMesh's comment.
  DeviceMesh device_mesh;

  // Replacement of ctor.
  static TensorPartitionSpec Create(
      const std::vector<AxisPartitionSpec>& axis_specs, const DeviceMesh& device_mesh) {
    TensorPartitionSpec spec;
    spec.axis_specs = axis_specs;
    spec.device_mesh = device_mesh;
    return spec;
  }

  // Copy-construct `spec` but with all tensor axes replicated.
  // The new spec have the same number of axis specs and the same device mesh.
  static TensorPartitionSpec CreateAllReplica(
      const TensorPartitionSpec& spec) {
    TensorPartitionSpec new_spec = spec;
    new_spec.axis_specs[spec.GetPartitionAxis()] = AxisPartitionSpec::CreateReplica();
    return new_spec;
  }

  // TODO(wechi): Create a halper to copy-construct a new spec with different sharding axis.
  // static TensorPartitionSpec CreateReshard(
  //     const TensorPartitionSpec& spec, int64_t new_shard_axis) {
  // }

  // Helper to debug and generate error message; e.g.,
  // "TensorPartitionSpec{RS[0], Device Mesh: DeviceMesh{Shape: [4,], Elements: [0,1,2,3,]}}".
  std::string ToString() const {
    std::ostringstream os;
    os << "TensorPartitionSpec{";
    for (const auto& spec : axis_specs)
      os << spec.ToString();
    os << ", DeviceMesh: " << device_mesh.ToString() << "}";
    return os.str();
  }

  // Call this in GDB to visualize the spec.
  void Print() const {
    std::cout << ToString() << std::endl;
  }

  // Return true if at least one tensor axis is sharded.
  // Otherwise, return false.
  bool HasShard() const {
    for (const auto& spec : axis_specs)
      if (spec.cond == AxisPartitionSpec::Condition::Shard) return true;
    return false;
  }

  // Return true if no tensor axis is sharded.
  // Otherwise, return false.
  bool HasNoShard() const {
    return !HasShard();
  }

  // Return true if the only sharded tensor axis is `axis`.
  // Otherwise, return false.
  bool OnlyShardAxis(int64_t axis) const {
    ValidateAxisIndex(axis, Rank());
    if (axis < 0) {
      axis += Rank();
    }
    bool answer = true;
    for (int64_t i = 0; i < Rank(); ++i) {
      if (i == axis && axis_specs[i].cond != AxisPartitionSpec::Condition::Shard) {
        answer = false;
      } else if (i != axis && axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        answer = false;
      }
    }
    return answer;
  }

  // Rank of the owing tensor of this spec.
  int64_t Rank() const {
    return gsl::narrow<int64_t>(axis_specs.size());
  }

  // Return the number of sharded tensor axes.
  // Currently we only support one sharded tensor axis, so
  // we may assert the returned value is 1 in related APIs.
  int64_t CountShardingAxes() const {
    int64_t count = 0;
    for (const auto& spec : axis_specs)
      if (spec.cond == AxisPartitionSpec::Condition::Shard) count++;
    return count;
  }

  // Return the AxisPartitionSpec for `axis`-th tensor axis.
  const AxisPartitionSpec& GetAxisSpec(int64_t axis) const {
    ValidateAxisIndex(axis, Rank());
    if (axis < 0) {
      axis += Rank();
    }
    return axis_specs.at(axis);
  }

  // Get the first sharded tensor axis' sharding spec.
  const AxisPartitionSpec& GetPartitionAxisSpec() const {
    // TODO: support multiple sharding axes.
    ORT_ENFORCE(CountShardingAxes() == 1, "TensorPartitionSpec must have exactly one sharding axis.");
    return GetAxisSpec(GetPartitionAxis());
  }

  // Get the first sharded tensor axis' index.
  // E.g., spec "RS[0]" should return 1, spec "S[0]R" should return 0, spec "RR" should return -1.
  // Returned value -1 means no sharded tensor axis.
  int64_t GetPartitionAxis() const {
    // TODO: support multiple sharding axes.
    ORT_ENFORCE(CountShardingAxes() == 1, "TensorPartitionSpec must have exactly one sharding axis.");
    for (int64_t i = 0; i < gsl::narrow<int64_t>(axis_specs.size()); ++i) {
      if (axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        return i;
      }
    }
    return -1;
  }

  // Similarly to GetPartitionAxis(), but returns the negative index of the first sharded tensor axis.
  // E.g., spec "RS[0]" should return -1, spec "S[0]R" should return -2, and spec "RR" should return 0.
  // Returned value 0 means no sharded tensor axis.
  int64_t GetNegativePartitionAxis() const {
    // TODO: support multiple sharding axes.
    ORT_ENFORCE(CountShardingAxes() == 1, "TensorPartitionSpec must have exactly one sharding axis.");
    for (int64_t i = 0; i < gsl::narrow<int64_t>(axis_specs.size()); ++i) {
      if (axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        return i - axis_specs.size();
      }
    }
    return 0;
  }

  // Return the number of shards along the first sharded tensor axis.
  // This value matches the number of devices along the associated mesh axis.
  // Return 1 if there is no sharding.
  int64_t GetPartitionCount(int64_t axis) const {
    ValidateAxisIndex(axis, Rank());
    auto axis_spec = GetAxisSpec(axis);
    if (axis_spec.cond == AxisPartitionSpec::Condition::Replica) {
      return 1;
    } else {
      return device_mesh.device_mesh_shape.at(axis_spec.device_mesh_axis);
    }
  }
};

// Parse "[0, 1, 2, 3]" as std::vector<int64_t>{0, 1, 2, 3}.
std::vector<int64_t> ParseStringAsInt64Vector(const std::string& str);

DeviceMesh CreateDeviceMesh(
    std::vector<int64_t> device_mesh_shape,
    std::vector<int64_t> device_mesh_elements);

TensorPartitionSpec CreateTensorPartitionSpec(
    std::string spec_string,
    std::vector<int64_t> device_mesh_shape,
    std::vector<int64_t> device_mesh_elements);

TensorPartitionSpec CreateTensorShardSpec(
    const DeviceMesh& device_mesh,
    int64_t device_mesh_axis,
    int64_t shard_axis,
    int64_t tensor_rank);

// Return the shape of the original tensor before sharding.
// E.g., assume tensor shard's shape is [5, 7] and sharding spec is "S[0]R"
// with 1-D device mesh [0, 1, 2].
// This function returns [15, 7].
//
// `shard_shape`: the shape of a shard.
// `spec`: the sharding spec of the original tensor.
TensorShape ComputeOriginShape(const TensorShape& shard_shape, const TensorPartitionSpec& spec);

// Return the shape of a shard.
// E.g., assume tensor's shape is [15, 7] and sharding spec is "S[0]R"
// with 1-D device mesh [0, 1, 2].
// This function returns [5, 7].
//
// `shape`: the shape of the original tensor.
// `spec`: the sharding spec of the original tensor.
TensorShape ComputeShardShape(const TensorShape& shape, const TensorPartitionSpec& spec);

// Similarly to ComputeShardShape(), but takes a shard axis and counts of all tensor shards
// instead of a spec.
TensorShape ComputeShardShape(const TensorShape source_shape, int64_t shard_axis, int64_t shard_count);

// Prepend 1's to `shape` to make `left` and `right` have the same rank.
// E.g., if `left` is [3, 7] and `right` is [5, 6, 7], this function returns [1, 3, 7] and [5, 6, 7].
std::tuple<TensorShape, TensorShape> NormalizeShapes(const TensorShape& left, const TensorShape& right);

// Prepend `R` (aks replicating axis) to `spec` to make `left` and `right` have the same rank.
// E.g., if `left` is S[0]R and `right` is `RRR`, this function returns `RS[0]R` and `RRR`.
std::tuple<TensorPartitionSpec, TensorPartitionSpec> NormalizeTensorPartitionSpecs(
    const TensorPartitionSpec& left, const TensorPartitionSpec& right);

// Return true if `shape` can be sharded according to `spec`.
// Otherwise, return false.
// Note that an axis is shardable along a device mesh axis only if
// the dimension of the axis is divisible by the number of devices along the device mesh axis.
bool CanShard(const TensorShape& shape, const TensorPartitionSpec& spec);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
