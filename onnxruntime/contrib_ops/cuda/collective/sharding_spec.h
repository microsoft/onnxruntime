// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"

#include <sstream>
#include <vector>

#pragma once

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
  //     GPU 0/1/2/3 owns 5/7/6/8.
  //     Visualization of sharding rows and then columns with 2-D mesh:
  //     - GPU: [[0, 1], [2, 3]], Value: [[5, 6], [7, 8]]
  //     (Split GPU mesh along axis 1 and tensor along axis 0)
  //     - GPU: [[0], [2]], Value: [[5, 6]]
  //       GPU: [[1], [3]], Value: [[7, 8]]
  //     (Split GPU mesh along axis 0 and tensor along axis 1)
  //     - GPU: [[0]], Value: [[5]]
  //     - GPU: [[2]], Value: [[6]]
  //     - GPU: [[1]], Value: [[7]]
  //     - GPU: [[3]], Value: [[8]]

  // Actual shape of device mesh represented by `device_mesh_elements`.
  std::vector<int64_t> device_mesh_shape;
  // Flattened device mesh.
  std::vector<int64_t> device_mesh_elements;
  // Helper to debug and generate error message.
  std::string to_string() const {
    std::ostringstream os;
    os << "DeviceMesh { Shape: [";
    for (const auto& shape : device_mesh_shape)
      os << shape << ",";
    os << "], Elements: [";
    for (const auto& element : device_mesh_elements)
      os << element << ",";
    os << "]}";
    return os.str();
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
  enum class Condition { Replica,
                         Shard };
  // This field tells if a tensor axis is sharded or not.
  Condition cond;
  // If a tensor axis is sharded, this field tells which device
  // mesh axis to distribute the shards along.
  // If a tensor axis is not sharded, this field is ignored.
  int device_mesh_axis;
  static AxisPartitionSpec CreateReplica() {
    return AxisPartitionSpec(Condition::Replica, -1);
  }
  static AxisPartitionSpec CreateShard(int device_mesh_axis) {
    return AxisPartitionSpec(Condition::Shard, device_mesh_axis);
  }
  AxisPartitionSpec(Condition cond_, int device_mesh_axis_) : device_mesh_axis(device_mesh_axis_), cond(cond_) {}
  std::string to_string() const {
    std::ostringstream os;
    os << (cond == Condition::Replica ? "R" : "S");
    if (cond == Condition::Shard) os << "[" << device_mesh_axis << "]";
    return os.str();
  }
};

class TensorPartitionSpec {
 public:
  std::vector<AxisPartitionSpec> axis_specs;
  DeviceMesh device_mesh;
  static TensorPartitionSpec Create(
      const std::vector<AxisPartitionSpec>& axis_specs, const DeviceMesh& device_mesh) {
    TensorPartitionSpec spec;
    spec.axis_specs = axis_specs;
    spec.device_mesh = device_mesh;
    return spec;
  }
  static TensorPartitionSpec CreateAllReplica(
      const TensorPartitionSpec& spec) {
    TensorPartitionSpec new_spec = spec;
    new_spec.axis_specs[spec.GetPartitionAxis()] = AxisPartitionSpec::CreateReplica();
    return new_spec;
  }
  std::string to_string() const {
    std::ostringstream os;
    os << "TensorPartitionSpec { ";
    for (const auto& spec : axis_specs)
      os << spec.to_string();
    os << ", Device Mesh: " << device_mesh.to_string() << " }";
    return os.str();
  }

  bool HasShard() const {
    for (const auto& spec : axis_specs)
      if (spec.cond == AxisPartitionSpec::Condition::Shard) return true;
    return false;
  }

  bool HasNoShard() const {
    return !HasShard();
  }

  bool OnlyShardAxis(int64_t axis) const {
    if (axis < 0) {
      axis += axis_specs.size();
    }
    bool answer = true;
    for (int64_t i = 0; i < gsl::narrow<int64_t>(axis_specs.size()); ++i) {
      if (i == axis && axis_specs[i].cond != AxisPartitionSpec::Condition::Shard) {
        answer = false;
      } else if (i != axis && axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        answer = false;
      }
    }
    return answer;
  }
  int64_t Rank() const {
    return gsl::narrow<int64_t>(axis_specs.size());
  }

  const AxisPartitionSpec& GetAxisSpec(int64_t axis) const {
    if (axis < 0) {
      axis += axis_specs.size();
    }
    return axis_specs.at(axis);
  }

  const AxisPartitionSpec& GetPartitionAxisSpec() const {
    return GetAxisSpec(GetPartitionAxis());
  }

  int64_t GetPartitionAxis() const {
    for (int64_t i = 0; i < gsl::narrow<int64_t>(axis_specs.size()); ++i) {
      if (axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        return i;
      }
    }
    return -1;
  }

  int64_t GetNegativePartitionAxis() const {
    for (int64_t i = 0; i < gsl::narrow<int64_t>(axis_specs.size()); ++i) {
      if (axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        return i - axis_specs.size();
      }
    }
    return 0;
  }

  int64_t GetPartitionCount(int64_t axis) const {
    auto axis_spec = GetAxisSpec(axis);
    if (axis_spec.cond == AxisPartitionSpec::Condition::Replica) {
      return 1;
    } else {
      return device_mesh.device_mesh_shape.at(axis_spec.device_mesh_axis);
    }
  }
};

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

TensorShape ComputeOriginShape(const TensorShape& shard_shape, const TensorPartitionSpec& spec);

TensorShape ComputeShardShape(const TensorShape& shape, const TensorPartitionSpec& spec);

TensorShape ComputeShardShape(const TensorShape source_shape, int64_t shard_axis, int64_t shard_count);

std::tuple<TensorShape, TensorShape> NormalizeShapes(const TensorShape& left, const TensorShape& right);

std::tuple<TensorPartitionSpec, TensorPartitionSpec> NormalizeTensorPartitionSpecs(
    const TensorPartitionSpec& left, const TensorPartitionSpec& right);

bool CanShard(const TensorShape& shape, const TensorPartitionSpec& spec);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
