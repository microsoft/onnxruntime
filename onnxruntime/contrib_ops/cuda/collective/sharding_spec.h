#include <sstream>
#include <vector>
#include "core/common/common.h"
#include "core/framework/tensor_shape.h"

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

class DeviceMesh {
 public:
  std::vector<int64_t> device_mesh_shape;
  std::vector<int64_t> device_mesh_elements;
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
 public:
  enum class Condition { Replica,
                         Shard };
  Condition cond;
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
  static TensorPartitionSpec Create(
      const TensorPartitionSpec& spec, int64_t new_shard_axis) {
    if (new_shard_axis < 0) {
      new_shard_axis += spec.axis_specs.size();
    }
    TensorPartitionSpec new_spec = spec;
    std::swap(new_spec.axis_specs[new_shard_axis], new_spec.axis_specs[spec.GetPartitionAxis()]);
    return new_spec;
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
    for (int64_t i = 0; i < axis_specs.size(); ++i) {
      if (i == axis && axis_specs[i].cond != AxisPartitionSpec::Condition::Shard) {
        answer = false;
      } else if (i != axis && axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        answer = false;
      }
    }
    return answer;
  }
  int64_t Rank() const {
    return axis_specs.size();
  }

  const AxisPartitionSpec& GetAxisSpec(int64_t axis) const {
    if (axis < 0) {
      axis += axis_specs.size();
    }
    return axis_specs.at(axis);
  }

  int64_t GetPartitionAxis() const {
    for (int64_t i = 0; i < axis_specs.size(); ++i) {
      if (axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        return i;
      }
    }
    return -1;
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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
