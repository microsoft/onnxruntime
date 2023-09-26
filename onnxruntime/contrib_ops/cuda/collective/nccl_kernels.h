// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#if defined(ORT_USE_NCCL)
#include <optional>
#include <string>
#include <nccl.h>
#include <sstream>
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

// -----------------------------------------------------------------------
// Defines a new version of nccl classes
// that independent with training::DistributedRunContext, only rely on MPI
// -----------------------------------------------------------------------
class NcclContext final {
 public:
  NcclContext();
  ~NcclContext();

  ncclComm_t Comm() {
    return comm_;
  }

  int Rank() const {
    return rank_;
  }

  int Size() const {
    return world_size_;
  }

 private:
  ncclComm_t comm_;
  int rank_;
  int world_size_;
};

class NcclKernel : public ::onnxruntime::cuda::CudaKernel {
 public:
  explicit NcclKernel(const OpKernelInfo& info);

 protected:
  NcclContext* nccl_ = nullptr;
};

/*
 * Defines new version of Nccl classes that independent with training::DistributedContext
 * only rely on MPI
 */
class AllReduce final : public NcclKernel {
 public:
  explicit AllReduce(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

class AllGather final : public NcclKernel {
 public:
  explicit AllGather(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t group_size_ = -1;
  int64_t axis_ = -1;
  const CUDAExecutionProvider* cuda_ep_;
};

class AllToAll final : public NcclKernel {
 public:
  explicit AllToAll(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t group_size_ = -1;
};

class DeviceMesh {
public:
  std::vector<int64_t> device_mesh_shape;
  std::vector<int64_t> device_mesh_elements;
  std::string to_string() const {
    std::ostringstream os;
    os << "DeviceMesh { Shape: [";
    for(const auto& shape : device_mesh_shape)
        os << shape << ",";
    os << "], Elements: [";
    for(const auto& element : device_mesh_elements)
        os << element << ",";
    os << "]}";
    return os.str();
  }
};

class AxisPartitionSpec {
public:
  enum class Condition { Replica, Shard };
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
    if(cond == Condition::Shard) os << "[" << device_mesh_axis << "]";
    return os.str();
  }
};

class TensorPartitionSpec {
public:
  enum class Condition { Replica, Shard };
  std::vector<AxisPartitionSpec> axis_specs;
  DeviceMesh device_mesh;
  TensorPartitionSpec(std::vector<AxisPartitionSpec> axis_specs_, DeviceMesh device_mesh_) : axis_specs(axis_specs_), device_mesh(device_mesh_) {};
  std::string to_string() const {
    std::ostringstream os;
    os << "TensorPartitionSpec { ";
    for(const auto& spec : axis_specs)
        os << spec.to_string();
    os << ", Device Mesh: " << device_mesh.to_string() << " }";
    return os.str();
  }

  bool HasShard() const {
    for(const auto& spec : axis_specs)
        if(spec.cond == AxisPartitionSpec::Condition::Shard) return true;
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

  std::vector<int64_t> ShardShape(const std::vector<int64_t> &origin_shape) {
    std::vector<int64_t> shard_shape;
    for (int64_t i = 0; i < origin_shape.size(); ++i) {
      if (axis_specs[i].cond == AxisPartitionSpec::Condition::Shard) {
        shard_shape.push_back(origin_shape[i] / device_mesh.device_mesh_shape[axis_specs[i].device_mesh_axis]);
      } else {
        shard_shape.push_back(origin_shape[i]);
      }
    }
    return shard_shape;
  }
};


DeviceMesh create_device_mesh(
    std::vector<int64_t> device_mesh_shape,
    std::vector<int64_t> device_mesh_elements
) {
  DeviceMesh device_mesh;
  device_mesh.device_mesh_shape = device_mesh_shape;
  device_mesh.device_mesh_elements = device_mesh_elements;
  return device_mesh;
}

TensorPartitionSpec create_tensor_partition_spec(std::string spec_string, std::vector<int64_t> device_mesh_shape, std::vector<int64_t> device_mesh_elements) {
  // "S[0]R"
  std::vector<AxisPartitionSpec> axis_specs;
  size_t dim_index = 0;
  size_t token_index = 0;
  while (token_index < spec_string.size()) {
    char token = spec_string.at(token_index);
    if (token == 'R') {
      AxisPartitionSpec axis_spec = AxisPartitionSpec::CreateReplica();
      axis_specs.push_back(axis_spec);
      ++token_index;
      ++dim_index;
    } else if (token == 'S') {
      std::stringstream ss;
      // Skip "[".
      ++token_index;
      while (spec_string.at(token_index) != ']') {
        // Now token_index should points to the first digit of
        // axis index.
        char digit = spec_string.at(token_index);
        ss << digit;
        // Loaded a digit. Go to next token.
        ++token_index;
      }
      int device_mesh_index = 0;
      ss >> device_mesh_index;
      AxisPartitionSpec axis_spec = AxisPartitionSpec::CreateShard(device_mesh_index);
      axis_specs.push_back(axis_spec);
      // Skip "]".
      ++token_index;
    } else {
      throw std::invalid_argument("Invalid partition token: " + token);
    }
  }
  DeviceMesh device_mesh = create_device_mesh(device_mesh_shape, device_mesh_elements);
  return TensorPartitionSpec(axis_specs, device_mesh);
}

void normalize_shapes(std::vector<int64_t>& left, std::vector<int64_t>& right) {
  if (left.size() > right.size()) {
    right.insert(right.begin(), left.size() - right.size(), 1);
  } else if (left.size() < right.size()) {
    left.insert(left.begin(), right.size() - left.size(), 1);
  }
}

void normalize_tensor_partition_specs(TensorPartitionSpec& left, TensorPartitionSpec& right) {
  if (left.axis_specs.size() > right.axis_specs.size()) {
    right.axis_specs.insert(right.axis_specs.begin(), left.axis_specs.size() - right.axis_specs.size(), AxisPartitionSpec::CreateReplica());
  } else if (left.axis_specs.size() < right.axis_specs.size()) {
    left.axis_specs.insert(left.axis_specs.begin(), right.axis_specs.size() - left.axis_specs.size(), AxisPartitionSpec::CreateReplica());
  }
}

void infer_matmul_output_shape(
    const std::vector<int64_t>& normalized_shape_A,
    const std::vector<int64_t>& normalized_shape_B,
    std::vector<int64_t>& shape_Y) {
  // left_shape: [M, K]
  // right_shape: [K, N]
  // output_shape: [M, N]
  ORT_ENFORCE(
    normalized_shape_A.size() >= 2 && normalized_shape_B.size() >= 2,
    "1-D tensor is not supported by this MatMul."
  );
  ORT_ENFORCE(
    normalized_shape_A.size() == normalized_shape_B.size(),
    "A and B must have the same rank after shape broadcasting."
  );
  ORT_ENFORCE(shape_Y.size() == 0, "shape_Y must be empty before shape inference.");

  size_t rank = normalized_shape_A.size();
  shape_Y.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim_A = normalized_shape_A.at(i);
    const int64_t dim_B = normalized_shape_B.at(i);
    if (i == rank - 1) {
      shape_Y.push_back(dim_B);
    } else if (i == rank - 2) {
      shape_Y.push_back(dim_A);
    } else if (dim_A == 1 && dim_B >= 1) {
      shape_Y.push_back(dim_B);
    } else if (dim_B == 1 && dim_A >= 1) {
      shape_Y.push_back(dim_A);
    } else {
      ORT_ENFORCE(
        dim_A == dim_B,
        "A and B must have the same shape after shape broadcasting."
      );
      shape_Y.push_back(dim_A);
    }
  }
};

template <typename T>
class DistributedMatMul final : public NcclKernel {
 public:
  explicit DistributedMatMul (const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::vector<TensorPartitionSpec> input_shard_specs_;
  std::vector<TensorPartitionSpec> output_shard_specs_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
