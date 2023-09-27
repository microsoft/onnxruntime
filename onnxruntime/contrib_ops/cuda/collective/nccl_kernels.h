// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#if defined(ORT_USE_NCCL)
#include <algorithm>
#include <tuple>
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

  ncclComm_t Comm() const {
    return nccl_->Comm();
  }

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
  std::vector<AxisPartitionSpec> axis_specs;
  DeviceMesh device_mesh;
  static TensorPartitionSpec Create(
    const std::vector<AxisPartitionSpec>& axis_specs, const DeviceMesh& device_mesh
  ) {
    TensorPartitionSpec spec;
    spec.axis_specs = axis_specs;
    spec.device_mesh = device_mesh;
    return spec;
  }
  static TensorPartitionSpec Create(
    const TensorPartitionSpec& spec, int64_t new_shard_axis
  ) {
    if (new_shard_axis < 0) {
      new_shard_axis += spec.axis_specs.size();
    }
    TensorPartitionSpec new_spec = spec;
    std::swap(new_spec.axis_specs[new_shard_axis], new_spec.axis_specs[spec.GetPartitionAxis()]);
    return new_spec;
  }
  static TensorPartitionSpec CreateAllReplica(
    const TensorPartitionSpec& spec
  ) {
    TensorPartitionSpec new_spec = spec;
    new_spec.axis_specs[spec.GetPartitionAxis()] = AxisPartitionSpec::CreateReplica();
    return new_spec;
  }
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
};


DeviceMesh CreateDeviceMesh(
    std::vector<int64_t> device_mesh_shape,
    std::vector<int64_t> device_mesh_elements
) {
  DeviceMesh device_mesh;
  device_mesh.device_mesh_shape = device_mesh_shape;
  device_mesh.device_mesh_elements = device_mesh_elements;
  return device_mesh;
}

TensorPartitionSpec CreateTensorPartitionSpec(std::string spec_string, std::vector<int64_t> device_mesh_shape, std::vector<int64_t> device_mesh_elements) {
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
  DeviceMesh device_mesh = CreateDeviceMesh(device_mesh_shape, device_mesh_elements);
  return TensorPartitionSpec::Create(axis_specs, device_mesh);
}

TensorPartitionSpec CreateTensorShardSpec(
  const DeviceMesh& device_mesh,
  int64_t device_mesh_axis,
  int64_t shard_axis,
  int64_t tensor_rank
) {
  if (shard_axis < 0) {
    shard_axis += tensor_rank;
  }
  std::vector<AxisPartitionSpec> axis_specs;
  for (int64_t i = 0; i < tensor_rank; ++i) {
    if (i == shard_axis) {
      axis_specs.push_back(AxisPartitionSpec::CreateShard(device_mesh_axis));
    } else {
      axis_specs.push_back(AxisPartitionSpec::CreateReplica());
    }
  }
  return TensorPartitionSpec::Create(axis_specs, device_mesh);
}

TensorShape ComputeOriginShape(const TensorShape& shard_shape, const TensorPartitionSpec& spec) {
  TensorShape shape(shard_shape);
  const int64_t axis = spec.GetPartitionAxis();
  shape[axis] *= spec.GetPartitionCount(axis);
  return shape;
}

TensorShape ComputeShardShape(const TensorShape& shape, const TensorPartitionSpec& spec) {
  TensorShape shard_shape(shape);
  if (spec.HasNoShard()) {
    return shard_shape;
  }
  const int64_t axis = spec.GetPartitionAxis();
  shard_shape[axis] /= spec.GetPartitionCount(axis);
  return shard_shape;
}

std::tuple<TensorShape, TensorShape> NormalizeShapes(const TensorShape& left, const TensorShape& right) {
  if (left.NumDimensions() > right.NumDimensions()) {
    std::vector<int64_t> right_vector(right.NumDimensions(), 0);
    right.CopyDims(right_vector.data(), right.NumDimensions());
    // Fill 1's to right shape. E.g.,
    // left: [1, 2, 3, 4], right: [5, 6, 7] -> left: [1, 2, 3, 4], right: [1, 5, 6, 7]
    right_vector.insert(right_vector.begin(), left.NumDimensions() - right.NumDimensions(), 1);
    return std::make_tuple(left, TensorShape(right_vector));
  } else if (left.NumDimensions() < right.NumDimensions()) {
    std::vector<int64_t> left_vector(left.NumDimensions(), 0);
    left.CopyDims(left_vector.data(), left.NumDimensions());
    // Fill 1's to left shape. E.g.,
    // left: [1, 2, 3], right: [4, 5, 6, 7] -> left: [1, 2, 3, 1], right: [4, 5, 6, 7]
    left_vector.insert(left_vector.begin(), right.NumDimensions() - left.NumDimensions(), 1);
    return std::make_tuple(TensorShape(left_vector), TensorShape(right));
  } else {
    return std::make_tuple(TensorShape(left), TensorShape(right));
  }
}

std::tuple<TensorPartitionSpec, TensorPartitionSpec> NormalizeTensorPartitionSpecs(
  const TensorPartitionSpec& left, const TensorPartitionSpec& right
) {
  // TODO: Make it to modify left and right instead of returning new values.
  if (left.axis_specs.size() > right.axis_specs.size()) {
    auto new_right = TensorPartitionSpec::Create(right.axis_specs, right.device_mesh);
    new_right.axis_specs.insert(new_right.axis_specs.begin(), left.axis_specs.size() - right.axis_specs.size(), AxisPartitionSpec::CreateReplica());
    return std::make_tuple(left, new_right);
  } else if (left.axis_specs.size() < right.axis_specs.size()) {
    auto new_left = TensorPartitionSpec::Create(left.axis_specs, left.device_mesh);
    new_left.axis_specs.insert(new_left.axis_specs.begin(), right.axis_specs.size() - left.axis_specs.size(), AxisPartitionSpec::CreateReplica());
    return std::make_tuple(new_left, right);
  } else {
    return std::make_tuple(left, right);
  }
}

TensorShape InferMatmulOutputShape(
    const TensorShape& shape_A,
    const TensorShape& shape_B
) {
  // left_shape: [M, K]
  // right_shape: [K, N]
  // output_shape: [M, N]
  ORT_ENFORCE(
    shape_A.NumDimensions() >= 2 && shape_B.NumDimensions() >= 2,
    "1-D tensor is not supported by this MatMul."
  );
  ORT_ENFORCE(
    shape_A.NumDimensions() == shape_B.NumDimensions(),
    "A and B must have the same rank after shape broadcasting."
  );
  size_t rank = shape_A.NumDimensions();
  std::vector<int64_t> shape_Y(rank, 0);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t dim_A = shape_A[i];
    const int64_t dim_B = shape_B[i];
    if (i == rank - 1) {
      shape_Y[i] = dim_B;
    } else if (i == rank - 2) {
      shape_Y[i] = dim_A;
    } else if (dim_A == 1 && dim_B >= 1) {
      // dim_A is 1.
      // dim_B can be either 1 or other positive integer.
      // due ot shape broadcast.
      shape_Y[i] = dim_B;
    } else if (dim_B == 1 && dim_A >= 1) {
      // dim_B is 1.
      // dim_A can be either 1 or other positive integer.
      // due ot shape broadcast.
      shape_Y[i] = dim_A;
    } else {
      ORT_ENFORCE(dim_A == dim_B, "Broadcasting can only happen when one of dim_A and dim_B is 1.");
      shape_Y[i] = dim_A;
    }
  }
  return TensorShape(shape_Y);
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
