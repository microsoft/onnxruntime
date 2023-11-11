// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sharding_spec.h"

#include "core/common/common.h"
#include "core/common/gsl.h"
#include "core/framework/tensor_shape.h"

#include <cctype>
#include <sstream>
#include <vector>

namespace onnxruntime {
namespace distributed {

#if defined(ORT_USE_NCCL)

void ValidateAxisIndex(const int64_t axis, const int64_t rank) {
  int64_t adjusted_axis = axis;
  if (axis < 0) {
    adjusted_axis = axis + rank;
  } else {
    adjusted_axis = axis;
  }
  ORT_ENFORCE(adjusted_axis >= 0 && adjusted_axis < rank, "axis,", axis, ", should be in [", -rank, ",", rank, ").");
}

std::vector<int64_t> ParseStringAsInt64Vector(const std::string& str) {
  if (str.empty() || str.front() != '[' || str.back() != ']') {
    throw std::invalid_argument("Invalid input string format");
  }
  // Parsed vector.
  // If input is "[0, 1, 2]", result should be {0, 1, 2}.
  std::vector<int64_t> result;
  // Skip '[' and ']'
  std::istringstream iss(str.substr(1, str.size() - 2));

  // Extract integers separated by ',' or whitespaces.
  int64_t num = -1;
  while (/* Read one number at a time */ iss >> num) {
    result.push_back(num);
    // Skip the comma
    if (iss.peek() == ',') {
      iss.ignore();
    }
  }
  return result;
}

DeviceMesh CreateDeviceMesh(
    std::vector<int64_t> device_mesh_shape,
    std::vector<int64_t> device_mesh_elements) {
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
      // Next should be "[".
      ++token_index;
      char left_bracket = spec_string.at(token_index);
      ORT_ENFORCE(left_bracket == '[', "Invalid partition token: ", left_bracket, " in ", spec_string);
      // Move to digit part.
      ++token_index;
      while (spec_string.at(token_index) != ']') {
        // Now token_index should points to the first digit of
        // axis index.
        char digit = spec_string.at(token_index);
        ORT_ENFORCE(std::isdigit(digit), "Invalid partition token: ", token, " in ", spec_string);
        ss << digit;
        // Loaded a digit. Go to next token.
        ++token_index;
      }
      int device_mesh_index = 0;
      ss >> device_mesh_index;
      AxisPartitionSpec axis_spec = AxisPartitionSpec::CreateShard(device_mesh_index);
      axis_specs.push_back(axis_spec);
      // Skip "]".
      char right_bracket = spec_string.at(token_index);
      ORT_ENFORCE(right_bracket == ']', "Invalid partition token: ", token, " in ", spec_string);
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
    int64_t tensor_rank) {
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
  ORT_ENFORCE(gsl::narrow<int64_t>(shard_shape.NumDimensions()) == spec.Rank(), "Shard shape and spec rank mismatch.");
  if (spec.HasNoShard()) {
    return shard_shape;
  }
  TensorShape shape(shard_shape);
  const int64_t axis = spec.GetPartitionAxis();
  shape[axis] *= spec.GetUniqueDeviceCount(axis);
  return shape;
}

TensorShape ComputeShardShape(const TensorShape& shape, const TensorPartitionSpec& spec) {
  ORT_ENFORCE(gsl::narrow<int64_t>(shape.NumDimensions()) == spec.Rank(), "Shape and spec rank mismatch.");
  TensorShape shard_shape(shape);
  if (spec.HasNoShard()) {
    return shard_shape;
  }
  const int64_t axis = spec.GetPartitionAxis();
  const int64_t unique_device_count = spec.GetUniqueDeviceCount(axis);
  ORT_ENFORCE(shard_shape[axis] % unique_device_count == 0, "Number of shards must be divisible by sharded axis' dimension.");
  // If a [8, 16]-tensor is shared by device mesh [0, 1, 0, 1] along axis=1 (2nd axis),
  // the local tensors on device 0 & 1 have same shape [8, 8 (from 16/2)] instead of
  // [8, 4 (from 16/4)]. The reason is that
  //  - First, the original tensor are split into 4 sub-tensors [8, 4] along the 2nd axis.
  //  - The 1st and 3rd sub-tensors are concatenated along axis=1 to one tensor on device 0.
  //  - The 2nd and 4th sub-tensors are concatenated along axis=1 to one tensor on device 1.
  shard_shape[axis] /= unique_device_count;
  return shard_shape;
}

TensorShape ComputeShardShape(const TensorShape source_shape, int64_t shard_axis, int64_t shard_count) {
  if (shard_axis < 0) {
    shard_axis += gsl::narrow<int64_t>(source_shape.NumDimensions());
  }
  TensorShape shard_shape(source_shape);
  ORT_ENFORCE(shard_axis < gsl::narrow<int64_t>(source_shape.NumDimensions()), "Shard axis must be less than the number of dimensions of the source tensor.");
  ORT_ENFORCE(source_shape[shard_axis] % shard_count == 0, "Number of shards must be divisible by sharded axis' dimension.");
  shard_shape[shard_axis] = source_shape[shard_axis] / shard_count;
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
    const TensorPartitionSpec& left, const TensorPartitionSpec& right) {
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

bool CanShard(const TensorShape& shape, const TensorPartitionSpec& spec) {
  if (spec.HasNoShard()) {
    return true;
  }
  if (gsl::narrow<int64_t>(shape.NumDimensions()) != spec.Rank()) {
    return false;
  }
  const int64_t axis = spec.GetPartitionAxis();
  if (axis < 0 || gsl::narrow<size_t>(axis) >= shape.NumDimensions()) {
    return false;
  }
  if (shape[axis] % spec.GetDeviceCount(axis) != 0) {
    return false;
  }
  return true;
}

#endif

}  // namespace distributed
}  // namespace onnxruntime
