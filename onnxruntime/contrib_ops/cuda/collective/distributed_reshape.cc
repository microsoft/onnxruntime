// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_reshape.h"
#include "sharding.h"
#include "sharding_spec.h"
#include "nccl_kernels.h"
#include "mpi_include.h"

// ORT system.
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/cuda_check_memory.h"

// std C++.
#include <iostream>

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
  size_t dst_begin, size_t dst_end
) {
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
  const std::vector<int64_t>& dst_shape
) {
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
            true, i_dst, i_src, 2
          );
        }
    }
  }
  return std::make_tuple(false, 0, 0, 0);
}

std::tuple<bool, size_t, size_t, size_t> IsTwoAxisDecomposition(
  const std::vector<int64_t>& src_shape,
  const std::vector<int64_t>& dst_shape
) {
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
  const DeviceMesh& source_mesh, const int64_t repeat
) {
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
    const int64_t fusion_axis_in_dst
) {
  // TODO(wechi): use device mesh stride to support non-1 stride.
  // Example: S[0]R, shape=[2, 3], device_mesh=[0, 1] -> S[0], shape = [6], device_mesh=[0, 1]
  // Example: RS[0], shape=[2, 3], device_mesh=[0, 1] -> S[0], shape = [6], device_mesh=[0, 1, 0, 1]
  // Example: S[0]RR, shape=[2, 3, 5], device_mesh=[0, 1] -> S[0]R, shape = [2, 15], device_mesh=[0, 1]
  if (src_spec.CountShardingAxes() != 1) {
    throw std::runtime_error("Too many sharding axes.");
  }
  if (src_spec.device_mesh.device_mesh_shape.size() != 1) {
    throw std::runtime_error("Source device mesh be 1-D.");
  }

  if (src_spec.HasNoShard()) {
    return std::make_tuple(true, TensorPartitionSpec::CreateAllReplica(dst_shape.size(), src_spec.device_mesh));
  } else if (src_spec.OnlyShardAxis(fused_axis_in_src)) {
    // Example: S[0]R, shape=[2, 3], device_mesh=[0, 1] -> S[0], shape = [6], device_mesh=[0, 1]
    auto dst_spec = TensorPartitionSpec::CreateOneTensorAxisOneDeviceMeshAxisSharding(
      dst_shape.size(), src_spec.device_mesh, fusion_axis_in_dst, /* 1-D mesh */ 0);
    return std::make_tuple(true, dst_spec);
  } else if (src_spec.OnlyShardAxis(fused_axis_in_src + 1)) {
    // Example: RS[0], shape=[2, 3], device_mesh=[0, 1] -> S[0], shape = [6], device_mesh=[0, 1, 0, 1]
    auto dst_device_mesh = CreateInterleaveDeviceMesh(src_spec.device_mesh, src_shape[fused_axis_in_src]);
    auto dst_spec = TensorPartitionSpec::CreateOneTensorAxisOneDeviceMeshAxisSharding(
      dst_shape.size(), dst_device_mesh, fusion_axis_in_dst, /* 1-D mesh */ 0);
    return std::make_tuple(true, dst_spec);
  } else {
    // It's two-axis fusion but the fusion region is not sharded.
    // Example: S[0]RR, shape=[2, 3, 5], device_mesh=[0, 1] -> S[0]R, shape = [2, 15], device_mesh=[0, 1]
    auto dst_spec = TensorPartitionSpec::CreateByDropOneAxis(
      src_spec, fused_axis_in_src + 1);
    return std::make_tuple(true, dst_spec);
  }
}

// Arguments:
//  - device_elements: a vector of device IDs.
//    It should only contain unique device IDs or
//    repeats of a list of unique device IDs. Otherwise,
//    (0, 0) is returned.
// Returns:
//  - counts per device
//  - unique device count
// Examples:
//  - [0, 1] -> (2, 1)
//  - [0, 1, 2, 0, 1, 2] -> (2, 3)
std::tuple<int64_t, int64_t> ComputeRepeatAndRepeatStride(
  const std::vector<int64_t>& device_elements
) {
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
  for (size_t i = 0; i < device_elements.size(); ++i) {
    for (size_t j = 0; j < repeat_stride; ++j) {
      if (device_elements.at(i + j * repeat_stride) != first_device_id) {
        // Unsupported device mesh patterns.
        return std::make_tuple(0, 0);
      }
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
    const int64_t decomposition_axis_in_dst
) {
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
          dst_axis_specs.push_back(AxisPartitionSpec::CreateCopy(src_spec.GetAxisSpec(src_axis)));
        } else if (dst_shape[decomposition_axis_in_dst] == 1) {
          // S[0] -> RS[0]
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
        } else {
          // S[0] -> S[0]R
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
          src_spec.device_mesh, 1
      );
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
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          dst_device_mesh = src_spec.device_mesh;
        } else if (dst_shape[decomposition_axis_in_dst + 1] == 1) {
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_device_mesh = src_spec.device_mesh;
        } else if (repeats == 1 && dst_shape[decomposition_axis_in_dst + 1] == device_stride * repeat_stride) {
          // S[0] -> RS[0]
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          dst_device_mesh = src_spec.device_mesh;
        } else if (repeats != 1 && repeats == dst_shape[decomposition_axis_in_dst] && dst_shape[decomposition_axis_in_dst + 1] / device_stride == repeat_stride) {
          // S[0] -> RS[0]
          dst_axis_specs.push_back(AxisPartitionSpec::CreateReplica());
          dst_axis_specs.push_back(AxisPartitionSpec::CreateShard(0));
          // Extract [0, 1] from [0, 1, 0, 1].
          dst_device_mesh.device_mesh_shape.push_back(repeat_stride);
          dst_device_mesh.device_mesh_elements.insert(
            dst_device_mesh.device_mesh_elements.end(),
            src_spec.device_mesh.device_mesh_elements.begin(),
            src_spec.device_mesh.device_mesh_elements.begin() + repeat_stride);
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

template <typename T>
DistributedReshape<T>::DistributedReshape(const OpKernelInfo& info) : DistributedKernel(info) {
  allow_zero_ = info.GetAttrOrDefault("allowzero", static_cast<int64_t>(0));
}

template <typename T>
Status DistributedReshape<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(context != nullptr);
  auto data_tensor = context->Input<Tensor>(0);
  auto shape_tensor = context->Input<Tensor>(1);

  const auto& data_sharding_spec = input_shard_specs_[0];
  const auto& shape_sharding_spec = input_shard_specs_[1];
  const auto& output_sharding_spec = output_shard_specs_[0];

  if (data_sharding_spec.HasNoShard() && shape_sharding_spec.HasNoShard() && output_sharding_spec.HasNoShard()) {
    // Case: all inputs and outputs are not sharded.
    const auto target_shape = onnxruntime::cuda::InferReshapeOutputShape(
      data_tensor,
      shape_tensor,
      allow_zero_
    );

    auto output_tensor = context->Output(0, target_shape);

    // Copy data from input from output.
    return FuncReshape(
      this,
      context,
      data_tensor,
      shape_tensor,
      allow_zero_,
      output_tensor
    );
  } else {
    // TODO: reshard shape if necessary.
    ORT_ENFORCE(shape_sharding_spec.HasNoShard());
    const auto original_data_shape = ComputeOriginShape(data_tensor->Shape(), data_sharding_spec);
    const auto original_shape_shape = ComputeOriginShape(shape_tensor->Shape(), shape_sharding_spec);
    const auto original_output_shape = ComputeOriginShape(context->Output(0, {})->Shape(), output_sharding_spec);
    std::vector<int64_t> src_shape(original_data_shape.GetDims().begin(), original_data_shape.GetDims().end());
    std::vector<int64_t> dst_shape(original_output_shape.GetDims().begin(), original_output_shape.GetDims().end());

    // Case: Two axis fusion
    bool is_two_axis_fusion = false;
    size_t two_axis_fusion_axis_in_dst = 0;
    size_t two_axis_fusion_first_fused_axis_in_src = 0;
    size_t two_axis_fusion_fused_axis_count = 0;
    std::tie(is_two_axis_fusion, two_axis_fusion_axis_in_dst, two_axis_fusion_first_fused_axis_in_src, two_axis_fusion_fused_axis_count) = IsTwoAxisFusion(src_shape, dst_shape);

    if (is_two_axis_fusion) {
      return Status::OK();
    }

    // Case: Two axis decomposition
    bool is_two_axis_decomposition = false;
    size_t two_axis_decomposition_decomposed_axis_in_src = 0;
    size_t two_axis_decomposition_first_factor_axis_in_dst = 0;
    size_t two_axis_decomposition_factor_axis_count_in_dst = 0;
    std::tie(is_two_axis_decomposition, two_axis_decomposition_decomposed_axis_in_src, two_axis_decomposition_first_factor_axis_in_dst, two_axis_decomposition_factor_axis_count_in_dst) = IsTwoAxisDecomposition(src_shape, dst_shape);
    if (is_two_axis_decomposition) {
      return Status::OK();
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
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
    DistributedReshape<int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReshape,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    DistributedReshape<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReshape,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .AllocateInputsContiguously()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    DistributedReshape<MLFloat16>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
