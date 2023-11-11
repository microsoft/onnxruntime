// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sharding.h"
#include "mpi_include.h"
#include "core/framework/sharding_spec.h"

#include <vector>
#include <string>
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/cuda_check_memory.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

void GatherTensor(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const NcclKernel* nccl_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const TensorPartitionSpec& spec,
    const Tensor* tensor,
    Tensor* gathered) {
  const int64_t shard_axis = spec.GetPartitionAxis();
  const int64_t shard_count = spec.GetUniqueDeviceCount(shard_axis);

  FuncAllGather(
      nccl_kernel,
      ctx,
      tensor,
      shard_count,
      shard_axis,
      gathered);
}

std::unique_ptr<Tensor> GatherTensor(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const NcclKernel* nccl_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const TensorPartitionSpec& spec,
    const Tensor* tensor) {
  const int64_t shard_axis = spec.GetPartitionAxis();
  const int64_t shard_count = spec.GetUniqueDeviceCount(shard_axis);
  TensorShape gathered_shape(tensor->Shape());
  gathered_shape[shard_axis] *= shard_count;

  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc) == Status::OK());
  auto gathered = Tensor::Create(tensor->DataType(), gathered_shape, alloc);

  FuncAllGather(
      nccl_kernel,
      ctx,
      tensor,
      shard_count,
      shard_axis,
      gathered.get());

  return gathered;
}

void ShardTensor(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const NcclKernel* nccl_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const TensorPartitionSpec& spec,
    const int64_t device_id,
    const Tensor* tensor,
    Tensor* shard_tensor) {
  const int64_t shard_axis = spec.GetPartitionAxis();
  const int64_t shard_count = spec.GetUniqueDeviceCount(shard_axis);
  TensorShape shard_shape = ComputeShardShape(
      tensor->Shape(),
      shard_axis,
      shard_count);
  const int64_t shard_dim = shard_shape[shard_axis];
  const std::vector<int64_t> starts = {shard_dim * device_id};
  const std::vector<int64_t> ends = {shard_dim * (device_id + 1)};
  const std::vector<int64_t> axes = {shard_axis};
  const std::vector<int64_t> steps = {1};

  ORT_ENFORCE(FuncSlice(
                  nccl_kernel,
                  ctx,
                  tensor,
                  starts,
                  ends,
                  axes,
                  steps,
                  shard_tensor) == Status::OK());
}

std::unique_ptr<Tensor> ShardTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& spec,
    const int64_t device_id,
    const Tensor* tensor) {
  // Shard all-replica tensor per spec.

  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc) == Status::OK());

  TensorShape shard_shape = ComputeShardShape(
      tensor->Shape(),
      spec.GetPartitionAxis(),
      spec.GetUniqueDeviceCount(spec.GetPartitionAxis()));
  auto shard_buffer = Tensor::Create(tensor->DataType(), shard_shape, alloc);

  // Shard with pre-allocated buffer.
  ShardTensor(
      nccl_kernel,
      ctx,
      spec,
      device_id,
      tensor,
      shard_buffer.get());

  return shard_buffer;
}

void ReshardTensor(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const NcclKernel* nccl_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const TensorPartitionSpec& src_spec,
    const TensorPartitionSpec& dst_spec,
    const int64_t device_id,
    const Tensor* src,
    Tensor* dst) {
  if (src_spec.HasShard() && dst_spec.HasNoShard()) {
    GatherTensor(
        nccl_kernel,
        ctx,
        src_spec,
        src,
        dst);
    return;
  } else if (src_spec.HasNoShard() && dst_spec.HasShard()) {
    ShardTensor(
        nccl_kernel,
        ctx,
        dst_spec,
        device_id,
        src,
        dst);
  } else if (src_spec.HasShard() && dst_spec.HasShard()) {
    int64_t src_axis = src_spec.GetPartitionAxis();
    int64_t dst_axis = dst_spec.GetPartitionAxis();
    ORT_ENFORCE(src_axis != dst_axis, "No reshard is needed. Don't call this function.");

    auto all_replica_buffer = GatherTensor(
        nccl_kernel,
        ctx,
        src_spec,
        src);

    ShardTensor(
        nccl_kernel,
        ctx,
        dst_spec,
        device_id,
        all_replica_buffer.get(),
        dst);
  } else {
    ORT_THROW("Not supported yet. Probably resharding is not needed.");
  }
}

std::unique_ptr<Tensor> ReshardTensor(
    // Use OpKernel and do a pointer cast to unify functional calls with other eps.
    // TODO: remove CudaKernel and OpKernelContext.
    const NcclKernel* nccl_kernel,
    // Do NOT use ctx to access inputs and outputs.
    // Inputs and outputs are passed in as function arguments.
    OpKernelContext* ctx,
    const TensorPartitionSpec& src_spec,
    const TensorPartitionSpec& dst_spec,
    const int64_t device_id,
    const Tensor* src) {
  // Implement ReshardTensor but returning a unique_ptr to Tensor instead.
  const auto origin_shape = ComputeOriginShape(src->Shape(), src_spec);
  const auto dst_shape = ComputeShardShape(origin_shape, dst_spec);
  ORT_ENFORCE(CanShard(origin_shape, dst_spec), "Cannot shard tensor. Shape:", origin_shape, ", sharding spec: ", dst_spec.ToString());

  AllocatorPtr alloc;
  ORT_ENFORCE(ctx->GetTempSpaceAllocator(&alloc) == Status::OK());
  auto dst = Tensor::Create(src->DataType(), dst_shape, alloc);
  ReshardTensor(
      nccl_kernel,
      ctx,
      src_spec,
      dst_spec,
      device_id,
      src,
      dst.get());
  return dst;
}

void ReshardTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& src_spec,
    const TensorPartitionSpec& dst_spec,
    const int64_t device_id,
    const Tensor* src,
    int output_idx) {
  // Implement ReshardTensor but returning a unique_ptr to Tensor instead.
  const auto origin_shape = ComputeOriginShape(src->Shape(), src_spec);
  const auto dst_shape = ComputeShardShape(origin_shape, dst_spec);
  ORT_ENFORCE(CanShard(origin_shape, dst_spec), "Cannot shard tensor. Shape:", origin_shape, ", sharding spec: ", dst_spec.ToString());

  auto* dst = ctx->Output(output_idx, dst_shape);
  ReshardTensor(
      nccl_kernel,
      ctx,
      src_spec,
      dst_spec,
      device_id,
      src,
      dst);
}

DistributedKernel::DistributedKernel(const OpKernelInfo& info) : NcclKernel(info) {
  // input_device_mesh_shapes[i] is the shape of device mesh for the i-th input.
  // E.g., device_mesh_shapes = ["[2]", "[1]"] means the first input is
  // stored on a 1-D mesh with 2 devices and the second input on another 1-D
  // mesh with 1 device.
  std::vector<std::string> attr_input_device_mesh_shapes;
  ORT_ENFORCE(info.GetAttrs<std::string>("input_device_mesh_shapes", attr_input_device_mesh_shapes).IsOK());

  // input_device_mesh_elements[i] is the flattened device mesh for the i-th input.
  // Note that its actual shape is input_device_mesh_shapes[i].
  // Example:
  //  Assume
  //   device_mesh_shapes = ["[2]", "[1]"]
  //   device_mesh_elements = ["[0,1]", "[0]"]
  //  Then the first input is stored on a 1-D mesh with 2 devices and the second
  //  input on another 1-D mesh with 1 device.
  std::vector<std::string> attr_input_device_mesh_elements;
  ORT_ENFORCE(info.GetAttrs<std::string>("input_device_mesh_elements", attr_input_device_mesh_elements).IsOK());

  // input_shard_specs[i] is the sharding spec of the i-th input; e.g.,
  // "RR" if the i-th input is not sharded.
  std::vector<std::string> input_shard_specs;
  ORT_ENFORCE(info.GetAttrs<std::string>("input_shard_specs", input_shard_specs).IsOK());

  ORT_ENFORCE(attr_input_device_mesh_shapes.size() == attr_input_device_mesh_elements.size());
  ORT_ENFORCE(attr_input_device_mesh_shapes.size() == input_shard_specs.size());

  // Begin parsing sharding metadata for inputs.
  for (size_t i = 0; i < input_shard_specs.size(); ++i) {
    auto device_mesh_shape = ParseStringAsInt64Vector(attr_input_device_mesh_shapes[i]);
    auto device_mesh_elements = ParseStringAsInt64Vector(attr_input_device_mesh_elements[i]);
    auto spec = CreateTensorPartitionSpec(input_shard_specs[i], device_mesh_shape, device_mesh_elements);
    input_shard_specs_.push_back(spec);
  }

  std::vector<std::string> attr_output_device_mesh_shapes;
  ORT_ENFORCE(info.GetAttrs<std::string>("output_device_mesh_shapes", attr_output_device_mesh_shapes).IsOK());

  std::vector<std::string> attr_output_device_mesh_elements;
  ORT_ENFORCE(info.GetAttrs<std::string>("output_device_mesh_elements", attr_output_device_mesh_elements).IsOK());

  std::vector<std::string> output_shard_specs;
  ORT_ENFORCE(info.GetAttrs<std::string>("output_shard_specs", output_shard_specs).IsOK());

  ORT_ENFORCE(attr_output_device_mesh_shapes.size() == attr_output_device_mesh_elements.size());
  ORT_ENFORCE(attr_output_device_mesh_shapes.size() == output_shard_specs.size());

  for (size_t i = 0; i < output_shard_specs.size(); ++i) {
    auto device_mesh_shape = ParseStringAsInt64Vector(attr_output_device_mesh_shapes[i]);
    auto device_mesh_elements = ParseStringAsInt64Vector(attr_output_device_mesh_elements[i]);
    auto spec = CreateTensorPartitionSpec(output_shard_specs[i], device_mesh_shape, device_mesh_elements);
    output_shard_specs_.push_back(spec);
  }
}

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
