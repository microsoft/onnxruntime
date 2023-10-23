// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sharding.h"
#include "mpi_include.h"
#include "sharding_spec.h"

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
  const int64_t shard_count = spec.GetPartitionCount(shard_axis);

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
  const int64_t shard_count = spec.GetPartitionCount(shard_axis);
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
  const int64_t shard_count = spec.GetPartitionCount(shard_axis);
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
      spec.GetPartitionCount(spec.GetPartitionAxis()));
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
  std::vector<int64_t> device_mesh_elements = info.GetAttrsOrDefault<int64_t>("device_mesh_elements");
  std::vector<int64_t> device_mesh_shape = info.GetAttrsOrDefault<int64_t>("device_mesh_shape");
  std::vector<std::string> input_shard_specs = info.GetAttrsOrDefault<std::string>("input_shard_specs");
  std::vector<std::string> output_shard_specs = info.GetAttrsOrDefault<std::string>("output_shard_specs");

  for (size_t i = 0; i < input_shard_specs.size(); ++i) {
    auto spec = CreateTensorPartitionSpec(input_shard_specs[i], device_mesh_shape, device_mesh_elements);
    input_shard_specs_.push_back(spec);
  }
  for (size_t i = 0; i < output_shard_specs.size(); ++i) {
    auto spec = CreateTensorPartitionSpec(output_shard_specs[i], device_mesh_shape, device_mesh_elements);
    output_shard_specs_.push_back(spec);
  }
}

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
