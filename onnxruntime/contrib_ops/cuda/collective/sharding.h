// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/framework/sharding_spec.h"
#include "nccl_kernels.h"

using namespace onnxruntime::distributed;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

void GatherTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& spec,
    const Tensor* tensor,
    Tensor* gathered);

std::unique_ptr<Tensor> GatherTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& spec,
    const Tensor* tensor);

void ShardTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& spec,
    const int64_t device_id,
    const Tensor* tensor,
    Tensor* shard_tensor);

std::unique_ptr<Tensor> ShardTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& spec,
    const int64_t device_id,
    const Tensor* tensor);

void ReshardTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& src_spec,
    const TensorPartitionSpec& dst_spec,
    const int64_t device_id,
    const Tensor* src,
    Tensor* dst);

// Output from ctx
void ReshardTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& src_spec,
    const TensorPartitionSpec& dst_spec,
    const int64_t device_id,
    const Tensor* src,
    int output_idx);

std::unique_ptr<Tensor> ReshardTensor(
    const NcclKernel* nccl_kernel,
    OpKernelContext* ctx,
    const TensorPartitionSpec& src_spec,
    const TensorPartitionSpec& dst_spec,
    const int64_t device_id,
    const Tensor* src);

class DistributedKernel : public NcclKernel {
 public:
  explicit DistributedKernel(const OpKernelInfo& info);

 protected:
  std::vector<TensorPartitionSpec> input_shard_specs_;
  std::vector<TensorPartitionSpec> output_shard_specs_;
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
