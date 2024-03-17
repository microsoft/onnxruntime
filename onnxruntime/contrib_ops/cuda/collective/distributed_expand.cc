// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_expand.h"
#include "sharding.h"
#include "nccl_kernels.h"
#include "mpi_include.h"

// ORT system.
#include "core/framework/sharding_spec.h"
#include "core/providers/cuda/tensor/expand.h"

// std C++.
#include <iostream>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

template <typename T>
DistributedExpand<T>::DistributedExpand(const OpKernelInfo& info) : DistributedKernel(info) {}

template <typename T>
Status DistributedExpand<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(context != nullptr);
  // Assumptions.
  //  - Shape is not sharded.
  // Algorithm.
  //  - Compute logical output shape.
  //  - Compute local output shape.
  //  - Expand from local input to local output.

  auto input_tensor = context->Input<Tensor>(0);
  auto shape_tensor = context->Input<Tensor>(1);
  const auto& input_sharding_spec = input_shard_specs_.at(0);
  const auto& shape_sharding_spec = input_shard_specs_.at(1);
  const auto& output_sharding_spec = output_shard_specs_.at(0);

  ORT_ENFORCE(shape_sharding_spec.HasNoShard(),
              "It's not worth to shard Shape tensor. "
              "If sharding shape is needed, please submit a feature request.");
  // Compute logical input shape.
  const auto original_input_shape = ComputeOriginShape(input_tensor->Shape(), input_sharding_spec);

  // Compute logical output shape.
  // This `shape_tensor` stores the logical output shape.
  const auto* p_shape = shape_tensor->Data<int64_t>();
  TensorShapeVector original_output_dims{p_shape, p_shape + shape_tensor->Shape().Size()};
  TensorShape original_output_shape(original_output_dims);
  ORT_ENFORCE(
      onnxruntime::cuda::ComputeOutputShape(
          Node().Name(),
          original_input_shape,
          original_output_dims, original_output_shape)
          .IsOK());

  // Compute local output shape.
  const auto local_output_shape = ComputeShardShape(original_output_shape, output_sharding_spec);

  auto output_tensor = context->Output(0, local_output_shape);

  return FuncExpand(
      this,
      context,
      input_tensor,
      shape_tensor,
      output_tensor);
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedExpand,
    kMSDomain,
    1,
    int64_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedExpand<int64_t>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedExpand,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedExpand<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedExpand,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedExpand<MLFloat16>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
