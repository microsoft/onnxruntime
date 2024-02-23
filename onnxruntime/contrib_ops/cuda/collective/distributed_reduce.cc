
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_reduce.h"
#include "sharding.h"
#include "sharding_spec.h"
#include "nccl_kernels.h"
#include "mpi_include.h"

// ORT system.
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/reduction/reduction_ops.h"

// std C++.
#include <iostream>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

template <typename T>
DistributedReduceBase<T>::DistributedReduceBase(
    const OpKernelInfo& info,
    cudnnReduceTensorOp_t cudnn_reduce_op) : DistributedKernel(info) {
  keepdims_ = info.GetAttrOrDefault<int64_t>("keepdims", 1);
  cudnn_reduce_op_ = cudnn_reduce_op;
};

template <typename T>
Status DistributedReduceBase<T>::ComputeInternal(OpKernelContext* context) const {
  const auto& input_sharding_spec = input_shard_specs_.at(0);
  const auto& axes_sharding_spec = input_shard_specs_.at(1);
  const auto& output_sharding_spec = output_shard_specs_.at(0);

  ORT_ENFORCE(axes_sharding_spec.HasNoShard(),
              "It's not worthy to shard axes tensor. "
              "If sharding axes is needed, please submit a feature request.");

  const Tensor* input_tensor = context->Input<Tensor>(0);
  const Tensor* axes_tensor = context->Input<Tensor>(1);
  ORT_ENFORCE(axes_tensor->Shape().NumDimensions() == 1, "Axes tensor must be an 1-D tensor.");
  auto axes_span = axes_tensor->DataAsSpan<int64_t>();

  // Case 1: empty axes means treating this reduction as an identity.
  if (axes_span.empty()) {
    ORT_ENFORCE(
        input_sharding_spec == output_sharding_spec,
        "Input and output sharding specs should be the same. Otherwise, resharding is needed.");
    auto* output_tensor = context->Output(0, input_tensor->Shape());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_tensor->MutableData<T>(), input_tensor->Data<T>(), input_tensor->SizeInBytes(),
                                         cudaMemcpyDeviceToDevice, Stream(context)));
    return Status::OK();
  }

  // Case 2: this is a valid reduction. Let's prepare for it.

  bool sharding_on_reduced_axes = false;
  for (auto axis_it = axes_span.begin(); input_sharding_spec.HasShard() && axis_it != axes_span.end(); ++axis_it) {
    if (*axis_it == input_sharding_spec.GetPartitionAxis()) {
      sharding_on_reduced_axes = true;
      break;
    }
  }

  if (sharding_on_reduced_axes) {
    // Case 2-1: sharding on reduced axes.
    ORT_THROW(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Not implemented. Resharding is required to make reduced axes replica.");
  } else {
    // Case 2-2: sharding on passing-through axes or no shard.
    ORT_ENFORCE(
        input_sharding_spec == output_sharding_spec,
        "Input and output sharding specs should be the same. Otherwise, resharding is needed.");
    onnxruntime::cuda::PrepareReduceMetadata metadata;
    ORT_RETURN_IF_ERROR(
        onnxruntime::cuda::PrepareForReduce(input_tensor, keepdims_, axes_span, metadata));
    auto output_tensor = context->Output(0, metadata.squeezed_output_dims);

    // Fast reduction is not deterministic, so sometimes we want to turn it off.
    const bool enable_fast_but_non_deterministic_reduction = !context->GetUseDeterministicCompute();
    return onnxruntime::cuda::ReduceComputeCore<T, CUDNN_REDUCE_TENSOR_NO_INDICES>(
        /* GPU allocator */ Info().GetAllocator(OrtMemType::OrtMemTypeDefault),
        *input_tensor, metadata, *output_tensor, cudnn_reduce_op_, axes_span,
        /* calculate_log */ false, /* calculate_sqt */ false, /* log_sum_exp_ */ false,
        enable_fast_but_non_deterministic_reduction, context->GetComputeStream());
  }
  return Status::OK();
}

template <typename T>
DistributedReduceSum<T>::DistributedReduceSum(
    const OpKernelInfo& info) : DistributedReduceBase<T>(info, CUDNN_REDUCE_TENSOR_ADD){};

template <typename T>
DistributedReduceMean<T>::DistributedReduceMean(
    const OpKernelInfo& info) : DistributedReduceBase<T>(info, CUDNN_REDUCE_TENSOR_AVG){};

template <typename T>
DistributedReduceMax<T>::DistributedReduceMax(
    const OpKernelInfo& info) : DistributedReduceBase<T>(info, CUDNN_REDUCE_TENSOR_MAX){};

// ReduceSum
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReduceSum,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceSum<float>);
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReduceSum,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceSum<MLFloat16>);

// ReduceMean
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReduceMean,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceMean<float>);
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReduceMean,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceMean<MLFloat16>);

// ReduceMax
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReduceMax,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceMax<float>);
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReduceMax,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceMax<MLFloat16>);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
