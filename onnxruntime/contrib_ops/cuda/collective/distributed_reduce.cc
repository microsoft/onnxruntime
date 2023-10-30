
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_reduce.h"
#include "sharding.h"
#include "sharding_spec.h"
#include "nccl_kernels.h"
#include "mpi_include.h"

// ORT system.
#include "core/providers/cuda/reduction/reduction_ops.h"

// std C++.
#include <iostream>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

template <typename T>
DistributedReduceSum<T>::DistributedReduceSum(const OpKernelInfo& info) : DistributedKernel(info) {};

template <typename T>
DistributedReduceMean<T>::DistributedReduceMean(const OpKernelInfo& info) : DistributedKernel(info) {};

template <typename T>
DistributedReduceMax<T>::DistributedReduceMax(const OpKernelInfo& info) : DistributedKernel(info) {};

template <typename T>
Status DistributedReduceSum<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(context != nullptr);
  return Status::OK();
}

template <typename T>
Status DistributedReduceMean<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(context != nullptr);
  return Status::OK();
}

template <typename T>
Status DistributedReduceMax<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(context != nullptr);
  return Status::OK();
}

// ReduceSum
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DistributedReduceSum,
    kMSDomain,
    1,
    int64_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        // Reduced axes are a small 1-D tensor, so we can use CPU memory.
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceSum<int64_t>);
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
    int64_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceMean<int64_t>);
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
    int64_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    DistributedReduceMax<int64_t>);
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
