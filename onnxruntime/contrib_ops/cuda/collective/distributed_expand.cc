// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Distributed computation.
#include "distributed_expand.h"
#include "sharding.h"
#include "sharding_spec.h"
#include "nccl_kernels.h"
#include "mpi_include.h"

// ORT system.
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
  return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, "Encounter unsupported expand pattern.");
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
