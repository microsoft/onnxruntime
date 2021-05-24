// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/math/einsum.h"
#include "einsum_utils/einsum_auxiliary_ops.h"
#include "core/providers/cuda/cuda_execution_provider.h"

namespace onnxruntime {
namespace cuda {

class Einsum final : public onnxruntime::Einsum {
 public:
  Einsum(const OpKernelInfo& info) : onnxruntime::Einsum(info) {
    // We need to cast away the const as PerThreadCublasHandle() is currently a non-const method
    // TODO: Clean up the CUDAExecutionProvider interface to avoid this
    cuda_ep_ = const_cast<CUDAExecutionProvider*>(
        static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  Status DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs,
                       AllocatorPtr allocator, concurrency::ThreadPool* tp) const override;

  // Members of Einsum CUDA kernel
  using onnxruntime::Einsum::einsum_equation_preprocessor_;
  using onnxruntime::Einsum::equation_;

  // We need to access to the CUDA EP instance to get the cublas/cudnn handles
  CUDAExecutionProvider* cuda_ep_;
};

}  // namespace cuda
}  // namespace onnxruntime
