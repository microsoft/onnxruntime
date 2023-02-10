// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/cpu/math/einsum.h"
#include "einsum_utils/einsum_auxiliary_ops.h"
#include "core/providers/rocm/rocm_execution_provider.h"

namespace onnxruntime {
namespace rocm {

class Einsum final : public onnxruntime::Einsum {
 public:
  Einsum(const OpKernelInfo& info) : onnxruntime::Einsum(info) {
    // We need to cast away the const as PerThreadRocblasHandle() is currently a non-const method
    // TODO: Clean up the ROCMExecutionProvider interface to avoid this
    rocm_ep_ = const_cast<ROCMExecutionProvider*>(
        static_cast<const ROCMExecutionProvider*>(info.GetExecutionProvider()));
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  Status DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs,
                       AllocatorPtr allocator, concurrency::ThreadPool* tp) const override;

  // Members of Einsum ROCM kernel
  using onnxruntime::Einsum::einsum_equation_preprocessor_;
  using onnxruntime::Einsum::equation_;

  // We need to access to the ROCM EP instance to get the rocblas/miopen handles
  ROCMExecutionProvider* rocm_ep_;
};

}  // namespace rocm
}  // namespace onnxruntime
