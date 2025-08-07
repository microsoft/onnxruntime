// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/framework/op_kernel.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace internal_testing_ep {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* /*context*/) const override;
};

}  // namespace internal_testing_ep
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
