// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"
#include "core/common/common.h"
#include "core/providers/cpu/controlflow/if.h"

namespace onnxruntime {
namespace webgpu {

#if defined(BUILD_WEBGPU_EP_STATIC_LIB)

// Use the CPU implementation for the logic
class If final : public onnxruntime::If {
 public:
  If(const OpKernelInfo& info) : onnxruntime::If(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};

#else

class If final : public OpKernel {
 public:
  If(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override;
};
#endif

}  // namespace webgpu
}  // namespace onnxruntime
