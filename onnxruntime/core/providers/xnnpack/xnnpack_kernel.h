// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {

class XnnpackKernel : public OpKernel {
 public:
  explicit XnnpackKernel(const OpKernelInfo& info)
      : OpKernel(info),
        xnnpack_threadpool_(
            static_cast<const XnnpackExecutionProvider*>(info.GetExecutionProvider())
                ->GetPrivateThreadPool()) {
  }
  pthreadpool_t GetThreadPool() const {
    return xnnpack_threadpool_;
  }

 private:
  mutable pthreadpool_t xnnpack_threadpool_{nullptr};
};
}  // namespace xnnpack
}  // namespace onnxruntime
