// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/op_kernel.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"

struct pthreadpool;

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
  [[nodiscard]] pthreadpool* GetThreadPool() const {
    return xnnpack_threadpool_;
  }

 private:
  pthreadpool* xnnpack_threadpool_;
};
}  // namespace xnnpack
}  // namespace onnxruntime
