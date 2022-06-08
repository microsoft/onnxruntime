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
        // Is this OK to have a non-const execution provider?
        provider_(const_cast<XnnpackExecutionProvider*>(
            static_cast<const XnnpackExecutionProvider*>(info.GetExecutionProvider()))),
        xnnpack_threadpool_(provider_->GetPrivateThreadPool()) {
  }
  pthreadpool_t GetThreadPool() const {
    return xnnpack_threadpool_;
  }

 protected:
  mutable bool is_op0_initilized{false};

 private:
  XnnpackExecutionProvider* provider_{nullptr};
  mutable pthreadpool_t xnnpack_threadpool_{nullptr};
};
}  // namespace xnnpack
}  // namespace onnxruntime
