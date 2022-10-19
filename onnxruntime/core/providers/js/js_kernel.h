// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <emscripten.h>

#include "core/framework/op_kernel.h"
#include "core/providers/js/js_execution_provider.h"

struct pthreadpool;

namespace onnxruntime {
namespace js {

class JsKernel : public OpKernel {
 public:
  explicit JsKernel(const OpKernelInfo& info)
      : OpKernel(info) {
        InitAttributes();
      }
  virtual ~JsKernel() {
    EM_ASM({ Module.jsepReleaseKernel($0); }, this);
  }

 protected:
  virtual void InitAttributes() {
    EM_ASM({ Module.jsepCreateKernel("abs", $0, undefined); }, this);
  };

};
}  // namespace js
}  // namespace onnxruntime
